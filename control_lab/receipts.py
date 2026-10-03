"""命令回执台账：区分排队、已确认发生、明确未发生与尚不能确认。

每条命令按 (执行器, 命令 ID) 归属登记，回执只按命令 ID 归账，
不会把另一台执行器的作用算成这条命令的完成。回执可能延迟、重复、
只报告部分执行量，或在重连后批量补报；台账保留原命令归属与全部
回执历史。目标状态已变化但回执未到的命令记为"尚不能确认"，既不
当作未执行重发，也不把它的作用重新安排；只有人工确认未发生后，
相应作用才允许重新计划。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUEUED = "queued"                        # 排队中（未到生效时刻）
UNCERTAIN = "uncertain"                  # 尚不能确认（已过生效时刻但无回执）
CONFIRMED_APPLIED = "confirmed_applied"  # 已确认发生
CONFIRMED_NOT_APPLIED = "confirmed_not_applied"  # 明确未发生

_PARTIAL_TOL = 1e-9


@dataclass
class CommandRecord:
    command_id: int
    actuator: str
    action: list
    submitted_step: int
    due_step: int
    status: str = QUEUED
    received_ack: bool = False           # 设备已确认收到（不代表已作用）
    applied_action: list | None = None   # 回执报告的实际作用量
    partial: bool = False                # 只收到部分执行量
    deficit: float = 0.0                 # 部分执行时明确未发生的作用量
    confirmed_by: str = ""               # receipt / resync / manual
    resolution_step: int | None = None
    receipt_count: int = 0
    duplicate_receipts: int = 0
    late_receipt: bool = False           # 回执到达时已有更新命令注册
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "command_id": int(self.command_id),
            "actuator": self.actuator,
            "action": list(self.action),
            "submitted_step": int(self.submitted_step),
            "due_step": int(self.due_step),
            "status": self.status,
            "received_ack": bool(self.received_ack),
            "applied_action": self.applied_action,
            "partial": bool(self.partial),
            "deficit": float(self.deficit),
            "confirmed_by": self.confirmed_by,
            "resolution_step": self.resolution_step,
            "receipt_count": int(self.receipt_count),
            "duplicate_receipts": int(self.duplicate_receipts),
            "late_receipt": bool(self.late_receipt),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommandRecord":
        return cls(
            command_id=int(data["command_id"]),
            actuator=str(data["actuator"]),
            action=[float(v) for v in data["action"]],
            submitted_step=int(data["submitted_step"]),
            due_step=int(data["due_step"]),
            status=str(data.get("status", QUEUED)),
            received_ack=bool(data.get("received_ack", False)),
            applied_action=data.get("applied_action"),
            partial=bool(data.get("partial", False)),
            deficit=float(data.get("deficit", 0.0)),
            confirmed_by=str(data.get("confirmed_by", "")),
            resolution_step=data.get("resolution_step"),
            receipt_count=int(data.get("receipt_count", 0)),
            duplicate_receipts=int(data.get("duplicate_receipts", 0)),
            late_receipt=bool(data.get("late_receipt", False)),
            note=str(data.get("note", "")),
        )


class ReceiptLedger:
    """按命令归属跟踪回执与人工核实决定的台账。"""

    def __init__(self, actuator_names=("a", "b")):
        self.actuator_names = tuple(str(n) for n in actuator_names)
        self._records = {}
        self._order = []
        self.orphan_receipts = []   # 无法归属的回执（如重连噪声）
        self.decisions = []         # 逐项核实决定（回执/人工/重连）

    def register(self, command: dict) -> CommandRecord:
        key = (str(command["actuator"]), int(command["command_id"]))
        if key in self._records:
            return self._records[key]
        record = CommandRecord(
            command_id=key[1],
            actuator=key[0],
            action=[float(v) for v in np.asarray(command["action"]).reshape(-1)],
            submitted_step=int(command["submitted_step"]),
            due_step=int(command["due_step"]),
        )
        self._records[key] = record
        self._order.append(key)
        return record

    def get(self, actuator: str, command_id: int):
        return self._records.get((str(actuator), int(command_id)))

    def mark_due(self, step: int) -> list:
        """生效时刻已过但仍无结果回执的命令转为"尚不能确认"。

        不会把它们当作未执行：只是标记不确定性，等待回执或人工核实。
        """
        newly = []
        for key in self._order:
            record = self._records[key]
            if record.status == QUEUED and record.due_step <= step:
                record.status = UNCERTAIN
                record.note = "due_step 已过但未收到结果回执"
                newly.append(record)
        return newly

    def receive_receipt(self, receipt: dict) -> dict:
        """处理一条回执，返回处理结果（含是否重复、迟到、部分执行）。"""
        actuator = str(receipt.get("actuator", ""))
        command_id = int(receipt.get("command_id", -1))
        step = int(receipt.get("step", 0))
        kind = str(receipt.get("kind", "receipt"))
        key = (actuator, command_id)
        record = self._records.get(key)
        if record is None:
            self.orphan_receipts.append(
                {"receipt": dict(receipt), "reason": "unknown_command"})
            return {"accepted": False, "reason": "unknown_command",
                    "command_id": command_id, "actuator": actuator}

        record.receipt_count += 1
        if kind == "ack":
            # 设备确认收到，不代表已作用
            record.received_ack = True
            return {"accepted": True, "reason": "ack", "command_id": command_id,
                    "actuator": actuator, "status": record.status}

        if record.status in (CONFIRMED_APPLIED, CONFIRMED_NOT_APPLIED):
            # 同一回执重复到达：保留首次归属与结论，不重复记账
            record.duplicate_receipts += 1
            return {"accepted": True, "reason": "duplicate",
                    "command_id": command_id, "actuator": actuator,
                    "status": record.status}

        # 较旧回执晚于新命令到达：仍按命令 ID 归账
        newer = [k for k in self._order
                 if k[0] == actuator
                 and self._records[k].submitted_step > record.submitted_step]
        if newer:
            record.late_receipt = True

        applied = bool(receipt.get("applied", True))
        applied_action = receipt.get("applied_action")
        record.resolution_step = step
        record.confirmed_by = "resync" if kind == "resync" else "receipt"
        if not applied:
            record.status = CONFIRMED_NOT_APPLIED
            record.applied_action = [0.0] * len(record.action)
            record.deficit = float(record.action[0])
        else:
            if applied_action is None:
                applied_value = float(record.action[0])
            else:
                applied_value = float(np.asarray(applied_action).reshape(-1)[0])
            record.applied_action = [applied_value]
            commanded = float(record.action[0])
            if abs(applied_value - commanded) > _PARTIAL_TOL:
                # 只收到部分执行量：已作用部分确认发生，
                # 差额是明确未发生的作用，允许重新安排
                record.partial = True
                record.deficit = commanded - applied_value
                record.note = "部分执行：差额可重新安排"
            record.status = CONFIRMED_APPLIED
        self.decisions.append({
            "command_id": command_id, "actuator": actuator, "step": step,
            "decision": record.status, "by": record.confirmed_by,
            "partial": record.partial, "deficit": record.deficit,
            "late": record.late_receipt,
        })
        return {"accepted": True, "reason": "resolved",
                "command_id": command_id, "actuator": actuator,
                "status": record.status, "partial": record.partial}

    def receive_resync(self, step: int, reports: list) -> list:
        """执行器重连后的批量补报；逐条按命令 ID 归账。"""
        results = []
        for report in reports:
            report = dict(report)
            report.setdefault("step", step)
            report["kind"] = "resync"
            results.append(self.receive_receipt(report))
        return results

    def confirm_applied(self, actuator: str, command_id: int, step: int) -> dict:
        return self._manual_confirm(actuator, command_id, step, True)

    def confirm_not_applied(self, actuator: str, command_id: int, step: int) -> dict:
        return self._manual_confirm(actuator, command_id, step, False)

    def _manual_confirm(self, actuator: str, command_id: int, step: int,
                        applied: bool) -> dict:
        record = self.get(actuator, command_id)
        if record is None:
            return {"accepted": False, "reason": "unknown_command",
                    "command_id": command_id}
        if record.status != UNCERTAIN:
            return {"accepted": False, "reason": "not_uncertain",
                    "command_id": command_id, "status": record.status}
        record.status = CONFIRMED_APPLIED if applied else CONFIRMED_NOT_APPLIED
        record.confirmed_by = "manual"
        record.resolution_step = int(step)
        if applied:
            record.applied_action = list(record.action)
            record.deficit = 0.0
        else:
            record.applied_action = [0.0] * len(record.action)
            record.deficit = float(record.action[0])
        self.decisions.append({
            "command_id": int(command_id), "actuator": record.actuator,
            "step": int(step), "decision": record.status, "by": "manual",
            "partial": False, "deficit": record.deficit, "late": False,
        })
        return {"accepted": True, "reason": "manual",
                "command_id": int(command_id), "status": record.status}

    def records(self, actuator: str | None = None) -> list:
        out = []
        for key in self._order:
            record = self._records[key]
            if actuator is None or record.actuator == actuator:
                out.append(record)
        return out

    def _by_status(self, status: str) -> dict:
        view = {name: [] for name in self.actuator_names}
        for record in self.records():
            if record.status == status:
                view[record.actuator].append(record.to_dict())
        return view

    def queued(self) -> dict:
        return self._by_status(QUEUED)

    def uncertain(self) -> dict:
        return self._by_status(UNCERTAIN)

    def confirmed_applied(self) -> dict:
        return self._by_status(CONFIRMED_APPLIED)

    def confirmed_not_applied(self) -> dict:
        return self._by_status(CONFIRMED_NOT_APPLIED)

    def view(self) -> dict:
        """使用者视图：每台执行器四类动作分列。"""
        return {
            "queued": self.queued(),
            "confirmed_applied": self.confirmed_applied(),
            "confirmed_not_applied": self.confirmed_not_applied(),
            "uncertain": self.uncertain(),
            "orphan_receipts": list(self.orphan_receipts),
        }

    def has_uncertain(self, actuator: str | None = None) -> bool:
        return any(r.status == UNCERTAIN for r in self.records(actuator))

    def uncertain_records(self, actuator: str) -> list:
        return [r for r in self.records(actuator) if r.status == UNCERTAIN]

    def last_confirmed_applied(self, actuator: str, default: float = 0.0) -> float:
        """该执行器最近一次已确认发生的实际作用量。"""
        value = float(default)
        for record in self.records(actuator):
            if record.status == CONFIRMED_APPLIED and record.applied_action:
                value = float(record.applied_action[0])
        return value

    def compensatable_deficit(self, actuator: str) -> float:
        """明确未发生（含部分执行差额）、允许重新安排的作用量。"""
        total = 0.0
        for record in self.records(actuator):
            if record.status == CONFIRMED_NOT_APPLIED:
                total += float(record.action[0])
            elif record.status == CONFIRMED_APPLIED and record.partial:
                total += float(record.deficit)
        return total

    def to_state(self) -> dict:
        return {
            "actuator_names": list(self.actuator_names),
            "records": [self._records[key].to_dict() for key in self._order],
            "orphan_receipts": list(self.orphan_receipts),
            "decisions": [dict(d) for d in self.decisions],
        }

    def load_state(self, state: dict) -> None:
        self.actuator_names = tuple(
            str(n) for n in state.get("actuator_names", self.actuator_names))
        self._records = {}
        self._order = []
        for data in state.get("records", []):
            record = CommandRecord.from_dict(data)
            key = (record.actuator, record.command_id)
            self._records[key] = record
            self._order.append(key)
        self.orphan_receipts = list(state.get("orphan_receipts", []))
        self.decisions = [dict(d) for d in state.get("decisions", [])]

