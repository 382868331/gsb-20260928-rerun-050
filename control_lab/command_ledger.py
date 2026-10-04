"""命令台账：把命令计划、设备接收与实际作用逐项对应。

每条命令在每台执行器上保留唯一 command_id，其状态只能是：

- queued：已提交，尚未到生效时刻；
- in_flight：已到生效时刻，但没有任何回执，是否实际作用未知；
- acked：设备确认收到命令，但尚未确认作用量；
- uncertain：有部分信息（例如部分执行量），仍不能确认全部作用；
- executed：已确认实际发生（含部分执行的已发生部分）；
- not_applied：已确认没有发生（人工确认或设备明确未收到）。

同一回执重复到达、旧命令的回执晚于新命令到达，都只按原 command_id
归档，不会写到新命令上，也不会重复计入补偿。人工核实结论为最终结论，
与之冲突的迟到回执会被标记为 superseded 而不是静默覆盖。

回执带计量方式 mode：cumulative 表示 applied_value 是到目前为止的累计
作用量（覆盖而不是累加），incremental 表示相对已确认量的增量，避免
先增量后累计被重复累计。执行回执只带量不带比例时，按量与目标的差距
决定部分或完成，缺失比例绝不推断为全部完成，未执行余额保留为
uncertain 区间。数量与比例矛盾、累计量倒退的回执标记 disputed 并记入
命令的 disputes，保留争议双方数字而不是静默选择一个。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

QUEUED = "queued"
IN_FLIGHT = "in_flight"
ACKED = "acked"
UNCERTAIN = "uncertain"
EXECUTED = "executed"
NOT_APPLIED = "not_applied"

CONFIRMED_STATES = frozenset({EXECUTED, NOT_APPLIED})

ACK = "ack"
EXEC_FULL = "executed_full"
EXEC_PARTIAL = "executed_partial"
NOT_RECEIVED = "not_received"
MANUAL_EXECUTED = "manual_executed"
MANUAL_NOT_APPLIED = "manual_not_applied"

# 回执计量方式：applied_value 是“到目前为止的累计作用量”还是
# “相对上一条已确认量的增量”。累计回执覆盖而不是累加，避免重复累计。
MODE_CUMULATIVE = "cumulative"
MODE_INCREMENTAL = "incremental"


@dataclass
class Receipt:
    command_id: int
    actuator: str
    kind: str
    arrived_step: int
    applied_value: float | None = None
    requested_value: float | None = None
    fraction: float | None = None
    mode: str = MODE_CUMULATIVE
    duplicate: bool = False
    superseded: bool = False
    disputed: bool = False
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "command_id": int(self.command_id),
            "actuator": self.actuator,
            "kind": self.kind,
            "arrived_step": int(self.arrived_step),
            "applied_value": (None if self.applied_value is None
                              else float(self.applied_value)),
            "requested_value": (None if self.requested_value is None
                                else float(self.requested_value)),
            "fraction": None if self.fraction is None else float(self.fraction),
            "mode": self.mode,
            "duplicate": bool(self.duplicate),
            "superseded": bool(self.superseded),
            "disputed": bool(self.disputed),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Receipt":
        return cls(
            command_id=int(data["command_id"]),
            actuator=str(data["actuator"]),
            kind=str(data["kind"]),
            arrived_step=int(data["arrived_step"]),
            applied_value=data.get("applied_value"),
            requested_value=data.get("requested_value"),
            fraction=data.get("fraction"),
            mode=str(data.get("mode", MODE_CUMULATIVE)),
            duplicate=bool(data.get("duplicate", False)),
            superseded=bool(data.get("superseded", False)),
            disputed=bool(data.get("disputed", False)),
            detail=str(data.get("detail", "")),
        )


@dataclass
class CommandRecord:
    command_id: int
    actuator: str
    action: float
    submitted_step: int
    due_step: int
    status: str = QUEUED
    confirmed_value: float | None = None
    hold_value: float = 0.0
    remaining_value: float | None = None
    manually_decided: bool = False
    planned_reschedule_id: int | None = None
    note: str = ""
    # 数量与比例矛盾、累计量倒退等争议回执的留存，不静默选择数字。
    disputes: list = None

    def __post_init__(self):
        if self.disputes is None:
            self.disputes = []

    @property
    def confirmed(self) -> bool:
        return self.status in CONFIRMED_STATES

    def to_dict(self) -> dict:
        return {
            "command_id": int(self.command_id),
            "actuator": self.actuator,
            "action": float(self.action),
            "submitted_step": int(self.submitted_step),
            "due_step": int(self.due_step),
            "status": self.status,
            "confirmed_value": (None if self.confirmed_value is None
                                else float(self.confirmed_value)),
            "hold_value": float(self.hold_value),
            "remaining_value": (None if self.remaining_value is None
                                else float(self.remaining_value)),
            "manually_decided": bool(self.manually_decided),
            "planned_reschedule_id": self.planned_reschedule_id,
            "note": self.note,
            "disputes": list(self.disputes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommandRecord":
        return cls(
            command_id=int(data["command_id"]),
            actuator=str(data["actuator"]),
            action=float(data["action"]),
            submitted_step=int(data["submitted_step"]),
            due_step=int(data["due_step"]),
            status=str(data["status"]),
            confirmed_value=data.get("confirmed_value"),
            hold_value=float(data.get("hold_value", 0.0)),
            remaining_value=data.get("remaining_value"),
            manually_decided=bool(data.get("manually_decided", False)),
            planned_reschedule_id=data.get("planned_reschedule_id"),
            note=str(data.get("note", "")),
            disputes=list(data.get("disputes", [])),
        )


class CommandLedger:
    """单台执行器的命令-回执台账（控制器侧的“信念”）。"""

    def __init__(self, actuator: str):
        self.actuator = str(actuator)
        self._records: dict[int, CommandRecord] = {}
        self._receipts: list[Receipt] = []
        # 同一命令的同内容事实回执只接受第一次；之后重复到达标记 duplicate。
        # 键包含计量方式与数值内容：同一命令先增量后累计、或多次不同的
        # 增量回执都是新事实，只有内容完全相同的重复投递才算 duplicate。
        self._accepted_kinds: dict[tuple, int] = {}

    @staticmethod
    def _dedupe_key(receipt: Receipt) -> tuple:
        return (
            receipt.command_id,
            receipt.kind,
            receipt.mode,
            None if receipt.applied_value is None
            else round(float(receipt.applied_value), 9),
            None if receipt.fraction is None
            else round(float(receipt.fraction), 9),
        )

    def register(self, command_id: int, action: float, submitted_step: int,
                 due_step: int, hold_value: float = 0.0) -> CommandRecord:
        if command_id in self._records:
            raise KeyError(
                f"执行器 {self.actuator} 命令 {command_id} 已登记，不能重复归属"
            )
        record = CommandRecord(
            command_id=int(command_id), actuator=self.actuator,
            action=float(action), submitted_step=int(submitted_step),
            due_step=int(due_step), hold_value=float(hold_value),
        )
        self._records[command_id] = record
        return record

    def mark_due(self, step: int) -> None:
        for record in self._records.values():
            if record.status == QUEUED and record.due_step <= step:
                record.status = IN_FLIGHT

    def get(self, command_id: int) -> CommandRecord:
        return self._records[command_id]

    def has(self, command_id: int) -> bool:
        return command_id in self._records

    def records(self, status: str | Iterable[str] | None = None) -> list[CommandRecord]:
        if status is None:
            wanted = None
        elif isinstance(status, str):
            wanted = {status}
        else:
            wanted = set(status)
        items = [r for r in self._records.values()
                 if wanted is None or r.status in wanted]
        items.sort(key=lambda r: (r.due_step, r.command_id))
        return items

    def queued(self) -> list[CommandRecord]:
        return self.records(QUEUED)

    def confirmed_occurred(self) -> list[CommandRecord]:
        return self.records(EXECUTED)

    def confirmed_not_occurred(self) -> list[CommandRecord]:
        return self.records(NOT_APPLIED)

    def unconfirmed(self) -> list[CommandRecord]:
        return self.records((IN_FLIGHT, ACKED, UNCERTAIN))

    def ingest(self, receipt: Receipt) -> Receipt:
        if receipt.actuator != self.actuator:
            raise ValueError("回执执行器与台账不匹配")
        if not self.has(receipt.command_id):
            raise KeyError(
                f"回执属于未知命令 {receipt.command_id}，不能挂到其他命令上"
            )
        key = self._dedupe_key(receipt)
        if key in self._accepted_kinds:
            receipt.duplicate = True
        else:
            self._accepted_kinds[key] = receipt.arrived_step
        self._receipts.append(receipt)
        if receipt.duplicate:
            return receipt

        record = self._records[receipt.command_id]
        if record.manually_decided:
            receipt.superseded = True
            return receipt

        if receipt.kind == ACK:
            # ack 只在命令已到生效时刻且没有执行/未收到事实时改变信念；
            # 排队阶段收到 ack 仅记录，状态仍由 due_step 决定。
            if record.status == IN_FLIGHT:
                record.status = ACKED
            if record.status in (QUEUED, ACKED):
                record.note = "device_acknowledged"
            return receipt

        if receipt.kind == NOT_RECEIVED:
            record.status = NOT_APPLIED
            baseline = float(record.hold_value)
            if receipt.applied_value is not None:
                baseline = float(receipt.applied_value)
                record.hold_value = baseline
            record.confirmed_value = baseline
            record.remaining_value = float(record.action - baseline)
            record.note = "device_reported_not_received"
            return receipt

        if receipt.kind == MANUAL_NOT_APPLIED:
            return self._ingest_manual_not_applied(record, receipt)

        if receipt.kind in (EXEC_FULL, EXEC_PARTIAL, MANUAL_EXECUTED):
            applied = self._resolve_applied(record, receipt)
            if applied is None:
                # 部分执行回执既未带量也未带比例：不能推断任何执行量，
                # 只能确认存在未决执行，保持不确定，余额不消失。
                if record.status != EXECUTED:
                    record.status = UNCERTAIN
                    base = record.confirmed_value
                    if base is None:
                        base = float(record.hold_value)
                    record.remaining_value = float(record.action - base)
                    record.note = "partial_receipt_without_amount"
                return receipt

            dispute = self._detect_dispute(record, receipt, applied)
            if dispute is not None:
                # 数量与比例矛盾或累计量倒退：保留争议信息，不静默选择。
                receipt.disputed = True
                receipt.detail = (receipt.detail + "|dispute:" + dispute
                                  ).strip("|")
                record.disputes.append({
                    "arrived_step": int(receipt.arrived_step),
                    "kind": receipt.kind,
                    "mode": receipt.mode,
                    "reported_value": receipt.applied_value,
                    "fraction": receipt.fraction,
                    "kept_confirmed_value": record.confirmed_value,
                    "reason": dispute,
                })
                record.note = "dispute:" + dispute
                if record.status != EXECUTED:
                    record.status = UNCERTAIN
                    base = record.confirmed_value
                    if base is None:
                        base = float(record.hold_value)
                    record.remaining_value = float(record.action - base)
                return receipt

            if receipt.kind == MANUAL_EXECUTED:
                record.manually_decided = True
                record.status = EXECUTED
                record.remaining_value = 0.0
            elif receipt.kind == EXEC_PARTIAL:
                # 缺少比例时绝不推断为全部完成：按量与目标的差距决定。
                full = False
                if receipt.fraction is not None:
                    full = float(receipt.fraction) >= 1.0 - 1e-12
                if not full:
                    full = abs(applied - float(record.action)) <= 1e-9 * max(
                        1.0, abs(float(record.action)))
                if full:
                    record.status = EXECUTED
                    record.remaining_value = 0.0
                else:
                    record.status = UNCERTAIN
                    record.remaining_value = float(record.action - applied)
            else:
                record.status = EXECUTED
                record.remaining_value = 0.0
            record.confirmed_value = applied
            return receipt

        raise ValueError(f"未知回执种类: {receipt.kind}")

    @staticmethod
    def _resolve_applied(record: CommandRecord, receipt: Receipt
                         ) -> float | None:
        """把回执数值折算成绝对作用量；增量回执在已确认量上累加。"""
        applied = receipt.applied_value
        if applied is None:
            if receipt.fraction is not None:
                frac = float(receipt.fraction)
                return float(record.hold_value) + frac * (
                    float(record.action) - float(record.hold_value))
            if receipt.kind == EXEC_PARTIAL:
                return None
            return float(record.action)
        applied = float(applied)
        if receipt.mode == MODE_INCREMENTAL:
            base = record.confirmed_value
            if base is None:
                base = float(record.hold_value)
            applied = float(base) + applied
        return applied

    @staticmethod
    def _detect_dispute(record: CommandRecord, receipt: Receipt,
                        applied: float) -> str | None:
        """识别数量与比例矛盾、累计量倒退；返回争议原因或 None。"""
        hold = float(record.hold_value)
        action = float(record.action)
        span = action - hold
        tol = 1e-6 * max(1.0, abs(action), abs(hold))
        if receipt.fraction is not None and receipt.applied_value is not None \
                and abs(span) > 1e-9:
            frac = float(receipt.fraction)
            implied = (applied - hold) / span
            # 设备限幅/中间命令会让“声称全量但量不同”成为正常情况，
            # 只有比例明显小于 1 而量明显超出比例所达位置才算矛盾。
            if frac < 1.0 - 1e-9 and abs(implied - frac) > 0.5:
                return (f"amount_fraction_mismatch"
                        f"(implied={implied:.6g},fraction={frac:.6g})")
        if receipt.mode == MODE_CUMULATIVE \
                and record.confirmed_value is not None:
            confirmed = float(record.confirmed_value)
            progress_old = confirmed - hold
            progress_new = applied - hold
            scale = max(1.0, abs(progress_old))
            if abs(progress_old) > 1e-9 and (
                    progress_new * progress_old < -1e-9
                    or abs(progress_new) < abs(progress_old) - 1e-6 * scale):
                return (f"cumulative_regression"
                        f"(previous={confirmed},reported={applied})")
        return None

    def _ingest_manual_not_applied(self, record: CommandRecord,
                                   receipt: Receipt) -> Receipt:
        record.manually_decided = True
        record.status = NOT_APPLIED
        baseline = float(record.hold_value)
        if receipt.applied_value is not None:
            baseline = float(receipt.applied_value)
            record.hold_value = baseline
        record.confirmed_value = baseline
        record.remaining_value = float(record.action - baseline)
        record.note = "manual_confirmation"
        return receipt

    def receipts(self, include_duplicates: bool = True) -> list[Receipt]:
        if include_duplicates:
            return list(self._receipts)
        return [r for r in self._receipts if not r.duplicate]

    def mark_reschedule(self, original_id: int, new_id: int) -> None:
        """人工确认未发生后，登记替代命令，保证逐项可追溯。"""
        original = self._records[original_id]
        if original.status != NOT_APPLIED:
            raise ValueError("只有确认未发生的命令允许重新安排")
        if new_id not in self._records:
            raise KeyError(f"替代命令 {new_id} 尚未登记")
        original.planned_reschedule_id = int(new_id)
        self._records[new_id].note = f"reschedule_of_command_{original_id}"

    def to_state(self) -> dict:
        return {
            "actuator": self.actuator,
            "records": [r.to_dict() for r in self.records()],
            "receipts": [r.to_dict() for r in self._receipts],
        }

    def load_state(self, state: dict) -> None:
        self.actuator = str(state.get("actuator", self.actuator))
        self._records = {}
        for data in state.get("records", []):
            record = CommandRecord.from_dict(data)
            self._records[record.command_id] = record
        self._receipts = [
            Receipt.from_dict(data) for data in state.get("receipts", [])
        ]
        self._accepted_kinds = {}
        for receipt in self._receipts:
            if receipt.duplicate:
                continue
            key = self._dedupe_key(receipt)
            self._accepted_kinds.setdefault(key, receipt.arrived_step)

    def possible_effect_interval(self, record: CommandRecord) -> tuple[float, float]:
        """尚不能确认命令在其生效时刻的可能作用量区间（含保持值）。"""
        if record.status == EXECUTED:
            value = float(record.confirmed_value)
            return value, value
        if record.status == NOT_APPLIED:
            return float(record.hold_value), float(record.hold_value)
        if record.status == UNCERTAIN and record.confirmed_value is not None:
            lo = min(float(record.confirmed_value), float(record.action))
            hi = max(float(record.confirmed_value), float(record.action))
            return lo, hi
        lo = min(record.hold_value, float(record.action))
        hi = max(record.hold_value, float(record.action))
        return lo, hi
