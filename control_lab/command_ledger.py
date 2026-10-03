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


@dataclass
class Receipt:
    command_id: int
    actuator: str
    kind: str
    arrived_step: int
    applied_value: float | None = None
    requested_value: float | None = None
    fraction: float | None = None
    duplicate: bool = False
    superseded: bool = False
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
            "duplicate": bool(self.duplicate),
            "superseded": bool(self.superseded),
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
            duplicate=bool(data.get("duplicate", False)),
            superseded=bool(data.get("superseded", False)),
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
        )


class CommandLedger:
    """单台执行器的命令-回执台账（控制器侧的“信念”）。"""

    def __init__(self, actuator: str):
        self.actuator = str(actuator)
        self._records: dict[int, CommandRecord] = {}
        self._receipts: list[Receipt] = []
        # 同一命令的同一种事实回执只接受第一次；之后重复到达标记 duplicate。
        self._accepted_kinds: dict[tuple[int, str], int] = {}

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
        key = (receipt.command_id, receipt.kind)
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

        if receipt.kind in (EXEC_FULL, EXEC_PARTIAL, MANUAL_EXECUTED):
            applied = receipt.applied_value
            if applied is None:
                applied = record.action
            applied = float(applied)
            if receipt.kind == MANUAL_EXECUTED:
                record.manually_decided = True
                record.status = EXECUTED
            elif receipt.kind == EXEC_PARTIAL and receipt.fraction is not None \
                    and 0.0 <= float(receipt.fraction) < 1.0 - 1e-12:
                record.status = UNCERTAIN
                record.remaining_value = float(record.action - applied)
            else:
                record.status = EXECUTED
                record.remaining_value = 0.0
            record.confirmed_value = applied
            return receipt

        if receipt.kind == MANUAL_NOT_APPLIED:
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

        raise ValueError(f"未知回执种类: {receipt.kind}")

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
            key = (receipt.command_id, receipt.kind)
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
