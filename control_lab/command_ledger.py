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
# 明确登记的“纠正事实”独立决定：只有它能够改写已经确认的执行事实，
# 普通迟到回执（无论声称更大还是更小）都不能自动完成纠正。
MANUAL_CORRECTION = "manual_correction"

# 回执计量方式：incremental 表示只携带“新增”执行量，cumulative 表示
# 携带截至目前的“累计”执行量。同一指令可能先收到增量回执、后收到累计
# 回执，台账按计量方式分别处理，避免重复累计。
MEASURE_INCREMENTAL = "incremental"
MEASURE_CUMULATIVE = "cumulative"

_VALUE_TOL = 1e-6


@dataclass
class Receipt:
    command_id: int
    actuator: str
    kind: str
    arrived_step: int
    applied_value: float | None = None
    requested_value: float | None = None
    fraction: float | None = None
    base_value: float | None = None
    clipped_value: float | None = None
    measure: str = MEASURE_CUMULATIVE
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
            "base_value": (None if self.base_value is None
                           else float(self.base_value)),
            "clipped_value": (None if self.clipped_value is None
                              else float(self.clipped_value)),
            "measure": self.measure,
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
            base_value=data.get("base_value"),
            clipped_value=data.get("clipped_value"),
            measure=str(data.get("measure", MEASURE_CUMULATIVE)),
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
    cumulative_value: float | None = None
    disputed: bool = False
    dispute_detail: str = ""
    disputed_values: list[float] | None = None
    manually_decided: bool = False
    planned_reschedule_id: int | None = None
    # 显式纠正决定的逐项留痕：每条包含纠正前后的状态、依据与剩余量。
    corrections: list[dict] | None = None
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
            "cumulative_value": (None if self.cumulative_value is None
                                 else float(self.cumulative_value)),
            "disputed": bool(self.disputed),
            "dispute_detail": self.dispute_detail,
            "disputed_values": (None if self.disputed_values is None
                                else [float(v) for v in self.disputed_values]),
            "manually_decided": bool(self.manually_decided),
            "planned_reschedule_id": self.planned_reschedule_id,
            "corrections": (None if self.corrections is None
                            else [dict(item) for item in self.corrections]),
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
            cumulative_value=data.get("cumulative_value"),
            disputed=bool(data.get("disputed", False)),
            dispute_detail=str(data.get("dispute_detail", "")),
            disputed_values=data.get("disputed_values"),
            manually_decided=bool(data.get("manually_decided", False)),
            planned_reschedule_id=data.get("planned_reschedule_id"),
            corrections=(data.get("corrections")
                         and [dict(item)
                              for item in data.get("corrections")]),
            note=str(data.get("note", "")),
        )


class CommandLedger:
    """单台执行器的命令-回执台账（控制器侧的“信念”）。"""

    def __init__(self, actuator: str):
        self.actuator = str(actuator)
        self._records: dict[int, CommandRecord] = {}
        self._receipts: list[Receipt] = []
        # 同一条事实回执（命令+种类+计量方式+内容指纹）只接受第一次；
        # 完全相同内容的再次投递标记 duplicate，而内容不同的后续回执
        # （例如先增量后累计）仍会被接受并按计量方式合并。
        self._accepted_kinds: dict[tuple, int] = {}

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
        key = (
            receipt.command_id, receipt.kind, receipt.measure,
            receipt.applied_value, receipt.fraction,
        )
        if key in self._accepted_kinds:
            receipt.duplicate = True
        else:
            self._accepted_kinds[key] = receipt.arrived_step
        self._receipts.append(receipt)
        if receipt.duplicate:
            return receipt

        record = self._records[receipt.command_id]

        # 显式纠正决定是唯一可以改写已确认事实的入口（即使此前已有
        # 人工结论，也允许用新的独立决定再次纠正）；它必须显式携带
        # 新的依据（applied_value 或 fraction），并逐项保留纠正前后事实。
        if receipt.kind == MANUAL_CORRECTION:
            self._ingest_correction(record, receipt)
            return receipt

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
            # 已经确认发生（含部分发生）后，迟到的“未收到/取消”回执
            # 不能撤销执行事实：只作为争议证据保留，状态不回退。
            if (record.status == EXECUTED
                    or (record.status == UNCERTAIN
                        and record.confirmed_value is not None
                        and self._signed_progress(
                            record, record.confirmed_value) > _VALUE_TOL)):
                self._keep_dispute(
                    record, receipt,
                    "取消/未收到回执与已确认执行事实冲突",
                    [float(record.confirmed_value)],
                )
                return receipt
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
            self._ingest_execution(record, receipt)
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

    def register_correction(self, command_id: int, step: int,
                            applied_value: float | None = None,
                            fraction: float | None = None,
                            detail: str = "") -> CommandRecord:
        """显式登记一条独立的“纠正事实”决定并返回更新后的命令记录。

        普通迟到回执不会触发纠正；只有通过本入口登记的独立决定才能
        改写已确认的执行事实，纠正前后的依据与剩余量逐项留痕。
        """
        receipt = Receipt(
            command_id=int(command_id), actuator=self.actuator,
            kind=MANUAL_CORRECTION, arrived_step=int(step),
            applied_value=(None if applied_value is None
                           else float(applied_value)),
            fraction=(None if fraction is None else float(fraction)),
            detail=str(detail),
        )
        self.ingest(receipt)
        return self._records[int(command_id)]

    # ------------------------------------------------------------------
    # 执行回执合并
    # ------------------------------------------------------------------
    @staticmethod
    def _signed_progress(record: CommandRecord, value: float) -> float:
        """沿请求动作方向的有符号进展；负向动作同样可比大小。"""
        direction = float(record.action) - float(record.hold_value)
        if abs(direction) <= _VALUE_TOL:
            return abs(float(value) - float(record.hold_value))
        sign = 1.0 if direction > 0 else -1.0
        return (float(value) - float(record.hold_value)) * sign

    def _execution_base(self, record: CommandRecord) -> float:
        """增量回执的累加基准：已确认的累计水平，否则保持值。"""
        if record.cumulative_value is not None:
            return float(record.cumulative_value)
        if record.confirmed_value is not None:
            return float(record.confirmed_value)
        return float(record.hold_value)

    def _keep_dispute(self, record: CommandRecord, receipt: Receipt,
                      detail: str, values: list[float]) -> None:
        """普通回执与事实/比例冲突：只追加争议证据，绝不改写已确认事实。"""
        record.disputed = True
        record.dispute_detail = detail
        receipt.disputed = True
        existing = record.disputed_values or []
        for value in values:
            value = float(value)
            if not any(abs(value - old) <= _VALUE_TOL for old in existing):
                existing.append(value)
        record.disputed_values = existing
        # 状态、confirmed_value、remaining_value 全部保持不变：
        # 已发生的动作不能因为一条迟到回执被重新当成未发生。

    def _mark_internal_dispute(self, record: CommandRecord, detail: str,
                               values: list[float]) -> None:
        """尚没有任何已确认事实时的内部矛盾：状态保持不确定。"""
        record.disputed = True
        record.dispute_detail = detail
        existing = record.disputed_values or []
        for value in values:
            value = float(value)
            if not any(abs(value - old) <= _VALUE_TOL for old in existing):
                existing.append(value)
        record.disputed_values = existing
        record.status = UNCERTAIN
        base = record.confirmed_value
        if base is None:
            base = float(record.hold_value)
        record.remaining_value = float(record.action - base)

    def _fact_confirmed_progress(self, record: CommandRecord) -> float:
        if record.status == NOT_APPLIED:
            return 0.0
        if record.confirmed_value is None:
            return 0.0
        return self._signed_progress(record, record.confirmed_value)

    def _ingest_execution(self, record: CommandRecord,
                          receipt: Receipt) -> None:
        action = float(record.action)
        hold = float(record.hold_value)
        manual = receipt.kind == MANUAL_EXECUTED
        measure = receipt.measure or MEASURE_CUMULATIVE

        # 已经给出最终结论（人工核实）的命令：普通回执一律只留证据。
        # 注意 manually_decided 的命令在 ingest 入口已被标记 superseded，
        # 这里主要防御直接调用 _ingest_execution 的路径。

        # 已经确认“未发生”（含取消）后，普通执行回执不能自动撤销该
        # 结论；设备声称执行过，只能登记为争议，等待显式纠正决定。
        if record.status == NOT_APPLIED and not manual:
            claim_values: list[float] = []
            if receipt.applied_value is not None:
                claim_values.append(float(receipt.applied_value))
            self._keep_dispute(
                record, receipt,
                "迟到执行回执与已确认未发生/取消结论冲突",
                claim_values,
            )
            return

        # 分别由“已执行量”和“比例”推出候选绝对作用量；两者都在时互相核对。
        candidates: list[float] = []
        if receipt.applied_value is not None:
            value = float(receipt.applied_value)
            if measure == MEASURE_INCREMENTAL:
                value = self._execution_base(record) + value
            candidates.append(value)
        if receipt.fraction is not None:
            fraction = float(receipt.fraction)
            if measure == MEASURE_INCREMENTAL:
                base = self._execution_base(record)
                candidates.append(base + fraction * (action - base))
            else:
                # 设备回执自带基准/限幅目标时，按同一基准核对数量与比例；
                # 外部回执缺少这些信息时退回台账登记的保持值与请求量。
                base = (float(receipt.base_value)
                        if receipt.base_value is not None else hold)
                target = (float(receipt.clipped_value)
                          if receipt.clipped_value is not None else action)
                candidates.append(base + fraction * (target - base))

        if not candidates:
            if receipt.kind == EXEC_FULL or manual:
                # 设备/人工明确声明“全部执行”但未给数字：按请求量确认。
                candidates.append(action)
            else:
                # 部分执行回执既无量也无比例：不能推断为全部完成，
                # 已确认部分保持不变，余额全部保留为待核实。
                record.status = UNCERTAIN
                base = record.confirmed_value
                if base is None:
                    base = hold
                record.remaining_value = float(action - base)
                record.note = "partial_receipt_without_amount"
                return

        if len(candidates) == 2 and \
                abs(candidates[0] - candidates[1]) > _VALUE_TOL:
            # 已经确认过部分/全部事实时，矛盾回执只留证据、不回退事实；
            # 尚未有事实时维持不确定并保留争议双方。
            if self._fact_confirmed_progress(record) > _VALUE_TOL:
                self._keep_dispute(
                    record, receipt,
                    "applied_value 与 fraction 矛盾",
                    candidates,
                )
            else:
                self._mark_internal_dispute(
                    record,
                    "applied_value 与 fraction 矛盾",
                    candidates,
            )
            return

        applied = float(candidates[0])

        # 倒退回执（任何计量方式、正向/负向动作统一处理）：后到回执
        # 声称的沿请求方向进展小于已确认事实时，只作为争议证据保留，
        # 不覆盖 confirmed_value、不重置 remaining_value、不改变状态。
        # 尚未有任何已确认事实时，倒退到保持值以外的说法也不能自动成立，
        # 仍维持不确定。
        if not manual:
            old_progress = self._fact_confirmed_progress(record)
            new_progress = self._signed_progress(record, applied)
            if new_progress < old_progress - _VALUE_TOL:
                if old_progress > _VALUE_TOL:
                    self._keep_dispute(
                        record, receipt, "执行量倒退",
                        [float(record.confirmed_value), applied],
                    )
                    return
                self._mark_internal_dispute(
                    record, "执行量倒退",
                    [float(record.confirmed_value
                           if record.confirmed_value is not None else hold),
                     applied],
                )
                return

        # 普通回执只能沿请求方向推进或确认事实，不能撤销已确认量。
        if not manual:
            old_progress = self._fact_confirmed_progress(record)
            new_progress = self._signed_progress(record, applied)
            if record.status == EXECUTED:
                # 与已确认水平不一致时只登记争议，改写必须走纠正决定。
                if abs(new_progress - old_progress) > _VALUE_TOL:
                    self._keep_dispute(
                        record, receipt, "迟到回执与已确认事实冲突",
                        [applied],
                    )
                return

        if measure == MEASURE_CUMULATIVE:
            record.cumulative_value = applied
        record.confirmed_value = applied

        if manual:
            record.manually_decided = True
            record.status = EXECUTED
            record.remaining_value = 0.0
            return

        full = receipt.kind == EXEC_FULL
        if receipt.fraction is not None and float(receipt.fraction) >= 1.0 - 1e-12:
            full = True
        if abs(applied - action) <= _VALUE_TOL:
            full = True
        if full:
            record.status = EXECUTED
            record.remaining_value = 0.0
        else:
            # 部分执行：已执行量保留，未执行余额保留为待核实，
            # 缺少比例绝不推断为全部完成。
            record.status = UNCERTAIN
            record.remaining_value = float(action - applied)

    def _ingest_correction(self, record: CommandRecord,
                           receipt: Receipt) -> None:
        action = float(record.action)
        hold = float(record.hold_value)

        candidates: list[float] = []
        if receipt.applied_value is not None:
            candidates.append(float(receipt.applied_value))
        if receipt.fraction is not None:
            candidates.append(
                hold + float(receipt.fraction) * (action - hold)
            )
        if not candidates:
            raise ValueError(
                "manual_correction must carry applied_value or fraction"
            )
        if len(candidates) == 2 and \
                abs(candidates[0] - candidates[1]) > _VALUE_TOL:
            raise ValueError("manual_correction value/fraction mismatch")
        new_value = float(candidates[0])

        old_status = str(record.status)
        old_confirmed = (None if record.confirmed_value is None
                         else float(record.confirmed_value))
        old_remaining = (None if record.remaining_value is None
                         else float(record.remaining_value))
        old_detail = str(record.dispute_detail)

        record.confirmed_value = new_value
        record.cumulative_value = new_value
        record.remaining_value = float(action - new_value)
        full = abs(new_value - action) <= _VALUE_TOL
        record.status = EXECUTED if full else UNCERTAIN
        record.manually_decided = True
        record.note = "manual_correction"
        entry = {
            "step": int(receipt.arrived_step),
            "receipt_detail": str(receipt.detail),
            "basis_value": float(new_value),
            "basis_fraction": (None if receipt.fraction is None
                               else float(receipt.fraction)),
            "old_status": old_status,
            "new_status": str(record.status),
            "old_confirmed_value": old_confirmed,
            "new_confirmed_value": float(new_value),
            "old_remaining_value": old_remaining,
            "new_remaining_value": float(record.remaining_value),
            "resolved_dispute_detail": old_detail,
            "resolved_disputed_values": (
                None if record.disputed_values is None
                else [float(v) for v in record.disputed_values]
            ),
        }
        if record.corrections is None:
            record.corrections = []
        record.corrections.append(entry)
        record.disputed = False
        record.dispute_detail = "resolved_by_manual_correction"

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
            key = (
                receipt.command_id, receipt.kind, receipt.measure,
                receipt.applied_value, receipt.fraction,
            )
            self._accepted_kinds.setdefault(key, receipt.arrived_step)

    def possible_effect_interval(self, record: CommandRecord) -> tuple[float, float]:
        """尚不能确认命令在其生效时刻的可能作用量区间（含保持值）。"""
        if record.status == EXECUTED:
            value = float(record.confirmed_value)
            return value, value
        if record.status == NOT_APPLIED:
            return float(record.hold_value), float(record.hold_value)
        if record.status == UNCERTAIN:
            action = float(record.action)
            if record.confirmed_value is not None:
                # 已确认部分是执行事实：供控制决策使用的可能作用区间
                # 只能从已确认量向“尚未核实的余额方向”延伸。倒退回执
                # 声称的更小值只保存在争议证据里，绝不进入该区间，
                # 已发生的动作不会被重新当成未发生（负向动作同理）。
                confirmed = float(record.confirmed_value)
                return min(confirmed, action), max(confirmed, action)
            points = [float(record.hold_value), action]
            if record.disputed_values:
                points.extend(float(v) for v in record.disputed_values)
            return min(points), max(points)
        lo = min(record.hold_value, float(record.action))
        hi = max(record.hold_value, float(record.action))
        return lo, hi
