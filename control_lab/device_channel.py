"""带回执的双执行器通道（真实设备侧）。

三个层面严格分开：

1. 命令计划：控制器决定并“提交”的动作（带原命令 ID）；
2. 设备接收：通道断开时设备收不到命令，会产生 not_received 回执；
3. 实际作用：即使收到命令，也可能只执行一部分（部分执行量），
   回执丢失时控制器只能看到“不确定”，而真实作用仍在此处保留。

通道只负责真实世界：幅度/变化速率限幅、延迟 FIFO、断连、部分执行
与回执生成。回执是否、何时到达控制器由 ReceiptBus 决定。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .capability import Capability
from .command_ledger import (
    ACK,
    EXEC_FULL,
    EXEC_PARTIAL,
    NOT_RECEIVED,
    Receipt,
)
from .plants.base import Plant


@dataclass
class DeviceCommand:
    command_id: int
    action: float
    submitted_step: int
    due_step: int


class ReceiptBus:
    """回执可能延迟、重复或丢失；到达时仍带原命令 ID。"""

    def __init__(self, loss_fn=None, delay_fn=None, duplicate_fn=None):
        # loss_fn(step, receipt) -> True 表示该条回执永久丢失。
        self.loss_fn = loss_fn
        # delay_fn(step, receipt) -> 延迟拍数。
        self.delay_fn = delay_fn
        # duplicate_fn(step, receipt) -> 重复投递次数。
        self.duplicate_fn = duplicate_fn
        self._inbox: dict[int, list[Receipt]] = {}

    def send(self, step: int, receipt: Receipt) -> None:
        if self.loss_fn is not None and self.loss_fn(step, receipt):
            return
        delay = 0 if self.delay_fn is None else int(self.delay_fn(step, receipt))
        delay = max(0, delay)
        self._inbox.setdefault(step + delay, []).append(receipt)
        repeats = 0 if self.duplicate_fn is None else int(
            self.duplicate_fn(step, receipt)
        )
        delivered = 0
        for extra in range(max(0, repeats)):
            # 重复投递必须是同一消息副本；用相同到达时刻/内容即可，
            # 台账按命令+种类+序号识别重复。
            self._inbox.setdefault(step + delay, []).append(receipt)
            delivered += 1
        receipt.detail = (receipt.detail + f"|copies={1 + max(0, repeats)}").strip("|")

    def receive(self, step: int) -> list[Receipt]:
        # 一拍内可能先取早到/迟到回执、设备推进后再取本拍执行回执，
        # 因此不清空，只取出尚未投递的条目（按对象身份去重）。
        due = self._inbox.pop(step, [])
        arrivals: list[Receipt] = []
        for index, receipt in enumerate(due):
            arrivals.append(Receipt(
                command_id=receipt.command_id,
                actuator=receipt.actuator,
                kind=receipt.kind,
                arrived_step=step,
                applied_value=receipt.applied_value,
                requested_value=receipt.requested_value,
                fraction=receipt.fraction,
                mode=receipt.mode,
                detail=f"{receipt.detail}|copy_index={index}",
            ))
        return arrivals

    def pending_steps(self) -> list[int]:
        return sorted(self._inbox)

    def reset(self) -> None:
        self._inbox.clear()


class DeviceChannel:
    """一台真实执行器：FIFO 延迟 + 限幅 + 断连 + 部分执行 + 回执。"""

    def __init__(self, plant: Plant, name: str, capability: Capability,
                 bus: ReceiptBus, command_id_offset: int = 0,
                 execution_fn=None, connected_fn=None, ack_fn=None):
        self.plant = plant
        self.name = str(name)
        self.capability = capability
        self.bus = bus
        # execution_fn(step, cmd) -> 实际执行比例 [0, 1]。
        self.execution_fn = execution_fn
        # connected_fn(step) -> 该周期设备是否连通（命令可送达）。
        self.connected_fn = connected_fn
        # ack_fn(step, cmd) -> 是否在接收时回 ack。
        self.ack_fn = ack_fn
        self.command_id_offset = int(command_id_offset)
        self._next_id = command_id_offset
        neutral = float(np.asarray(plant.neutral_action(), dtype=float).reshape(-1)[0])
        self.previous_applied = neutral
        self.last_applied_command_id = -1
        self._queue: deque[DeviceCommand] = deque()
        self._disconnected_commands: list[DeviceCommand] = []
        # 真实作用历史：command_id -> applied，便于测试与恢复对照。
        self.ground_truth_applied: dict[int, float] = {}

    @property
    def next_command_id(self) -> int:
        return self._next_id

    def due_step_for(self, step: int) -> int:
        if self._queue:
            return max(self._queue[-1].due_step + 1,
                       step + self.capability.delay_steps)
        return step + self.capability.delay_steps

    def submit(self, action: float, step: int) -> tuple[int, int]:
        """提交一条命令；返回 (command_id, due_step)，不立即施加。"""
        command_id = self._next_id
        self._next_id += 1
        due_step = self.due_step_for(step)
        command = DeviceCommand(
            command_id=command_id, action=float(action),
            submitted_step=int(step), due_step=int(due_step),
        )
        connected = True if self.connected_fn is None \
            else bool(self.connected_fn(step))
        if connected:
            self._queue.append(command)
            if self.ack_fn is None or self.ack_fn(step, command):
                self.bus.send(step, Receipt(
                    command_id=command_id, actuator=self.name,
                    kind=ACK, arrived_step=step,
                    requested_value=command.action,
                ))
        else:
            # 断连：设备没有收到，更不会作用；重连时补一条未收到回执。
            self._disconnected_commands.append(command)
        return command_id, due_step

    def reconnect(self, step: int) -> None:
        """设备重连：把断连期间未收到的命令明确回执为未收到。"""
        for command in self._disconnected_commands:
            self.ground_truth_applied.setdefault(command.command_id,
                                                 self.previous_applied)
            self.bus.send(step, Receipt(
                command_id=command.command_id, actuator=self.name,
                kind=NOT_RECEIVED, arrived_step=step,
                requested_value=command.action,
                applied_value=self.previous_applied,
                detail="reported_after_reconnect",
            ))
        self._disconnected_commands.clear()

    def update_capability(self, capability: Capability) -> None:
        self.capability = capability

    def step(self, step: int) -> float:
        """推进一拍，施加到期命令并产生执行回执，返回本拍真实作用量。"""
        applied = self.previous_applied
        due_commands: list[DeviceCommand] = []
        while self._queue and self._queue[0].due_step <= step:
            due_commands.append(self._queue.popleft())
        for command in due_commands:
            fraction = 1.0
            if self.execution_fn is not None:
                fraction = float(self.execution_fn(step, command))
            fraction = min(1.0, max(0.0, fraction))
            cap = self.capability
            lower = max(cap.u_min, self.previous_applied - cap.du_down)
            upper = min(cap.u_max, self.previous_applied + cap.du_up)
            full = float(np.clip(command.action, lower, upper))
            # 部分执行：作用量在“保持值 -> 限幅后的请求值”之间按比例实现。
            applied = self.previous_applied + fraction * (
                full - self.previous_applied
            )
            applied = float(applied)
            self.previous_applied = applied
            self.last_applied_command_id = command.command_id
            self.ground_truth_applied[command.command_id] = applied
            kind = EXEC_FULL if fraction >= 1.0 - 1e-12 else EXEC_PARTIAL
            self.bus.send(step, Receipt(
                command_id=command.command_id, actuator=self.name,
                kind=kind, arrived_step=step,
                requested_value=command.action,
                applied_value=applied,
                fraction=fraction,
            ))
        return applied

    def pending_commands(self) -> list[DeviceCommand]:
        return list(self._queue)
