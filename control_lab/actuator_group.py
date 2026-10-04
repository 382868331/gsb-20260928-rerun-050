"""两台执行器组成的协同执行器组。

职责：

- 每台执行器一条独立的设备通道（延迟、限幅、允许方向各自不同）；
- 每台执行器一本命令台账，把计划命令、设备接收回执、实际作用回执
  逐项对应；
- 回执经 ReceiptBus 到达后只按原 command_id 归档，重复回执、旧命令
  的迟到回执都不会改写新命令；
- 支持人工补充“确认发生/确认未发生”，只有确认未发生才允许重新
  安排相应作用，确认发生后同一作用不会被再次补偿；
- 断连/重连、部分执行、能力变化与保存/恢复均保留原命令归属，
  已生效命令不会因保存/重开再次施加。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .capability import Capability
from .command_ledger import (
    MANUAL_EXECUTED,
    MANUAL_NOT_APPLIED,
    CommandLedger,
    Receipt,
)
from .device_channel import DeviceChannel, ReceiptBus
from .plants.base import Plant


class ActuatorGroup:
    def __init__(self, plant: Plant,
                 capabilities: dict[str, Capability] | None = None,
                 names: tuple[str, str] = ("a", "b"),
                 bus: ReceiptBus | None = None,
                 channel_options: dict[str, dict] | None = None):
        self.plant = plant
        self.names = (str(names[0]), str(names[1]))
        caps = capabilities or {
            name: Capability.from_plant(plant) for name in self.names
        }
        self.bus = bus or ReceiptBus()
        self.ledgers: dict[str, CommandLedger] = {
            name: CommandLedger(name) for name in self.names
        }
        options = channel_options or {}
        self.channels: dict[str, DeviceChannel] = {}
        for index, name in enumerate(self.names):
            opt = options.get(name, {})
            self.channels[name] = DeviceChannel(
                plant, name=name, capability=caps[name], bus=self.bus,
                command_id_offset=index * 10_000_000,
                execution_fn=opt.get("execution_fn"),
                connected_fn=opt.get("connected_fn"),
                ack_fn=opt.get("ack_fn"),
            )

    def reset(self) -> None:
        for name in self.names:
            self.bus.reset()
            self.ledgers[name] = CommandLedger(name)
            channel = self.channels[name]
            channel._next_id = channel.command_id_offset
            channel.previous_applied = float(
                np.asarray(self.plant.neutral_action()).reshape(-1)[0]
            )
            channel.last_applied_command_id = -1
            channel.ground_truth_applied.clear()
            channel._queue.clear()
            channel._disconnected_commands.clear()

    def capability(self, name: str) -> Capability:
        return self.channels[name].capability

    def capabilities(self) -> dict[str, Capability]:
        return {name: ch.capability for name, ch in self.channels.items()}

    # ------------------------------------------------------------------
    # 命令提交与回执
    # ------------------------------------------------------------------
    def submit_plan(self, actions: dict[str, float], step: int) -> dict[str, int]:
        """把控制器本拍计划提交给两台设备；台账登记计划命令。"""
        ids: dict[str, int] = {}
        for name in self.names:
            channel = self.channels[name]
            hold = self._confirmed_baseline(name)
            command_id, due_step = channel.submit(float(actions[name]), step)
            self.ledgers[name].register(
                command_id, float(actions[name]),
                submitted_step=step, due_step=due_step, hold_value=hold,
            )
            ids[name] = command_id
        return ids

    def _confirmed_baseline(self, name: str) -> float:
        occurred = self.ledgers[name].confirmed_occurred()
        if occurred:
            return float(occurred[-1].confirmed_value)
        return float(self.channels[name].previous_applied)

    def advance(self, step: int) -> dict[str, float]:
        """设备推进一拍（真实作用）并产生回执；回执本拍入总线。"""
        applied: dict[str, float] = {}
        for name in self.names:
            applied[name] = float(self.channels[name].step(step))
        return applied

    def collect_receipts(self, step: int, mark_due: bool = True
                         ) -> list[Receipt]:
        """收取本拍到达的回执并按原命令 ID 归档，返回到达列表。"""
        arrivals = self.bus.receive(step)
        accepted: list[Receipt] = []
        for receipt in arrivals:
            ledger = self.ledgers.get(receipt.actuator)
            if ledger is not None and ledger.has(receipt.command_id):
                accepted.append(ledger.ingest(receipt))
        if mark_due:
            for name in self.names:
                self.ledgers[name].mark_due(step)
        return accepted

    def deliver_receipt(self, receipt: Receipt) -> Receipt:
        """实验外补充的观测/回执（含人工核实）立即归档。"""
        return self.ledgers[receipt.actuator].ingest(receipt)

    def manual_confirm(self, step: int, name: str, command_id: int,
                       occurred: bool, applied_value: float | None = None
                       ) -> Receipt:
        """人工补充核实：确认未发生才可重排；确认发生避免重复补偿。"""
        kind = MANUAL_EXECUTED if occurred else MANUAL_NOT_APPLIED
        receipt = Receipt(
            command_id=int(command_id), actuator=name, kind=kind,
            arrived_step=int(step), applied_value=applied_value,
        )
        return self.ledgers[name].ingest(receipt)

    def register_correction(self, name: str, command_id: int, step: int,
                            applied_value: float | None = None,
                            fraction: float | None = None,
                            detail: str = ""):
        """显式登记独立的纠正事实决定（普通迟到回执不会自动纠正）。

        返回纠正后的命令记录；纠正前后的状态、依据与剩余量在记录的
        corrections 中逐项留痕，控制器可据此比较剩余控制能力。
        """
        return self.ledgers[name].register_correction(
            command_id=int(command_id), step=int(step),
            applied_value=applied_value, fraction=fraction, detail=detail,
        )

    def reconnect(self, step: int, name: str | None = None) -> None:
        """执行器重连：断连命令保留原归属，未收到回执入总线，不重发。"""
        targets = [name] if name is not None else list(self.names)
        for target in targets:
            self.channels[target].reconnect(step)

    def reschedule(self, name: str, original_id: int,
                   action: float, step: int) -> int:
        """只有原命令已确认未发生时，才安排替代作用并建立逐项对应。"""
        ledger = self.ledgers[name]
        original = ledger.get(original_id)
        if not original.manually_decided and original.status != "not_applied":
            raise ValueError("命令尚未确认未发生，不能重新安排该作用")
        new_ids = self.submit_plan({
            n: (action if n == name else self.channels[n].previous_applied)
            for n in self.names
        }, step)
        new_id = new_ids[name]
        ledger.mark_reschedule(original_id, new_id)
        return new_id

    def update_capability(self, changes: dict[str, Capability]) -> None:
        for name, capability in changes.items():
            self.channels[name].update_capability(capability)

    def confirmed_actions(self) -> dict[str, float]:
        result = {}
        for name in self.names:
            applied = [
                float(r.confirmed_value)
                for r in self.ledgers[name].confirmed_occurred()
            ]
            result[name] = applied[-1] if applied else float(
                self.channels[name].previous_applied * 0.0
            )
        return result

    def total_applied(self, applied: dict[str, float]) -> float:
        return float(sum(applied[name] for name in self.names))

    # ------------------------------------------------------------------
    # 保存 / 恢复
    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        return {
            "names": list(self.names),
            "ledgers": {
                name: self.ledgers[name].to_state() for name in self.names
            },
            "channels": {
                name: {
                    "capability": self.channels[name].capability.to_dict(),
                    "previous_applied": float(
                        self.channels[name].previous_applied
                    ),
                    "last_applied_command_id": int(
                        self.channels[name].last_applied_command_id
                    ),
                    "next_command_id": int(self.channels[name]._next_id),
                    "queue": [
                        {
                            "command_id": int(c.command_id),
                            "action": float(c.action),
                            "submitted_step": int(c.submitted_step),
                            "due_step": int(c.due_step),
                        }
                        for c in self.channels[name]._queue
                    ],
                    "disconnected_commands": [
                        {
                            "command_id": int(c.command_id),
                            "action": float(c.action),
                            "submitted_step": int(c.submitted_step),
                            "due_step": int(c.due_step),
                        }
                        for c in self.channels[name]._disconnected_commands
                    ],
                    "ground_truth_applied": {
                        int(k): float(v)
                        for k, v in self.channels[name]
                        .ground_truth_applied.items()
                    },
                }
                for name in self.names
            },
        }

    def load_state(self, state: dict) -> None:
        from .device_channel import DeviceCommand

        names = tuple(state.get("names", list(self.names)))
        self.names = (str(names[0]), str(names[1]))
        for name in self.names:
            self.ledgers[name].load_state(state["ledgers"][name])
            channel_data = state["channels"][name]
            cap_data = channel_data["capability"]
            capability = Capability(
                float(cap_data["u_min"]), float(cap_data["u_max"]),
                float(cap_data["du_up"]), float(cap_data["du_down"]),
                int(cap_data["delay_steps"]),
                bool(cap_data.get("withdraw_allowed", False)),
            )
            channel = self.channels[name]
            channel.capability = capability
            channel.previous_applied = float(channel_data["previous_applied"])
            channel.last_applied_command_id = int(
                channel_data.get("last_applied_command_id", -1)
            )
            channel._next_id = int(channel_data["next_command_id"])
            channel._queue = type(channel._queue)(
                DeviceCommand(
                    command_id=int(item["command_id"]),
                    action=float(item["action"]),
                    submitted_step=int(item["submitted_step"]),
                    due_step=int(item["due_step"]),
                )
                for item in channel_data.get("queue", [])
            )
            channel._disconnected_commands = [
                DeviceCommand(
                    command_id=int(item["command_id"]),
                    action=float(item["action"]),
                    submitted_step=int(item["submitted_step"]),
                    due_step=int(item["due_step"]),
                )
                for item in channel_data.get("disconnected_commands", [])
            ]
            channel.ground_truth_applied = {
                int(k): float(v)
                for k, v in channel_data.get("ground_truth_applied", {}).items()
            }

    def save_state(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(self.to_state(), handle, ensure_ascii=False, indent=2)
        return destination

    def load_state_file(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as handle:
            self.load_state(json.load(handle))
