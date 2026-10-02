"""两台执行器共同作用于同一目标的执行器组。

每台执行器保留独立的 FIFO 延迟队列、命令 ID 与实际生效时刻；
组在一个控制周期内只让每条命令入队一次，已生效命令不会因为
保存/恢复而被再次施加。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .actuator import Actuator
from .capability import Capability
from .plants.base import Plant


def disabled_capability() -> Capability:
    """对照实验中单台停用：只能保持零力，不承担任何调节量。"""
    return Capability(0.0, 0.0, 0.0, 0.0, delay_steps=0)


class ActuatorGroup:
    def __init__(self, plant: Plant,
                 capabilities: dict[str, Capability] | None = None,
                 names: tuple[str, str] = ("a", "b")):
        self.plant = plant
        self.names = (str(names[0]), str(names[1]))
        caps = capabilities or {
            self.names[0]: Capability.from_plant(plant),
            self.names[1]: Capability.from_plant(plant),
        }
        self.actuators: dict[str, Actuator] = {}
        for index, name in enumerate(self.names):
            cap = caps[name]
            self.actuators[name] = Actuator(
                plant,
                delay_steps=cap.delay_steps,
                withdraw_allowed=cap.withdraw_allowed,
                capability=cap,
                name=name,
                command_id_offset=index * 10_000_000,
            )

    def reset(self) -> None:
        for actuator in self.actuators.values():
            actuator.reset()

    def capability(self, name: str) -> Capability:
        return self.actuators[name].capability

    def capabilities(self) -> dict[str, Capability]:
        return {name: actuator.capability
                for name, actuator in self.actuators.items()}

    def previous_applied(self) -> dict[str, np.ndarray]:
        return {name: actuator.previous_applied.copy()
                for name, actuator in self.actuators.items()}

    def projected_pending(self) -> dict[str, list[np.ndarray]]:
        return {name: actuator.projected_pending()
                for name, actuator in self.actuators.items()}

    def pending_commands(self) -> dict[str, list[dict]]:
        return {
            name: [item.to_dict() for item in actuator.pending_commands()]
            for name, actuator in self.actuators.items()
        }

    def submit(self, commands: dict[str, np.ndarray], step: int) -> dict[str, np.ndarray]:
        """两台各提交一条命令，返回各自本周期真正施加的动作。"""
        return {
            name: self.actuators[name].submit(
                np.asarray(commands[name], dtype=float).reshape(-1), step
            )
            for name in self.names
        }

    def total_applied(self, applied: dict[str, np.ndarray]) -> np.ndarray:
        total = np.zeros(1, dtype=float)
        for name in self.names:
            total += np.asarray(applied[name], dtype=float).reshape(-1)
        return total

    def update_capability(self, step: int,
                          changes: dict[str, Capability]) -> dict:
        events = {}
        for name, capability in changes.items():
            events[name] = self.actuators[name].update_capability(
                step, capability
            )
        return {"step": int(step), "events": events}

    def to_state(self) -> dict:
        return {
            "names": list(self.names),
            "actuators": {
                name: actuator.to_state()
                for name, actuator in self.actuators.items()
            },
        }

    def load_state(self, state: dict) -> None:
        names = tuple(state.get("names", list(self.names)))
        for name in names:
            if name not in self.actuators:
                raise KeyError(f"保存状态中包含未知执行器: {name}")
            self.actuators[name].load_state(state["actuators"][name])

    def save_state(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as f:
            json.dump(self.to_state(), f, ensure_ascii=False, indent=2)
        return destination

    def load_state_file(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as f:
            self.load_state(json.load(f))
