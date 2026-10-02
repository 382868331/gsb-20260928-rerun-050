"""双执行器组合：两台执行器共同作用于同一被控对象。

每台执行器保留独立的 FIFO 延迟队列（命令 ID 与生效时刻各自归属），
总施加动作为两台实际施加动作之和；方向相反时报告抵消量。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .actuator import Actuator, QueuedCommand
from .capability import Capability
from .plants.base import Plant

NAMES = ("a", "b")


class DualActuator:
    """两台执行器，各自独立的延迟队列与能力，动作叠加到同一对象。"""

    def __init__(self, plant: Plant, delay_a: int = 0, delay_b: int = 0,
                 withdraw_allowed: bool = False):
        self.plant = plant
        self.actuators = {
            "a": Actuator(plant, delay_steps=delay_a,
                          withdraw_allowed=withdraw_allowed),
            "b": Actuator(plant, delay_steps=delay_b,
                          withdraw_allowed=withdraw_allowed),
        }

    def __getitem__(self, name: str) -> Actuator:
        return self.actuators[name]

    def reset(self) -> None:
        for actuator in self.actuators.values():
            actuator.reset()

    def submit(self, command_a, command_b, step=None):
        """分别提交两台命令，返回两台本周期真正施加的动作。"""
        applied_a = self.actuators["a"].submit(command_a, step)
        applied_b = self.actuators["b"].submit(command_b, step)
        return applied_a, applied_b

    def total_applied(self):
        return (self.actuators["a"].previous_applied
                + self.actuators["b"].previous_applied)

    def update_capability(self, name, step, capability):
        event = self.actuators[name].update_capability(step, capability)
        event["actuator"] = name
        return event

    def projected_pending(self, name):
        return self.actuators[name].projected_pending()

    def pending_commands(self, name):
        return self.actuators[name].pending_commands()

    def to_state(self):
        return {
            name: actuator.to_state()
            for name, actuator in self.actuators.items()
        }

    def load_state(self, state):
        for name in NAMES:
            if name in state:
                self.actuators[name].load_state(state[name])

    def save_state(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as f:
            json.dump(self.to_state(), f, ensure_ascii=False, indent=2)
        return destination

    def load_state_file(self, path):
        with open(path, encoding="utf-8") as f:
            self.load_state(json.load(f))


def conflict_report(applied_a, applied_b):
    """两台施加动作方向相反时，报告实际抵消量与净作用。"""
    ua = float(np.asarray(applied_a, dtype=float).reshape(-1)[0])
    ub = float(np.asarray(applied_b, dtype=float).reshape(-1)[0])
    cancellation = min(abs(ua), abs(ub)) if ua * ub < 0.0 else 0.0
    return {
        "conflict": bool(cancellation > 1e-12),
        "cancellation": float(cancellation),
        "applied_a": ua,
        "applied_b": ub,
        "net": ua + ub,
    }
