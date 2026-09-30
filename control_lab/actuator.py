"""执行器模型：固定延迟、幅度饱和、方向不对称变化限制与运行期能力变化。"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from .capability import ActuatorCapability
from .plants.base import Plant


class Actuator:
    """FIFO 延迟执行器，能力（范围/变化率/延迟/方向）可在运行期改变。

    已排队但尚未生效的命令不可撤回：能力变化时它们保留在原队列位置，
    仍按原生效时刻弹出，只在真正生效时按新能力裁剪。延迟增长用队尾
    保持（hold-last）填充新槽位；延迟缩短不会丢弃已排队命令，因此
    实际延迟取 max(新延迟, 当前队列长度)。
    """

    def __init__(self, plant: Plant, delay_steps: int = 0,
                 capability: ActuatorCapability | None = None):
        if delay_steps < 0:
            raise ValueError("delay_steps 必须非负")
        self.plant = plant
        self.capability = capability or ActuatorCapability.from_plant(
            plant, delay_steps)
        self.delay_steps = self.capability.delay_steps
        neutral = np.asarray(plant.neutral_action(), dtype=float).copy()
        self.previous_applied = neutral.copy()
        self._queue: deque[np.ndarray] = deque(
            neutral.copy() for _ in range(self.delay_steps)
        )

    def reset(self) -> None:
        neutral = np.asarray(self.plant.neutral_action(), dtype=float).copy()
        self.previous_applied = neutral.copy()
        self._queue = deque(neutral.copy() for _ in range(self.delay_steps))

    def pending(self) -> list[np.ndarray]:
        return [action.copy() for action in self._queue]

    def projected_pending(self) -> list[np.ndarray]:
        projected: list[np.ndarray] = []
        previous = self.previous_applied.copy()
        lo, hi = self.capability.action_bounds()
        rate_lo, rate_hi = self.capability.rate_bounds()
        for command in self._queue:
            lower = np.maximum(lo, previous + rate_lo)
            upper = np.minimum(hi, previous + rate_hi)
            previous = np.clip(command, lower, upper).copy()
            projected.append(previous.copy())
        return projected

    def set_capability(self, capability: ActuatorCapability,
                       step: int) -> dict:
        """切换执行器能力，返回变更报告（含排队命令去向与生效时刻）。

        报告中的 pending 条目给出每条已排队命令的生效拍：队列位置 i
        的命令将在第 step+i 拍弹出并施加（按新能力裁剪）。
        """
        old = self.capability
        queued = [q.copy() for q in self._queue]
        effective_delay = max(capability.delay_steps, len(self._queue))
        while len(self._queue) < effective_delay:
            filler = (self._queue[-1].copy() if self._queue
                      else np.asarray(self.plant.neutral_action(), dtype=float))
            self._queue.append(filler)
        self.delay_steps = effective_delay
        self.capability = replace(capability, delay_steps=effective_delay)
        return {
            "step": int(step),
            "old": old.to_dict(),
            "new": self.capability.to_dict(),
            "requested_delay_steps": int(capability.delay_steps),
            "effective_delay_steps": int(effective_delay),
            "pending_policy": "kept_not_withdrawable",
            "pending": [
                {"command": [float(v) for v in q],
                 "effective_step": int(step + i)}
                for i, q in enumerate(queued)
            ],
            "note": "已排队命令不可撤回，按原生效时刻施加并按新能力裁剪；"
                    "延迟缩短不影响已排队命令，实际延迟取 max(新延迟, 队列长度)",
        }

    def submit(self, command: np.ndarray) -> np.ndarray:
        """提交当前规划命令，返回本周期真正生效的命令。

        纯延迟为 FIFO；变化限制只在命令真正生效时，相对上一周期实际
        施加动作进行裁剪，因此不会把队列中尚未生效的命令当作即时效果。
        """
        requested = np.asarray(command, dtype=float).reshape(-1).copy()
        self._queue.append(requested)
        due = self._queue.popleft()
        return self._apply_safety(due)

    def _apply_safety(self, due: np.ndarray) -> np.ndarray:
        lo, hi = self.capability.action_bounds()
        rate_lo, rate_hi = self.capability.rate_bounds()
        lower = np.maximum(lo, self.previous_applied + rate_lo)
        upper = np.minimum(hi, self.previous_applied + rate_hi)
        applied = np.clip(due, lower, upper)
        self.previous_applied = applied.copy()
        return applied

    def checkpoint(self) -> dict:
        """持久化执行器状态；重开时恢复而不是重放历史提交，
        避免队列里尚未生效的动作被重复施加。"""
        return {
            "delay_steps": int(self.delay_steps),
            "previous_applied": [float(v) for v in self.previous_applied],
            "queue": [[float(v) for v in q] for q in self._queue],
            "capability": self.capability.to_dict(),
        }

    @classmethod
    def restore(cls, plant: Plant, checkpoint: dict) -> "Actuator":
        capability = ActuatorCapability.from_dict(checkpoint["capability"])
        actuator = cls(plant, capability=capability)
        actuator.delay_steps = int(checkpoint["delay_steps"])
        actuator.previous_applied = np.asarray(
            checkpoint["previous_applied"], dtype=float)
        actuator._queue = deque(
            np.asarray(q, dtype=float) for q in checkpoint["queue"])
        return actuator

