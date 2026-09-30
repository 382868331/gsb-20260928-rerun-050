"""执行器模型：固定延迟、幅度饱和与方向不对称变化限制。"""

from __future__ import annotations

from collections import deque

import numpy as np

from .plants.base import Plant


class Actuator:
    def __init__(self, plant: Plant, delay_steps: int = 0):
        if delay_steps < 0:
            raise ValueError("delay_steps 必须非负")
        self.plant = plant
        self.delay_steps = int(delay_steps)
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
        lo, hi = self.plant.action_bounds()
        rate_lo, rate_hi = self.plant.action_rate_bounds()
        for command in self._queue:
            lower = np.maximum(lo, previous + rate_lo)
            upper = np.minimum(hi, previous + rate_hi)
            previous = np.clip(command, lower, upper).copy()
            projected.append(previous.copy())
        return projected

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
        lo, hi = self.plant.action_bounds()
        rate_lo, rate_hi = self.plant.action_rate_bounds()
        lower = np.maximum(lo, self.previous_applied + rate_lo)
        upper = np.minimum(hi, self.previous_applied + rate_hi)
        applied = np.clip(due, lower, upper)
        self.previous_applied = applied.copy()
        return applied
