"""运行时执行器能力描述。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Capability:
    u_min: float
    u_max: float
    du_up: float
    du_down: float
    delay_steps: int
    withdraw_allowed: bool = False

    def __post_init__(self) -> None:
        if self.u_min > self.u_max:
            raise ValueError("u_min 不能大于 u_max")
        if self.du_up < 0.0 or self.du_down < 0.0:
            raise ValueError("动作变化限制必须非负")
        if self.delay_steps < 0:
            raise ValueError("delay_steps 必须非负")

    @property
    def allows_increase(self) -> bool:
        return self.du_up > 1e-12

    @property
    def allows_decrease(self) -> bool:
        return self.du_down > 1e-12

    def direction(self) -> str:
        if self.allows_increase and self.allows_decrease:
            return "both"
        if self.allows_increase:
            return "increase_only"
        if self.allows_decrease:
            return "decrease_only"
        return "hold_only"

    def to_dict(self) -> dict:
        return {
            "u_min": float(self.u_min),
            "u_max": float(self.u_max),
            "du_up": float(self.du_up),
            "du_down": float(self.du_down),
            "delay_steps": int(self.delay_steps),
            "direction": self.direction(),
            "withdraw_allowed": bool(self.withdraw_allowed),
        }

    @classmethod
    def from_plant(cls, plant, delay_steps: int = 0, withdraw_allowed: bool = False) -> "Capability":
        lo, hi = plant.action_bounds()
        rate_lo, rate_hi = plant.action_rate_bounds()
        return cls(
            float(np.asarray(lo).reshape(-1)[0]),
            float(np.asarray(hi).reshape(-1)[0]),
            float(np.asarray(rate_hi).reshape(-1)[0]),
            float(-np.asarray(rate_lo).reshape(-1)[0]),
            int(delay_steps),
            bool(withdraw_allowed),
        )
