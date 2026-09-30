"""执行器能力描述：运行期可降级/恢复的动作范围、变化限制、延迟与方向。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DIRECTIONS = ("both", "increase_only", "decrease_only")


@dataclass(frozen=True)
class ActuatorCapability:
    """当前执行器能力。

    u_lo/u_hi: 动作幅度范围（可随降级缩小）。
    du_up/du_down: 每拍动作增大/减小的最大幅度（正数）。
    delay_steps: 提交到生效的 FIFO 延迟拍数（运行期可增长）。
    direction: both 双向调节；increase_only 只能增大；
    decrease_only 只能减小（单向调节降级）。
    """

    u_lo: float
    u_hi: float
    du_up: float
    du_down: float
    delay_steps: int = 0
    direction: str = "both"

    def __post_init__(self) -> None:
        if self.u_lo > self.u_hi:
            raise ValueError("u_lo 不能大于 u_hi")
        if self.du_up < 0 or self.du_down < 0:
            raise ValueError("变化限制必须非负")
        if self.delay_steps < 0:
            raise ValueError("delay_steps 必须非负")
        if self.direction not in DIRECTIONS:
            raise ValueError("direction 必须是 " + "/".join(DIRECTIONS) + " 之一")

    @classmethod
    def from_plant(cls, plant, delay_steps: int = 0) -> "ActuatorCapability":
        """以被控对象契约中的标称边界作为初始（未降级）能力。"""
        lo, hi = plant.action_bounds()
        rate_lo, rate_hi = plant.action_rate_bounds()
        return cls(
            u_lo=float(lo[0]), u_hi=float(hi[0]),
            du_up=float(rate_hi[0]), du_down=float(-rate_lo[0]),
            delay_steps=int(delay_steps),
        )

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self.u_lo]), np.array([self.u_hi])

    def rate_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """方向限制折算进每拍变化率边界。"""
        lo, hi = -float(self.du_down), float(self.du_up)
        if self.direction == "increase_only":
            lo = 0.0
        elif self.direction == "decrease_only":
            hi = 0.0
        return np.array([lo]), np.array([hi])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ActuatorCapability":
        return cls(**data)

