from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """Observation boundary shared by every replaceable plant."""

    time: float
    position: float
    velocity: float
    target_position: float


@dataclass(frozen=True)
class Action:
    """Command boundary shared by every replaceable plant."""

    acceleration: float


@dataclass(frozen=True)
class ActionBounds:
    min_acceleration: float
    max_acceleration: float
    max_slew: float

    def __post_init__(self) -> None:
        if self.min_acceleration >= self.max_acceleration:
            raise ValueError("action bounds must satisfy min < max")
        if self.max_slew <= 0.0:
            raise ValueError("max_slew must be positive")

    def clip_action(self, value: float) -> float:
        return min(self.max_acceleration, max(self.min_acceleration, value))

    def apply_slew(self, desired: float, previous: float) -> float:
        limit_low = previous - self.max_slew
        limit_high = previous + self.max_slew
        return min(limit_high, max(limit_low, self.clip_action(desired)))
