from __future__ import annotations

import math

from .interfaces import Observation
from .experiment import Scenario


def constant_target(step: int, time: float, observation: Observation) -> float:
    return observation.target_position


def step_target(step: int, time: float, observation: Observation) -> float:
    return -2.0 if step >= 24 else 1.5


def wind_disturbance(step: int, time: float, observation: Observation) -> float:
    if 10 <= step < 35:
        return 0.75 + 0.15 * math.sin(0.8 * time)
    return 0.0


def combined_target(step: int, time: float, observation: Observation) -> float:
    if step < 18:
        return 1.2
    if step < 38:
        return -1.4
    return 0.8


SCENARIOS = [
    Scenario(
        name="nominal",
        description="固定目标，无外部扰动",
        steps=60,
        initial_target=1.5,
        target=constant_target,
    ),
    Scenario(
        name="wind_disturbance",
        description="中段持续风扰，控制器只能从新观测重新规划",
        steps=70,
        initial_target=1.5,
        disturbance=wind_disturbance,
        target=constant_target,
    ),
    Scenario(
        name="target_jump",
        description="运行中途目标从 1.5 m 突变为 -2.0 m",
        steps=75,
        initial_target=1.5,
        target=step_target,
    ),
    Scenario(
        name="disturbance_and_target_jumps",
        description="持续扰动叠加两次目标突变",
        steps=90,
        initial_target=1.2,
        disturbance=wind_disturbance,
        target=combined_target,
    ),
]
