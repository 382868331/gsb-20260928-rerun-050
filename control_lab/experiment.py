"""滚动时域实验循环。

每个周期：读取最新观测 -> 基于当前目标重新规划 -> 只执行计划第一步 ->
真实对象（可能含扰动/饱和）前进一步 -> 记录。规划失败执行保守动作。
"""

from __future__ import annotations
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .controller import MPCController
from .logger import ExperimentLogger
from .plants.base import Plant

DisturbanceFn = Callable[[int, np.ndarray], float]
TargetFn = Callable[[int], float]
ActionHook = Callable[[int, np.ndarray], np.ndarray]


def run_experiment(
    plant: Plant,
    controller: MPCController,
    n_steps: int,
    initial_state: np.ndarray,
    target_fn: TargetFn,
    disturbance_fn: DisturbanceFn | None = None,
    action_hook: ActionHook | None = None,
    out_dir: str | Path | None = None,
) -> ExperimentLogger:
    logger = ExperimentLogger(plant.dt)
    obs = plant.reset(np.asarray(initial_state, dtype=float))
    controller.reset()

    for step in range(n_steps):
        target = float(target_fn(step))
        result = controller.plan(obs, target)
        action = result.action.copy()
        # action_hook 可用于注入“执行器异常”，例如强制规划失败路径测试
        if action_hook is not None:
            action = action_hook(step, action)
        disturbance = 0.0 if disturbance_fn is None else float(disturbance_fn(step, obs))
        obs_after = plant.step(action, np.array([disturbance]))

        logger.record(
            step, obs, target, action, disturbance, obs_after,
            success=result.success, fallback_used=result.fallback_used,
            reason=result.reason, cost=result.cost,
            planned_states=result.planned_states, planned_actions=result.planned_actions,
        )
        # 关键：即使执行动作被外部改变，下一周期规划也以真实上一动作为基准
        controller.prev_action = action.copy()
        obs = obs_after

    if out_dir is not None:
        logger.save(out_dir)
    return logger
