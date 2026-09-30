"""滚动时域实验循环。

控制器每个周期提交动作命令；执行器可能延迟该命令，并在命令真正生效时
施加幅度与变化限制。观测短时缺失时，规划器只使用最后已知观测并输出
明确标记的保守动作，不把旧预测当作当前计划。
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import numpy as np

from .actuator import Actuator
from .capability import ActuatorCapability
from .controller import MPCController
from .logger import ExperimentLogger
from .plants.base import Plant

DisturbanceFn = Callable[[int, np.ndarray], float]
TargetFn = Callable[[int], float]
ActionHook = Callable[[int, np.ndarray], np.ndarray]
ObservationAvailableFn = Callable[[int], bool]
CapabilityFn = Callable[[int], ActuatorCapability | None]

# 能力变化时评估可达性的候选目标（含当前目标）
REACHABILITY_CANDIDATES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def run_experiment(
    plant: Plant,
    controller: MPCController,
    n_steps: int,
    initial_state: np.ndarray,
    target_fn: TargetFn,
    disturbance_fn: DisturbanceFn | None = None,
    action_hook: ActionHook | None = None,
    observation_available_fn: ObservationAvailableFn | None = None,
    capability_fn: CapabilityFn | None = None,
    actuator_delay_steps: int = 0,
    actuator_checkpoint: dict | None = None,
    out_dir: str | Path | None = None,
) -> ExperimentLogger:
    logger = ExperimentLogger(plant.dt)
    plant.reset(np.asarray(initial_state, dtype=float))
    actual_obs = plant.get_observation()
    last_known_obs = actual_obs.copy()
    obs_age = 0
    if actuator_checkpoint is not None:
        # 从检查点恢复：队列中尚未生效的动作保持原样，不重复施加
        actuator = Actuator.restore(plant, actuator_checkpoint)
    else:
        actuator = Actuator(plant, delay_steps=actuator_delay_steps)
    controller.reset()
    controller.set_capability(actuator.capability)
    controller.prev_action = actuator.previous_applied.copy()
    controller.pending_actions = actuator.projected_pending()

    for step in range(n_steps):
        target = float(target_fn(step))
        obs_available = (
            True if observation_available_fn is None
            else bool(observation_available_fn(step))
        )
        if obs_available:
            controller_obs = actual_obs.copy()
            effective_age = obs_age
        else:
            controller_obs = last_known_obs.copy()
            effective_age = -1

        capability_change = None
        if capability_fn is not None:
            new_capability = capability_fn(step)
            if new_capability is not None:
                # 变化前计划：用旧能力在同一观测下规划，仅作对照，不污染热启动
                warm_backup = controller._warm.copy()
                plan_before = controller.plan(
                    controller_obs, target,
                    observation_available=obs_available,
                    observation_age=effective_age,
                )
                controller._warm = warm_backup
                report = actuator.set_capability(new_capability, step)
                controller.set_capability(actuator.capability)
                candidates = sorted(set(REACHABILITY_CANDIDATES) | {target})
                capability_change = {
                    "report": report,
                    "reachability": controller.assess_reachability(candidates),
                    "plan_before": {
                        "success": bool(plan_before.success),
                        "reason": plan_before.reason,
                        "action": [float(v) for v in plan_before.action],
                        "planned_actions": plan_before.planned_actions.tolist(),
                        "planned_states": plan_before.planned_states.tolist(),
                    },
                }

        result = controller.plan(
            controller_obs, target,
            observation_available=obs_available,
            observation_age=effective_age,
        )
        if capability_change is not None:
            capability_change["plan_after"] = {
                "success": bool(result.success),
                "reason": result.reason,
                "action": [float(v) for v in result.action],
                "planned_actions": result.planned_actions.tolist(),
                "planned_states": result.planned_states.tolist(),
            }
        commanded_action = result.action.copy()
        submitted_action = commanded_action.copy()
        if action_hook is not None:
            submitted_action = action_hook(step, submitted_action)

        applied_action = actuator.submit(submitted_action)
        controller.pending_actions = actuator.projected_pending()
        disturbance = (
            0.0 if disturbance_fn is None
            else float(disturbance_fn(step, actual_obs))
        )
        obs_after = plant.step(applied_action, np.array([disturbance]))

        logger.record(
            step=step,
            actual_obs_before=actual_obs,
            controller_obs=controller_obs,
            target=target,
            commanded_action=commanded_action,
            applied_action=applied_action,
            submitted_action=submitted_action,
            actuator_delay_steps=actuator_delay_steps,
            disturbance=disturbance,
            obs_after=obs_after,
            observation_available=obs_available,
            observation_age=effective_age,
            success=result.success,
            fallback_used=result.fallback_used,
            reason=result.reason,
            cost=result.cost,
            planned_states=result.planned_states,
            planned_actions=result.planned_actions,
            capability=actuator.capability,
            capability_change=capability_change,
        )

        # 下一周期的变化限制基准是设备真正施加的动作；延迟队列单独用于预测。
        controller.prev_action = applied_action.copy()
        actual_obs = obs_after
        if obs_available:
            last_known_obs = actual_obs.copy()
            obs_age = 0
        else:
            obs_age += 1

    if out_dir is not None:
        logger.save(out_dir)
        checkpoint_path = Path(out_dir) / "actuator_checkpoint.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"step": n_steps - 1, **actuator.checkpoint()}, f,
                      ensure_ascii=False, indent=2)
    return logger
