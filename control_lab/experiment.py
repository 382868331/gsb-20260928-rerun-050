"""滚动时域实验循环。

控制器每个周期提交动作命令；执行器可能延迟该命令，并在命令真正生效时
施加幅度与变化限制。观测短时缺失时，规划器只使用最后已知观测并输出
明确标记的保守动作，不把旧预测当作当前计划。
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np

from .actuator import Actuator
from .capability import Capability
from .controller import MPCController
from .logger import ExperimentLogger
from .plants.base import Plant

DisturbanceFn = Callable[[int, np.ndarray], float]
TargetFn = Callable[[int], float]
ActionHook = Callable[[int, np.ndarray], np.ndarray]
ObservationAvailableFn = Callable[[int], bool]
CapabilityChangeFn = Callable[[int], Capability | None]


def run_experiment(
    plant: Plant,
    controller: MPCController,
    n_steps: int,
    initial_state: np.ndarray,
    target_fn: TargetFn,
    disturbance_fn: DisturbanceFn | None = None,
    action_hook: ActionHook | None = None,
    observation_available_fn: ObservationAvailableFn | None = None,
    capability_change_fn: CapabilityChangeFn | None = None,
    actuator_delay_steps: int = 0,
    initial_step: int = 0,
    resume_actuator_state: dict | None = None,
    out_dir: str | Path | None = None,
) -> ExperimentLogger:
    logger = ExperimentLogger(plant.dt)
    plant.reset(np.asarray(initial_state, dtype=float))
    actual_obs = plant.get_observation()
    last_known_obs = actual_obs.copy()
    obs_age = 0
    actuator = Actuator(plant, delay_steps=actuator_delay_steps)
    if resume_actuator_state is not None:
        actuator.load_state(resume_actuator_state)
    controller.reset()
    if resume_actuator_state is not None:
        controller.update_capability(actuator.capability)
    controller.prev_action = actuator.previous_applied.copy()
    controller.pending_actions = actuator.projected_pending()

    for step in range(initial_step, initial_step + n_steps):
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

        capability_event = None
        previous_plan_record = None
        if capability_change_fn is not None:
            next_capability = capability_change_fn(step)
            if next_capability is not None:
                previous_plan = controller.plan(
                    controller_obs, target,
                    observation_available=obs_available,
                    observation_age=effective_age,
                )
                capability_event = actuator.update_capability(step, next_capability)
                controller.update_capability(next_capability)
                controller.pending_actions = actuator.projected_pending()
                previous_plan_record = deepcopy({
                    "success": previous_plan.success,
                    "reason": previous_plan.reason,
                    "commanded_action": previous_plan.action.tolist(),
                    "planned_actions": previous_plan.planned_actions.tolist(),
                    "planned_states": previous_plan.planned_states.tolist(),
                    "capability": previous_plan.capability,
                    "target_attainable": previous_plan.target_attainable,
                    "reachable_target_min": previous_plan.reachable_target_min,
                    "reachable_target_max": previous_plan.reachable_target_max,
                })

        result = controller.plan(
            controller_obs, target,
            observation_available=obs_available,
            observation_age=effective_age,
        )
        commanded_action = result.action.copy()
        submitted_action = commanded_action.copy()
        if action_hook is not None:
            submitted_action = action_hook(step, submitted_action)

        applied_action = actuator.submit(submitted_action, step)
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
            capability=result.capability,
            target_attainable=result.target_attainable,
            reachable_target_min=result.reachable_target_min,
            reachable_target_max=result.reachable_target_max,
            capability_event=capability_event,
            previous_plan=previous_plan_record,
            applied_command_id=actuator.last_applied_command_id,
            applied_due_step=actuator.last_applied_due_step,
            pending_commands=[item.to_dict() for item in actuator.pending_commands()],
            actuator_state=actuator.to_state(),
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
    return logger
