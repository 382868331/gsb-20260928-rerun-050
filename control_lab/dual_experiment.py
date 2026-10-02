"""双执行器滚动时域实验循环。

两台执行器各自提交命令、各自按延迟队列生效，施加动作叠加到同一
被控对象；每台的能力可独立变化。缺观测时输出明确保守动作；能力
变化前后保留对照计划；保存/恢复发生在两条不同延迟队列中间时，
已生效命令不会重复施加。
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np

from .capability import Capability
from .dual_actuator import DualActuator, conflict_report
from .dual_controller import DualMPCController
from .dual_logger import DualExperimentLogger
from .plants.base import Plant

DisturbanceFn = Callable[[int, np.ndarray], float]
TargetFn = Callable[[int], float]
ObservationAvailableFn = Callable[[int], bool]
DualCapabilityChangeFn = Callable[[int], dict | None]


def _plan_snapshot(result) -> dict:
    return deepcopy({
        "success": result.success,
        "reason": result.reason,
        "commanded_a": result.action_a.tolist(),
        "commanded_b": result.action_b.tolist(),
        "planned_actions_a": result.planned_actions_a.tolist(),
        "planned_actions_b": result.planned_actions_b.tolist(),
        "planned_states": result.planned_states.tolist(),
        "capability": result.capability,
        "target_attainable": result.target_attainable,
        "reachable_target_min": result.reachable_target_min,
        "reachable_target_max": result.reachable_target_max,
    })


def run_dual_experiment(
    plant: Plant,
    controller: DualMPCController,
    n_steps: int,
    initial_state: np.ndarray,
    target_fn: TargetFn,
    disturbance_fn: DisturbanceFn | None = None,
    observation_available_fn: ObservationAvailableFn | None = None,
    capability_change_fn: DualCapabilityChangeFn | None = None,
    actuator_delay_a: int = 0,
    actuator_delay_b: int = 0,
    initial_step: int = 0,
    resume_actuator_state: dict | None = None,
    out_dir: str | Path | None = None,
) -> DualExperimentLogger:
    logger = DualExperimentLogger(plant.dt)
    plant.reset(np.asarray(initial_state, dtype=float))
    actual_obs = plant.get_observation()
    last_known_obs = actual_obs.copy()
    obs_age = 0
    actuator = DualActuator(
        plant, delay_a=actuator_delay_a, delay_b=actuator_delay_b
    )
    if resume_actuator_state is not None:
        actuator.load_state(resume_actuator_state)
    controller.reset()
    for name in ("a", "b"):
        if resume_actuator_state is not None:
            controller.update_capability(name, actuator[name].capability)
        else:
            controller.capability[name] = actuator[name].capability
        controller.prev_action[name] = actuator[name].previous_applied.copy()
        controller.pending_actions[name] = actuator[name].projected_pending()

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
            changes = capability_change_fn(step)
            if changes:
                previous_plan_record = _plan_snapshot(controller.plan(
                    controller_obs, target,
                    observation_available=obs_available,
                    observation_age=effective_age,
                ))
                capability_event = {}
                for name in ("a", "b"):
                    next_capability = changes.get(name)
                    if next_capability is None:
                        continue
                    capability_event[name] = actuator.update_capability(
                        name, step, next_capability
                    )
                    controller.update_capability(name, next_capability)
                    controller.pending_actions[name] = (
                        actuator[name].projected_pending()
                    )

        result = controller.plan(
            controller_obs, target,
            observation_available=obs_available,
            observation_age=effective_age,
        )
        commanded_a = result.action_a.copy()
        commanded_b = result.action_b.copy()
        applied_a, applied_b = actuator.submit(commanded_a, commanded_b, step)
        for name in ("a", "b"):
            controller.pending_actions[name] = (
                actuator[name].projected_pending()
            )
        conflict = conflict_report(applied_a, applied_b)
        unmet_reasons = []
        if conflict["conflict"]:
            unmet_reasons.append("direction_conflict")
        if not result.target_attainable:
            unmet_reasons.append("target_unattainable")
        if result.fallback_used:
            unmet_reasons.append(result.reason)
        unmet_reason = "+".join(unmet_reasons)

        disturbance = (
            0.0 if disturbance_fn is None
            else float(disturbance_fn(step, actual_obs))
        )
        total_applied = applied_a + applied_b
        obs_after = plant.step(total_applied, np.array([disturbance]))

        logger.record(
            step=step,
            actual_obs_before=actual_obs,
            controller_obs=controller_obs,
            target=target,
            commanded_a=commanded_a,
            commanded_b=commanded_b,
            submitted_a=commanded_a,
            submitted_b=commanded_b,
            applied_a=applied_a,
            applied_b=applied_b,
            disturbance=disturbance,
            obs_after=obs_after,
            observation_available=obs_available,
            observation_age=effective_age,
            success=result.success,
            fallback_used=result.fallback_used,
            reason=result.reason,
            cost=result.cost,
            planned_states=result.planned_states,
            planned_actions_a=result.planned_actions_a,
            planned_actions_b=result.planned_actions_b,
            capability=result.capability,
            target_attainable=result.target_attainable,
            reachable_target_min=result.reachable_target_min,
            reachable_target_max=result.reachable_target_max,
            comparison=result.comparison,
            conflict=conflict,
            unmet_reason=unmet_reason,
            capability_event=capability_event,
            previous_plan=previous_plan_record,
            applied_command_id_a=actuator["a"].last_applied_command_id,
            applied_command_id_b=actuator["b"].last_applied_command_id,
            pending_a=[item.to_dict()
                       for item in actuator["a"].pending_commands()],
            pending_b=[item.to_dict()
                       for item in actuator["b"].pending_commands()],
            actuator_state=actuator.to_state(),
        )

        # 下一周期变化限制基准是每台设备真正施加的动作。
        controller.prev_action["a"] = applied_a.copy()
        controller.prev_action["b"] = applied_b.copy()
        actual_obs = obs_after
        if obs_available:
            last_known_obs = actual_obs.copy()
            obs_age = 0
        else:
            obs_age += 1

    if out_dir is not None:
        logger.save(out_dir)
    return logger
