"""双执行器闭环实验循环。

每拍顺序：设备重连事件 -> 回执到达 -> 控制器基于“台账信念”规划 ->
计划命令提交设备（区分命令计划/设备接收）-> 设备推进并实际作用 ->
目标状态按合力真实推进。目标状态变化但命令回执缺失时，命令仍以
in_flight/uncertain 保留原归属，控制器不会立即当作未执行重发。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .actuator_group import ActuatorGroup
from .capability import Capability
from .device_channel import ReceiptBus
from .dual_controller import DualMPCController
from .dual_logger import DualExperimentLogger
from .plants.base import Plant

DisturbanceFn = Callable[[int, np.ndarray], float]
TargetFn = Callable[[int], float]
CapabilityChangeFn = Callable[[int], "dict[str, Capability] | None"]
ReconnectFn = Callable[[int], "tuple[str, ...]"]
ReceiptInjectFn = Callable[[int, ActuatorGroup], list]


def run_dual_experiment(
    plant: Plant,
    controller: DualMPCController,
    n_steps: int,
    initial_state: np.ndarray,
    target_fn: TargetFn,
    group: ActuatorGroup | None = None,
    disturbance_fn: DisturbanceFn | None = None,
    capability_change_fn: CapabilityChangeFn | None = None,
    reconnect_fn: ReconnectFn | None = None,
    receipt_inject_fn: ReceiptInjectFn | None = None,
    initial_step: int = 0,
    resume_state: dict | None = None,
    out_dir: str | Path | None = None,
) -> DualExperimentLogger:
    plant.reset(np.asarray(initial_state, dtype=float))
    if group is None:
        group = ActuatorGroup(plant, names=controller.names,
                              capabilities=controller.caps)
    if resume_state is not None:
        group.load_state(resume_state)
        controller.update_capabilities(group.capabilities(), group.ledgers)
        for name in controller.names:
            occurred = group.ledgers[name].confirmed_occurred()
            if occurred:
                controller.last_confirmed[name] = float(
                    occurred[-1].confirmed_value
                )
            else:
                controller.last_confirmed[name] = float(
                    group.channels[name].previous_applied
                )

    logger = DualExperimentLogger(plant.dt, group.names)
    obs = plant.get_observation()

    for step in range(initial_step, initial_step + n_steps):
        target = float(target_fn(step))

        # 1) 能力变化：先重核未发生动作（已发生事实不回退）。
        if capability_change_fn is not None:
            changes = capability_change_fn(step)
            if changes:
                group.update_capability(changes)
                controller.update_capabilities(changes, group.ledgers)

        # 2) 重连：断连期间未收到的命令补明确回执（入总线），不自动重发。
        if reconnect_fn is not None:
            for name in reconnect_fn(step) or ():
                group.reconnect(step, name)

        # 3) 先收取早到的回执（迟到回执、重连通知等）。
        arrived_receipts = group.collect_receipts(step, mark_due=False)

        # 4) 外部补充观测/人工核实（逐项对应原命令）。
        injected: list = []
        if receipt_inject_fn is not None:
            injected = list(receipt_inject_fn(step, group) or [])

        # 5) 用已确认（含迟到回执与人工核实）的实际作用刷新规划基准，
        #    确认发生后避免重复补偿；已发生事实不回退。
        for name in controller.names:
            occurred = group.ledgers[name].confirmed_occurred()
            if occurred:
                controller.last_confirmed[name] = float(
                    occurred[-1].confirmed_value
                )

        # 6) 控制器规划（等待 vs 补偿两套方案同时给出）。
        plan = controller.plan(obs, target, group.ledgers, current_step=step)

        # 7) 提交计划命令（设备断开时设备侧记录为未收到）。
        submitted_ids = group.submit_plan(plan.actions, step)
        submitted = {
            name: {
                "command_id": submitted_ids[name],
                "action": float(plan.actions[name]),
            }
            for name in group.names
        }

        # 8) 设备真实作用与目标真实推进（执行回执下一拍才可见，
        #    模拟回执不可能先于本拍观测到达）。
        applied = group.advance(step)
        # 本拍生效命令的零延迟执行回执在设备推进后立即可见；这是
        # “实际作用已确认”的事实更新，不会改变本拍已提交的计划归属。
        arrived_receipts.extend(group.collect_receipts(step, mark_due=True))
        disturbance = (
            0.0 if disturbance_fn is None
            else float(disturbance_fn(step, obs))
        )
        obs_after = plant.step(
            np.array([group.total_applied(applied)]),
            np.array([disturbance]),
        )

        connected = {
            name: (
                True if group.channels[name].connected_fn is None
                else bool(group.channels[name].connected_fn(step))
            )
            for name in group.names
        }
        all_receipts = [r.to_dict() for r in arrived_receipts] + [
            r.to_dict() for r in injected
        ]
        logger.record(
            step=step, obs_before=obs, target=target,
            planned_actions=dict(plan.actions),
            submitted=submitted, applied=applied,
            disturbance=disturbance, obs_after=obs_after,
            plan=plan, group=group, connected=connected,
            receipts_arrived=all_receipts,
        )
        obs = obs_after

    if out_dir is not None:
        logger.save(out_dir)
    return logger
