"""双执行器滚动实验：命令计划、设备接收与实际作用分离记录。

每条提交命令登记入回执台账；设备实际生效的命令产生事件，是否、何时、
以什么内容回执由 receipt_fn 决定（可模拟丢失、重复、迟到、部分执行、
重连补报）。尚不能确认的命令不会被当作未执行重发：

- strategy="wait"：存在未确认命令时保守等待，两通道都保持上一作用，
  直到回执或人工核实到达。
- strategy="compensate"：把不确定命令可能的作用从能力中扣除，只在
  剩余能力范围内继续补偿，不为不确定命令的效果重复补偿。

人工确认某条命令未发生后，其作用（含部分执行的差额）才允许重新
安排；确认发生则避免重复补偿。已发生事实（previous_applied 与台账
已确认记录）在能力变化、保存/恢复后都不回退。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .actuator_group import ActuatorGroup
from .capability import Capability
from .dual_controller import DualMPCController
from .plants.base import Plant
from .receipts import ReceiptLedger, UNCERTAIN


class DualExperimentLogger:
    def __init__(self, dt: float):
        self.dt = dt
        self.rows: list[dict] = []
        self.plans: list[dict] = []

    def record(self, row: dict, plan: dict) -> None:
        self.rows.append(row)
        self.plans.append(plan)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "steps.csv"
        self.to_frame().to_csv(csv_path, index=False)
        json_path = out / "plans.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.plans, f, ensure_ascii=False)
        return csv_path, json_path


def _uncertain_force_interval(ledger: ReceiptLedger, name: str,
                              capability: Capability) -> tuple[float, float]:
    """该执行器当前实际作用力的可能区间（不确定命令作用与否的组合）。

    基线为最近一次已确认发生的作用量；不确定命令按 FIFO 顺序逐个
    "作用/未作用"取前缀投影，区间覆盖所有可能，不给精确单值。
    """
    baseline = ledger.last_confirmed_applied(name, default=0.0)
    uncertain = ledger.uncertain_records(name)
    lo = hi = baseline
    value = baseline
    for record in uncertain:
        lower = max(capability.u_min, value - capability.du_down)
        upper = min(capability.u_max, value + capability.du_up)
        value = float(np.clip(record.action[0], lower, upper))
        lo = min(lo, value)
        hi = max(hi, value)
    return lo, hi


def _shrink_capability(capability: Capability, lo: float,
                       hi: float) -> Capability:
    """剩余能力：新命令必须对 [lo, hi] 内任何实际作用都满足变化限制。

    实际作用力只确定在某个区间内时，新命令 z 需要满足
    z <= lo + du_up 且 z >= hi - du_down，才能保证无论真实作用落在
    区间何处都不突破设备变化限制；区间过宽时退化为区间中点（设备
    自身仍会裁剪，不会越界）。不确定命令可能占用的作用量 thus 从
    能力中扣除，而不是被藏进一个精确预测值。
    """
    u_min = max(capability.u_min, hi - capability.du_down)
    u_max = min(capability.u_max, lo + capability.du_up)
    if u_min > u_max:
        mid = float(np.clip(0.5 * (lo + hi),
                            capability.u_min, capability.u_max))
        u_min = u_max = mid
    return Capability(
        u_min, u_max, capability.du_up, capability.du_down,
        capability.delay_steps, capability.withdraw_allowed,
    )


def run_dual_experiment(
    plant: Plant,
    controller: DualMPCController | None = None,
    group: ActuatorGroup | None = None,
    capabilities: dict | None = None,
    n_steps: int = 120,
    initial_state: np.ndarray | None = None,
    target_fn=lambda step: 0.0,
    disturbance_fn=None,
    receipt_fn=None,
    manual_confirmation_fn=None,
    capability_change_fn=None,
    observation_available_fn=None,
    strategy: str = "wait",
    snapshot_step: int | None = None,
    snapshot_path: str | Path | None = None,
    resume_state: dict | None = None,
    out_dir: str | Path | None = None,
    initial_step: int = 0,
) -> DualExperimentLogger:
    if strategy not in ("wait", "compensate"):
        raise ValueError("strategy 必须是 'wait' 或 'compensate'")
    group = group or ActuatorGroup(plant, capabilities)
    controller = controller or DualMPCController(
        plant, capabilities=group.capabilities())
    ledger = ReceiptLedger(group.names)
    logger = DualExperimentLogger(plant.dt)

    plant.reset(np.asarray(initial_state, dtype=float))
    actual_obs = plant.get_observation()
    last_known_obs = actual_obs.copy()
    obs_age = 0

    if resume_state is not None:
        group.load_state(resume_state["group"])
        ledger.load_state(resume_state["ledger"])
        ctrl_state = resume_state["controller"]
        controller.share = {k: float(v) for k, v in ctrl_state["share"].items()}
        controller.target_share = {
            k: float(v) for k, v in ctrl_state["target_share"].items()}
        controller.reallocation_active = bool(ctrl_state["reallocation_active"])
        controller.actuator_available = {
            k: bool(v) for k, v in ctrl_state["actuator_available"].items()}
        controller._warm = np.asarray(ctrl_state["warm"], dtype=float)
        initial_step = int(resume_state["step"]) + 1
        actual_obs = np.asarray(resume_state["actual_obs"], dtype=float)
        plant.reset(actual_obs)
        last_known_obs = actual_obs.copy()

    controller.update_capabilities(group.capabilities())
    for name in group.names:
        controller.prev_action[name] = group.actuators[name].previous_applied.copy()
    controller.pending_actions = group.projected_pending()

    for step in range(initial_step, initial_step + n_steps):
        target = float(target_fn(step))
        obs_available = (True if observation_available_fn is None
                         else bool(observation_available_fn(step)))
        if obs_available:
            controller_obs = actual_obs.copy()
            effective_age = obs_age
        else:
            controller_obs = last_known_obs.copy()
            effective_age = -1

        # 1) 运行时能力变化：重新核对尚未发生的动作，已发生事实不回退
        capability_event = None
        if capability_change_fn is not None:
            changes = capability_change_fn(step)
            if changes:
                capability_event = group.update_capability(step, changes)
                controller.update_capabilities(group.capabilities())
                recheck = {}
                for name in group.names:
                    cap = group.capability(name)
                    flagged = []
                    prev = float(group.actuators[name].previous_applied[0])
                    for item in group.actuators[name].pending_commands():
                        lo = max(cap.u_min, prev - cap.du_down)
                        hi = min(cap.u_max, prev + cap.du_up)
                        clipped = float(np.clip(item.action[0], lo, hi))
                        if abs(clipped - float(item.action[0])) > 1e-9:
                            flagged.append({
                                "command_id": int(item.command_id),
                                "requested": float(item.action[0]),
                                "will_apply": clipped,
                            })
                        prev = clipped
                    recheck[name] = {
                        "pending_exceeding_new_capability": flagged,
                        "uncertain_command_ids": [
                            r.command_id for r in ledger.uncertain_records(name)],
                        "applied_facts_kept": True,
                    }
                capability_event["recheck"] = recheck

        # 2) 人工核实：确认未发生后才允许重新安排相应作用
        manual_results = []
        if manual_confirmation_fn is not None:
            for decision in manual_confirmation_fn(step, ledger) or []:
                if decision["applied"]:
                    manual_results.append(ledger.confirm_applied(
                        decision["actuator"], decision["command_id"], step))
                else:
                    manual_results.append(ledger.confirm_not_applied(
                        decision["actuator"], decision["command_id"], step))

        # 3) 不确定性包络：每台当前作用力的可能区间
        intervals = {}
        for name in group.names:
            intervals[name] = _uncertain_force_interval(
                ledger, name, group.capability(name))

        # 4) 规划（wait 策略在存在未确认命令时保守保持）
        holding = False
        reason_extra = ""
        overrides = None
        if ledger.has_uncertain() and strategy == "wait":
            holding = True
            reason_extra = "waiting_uncertain_confirmation"
        elif ledger.has_uncertain() and strategy == "compensate":
            overrides = {}
            for name in group.names:
                cap = group.capability(name)
                nominal = float(group.actuators[name].previous_applied[0])
                lo, hi = intervals[name]
                overrides[name] = _shrink_capability(cap, lo, hi)

        result = controller.plan(
            controller_obs, target,
            observation_available=obs_available,
            observation_age=effective_age,
            capability_overrides=overrides,
        )
        commanded = {n: result.actions[n].copy() for n in group.names}
        submitted = {n: commanded[n].copy() for n in group.names}
        if holding:
            # 保守等待：输出明确保守刹车动作，而不是重复上一作用
            brake = controller._brake_actions(controller_obs,
                                              controller.caps)
            for name in group.names:
                submitted[name] = brake[name]

        # 5) 提交并登记台账；设备真实作用与回执分离
        last_applied_before = {
            n: group.actuators[n].last_applied_command_id for n in group.names}
        applied = group.submit(submitted, step)
        for name in group.names:
            pending = group.actuators[name].pending_commands()
            if pending:
                ledger.register(pending[-1].to_dict())
        events = []
        for name in group.names:
            actuator = group.actuators[name]
            if actuator.last_applied_command_id != last_applied_before[name]:
                record = ledger.get(name, actuator.last_applied_command_id)
                events.append({
                    "actuator": name,
                    "command_id": int(actuator.last_applied_command_id),
                    "applied_action": float(applied[name][0]),
                    "commanded_action": (float(record.action[0])
                                         if record else float(applied[name][0])),
                })
        receipt_results = []
        if receipt_fn is None:
            receipts = [{**e, "step": step, "applied": True,
                         "applied_action": [e["applied_action"]]}
                        for e in events]
        else:
            receipts = receipt_fn(step, events) or []
        for receipt in receipts:
            if receipt.get("kind") == "resync_batch":
                receipt_results.extend(ledger.receive_resync(
                    step, receipt.get("reports", [])))
            else:
                receipt.setdefault("step", step)
                receipt_results.append(ledger.receive_receipt(receipt))
        newly_uncertain = ledger.mark_due(step)

        # 6) 真实对象推进（合力 + 扰动）
        total_applied = group.total_applied(applied)
        disturbance = (0.0 if disturbance_fn is None
                       else float(disturbance_fn(step, actual_obs)))
        obs_after = plant.step(total_applied, np.array([disturbance]))

        # 7) 记录：实际抵消量、剩余误差、不能继续的原因
        ua = float(applied[group.names[0]][0])
        ub = float(applied[group.names[1]][0])
        cancellation = min(abs(ua), abs(ub)) if ua * ub < 0 else 0.0
        residual_error = target - float(obs_after[0])
        unmet = result.unmet_reason
        if holding:
            unmet = (unmet + ";" if unmet else "") + reason_extra
        elif not result.target_attainable and not unmet:
            unmet = "capability_saturated"

        counts = {status: sum(1 for r in ledger.records() if r.status == status)
                  for status in ("queued", "uncertain", "confirmed_applied",
                                 "confirmed_not_applied")}
        spread_up = sum(max(0.0, intervals[n][1]
                            - float(group.actuators[n].previous_applied[0]))
                        for n in group.names)
        spread_down = sum(max(0.0, float(group.actuators[n].previous_applied[0])
                              - intervals[n][0]) for n in group.names)
        planned_total = np.asarray(result.planned_actions)[:, 2]
        uncertainty_record = {
            "possible_applied_now": {
                n: [float(intervals[n][0]), float(intervals[n][1])]
                for n in group.names},
            "planned_total_min": (planned_total - spread_down).tolist(),
            "planned_total_max": (planned_total + spread_up).tolist(),
            "uncertain_commands": [
                {"actuator": r.actuator, "command_id": r.command_id,
                 "action": r.action, "due_step": r.due_step}
                for r in ledger.records() if r.status == UNCERTAIN],
            "strategy": strategy,
            "holding": holding,
            "capability_overrides": (
                {n: overrides[n].to_dict() for n in group.names}
                if overrides else None),
            "target_attainable_under_remaining_capability": bool(
                result.target_attainable),
        }

        row = {
            "step": int(step), "time": step * plant.dt,
            "x": float(actual_obs[0]), "v": float(actual_obs[1]),
            "target": target,
            "observation_available": bool(obs_available),
            "disturbance": disturbance,
            "commanded_a": float(commanded[group.names[0]][0]),
            "commanded_b": float(commanded[group.names[1]][0]),
            "submitted_a": float(submitted[group.names[0]][0]),
            "submitted_b": float(submitted[group.names[1]][0]),
            "applied_a": ua, "applied_b": ub,
            "applied_total": float(total_applied[0]),
            "cancellation": float(cancellation),
            "residual_error": float(residual_error),
            "strategy": strategy, "holding": bool(holding),
            "success": bool(result.success),
            "fallback": bool(result.fallback_used),
            "reason": result.reason if not holding else reason_extra,
            "unmet_reason": unmet,
            "target_attainable": bool(result.target_attainable),
            "reachable_target_min": float(result.reachable_target_min),
            "reachable_target_max": float(result.reachable_target_max),
            "possible_a_lo": float(intervals[group.names[0]][0]),
            "possible_a_hi": float(intervals[group.names[0]][1]),
            "possible_b_lo": float(intervals[group.names[1]][0]),
            "possible_b_hi": float(intervals[group.names[1]][1]),
            "n_queued": counts["queued"],
            "n_uncertain": counts["uncertain"],
            "n_confirmed_applied": counts["confirmed_applied"],
            "n_confirmed_not_applied": counts["confirmed_not_applied"],
            "capability_changed": capability_event is not None,
        }
        plan_record = {
            "step": int(step),
            "success": bool(result.success),
            "fallback_used": bool(result.fallback_used),
            "reason": row["reason"],
            "commanded_actions": {n: commanded[n].tolist() for n in group.names},
            "submitted_actions": {n: submitted[n].tolist() for n in group.names},
            "applied_actions": {n: applied[n].tolist() for n in group.names},
            "planned_states": np.asarray(result.planned_states).tolist(),
            "planned_actions": np.asarray(result.planned_actions).tolist(),
            "capabilities": result.capabilities,
            "target_attainable": bool(result.target_attainable),
            "reachable_target_min": float(result.reachable_target_min),
            "reachable_target_max": float(result.reachable_target_max),
            "cancellation_first_step": float(result.cancellation_first_step),
            "cancellation_horizon": float(result.cancellation_horizon),
            "allocation_share": result.allocation_share,
            "reallocation_active": bool(result.reallocation_active),
            "uncertainty": uncertainty_record,
            "capability_event": capability_event,
            "receipt_results": receipt_results,
            "manual_confirmations": manual_results,
            "newly_uncertain": [
                {"actuator": r.actuator, "command_id": r.command_id}
                for r in newly_uncertain],
            "ledger_view": ledger.view(),
            "pending_commands": group.pending_commands(),
        }
        logger.record(row, plan_record)

        # 8) 控制器状态推进：以真实施加动作为变化限制基准
        for name in group.names:
            controller.prev_action[name] = applied[name].copy()
        controller.pending_actions = group.projected_pending()
        actual_obs = obs_after
        if obs_available:
            last_known_obs = actual_obs.copy()
            obs_age = 0
        else:
            obs_age += 1

        # 9) 中途保存：不确定回执状态下原命令与核实决定可逐项对应
        if snapshot_step is not None and step == snapshot_step \
                and snapshot_path is not None:
            state = {
                "step": int(step),
                "actual_obs": actual_obs.tolist(),
                "group": group.to_state(),
                "ledger": ledger.to_state(),
                "controller": {
                    "share": dict(controller.share),
                    "target_share": dict(controller.target_share),
                    "reallocation_active": controller.reallocation_active,
                    "actuator_available": dict(controller.actuator_available),
                    "warm": controller._warm.tolist(),
                },
            }
            destination = Path(snapshot_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)

    if out_dir is not None:
        logger.save(out_dir)
    return logger

