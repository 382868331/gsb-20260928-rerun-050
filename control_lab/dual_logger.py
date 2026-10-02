"""双执行器实验记录：对照计划、每台实际动作与目标状态。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class DualExperimentLogger:
    """逐步记录两台执行器的命令链路、冲突报告与完整预测。"""

    def __init__(self, dt: float):
        self.dt = dt
        self.rows: list[dict] = []
        self.plans: list[dict] = []

    def record(
        self,
        step: int,
        actual_obs_before: np.ndarray,
        controller_obs: np.ndarray,
        target: float,
        commanded_a: np.ndarray,
        commanded_b: np.ndarray,
        submitted_a: np.ndarray,
        submitted_b: np.ndarray,
        applied_a: np.ndarray,
        applied_b: np.ndarray,
        disturbance: float,
        obs_after: np.ndarray,
        observation_available: bool,
        observation_age: int,
        success: bool,
        fallback_used: bool,
        reason: str,
        cost: float,
        planned_states: np.ndarray,
        planned_actions_a: np.ndarray,
        planned_actions_b: np.ndarray,
        capability: dict,
        target_attainable: bool,
        reachable_target_min: float,
        reachable_target_max: float,
        comparison: dict | None = None,
        conflict: dict | None = None,
        unmet_reason: str = "",
        capability_event: dict | None = None,
        previous_plan: dict | None = None,
        applied_command_id_a: int = -1,
        applied_command_id_b: int = -1,
        pending_a: list[dict] | None = None,
        pending_b: list[dict] | None = None,
        actuator_state: dict | None = None,
    ) -> None:
        conflict = conflict or {}
        capability = capability or {}
        cap_a = capability.get("a", {})
        cap_b = capability.get("b", {})
        ua = float(applied_a[0])
        ub = float(applied_b[0])
        self.rows.append({
            "step": int(step),
            "time": step * self.dt,
            "x": float(actual_obs_before[0]),
            "v": float(actual_obs_before[1]),
            "x_used": float(controller_obs[0]),
            "v_used": float(controller_obs[1]),
            "target": float(target),
            "commanded_u_a": float(commanded_a[0]),
            "commanded_u_b": float(commanded_b[0]),
            "submitted_u_a": float(submitted_a[0]),
            "submitted_u_b": float(submitted_b[0]),
            "applied_u_a": ua,
            "applied_u_b": ub,
            "applied_u_total": ua + ub,
            "conflict": bool(conflict.get("conflict", False)),
            "conflict_cancellation": float(conflict.get("cancellation", 0.0)),
            "unmet_reason": unmet_reason,
            "observation_available": bool(observation_available),
            "observation_age": int(observation_age),
            "disturbance": float(disturbance),
            "success": bool(success),
            "fallback": bool(fallback_used),
            "reason": reason,
            "target_attainable": bool(target_attainable),
            "reachable_target_min": float(reachable_target_min),
            "reachable_target_max": float(reachable_target_max),
            "single_a_attainable": bool(
                capability.get("single_a_attainable", False)),
            "single_b_attainable": bool(
                capability.get("single_b_attainable", False)),
            "capability_changed": capability_event is not None,
            "a_u_min": float(cap_a.get("u_min", np.nan)),
            "a_u_max": float(cap_a.get("u_max", np.nan)),
            "a_du_up": float(cap_a.get("du_up", np.nan)),
            "a_du_down": float(cap_a.get("du_down", np.nan)),
            "a_delay_steps": int(cap_a.get("delay_steps", 0)),
            "a_ramp_factor": float(cap_a.get("recovery_ramp_factor", 1.0)),
            "b_u_min": float(cap_b.get("u_min", np.nan)),
            "b_u_max": float(cap_b.get("u_max", np.nan)),
            "b_du_up": float(cap_b.get("du_up", np.nan)),
            "b_du_down": float(cap_b.get("du_down", np.nan)),
            "b_delay_steps": int(cap_b.get("delay_steps", 0)),
            "b_ramp_factor": float(cap_b.get("recovery_ramp_factor", 1.0)),
            "applied_command_id_a": int(applied_command_id_a),
            "applied_command_id_b": int(applied_command_id_b),
            "pending_count_a": int(len(pending_a or [])),
            "pending_count_b": int(len(pending_b or [])),
            "cost": float(cost) if np.isfinite(cost) else np.nan,
            "x_next": float(obs_after[0]),
            "v_next": float(obs_after[1]),
        })
        self.plans.append({
            "step": int(step),
            "time": step * self.dt,
            "success": bool(success),
            "fallback_used": bool(fallback_used),
            "reason": reason,
            "observation_available": bool(observation_available),
            "observation_age": int(observation_age),
            "commanded_a": np.asarray(commanded_a, dtype=float).tolist(),
            "commanded_b": np.asarray(commanded_b, dtype=float).tolist(),
            "submitted_a": np.asarray(submitted_a, dtype=float).tolist(),
            "submitted_b": np.asarray(submitted_b, dtype=float).tolist(),
            "applied_a": np.asarray(applied_a, dtype=float).tolist(),
            "applied_b": np.asarray(applied_b, dtype=float).tolist(),
            "planned_states": np.asarray(planned_states, dtype=float).tolist(),
            "planned_actions_a": np.asarray(
                planned_actions_a, dtype=float).tolist(),
            "planned_actions_b": np.asarray(
                planned_actions_b, dtype=float).tolist(),
            "capability": capability,
            "target_attainable": bool(target_attainable),
            "reachable_target_min": float(reachable_target_min),
            "reachable_target_max": float(reachable_target_max),
            "comparison": comparison or {},
            "conflict": conflict,
            "unmet_reason": unmet_reason,
            "capability_event": capability_event,
            "plan_before_capability_change": previous_plan,
            "pending_a": pending_a or [],
            "pending_b": pending_b or [],
            "actuator_state": actuator_state or {},
        })

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
