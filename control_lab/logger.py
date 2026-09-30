"""实验记录：保存计划命令、设备实际动作与预测轨迹。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class ExperimentLogger:
    """逐步记录控制器所见观测、真实观测、动作链路和完整预测。"""

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
        commanded_action: np.ndarray,
        applied_action: np.ndarray,
        submitted_action: np.ndarray,
        actuator_delay_steps: int,
        disturbance: float,
        obs_after: np.ndarray,
        observation_available: bool,
        observation_age: int,
        success: bool,
        fallback_used: bool,
        reason: str,
        cost: float,
        planned_states: np.ndarray,
        planned_actions: np.ndarray,
        capability: dict | None = None,
        target_attainable: bool = True,
        reachable_target_min: float = -np.inf,
        reachable_target_max: float = np.inf,
        capability_event: dict | None = None,
        previous_plan: dict | None = None,
        applied_command_id: int = -1,
        applied_due_step: int = -1,
        pending_commands: list[dict] | None = None,
        actuator_state: dict | None = None,
    ) -> None:
        self.rows.append({
            "step": int(step),
            "time": step * self.dt,
            "x": float(actual_obs_before[0]),
            "v": float(actual_obs_before[1]),
            "x_used": float(controller_obs[0]),
            "v_used": float(controller_obs[1]),
            "target": float(target),
            "planned_u0": float(commanded_action[0]),
            "commanded_u": float(commanded_action[0]),
            "submitted_u": float(submitted_action[0]),
            "applied_u": float(applied_action[0]),
            "actuator_delay_steps": int(actuator_delay_steps),
            "observation_available": bool(observation_available),
            "observation_age": int(observation_age),
            "disturbance": float(disturbance),
            "success": bool(success),
            "fallback": bool(fallback_used),
            "reason": reason,
            "target_attainable": bool(target_attainable),
            "reachable_target_min": float(reachable_target_min),
            "reachable_target_max": float(reachable_target_max),
            "applied_command_id": int(applied_command_id),
            "applied_due_step": int(applied_due_step),
            "capability_changed": capability_event is not None,
            "effective_delay_steps": int((capability or {}).get("delay_steps", 0)),
            "capability_u_min": float((capability or {}).get("u_min", np.nan)),
            "capability_u_max": float((capability or {}).get("u_max", np.nan)),
            "capability_du_up": float((capability or {}).get("du_up", np.nan)),
            "capability_du_down": float((capability or {}).get("du_down", np.nan)),
            "pending_count": int(len(pending_commands or [])),
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
            "commanded_action": np.asarray(commanded_action, dtype=float).tolist(),
            "submitted_action": np.asarray(submitted_action, dtype=float).tolist(),
            "applied_action": np.asarray(applied_action, dtype=float).tolist(),
            "planned_states": np.asarray(planned_states, dtype=float).tolist(),
            "planned_actions": np.asarray(planned_actions, dtype=float).tolist(),
            "capability": capability or {},
            "target_attainable": bool(target_attainable),
            "reachable_target_min": float(reachable_target_min),
            "reachable_target_max": float(reachable_target_max),
            "capability_event": capability_event,
            "plan_before_capability_change": previous_plan,
            "pending_commands": pending_commands or [],
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
