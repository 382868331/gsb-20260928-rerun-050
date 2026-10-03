"""双执行器实验的逐步记录：队列/已发生/未发生/不确定分类可见。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .command_ledger import (
    ACKED,
    EXECUTED,
    IN_FLIGHT,
    NOT_APPLIED,
    QUEUED,
    UNCERTAIN,
)

STATUS_COLUMNS = (
    QUEUED, IN_FLIGHT, ACKED, UNCERTAIN, EXECUTED, NOT_APPLIED,
)


class DualExperimentLogger:
    def __init__(self, dt: float, names: tuple[str, str]):
        self.dt = float(dt)
        self.names = (str(names[0]), str(names[1]))
        self.rows: list[dict] = []
        self.plans: list[dict] = []

    def record(self, *, step: int, obs_before: np.ndarray, target: float,
               planned_actions: dict[str, float],
               submitted: dict[str, dict], applied: dict[str, float],
               disturbance: float, obs_after: np.ndarray,
               plan, group, connected: dict[str, bool],
               receipts_arrived: list[dict]) -> None:
        row: dict = {
            "step": int(step),
            "time": step * self.dt,
            "x": float(obs_before[0]),
            "v": float(obs_before[1]),
            "target": float(target),
            "disturbance": float(disturbance),
            "applied_total": float(sum(applied.values())),
            "x_next": float(obs_after[0]),
            "v_next": float(obs_after[1]),
            "policy": plan.policy,
            "success": bool(plan.success),
            "fallback": bool(plan.fallback_used),
            "reason": plan.reason,
            "target_attainable": bool(plan.target_attainable),
            "reachable_target_min": float(plan.reachable_target_min),
            "reachable_target_max": float(plan.reachable_target_max),
            "cancellation_first_step": float(plan.cancellation_first_step),
            "cancellation_horizon": float(plan.cancellation_horizon),
            "residual_error_first_step": float(plan.residual_error_first_step),
            "residual_error_terminal": float(plan.residual_error_terminal),
            "unmet_reason": plan.unmet_reason,
            "n_receipts_arrived": int(len(receipts_arrived)),
            "n_duplicate_receipts": int(
                sum(1 for r in receipts_arrived if r.get("duplicate"))
            ),
        }
        for name in self.names:
            row[f"planned_{name}"] = float(planned_actions[name])
            row[f"submitted_{name}"] = float(submitted[name]["action"])
            row[f"command_id_{name}"] = int(submitted[name]["command_id"])
            row[f"applied_{name}"] = float(applied[name])
            row[f"connected_{name}"] = bool(connected[name])
            records = group.ledgers[name].records()
            for status in STATUS_COLUMNS:
                row[f"{name}_{status}"] = int(
                    sum(1 for r in records if r.status == status)
                )
        self.rows.append(row)

        plan_payload = {
            "step": int(step),
            "time": step * self.dt,
            "policy": plan.policy,
            "success": bool(plan.success),
            "fallback_used": bool(plan.fallback_used),
            "reason": plan.reason,
            "target": float(target),
            "planned_actions": {
                name: float(planned_actions[name]) for name in self.names
            },
            "submitted_commands": {
                name: {
                    "command_id": int(submitted[name]["command_id"]),
                    "action": float(submitted[name]["action"]),
                }
                for name in self.names
            },
            "applied_actions": {
                name: float(applied[name]) for name in self.names
            },
            "applied_total": float(sum(applied.values())),
            "planned_states": np.asarray(plan.planned_states).tolist(),
            "planned_channel_actions": np.asarray(
                plan.planned_actions
            ).tolist(),
            "envelope_states": (
                None if plan.envelope_states is None
                else np.asarray(plan.envelope_states).tolist()
            ),
            "policies": plan.policies,
            "capabilities": plan.capabilities,
            "channels": plan.channels,
            "target_attainable": bool(plan.target_attainable),
            "reachable_target_min": float(plan.reachable_target_min),
            "reachable_target_max": float(plan.reachable_target_max),
            "cancellation_first_step": float(plan.cancellation_first_step),
            "cancellation_horizon": float(plan.cancellation_horizon),
            "residual_error_first_step": float(plan.residual_error_first_step),
            "residual_error_terminal": float(plan.residual_error_terminal),
            "unmet_reason": plan.unmet_reason,
            "uncertain_commands": plan.uncertain_commands,
            "blocked_reasons": plan.blocked_reasons,
            "connected": dict(connected),
            "receipts_arrived": list(receipts_arrived),
            "commands": {
                name: [r.to_dict() for r in group.ledgers[name].records()]
                for name in self.names
            },
            "receipt_log": {
                name: [
                    r.to_dict()
                    for r in group.ledgers[name].receipts()
                ]
                for name in self.names
            },
        }
        self.plans.append(plan_payload)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "steps.csv"
        self.to_frame().to_csv(csv_path, index=False)
        json_path = out / "plans.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(self.plans, handle, ensure_ascii=False)
        return csv_path, json_path

