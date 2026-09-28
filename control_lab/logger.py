"""实验记录：逐步保存计划与执行的差异。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class ExperimentLogger:
    """记录每一步：观测、目标、计划首动作、实际执行动作、扰动、
    是否触发保守动作，以及完整预测轨迹（用于计划 vs 执行分析）。"""

    def __init__(self, dt: float):
        self.dt = dt
        self.rows: list[dict] = []
        self.plans: list[dict] = []

    def record(self, step: int, obs_before: np.ndarray, target: float,
               action: np.ndarray, disturbance: float, obs_after: np.ndarray,
               success: bool, fallback_used: bool, reason: str, cost: float,
               planned_states: np.ndarray, planned_actions: np.ndarray) -> None:
        self.rows.append({
            "step": step,
            "time": step * self.dt,
            "x": float(obs_before[0]),
            "v": float(obs_before[1]),
            "target": float(target),
            "planned_u0": float(planned_actions.reshape(-1)[0]),
            "applied_u": float(action[0]),
            "disturbance": float(disturbance),
            "success": bool(success),
            "fallback": bool(fallback_used),
            "reason": reason,
            "cost": float(cost) if np.isfinite(cost) else np.nan,
            "x_next": float(obs_after[0]),
            "v_next": float(obs_after[1]),
        })
        self.plans.append({
            "step": step,
            "time": step * self.dt,
            "success": bool(success),
            "planned_states": np.asarray(planned_states, dtype=float).tolist(),
            "planned_actions": np.asarray(planned_actions, dtype=float).tolist(),
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
