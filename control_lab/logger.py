"""实验记录：逐步保存“计划/下发”与“真正生效”的差异。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class ExperimentLogger:
    """每步记录：

    * ``x``/``v``：规划时刻控制器使用的状态（观测缺失时为开环估计值）；
    * ``x_true``/``v_true``：设备真实状态（仅用于事后分析，控制器不可见）；
    * ``issued_u``：本周期规划并下发的命令（延迟 D 步后才生效）；
    * ``applied_u``：本周期设备真正施加的动作（在途队列最旧项）；
    * ``obs_available``/``dropout_streak``：观测链路状态；
    * ``committed_*``：本周期在途命令队列快照；
    * 完整预测轨迹（含在途前缀），用于预测 vs 实际差异分析。
    """

    def __init__(self, dt: float):
        self.dt = dt
        self.rows: list[dict] = []
        self.plans: list[dict] = []

    def record(self, step: int, obs_before: np.ndarray, obs_used: np.ndarray,
               target: float, planned_u0: float, issued_u: float, applied_u: float,
               disturbance: float, obs_after: np.ndarray, obs_available: bool,
               dropout_streak: int, delay_steps: int, success: bool,
               fallback_used: bool, reason: str, cost: float,
               planned_states: np.ndarray, planned_actions: np.ndarray,
               committed: np.ndarray) -> None:
        committed = np.asarray(committed, dtype=float).reshape(-1)
        row = {
            "step": step,
            "time": step * self.dt,
            "x": float(obs_used[0]),
            "v": float(obs_used[1]),
            "x_true": float(obs_after[0]),
            "v_true": float(obs_after[1]),
            "target": float(target),
            "issued_u": float(issued_u),
            "planned_u0": float(planned_u0),
            "applied_u": float(applied_u),
            "disturbance": float(disturbance),
            "obs_available": bool(obs_available),
            "dropout_streak": int(dropout_streak),
            "delay_steps": int(delay_steps),
            "success": bool(success),
            "fallback": bool(fallback_used),
            "reason": reason,
            "cost": float(cost) if np.isfinite(cost) else np.nan,
            "x_next": float(obs_after[0]),
            "v_next": float(obs_after[1]),
        }
        for j, value in enumerate(committed):
            row[f"committed_{j}"] = float(value)
        self.rows.append(row)
        self.plans.append({
            "step": step,
            "time": step * self.dt,
            "success": bool(success),
            "fallback_used": bool(fallback_used),
            "reason": reason,
            "obs_available": bool(obs_available),
            "planned_states": np.asarray(planned_states, dtype=float).tolist(),
            "planned_actions": np.asarray(planned_actions, dtype=float).tolist(),
            "issued_u": float(issued_u),
            "applied_u": float(applied_u),
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
