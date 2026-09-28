"""计划 vs 执行差异分析。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def plan_vs_actual(csv_path: str | Path, plans_path: str | Path, max_horizon: int | None = None) -> pd.DataFrame:
    """比较每个周期预测的第 k 步与之后真实发生的状态（开环预测 vs 闭环实际）。

    返回长表：plan_step, horizon, dim, predicted, actual, error。
    """
    df = pd.read_csv(csv_path)
    with open(plans_path, encoding="utf-8") as f:
        plans = {p["step"]: p for p in json.load(f)}

    n_total = len(df)
    states = df[["x", "v"]].to_numpy()
    H = max_horizon or max(len(p["planned_states"]) - 1 for p in plans.values())
    records = []
    names = ["x", "v"]
    for i in range(n_total):
        if not bool(plans[i]["success"]):
            continue
        pred = np.asarray(plans[i]["planned_states"])
        for k in range(1, min(H, pred.shape[0] - 1, n_total - i) + 1):
            for dim in range(2):
                if i + k == n_total:
                    actual = float(df.iloc[-1]["x_next" if dim == 0 else "v_next"])
                else:
                    actual = float(states[i + k, dim])
                records.append({
                    "plan_step": i, "horizon": k, "dim": names[dim],
                    "predicted": float(pred[k, dim]),
                    "actual": actual,
                    "error": float(pred[k, dim] - actual),
                })
    return pd.DataFrame(records)


def summarize(df_steps: pd.DataFrame, diff: pd.DataFrame) -> dict:
    """汇总约束满足、失败次数与预测偏差。"""
    du = np.abs(np.diff(df_steps["applied_u"], prepend=df_steps["applied_u"].iloc[0]))
    summary = {
        "n_steps": int(len(df_steps)),
        "n_fallback": int(df_steps["fallback"].sum()),
        "max_applied_u": float(df_steps["applied_u"].abs().max()),
        "max_abs_delta_u_per_step": float(du.max()),
        "final_position_error": float(abs(
            df_steps["x"].iloc[-1] - df_steps["target"].iloc[-1])),
        "rms_plan_actual_error_x": float(
            np.sqrt(np.mean(diff.loc[diff["dim"] == "x", "error"] ** 2))) if len(diff) else 0.0,
        "rms_plan_actual_error_v": float(
            np.sqrt(np.mean(diff.loc[diff["dim"] == "v", "error"] ** 2))) if len(diff) else 0.0,
    }
    return summary
