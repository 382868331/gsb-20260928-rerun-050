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


def summarize(df_steps: pd.DataFrame, diff: pd.DataFrame,
              action_bound: float | None = None,
              rate_up_bound: float | None = None,
              rate_down_bound: float | None = None) -> dict:
    """汇总约束满足、失败次数与预测偏差。"""
    initial_applied = 0.0
    applied = df_steps["applied_u"].to_numpy(dtype=float)
    du = np.diff(applied, prepend=initial_applied)
    summary = {
        "n_steps": int(len(df_steps)),
        "n_fallback": int(df_steps["fallback"].sum()),
        "n_missing_observations": int((~df_steps["observation_available"]).sum()),
        "n_recovery_conservative": int(
            df_steps["reason"].str.startswith("observation_recovery_").sum()
        ),
        "max_commanded_u": float(df_steps["commanded_u"].abs().max()),
        "max_applied_u": float(np.abs(applied).max()),
        "max_abs_delta_u_per_step": float(np.abs(du).max()),
        "max_u_increase": float(du.max()),
        "max_u_decrease_magnitude": float(-du.min()),
        "max_command_vs_applied_abs": float(
            (df_steps["commanded_u"] - df_steps["applied_u"]).abs().max()
        ),
        "final_position_error": float(abs(
            df_steps["x"].iloc[-1] - df_steps["target"].iloc[-1])),
        "rms_plan_actual_error_x": float(
            np.sqrt(np.mean(diff.loc[diff["dim"] == "x", "error"] ** 2))) if len(diff) else 0.0,
        "rms_plan_actual_error_v": float(
            np.sqrt(np.mean(diff.loc[diff["dim"] == "v", "error"] ** 2))) if len(diff) else 0.0,
    }
    if action_bound is not None:
        summary["action_bound_violation"] = float(max(0.0, summary["max_applied_u"] - action_bound))
    if rate_up_bound is not None:
        summary["rate_up_violation"] = float(max(0.0, summary["max_u_increase"] - rate_up_bound))
    if rate_down_bound is not None:
        summary["rate_down_violation"] = float(
            max(0.0, summary["max_u_decrease_magnitude"] - rate_down_bound))
    return summary
