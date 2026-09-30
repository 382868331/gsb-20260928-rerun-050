"""计划 vs 执行差异分析，以及逐步动作边界核对。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def plan_vs_actual(csv_path: str | Path, plans_path: str | Path,
                   max_horizon: int | None = None) -> pd.DataFrame:
    """比较每个成功周期预测的第 k 步与之后真实发生的状态。

    实际轨迹取设备**真实状态**（``x_true``/``v_true``），即观测缺失期间也以
    真值核对，而预测来自含在途命令前缀的标称（无扰动）模型。返回长表：
    plan_step, horizon, dim, predicted, actual, error。
    """
    df = pd.read_csv(csv_path)
    with open(plans_path, encoding="utf-8") as f:
        plans = {p["step"]: p for p in json.load(f)}

    n_total = len(df)
    states = df[["x_true", "v_true"]].to_numpy()
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


def check_action_bounds(df: pd.DataFrame, u_max: float, du_inc_max: float,
                        du_dec_max: float, tol: float = 1e-8) -> dict:
    """逐步核对每一条**真正生效**动作的幅度与方向不对称变化限制。"""
    applied = df["applied_u"].to_numpy(dtype=float)
    deltas = np.diff(applied, prepend=applied[0])
    violations = {
        "magnitude": int(np.sum(np.abs(applied) > u_max + tol)),
        "increase_rate": int(np.sum(deltas > du_inc_max + tol)),
        "decrease_rate": int(np.sum(-deltas > du_dec_max + tol)),
    }
    return {
        "max_applied_abs_u": float(np.max(np.abs(applied))),
        "max_increase_per_step": float(deltas.max()),
        "max_decrease_per_step": float((-deltas).max()),
        "bound_violations": violations,
        "all_bounds_satisfied": all(v == 0 for v in violations.values()),
    }


def summarize(df_steps: pd.DataFrame, diff: pd.DataFrame, cfg=None) -> dict:
    """汇总：延迟/丢观测统计、保守动作原因、约束满足与预测偏差。"""
    applied = df_steps["applied_u"].to_numpy()
    deltas = np.diff(applied, prepend=applied[0])
    fb = df_steps[df_steps["fallback"]]
    summary = {
        "n_steps": int(len(df_steps)),
        "actuator_delay_steps": int(df_steps["delay_steps"].iloc[0]),
        "n_obs_dropouts": int((~df_steps["obs_available"]).sum()),
        "max_dropout_streak": int(df_steps["dropout_streak"].max()),
        "n_fallback": int(df_steps["fallback"].sum()),
        "fallback_reasons": {
            reason: int((fb["reason"] == reason).sum())
            for reason in sorted(fb["reason"].unique())
        } if len(fb) else {},
        "max_applied_u": float(np.max(np.abs(applied))),
        "max_increase_per_step": float(deltas.max()),
        "max_decrease_per_step": float((-deltas).max()),
        "max_command_minus_applied_abs": float(
            (df_steps["issued_u"] - df_steps["applied_u"]).abs().max()),
        "final_position_error": float(abs(
            df_steps["x_true"].iloc[-1] - df_steps["target"].iloc[-1])),
        "rms_plan_actual_error_x": float(
            np.sqrt(np.mean(diff.loc[diff["dim"] == "x", "error"] ** 2)))
            if len(diff) else 0.0,
        "rms_plan_actual_error_v": float(
            np.sqrt(np.mean(diff.loc[diff["dim"] == "v", "error"] ** 2)))
            if len(diff) else 0.0,
    }
    if cfg is not None:
        summary["action_bounds"] = check_action_bounds(
            df_steps, cfg.u_max, cfg.du_inc_max, cfg.du_dec_max)
    return summary
