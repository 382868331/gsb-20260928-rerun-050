"""含执行器延迟、方向不对称动作限制与观测短时缺失的闭环场景实验。

运行：.venv/bin/python -m experiments.run_scenarios

每个场景目录额外保存 ``scenario.json``，可用 ``replay`` 完全重放。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from control_lab.analysis import plan_vs_actual, summarize
from control_lab.config import CartConfig, MPCConfig
from control_lab.controller import MPCController
from control_lab.experiment import build_scenario, run_from_spec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def build_specs() -> list:
    cfg = CartConfig()          # u_max=2, 增力斜率 1.2/步, 减力斜率 4/步
    mpc_cfg = MPCConfig(actuator_delay_steps=2, max_dropout_hold=3)
    n = 120
    specs = []

    # 场景 1：持续扰动（t=4s 起恒定向右推车）+ 执行器延迟
    specs.append(build_scenario(
        "01_constant_disturbance", n, (-3.0, 0.0),
        target_fn=lambda s: 1.0,
        disturbance_fn=lambda s, o: 0.9 if s >= 40 else 0.0,
        cart_config=cfg, mpc_config=mpc_cfg,
    ))

    # 场景 2：严格（不对称）变化限制下目标突变，验证安全退化而非激进机动
    def target_jump(step):
        if step < 40:
            return 0.0
        if step < 80:
            return 3.0
        return -2.5
    specs.append(build_scenario(
        "02_target_jump", n, (0.0, 0.0), target_jump,
        disturbance_fn=lambda s, o: 0.15 * np.sin(0.5 * s),
        cart_config=CartConfig(du_inc_max=0.8, du_dec_max=2.0),
        mpc_config=MPCConfig(actuator_delay_steps=2, max_dropout_hold=3),
    ))

    # 场景 3：连续丢观测 + 延迟恢复：t=4.0~4.7s 丢 8 步（超过保持上限 3 步），
    # 前 3 步开环保持规划，之后强制保守刹车；观测恢复后重新收敛。
    def obs_gate(step):
        return not (40 <= step < 48)
    specs.append(build_scenario(
        "03_observation_dropout", n, (2.5, 0.8),
        target_fn=lambda s: 0.0,
        disturbance_fn=lambda s, o: 0.2,
        obs_available_fn=obs_gate,
        cart_config=cfg, mpc_config=mpc_cfg,
    ))

    # 场景 4：规划器在若干周期故障（含丢观测），必须使用明确保守动作
    specs.append(build_scenario(
        "04_planner_failure", n, (2.0, 0.0),
        target_fn=lambda s: -1.5,
        disturbance_fn=lambda s, o: 0.0,
        planner_fail_steps={30, 31, 60},
        cart_config=cfg, mpc_config=mpc_cfg,
    ))
    return specs


def plot_scenario(df: pd.DataFrame, plans: list[dict], cfg: CartConfig,
                  title: str, path: Path) -> None:
    t = df["time"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    ax = axes[0]
    for p in plans[::2]:
        states = np.asarray(p["planned_states"])
        tt = (p["step"] + np.arange(states.shape[0])) * cfg.dt
        color = "tab:orange" if p["success"] else "tab:red"
        ax.plot(tt, states[:, 0], color=color, alpha=0.12, lw=1)
    ax.plot(t, df["x_true"], color="tab:blue", lw=2, label="actual position")
    ax.plot(t, df["x"], color="tab:cyan", ls=":", lw=1.2,
            label="state used by controller (estimate during dropout)")
    ax.plot(t, df["target"], color="black", ls="--", lw=1.5, label="target")
    lost = df[~df["obs_available"]]
    if len(lost):
        ax.axvspan(lost["time"].min(), lost["time"].max() + cfg.dt,
                   color="gray", alpha=0.18, label="observation dropout")
    fb = df[df["fallback"]]
    if len(fb):
        ax.scatter(fb["time"], fb["x_true"], color="red", zorder=5,
                   label="conservative (safe) action")
    ax.set_ylabel("position")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, df["v_true"], color="tab:green", lw=2, label="actual velocity")
    ax.axhline(cfg.v_bounds[0], color="gray", ls=":")
    ax.axhline(cfg.v_bounds[1], color="gray", ls=":")
    ax.set_ylabel("velocity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, df["issued_u"], color="tab:orange", lw=1.6,
            label="issued command (delayed)")
    ax.plot(t, df["applied_u"], color="tab:purple", lw=2,
            label="actually applied action")
    ax.axhline(cfg.u_max, color="gray", ls=":")
    ax.axhline(-cfg.u_max, color="gray", ls=":")
    ax.set_ylabel("force")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(t, df["disturbance"], color="tab:red", alpha=0.6, label="disturbance d")
    ax.plot(t, df["dropout_streak"], color="tab:blue", alpha=0.6,
            label="consecutive lost observations")
    ax.set_ylabel("disturbance / dropout")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    titles = {
        "01_constant_disturbance": "Scenario 1: constant disturbance + actuator delay",
        "02_target_jump": "Scenario 2: target jumps under asymmetric slew limits",
        "03_observation_dropout": "Scenario 3: observation dropout -> open-loop hold -> safe brake -> recovery",
        "04_planner_failure": "Scenario 4: planner failure -> conservative braking",
    }
    summaries = {}
    for spec in build_specs():
        d = OUT / spec.name
        log = run_from_spec(spec, out_dir=d)
        df = log.to_frame()
        cfg = CartConfig(**spec.cart)
        diff = plan_vs_actual(d / "steps.csv", d / "plans.json")
        diff.to_csv(d / "plan_vs_actual.csv", index=False)
        summaries[spec.name] = summarize(df, diff, cfg)
        plot_scenario(df, log.plans, cfg, titles[spec.name], d / "plot.png")

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
