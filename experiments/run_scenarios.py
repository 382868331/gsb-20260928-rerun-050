"""受扰动与目标突变等场景下的闭环实验，输出记录与计划-执行差异图。

运行：.venv/bin/python -m experiments.run_scenarios
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from control_lab.analysis import plan_vs_actual, summarize
from control_lab.config import CartConfig, MPCConfig
from control_lab.controller import MPCController
from control_lab.experiment import run_experiment
from control_lab.plants import CartPlant

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


class FailureInjectingController(MPCController):
    """在指定周期令求解器抛异常，验证失败时走明确保守动作而非旧预测。"""

    def __init__(self, *args, fail_steps: set[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_count = 0
        self.fail_steps = fail_steps or set()

    def plan(self, obs, target):
        current = self._step_count
        self._step_count += 1
        if current in self.fail_steps:
            import scipy.optimize as opt
            orig = opt.minimize
            def boom(*a, **k):
                raise RuntimeError("injected_solver_failure")
            opt.minimize = boom  # type: ignore[assignment]
            module = sys.modules["control_lab.controller"]
            orig_ctrl = module.minimize
            module.minimize = boom  # type: ignore[assignment]
            try:
                return super().plan(obs, target)
            finally:
                opt.minimize = orig  # type: ignore[assignment]
                module.minimize = orig_ctrl  # type: ignore[assignment]
        return super().plan(obs, target)


def plot_scenario(df: pd.DataFrame, plans: list[dict], cfg: CartConfig, title: str, path: Path) -> None:
    t = df["time"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax = axes[0]
    # 每次重规划的整条预测轨迹（淡色）：展示计划随新观测不断修正
    for p in plans[::2]:
        states = np.asarray(p["planned_states"])
        k0 = p["step"]
        tt = (k0 + np.arange(states.shape[0])) * cfg.dt
        color = "tab:orange" if p["success"] else "tab:red"
        ax.plot(tt, states[:, 0], color=color, alpha=0.12, lw=1)
    ax.plot(t, df["x"], color="tab:blue", lw=2, label="actual position x")
    ax.plot(t, df["target"], color="black", ls="--", lw=1.5, label="target")
    ax.fill_between(t, *cfg.x_bounds, color="gray", alpha=0.08)
    fb = df[df["fallback"]]
    if len(fb):
        ax.scatter(fb["time"], fb["x"], color="red", zorder=5, label="plan failed (safe action)")
    ax.set_ylabel("position")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, df["v"], color="tab:green", lw=2, label="actual velocity v")
    ax.axhline(cfg.v_bounds[0], color="gray", ls=":")
    ax.axhline(cfg.v_bounds[1], color="gray", ls=":")
    ax.set_ylabel("velocity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, df["applied_u"], color="tab:purple", lw=2, label="applied action u")
    ax.plot(t, df["planned_u0"], color="tab:orange", ls="--", lw=1, alpha=0.7,
            label="planned first action (brake on failure)")
    ax.plot(t, df["disturbance"], color="tab:red", alpha=0.5, label="disturbance d")
    ax.axhline(cfg.u_max, color="gray", ls=":")
    ax.axhline(-cfg.u_max, color="gray", ls=":")
    ax.set_ylabel("force")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cfg = CartConfig()
    mpc_cfg = MPCConfig()
    n_steps = 120
    summaries = {}

    # 场景 1：持续外部扰动力（t=4s 起恒定向右推车，控制器不知道该扰动）
    def disturbance_const(step, obs):
        return 0.9 if step >= 40 else 0.0
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, mpc_cfg)
    log = run_experiment(plant, ctrl, n_steps, np.array([-3.0, 0.0]),
                         target_fn=lambda s: 1.0, disturbance_fn=disturbance_const,
                         out_dir=OUT / "01_constant_disturbance")

    # 场景 2：目标突变（4s 与 8s 两次阶跃）
    def target_jump(step):
        if step < 40:
            return 0.0
        if step < 80:
            return 3.0
        return -2.5
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, mpc_cfg)
    log2 = run_experiment(plant, ctrl, n_steps, np.array([0.0, 0.0]),
                          target_fn=target_jump,
                          disturbance_fn=lambda s, o: 0.15 * np.sin(0.5 * s),
                          out_dir=OUT / "02_target_jump")

    # 场景 3：规划器在若干周期故障，必须使用明确保守刹车动作
    plant = CartPlant(cfg)
    ctrl = FailureInjectingController(plant, mpc_cfg, fail_steps={30, 31, 60})
    log3 = run_experiment(plant, ctrl, n_steps, np.array([2.0, 0.0]),
                          target_fn=lambda s: -1.5,
                          disturbance_fn=lambda s, o: 0.0,
                          out_dir=OUT / "03_planner_failure")

    for name, lg, title in [
        ("01_constant_disturbance", log, "Scenario 1: constant disturbance, receding-horizon replanning"),
        ("02_target_jump", log2, "Scenario 2: target jumps + mild periodic disturbance"),
        ("03_planner_failure", log3, "Scenario 3: planner failure -> conservative braking"),
    ]:
        d = OUT / name
        df = lg.to_frame()
        diff = plan_vs_actual(d / "steps.csv", d / "plans.json")
        diff.to_csv(d / "plan_vs_actual.csv", index=False)
        summaries[name] = summarize(df, diff)
        plot_scenario(df, lg.plans, cfg, title, d / "plot.png")

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
