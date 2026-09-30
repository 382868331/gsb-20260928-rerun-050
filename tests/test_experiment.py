import json

import numpy as np
import pandas as pd

from control_lab.analysis import check_action_bounds, plan_vs_actual, summarize
from control_lab.config import CartConfig, MPCConfig
from control_lab.controller import MPCController
from control_lab.experiment import build_scenario, replay, run_experiment, run_from_spec
from control_lab.plants import CartPlant


def _run(tmp_path, **kw):
    cfg = kw.pop("cfg", CartConfig())
    mpc_cfg = kw.pop("mpc", MPCConfig(actuator_delay_steps=2, max_dropout_hold=3))
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, mpc_cfg)
    log = run_experiment(plant, ctrl, out_dir=tmp_path, **kw)
    return cfg, mpc_cfg, log


def test_disturbance_causes_plan_actual_divergence(tmp_path):
    cfg, _, log = _run(
        tmp_path,
        n_steps=80, initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 1.0,
        disturbance_fn=lambda s, o: 1.5 if s >= 20 else 0.0,
    )
    df = log.to_frame()
    diff = plan_vs_actual(tmp_path / "steps.csv", tmp_path / "plans.json")
    late = diff[(diff["plan_step"] >= 20) & (diff["dim"] == "x") & (diff["horizon"] >= 3)]
    assert (late["error"].abs() > 0.02).any()
    assert abs(df["x_true"].iloc[-1] - 1.0) < 0.4


def test_issued_vs_applied_differ_during_delay_then_converge(tmp_path):
    cfg, _, log = _run(
        tmp_path, n_steps=100, initial_state=np.array([-2.0, 0.0]),
        target_fn=lambda s: 2.0,
    )
    df = log.to_frame()
    # 延迟 2 步：issued 是 applied 的前移，且前若干步两者不同
    assert df["delay_steps"].iloc[0] == 2
    # 瞬态阶段（起步时新命令与中性填充不同）两者不同
    assert (df["issued_u"] - df["applied_u"]).abs().iloc[2] > 1e-6
    assert np.allclose(df["issued_u"].to_numpy()[:-2], df["applied_u"].to_numpy()[2:])


def test_short_dropout_holds_then_long_dropout_brakes_and_recovers(tmp_path):
    def avail(s):
        return not (20 <= s < 29)   # 连续丢 9 步
    cfg, mpc, log = _run(
        tmp_path, n_steps=120, initial_state=np.array([1.5, 1.0]),
        target_fn=lambda s: 0.0,
        disturbance_fn=lambda s, o: 0.0,
        obs_available_fn=avail,
    )
    df = log.to_frame()
    lost = df[~df["obs_available"]]
    assert len(lost) == 9
    # 前 max_hold=3 步仍尝试基于开环估计规划；之后强制保守动作
    timeout = df[df["reason"] == "observation_dropout_timeout"]
    assert len(timeout) == 9 - mpc.max_dropout_hold
    assert bool(timeout["fallback"].all())
    # 保守动作必须与速度反向（刹车）
    for _, row in timeout.iterrows():
        assert np.sign(row["issued_u"]) == -np.sign(row["v"]) or row["v"] == 0.0
    # 恢复观测后规划恢复成功，最终在目标附近停住
    assert bool(df[df["step"] >= 30]["success"].all())
    assert abs(df["x_true"].iloc[-1]) < 0.1
    assert abs(df["v_true"].iloc[-1]) < 0.1


def test_dropout_open_loop_estimate_uses_applied_not_issued(tmp_path):
    def avail(s):
        return not (10 <= s < 13)
    cfg, _, log = _run(
        tmp_path, n_steps=30, initial_state=np.array([0.0, 0.5]),
        target_fn=lambda s: 0.0, disturbance_fn=lambda s, o: 0.3,
        obs_available_fn=avail,
    )
    df = log.to_frame()
    # 丢观测期间控制器使用的是估计值，且因扰动未知，估计偏离真实状态
    lost = df[~df["obs_available"]]
    assert (lost["x"] - lost["x_true"]).abs().max() > 0.0
    # 记录中同时保留两种状态，恢复后重新对齐真实观测
    rec = df[df["step"] == 13].iloc[0]
    assert rec["obs_available"] and np.isclose(rec["x"], rec["x_true"])


def test_every_applied_action_within_asymmetric_bounds(tmp_path):
    cfg = CartConfig(du_inc_max=0.8, du_dec_max=2.0)
    cfg, mpc, log = _run(
        tmp_path, cfg=cfg,
        n_steps=120, initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 3.0 if s >= 40 else 0.0,
        disturbance_fn=lambda s, o: 0.15 * np.sin(0.5 * s),
    )
    df = log.to_frame()
    report = check_action_bounds(df, cfg.u_max, cfg.du_inc_max, cfg.du_dec_max)
    assert report["all_bounds_satisfied"]
    assert report["max_increase_per_step"] <= cfg.du_inc_max + 1e-8
    assert report["max_decrease_per_step"] <= cfg.du_dec_max + 1e-8


def test_planner_failure_uses_conservative_braking(tmp_path):
    plant = CartPlant(CartConfig())
    ctrl = MPCController(plant, MPCConfig(horizon=20, actuator_delay_steps=2))
    log = run_experiment(
        plant, ctrl, n_steps=80, initial_state=np.array([2.0, 0.0]),
        target_fn=lambda s: -1.5,
        planner_fail_steps={30, 31, 60}, out_dir=tmp_path,
    )
    df = log.to_frame()
    assert df["fallback"].sum() == 3
    assert df.loc[df["fallback"], "reason"].str.startswith("solver_exception").all()


def test_spec_replay_is_bit_identical(tmp_path):
    cfg = CartConfig()
    mpc = MPCConfig(actuator_delay_steps=2)
    spec = build_scenario(
        "replay", n_steps=60, initial_state=(1.0, -0.5),
        target_fn=lambda s: -1.0 if s >= 20 else 1.0,
        disturbance_fn=lambda s, o: 0.2 * np.sin(0.3 * s),
        obs_available_fn=lambda s: s not in {15, 16, 17},
        planner_fail_steps={40},
        cart_config=cfg, mpc_config=mpc,
    )
    d1, d2 = tmp_path / "a", tmp_path / "b"
    log1 = run_from_spec(spec, out_dir=d1)
    log2 = replay(d1 / "scenario.json", out_dir=d2)
    pd.testing.assert_frame_equal(log1.to_frame(), log2.to_frame())
    with open(d1 / "plans.json", encoding="utf-8") as f:
        plans1 = json.load(f)
    with open(d2 / "plans.json", encoding="utf-8") as f:
        plans2 = json.load(f)
    assert plans1 == plans2


def test_summary_reports_dropout_and_bounds(tmp_path):
    cfg, _, log = _run(
        tmp_path, n_steps=80, initial_state=np.array([-2.0, 0.0]),
        target_fn=lambda s: 2.0,
        obs_available_fn=lambda s: s not in {30, 31, 32, 33},
    )
    df = log.to_frame()
    diff = plan_vs_actual(tmp_path / "steps.csv", tmp_path / "plans.json")
    s = summarize(df, diff, cfg)
    assert s["n_obs_dropouts"] == 4
    assert s["max_dropout_streak"] == 4
    assert s["actuator_delay_steps"] == 2
    assert s["action_bounds"]["all_bounds_satisfied"]
    assert s["max_applied_u"] <= cfg.u_max + 1e-8
