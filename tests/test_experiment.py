import json

import numpy as np

from control_lab.analysis import plan_vs_actual, summarize
from control_lab.config import CartConfig, MPCConfig
from control_lab.controller import MPCController
from control_lab.experiment import run_experiment
from control_lab.plants import CartPlant
from control_lab.replay import SPECS, replay


def test_disturbance_causes_plan_actual_divergence(tmp_path):
    cfg = CartConfig()
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, MPCConfig(horizon=20))
    log = run_experiment(
        plant, ctrl, n_steps=60, initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 1.0,
        disturbance_fn=lambda s, o: 1.5 if s >= 20 else 0.0,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    diff = plan_vs_actual(tmp_path / "steps.csv", tmp_path / "plans.json")
    # 扰动后真实位置明显偏离无扰动预测
    late = diff[(diff["plan_step"] >= 20) & (diff["dim"] == "x") & (diff["horizon"] >= 3)]
    assert (late["error"].abs() > 0.02).any()
    # 但闭环最终仍把扰动造成的偏差压到有界范围（无积分作用存在小稳态静差）
    assert abs(df["x"].iloc[-1] - 1.0) < 0.2


def test_target_jump_is_tracked(tmp_path):
    plant = CartPlant(CartConfig())
    ctrl = MPCController(plant, MPCConfig(horizon=25))
    log = run_experiment(
        plant, ctrl, n_steps=120, initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 2.0 if s >= 40 else 0.0,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    assert abs(df["x"].iloc[35] - 0.0) < 0.1
    assert abs(df["x"].iloc[-1] - 2.0) < 0.1


def test_logger_records_every_step_and_reason(tmp_path):
    plant = CartPlant(CartConfig())
    ctrl = MPCController(plant, MPCConfig(horizon=10))
    log = run_experiment(
        plant, ctrl, n_steps=10, initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 1.0, out_dir=tmp_path,
    )
    df = log.to_frame()
    assert len(df) == 10
    assert set(df["reason"]) == {"ok"}
    plans = json.loads((tmp_path / "plans.json").read_text())
    assert len(plans) == 10 and len(plans[0]["planned_states"]) == 11


def test_summary_reports_constraint_satisfaction(tmp_path):
    cfg = CartConfig()
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, MPCConfig(horizon=20))
    log = run_experiment(
        plant, ctrl, n_steps=80, initial_state=np.array([-2.0, 0.0]),
        target_fn=lambda s: 2.0, out_dir=tmp_path,
    )
    df = log.to_frame()
    diff = plan_vs_actual(tmp_path / "steps.csv", tmp_path / "plans.json")
    s = summarize(df, diff)
    assert s["n_fallback"] == 0
    assert s["max_applied_u"] <= cfg.u_max + 1e-8
    assert s["max_abs_delta_u_per_step"] <= cfg.du_max + 1e-8
    assert s["final_position_error"] < 0.05


def test_delayed_commands_are_not_applied_immediately(tmp_path):
    cfg = CartConfig(du_up_max=4.0, du_down_max=4.0)
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, MPCConfig(horizon=10))
    log = run_experiment(
        plant, ctrl, n_steps=8, initial_state=np.array([-3.0, 0.0]),
        target_fn=lambda s: 3.0,
        actuator_delay_steps=3,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    assert np.all(df.loc[df["step"] < 3, "applied_u"] == 0.0)
    assert (df.loc[df["step"] < 3, "commanded_u"].abs() > 0.0).all()
    assert np.all(df["actuator_delay_steps"] == 3)


def test_missing_observations_use_conservative_action_and_recovery_step(tmp_path):
    cfg = CartConfig(du_up_max=0.8, du_down_max=2.5)
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, MPCConfig(horizon=15))
    log = run_experiment(
        plant, ctrl, n_steps=30, initial_state=np.array([0.0, 1.5]),
        target_fn=lambda s: 0.0,
        disturbance_fn=lambda s, o: 0.2,
        observation_available_fn=lambda s: not (10 <= s <= 13),
        actuator_delay_steps=2,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    missing = df[df["reason"] == "observation_missing"]
    recovery = df[df["reason"] == "observation_recovery_age_4"]
    assert list(missing["step"]) == [10, 11, 12, 13]
    assert list(recovery["step"]) == [14]
    assert missing["fallback"].all() and recovery["fallback"].all()
    assert (missing["x_used"] == missing["x"].iloc[0]).all()


def test_asymmetric_applied_action_bounds_hold_under_target_jump(tmp_path):
    cfg = CartConfig(du_up_max=0.8, du_down_max=2.5)
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, MPCConfig(horizon=20))
    log = run_experiment(
        plant, ctrl, n_steps=80, initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 3.0 if s >= 20 else 0.0,
        disturbance_fn=lambda s, o: 0.4,
        actuator_delay_steps=2,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    applied = df["applied_u"].to_numpy()
    delta = np.diff(applied, prepend=0.0)
    assert np.abs(applied).max() <= cfg.u_max + 1e-8
    assert delta.max() <= cfg.du_up_max + 1e-8
    assert -delta.min() <= cfg.du_down_max + 1e-8


def test_builtin_scenario_replay_writes_manifest(tmp_path):
    name = "04_delay_asymmetric_rate_observation_loss"
    destination = replay(name, tmp_path)
    assert (destination / "steps.csv").exists()
    assert (destination / "plans.json").exists()
    assert (destination / "replay.json").exists()
    assert SPECS[name].delay_steps == 3
