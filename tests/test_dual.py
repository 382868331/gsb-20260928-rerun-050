import json

import numpy as np
import pandas as pd

from control_lab.analysis import summarize_dual
from control_lab.capability import Capability
from control_lab.config import CartConfig, MPCConfig
from control_lab.dual_actuator import DualActuator, conflict_report
from control_lab.dual_controller import DualMPCController
from control_lab.dual_experiment import run_dual_experiment
from control_lab.plants import CartPlant
from control_lab.replay import replay, replay_dual


def _make(delay_a=1, delay_b=3, horizon=15):
    plant = CartPlant(CartConfig(du_up_max=1.0, du_down_max=1.0))
    ctrl = DualMPCController(plant, MPCConfig(horizon=horizon),
                             recovery_ramp_steps=4)
    return plant, ctrl, delay_a, delay_b


def test_dual_queues_keep_identity_and_separate_due_steps(tmp_path):
    plant, ctrl, da, db = _make()
    log = run_dual_experiment(
        plant, ctrl, n_steps=12, initial_state=np.array([-2.0, 0.0]),
        target_fn=lambda s: 1.5,
        actuator_delay_a=da, actuator_delay_b=db,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    # b 延迟 3 拍：前 3 拍 b 不生效；a 延迟 1 拍：第 1 拍起生效
    assert (df.loc[df["step"] < 3, "applied_u_b"] == 0.0).all()
    assert (df.loc[df["step"] >= 1, "applied_u_a"].abs() >= 0.0).all()
    # 两台队列各自保留命令身份与生效时刻，互不混用
    state = log.plans[-1]["actuator_state"]
    ids_a = [c["command_id"] for c in state["a"]["queue"]]
    ids_b = [c["command_id"] for c in state["b"]["queue"]]
    assert len(ids_a) == da and len(ids_b) == db
    assert df["applied_command_id_a"].is_unique
    assert df["applied_command_id_b"].is_unique
    # a 的未生效命令不会被算成 b 的已生效补偿
    assert log.plans[-1]["pending_a"] != log.plans[-1]["pending_b"]


def test_conflict_report_counts_cancellation(tmp_path):
    report = conflict_report(np.array([1.5]), np.array([-0.5]))
    assert report["conflict"] and report["cancellation"] == 0.5
    assert report["net"] == 1.0
    assert not conflict_report(np.array([1.0]), np.array([0.5]))["conflict"]

    # b 只能增大/保持（du_down=0）：目标反向需要刹车时，b 仍维持正向
    # 输出，与 a 的刹车方向冲突，须报告实际抵消量
    plant, ctrl, da, db = _make()
    log = run_dual_experiment(
        plant, ctrl, n_steps=30, initial_state=np.array([-2.0, 0.0]),
        target_fn=lambda s: 2.0 if s < 15 else -2.0,
        disturbance_fn=lambda s, o: -0.5,
        capability_change_fn=lambda s: {
            "a": Capability(-2.0, 2.0, 1.0, 1.0, 1),
            "b": Capability(-2.0, 2.0, 0.8, 0.0, 3),
        } if s == 0 else None,
        actuator_delay_a=da, actuator_delay_b=db,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    conflicts = df[df["conflict"]]
    assert len(conflicts) > 0
    for _, row in conflicts.iterrows():
        assert row["applied_u_a"] * row["applied_u_b"] < 0
        expected = min(abs(row["applied_u_a"]), abs(row["applied_u_b"]))
        assert abs(row["conflict_cancellation"] - expected) < 1e-9
        assert "direction_conflict" in row["unmet_reason"]


def test_degradation_restates_reachability_and_respects_limits(tmp_path):
    plant, ctrl, da, db = _make()
    changes = {
        0: {"a": Capability(-0.5, 2.0, 0.8, 0.8, 1),
            "b": Capability(-2.0, 0.5, 0.8, 0.8, 3)},
        # a 几乎失去正向能力：可达性须按当前能力重新说明
        20: {"a": Capability(-0.5, 0.2, 0.1, 0.1, 2)},
    }
    log = run_dual_experiment(
        plant, ctrl, n_steps=60, initial_state=np.array([-1.0, 0.0]),
        target_fn=lambda s: 8.0 if s >= 20 else 0.0,
        capability_change_fn=lambda s: changes.get(s),
        actuator_delay_a=da, actuator_delay_b=db,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    plans = json.loads((tmp_path / "plans.json").read_text())
    assert plans[20]["capability_event"]["a"] is not None
    assert plans[20]["plan_before_capability_change"] is not None
    # 变化后：目标超出协同可达范围时被明确报告，并给出单台可达区间
    row = df.loc[20]
    assert not row["target_attainable"]
    assert "target_unattainable" in row["unmet_reason"]
    cap = plans[20]["capability"]
    assert cap["single_b_reachable_max"] < 8.0
    assert not cap["single_b_attainable"]
    # 不为保持计划而超限：每台实际动作都在各自当前能力内
    summary = summarize_dual(df)
    assert summary["a_action_bound_violation"] == 0.0
    assert summary["b_action_bound_violation"] == 0.0
    assert summary["a_rate_up_violation"] == 0.0
    assert summary["b_rate_down_violation"] == 0.0


def test_recovery_ramps_reallocation_gradually(tmp_path):
    plant, ctrl, da, db = _make()
    changes = {
        0: {"a": Capability(-1.0, 0.3, 0.2, 0.2, 1),
            "b": Capability(-1.0, 1.0, 0.6, 0.6, 3)},
        25: {"a": Capability(-1.0, 2.0, 0.8, 0.8, 1)},
    }
    log = run_dual_experiment(
        plant, ctrl, n_steps=60, initial_state=np.array([-2.0, 0.0]),
        target_fn=lambda s: 2.0,
        capability_change_fn=lambda s: changes.get(s),
        actuator_delay_a=da, actuator_delay_b=db,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    ramps = df.loc[df["step"] >= 25, "a_ramp_factor"].to_numpy()
    assert ramps[0] < 1.0                      # 恢复首拍只放开一部分
    assert np.all(np.diff(ramps[:4]) > 0)      # 逐步放开
    assert (ramps[4:] == 1.0).all()            # 爬坡结束后恢复全量
    summary = summarize_dual(df)
    assert summary["a_rate_up_violation"] == 0.0
    assert summary["a_rate_down_violation"] == 0.0


def test_missing_observation_dual_conservative(tmp_path):
    plant, ctrl, da, db = _make()
    log = run_dual_experiment(
        plant, ctrl, n_steps=30, initial_state=np.array([0.0, 1.0]),
        target_fn=lambda s: 0.0,
        observation_available_fn=lambda s: not (10 <= s <= 12),
        actuator_delay_a=da, actuator_delay_b=db,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    missing = df[df["reason"] == "observation_missing"]
    assert list(missing["step"]) == [10, 11, 12]
    assert missing["fallback"].all()
    assert list(df[df["reason"] == "observation_recovery_age_3"]["step"]) == [13]


def test_resume_dual_state_applies_commands_once(tmp_path):
    plant, ctrl, da, db = _make()
    log = run_dual_experiment(
        plant, ctrl, n_steps=10, initial_state=np.array([-2.0, 0.0]),
        target_fn=lambda s: 1.0,
        actuator_delay_a=da, actuator_delay_b=db,
        out_dir=tmp_path / "first",
    )
    # 在两条不同延迟队列中间保存：a 队列 1 条、b 队列 3 条未生效命令
    state = log.plans[6]["actuator_state"]
    assert len(state["a"]["queue"]) == 1
    assert len(state["b"]["queue"]) == 3
    assert all(c["command_id"] > state["a"]["last_applied_command_id"]
               for c in state["a"]["queue"])

    resumed_plant = CartPlant(CartConfig(du_up_max=1.0, du_down_max=1.0))
    resumed_ctrl = DualMPCController(resumed_plant, MPCConfig(horizon=15),
                                     recovery_ramp_steps=4)
    resumed = run_dual_experiment(
        resumed_plant, resumed_ctrl, n_steps=3,
        initial_state=np.array([log.rows[6]["x_next"], log.rows[6]["v_next"]]),
        target_fn=lambda s: 1.0,
        actuator_delay_a=da, actuator_delay_b=db,
        initial_step=7, resume_actuator_state=state,
        out_dir=tmp_path / "resumed",
    )
    df = log.to_frame()
    # 续跑施加的命令与原运行同一拍一致：已生效命令不会重复施加
    for k in range(3):
        original = df.loc[7 + k]
        assert resumed.rows[k]["applied_command_id_a"] == +            original["applied_command_id_a"]
        assert resumed.rows[k]["applied_command_id_b"] == +            original["applied_command_id_b"]
        assert abs(resumed.rows[k]["applied_u_a"]
                   - original["applied_u_a"]) < 1e-9
        assert abs(resumed.rows[k]["applied_u_b"]
                   - original["applied_u_b"]) < 1e-9


def test_comparison_plans_recorded(tmp_path):
    plant, ctrl, da, db = _make()
    log = run_dual_experiment(
        plant, ctrl, n_steps=20, initial_state=np.array([-1.0, 0.0]),
        target_fn=lambda s: 1.5,
        actuator_delay_a=da, actuator_delay_b=db,
        out_dir=tmp_path,
    )
    plans = json.loads((tmp_path / "plans.json").read_text())
    entry = plans[5]
    assert set(entry["comparison"]) == {"single_a", "single_b"}
    assert entry["comparison"]["single_a"]["success"]
    assert "cost" in entry["comparison"]["single_a"]
    assert "single_a_attainable" in entry["capability"]


def test_dual_actuator_state_roundtrip(tmp_path):
    plant = CartPlant(CartConfig())
    dual = DualActuator(plant, delay_a=1, delay_b=3)
    dual.submit(np.array([0.5]), np.array([-0.3]), 0)
    dual.submit(np.array([0.6]), np.array([-0.2]), 1)
    state = dual.to_state()
    clone = DualActuator(plant, delay_a=0, delay_b=0)
    clone.load_state(state)
    # 第 1 拍时 a 的首条命令已生效，队列中只剩第 2 条
    assert clone["a"].pending_commands()[0].action[0] == 0.6
    assert clone["a"].last_applied_command_id == 1
    assert len(clone["b"].pending_commands()) == 3
    path = dual.save_state(tmp_path / "dual_state.json")
    again = DualActuator(plant)
    again.load_state_file(path)
    assert again.to_state() == state


def test_single_actuator_scenarios_still_reproduce(tmp_path):
    # 原有单执行器实验不受双执行器改动影响，结果与仓库记录一致
    destination = replay("01_constant_disturbance", tmp_path / "01")
    reproduced = pd.read_csv(destination / "steps.csv")
    recorded = pd.read_csv(
        "results/01_constant_disturbance/steps.csv")
    pd.testing.assert_frame_equal(reproduced, recorded)


def test_dual_scenario_replay_writes_manifest(tmp_path):
    destination = replay_dual("06_dual_actuator_coordination", tmp_path / "06")
    assert (destination / "steps.csv").exists()
    assert (destination / "plans.json").exists()
    spec = json.loads((destination / "replay.json").read_text())
    assert spec["kind"] == "dual"
    assert spec["actuator_delay_a"] == 1
    assert spec["actuator_delay_b"] == 3
