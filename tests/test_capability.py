import json

import numpy as np

from control_lab.actuator import Actuator
from control_lab.analysis import plan_vs_actual, summarize
from control_lab.capability import ActuatorCapability
from control_lab.config import CartConfig, MPCConfig
from control_lab.controller import MPCController
from control_lab.experiment import run_experiment
from control_lab.plants import CartPlant
from control_lab.replay import capability_change, replay


def _plant(**kw):
    return CartPlant(CartConfig(**kw))


def test_capability_change_keeps_pending_and_reports_effective_steps():
    plant = _plant(du_up_max=4.0, du_down_max=4.0)
    actuator = Actuator(plant, delay_steps=2)
    actuator.submit(np.array([1.0]))
    actuator.submit(np.array([0.5]))

    degraded = ActuatorCapability(u_lo=-0.5, u_hi=0.5, du_up=1.0,
                                  du_down=1.0, delay_steps=4)
    report = actuator.set_capability(degraded, step=10)

    assert report["pending_policy"] == "kept_not_withdrawable"
    # 已排队的两条命令保留在原位置，生效时刻为第 10、11 拍
    assert [p["effective_step"] for p in report["pending"]] == [10, 11]
    assert [p["command"] for p in report["pending"]] == [[1.0], [0.5]]
    # 延迟增长后队列变长，新槽位用队尾保持填充
    assert actuator.delay_steps == 4
    assert len(actuator.pending()) == 4
    # 排队命令仍按原顺序生效（生效时按新能力裁剪），不被重复施加
    assert actuator.submit(np.array([0.0])).item() == 0.5
    assert actuator.submit(np.array([0.0])).item() == 0.5


def test_delay_shrink_does_not_drop_queued_commands():
    plant = _plant()
    actuator = Actuator(plant, delay_steps=3)
    for u in (0.3, 0.2, 0.1):
        actuator.submit(np.array([u]))
    restored = ActuatorCapability(u_lo=-2.0, u_hi=2.0, du_up=4.0,
                                  du_down=4.0, delay_steps=1)
    report = actuator.set_capability(restored, step=5)
    # 实际延迟取 max(新延迟, 队列长度)，已排队命令不丢失
    assert report["effective_delay_steps"] == 3
    assert actuator.delay_steps == 3
    applied = [actuator.submit(np.array([0.0])).item() for _ in range(3)]
    assert applied == [0.3, 0.2, 0.1]


def test_degraded_capability_clips_applied_action():
    plant = _plant()
    actuator = Actuator(plant, delay_steps=0)
    actuator.set_capability(ActuatorCapability(
        u_lo=-0.4, u_hi=0.4, du_up=1.0, du_down=1.0), step=0)
    assert abs(actuator.submit(np.array([2.0])).item()) <= 0.4 + 1e-12
    # 只能单向（减小）调节：动作不得增大
    actuator.set_capability(ActuatorCapability(
        u_lo=-0.4, u_hi=0.4, du_up=1.0, du_down=1.0,
        direction="decrease_only"), step=1)
    prev = actuator.previous_applied.item()
    applied = actuator.submit(np.array([0.4])).item()
    assert applied <= prev + 1e-12


def test_controller_replans_with_new_capability_and_reports_reachability():
    plant = _plant()
    ctrl = MPCController(plant, MPCConfig(horizon=15))
    plant.reset(np.array([0.0, 0.0]))
    res = ctrl.plan(plant.get_observation(), 2.0)
    assert abs(res.action[0]) > 0.5

    ctrl.set_capability(ActuatorCapability(
        u_lo=-0.3, u_hi=0.3, du_up=0.5, du_down=0.5))
    res2 = ctrl.plan(plant.get_observation(), 2.0)
    assert np.all(np.abs(res2.planned_actions) <= 0.3 + 1e-8)

    reach = ctrl.assess_reachability([2.0, -1.0])
    assert reach["can_hold_steady_state"]
    assert all(e["reachable"] for e in reach["targets"])

    # 动作范围不含 0（只能正向推）时，任何定点目标都无法长期保持
    ctrl.set_capability(ActuatorCapability(
        u_lo=0.2, u_hi=0.6, du_up=0.5, du_down=0.5))
    reach2 = ctrl.assess_reachability([2.0])
    assert not reach2["can_hold_steady_state"]
    assert not reach2["targets"][0]["reachable"]


def test_observation_loss_during_degraded_capability_stays_conservative(tmp_path):
    plant = _plant()
    ctrl = MPCController(plant, MPCConfig(horizon=15))
    degraded = ActuatorCapability(u_lo=-0.6, u_hi=0.6, du_up=0.5,
                                  du_down=0.5, delay_steps=2)
    log = run_experiment(
        plant, ctrl, n_steps=40, initial_state=np.array([0.0, 1.0]),
        target_fn=lambda s: 1.0,
        disturbance_fn=lambda s, o: 0.2,
        observation_available_fn=lambda s: not (15 <= s <= 19),
        capability_fn=lambda s: degraded if s == 10 else None,
        actuator_delay_steps=2,
        out_dir=tmp_path,
    )
    df = log.to_frame()
    missing = df[df["reason"] == "observation_missing"]
    assert list(missing["step"]) == [15, 16, 17, 18, 19]
    # 缺观测保守动作同样遵守降级后的能力边界
    assert (missing["applied_u"].abs() <= 0.6 + 1e-8).all()
    applied = df["applied_u"].to_numpy()
    du = np.diff(applied, prepend=0.0)
    after = df["step"] >= 10
    assert np.abs(applied[after]).max() <= 0.6 + 1e-8
    assert du[after].max() <= 0.5 + 1e-8
    assert -du[after].min() <= 0.5 + 1e-8
    # 能力变化拍记录了对照计划与可达性说明
    plans = json.loads((tmp_path / "plans.json").read_text())
    change = plans[10]["capability_change"]
    assert "plan_before" in change and "plan_after" in change
    assert change["reachability"]["can_hold_steady_state"]
    assert change["report"]["pending_policy"] == "kept_not_withdrawable"


def test_checkpoint_restore_does_not_duplicate_queued_actions(tmp_path):
    plant = _plant(du_up_max=4.0, du_down_max=4.0)
    actuator = Actuator(plant, delay_steps=3)
    for u in (0.7, -0.2, 0.4):
        actuator.submit(np.array([u]))
    checkpoint = actuator.checkpoint()

    restored = Actuator.restore(plant, checkpoint)
    assert [q.tolist() for q in restored.pending()] == \
        [q.tolist() for q in actuator.pending()]
    # 恢复后续提交的生效序列与不中断运行一致：队列动作只施加一次
    expected = [actuator.submit(np.array([0.1])).item() for _ in range(4)]
    actual = [restored.submit(np.array([0.1])).item() for _ in range(4)]
    assert np.allclose(actual, expected)


def test_experiment_checkpoint_allows_resume_without_reapply(tmp_path):
    plant = _plant()
    ctrl = MPCController(plant, MPCConfig(horizon=15))
    log = run_experiment(
        plant, ctrl, n_steps=20, initial_state=np.array([-1.0, 0.0]),
        target_fn=lambda s: 1.0, actuator_delay_steps=3, out_dir=tmp_path,
    )
    checkpoint = json.loads((tmp_path / "actuator_checkpoint.json").read_text())
    assert len(checkpoint["queue"]) == 3
    # 用检查点重开：队列中动作来自恢复的状态，而非重新提交历史
    plant2 = _plant()
    ctrl2 = MPCController(plant2, MPCConfig(horizon=15))
    actuator = Actuator.restore(plant2, checkpoint)
    queued = [q[0] for q in checkpoint["queue"]]
    applied = [actuator.submit(np.array([0.0])).item() for _ in range(3)]
    lo, hi = actuator.capability.action_bounds()
    expected = [float(np.clip(u, lo[0], hi[0])) for u in queued]
    assert np.allclose(applied, expected)


def test_capability_scenario_recovers_and_satisfies_bounds(tmp_path):
    destination = replay("05_capability_degradation_recovery", tmp_path)
    assert (destination / "actuator_checkpoint.json").exists()
    import pandas as pd
    df = pd.read_csv(destination / "steps.csv")
    diff = plan_vs_actual(destination / "steps.csv", destination / "plans.json")
    s = summarize(df, diff)
    assert s["action_bound_violation"] == 0.0
    assert s["rate_up_violation"] == 0.0
    assert s["rate_down_violation"] == 0.0
    assert s["n_capability_changes"] == 3
    # 缺观测与单向降级重叠期间仍走保守动作
    overlap = df[(df["step"] >= 72) & (df["step"] <= 77)]
    assert (overlap["reason"] == "observation_missing").all()
    # 恢复正常能力后平稳回到目标
    assert s["final_position_error"] < 0.15
    plans = json.loads((destination / "plans.json").read_text())
    changes = [p for p in plans if "capability_change" in p]
    assert [p["step"] for p in changes] == [45, 70, 95]
    for p in changes:
        assert "plan_before" in p["capability_change"]
        assert "plan_after" in p["capability_change"]
        assert "reachability" in p["capability_change"]
