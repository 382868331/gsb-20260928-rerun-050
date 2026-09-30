import numpy as np

from control_lab.config import CartConfig, MPCConfig
from control_lab.controller import MPCController
from control_lab.plants import CartPlant


def _setup(**kw):
    cfg = CartConfig(**{"du_max": 4.0, "u_max": 2.0, **kw.pop("cart", {})})
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, MPCConfig(**kw))
    return cfg, plant, ctrl


def test_mpc_reaches_target_and_stops():
    cfg, plant, ctrl = _setup(horizon=30)
    plant.reset(np.array([0.0, 0.0]))
    obs = plant.get_observation()
    for _ in range(100):
        res = ctrl.plan(obs, 2.0)
        assert res.success
        obs = plant.step(res.action)
        ctrl.prev_action = res.action.copy()
    assert abs(obs[0] - 2.0) < 5e-2
    assert abs(obs[1]) < 5e-2


def test_action_magnitude_and_rate_constraints_hold():
    cfg, plant, ctrl = _setup(horizon=20)
    plant.reset(np.array([-4.0, 0.0]))
    obs = plant.get_observation()
    prev = np.array([0.0])
    for _ in range(60):
        res = ctrl.plan(obs, 3.0)
        assert res.success
        assert np.abs(res.action)[0] <= cfg.u_max + 1e-8
        assert abs(float(res.action[0]) - float(prev[0])) <= cfg.du_max + 1e-8
        # 整条计划内也要满足两类约束
        u = res.planned_actions.reshape(-1)
        assert np.all(u <= cfg.u_max + 1e-8) and np.all(u >= -cfg.u_max - 1e-8)
        assert np.max(np.abs(np.diff(u))) <= cfg.du_max + 1e-8
        assert abs(u[0] - float(prev[0])) <= cfg.du_max + 1e-8
        obs = plant.step(res.action)
        prev = res.action.copy()
        ctrl.prev_action = prev.copy()


def test_only_first_action_executed_and_replan_changes_plan():
    _, plant, ctrl = _setup(horizon=25)
    plant.reset(np.array([0.0, 0.0]))
    obs = plant.get_observation()
    r1 = ctrl.plan(obs, 2.0)
    obs = plant.step(r1.action)
    r2 = ctrl.plan(obs, 2.0)
    # 基于新观测重新规划，第二份计划从新状态出发
    assert np.allclose(r2.planned_states[0], obs)
    assert not np.allclose(r1.planned_states[1], r2.planned_states[1], atol=1e-4) or \
           np.linalg.norm(r1.action - r2.action) > 1e-6


def test_delayed_prediction_uses_pending_action_before_new_command():
    _, plant, ctrl = _setup(horizon=8)
    ctrl.pending_actions = [np.array([0.0])]
    plant.reset(np.array([1.0, 0.5]))
    res = ctrl.plan(plant.get_observation(), 0.0)
    assert res.success
    assert np.isclose(res.planned_actions[0, 0], 0.0)
    assert np.isclose(res.planned_actions[1, 0], res.action[0])


def test_fallback_is_braking_and_respects_hard_constraints():
    cfg, plant, ctrl = _setup(horizon=10, max_iter=0)
    plant.reset(np.array([0.0, 1.5]))
    res = ctrl.plan(plant.get_observation(), 0.0)
    # max_iter=0 时 SLSQP 无法收敛 -> 必须走保守动作
    assert res.fallback_used and not res.success
    u = float(res.action[0])
    assert u < 0.0  # 与速度反向 = 刹车
    assert abs(u) <= cfg.u_max + 1e-8
    # 保守动作也不能违反变化速度限制
    assert abs(u) <= cfg.du_max + 1e-8


def test_fallback_after_moving_action_respects_rate_limit():
    cfg, plant, ctrl = _setup(horizon=10, max_iter=50)
    plant.reset(np.array([-3.0, 0.0]))
    obs = plant.get_observation()
    # 先让上一动作达到正饱和附近
    r = ctrl.plan(obs, 3.0)
    ctrl.prev_action = np.array([2.0])
    plant.step(np.array([2.0]))
    obs = plant.get_observation()
    ctrl.cfg = MPCConfig(horizon=10, max_iter=0)
    res = ctrl.plan(obs, 3.0)
    assert res.fallback_used
    # 从 +2 起步，保守动作最大变化为 du_max=4，区间 [-2, 2] 内合法
    assert -2.0 - 1e-8 <= float(res.action[0]) <= 2.0 + 1e-8
    assert abs(float(res.action[0]) - 2.0) <= cfg.du_max + 1e-8


def test_fallback_respects_asymmetric_rate_limits():
    cfg = CartConfig(du_up_max=0.5, du_down_max=1.5)
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, MPCConfig(horizon=10, max_iter=0))
    ctrl.prev_action = np.array([1.0])
    plant.reset(np.array([0.0, 1.0]))
    res = ctrl.plan(plant.get_observation(), 0.0)
    assert res.fallback_used
    delta = float(res.action[0] - 1.0)
    assert -cfg.du_down_max - 1e-8 <= delta <= cfg.du_up_max + 1e-8
