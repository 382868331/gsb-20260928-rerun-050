import numpy as np

from control_lab.config import CartConfig, MPCConfig
from control_lab.controller import MPCController
from control_lab.plants import CartPlant


def _setup(**kw):
    cart_kw = kw.pop("cart", {})
    delay = kw.pop("delay", 0)
    cfg = CartConfig(**{"du_inc_max": 4.0, "du_dec_max": 4.0, **cart_kw})
    plant = CartPlant(cfg)
    ctrl = MPCController(plant, MPCConfig(actuator_delay_steps=delay, **kw))
    return cfg, plant, ctrl


def test_mpc_reaches_target_and_stops():
    cfg, plant, ctrl = _setup(horizon=30)
    plant.reset(np.array([0.0, 0.0]))
    obs = plant.get_observation()
    committed = ctrl.initial_committed()
    for _ in range(100):
        res = ctrl.plan(obs, 2.0, committed)
        assert res.success
        applied = ctrl.applied_this_step(committed)
        obs = plant.step(applied)
        committed = ctrl.advance_committed(committed, res.action)
    assert abs(obs[0] - 2.0) < 5e-2
    assert abs(obs[1]) < 5e-2


def test_action_magnitude_and_asymmetric_rate_constraints_hold():
    cfg, plant, ctrl = _setup(
        horizon=20, cart={"du_inc_max": 1.2, "du_dec_max": 4.0})
    plant.reset(np.array([-4.0, 0.0]))
    obs = plant.get_observation()
    committed = ctrl.initial_committed()
    prev = 0.0
    for _ in range(60):
        res = ctrl.plan(obs, 3.0, committed)
        assert res.success
        u_seq = res.planned_actions.reshape(-1)
        assert np.all(u_seq <= cfg.u_max + 1e-8)
        assert np.all(u_seq >= -cfg.u_max - 1e-8)
        diffs = np.diff(u_seq)
        assert np.all(diffs <= cfg.du_inc_max + 1e-8)
        assert np.all(diffs >= -cfg.du_dec_max - 1e-8)
        # 本周期生效的是队列最旧命令，其相对上一生效动作也要满足限制
        applied = ctrl.applied_this_step(committed)
        delta = float(applied[0]) - prev
        assert -cfg.du_dec_max - 1e-8 <= delta <= cfg.du_inc_max + 1e-8
        obs = plant.step(applied)
        committed = ctrl.advance_committed(committed, res.action)
        prev = float(applied[0])


def test_delay_keeps_committed_prefix_fixed():
    cfg, plant, ctrl = _setup(horizon=25, delay=3)
    plant.reset(np.array([0.0, 0.0]))
    obs = plant.get_observation()
    committed = ctrl.initial_committed()
    res = ctrl.plan(obs, 2.0, committed)
    # 前 D 步预测动作必须等于在途队列（中性），新命令到第 D 步才出现
    u_seq = res.planned_actions.reshape(-1)
    assert np.allclose(u_seq[:3], committed)
    assert not np.isclose(u_seq[3], committed[-1]) or res.action[0] != committed[-1]
    # 本周期生效动作绝不是刚规划出的命令
    applied = ctrl.applied_this_step(committed)
    assert np.allclose(applied, committed[0])
    assert not np.allclose(applied, res.action)


def test_delay_queue_fifo_ordering():
    cfg, plant, ctrl = _setup(horizon=25, delay=2)
    committed = ctrl.initial_committed()
    issued = [np.array([0.5]), np.array([-0.5]), np.array([1.0])]
    applied = []
    for u in issued:
        applied.append(ctrl.applied_this_step(committed).copy())
        committed = ctrl.advance_committed(committed, u)
    # 延迟 D=2：命令在队列里走两格后生效（FIFO）。
    # [n,n] -> 步1 生效 n，追加 0.5 -> [n,0.5]
    #        -> 步2 生效 n，追加 -0.5 -> [0.5,-0.5]
    #        -> 步3 生效 0.5，追加 1.0 -> [-0.5,1.0]
    assert np.allclose(applied[0], ctrl.neutral)
    assert np.allclose(applied[1], ctrl.neutral)
    assert np.isclose(float(applied[2][0]), 0.5)
    # 再前进一步：步4 生效步2发出的 -0.5
    assert np.isclose(float(ctrl.applied_this_step(committed)[0]), -0.5)


def test_delayed_closed_loop_reaches_target():
    cfg, plant, ctrl = _setup(horizon=30, delay=2)
    plant.reset(np.array([0.0, 0.0]))
    obs = plant.get_observation()
    committed = ctrl.initial_committed()
    for _ in range(120):
        res = ctrl.plan(obs, 2.0, committed)
        assert res.success
        applied = ctrl.applied_this_step(committed)
        obs = plant.step(applied)
        committed = ctrl.advance_committed(committed, res.action)
    assert abs(obs[0] - 2.0) < 0.08
    assert abs(obs[1]) < 0.08


def test_fallback_is_braking_and_respects_asymmetric_limits():
    cfg, plant, ctrl = _setup(
        horizon=10, max_iter=0, cart={"du_inc_max": 0.5, "du_dec_max": 2.0})
    plant.reset(np.array([0.0, 1.5]))
    res = ctrl.plan(plant.get_observation(), 0.0, ctrl.initial_committed())
    assert res.fallback_used and not res.success
    u = float(res.action[0])
    assert u < 0.0
    assert abs(u) <= cfg.u_max + 1e-8
    # 增力（此处数值变大）方向不得超过严格的 0.5/步
    assert u - 0.0 <= cfg.du_inc_max + 1e-8


def test_force_fallback_on_untrusted_prediction():
    cfg, plant, ctrl = _setup(horizon=20, delay=2)
    res = ctrl.plan(np.array([1.0, -1.2]), 0.0, ctrl.initial_committed(),
                    force_fallback=True)
    assert res.fallback_used
    assert res.reason == "observation_dropout_timeout"
    assert float(res.action[0]) > 0.0  # 与负速度反向 = 刹车
