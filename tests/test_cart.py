import numpy as np

from control_lab.config import CartConfig
from control_lab.plants import CartPlant


def test_step_matches_nominal_dynamics_without_disturbance():
    cfg = CartConfig(dt=0.1, mass=2.0)
    plant = CartPlant(cfg)
    plant.reset(np.array([1.0, 0.5]))
    obs = plant.step(np.array([2.0]))
    # v' = 0.5 + 0.1*(2/2) = 0.6; x' = 1 + 0.1*0.6 = 1.06
    assert np.allclose(obs, [1.06, 0.6], atol=1e-12)


def test_disturbance_only_affects_real_step_not_predict():
    cfg = CartConfig(mass=1.0)
    plant = CartPlant(cfg)
    plant.reset(np.array([0.0, 0.0]))
    real = plant.step(np.array([0.0]), disturbance=np.array([1.0]))
    nominal = plant.predict(np.array([0.0, 0.0]), np.array([0.0]))
    assert real[1] == 0.1            # 真实速度受扰动推动
    assert nominal[1] == 0.0         # 规划模型不含扰动


def test_state_saturation_respects_bounds():
    plant = CartPlant(CartConfig())
    plant.reset(np.array([9.99, 4.99]))
    obs = plant.step(np.array([2.0]))
    assert obs[0] <= 10.0 and obs[1] <= 5.0


def test_boundary_contract_shapes():
    plant = CartPlant()
    lo, hi = plant.observation_bounds()
    ulo, uhi = plant.action_bounds()
    dlo, dhi = plant.action_rate_bounds()
    assert lo.shape == hi.shape == (2,)
    assert ulo.shape == uhi.shape == dlo.shape == dhi.shape == (1,)
    assert np.all(lo < hi) and ulo[0] == -2.0 and uhi[0] == 2.0
    # 方向不对称：增力斜率上界与减力（卸载）斜率下界幅度不同
    assert abs(dhi[0] - 1.2) < 1e-12 and dlo[0] == -4.0


def test_asymmetric_slew_bounds_configurable():
    plant = CartPlant(CartConfig(du_inc_max=0.8, du_dec_max=2.0))
    dlo, dhi = plant.action_rate_bounds()
    assert dhi[0] == 0.8 and dlo[0] == -2.0
