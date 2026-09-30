import numpy as np

from control_lab.actuator import Actuator
from control_lab.config import CartConfig
from control_lab.plants import CartPlant


def test_actuator_delay_holds_past_actions_and_then_fifo():
    plant = CartPlant(CartConfig(du_up_max=4.0, du_down_max=4.0))
    actuator = Actuator(plant, delay_steps=2)

    first = actuator.submit(np.array([1.0]))
    second = actuator.submit(np.array([-1.0]))
    third = actuator.submit(np.array([2.0]))

    assert first == 0.0
    assert second == 0.0
    assert third == 1.0


def test_actuator_enforces_directional_slew_bounds():
    cfg = CartConfig(du_up_max=0.5, du_down_max=1.5)
    plant = CartPlant(cfg)
    actuator = Actuator(plant, delay_steps=0)
    actuator.previous_applied = np.array([0.0])

    up = actuator.submit(np.array([2.0]))
    down = actuator.submit(np.array([-2.0]))

    assert up.item() == 0.5
    assert down.item() == -1.0
    assert abs(up.item()) <= cfg.u_max


def test_projected_pending_previews_safety_clamped_delay_queue():
    cfg = CartConfig(du_up_max=0.5, du_down_max=1.5)
    plant = CartPlant(cfg)
    actuator = Actuator(plant, delay_steps=2)

    actuator.submit(np.array([2.0]))
    pending = actuator.projected_pending()

    assert np.allclose(pending, [[0.0], [0.5]])

