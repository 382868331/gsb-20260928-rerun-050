import numpy as np

from control_lab.actuator import Actuator
from control_lab.capability import Capability
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


def test_capability_change_preserves_queue_and_records_effective_times():
    cfg = CartConfig(du_up_max=0.9, du_down_max=1.2)
    plant = CartPlant(cfg)
    actuator = Actuator(plant, delay_steps=2)
    actuator.submit(np.array([0.7]), step=10)
    actuator.submit(np.array([0.8]), step=11)
    retained = [item.command_id for item in actuator.pending_commands()]

    event = actuator.update_capability(
        12, Capability(-2.0, 1.0, 0.2, 0.3, delay_steps=4)
    )

    assert event["retained_command_ids"] == retained
    assert event["withdrawn_command_ids"] == []
    assert not event["withdraw_allowed"]
    assert event["inserted_neutral_commands"] == 2
    pending = actuator.pending_commands()
    assert [item.command_id for item in pending[-2:]] == retained
    assert [item.due_step for item in pending[-2:]] == [14, 15]

    withdrawal = actuator.withdraw_pending(12)
    assert not withdrawal["allowed"]
    assert len(actuator.pending_commands()) == 4
