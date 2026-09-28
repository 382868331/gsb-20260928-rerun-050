from __future__ import annotations

import unittest

import numpy as np

from cart_control import Action, ActionBounds, CartPlant, MPCParameters, RecedingHorizonMPC
from cart_control.experiment import Scenario, run_scenario
from cart_control.mpc import Plan


class CartControlTests(unittest.TestCase):
    def test_cart_model_step(self) -> None:
        plant = CartPlant(dt=0.1)
        observation = plant.step(Action(1.0))
        self.assertEqual(observation.time, 0.1)
        self.assertEqual(observation.position, 0.0)
        self.assertAlmostEqual(observation.velocity, 0.1)

    def test_action_and_slew_bounds(self) -> None:
        bounds = ActionBounds(-2.0, 2.0, 0.5)
        self.assertEqual(bounds.apply_slew(3.0, 1.0), 1.5)
        self.assertEqual(bounds.apply_slew(-3.0, 1.0), 0.5)
        self.assertEqual(bounds.apply_slew(0.0, 1.0), 0.5)

    def test_mpc_respects_bounds_and_reaches_target(self) -> None:
        plant = CartPlant(dt=0.1, target_position=1.0)
        bounds = ActionBounds(-2.0, 2.0, 1.5)
        controller = RecedingHorizonMPC(0.1, bounds, MPCParameters(horizon=20))
        actions = []

        for _ in range(60):
            action, plan = controller.step(plant.observe())
            self.assertTrue(plan.success, msg=plan.message)
            plant.step(action)
            actions.append(action.acceleration)

        actions = np.asarray(actions)
        self.assertLessEqual(np.max(np.abs(actions)), 2.0 + 1e-9)
        self.assertLessEqual(
            np.max(np.abs(np.diff(np.r_[0.0, actions]))), 1.5 + 1e-9
        )
        self.assertAlmostEqual(plant.position, 1.0, delta=1e-3)
        self.assertAlmostEqual(plant.velocity, 0.0, delta=1e-2)

    def test_disturbance_creates_recorded_plan_execution_gap(self) -> None:
        plant = CartPlant(dt=0.1, target_position=1.0)
        bounds = ActionBounds(-2.0, 2.0, 1.5)
        controller = RecedingHorizonMPC(0.1, bounds)
        scenario = Scenario(
            name="disturbance",
            description="test disturbance",
            steps=3,
            initial_target=1.0,
            disturbance=lambda step, time, observation: 0.5,
        )

        records = run_scenario(scenario, controller, plant)

        self.assertTrue(all(record["plan_success"] for record in records))
        self.assertTrue(
            any(
                abs(record["velocity_prediction_error"]) > 1e-3
                for record in records
            )
        )

    def test_failure_uses_explicit_conservative_braking(self) -> None:
        plant = CartPlant(dt=0.1, initial_velocity=1.0)
        bounds = ActionBounds(-2.0, 2.0, 0.5)
        controller = RecedingHorizonMPC(
            0.1,
            bounds,
            MPCParameters(horizon=2, braking_gain=3.0),
        )

        original_plan = controller.plan

        def failed_plan(observation):
            plan = original_plan(observation)
            return Plan(
                success=False,
                message="injected failure",
                actions=plan.actions,
                predicted_positions=plan.predicted_positions,
                predicted_velocities=plan.predicted_velocities,
                target_positions=plan.target_positions,
            )

        controller.plan = failed_plan
        action, plan = controller.step(plant.observe())

        self.assertFalse(plan.success)
        self.assertEqual(action.acceleration, -0.5)


if __name__ == "__main__":
    unittest.main()
