from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .interfaces import Action, ActionBounds, Observation
from .model import CartParameters


@dataclass(frozen=True)
class MPCParameters:
    horizon: int = 25
    position_weight: float = 8.0
    velocity_weight: float = 0.8
    acceleration_weight: float = 0.02
    slew_weight: float = 0.15
    terminal_weight: float = 4.0
    braking_gain: float = 2.0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        for name in (
            "position_weight",
            "velocity_weight",
            "acceleration_weight",
            "slew_weight",
            "terminal_weight",
            "braking_gain",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class Plan:
    success: bool
    message: str
    actions: np.ndarray
    predicted_positions: np.ndarray
    predicted_velocities: np.ndarray
    target_positions: np.ndarray

    @property
    def first_action(self) -> Action:
        return Action(float(self.actions[0]))


class RecedingHorizonMPC:
    def __init__(
        self,
        dt: float,
        action_bounds: ActionBounds,
        params: MPCParameters | None = None,
        plant_params: CartParameters | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self.action_bounds = action_bounds
        self.params = params or MPCParameters()
        self.plant_params = plant_params or CartParameters()
        self._last_action_value = 0.0
        self._previous_solution: np.ndarray | None = None

    def reset(self, last_action: float = 0.0) -> None:
        self._last_action_value = self.action_bounds.clip_action(last_action)
        self._previous_solution = None

    def fallback_action(self, observation: Observation) -> Action:
        braking = -self.params.braking_gain * observation.velocity
        value = self.action_bounds.apply_slew(braking, self._last_action_value)
        return Action(float(value))

    def plan(
        self,
        observation: Observation,
        target_trajectory: np.ndarray | None = None,
    ) -> Plan:
        n = self.params.horizon
        if target_trajectory is None:
            target_trajectory = np.full(n, observation.target_position)
        target = np.asarray(target_trajectory, dtype=float)
        if target.shape != (n,):
            raise ValueError(f"target trajectory must have shape ({n},)")

        slew_matrix = np.eye(n)
        slew_matrix[1:, :] -= np.eye(n, k=-1)[1:, :]
        slew_lower = np.full(n, -self.action_bounds.max_slew)
        slew_upper = np.full(n, self.action_bounds.max_slew)
        slew_lower[0] = self._last_action_value - self.action_bounds.max_slew
        slew_upper[0] = self._last_action_value + self.action_bounds.max_slew

        constraints = LinearConstraint(
            slew_matrix,
            slew_lower,
            slew_upper,
        )
        bounds = Bounds(
            np.full(n, self.action_bounds.min_acceleration),
            np.full(n, self.action_bounds.max_acceleration),
        )

        def candidate(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            positions = np.empty(n + 1)
            velocities = np.empty(n + 1)
            positions[0] = observation.position
            velocities[0] = observation.velocity
            previous = self._last_action_value
            for k, acceleration in enumerate(actions):
                positions[k + 1] = positions[k] + self.dt * velocities[k]
                velocities[k + 1] = (
                    velocities[k]
                    + self.dt
                    * (
                        acceleration
                        - self.plant_params.damping * velocities[k]
                    )
                    / self.plant_params.mass
                )
                previous = acceleration
            return positions, velocities

        def objective(actions: np.ndarray) -> float:
            positions, velocities = candidate(actions)
            position_error = positions[1:] - target
            command_change = slew_matrix @ actions
            command_change[0] = actions[0] - self._last_action_value
            stage_cost = (
                self.params.position_weight * np.sum(position_error[:-1] ** 2)
                + self.params.velocity_weight * np.sum(velocities[1:-1] ** 2)
                + self.params.acceleration_weight * np.sum(actions**2)
                + self.params.slew_weight * np.sum(command_change**2)
            )
            terminal_cost = self.params.terminal_weight * (
                position_error[-1] ** 2 + velocities[-1] ** 2
            )
            return float(stage_cost + terminal_cost)

        if self._previous_solution is not None and len(self._previous_solution) == n:
            initial = np.r_[self._previous_solution[1:], self._previous_solution[-1]]
        else:
            initial = np.zeros(n)
        initial = np.clip(
            initial,
            self.action_bounds.min_acceleration,
            self.action_bounds.max_acceleration,
        )

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 120, "ftol": 1e-9, "disp": False},
        )

        actions = np.asarray(result.x, dtype=float)
        finite = np.all(np.isfinite(actions))
        within_action = np.all(
            (actions >= self.action_bounds.min_acceleration - 1e-7)
            & (actions <= self.action_bounds.max_acceleration + 1e-7)
        )
        command_sequence = np.r_[self._last_action_value, actions]
        changes = np.diff(command_sequence)
        within_slew = np.all(np.abs(changes) <= self.action_bounds.max_slew + 1e-7)
        if not (result.success and finite and within_action and within_slew):
            return Plan(
                success=False,
                message=result.message if hasattr(result, "message") else "invalid plan",
                actions=actions,
                predicted_positions=np.empty(0),
                predicted_velocities=np.empty(0),
                target_positions=target,
            )

        positions, velocities = candidate(actions)
        return Plan(
            success=True,
            message=str(result.message),
            actions=actions,
            predicted_positions=positions,
            predicted_velocities=velocities,
            target_positions=target,
        )

    def step(self, observation: Observation) -> tuple[Action, Plan]:
        plan = self.plan(observation)
        if not plan.success:
            action = self.fallback_action(observation)
            self._last_action_value = action.acceleration
            self._previous_solution = None
            return action, plan

        action = plan.first_action
        self._last_action_value = action.acceleration
        self._previous_solution = plan.actions.copy()
        return action, plan
