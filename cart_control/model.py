from __future__ import annotations

from dataclasses import dataclass

from .interfaces import Action, Observation


@dataclass(frozen=True)
class CartParameters:
    mass: float = 1.0
    damping: float = 0.0

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")


class CartPlant:
    """A transparent point-mass cart with damped velocity dynamics.

    The discrete model is::

        x[k+1] = x[k] + dt * v[k]
        v[k+1] = v[k] + dt * (u[k] + disturbance[k] - damping * v[k]) / mass

    Disturbance is intentionally not included in :class:`Observation`. It acts
    as an unmeasured process disturbance so the controller must replan from the
    returned state.
    """

    def __init__(
        self,
        dt: float,
        params: CartParameters | None = None,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
        target_position: float = 1.0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self.params = params or CartParameters()
        self.time = 0.0
        self.position = float(initial_position)
        self.velocity = float(initial_velocity)
        self.target_position = float(target_position)

    def observe(self) -> Observation:
        return Observation(
            time=self.time,
            position=self.position,
            velocity=self.velocity,
            target_position=self.target_position,
        )

    def predict(
        self,
        position: float,
        velocity: float,
        acceleration: float,
        disturbance: float = 0.0,
    ) -> tuple[float, float]:
        next_position = position + self.dt * velocity
        next_velocity = (
            velocity
            + self.dt
            * (
                acceleration
                + disturbance
                - self.params.damping * velocity
            )
            / self.params.mass
        )
        return next_position, next_velocity

    def step(self, action: Action, disturbance: float = 0.0) -> Observation:
        self.position, self.velocity = self.predict(
            self.position,
            self.velocity,
            action.acceleration,
            disturbance,
        )
        self.time += self.dt
        return self.observe()
