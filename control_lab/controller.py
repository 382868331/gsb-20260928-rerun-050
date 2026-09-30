"""有限时域 MPC 控制器。

优化变量是控制器新提交的动作序列；若执行器存在纯延迟，状态预测前几拍
使用执行器队列中已提交但尚未生效的旧命令。动作幅度和方向不对称变化限制
均作为硬约束，规划失败或观测缺失时输出保守刹车命令。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .config import MPCConfig
from .plants.base import Plant


@dataclass
class PlanResult:
    success: bool
    action: np.ndarray
    planned_states: np.ndarray
    planned_actions: np.ndarray
    target: float
    cost: float
    fallback_used: bool
    reason: str
    solver_status: dict = field(default_factory=dict)


class MPCController:
    def __init__(self, plant: Plant, config: MPCConfig | None = None):
        self.plant = plant
        self.cfg = config or MPCConfig()
        if plant.n_obs != 2 or plant.n_act != 1:
            raise ValueError("当前 MPC 针对一维小车：n_obs=2, n_act=1")
        self.n = self.cfg.horizon
        dt = plant.dt
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.B = np.array([[0.0], [dt / plant.cfg.mass]])  # type: ignore[attr-defined]
        self.u_lo, self.u_hi = plant.action_bounds()
        self.du_lo, self.du_hi = plant.action_rate_bounds()
        self.prev_action = plant.neutral_action()
        self.pending_actions: list[np.ndarray] = []
        self.observation_recovery_steps = 1
        self._warm = np.zeros(self.n)

    def reset(self) -> None:
        self.prev_action = self.plant.neutral_action()
        self.pending_actions = []
        self._warm = np.zeros(self.n)

    def _delay_steps(self) -> int:
        return min(len(self.pending_actions), self.n)

    def _effective_actions(self, commands: np.ndarray) -> np.ndarray:
        delay = self._delay_steps()
        effective = np.zeros(self.n)
        if delay:
            pending = np.asarray(self.pending_actions, dtype=float).reshape(-1)
            effective[:delay] = pending[:delay]
        if delay < self.n:
            effective[delay:] = commands[:self.n - delay]
        return effective

    def _rollout(self, x0: np.ndarray, z: np.ndarray) -> np.ndarray:
        effective = self._effective_actions(z)
        states = np.zeros((self.n + 1, 2))
        states[0] = x0
        for k in range(self.n):
            states[k + 1] = self.A @ states[k] + self.B[:, 0] * effective[k]
        return states

    def _cost_and_grad(self, z: np.ndarray, x0: np.ndarray, target: float):
        states = self._rollout(x0, z)
        effective = self._effective_actions(z)
        pos = states[:, 0]
        vel = states[:, 1]
        err = pos - target
        stage = (
            self.cfg.q_pos * err[:-1] ** 2
            + self.cfg.q_vel * vel[:-1] ** 2
            + self.cfg.r_u * effective ** 2
        )
        terminal = self.cfg.q_term_pos * err[-1] ** 2 + self.cfg.q_term_vel * vel[-1] ** 2
        cost = float(stage.sum() + terminal)

        delay = self._delay_steps()
        lam = np.array([
            2.0 * self.cfg.q_term_pos * err[-1],
            2.0 * self.cfg.q_term_vel * vel[-1],
        ])
        grad = np.zeros(self.n)
        for k in range(self.n - 1, -1, -1):
            action_grad = float((self.B.T @ lam).item())
            if k >= delay:
                j = k - delay
                grad[j] = action_grad + 2.0 * self.cfg.r_u * effective[k]
            gx = np.array([
                2.0 * self.cfg.q_pos * err[k],
                2.0 * self.cfg.q_vel * vel[k],
            ])
            lam = gx + self.A.T @ lam
        return cost, grad

    def _bounds_first_new_command(self) -> tuple[float, float]:
        delay = self._delay_steps()
        prev = float(self.pending_actions[-1][0]) if delay else float(self.prev_action[0])
        lo = max(float(self.u_lo[0]), prev + float(self.du_lo[0]))
        hi = min(float(self.u_hi[0]), prev + float(self.du_hi[0]))
        return lo, hi

    def _fallback(self, obs: np.ndarray, reason: str, target: float,
                  cost: float, status: dict) -> PlanResult:
        vel = float(obs[1])
        lo0, hi0 = self._bounds_first_new_command()
        u_brake = float(np.clip(-self.cfg.brake_gain * vel, lo0, hi0))
        commands = np.full(self.n, u_brake)
        planned_actions = self._effective_actions(commands).reshape(-1, 1)
        planned_states = self._rollout(obs, commands)
        return PlanResult(
            success=False, action=np.array([u_brake]),
            planned_states=planned_states, planned_actions=planned_actions,
            target=target, cost=cost, fallback_used=True, reason=reason,
            solver_status=status,
        )

    def conservative_action(self, obs: np.ndarray, target: float, reason: str) -> PlanResult:
        x0 = np.asarray(obs, dtype=float).reshape(2)
        lo0, hi0 = self._bounds_first_new_command()
        u_brake = float(np.clip(-self.cfg.brake_gain * float(x0[1]), lo0, hi0))
        commands = np.full(self.n, u_brake)
        actions = self._effective_actions(commands).reshape(-1, 1)
        states = self._rollout(x0, commands)
        return PlanResult(
            success=False, action=np.array([u_brake]),
            planned_states=states, planned_actions=actions, target=target,
            cost=np.nan, fallback_used=True, reason=reason,
            solver_status={"conservative": True},
        )

    def plan(self, obs: np.ndarray, target: float,
             observation_available: bool = True,
             observation_age: int = 0) -> PlanResult:
        x0 = np.asarray(obs, dtype=float).reshape(2)
        commands = np.tile(self.prev_action, self.n)

        if not observation_available or observation_age > 0:
            reason = (
                "observation_missing"
                if not observation_available
                else f"observation_recovery_age_{observation_age}"
            )
            return self.conservative_action(x0, target, reason)

        lo0, hi0 = self._bounds_first_new_command()
        if lo0 > hi0 + 1e-12:
            return self._fallback(x0, "infeasible_first_step", target,
                                  np.inf, {})

        box = Bounds(
            np.concatenate(([lo0], np.full(self.n - 1, float(self.u_lo[0])))),
            np.concatenate(([hi0], np.full(self.n - 1, float(self.u_hi[0])))),
        )
        constraints = []
        if self.n > 1:
            rows = []
            for k in range(1, self.n):
                row = np.zeros(self.n)
                row[k] = 1.0
                row[k - 1] = -1.0
                rows.append(row)
            constraints.append(LinearConstraint(
                np.array(rows),
                np.full(self.n - 1, float(self.du_lo[0])),
                np.full(self.n - 1, float(self.du_hi[0])),
            ))

        z0 = np.clip(self._warm, box.lb, box.ub)
        status: dict = {}
        try:
            res = minimize(
                lambda z: self._cost_and_grad(z, x0, target),
                z0, jac=True, method="SLSQP", bounds=box,
                constraints=constraints,
                options={"maxiter": self.cfg.max_iter, "ftol": self.cfg.tol,
                         "disp": False},
            )
            status = {
                "success": bool(res.success),
                "status": int(getattr(res, "status", -1)),
                "message": str(res.message),
                "nit": int(getattr(res, "nit", -1)),
            }
        except Exception as exc:
            return self._fallback(x0, f"solver_exception:{exc}", target,
                                  np.inf, status)

        z = np.asarray(res.x, dtype=float)
        states = self._rollout(x0, z)
        actions = self._effective_actions(z).reshape(-1, 1)
        if not res.success:
            return self._fallback(x0, f"solver_failed:{res.message}", target,
                                  float(res.fun), status)

        action = np.array([float(z[0])])
        self._warm = np.concatenate((z[1:], z[-1:]))
        return PlanResult(
            success=True, action=action, planned_states=states,
            planned_actions=actions, target=target, cost=float(res.fun),
            fallback_used=False, reason="ok", solver_status=status,
        )
