"""有限时域 MPC 控制器（线性小车模型上的二次规划，SLSQP 求解）。

约束：
  * 动作幅度：u_low <= u_k <= u_high；
  * 动作变化速度：-du_max <= u_k - u_{k-1} <= du_max，k=0 时以上一周期
    实际执行动作为基准。
规划失败时不延用旧预测，输出明确的保守刹车动作。
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
    action: np.ndarray                       # 本周期实际采用的动作
    planned_states: np.ndarray               # (N+1, n_obs) 本次规划预测状态
    planned_actions: np.ndarray              # (N, n_act) 本次规划预测动作
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
        self._warm = np.zeros(self.n)

    def reset(self) -> None:
        self.prev_action = self.plant.neutral_action()
        self._warm = np.zeros(self.n)

    def _rollout(self, x0: np.ndarray, z: np.ndarray, target: float):
        states = np.zeros((self.n + 1, 2))
        states[0] = x0
        for k in range(self.n):
            states[k + 1] = self.A @ states[k] + self.B[:, 0] * z[k]
        return states

    def _cost_and_grad(self, z: np.ndarray, x0: np.ndarray, target: float):
        states = self._rollout(x0, z, target)
        pos = states[:, 0]
        vel = states[:, 1]
        err = pos - target
        stage = self.cfg.q_pos * err[:-1] ** 2 + self.cfg.q_vel * vel[:-1] ** 2 + self.cfg.r_u * z ** 2
        terminal = self.cfg.q_term_pos * err[-1] ** 2 + self.cfg.q_term_vel * vel[-1] ** 2
        cost = float(stage.sum() + terminal)

        # 伴随法求精确梯度
        lam = np.array([2.0 * self.cfg.q_term_pos * err[-1],
                        2.0 * self.cfg.q_term_vel * vel[-1]])
        grad = np.zeros(self.n)
        for k in range(self.n - 1, -1, -1):
            grad[k] = (self.B.T @ lam).item() + 2.0 * self.cfg.r_u * z[k]
            gx = np.array([2.0 * self.cfg.q_pos * err[k],
                           2.0 * self.cfg.q_vel * vel[k]])
            lam = gx + self.A.T @ lam
        return cost, grad

    def _bounds_first_step(self) -> tuple[float, float]:
        """首动作同时满足幅度与相对上一步的变化速度限制。"""
        prev = float(self.prev_action[0])
        lo = max(float(self.u_lo[0]), prev + float(self.du_lo[0]))
        hi = min(float(self.u_hi[0]), prev + float(self.du_hi[0]))
        return lo, hi

    def _fallback(self, obs: np.ndarray, reason: str, planned_states: np.ndarray,
                  planned_actions: np.ndarray, target: float, cost: float,
                  status: dict) -> PlanResult:
        """明确的保守动作：限速刹车（反速度反馈），并同时满足两类硬约束。"""
        vel = float(obs[1])
        lo0, hi0 = self._bounds_first_step()
        u_brake = float(np.clip(-self.cfg.brake_gain * vel, lo0, hi0))
        action = np.array([u_brake])
        return PlanResult(
            success=False, action=action, planned_states=planned_states,
            planned_actions=planned_actions, target=target, cost=cost,
            fallback_used=True, reason=reason, solver_status=status,
        )

    def plan(self, obs: np.ndarray, target: float) -> PlanResult:
        x0 = np.asarray(obs, dtype=float).reshape(2)
        lo0, hi0 = self._bounds_first_step()
        empty_states = np.tile(x0, (self.n + 1, 1))
        empty_actions = np.tile(self.prev_action, (self.n, 1))
        if lo0 > hi0 + 1e-12:
            # 理论上不应发生（上一步动作始终可行）；直接给中性保守动作
            return self._fallback(x0, "infeasible_first_step", empty_states,
                                  empty_actions, target, np.inf, {})

        box = Bounds(
            np.concatenate(([lo0], np.full(self.n - 1, float(self.u_lo[0])))),
            np.concatenate(([hi0], np.full(self.n - 1, float(self.u_hi[0])))),
        )
        if self.n > 1:
            rows = []
            for k in range(1, self.n):
                row = np.zeros(self.n)
                row[k] = 1.0
                row[k - 1] = -1.0
                rows.append(row)
            slew = LinearConstraint(
                np.array(rows),
                np.full(self.n - 1, float(self.du_lo[0])),
                np.full(self.n - 1, float(self.du_hi[0])),
            )
            constraints = [slew]
        else:
            constraints = []

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
            status = {"success": bool(res.success), "status": int(getattr(res, "status", -1)),
                      "message": str(res.message), "nit": int(getattr(res, "nit", -1))}
        except Exception as exc:  # 求解器异常 -> 保守动作
            return self._fallback(x0, f"solver_exception:{exc}", empty_states,
                                  empty_actions, target, np.inf, status)

        z = np.asarray(res.x, dtype=float)
        states = self._rollout(x0, z, target)
        actions = z.reshape(-1, 1)
        if not res.success:
            # 失败：不延用本次（或上次）预测，改用保守刹车
            return self._fallback(x0, f"solver_failed:{res.message}", states, actions,
                                  target, float(res.fun), status)

        action = np.array([float(z[0])])  # 只执行当前一步
        # 热启动：整体前移，末尾补最后一个值
        self._warm = np.concatenate((z[1:], z[-1:]))
        self.prev_action = action.copy()
        return PlanResult(
            success=True, action=action, planned_states=states,
            planned_actions=actions, target=target, cost=float(res.fun),
            fallback_used=False, reason="ok", solver_status=status,
        )
