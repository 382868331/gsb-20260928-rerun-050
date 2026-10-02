"""双执行器有限时域 MPC。

两台执行器共同作用于同一一维对象，能力、延迟与允许方向可以不同。
优化变量是双方各自新提交的动作序列；每台执行器尚未生效的排队命令
按各自的生效时刻进入预测前缀，不会把一台未到点的动作算成另一台
已完成的补偿。能力恢复后通过爬坡系数逐步放开变化限制，避免两台
同时补偿同一误差而超调。每个周期可输出只用单台与协同使用的对照
计划，便于比较。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .config import MPCConfig
from .capability import Capability
from .plants.base import Plant


@dataclass
class DualPlanResult:
    success: bool
    action_a: np.ndarray
    action_b: np.ndarray
    planned_states: np.ndarray
    planned_actions_a: np.ndarray
    planned_actions_b: np.ndarray
    target: float
    cost: float
    fallback_used: bool
    reason: str
    solver_status: dict = field(default_factory=dict)
    capability: dict = field(default_factory=dict)
    target_attainable: bool = True
    reachable_target_min: float = -np.inf
    reachable_target_max: float = np.inf
    comparison: dict = field(default_factory=dict)


class DualMPCController:
    """两台执行器的联合 MPC；预测模型与单机 MPC 相同的线性小车。"""

    def __init__(self, plant: Plant, config: MPCConfig | None = None,
                 recovery_ramp_steps: int = 5,
                 compare_single: bool = True):
        self.plant = plant
        self.cfg = config or MPCConfig()
        if plant.n_obs != 2 or plant.n_act != 1:
            raise ValueError("当前双执行器 MPC 针对一维小车：n_obs=2, n_act=1")
        self.n = self.cfg.horizon
        dt = plant.dt
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.B = np.array([[0.0], [dt / plant.cfg.mass]])  # type: ignore[attr-defined]
        self.capability = {
            "a": Capability.from_plant(plant),
            "b": Capability.from_plant(plant),
        }
        self.prev_action = {
            "a": plant.neutral_action(),
            "b": plant.neutral_action(),
        }
        self.pending_actions: dict[str, list[np.ndarray]] = {"a": [], "b": []}
        self.recovery_ramp_steps = max(1, int(recovery_ramp_steps))
        self.compare_single = bool(compare_single)
        self._ramp_age = {"a": None, "b": None}
        self._warm = {"a": np.zeros(self.n), "b": np.zeros(self.n)}

    def reset(self) -> None:
        for name in ("a", "b"):
            self.prev_action[name] = self.plant.neutral_action()
            self.pending_actions[name] = []
            self._warm[name] = np.zeros(self.n)
            self._ramp_age[name] = None

    def update_capability(self, name: str, capability: Capability) -> None:
        old = self.capability[name]
        recovered = (
            capability.u_max > old.u_max + 1e-12
            or capability.u_min < old.u_min - 1e-12
            or capability.du_up > old.du_up + 1e-12
            or capability.du_down > old.du_down + 1e-12
            or capability.delay_steps < old.delay_steps
        )
        self.capability[name] = capability
        # 能力恢复：从爬坡起点逐步放开；能力下降：立即使用更严约束。
        self._ramp_age[name] = 0 if recovered else None
        self._warm[name] = np.clip(
            self._warm[name], capability.u_min, capability.u_max
        )

    def _ramp_factor(self, name: str) -> float:
        age = self._ramp_age[name]
        if age is None:
            return 1.0
        return min(1.0, (age + 1) / self.recovery_ramp_steps)

    def _effective_rates(self, name: str) -> tuple[float, float]:
        cap = self.capability[name]
        factor = self._ramp_factor(name)
        return cap.du_up * factor, cap.du_down * factor

    def _delay(self, name: str) -> int:
        return min(len(self.pending_actions[name]), self.n)

    def _anchor(self, name: str) -> float:
        """新命令变化限制的基准：队列尾（若有）否则最后施加动作。"""
        if self._delay(name):
            return float(self.pending_actions[name][-1][0])
        return float(self.prev_action[name][0])

    def _first_bounds(self, name: str) -> tuple[float, float]:
        cap = self.capability[name]
        du_up, du_down = self._effective_rates(name)
        anchor = self._anchor(name)
        lo = max(cap.u_min, anchor - du_down)
        hi = min(cap.u_max, anchor + du_up)
        return lo, hi

    def _effective_actions(self, name: str,
                           commands: np.ndarray) -> np.ndarray:
        delay = self._delay(name)
        effective = np.zeros(self.n)
        if delay:
            pending = np.asarray(
                self.pending_actions[name], dtype=float
            ).reshape(-1)
            effective[:delay] = pending[:delay]
        if delay < self.n:
            effective[delay:] = commands[:self.n - delay]
        return effective

    def _rollout(self, x0: np.ndarray, total: np.ndarray) -> np.ndarray:
        states = np.zeros((self.n + 1, 2))
        states[0] = x0
        for k in range(self.n):
            states[k + 1] = self.A @ states[k] + self.B[:, 0] * total[k]
        return states

    def _cost_and_grad(self, z: np.ndarray, x0: np.ndarray, target: float):
        za, zb = z[:self.n], z[self.n:]
        eff_a = self._effective_actions("a", za)
        eff_b = self._effective_actions("b", zb)
        total = eff_a + eff_b
        states = self._rollout(x0, total)
        pos, vel = states[:, 0], states[:, 1]
        err = pos - target
        stage = (
            self.cfg.q_pos * err[:-1] ** 2
            + self.cfg.q_vel * vel[:-1] ** 2
            + self.cfg.r_u * (eff_a ** 2 + eff_b ** 2)
        )
        terminal = (self.cfg.q_term_pos * err[-1] ** 2
                    + self.cfg.q_term_vel * vel[-1] ** 2)
        cost = float(stage.sum() + terminal)

        lam = np.array([
            2.0 * self.cfg.q_term_pos * err[-1],
            2.0 * self.cfg.q_term_vel * vel[-1],
        ])
        grad = np.zeros(2 * self.n)
        delay_a, delay_b = self._delay("a"), self._delay("b")
        for k in range(self.n - 1, -1, -1):
            shared = float((self.B.T @ lam).item())
            if k >= delay_a:
                grad[k - delay_a] = shared + 2.0 * self.cfg.r_u * eff_a[k]
            if k >= delay_b:
                grad[self.n + k - delay_b] = (
                    shared + 2.0 * self.cfg.r_u * eff_b[k]
                )
            gx = np.array([
                2.0 * self.cfg.q_pos * err[k],
                2.0 * self.cfg.q_vel * vel[k],
            ])
            lam = gx + self.A.T @ lam
        return cost, grad

    def _neutral_profile(self, name: str) -> np.ndarray:
        """退出使用的执行器：从队列锚点按变化限制尽快回到中性动作。"""
        cap = self.capability[name]
        du_up, du_down = self._effective_rates(name)
        neutral = float(np.asarray(
            self.plant.neutral_action(), dtype=float).reshape(-1)[0])
        neutral = float(np.clip(neutral, cap.u_min, cap.u_max))
        profile = np.zeros(self.n)
        value = self._anchor(name)
        for k in range(self.n):
            if value < neutral:
                value = min(neutral, value + du_up)
            else:
                value = max(neutral, value - du_down)
            profile[k] = value
        return profile

    def _reachable_interval(self, x0: np.ndarray,
                            plan_a: bool, plan_b: bool,
                            ramp_aware: bool = True
                            ) -> tuple[float, float]:
        """时域末端位置可达区间；不参与规划的执行器播完队列后退回中性。

        双方前缀队列都不可撤回，按各自生效时刻计入；区间按每台
        执行器各自的变化限制独立推进后再求和。
        """
        dt = float(self.plant.dt)
        mass = float(self.plant.cfg.mass)  # type: ignore[attr-defined]
        x_lo = x_hi = float(x0[0])
        v_lo = v_hi = float(x0[1])
        lo_u = {"a": float(self.prev_action["a"][0]),
                "b": float(self.prev_action["b"][0])}
        hi_u = dict(lo_u)
        prefixes = {
            name: [float(np.asarray(item).reshape(-1)[0])
                   for item in self.pending_actions[name][:self.n]]
            for name in ("a", "b")
        }
        planning = {"a": plan_a, "b": plan_b}
        retire = {
            name: self._neutral_profile(name) for name in ("a", "b")
        }
        for k in range(self.n):
            step_lo = step_hi = 0.0
            for name in ("a", "b"):
                cap = self.capability[name]
                prefix = prefixes[name]
                if k < len(prefix):
                    value = float(np.clip(
                        prefix[k],
                        max(cap.u_min, lo_u[name] - cap.du_down),
                        min(cap.u_max, hi_u[name] + cap.du_up),
                    ))
                    lo_u[name] = hi_u[name] = value
                elif planning[name]:
                    if ramp_aware:
                        du_up, du_down = self._effective_rates(name)
                    else:
                        du_up, du_down = cap.du_up, cap.du_down
                    lo_u[name] = max(cap.u_min, lo_u[name] - du_down)
                    hi_u[name] = min(cap.u_max, hi_u[name] + du_up)
                else:
                    # 不参与规划：按变化限制退回中性动作
                    lo_u[name] = hi_u[name] = float(
                        retire[name][k - len(prefix)]
                    ) if k - len(prefix) < self.n else lo_u[name]
                step_lo += lo_u[name]
                step_hi += hi_u[name]
            a_lo, a_hi = step_lo / mass, step_hi / mass
            x_lo += v_lo * dt + 0.5 * a_lo * dt * dt
            x_hi += v_hi * dt + 0.5 * a_hi * dt * dt
            v_lo += a_lo * dt
            v_hi += a_hi * dt
        return x_lo, x_hi

    def _annotate_reachability(self, result: DualPlanResult,
                               x0: np.ndarray,
                               target: float) -> DualPlanResult:
        lo, hi = self._reachable_interval(x0, True, True)
        lo_a, hi_a = self._reachable_interval(x0, True, False)
        lo_b, hi_b = self._reachable_interval(x0, False, True)
        result.reachable_target_min = lo
        result.reachable_target_max = hi
        result.target_attainable = lo - 1e-8 <= target <= hi + 1e-8
        result.capability = {
            "a": {
                **self.capability["a"].to_dict(),
                "recovery_ramp_factor": self._ramp_factor("a"),
            },
            "b": {
                **self.capability["b"].to_dict(),
                "recovery_ramp_factor": self._ramp_factor("b"),
            },
            "target_attainable": result.target_attainable,
            "reachable_target_min": float(lo),
            "reachable_target_max": float(hi),
            "single_a_attainable": bool(lo_a - 1e-8 <= target <= hi_a + 1e-8),
            "single_a_reachable_min": float(lo_a),
            "single_a_reachable_max": float(hi_a),
            "single_b_attainable": bool(lo_b - 1e-8 <= target <= hi_b + 1e-8),
            "single_b_reachable_min": float(lo_b),
            "single_b_reachable_max": float(hi_b),
        }
        return result

    def _solve(self, x0: np.ndarray, target: float,
               hold: frozenset[str] = frozenset()):
        """求解联合优化；hold 中的执行器新命令被固定为保持值（对照用）。"""
        lo_a, hi_a = self._first_bounds("a")
        lo_b, hi_b = self._first_bounds("b")
        if lo_a > hi_a + 1e-12 or lo_b > hi_b + 1e-12:
            return None, {"success": False, "message": "infeasible_first_step"}
        cap_a, cap_b = self.capability["a"], self.capability["b"]
        lb = np.concatenate((
            [lo_a], np.full(self.n - 1, cap_a.u_min),
            [lo_b], np.full(self.n - 1, cap_b.u_min),
        ))
        ub = np.concatenate((
            [hi_a], np.full(self.n - 1, cap_a.u_max),
            [hi_b], np.full(self.n - 1, cap_b.u_max),
        ))
        for name in hold:
            profile = self._neutral_profile(name)
            offset = 0 if name == "a" else self.n
            lb[offset:offset + self.n] = profile
            ub[offset:offset + self.n] = profile
        box = Bounds(lb, ub)

        rows, r_lo, r_hi = [], [], []
        for name in ("a", "b"):
            offset = 0 if name == "a" else self.n
            du_up, du_down = self._effective_rates(name)
            for k in range(1, self.n):
                row = np.zeros(2 * self.n)
                row[offset + k] = 1.0
                row[offset + k - 1] = -1.0
                rows.append(row)
                r_lo.append(-du_down)
                r_hi.append(du_up)
        constraints = []
        if rows:
            constraints.append(LinearConstraint(
                np.array(rows), np.array(r_lo), np.array(r_hi)
            ))

        z0 = np.clip(
            np.concatenate((self._warm["a"], self._warm["b"])), lb, ub
        )
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
            return res, status
        except Exception as exc:
            return None, {"success": False,
                          "message": f"solver_exception:{exc}"}

    def _allocate(self, total: float) -> tuple[float, float]:
        """把总保守动作分摊到两台，均不超出各自首拍允许区间。"""
        lo_a, hi_a = self._first_bounds("a")
        lo_b, hi_b = self._first_bounds("b")
        total = float(np.clip(total, lo_a + lo_b, hi_a + hi_b))
        ua = float(np.clip(total / 2.0, lo_a, hi_a))
        ub = float(np.clip(total - ua, lo_b, hi_b))
        ua = float(np.clip(total - ub, lo_a, hi_a))
        return ua, ub

    def conservative_action(self, obs: np.ndarray, target: float,
                            reason: str) -> DualPlanResult:
        x0 = np.asarray(obs, dtype=float).reshape(2)
        ua, ub = self._allocate(-self.cfg.brake_gain * float(x0[1]))
        eff_a = self._effective_actions("a", np.full(self.n, ua))
        eff_b = self._effective_actions("b", np.full(self.n, ub))
        states = self._rollout(x0, eff_a + eff_b)
        result = DualPlanResult(
            success=False, action_a=np.array([ua]), action_b=np.array([ub]),
            planned_states=states,
            planned_actions_a=eff_a.reshape(-1, 1),
            planned_actions_b=eff_b.reshape(-1, 1),
            target=target, cost=np.nan, fallback_used=True, reason=reason,
            solver_status={"conservative": True},
        )
        return self._annotate_reachability(result, x0, target)

    def _fallback(self, obs: np.ndarray, reason: str, target: float,
                  cost: float, status: dict) -> DualPlanResult:
        result = self.conservative_action(obs, target, reason)
        result.cost = cost
        result.solver_status = status
        return result

    def _comparison(self, x0: np.ndarray, target: float) -> dict:
        """只用单台的对照计划：另一台新命令被固定为保持值。"""
        comparison = {}
        for name, hold in (("a", frozenset({"b"})),
                           ("b", frozenset({"a"}))):
            res, status = self._solve(x0, target, hold=hold)
            if res is None or not getattr(res, "success", False):
                comparison[f"single_{name}"] = {
                    "success": False,
                    "reason": status.get("message", "solver_failed"),
                }
                continue
            lo, hi = self._reachable_interval(
                x0, name == "a", name == "b"
            )
            comparison[f"single_{name}"] = {
                "success": True,
                "cost": float(res.fun),
                "target_attainable": bool(lo - 1e-8 <= target <= hi + 1e-8),
                "reachable_target_min": float(lo),
                "reachable_target_max": float(hi),
            }
        return comparison

    def plan(self, obs: np.ndarray, target: float,
             observation_available: bool = True,
             observation_age: int = 0) -> DualPlanResult:
        x0 = np.asarray(obs, dtype=float).reshape(2)
        try:
            if not observation_available or observation_age > 0:
                reason = (
                    "observation_missing"
                    if not observation_available
                    else f"observation_recovery_age_{observation_age}"
                )
                return self.conservative_action(x0, target, reason)

            res, status = self._solve(x0, target)
            if res is None:
                return self._fallback(x0, status.get("message", "solver_failed"),
                                      target, np.inf, status)
            if not res.success:
                return self._fallback(
                    x0, f"solver_failed:{res.message}", target,
                    float(res.fun), status,
                )
            z = np.asarray(res.x, dtype=float)
            za, zb = z[:self.n], z[self.n:]
            eff_a = self._effective_actions("a", za)
            eff_b = self._effective_actions("b", zb)
            states = self._rollout(x0, eff_a + eff_b)
            self._warm["a"] = np.concatenate((za[1:], za[-1:]))
            self._warm["b"] = np.concatenate((zb[1:], zb[-1:]))
            result = DualPlanResult(
                success=True,
                action_a=np.array([float(za[0])]),
                action_b=np.array([float(zb[0])]),
                planned_states=states,
                planned_actions_a=eff_a.reshape(-1, 1),
                planned_actions_b=eff_b.reshape(-1, 1),
                target=target, cost=float(res.fun),
                fallback_used=False, reason="ok", solver_status=status,
            )
            result = self._annotate_reachability(result, x0, target)
            if self.compare_single:
                result.comparison = self._comparison(x0, target)
            return result
        finally:
            for name in ("a", "b"):
                if self._ramp_age[name] is not None:
                    self._ramp_age[name] += 1
