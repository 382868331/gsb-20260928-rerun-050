"""有限时域 MPC 控制器（线性小车模型上的二次规划，SLSQP 求解）。

支持执行器延迟：规划动作需要经过 ``delay_steps`` 个周期才真正施加到设备。
在途命令（已发出但尚未生效）通过 ``committed`` 前缀传入，控制器把它们固定
为预测轨迹的前 D 步，只优化其后的自由动作，因此**不会把尚未生效的控制量
当成即时效果**。

支持方向不对称的动作变化限制：
  * 增力方向：0 <= u[k]-u[k-1] <= du_inc_max；
  * 减力方向：-du_dec_max <= u[k]-u[k-1] < 0；
  * 同时满足动作幅度 |u| <= u_max。

规划失败或观测长时间缺失时，不延用旧预测，输出明确的保守刹车动作；该动作
本身也必须满足幅度与变化速度硬约束。
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
    action: np.ndarray                       # 本周期下发的命令（可能要延迟才生效）
    applied_action: np.ndarray                # 本周期真正施加到设备的动作
    committed_actions: np.ndarray             # (D,) 本周期在途（尚未生效）命令
    planned_states: np.ndarray               # (H+1, n_obs) 含在途前缀的预测状态
    planned_actions: np.ndarray              # (H,) 含在途前缀 + 自由动作
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
        self.D = max(0, int(self.cfg.actuator_delay_steps))
        dt = plant.dt
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.B = np.array([[0.0], [dt / plant.cfg.mass]])  # type: ignore[attr-defined]
        self.u_lo, self.u_hi = plant.action_bounds()
        self.du_lo, self.du_hi = plant.action_rate_bounds()
        self.neutral = plant.neutral_action()
        self.reset()

    def reset(self) -> None:
        # 上电瞬间设备施加中性动作；在途命令队列也用中性动作填充，
        # 表示“延迟期内设备仍维持中性”，而不是提前施加新命令。
        self._applied = self.neutral.copy()
        self._last_issued = self.neutral.copy()
        self._warm = np.zeros(self.n)

    # -- 在途命令 ----------------------------------------------------------

    def initial_committed(self) -> np.ndarray:
        """复位后（尚未发任何命令时）的在途序列：D 个中性动作。"""
        return np.repeat(self.neutral.reshape(1), self.D, axis=0).reshape(-1)

    def advance_committed(self, committed: np.ndarray, issued: np.ndarray) -> np.ndarray:
        """设备前进一步后更新在途队列：最旧命令生效，队尾补本次新命令。"""
        committed = np.asarray(committed, dtype=float).reshape(-1)
        if self.D == 0:
            return np.zeros(0)
        if len(committed) != self.D:
            raise ValueError(f"committed 长度应为 {self.D}，实际 {len(committed)}")
        return np.concatenate((committed[1:], np.asarray(issued, dtype=float).reshape(1)))

    def applied_this_step(self, committed: np.ndarray) -> np.ndarray:
        """本周期真正施加的动作：延迟为 0 时即上一条命令，否则为队列最旧项。"""
        if self.D == 0:
            return self._last_issued.copy()
        return np.asarray(committed, dtype=float).reshape(-1)[0:1].copy()

    # -- 约束辅助 ----------------------------------------------------------

    def _bounds_after(self, prev: float) -> tuple[float, float]:
        """紧跟 prev 之后的第一个动作的可行区间（幅度 + 方向不对称变化限制）。"""
        lo = max(float(self.u_lo[0]), prev + float(self.du_lo[0]))
        hi = min(float(self.u_hi[0]), prev + float(self.du_hi[0]))
        return lo, hi

    def _slew_pair(self, predecessor: float, value: float) -> float:
        return float(value) - float(predecessor)

    def _fallback(self, obs: np.ndarray, reason: str, committed: np.ndarray,
                  full_actions: np.ndarray, target: float, cost: float,
                  status: dict, applied: np.ndarray) -> PlanResult:
        """明确的保守动作：限速刹车（反速度反馈），同时满足两类硬约束。"""
        vel = float(obs[1])
        predecessor = float(full_actions[self.D - 1]) if self.D > 0 else float(self._last_issued[0])
        lo0, hi0 = self._bounds_after(predecessor)
        u_brake = float(np.clip(-self.cfg.brake_gain * vel, lo0, hi0))
        action = np.array([u_brake])
        return PlanResult(
            success=False, action=action, applied_action=applied.copy(),
            committed_actions=committed.copy(), planned_states=self._rollout(obs, full_actions),
            planned_actions=full_actions.reshape(-1, 1), target=target, cost=cost,
            fallback_used=True, reason=reason, solver_status=status,
        )

    # -- 预测与优化 --------------------------------------------------------

    def _rollout(self, x0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = np.zeros((len(actions) + 1, 2))
        states[0] = x0
        for k, u in enumerate(actions):
            states[k + 1] = self.A @ states[k] + self.B[:, 0] * float(u)
        return states

    def _cost_and_grad(self, z: np.ndarray, x0: np.ndarray, actions: np.ndarray,
                       n_free: int, target: float):
        """成本只对自由段（含其首个动作对应状态）与终端计权；在途前缀固定。"""
        full = actions.copy()
        full[self.D:] = z
        states = self._rollout(x0, full)
        pos, vel = states[:, 0], states[:, 1]
        err = pos - target
        stage = (self.cfg.q_pos * err ** 2 + self.cfg.q_vel * vel ** 2)
        stage = stage[self.D:]  # 在途前缀期间状态已定，不计优化成本
        r = self.cfg.r_u * z ** 2
        terminal = self.cfg.q_term_pos * err[-1] ** 2 + self.cfg.q_term_vel * vel[-1] ** 2
        cost = float(stage[:-1].sum() + terminal + r.sum())

        lam = np.array([2.0 * self.cfg.q_term_pos * err[-1],
                        2.0 * self.cfg.q_term_vel * vel[-1]])
        grad = np.zeros(n_free)
        for k in range(self.n - 1, self.D - 1, -1):
            j = k - self.D
            grad[j] = (self.B.T @ lam).item() + 2.0 * self.cfg.r_u * z[j]
            gx = np.array([2.0 * self.cfg.q_pos * err[k],
                           2.0 * self.cfg.q_vel * vel[k]])
            lam = gx + self.A.T @ lam
        return cost, grad

    def plan(self, obs: np.ndarray, target: float, committed: np.ndarray | None = None,
             force_fallback: bool = False) -> PlanResult:
        """基于当前（可能是开环估计的）状态与在途命令重新规划。

        参数
        ----
        committed: 长度 D 的在途命令序列，committed[0] 是本周期设备即将施加的
            动作，committed[D-1] 是上一周期刚下发、本周期仍未生效的命令。
            D=0 时忽略。
        force_fallback: True 时跳过求解器，直接给保守刹车（用于观测长时间缺失
            等“无法信任预测”的情形）。
        """
        x0 = np.asarray(obs, dtype=float).reshape(2)
        if self.D == 0:
            committed = np.zeros(0)
            applied = self._last_issued.copy()
        else:
            committed = np.asarray(committed, dtype=float).reshape(-1)
            if len(committed) != self.D:
                raise ValueError(f"committed 长度应为 {self.D}，实际 {len(committed)}")
            applied = committed[0:1].copy()

        full = np.zeros(self.n)
        if self.D > 0:
            full[:self.D] = committed
        predecessor = float(full[self.D - 1]) if self.D > 0 else float(self._last_issued[0])
        n_free = self.n - self.D

        lo0, hi0 = self._bounds_after(predecessor)
        infeasible = n_free <= 0 or lo0 > hi0 + 1e-12
        if infeasible:
            if n_free <= 0:
                full_actions = full
            else:
                full_actions = full
            return self._fallback(x0, "no_free_horizon" if n_free <= 0 else
                                  "infeasible_first_step", committed, full, target,
                                  np.inf, {}, applied)

        # 保守动作（与求解失败共用同一条安全路径）
        if force_fallback:
            return self._fallback(x0, "observation_dropout_timeout", committed, full,
                                  target, np.inf, {}, applied)

        lb = np.concatenate(([lo0], np.full(n_free - 1, float(self.u_lo[0]))))
        ub = np.concatenate(([hi0], np.full(n_free - 1, float(self.u_hi[0]))))
        box = Bounds(lb, ub)

        constraints = []
        if n_free > 1:
            rows = []
            for k in range(1, n_free):
                row = np.zeros(n_free)
                row[k] = 1.0
                row[k - 1] = -1.0
                rows.append(row)
            constraints = [LinearConstraint(
                np.array(rows),
                np.full(n_free - 1, float(self.du_lo[0])),
                np.full(n_free - 1, float(self.du_hi[0])),
            )]

        z0 = np.clip(self._warm[:n_free] if len(self._warm) >= n_free else
                     np.zeros(n_free), lb, ub)
        status: dict = {}
        try:
            res = minimize(
                lambda z: self._cost_and_grad(z, x0, full, n_free, target),
                z0, jac=True, method="SLSQP", bounds=box,
                constraints=constraints,
                options={"maxiter": self.cfg.max_iter, "ftol": self.cfg.tol,
                         "disp": False},
            )
            status = {"success": bool(res.success), "status": int(getattr(res, "status", -1)),
                      "message": str(res.message), "nit": int(getattr(res, "nit", -1))}
        except Exception as exc:  # 求解器异常 -> 保守动作
            return self._fallback(x0, f"solver_exception:{exc}", committed, full,
                                  target, np.inf, status, applied)

        z = np.asarray(res.x, dtype=float)
        full[self.D:] = z
        states = self._rollout(x0, full)
        if not res.success:
            return self._fallback(x0, f"solver_failed:{res.message}", committed, full,
                                  target, float(res.fun), status, applied)

        action = np.array([float(z[0])])  # 本周期下发；要 D 步后才生效
        # 热启动：自由段前移，末尾补最后一个值
        self._warm = np.concatenate((z[1:], z[-1:]))
        self._last_issued = action.copy()
        return PlanResult(
            success=True, action=action, applied_action=applied,
            committed_actions=committed.copy(), planned_states=states,
            planned_actions=full.reshape(-1, 1), target=target,
            cost=float(res.fun), fallback_used=False, reason="ok",
            solver_status=status,
        )
