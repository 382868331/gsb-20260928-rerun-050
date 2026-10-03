"""双执行器有限时域 MPC。

优化变量为两台执行器各自新提交的命令序列；状态预测使用各队列中
尚未生效命令的投影。两台的幅度、变化速率、延迟和允许方向分别作为
硬约束，控制器不会让一台承担超过其能力的力。

能力恢复后，每台承担的目标比例只按固定步长逐步调整，避免两台同时
补偿同一误差造成合力过冲。单台失去观测时冻结该通道新命令，只用
另一台在自身限制内调节，并重新说明目标可达区间与未满足原因。
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
    actions: dict[str, np.ndarray]
    planned_states: np.ndarray
    planned_actions: np.ndarray  # 形状 (N, 3): ua, ub, total
    target: float
    cost: float
    fallback_used: bool
    reason: str
    solver_status: dict = field(default_factory=dict)
    capabilities: dict[str, dict] = field(default_factory=dict)
    target_attainable: bool = True
    reachable_target_min: float = -np.inf
    reachable_target_max: float = np.inf
    cancellation_first_step: float = 0.0
    cancellation_horizon: float = 0.0
    unmet_reason: str = ""
    allocation_share: dict[str, float] = field(default_factory=dict)
    allocation_target_share: dict[str, float] = field(default_factory=dict)
    reallocation_active: bool = False
    actuator_observation_available: dict[str, bool] = field(default_factory=dict)


class DualMPCController:
    """两台执行器协同作用于一维小车；合力 u_a + u_b 驱动同一目标。"""

    def __init__(self, plant: Plant, config: MPCConfig | None = None,
                 names: tuple[str, str] = ("a", "b"),
                 capabilities: dict[str, Capability] | None = None,
                 share_ramp: float = 0.05,
                 allocation_weight: float = 0.6):
        self.plant = plant
        self.cfg = config or MPCConfig()
        if plant.n_obs != 2 or plant.n_act != 1:
            raise ValueError("双执行器 MPC 针对一维小车")
        self.names = (str(names[0]), str(names[1]))
        self.n = self.cfg.horizon
        dt = plant.dt
        mass = plant.cfg.mass  # type: ignore[attr-defined]
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.b = np.array([0.5 * dt * dt / mass, dt / mass])
        self.caps: dict[str, Capability] = {
            name: (capabilities or {}).get(name)
            or Capability.from_plant(plant)
            for name in self.names
        }
        self.share_ramp = float(share_ramp)
        self.allocation_weight = float(allocation_weight)
        self.prev_action: dict[str, np.ndarray] = {
            name: plant.neutral_action().copy() for name in self.names
        }
        self.pending_actions: dict[str, list[np.ndarray]] = {
            name: [] for name in self.names
        }
        self.share = {name: 0.5 for name in self.names}
        self.target_share = {name: 0.5 for name in self.names}
        self.reallocation_active = False
        self.actuator_available = {name: True for name in self.names}
        self._warm = np.zeros(2 * self.n)

    def reset(self) -> None:
        self.prev_action = {
            name: self.plant.neutral_action().copy() for name in self.names
        }
        self.pending_actions = {name: [] for name in self.names}
        self.share = {name: 0.5 for name in self.names}
        self.target_share = {name: 0.5 for name in self.names}
        self.reallocation_active = False
        self.actuator_available = {name: True for name in self.names}
        self._warm = np.zeros(2 * self.n)

    def update_capabilities(self, capabilities: dict[str, Capability]) -> None:
        for name, capability in capabilities.items():
            self.caps[name] = capability
        self._update_target_shares()

    def _authority_weight(self, name: str) -> float:
        cap = self.caps[name]
        if not self.actuator_available.get(name, True):
            return 0.0
        # 同时反映可用幅度与双向调节能力；完全单向执行器权重减半。
        span = max(0.0, cap.u_max - cap.u_min)
        direction_factor = {
            "both": 1.0, "increase_only": 0.5,
            "decrease_only": 0.5, "hold_only": 0.0,
        }[cap.direction()]
        return span * direction_factor + 1e-9

    def _update_target_shares(self) -> dict[str, float]:
        weights = {name: self._authority_weight(name) for name in self.names}
        total_weight = sum(weights.values())
        if total_weight <= 1e-12:
            self.target_share = {name: 0.5 for name in self.names}
        else:
            self.target_share = {
                name: float(np.clip(weights[name] / total_weight, 0.0, 1.0))
                for name in self.names
            }
        return self.target_share

    def _ramp_shares(self) -> None:
        self._update_target_shares()
        active = False
        for name in self.names:
            if abs(self.share[name] - self.target_share[name]) > 1e-9:
                active = True
                step = float(np.clip(
                    self.target_share[name] - self.share[name],
                    -self.share_ramp, self.share_ramp,
                ))
                self.share[name] = float(np.clip(
                    self.share[name] + step, 0.0, 1.0
                ))
        other = self.names[1]
        self.share[other] = max(0.0, 1.0 - self.share[self.names[0]])
        self.reallocation_active = active

    def _delay(self, name: str) -> int:
        return min(len(self.pending_actions[name]), self.n)

    def _channel_series(self, z: np.ndarray, name: str) -> np.ndarray:
        offset = 0 if name == self.names[0] else self.n
        delay = self._delay(name)
        series = np.zeros(self.n)
        if delay:
            pending = np.asarray(self.pending_actions[name], dtype=float).reshape(-1)
            series[:delay] = pending[:delay]
        if delay < self.n:
            series[delay:] = z[offset + delay:offset + self.n]
        return series

    def _effective(self, z: np.ndarray) -> np.ndarray:
        channels = np.column_stack([
            self._channel_series(z, name) for name in self.names
        ])
        total = channels.sum(axis=1)
        return np.column_stack([channels, total])

    def _rollout(self, x0: np.ndarray, effective: np.ndarray) -> np.ndarray:
        states = np.zeros((self.n + 1, 2))
        states[0] = x0
        for k in range(self.n):
            acc = effective[k, 2] / self.plant.cfg.mass  # type: ignore[attr-defined]
            states[k + 1] = self.A @ states[k] + self.b * acc
        return states


    def set_actuator_available(self, name: str, available: bool) -> None:
        self.actuator_available[str(name)] = bool(available)

    def _reachable_interval(self, x0: np.ndarray,
                            caps: dict[str, Capability] | None = None,
                            ) -> tuple[float, float]:
        """当前能力（含排队命令）下，时域末端位置的可达区间。"""
        caps = caps or self.caps
        x_lo = x_hi = float(x0[0])
        v_lo = v_hi = float(x0[1])
        mass = float(self.plant.cfg.mass)  # type: ignore[attr-defined]
        dt = float(self.plant.dt)
        prev_lo = {n: float(self.prev_action[n][0]) for n in self.names}
        prev_hi = dict(prev_lo)
        for k in range(self.n):
            force_lo = force_hi = 0.0
            for name in self.names:
                cap = caps[name]
                pending = self.pending_actions[name]
                if k < len(pending):
                    # 排队命令固定，只按变化限制裁剪出单值
                    lo = max(cap.u_min, prev_lo[name] - cap.du_down)
                    hi = min(cap.u_max, prev_hi[name] + cap.du_up)
                    value_lo = float(np.clip(pending[k][0], lo, hi))
                    value_hi = float(np.clip(pending[k][0], lo, hi))
                elif self.actuator_available.get(name, True):
                    value_lo = max(cap.u_min, prev_lo[name] - cap.du_down)
                    value_hi = min(cap.u_max, prev_hi[name] + cap.du_up)
                else:
                    value_lo = value_hi = prev_lo[name]
                prev_lo[name] = value_lo
                prev_hi[name] = value_hi
                force_lo += value_lo
                force_hi += value_hi
            a_lo = force_lo / mass
            a_hi = force_hi / mass
            x_lo += v_lo * dt + 0.5 * a_lo * dt * dt
            x_hi += v_hi * dt + 0.5 * a_hi * dt * dt
            v_lo += a_lo * dt
            v_hi += a_hi * dt
        return x_lo, x_hi

    def _brake_actions(self, obs: np.ndarray,
                       caps: dict[str, Capability]) -> dict[str, np.ndarray]:
        """明确保守动作：按分担比例刹车并裁剪到各台能力内。"""
        v = float(obs[1])
        actions: dict[str, np.ndarray] = {}
        for name in self.names:
            cap = caps[name]
            prev = float(self.prev_action[name][0])
            raw = -self.cfg.brake_gain * v * self.share[name]
            lo = max(cap.u_min, prev - cap.du_down)
            hi = min(cap.u_max, prev + cap.du_up)
            actions[name] = np.array([float(np.clip(raw, lo, hi))])
        return actions

    def plan(self, obs: np.ndarray, target: float,
             observation_available: bool = True,
             observation_age: int = 0,
             capability_overrides: dict[str, Capability] | None = None,
             ) -> DualPlanResult:
        """滚动优化两台执行器的新命令序列。

        capability_overrides 用于"剩余能力内继续补偿"：调用方把不确定
        命令可能的作用从能力中扣除后传入，控制器只在收缩后的能力内
        规划，不确定性不会被藏进一个精确预测值。
        """
        obs = np.asarray(obs, dtype=float).reshape(-1)
        target = float(target)
        caps = dict(self.caps)
        if capability_overrides:
            caps.update(capability_overrides)
        self._ramp_shares()

        if not observation_available:
            actions = self._brake_actions(obs, caps)
            effective = np.column_stack([
                np.full(self.n, actions[self.names[0]][0]),
                np.full(self.n, actions[self.names[1]][0]),
            ])
            effective = np.column_stack([effective, effective.sum(axis=1)])
            states = self._rollout(obs, effective)
            return DualPlanResult(
                success=False, actions=actions, planned_states=states,
                planned_actions=effective, target=target, cost=np.inf,
                fallback_used=True, reason="observation_missing",
                capabilities={n: caps[n].to_dict() for n in self.names},
                target_attainable=False,
                unmet_reason="observation_missing",
                allocation_share=dict(self.share),
                allocation_target_share=dict(self.target_share),
                reallocation_active=self.reallocation_active,
                actuator_observation_available=dict(self.actuator_available),
            )

        n = self.n
        cfg = self.cfg
        delays = {name: self._delay(name) for name in self.names}

        def objective(z: np.ndarray) -> float:
            eff = self._effective(z)
            states = self._rollout(obs, eff)
            pos_err = states[1:, 0] - target
            cost = cfg.q_pos * float(pos_err @ pos_err)
            cost += cfg.q_vel * float(states[1:, 1] @ states[1:, 1])
            cost += cfg.q_term_pos * (states[-1, 0] - target) ** 2
            cost += cfg.q_term_vel * states[-1, 1] ** 2
            cost += cfg.r_u * float(z @ z)
            total = eff[:, 2]
            for idx, name in enumerate(self.names):
                dev = eff[:, idx] - self.share[name] * total
                cost += self.allocation_weight * float(dev @ dev)
            return cost

        bounds: list[tuple[float, float]] = []
        for name in self.names:
            cap = caps[name]
            for k in range(n):
                if k < delays[name]:
                    bounds.append((0.0, 0.0))  # 延迟段由排队命令占据
                else:
                    bounds.append((cap.u_min, cap.u_max))

        rows: list[np.ndarray] = []
        lb: list[float] = []
        ub: list[float] = []
        for c, name in enumerate(self.names):
            cap = caps[name]
            offset = c * n
            delay = delays[name]
            for k in range(delay, n):
                row = np.zeros(2 * n)
                row[offset + k] = 1.0
                if k > delay:
                    row[offset + k - 1] = -1.0
                    prev = 0.0
                elif delay > 0:
                    prev = float(np.asarray(
                        self.pending_actions[name], dtype=float
                    ).reshape(-1)[-1])
                else:
                    prev = float(self.prev_action[name][0])
                rows.append(row)
                lb.append(prev - cap.du_down)
                ub.append(prev + cap.du_up)
        constraints = []
        if rows:
            constraints.append(LinearConstraint(
                np.vstack(rows), np.array(lb), np.array(ub)))

        z0 = self._warm.copy()
        for c, name in enumerate(self.names):
            z0[c * n:c * n + delays[name]] = 0.0
        solver_status: dict = {}
        try:
            res = minimize(objective, z0, method="SLSQP", bounds=Bounds(
                [b[0] for b in bounds], [b[1] for b in bounds]),
                constraints=constraints,
                options={"maxiter": cfg.max_iter, "ftol": cfg.tol})
            solver_status = {"success": bool(res.success),
                             "message": str(res.message),
                             "nit": int(getattr(res, "nit", 0))}
            z = res.x if np.all(np.isfinite(res.x)) else None
        except Exception as exc:  # 求解器异常：走明确保守动作
            solver_status = {"success": False, "message": f"exception: {exc}"}
            z = None

        if z is None or not solver_status.get("success", False):
            actions = self._brake_actions(obs, caps)
            effective = np.column_stack([
                np.full(n, actions[self.names[0]][0]),
                np.full(n, actions[self.names[1]][0]),
            ])
            effective = np.column_stack([effective, effective.sum(axis=1)])
            states = self._rollout(obs, effective)
            return DualPlanResult(
                success=False, actions=actions, planned_states=states,
                planned_actions=effective, target=target, cost=np.inf,
                fallback_used=True,
                reason="solver_failure:" + solver_status.get("message", ""),
                solver_status=solver_status,
                capabilities={nm: caps[nm].to_dict() for nm in self.names},
                target_attainable=False,
                unmet_reason="solver_failure",
                allocation_share=dict(self.share),
                allocation_target_share=dict(self.target_share),
                reallocation_active=self.reallocation_active,
                actuator_observation_available=dict(self.actuator_available),
            )

        self._warm = np.roll(z, -1)
        self._warm[-1] = z[-1]
        effective = self._effective(z)
        states = self._rollout(obs, effective)
        actions = {}
        for c, name in enumerate(self.names):
            delay = delays[name]
            if delay < n:
                value = z[c * n + delay]
            else:
                value = float(self.prev_action[name][0])
            cap = caps[name]
            prev = float(self.prev_action[name][0])
            value = float(np.clip(
                value, max(cap.u_min, prev - cap.du_down),
                min(cap.u_max, prev + cap.du_up)))
            actions[name] = np.array([value])

        reach_lo, reach_hi = self._reachable_interval(obs, caps)
        attainable = bool(reach_lo - 1e-9 <= target <= reach_hi + 1e-9)
        unmet = ""
        if not attainable:
            unmet = ("target_outside_reachable:"
                     f"[{reach_lo:.3f},{reach_hi:.3f}] 不含 {target:.3f}")

        ua0 = float(effective[0, 0])
        ub0 = float(effective[0, 1])
        cancel_first = min(abs(ua0), abs(ub0)) if ua0 * ub0 < 0 else 0.0
        cancel_horizon = float(sum(
            min(abs(effective[k, 0]), abs(effective[k, 1]))
            for k in range(n) if effective[k, 0] * effective[k, 1] < 0
        ))

        return DualPlanResult(
            success=True, actions=actions, planned_states=states,
            planned_actions=effective, target=target,
            cost=float(objective(z)), fallback_used=False, reason="ok",
            solver_status=solver_status,
            capabilities={nm: caps[nm].to_dict() for nm in self.names},
            target_attainable=attainable,
            reachable_target_min=float(reach_lo),
            reachable_target_max=float(reach_hi),
            cancellation_first_step=float(cancel_first),
            cancellation_horizon=float(cancel_horizon),
            unmet_reason=unmet,
            allocation_share=dict(self.share),
            allocation_target_share=dict(self.target_share),
            reallocation_active=self.reallocation_active,
            actuator_observation_available=dict(self.actuator_available),
        )

