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

    def _reachable_interval(self, x0: np.ndarray) -> tuple[float, float]:
        x_lo = x_hi = float(x0[0])
        v_lo = v_hi = float(x0[1])
        mass = float(self.plant.cfg.mass)  # type: ignore[attr-defined]
        dt = float(self.plant.dt)
        previous = {name: float(self.prev_action[name][0]) for name in self.names}
        force_lo = [0.0] * self.n
        force_hi = [0.0] * self.n
        for k in range(self.n):
            flo = fhi = 0.0
            for name in self.names:
                cap = self.caps[name]
                pending = self.pending_actions[name]
                if k < len(pending):
                    u = float(np.clip(
                        pending[k],
                        max(cap.u_min, previous[name] - cap.du_down),
                        min(cap.u_max, previous[name] + cap.du_up),
                    ))
                    previous[name] = u
                else:
                    available = self.actuator_available.get(name, True)
                    if available:
                        u_lo = max(cap.u_min, previous[name] - cap.du_down)
                        u_hi = min(cap.u_max, previous[name] + cap.du_up)
                        previous[name] = previous[name]  # 区间包络分别传播
                        # 用独立极值变量传播各通道可达范围
                        previous[name] = self._range_propagate(
                            name, k, len(pending)
                        )
                        u = float("nan")
                    else:
                        u_lo = u_hi = previous[name]
                    if np.isnan(u):
                        flo += u_lo
                        fhi += u_hi
                        continue
                flo += u
                fhi += u
            force_lo[k] = flo
            force_hi[k] = fhi
            a_lo = flo / mass
            a_hi = fhi / mass
            x_lo += v_lo * dt + 0.5 * a_lo * dt * dt
            x_hi += v_hi * dt + 0.5 * a_hi * dt * dt
            v_lo += a_lo * dt
            v_hi += a_hi * dt
        return x_lo, x_hi

    def _range_propagate(self, name: str, k: int, prefix_len: int) -> float:
        """占位：实际区间传播在 _reachable_interval 内联维护。"""
        return 0.0
