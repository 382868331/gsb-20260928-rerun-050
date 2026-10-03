"""双执行器协同控制（含命令回执不确定时的保守决策）。

两台能力、延迟、允许方向不同的执行器共同作用于一维小车，合力
u_a + u_b 驱动同一目标。控制器为每条命令保留原命令归属（台账在
实验循环中维护），规划时区分三种信息：

- 已确认发生/未发生：作为确定性事实，已发生事实不回退；
- 已排队但尚未到确认时刻：按计划投影为名义点；
- 尚不能确认（in_flight/acked/uncertain）：用可能作用区间包住，
  状态以区间包络传播，不把不确定性压成一个精确预测值。

对不确定动作同时给出两套可比较方案：

- wait：该通道冻结新命令、保守等待回执，其余通道在自身剩余
  能力内继续调节；
- compensate：在最坏情形（区间鲁棒）仍满足目标与能力限制时，
  才允许在剩余能力范围内继续补偿；否则说明不能继续的原因。

能力降低后对所有“尚未发生”的动作重新用新约束核对，可能作用区间
按新限幅收缩；另一台的已发生作用不会算成本台某条命令的完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .capability import Capability
from .command_ledger import (
    ACKED,
    EXECUTED,
    IN_FLIGHT,
    NOT_APPLIED,
    QUEUED,
    UNCERTAIN,
    CommandLedger,
)
from .config import MPCConfig
from .plants.base import Plant

WAIT = "wait"
COMPENSATE = "compensate"


@dataclass
class ChannelProjection:
    """一个通道在预测时域内每拍的作用量描述。"""

    nominal: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    free_from: int
    uncertain_steps: list[int] = field(default_factory=list)
    rechecked_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nominal": np.nan_to_num(self.nominal).tolist(),
            "possible_min": np.nan_to_num(self.lower).tolist(),
            "possible_max": np.nan_to_num(self.upper).tolist(),
            "free_from": int(self.free_from),
            "uncertain_steps": [int(k) for k in self.uncertain_steps],
            "capability_rechecked_command_ids": [
                int(i) for i in self.rechecked_ids
            ],
        }


@dataclass
class DualPlanResult:
    success: bool
    policy: str
    actions: dict[str, float]
    planned_states: np.ndarray
    planned_actions: np.ndarray
    target: float
    fallback_used: bool
    reason: str
    policies: dict[str, dict] = field(default_factory=dict)
    capabilities: dict[str, dict] = field(default_factory=dict)
    channels: dict[str, dict] = field(default_factory=dict)
    target_attainable: bool = True
    reachable_target_min: float = -np.inf
    reachable_target_max: float = np.inf
    envelope_states: np.ndarray | None = None
    cancellation_first_step: float = 0.0
    cancellation_horizon: float = 0.0
    residual_error_first_step: float = 0.0
    residual_error_terminal: float = 0.0
    unmet_reason: str = ""
    uncertain_commands: dict[str, list[int]] = field(default_factory=dict)
    blocked_reasons: dict[str, str] = field(default_factory=dict)


class DualMPCController:
    def __init__(self, plant: Plant, config: MPCConfig | None = None,
                 names: tuple[str, str] = ("a", "b"),
                 capabilities: dict[str, Capability] | None = None,
                 policy: str = COMPENSATE):
        self.plant = plant
        self.cfg = config or MPCConfig()
        if plant.n_obs != 2 or plant.n_act != 1:
            raise ValueError("双执行器 MPC 针对一维小车")
        self.names = (str(names[0]), str(names[1]))
        self.n = self.cfg.horizon
        self.dt = float(plant.dt)
        self.mass = float(plant.cfg.mass)  # type: ignore[attr-defined]
        self.caps: dict[str, Capability] = {
            name: (capabilities or {}).get(name)
            or Capability.from_plant(plant)
            for name in self.names
        }
        if policy not in (WAIT, COMPENSATE):
            raise ValueError("policy 必须是 wait 或 compensate")
        self.policy = policy
        # 已确认的“上一拍实际作用”，是变化速率约束的基准（事实，不回退）。
        neutral = float(np.asarray(plant.neutral_action()).reshape(-1)[0])
        self.last_confirmed: dict[str, float] = {
            name: neutral for name in self.names
        }

    def reset(self) -> None:
        neutral = float(np.asarray(self.plant.neutral_action()).reshape(-1)[0])
        self.last_confirmed = {name: neutral for name in self.names}

    def update_capabilities(
        self,
        capabilities: dict[str, Capability],
        ledgers: dict[str, CommandLedger] | None = None,
    ) -> None:
        for name, capability in capabilities.items():
            self.caps[name] = capability
        # 能力降低后立即重新核对尚未发生的动作：可能作用区间用新限幅，
        # 已发生事实（confirmed_value）不回退。
        if ledgers is not None:
            for name, ledger in ledgers.items():
                cap = self.caps[name]
                for record in ledger.records():
                    if record.status in (EXECUTED, NOT_APPLIED):
                        continue
                    lo = max(cap.u_min, record.hold_value - cap.du_down)
                    hi = min(cap.u_max, record.hold_value + cap.du_up)
                    requested = float(np.clip(record.action, lo, hi))
                    record.action = requested
                    # 已实际执行的量（含部分执行的已发生部分）是事实，
                    # 不随能力重核改写；待核实余额也保持原记录。

    # ------------------------------------------------------------------
    # 台账 -> 时域区间投影
    # ------------------------------------------------------------------
    def project_channel(self, name: str, ledger: CommandLedger,
                        policy: str, current_step: int = 0) -> ChannelProjection:
        cap = self.caps[name]
        n = self.n
        nominal = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        uncertain_steps: list[int] = []
        rechecked: list[int] = []

        previous = self.last_confirmed[name]
        self._current_step = int(current_step)
        # 自由段起点：
        # - wait 策略且本通道存在已到生效时刻但未确认的命令：整个通道
        #   冻结等待人工核实，绝不把该命令当作未执行而重发；
        # - 否则只由“尚未到生效时刻”的排队命令占据前缀，已到生效时刻
        #   但未确认的命令只在其生效拍贡献可能区间，不冻结之后的新命令。
        waiting_unconfirmed = any(
            r.status in (IN_FLIGHT, UNCERTAIN)
            and r.due_step <= current_step
            for r in ledger.records()
        )
        if policy == WAIT and waiting_unconfirmed:
            slot = n
        else:
            future_due = [
                int(r.due_step) - int(current_step)
                for r in ledger.records()
                if r.status in (QUEUED, ACKED)
                and int(r.due_step) > int(current_step)
            ]
            slot = (max(future_due) + 1) if future_due else 0
            slot = min(n, max(0, slot))
        for record in ledger.records():
            if record.status == EXECUTED:
                previous = float(record.confirmed_value)
                continue
            if record.status == NOT_APPLIED:
                previous = float(record.hold_value)
                continue
            k = int(record.due_step) - int(current_step)
            if k < 0 or k >= n:
                continue
            lo_rate = max(cap.u_min, previous - cap.du_down)
            hi_rate = min(cap.u_max, previous + cap.du_up)
            requested = float(np.clip(record.action, lo_rate, hi_rate))
            rechecked.append(record.command_id)
            if record.status in (QUEUED, ACKED) and record.due_step > self._current_step:
                nominal[k] = lower[k] = upper[k] = requested
                previous = requested
                continue

            if record.status == ACKED and record.due_step <= self._current_step:
                # 设备确认收到但尚未确认作用量：仍是不确定，落入下面区间。
                pass

            # in_flight / acked / uncertain —— 尚不能确认。
            raw_lo, raw_hi = ledger.possible_effect_interval(record)
            bound_lo = max(cap.u_min, record.hold_value - cap.du_down)
            bound_hi = min(cap.u_max, record.hold_value + cap.du_up)
            lo_v = float(np.clip(raw_lo, bound_lo, bound_hi))
            hi_v = float(np.clip(raw_hi, bound_lo, bound_hi))
            lo_v, hi_v = min(lo_v, hi_v), max(lo_v, hi_v)
            if policy == WAIT:
                nominal[k] = float(record.hold_value)
                lower[k] = min(lo_v, float(record.hold_value))
                upper[k] = max(hi_v, float(record.hold_value))
            else:
                nominal[k] = float(np.clip(requested, lo_v, hi_v))
                lower[k] = lo_v
                upper[k] = hi_v
            uncertain_steps.append(k)
            previous = hi_v

        free_from = n if slot >= n else slot
        hold = self.last_confirmed[name]
        range_lo = range_hi = hold
        for k in range(n):
            if not np.isnan(nominal[k]):
                hold = float(nominal[k])
                range_lo = float(lower[k])
                range_hi = float(upper[k])
                continue
            if k < free_from:
                # 该拍被队列/不确定前缀锁定为保持值。
                nominal[k] = lower[k] = upper[k] = hold
                range_lo = range_hi = hold
                continue
            # 自由段：名义值留给优化器；可能作用区间按变化速率独立传播。
            next_lo = max(cap.u_min, range_lo - cap.du_down)
            next_hi = min(cap.u_max, range_hi + cap.du_up)
            lower[k], upper[k] = next_lo, next_hi
            range_lo, range_hi = next_lo, next_hi

        return ChannelProjection(
            nominal=nominal, lower=lower, upper=upper,
            free_from=free_from, uncertain_steps=uncertain_steps,
        rechecked_ids=sorted(set(rechecked)),
        )

    # ------------------------------------------------------------------
    # 区间包络与名义轨迹
    # ------------------------------------------------------------------
    def envelope_rollout(self, x0: np.ndarray,
                         projections: dict[str, ChannelProjection]
                         ) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        force_lo = np.zeros(n)
        force_hi = np.zeros(n)
        for name in self.names:
            proj = projections[name]
            force_lo += np.nan_to_num(proj.lower, nan=0.0)
            force_hi += np.nan_to_num(proj.upper, nan=0.0)
        x_lo = np.zeros((n + 1, 2))
        x_hi = np.zeros((n + 1, 2))
        x_lo[0] = x_hi[0] = np.asarray(x0, dtype=float)
        for k in range(n):
            a_lo = force_lo[k] / self.mass
            a_hi = force_hi[k] / self.mass
            v_new_lo = x_lo[k, 1] + self.dt * a_lo
            v_new_hi = x_hi[k, 1] + self.dt * a_hi
            x_lo[k + 1, 1] = min(v_new_lo, v_new_hi)
            x_hi[k + 1, 1] = max(v_new_lo, v_new_hi)
            x_lo[k + 1, 0] = x_lo[k, 0] + self.dt * x_lo[k + 1, 1]
            x_hi[k + 1, 0] = x_hi[k, 0] + self.dt * x_hi[k + 1, 1]
        # 形状 (T, 2, 2)：[时刻][状态维 0=位置/1=速度][0=下界,1=上界]。
        envelope = np.stack([x_lo, x_hi], axis=2)
        forces = np.column_stack([force_lo, force_hi])
        return envelope, forces

    def nominal_rollout(self, x0: np.ndarray, z: np.ndarray,
                        projections: dict[str, ChannelProjection]
                        ) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        channel_series: dict[str, np.ndarray] = {}
        for idx, name in enumerate(self.names):
            proj = projections[name]
            series = proj.nominal.copy()
            ff = proj.free_from
            if ff < n:
                free_values = z[idx * n + ff:(idx + 1) * n]
                series = np.concatenate([series[:ff], free_values])
            for k in range(n):
                if np.isnan(series[k]):
                    series[k] = self.last_confirmed[name]
            channel_series[name] = series
        ua = channel_series[self.names[0]]
        ub = channel_series[self.names[1]]
        total = ua + ub
        states = np.zeros((n + 1, 2))
        states[0] = np.asarray(x0, dtype=float)
        for k in range(n):
            v = states[k, 1] + self.dt * total[k] / self.mass
            states[k + 1] = np.array([states[k, 0] + self.dt * v, v])
        return states, np.column_stack([ua, ub, total])

    @staticmethod
    def _cancellation(force_interval: np.ndarray) -> float:
        """两通道方向相反时能确认的实际抵消量（合力区间同号部分）。"""
        lo, hi = float(force_interval[0]), float(force_interval[1])
        if lo > 0.0 or hi < 0.0:
            return min(abs(lo), abs(hi))
        return 0.0

    def _free_bounds(self, projection: ChannelProjection,
                     freeze: bool) -> tuple[np.ndarray, np.ndarray]:
        cap = self.caps[self._current_name]
        n = self.n
        ff = projection.free_from
        lo = np.empty(n)
        hi = np.empty(n)
        # 前缀（排队/保持）不是优化变量：锁定为名义值，变化速率从
        # 名义前缀的最后一拍接续，杜绝自由段首步越过变化限制。
        for k in range(ff):
            lo[k] = hi[k] = float(projection.nominal[k])
        base = float(projection.nominal[ff - 1]) if ff > 0 \
            else self.last_confirmed[self._current_name]
        for k in range(ff, n):
            if freeze:
                lo[k] = hi[k] = self.last_confirmed[self._current_name]
                continue
            prev_lo = base if k == ff else lo[k - 1]
            prev_hi = base if k == ff else hi[k - 1]
            lo[k] = max(cap.u_min, prev_lo - cap.du_down)
            hi[k] = min(cap.u_max, prev_hi + cap.du_up)
            # 能力在规划当拍收缩时，前缀保持值可能暂时落在新可达区间外，
            # 退化为以当前已确认值冻结，避免上下界反转导致求解失败。
            if lo[k] > hi[k] + 1e-12:
                lo[k] = hi[k] = float(np.clip(
                    self.last_confirmed[self._current_name],
                    max(cap.u_min, hi[k]), min(cap.u_max, lo[k]),
                ))
        return lo, hi

    # ------------------------------------------------------------------
    # 单方案规划（wait / compensate）
    # ------------------------------------------------------------------
    def plan_policy(self, obs: np.ndarray, target: float,
                    ledgers: dict[str, CommandLedger], policy: str,
                    current_step: int = 0) -> dict:
        x0 = np.asarray(obs, dtype=float).reshape(2)
        projections = {
            name: self.project_channel(
                name, ledgers[name], policy, current_step=int(current_step)
            )
            for name in self.names
        }
        envelope, force_envelope = self.envelope_rollout(x0, projections)
        terminal_lo = float(envelope[-1, 0, 0])
        terminal_hi = float(envelope[-1, 0, 1])
        target_attainable = terminal_lo - 1e-7 <= target <= terminal_hi + 1e-7

        n = self.n
        x0z = np.zeros(2 * n)
        lb = np.zeros(2 * n)
        ub = np.zeros(2 * n)
        rate_rows: list[np.ndarray] = []
        rate_lb: list[float] = []
        rate_ub: list[float] = []
        freeze_flags: dict[str, bool] = {}
        blocked_reasons: dict[str, str] = {}
        for idx, name in enumerate(self.names):
            self._current_name = name
            proj = projections[name]
            freeze = bool(policy == WAIT and proj.free_from >= n)
            freeze_flags[name] = freeze
            if freeze:
                blocked_reasons[name] = (
                    "waiting_for_receipt:command_outcome_unconfirmed"
                )
            clo, chi = self._free_bounds(proj, freeze)
            seg = slice(idx * n, (idx + 1) * n)
            lb[seg] = clo
            ub[seg] = chi
            x0z[seg] = np.clip(np.zeros(n), clo, chi)
            cap = self.caps[name]
            ff = proj.free_from
            start = idx * n
            # 从前缀末拍到自由段首拍、以及自由段内部，相邻命令差
            # 都受该通道变化速率限制。
            for k in range(ff, n):
                row = np.zeros(2 * n)
                row[start + k] = 1.0
                row[start + k - 1] = -1.0
                rate_rows.append(row)
                rate_lb.append(-cap.du_down)
                rate_ub.append(cap.du_up)

        def objective(z: np.ndarray) -> float:
            states, actions = self.nominal_rollout(x0, z, projections)
            err = states[:, 0] - target
            stage = (self.cfg.q_pos * err[:-1] ** 2
                     + self.cfg.q_vel * states[:-1, 1] ** 2
                     + self.cfg.r_u * actions[:, 2] ** 2).sum()
            terminal = (self.cfg.q_term_pos * err[-1] ** 2
                        + self.cfg.q_term_vel * states[-1, 1] ** 2)
            worst_miss = max(0.0, terminal_lo - target, target - terminal_hi)
            robust = self.cfg.q_term_pos * worst_miss ** 2
            return float(stage + terminal + robust)

        constraints = []
        if rate_rows:
            constraints.append(LinearConstraint(
                np.array(rate_rows), np.array(rate_lb), np.array(rate_ub)
            ))
        success = True
        reason = "ok"
        try:
            res = minimize(
                objective, x0z, method="SLSQP",
                bounds=Bounds(lb, ub), constraints=constraints,
                options={"maxiter": self.cfg.max_iter, "ftol": self.cfg.tol,
                         "disp": False},
            )
            z = np.asarray(res.x, dtype=float)
            success = bool(res.success)
            if not success:
                reason = f"solver_failed:{res.message}"
        except Exception as exc:  # pragma: no cover
            z = x0z
            success = False
            reason = f"solver_exception:{exc}"

        states, actions = self.nominal_rollout(x0, z, projections)
        first_actions = {}
        for idx, name in enumerate(self.names):
            ff = projections[name].free_from
            if ff >= n:
                first_actions[name] = float(self.last_confirmed[name])
            else:
                first_actions[name] = float(z[idx * n + ff])
        for name in self.names:
            if freeze_flags[name]:
                first_actions[name] = self.last_confirmed[name]

        unmet = ""
        if not target_attainable:
            unmet = (
                "target_outside_reachable_envelope:"
                f"[{terminal_lo:.4f},{terminal_hi:.4f}]"
            )
        cancellation_first = self._cancellation(force_envelope[0])
        cancellation_horizon = float(sum(
            self._cancellation(force_envelope[k]) for k in range(n)
        ))
        nominal_first_force = float(actions[0, 2])
        residual_first = -(
            x0[1] + 0.5 * self.dt * nominal_first_force / self.mass
        ) * self.dt
        residual_terminal = (
            max(abs(target - terminal_lo), abs(target - terminal_hi))
            if not target_attainable else 0.0
        )

        return {
            "success": success,
            "reason": reason,
            "actions": first_actions,
            "z": z,
            "planned_states": states,
            "planned_actions": actions,
            "envelope": envelope,
            "force_envelope": force_envelope,
            "reachable_target_min": terminal_lo,
            "reachable_target_max": terminal_hi,
            "target_attainable": target_attainable,
            "cancellation_first_step": float(cancellation_first),
            "cancellation_horizon": cancellation_horizon,
            "residual_error_first_step": float(residual_first),
            "residual_error_terminal": float(residual_terminal),
            "unmet_reason": unmet,
            "blocked_reasons": blocked_reasons,
            "projections": projections,
            "freeze": freeze_flags,
        }

    # ------------------------------------------------------------------
    # 对外规划入口：同时计算等待与补偿两套方案
    # ------------------------------------------------------------------
    def plan(self, obs: np.ndarray, target: float,
             ledgers: dict[str, CommandLedger],
             current_step: int = 0) -> DualPlanResult:
        policies = {
            WAIT: self.plan_policy(obs, target, ledgers, WAIT, current_step),
            COMPENSATE: self.plan_policy(
                obs, target, ledgers, COMPENSATE, current_step
            ),
        }
        chosen = policies[self.policy]
        fallback = not chosen["success"]
        actions = dict(chosen["actions"])
        if fallback:
            # 保守动作：各通道在自身变化限制内与速度反向刹车；
            # 不确定通道冻结，绝不把未确认命令当作未执行而重发。
            vel = float(np.asarray(obs, dtype=float).reshape(2)[1])
            for name in self.names:
                cap = self.caps[name]
                base = self.last_confirmed[name]
                brake = float(np.clip(
                    -self.cfg.brake_gain * vel,
                    max(cap.u_min, base - cap.du_down),
                    min(cap.u_max, base + cap.du_up),
                ))
                actions[name] = base if chosen["freeze"].get(name) else brake

        uncertain = {
            name: [r.command_id for r in ledgers[name].unconfirmed()]
            for name in self.names
        }
        return DualPlanResult(
            success=bool(chosen["success"]),
            policy=self.policy,
            actions=actions,
            planned_states=chosen["planned_states"],
            planned_actions=chosen["planned_actions"],
            target=float(target),
            fallback_used=fallback,
            reason=chosen["reason"],
            policies={
                policy_name: {
                    "success": bool(info["success"]),
                    "target_attainable": bool(info["target_attainable"]),
                    "reachable_target_min": float(info["reachable_target_min"]),
                    "reachable_target_max": float(info["reachable_target_max"]),
                    "first_actions": {
                        k: float(v) for k, v in info["actions"].items()
                    },
                    "cancellation_first_step": float(
                        info["cancellation_first_step"]
                    ),
                    "cancellation_horizon": float(info["cancellation_horizon"]),
                    "residual_error_first_step": float(
                        info["residual_error_first_step"]
                    ),
                    "residual_error_terminal": float(
                        info["residual_error_terminal"]
                    ),
                    "unmet_reason": info["unmet_reason"],
                    "blocked_reasons": dict(info["blocked_reasons"]),
                    "frozen_channels": [
                        ch for ch, frozen in info["freeze"].items() if frozen
                    ],
                }
                for policy_name, info in policies.items()
            },
            capabilities={
                name: cap.to_dict() for name, cap in self.caps.items()
            },
            channels={
                name: chosen["projections"][name].to_dict()
                for name in self.names
            },
            target_attainable=bool(chosen["target_attainable"]),
            reachable_target_min=float(chosen["reachable_target_min"]),
            reachable_target_max=float(chosen["reachable_target_max"]),
            envelope_states=np.asarray(chosen["envelope"]),
            cancellation_first_step=float(chosen["cancellation_first_step"]),
            cancellation_horizon=float(chosen["cancellation_horizon"]),
            residual_error_first_step=float(
                chosen["residual_error_first_step"]
            ),
            residual_error_terminal=float(chosen["residual_error_terminal"]),
            unmet_reason=chosen["unmet_reason"],
            uncertain_commands=uncertain,
            blocked_reasons=dict(chosen["blocked_reasons"]),
        )
