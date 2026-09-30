"""滚动时域实验循环（含执行器延迟与观测短时缺失）。

时间线（每个周期）：

1. 设备施加在途队列最旧的命令 ``applied``；延迟期间它**不是**刚规划出的量。
2. 读取观测；观测可能缺失（``observation_available_fn`` 返回 False）。缺失时
   用标称模型 + 真实生效动作做开环状态估计，短时保持规划；连续缺失超过
   ``MPCConfig.max_dropout_hold`` 后不再信任预测，直接发保守刹车命令。
3. 基于当前（估计）状态与剩余在途命令重新规划，只把新命令追加到队列尾部。
4. 真实对象（可能含扰动/饱和）前进一步，并记录全部计划与执行信息。

所有场景输入（初始状态、目标/扰动/丢观测函数、随机种子等）都在
:class:`ScenarioSpec` 中声明式保存，实验记录可原样重放（见 ``replay``）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import CartConfig, MPCConfig
from .controller import MPCController
from .logger import ExperimentLogger
from .plants.base import Plant

DisturbanceFn = Callable[[int, np.ndarray], float]
TargetFn = Callable[[int], float]
ObsAvailableFn = Callable[[int], bool]


@dataclass
class ScenarioSpec:
    """可重放的场景描述：据此可以重建完全相同的一次实验。"""

    name: str
    n_steps: int
    initial_state: tuple[float, float]
    # 函数用确定性序列表示（每步一个值），保证跨进程可重放
    targets: list[float]
    disturbances: list[float]
    obs_available: list[bool]
    actuator_delay_steps: int
    cart: dict[str, Any] = field(default_factory=dict)
    mpc: dict[str, Any] = field(default_factory=dict)
    planner_fail_steps: list[int] = field(default_factory=list)
    seed: int = 0

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str | Path) -> "ScenarioSpec":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return ScenarioSpec(**data)


def build_scenario(
    name: str,
    n_steps: int,
    initial_state: np.ndarray | tuple[float, float],
    target_fn: TargetFn,
    disturbance_fn: DisturbanceFn | None = None,
    obs_available_fn: ObsAvailableFn | None = None,
    planner_fail_steps: set[int] | list[int] | None = None,
    cart_config: CartConfig | None = None,
    mpc_config: MPCConfig | None = None,
) -> ScenarioSpec:
    """把函数形式的场景定义物化成确定性序列（便于保存与重放）。"""
    cart_config = cart_config or CartConfig()
    mpc_config = mpc_config or MPCConfig()
    targets, disturbances, available = [], [], []
    x0 = np.asarray(initial_state, dtype=float)
    for step in range(n_steps):
        targets.append(float(target_fn(step)))
        disturbances.append(0.0 if disturbance_fn is None
                            else float(disturbance_fn(step, x0)))
        available.append(True if obs_available_fn is None
                         else bool(obs_available_fn(step)))
    cart_fields = {"mass": cart_config.mass, "dt": cart_config.dt,
                   "u_max": cart_config.u_max,
                   "du_inc_max": cart_config.du_inc_max,
                   "du_dec_max": cart_config.du_dec_max,
                   "x_bounds": list(cart_config.x_bounds),
                   "v_bounds": list(cart_config.v_bounds)}
    mpc_fields = {k: getattr(mpc_config, k) for k in (
        "horizon", "q_pos", "q_vel", "r_u", "q_term_pos", "q_term_vel",
        "brake_gain", "actuator_delay_steps", "max_dropout_hold",
        "max_iter", "tol")}
    return ScenarioSpec(
        name=name, n_steps=n_steps,
        initial_state=(float(x0[0]), float(x0[1])),
        targets=targets, disturbances=disturbances, obs_available=available,
        actuator_delay_steps=mpc_config.actuator_delay_steps,
        cart=cart_fields, mpc=mpc_fields,
        planner_fail_steps=sorted(int(s) for s in (planner_fail_steps or [])),
    )


def run_experiment(
    plant: Plant,
    controller: MPCController,
    n_steps: int,
    initial_state: np.ndarray,
    target_fn: TargetFn,
    disturbance_fn: DisturbanceFn | None = None,
    obs_available_fn: ObsAvailableFn | None = None,
    planner_fail_steps: set[int] | list[int] | None = None,
    out_dir: str | Path | None = None,
    spec: ScenarioSpec | None = None,
) -> ExperimentLogger:
    logger = ExperimentLogger(plant.dt)
    obs = plant.reset(np.asarray(initial_state, dtype=float))
    controller.reset()

    D = controller.D
    committed = controller.initial_committed()
    estimate = obs.copy()          # 控制器相信的状态（观测缺失时为开环估计）
    dropout_streak = 0
    fail_steps = set(int(s) for s in (planner_fail_steps or []))

    for step in range(n_steps):
        target = float(target_fn(step))

        # 1) 本周期真正施加的动作 = 在途队列最旧命令（D=0 时为上一条命令）
        applied = controller.applied_this_step(committed)
        disturbance = 0.0 if disturbance_fn is None else float(disturbance_fn(step, plant.get_state()))
        obs_after = plant.step(applied, np.array([disturbance]))

        # 2) 观测链路：可用则用真实观测并重置计数；缺失则用标称模型开环估计
        obs_ok = True if obs_available_fn is None else bool(obs_available_fn(step))
        if obs_ok:
            dropout_streak = 0
            obs_used = obs_after.copy()
            estimate = obs_after.copy()
        else:
            dropout_streak += 1
            # 开环传播用的是**真正生效**的 applied（而非新规划量），且不含扰动
            estimate = np.asarray(
                plant.predict(estimate, applied), dtype=float).reshape(2)
            obs_used = estimate.copy()

        # 连续缺失超过允许保持步数 -> 不再信任预测，直接走保守刹车
        force_fallback = dropout_streak > controller.cfg.max_dropout_hold
        inject_failure = step in fail_steps

        # 3) 重新规划：在途前缀固定，只优化其后的自由动作
        if inject_failure:
            result = _plan_with_failure(controller, obs_used, target, committed,
                                        force_fallback)
        else:
            result = controller.plan(obs_used, target, committed,
                                     force_fallback=force_fallback)
        issued = result.action.copy()

        logger.record(
            step=step, obs_before=obs, obs_used=obs_used, target=target,
            planned_u0=issued[0], issued_u=issued[0], applied_u=applied[0],
            disturbance=disturbance, obs_after=obs_after, obs_available=obs_ok,
            dropout_streak=dropout_streak, delay_steps=D,
            success=result.success, fallback_used=result.fallback_used,
            reason=result.reason, cost=result.cost,
            planned_states=result.planned_states,
            planned_actions=result.planned_actions,
            committed=committed,
        )

        # 4) 队列前移：最旧命令已生效，新命令进入队尾（D 步后才施加）
        committed = controller.advance_committed(committed, issued)
        obs = obs_after

    if out_dir is not None:
        logger.save(out_dir)
        if spec is not None:
            spec.save(Path(out_dir) / "scenario.json")
    return logger


def _plan_with_failure(controller: MPCController, obs: np.ndarray, target: float,
                       committed: np.ndarray, force_fallback: bool):
    """临时令求解器抛异常，验证失败路径走明确保守动作而非旧预测。"""
    import scipy.optimize as opt

    module = __import__("control_lab.controller", fromlist=["minimize"])
    orig_pkg = module.minimize
    orig_scipy = opt.minimize

    def boom(*a, **k):
        raise RuntimeError("injected_solver_failure")

    module.minimize = boom  # type: ignore[assignment]
    opt.minimize = boom  # type: ignore[assignment]
    try:
        return controller.plan(obs, target, committed, force_fallback=force_fallback)
    finally:
        module.minimize = orig_pkg  # type: ignore[assignment]
        opt.minimize = orig_scipy  # type: ignore[assignment]


def run_from_spec(spec: ScenarioSpec, out_dir: str | Path | None = None) -> ExperimentLogger:
    """按 ScenarioSpec 重建并运行实验（确定性重放入口）。"""
    cart_cfg = CartConfig(**spec.cart)
    mpc_cfg = MPCConfig(**spec.mpc)
    plant = __import__(
        "control_lab.plants", fromlist=["CartPlant"]).CartPlant(cart_cfg)
    controller = MPCController(plant, mpc_cfg)
    targets = list(spec.targets)
    disturbances = list(spec.disturbances)
    available = list(spec.obs_available)
    return run_experiment(
        plant, controller, n_steps=spec.n_steps,
        initial_state=np.asarray(spec.initial_state, dtype=float),
        target_fn=lambda s: targets[s],
        disturbance_fn=lambda s, o: disturbances[s],
        obs_available_fn=lambda s: available[s],
        planner_fail_steps=set(spec.planner_fail_steps),
        out_dir=out_dir, spec=spec,
    )


def replay(scenario_path: str | Path, out_dir: str | Path | None = None) -> ExperimentLogger:
    """读取保存的场景描述并重放，便于核对历史实验。"""
    return run_from_spec(ScenarioSpec.load(scenario_path), out_dir=out_dir)
