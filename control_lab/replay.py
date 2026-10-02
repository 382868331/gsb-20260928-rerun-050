"""内置场景重放工具。

记录中的 target/disturbance/observation mask 使用确定性公式，因此可根据
scenario 名称和参数重新生成相同轨迹。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CartConfig, MPCConfig
from .controller import MPCController
from .dual_controller import DualMPCController
from .dual_experiment import run_dual_experiment
from .experiment import run_experiment
from .capability import Capability
from .plants import CartPlant

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class ReplaySpec:
    name: str
    initial_state: tuple[float, float]
    n_steps: int = 120
    delay_steps: int = 0
    u_max: float = 2.0
    du_up_max: float = 4.0
    du_down_max: float = 4.0
    fail_steps: tuple[int, ...] = ()
    capability_changes: tuple[tuple[int, float, float, float, float, int, bool], ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "initial_state": list(self.initial_state),
            "n_steps": self.n_steps,
            "actuator_delay_steps": self.delay_steps,
            "cart": {
                "u_max": self.u_max,
                "du_up_max": self.du_up_max,
                "du_down_max": self.du_down_max,
            },
            "fail_steps": list(self.fail_steps),
            "capability_changes": [
                {
                    "step": item[0],
                    "u_min": item[1],
                    "u_max": item[2],
                    "du_up": item[3],
                    "du_down": item[4],
                    "delay_steps": item[5],
                    "withdraw_allowed": item[6],
                }
                for item in self.capability_changes
            ],
        }


class ForcedFailureController(MPCController):
    def __init__(self, *args, fail_steps: set[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_count = 0
        self.fail_steps = fail_steps or set()

    def plan(self, obs, target, **kwargs):
        current = self._step_count
        self._step_count += 1
        if current in self.fail_steps:
            return self.conservative_action(
                np.asarray(obs, dtype=float), target,
                "solver_exception:injected_solver_failure",
            )
        return super().plan(obs, target, **kwargs)


def target(name: str, step: int) -> float:
    if name == "01_constant_disturbance":
        return 1.0
    if name == "02_target_jump":
        if step < 40:
            return 0.0
        if step < 80:
            return 3.0
        return -2.5
    if name == "03_planner_failure":
        return -1.5
    if name == "04_delay_asymmetric_rate_observation_loss":
        if step < 35:
            return 0.0
        if step < 75:
            return 3.0
        return -2.0
    if name == "05_runtime_capability_change":
        return 2.0 if step >= 25 else 0.0
    raise KeyError(f"未知场景: {name}")


def disturbance(name: str, step: int, obs: np.ndarray) -> float:
    if name == "01_constant_disturbance":
        return 0.9 if step >= 40 else 0.0
    if name == "02_target_jump":
        return 0.15 * np.sin(0.5 * step)
    if name == "04_delay_asymmetric_rate_observation_loss":
        return 0.6 + 0.1 * np.sin(0.2 * step)
    if name == "05_runtime_capability_change":
        return 0.25
    return 0.0


def observation_available(name: str, step: int) -> bool:
    if name == "04_delay_asymmetric_rate_observation_loss":
        return not (45 <= step <= 52)
    if name == "05_runtime_capability_change":
        return not (40 <= step <= 45)
    return True


SPECS = {
    "01_constant_disturbance": ReplaySpec(
        "01_constant_disturbance", (-3.0, 0.0)),
    "02_target_jump": ReplaySpec(
        "02_target_jump", (0.0, 0.0)),
    "03_planner_failure": ReplaySpec(
        "03_planner_failure", (2.0, 0.0), fail_steps=(30, 31, 60)),
    "04_delay_asymmetric_rate_observation_loss": ReplaySpec(
        "04_delay_asymmetric_rate_observation_loss", (-2.0, 0.0),
        delay_steps=3, du_up_max=0.8, du_down_max=2.5),
    "05_runtime_capability_change": ReplaySpec(
        "05_runtime_capability_change", (-3.0, 0.0),
        n_steps=160,
        delay_steps=2, du_up_max=0.9, du_down_max=1.2,
        capability_changes=(
            (40, -2.0, 1.2, 0.25, 0.35, 4, False),
            (55, -1.2, 1.2, 0.0, 0.35, 3, False),
            (60, -2.0, 2.0, 0.9, 1.2, 2, False),
        ),
    ),
}


def _capability_change_fn(spec: ReplaySpec):
    schedule = {
        item[0]: Capability(
            item[1], item[2], item[3], item[4], item[5], item[6]
        )
        for item in spec.capability_changes
    }

    def change(step: int):
        return schedule.get(step)

    return change if schedule else None


def replay(spec_name: str, out_dir: str | Path | None = None) -> Path:
    spec = SPECS[spec_name]
    cfg = CartConfig(
        u_max=spec.u_max,
        du_up_max=spec.du_up_max,
        du_down_max=spec.du_down_max,
    )
    plant = CartPlant(cfg)
    if spec.fail_steps:
        controller = ForcedFailureController(
            plant, MPCConfig(), fail_steps=set(spec.fail_steps)
        )
    else:
        controller = MPCController(plant, MPCConfig())
    destination = Path(out_dir) if out_dir is not None else RESULTS / spec.name
    run_experiment(
        plant, controller, spec.n_steps,
        np.asarray(spec.initial_state, dtype=float),
        target_fn=lambda s: target(spec.name, s),
        disturbance_fn=lambda s, o: disturbance(spec.name, s, o),
        observation_available_fn=lambda s: observation_available(spec.name, s),
        actuator_delay_steps=spec.delay_steps,
        capability_change_fn=_capability_change_fn(spec),
        out_dir=destination,
    )
    with open(destination / "replay.json", "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
    return destination


def save_manifest(spec_name: str, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "replay.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(SPECS[spec_name].to_dict(), f, ensure_ascii=False, indent=2)
    return path


@dataclass(frozen=True)
class DualReplaySpec:
    """双执行器场景：两台能力/延迟/允许方向不同，含能力下降与恢复。"""

    name: str
    initial_state: tuple[float, float]
    n_steps: int
    delay_a: int
    delay_b: int
    u_max: float
    du_up_max: float
    du_down_max: float
    recovery_ramp_steps: int
    capability_changes: tuple[tuple[int, str, float, float, float, float, int], ...]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": "dual",
            "initial_state": list(self.initial_state),
            "n_steps": self.n_steps,
            "actuator_delay_a": self.delay_a,
            "actuator_delay_b": self.delay_b,
            "cart": {
                "u_max": self.u_max,
                "du_up_max": self.du_up_max,
                "du_down_max": self.du_down_max,
            },
            "recovery_ramp_steps": self.recovery_ramp_steps,
            "capability_changes": [
                {
                    "step": item[0],
                    "actuator": item[1],
                    "u_min": item[2],
                    "u_max": item[3],
                    "du_up": item[4],
                    "du_down": item[5],
                    "delay_steps": item[6],
                }
                for item in self.capability_changes
            ],
        }


DUAL_SPECS = {
    "06_dual_actuator_coordination": DualReplaySpec(
        "06_dual_actuator_coordination",
        initial_state=(-3.0, 0.0),
        n_steps=140,
        delay_a=1,
        delay_b=3,
        u_max=2.0,
        du_up_max=1.0,
        du_down_max=1.0,
        recovery_ramp_steps=6,
        capability_changes=(
            # 两台分工：a 擅长正向快速调节，b 擅长负向、延迟更大
            (0, "a", -0.8, 2.0, 0.6, 0.9, 1),
            (0, "b", -2.0, 0.8, 0.9, 0.5, 3),
            # a 能力下降且延迟增长，同时观测短暂缺失
            (50, "a", -0.8, 0.5, 0.15, 0.3, 2),
            # a 能力恢复：逐步重新分配，避免两台同时补偿同一误差
            (85, "a", -0.8, 2.0, 0.6, 0.9, 1),
        ),
    ),
}


def dual_target(name: str, step: int) -> float:
    if name == "06_dual_actuator_coordination":
        return 2.0 if step >= 25 else 0.0
    raise KeyError(f"未知双执行器场景: {name}")


def dual_disturbance(name: str, step: int, obs: np.ndarray) -> float:
    if name == "06_dual_actuator_coordination":
        return 0.2 if step >= 40 else 0.0
    return 0.0


def dual_observation_available(name: str, step: int) -> bool:
    if name == "06_dual_actuator_coordination":
        return not (55 <= step <= 60)
    return True


def _dual_capability_change_fn(spec: DualReplaySpec):
    schedule: dict[int, dict] = {}
    for item in spec.capability_changes:
        step, name = item[0], item[1]
        schedule.setdefault(step, {})[name] = Capability(
            item[2], item[3], item[4], item[5], item[6]
        )

    def change(step: int):
        return schedule.get(step)

    return change if schedule else None


def run_dual(spec_name: str, out_dir: str | Path):
    """按内置双执行器场景定义运行，返回 (logger, 输出目录)。"""
    spec = DUAL_SPECS[spec_name]
    cfg = CartConfig(
        u_max=spec.u_max,
        du_up_max=spec.du_up_max,
        du_down_max=spec.du_down_max,
    )
    plant = CartPlant(cfg)
    controller = DualMPCController(
        plant, MPCConfig(), recovery_ramp_steps=spec.recovery_ramp_steps
    )
    destination = Path(out_dir)
    logger = run_dual_experiment(
        plant, controller, spec.n_steps,
        np.asarray(spec.initial_state, dtype=float),
        target_fn=lambda s: dual_target(spec.name, s),
        disturbance_fn=lambda s, o: dual_disturbance(spec.name, s, o),
        observation_available_fn=lambda s: dual_observation_available(
            spec.name, s),
        capability_change_fn=_dual_capability_change_fn(spec),
        actuator_delay_a=spec.delay_a,
        actuator_delay_b=spec.delay_b,
        out_dir=destination,
    )
    return logger, destination


def replay_dual(spec_name: str, out_dir: str | Path | None = None) -> Path:
    destination = Path(out_dir) if out_dir is not None else RESULTS / spec_name
    run_dual(spec_name, destination)
    with open(destination / "replay.json", "w", encoding="utf-8") as f:
        json.dump(DUAL_SPECS[spec_name].to_dict(), f,
                  ensure_ascii=False, indent=2)
    return destination


def save_dual_manifest(spec_name: str, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "replay.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DUAL_SPECS[spec_name].to_dict(), f,
                  ensure_ascii=False, indent=2)
    return path
