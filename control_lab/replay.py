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
from .experiment import run_experiment
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
    raise KeyError(f"未知场景: {name}")


def disturbance(name: str, step: int, obs: np.ndarray) -> float:
    if name == "01_constant_disturbance":
        return 0.9 if step >= 40 else 0.0
    if name == "02_target_jump":
        return 0.15 * np.sin(0.5 * step)
    if name == "04_delay_asymmetric_rate_observation_loss":
        return 0.6 + 0.1 * np.sin(0.2 * step)
    return 0.0


def observation_available(name: str, step: int) -> bool:
    if name == "04_delay_asymmetric_rate_observation_loss":
        return not (45 <= step <= 52)
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
}


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
