from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .interfaces import Observation
from .model import CartPlant
from .mpc import Plan, RecedingHorizonMPC

DisturbanceFunction = Callable[[int, float, Observation], float]
TargetFunction = Callable[[int, float, Observation], float]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    steps: int
    initial_position: float = 0.0
    initial_velocity: float = 0.0
    initial_target: float = 1.0
    disturbance: DisturbanceFunction = lambda step, time, observation: 0.0
    target: TargetFunction = lambda step, time, observation: observation.target_position


def run_scenario(
    scenario: Scenario,
    controller: RecedingHorizonMPC,
    plant: CartPlant,
) -> list[dict]:
    controller.reset()
    records: list[dict] = []

    for step in range(scenario.steps):
        observation = plant.observe()
        target_before_plan = scenario.target(step, plant.time, observation)
        target_changed = target_before_plan != plant.target_position
        plant.target_position = float(target_before_plan)
        observation = plant.observe()

        action, plan = controller.step(observation)
        disturbance = float(
            scenario.disturbance(step, plant.time, observation)
        )
        before_position = plant.position
        before_velocity = plant.velocity
        next_observation = plant.step(action, disturbance)

        if plan.success:
            predicted_position = float(plan.predicted_positions[1])
            predicted_velocity = float(plan.predicted_velocities[1])
        else:
            predicted_position = None
            predicted_velocity = None

        record = {
            "scenario": scenario.name,
            "description": scenario.description,
            "step": step,
            "time": observation.time,
            "target": observation.target_position,
            "target_changed": target_changed,
            "disturbance": disturbance,
            "position_before": before_position,
            "velocity_before": before_velocity,
            "planned_action": float(plan.actions[0]) if len(plan.actions) else None,
            "executed_action": action.acceleration,
            "plan_success": plan.success,
            "plan_message": plan.message,
            "predicted_next_position": predicted_position,
            "predicted_next_velocity": predicted_velocity,
            "actual_next_position": next_observation.position,
            "actual_next_velocity": next_observation.velocity,
            "position_prediction_error": (
                next_observation.position - predicted_position
                if predicted_position is not None
                else None
            ),
            "velocity_prediction_error": (
                next_observation.velocity - predicted_velocity
                if predicted_velocity is not None
                else None
            ),
            "plan_actions": np.asarray(plan.actions, dtype=float).tolist(),
            "plan_positions": np.asarray(plan.predicted_positions, dtype=float).tolist(),
            "plan_velocities": np.asarray(plan.predicted_velocities, dtype=float).tolist(),
        }
        records.append(record)

    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_summary(records_by_scenario: dict[str, list[dict]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "steps",
        "plan_failures",
        "target_changes",
        "max_abs_action",
        "max_abs_slew",
        "max_abs_disturbance",
        "max_abs_position_error",
        "max_abs_velocity",
        "max_abs_position_prediction_error",
        "max_abs_velocity_prediction_error",
        "final_position",
        "final_target",
        "final_position_error",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_name, records in records_by_scenario.items():
            actions = np.array([r["executed_action"] for r in records])
            slew = np.diff(np.r_[0.0, actions])
            positions = np.array([r["actual_next_position"] for r in records])
            velocities = np.array([r["actual_next_velocity"] for r in records])
            targets = np.array([r["target"] for r in records])
            disturbances = np.array([r["disturbance"] for r in records])
            position_errors = positions - targets
            predicted_position_errors = [
                r["position_prediction_error"]
                for r in records
                if r["position_prediction_error"] is not None
            ]
            predicted_velocity_errors = [
                r["velocity_prediction_error"]
                for r in records
                if r["velocity_prediction_error"] is not None
            ]
            writer.writerow(
                {
                    "scenario": scenario_name,
                    "steps": len(records),
                    "plan_failures": sum(not r["plan_success"] for r in records),
                    "target_changes": sum(r["target_changed"] for r in records),
                    "max_abs_action": float(np.max(np.abs(actions))),
                    "max_abs_slew": float(np.max(np.abs(slew))),
                    "max_abs_disturbance": float(np.max(np.abs(disturbances))),
                    "max_abs_position_error": float(np.max(np.abs(position_errors))),
                    "max_abs_velocity": float(np.max(np.abs(velocities))),
                    "max_abs_position_prediction_error": float(
                        np.max(np.abs(predicted_position_errors))
                        if predicted_position_errors
                        else 0.0
                    ),
                    "max_abs_velocity_prediction_error": float(
                        np.max(np.abs(predicted_velocity_errors))
                        if predicted_velocity_errors
                        else 0.0
                    ),
                    "final_position": positions[-1],
                    "final_target": targets[-1],
                    "final_position_error": position_errors[-1],
                }
            )
