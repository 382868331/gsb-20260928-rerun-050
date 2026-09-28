from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cart_control import (
    ActionBounds,
    CartPlant,
    MPCParameters,
    RecedingHorizonMPC,
)
from cart_control.experiment import run_scenario, write_jsonl, write_summary
from cart_control.scenarios import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(description="运行小车滚动时域控制实验")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=25)
    args = parser.parse_args()

    bounds = ActionBounds(
        min_acceleration=-2.0,
        max_acceleration=2.0,
        max_slew=1.5,
    )
    mpc_parameters = MPCParameters(horizon=args.horizon)

    records_by_scenario: dict[str, list[dict]] = {}
    for scenario in SCENARIOS:
        plant = CartPlant(
            dt=args.dt,
            initial_position=scenario.initial_position,
            initial_velocity=scenario.initial_velocity,
            target_position=scenario.initial_target,
        )
        controller = RecedingHorizonMPC(
            dt=args.dt,
            action_bounds=bounds,
            params=mpc_parameters,
        )
        records = run_scenario(scenario, controller, plant)
        records_by_scenario[scenario.name] = records
        write_jsonl(records, args.output_dir / f"{scenario.name}.jsonl")

    write_summary(records_by_scenario, args.output_dir / "summary.csv")
    print(f"已写入 {len(records_by_scenario)} 个场景到 {args.output_dir}")


if __name__ == "__main__":
    main()
