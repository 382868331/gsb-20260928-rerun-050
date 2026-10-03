"""双执行器不确定回执场景实验。

运行：.venv/bin/python -m experiments.run_dual_scenarios

场景：
- dual_01_*：回执丢失时保守等待 vs 剩余能力内继续补偿的对照。
- dual_02_*：重复回执、部分执行与执行器重连补报，归属保持不变。
- dual_03_*：能力收缩后重新核对未发生动作，人工确认未发生后重新安排。
- dual_04_*：两台相反作用、不同延迟下的抵消量、剩余误差与不能继续原因。
- dual_05_*：含不确定回执的中途保存与恢复，恢复结果与连续运行一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from control_lab.capability import Capability
from control_lab.config import CartConfig, MPCConfig
from control_lab.dual_controller import DualMPCController
from control_lab.dual_experiment import run_dual_experiment
from control_lab.plants import CartPlant

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

CAPS = {
    "a": Capability(-2.0, 2.0, 1.0, 1.0, 2),
    "b": Capability(-1.0, 1.0, 0.6, 0.6, 0),
}


def _make():
    plant = CartPlant(CartConfig())
    ctrl = DualMPCController(plant, MPCConfig(), capabilities=CAPS)
    return plant, ctrl


def _honest_with_drop(drop_steps):
    def receipt_fn(step, events):
        if step in drop_steps:
            return []
        return [{**e, "step": step, "applied": True,
                 "applied_action": [e["applied_action"]]} for e in events]
    return receipt_fn


def _summarize(name, log):
    df = log.to_frame()
    return {
        "scenario": name,
        "steps": int(len(df)),
        "final_residual_error": float(df["residual_error"].iloc[-1]),
        "max_abs_residual": float(df["residual_error"].abs().max()),
        "total_cancellation": float(df["cancellation"].sum()),
        "steps_holding": int(df["holding"].sum()),
        "max_uncertain": int(df["n_uncertain"].max()),
        "confirmed_not_applied": int(df["n_confirmed_not_applied"].max()),
        "unmet_reasons": sorted({
            r for r in df["unmet_reason"].unique() if isinstance(r, str) and r
        }),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    summaries = {}

    # dual_01：回执丢失（5-9 拍），保守等待 vs 剩余能力内补偿；
    # 第 15 拍人工核实丢失期间的命令均已发生，等待策略随后恢复跟踪
    def confirm_applied_15(step, ledger):
        if step == 15:
            return [{"actuator": r.actuator, "command_id": r.command_id,
                     "applied": True}
                    for r in ledger.records() if r.status == "uncertain"]
        return []

    for strategy in ("wait", "compensate"):
        plant, ctrl = _make()
        log = run_dual_experiment(
            plant, ctrl, capabilities=CAPS, n_steps=60,
            initial_state=np.array([0.0, 0.0]),
            target_fn=lambda s: 1.5, strategy=strategy,
            receipt_fn=_honest_with_drop({5, 6, 7, 8, 9}),
            manual_confirmation_fn=confirm_applied_15,
            out_dir=OUT / f"dual_01_lost_receipt_{strategy}",
        )
        summaries[f"dual_01_lost_receipt_{strategy}"] = _summarize(
            f"dual_01_{strategy}", log)

    # dual_02：重复回执 + 部分执行 + 重连补报
    def receipt_fn_02(step, events):
        receipts = []
        for e in events:
            base = {**e, "step": step}
            if step == 6:
                # 只收到部分执行量，且同一回执重复到达
                partial = {**base, "applied": True,
                           "applied_action": [0.5 * e["applied_action"]]}
                receipts.extend([partial, dict(partial)])
            elif 7 <= step <= 10:
                continue  # 连接中断：无回执
            else:
                receipts.append({**base, "applied": True,
                                 "applied_action": [e["applied_action"]]})
        if step == 11:
            # 重连：补报中断期间各命令的真实结果
            receipts.append({"kind": "resync_batch", "reports": [
                {"actuator": "a", "command_id": cid, "applied": True}
                for cid in []  # 由下方 manual 钩子按台账补报
            ]})
        return receipts

    def resync_fn(step, ledger):
        # 重连后按台账逐项补报（设备报告全部已作用）
        if step == 11:
            return [{"actuator": r.actuator, "command_id": r.command_id,
                     "applied": True}
                    for r in ledger.records() if r.status == "uncertain"]
        return []

    plant, ctrl = _make()
    log2 = run_dual_experiment(
        plant, ctrl, capabilities=CAPS, n_steps=60,
        initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 1.0, strategy="wait",
        receipt_fn=receipt_fn_02,
        manual_confirmation_fn=lambda s, l: [
            {**d, "applied": d.pop("applied")} for d in (
                [{"actuator": d["actuator"], "command_id": d["command_id"],
                  "applied": d["applied"]} for d in resync_fn(s, l)]
            )],
        out_dir=OUT / "dual_02_duplicate_partial_reconnect",
    )
    summaries["dual_02_duplicate_partial_reconnect"] = _summarize(
        "dual_02", log2)

    # dual_03：能力收缩 + 人工确认未发生后重新安排
    def capability_change_03(step):
        if step == 20:
            return {"a": Capability(-1.0, 1.0, 0.3, 0.3, 3)}
        return None

    def confirm_fn_03(step, ledger):
        if step == 14:
            return [{"actuator": r.actuator, "command_id": r.command_id,
                     "applied": False}
                    for r in ledger.records() if r.status == "uncertain"]
        return []

    plant, ctrl = _make()
    log3 = run_dual_experiment(
        plant, ctrl, capabilities=CAPS, n_steps=80,
        initial_state=np.array([-1.0, 0.0]),
        target_fn=lambda s: 2.0, strategy="wait",
        receipt_fn=_honest_with_drop({10, 11, 12, 13}),
        capability_change_fn=capability_change_03,
        manual_confirmation_fn=confirm_fn_03,
        out_dir=OUT / "dual_03_capability_reduction_manual_confirm",
    )
    summaries["dual_03_capability_reduction_manual_confirm"] = _summarize(
        "dual_03", log3)

    # dual_04：相反作用 + 不同延迟，展示抵消量与剩余误差
    caps_opp = {
        "a": Capability(-2.0, 2.0, 1.5, 1.5, 3),
        "b": Capability(-2.0, 2.0, 1.5, 1.5, 0),
    }
    plant = CartPlant(CartConfig())
    ctrl = DualMPCController(plant, MPCConfig(), capabilities=caps_opp)
    log4 = run_dual_experiment(
        plant, ctrl, capabilities=caps_opp, n_steps=80,
        initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 2.0 if s < 40 else -2.0,
        strategy="compensate",
        out_dir=OUT / "dual_04_opposing_delays_cancellation",
    )
    summaries["dual_04_opposing_delays_cancellation"] = _summarize(
        "dual_04", log4)

    # dual_05：含不确定回执的中途保存与恢复
    snapshot_path = OUT / "dual_05_snapshot_resume" / "snapshot.json"
    plant, ctrl = _make()
    continuous = run_dual_experiment(
        plant, ctrl, capabilities=CAPS, n_steps=50,
        initial_state=np.array([0.5, 0.0]),
        target_fn=lambda s: 1.0, strategy="compensate",
        receipt_fn=_honest_with_drop({18, 19, 20, 21, 22}),
        snapshot_step=25, snapshot_path=snapshot_path,
        out_dir=OUT / "dual_05_snapshot_resume" / "continuous",
    )
    state = json.loads(snapshot_path.read_text(encoding="utf-8"))
    plant, ctrl = _make()
    resumed = run_dual_experiment(
        plant, ctrl, capabilities=CAPS, n_steps=24,
        initial_state=np.zeros(2),
        target_fn=lambda s: 1.0, strategy="compensate",
        receipt_fn=_honest_with_drop({18, 19, 20, 21, 22}),
        resume_state=state,
        out_dir=OUT / "dual_05_snapshot_resume" / "resumed",
    )
    cont = continuous.to_frame().set_index("step")
    res = resumed.to_frame().set_index("step")
    max_dev = max(abs(cont.loc[s, "x"] - res.loc[s, "x"]) for s in res.index)
    summaries["dual_05_snapshot_resume"] = {
        "scenario": "dual_05",
        "snapshot_step": 25,
        "resumed_steps": int(len(res)),
        "uncertain_at_snapshot": int(sum(
            1 for r in state["ledger"]["records"]
            if r["status"] == "uncertain")),
        "decisions_at_snapshot": int(len(state["ledger"]["decisions"])),
        "max_position_deviation_after_resume": float(max_dev),
    }

    with open(OUT / "dual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

