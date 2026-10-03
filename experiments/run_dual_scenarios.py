"""双执行器协同场景：命令回执不确定时的计划、补偿与核实。

运行：.venv/bin/python -m experiments.run_dual_scenarios

场景 06 覆盖：两台能力/延迟/允许方向不同；一条正力命令的执行回执
丢失（目标仍在变化），控制器不得立即当作未执行重发，也不得把另一台
的作用算到它头上；同一回执重复到达、旧回执晚于新命令到达只按原 ID
归档；一条命令只收到部分执行量；执行器断连后重连，断连期间命令
保留原归属并收到明确未收到回执。等待方案与补偿方案的可达区间、
实际抵消量、剩余误差和不能继续的原因同时输出。

场景 07 覆盖：运行中某台能力降低（含变为单向），尚未发生的动作按
新能力重新核对；人工确认某条命令未发生后才重新安排作用，确认发生
后避免重复补偿；保存不确定中途状态后恢复，原命令与核实决定逐项
对应。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from control_lab.actuator_group import ActuatorGroup
from control_lab.capability import Capability
from control_lab.command_ledger import (
    ACK,
    EXEC_FULL,
    EXEC_PARTIAL,
    Receipt,
)
from control_lab.config import CartConfig, MPCConfig
from control_lab.device_channel import ReceiptBus
from control_lab.dual_controller import COMPENSATE, WAIT, DualMPCController
from control_lab.dual_experiment import run_dual_experiment
from control_lab.plants import CartPlant

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def build_components(policy: str = COMPENSATE,
                     bus: ReceiptBus | None = None,
                     channel_options: dict | None = None,
                     caps: dict[str, Capability] | None = None):
    plant = CartPlant(CartConfig())
    if caps is None:
        caps = {
            # a：双向、小幅、变化快、延迟 1 拍。
            "a": Capability(-2.0, 2.0, 1.0, 1.0, delay_steps=1),
            # b：只能正向、幅度小、变化慢、延迟 2 拍。
            "b": Capability(0.0, 1.2, 0.4, 0.4, delay_steps=2),
        }
    controller = DualMPCController(
        plant, MPCConfig(horizon=15), capabilities=caps, policy=policy
    )
    group = ActuatorGroup(
        plant, capabilities=caps, bus=bus, channel_options=channel_options
    )
    return plant, controller, group, caps


def scenario_06(out_dir: Path) -> dict:
    # b 的执行回执在指定拍永久丢失；ack 保留，让状态停留在 acked。
    def loss(step, receipt):
        if receipt.actuator == "b" and receipt.kind == EXEC_FULL \
                and receipt.detail == "" and step == 6:
            return True
        return False

    # b 的第 10 拍命令只执行 40%。
    def execution(step, command):
        if step == 12 and command.submitted_step == 10:
            return 0.4
        return 1.0

    # 一条旧命令（b, step=5 提交、due=7）的执行回执延迟 5 拍，
    # 晚于多条新命令到达，仍必须归属原命令。
    def delay(step, receipt):
        if receipt.actuator == "b" and receipt.kind == EXEC_FULL \
                and step == 7:
            return 5
        return 0

    # 第 8 拍一条 a 的执行回执重复到达一次。
    def duplicate(step, receipt):
        if receipt.actuator == "a" and receipt.kind == EXEC_FULL \
                and step == 8:
            return 1
        return 0

    bus = ReceiptBus(loss_fn=loss, delay_fn=delay, duplicate_fn=duplicate)
    plant, controller, group, caps = build_components(
        policy=COMPENSATE, bus=bus,
        channel_options={"b": {"execution_fn": execution}},
    )

    # b 在 18 拍短暂断连一拍；19 拍重连后只补一条明确的未收到回执，
    # 不自动重发该命令。
    group.channels["b"].connected_fn = lambda step: step != 18

    def reconnect(step):
        return ("b",) if step == 19 else ()

    def inject(step, group: ActuatorGroup):
        # 部分执行命令（due=12）的剩余部分在 15 拍才执行完毕，
        # 迟到的完成回执仍归属原命令；在此之前保留 uncertain 区间。
        if step == 15:
            for record in group.ledgers["b"].records():
                if record.submitted_step == 10 and record.status == "uncertain":
                    return [group.deliver_receipt(Receipt(
                        command_id=record.command_id, actuator="b",
                        kind=EXEC_FULL, arrived_step=step,
                        applied_value=record.action,
                        detail="remaining_completed_late",
                    ))]
        return []

    log = run_dual_experiment(
        plant, controller, n_steps=90,
        initial_state=np.array([-2.0, 0.0]),
        target_fn=lambda s: 1.5 if s >= 5 else 0.0,
        disturbance_fn=lambda s, o: 0.1,
        group=group, reconnect_fn=reconnect,
        receipt_inject_fn=inject,
        out_dir=out_dir,
    )
    return _summarize("06_dual_uncertain_receipt", log, caps)


def scenario_07(out_dir: Path) -> dict:
    plant, controller, group, caps = build_components(policy=WAIT)

    def capability_change(step):
        if step == 18:
            return {"a": Capability(-0.8, 0.8, 0.25, 0.25, delay_steps=2)}
        if step == 40:
            # a 变成只能减小方向。
            return {"a": Capability(-2.0, 0.0, 0.0, 0.6, delay_steps=2)}
        if step == 55:
            return {"a": Capability(-2.0, 2.0, 1.0, 1.0, delay_steps=1)}
        return None

    # 让 a 的若干条执行回执永久丢失（含能力收缩前后各一条），
    # 命令停留在 in_flight/acked，等待人工核实。
    def loss(step, receipt):
        if receipt.actuator != "a" or receipt.kind != EXEC_FULL:
            return False
        return step in (19, 20, 21)

    group.bus.loss_fn = loss
    injected_notes: list[str] = []

    def inject(step, group: ActuatorGroup):
        receipts = []
        # 22 拍：人工确认 a 的一条在能力收缩边缘提交的命令确实发生，
        # 之后即使有迟到的相反回执也不能重复补偿。
        if step == 22:
            candidates = [
                r for r in group.ledgers["a"].records()
                if r.due_step in (19, 20) and not r.manually_decided
                and r.status in ("in_flight", "acked", "uncertain")
            ]
            if candidates:
                receipts.append(group.manual_confirm(
                    step, "a", candidates[0].command_id, occurred=True,
                    applied_value=float(candidates[0].action),
                ))
                injected_notes.append(f"manual_executed:{candidates[0].command_id}")
        # 30 拍：人工确认 a 的某条命令未发生 -> 才允许重新安排作用。
        if step == 30:
            candidates = [
                r for r in group.ledgers["a"].records()
                if not r.confirmed and r.due_step <= step and r.status in (
                    "in_flight", "acked", "uncertain"
                )
            ]
            if candidates:
                original = candidates[0]
                receipts.append(group.manual_confirm(
                    step, "a", original.command_id, occurred=False,
                ))
                # 替代作用必须在能力收缩后的新限制内。
                cap = group.capability("a")
                replacement = max(
                    cap.u_min,
                    min(cap.u_max, original.hold_value + cap.du_up),
                )
                new_id = group.reschedule("a", original.command_id,
                                          replacement, step)
                injected_notes.append(
                    f"manual_not_applied:{original.command_id}->reschedule:{new_id}"
                )
        return receipts

    log = run_dual_experiment(
        plant, controller, n_steps=120,
        initial_state=np.array([-1.5, 0.0]),
        target_fn=lambda s: 1.0,
        disturbance_fn=lambda s, o: 0.05 * np.sin(0.3 * s),
        capability_change_fn=capability_change,
        receipt_inject_fn=inject,
        group=group, out_dir=out_dir,
    )
    summary = _summarize("07_dual_capability_manual", log, caps)
    summary["manual_events"] = injected_notes
    return summary


def _summarize(name: str, log, caps: dict[str, Capability]) -> dict:
    df = log.to_frame()
    summary = {
        "name": name,
        "n_steps": int(len(df)),
        "n_fallback": int(df["fallback"].sum()),
        "final_position_error": float(abs(
            df["x_next"].iloc[-1] - df["target"].iloc[-1]
        )),
        "max_applied_a": float(df["applied_a"].abs().max()),
        "max_applied_b": float(df["applied_b"].abs().max()),
        "max_cancellation_first_step": float(
            df["cancellation_first_step"].max()
        ),
        "max_residual_error_terminal": float(
            df["residual_error_terminal"].max()
        ),
        "n_uncertain_reports": int(
            (df.filter(regex="_(in_flight|acked|uncertain)$").sum(axis=1) > 0)
            .sum()
        ),
        "n_receipts_arrived": int(df["n_receipts_arrived"].sum()),
        "n_duplicate_receipts": int(df["n_duplicate_receipts"].sum()),
        "b_min_violation": float((-df["applied_b"]).max()),
        "a_bound_violation": float(max(
            0.0,
            (df["applied_a"] - df["target"].map(lambda _: 2.0)).max(),
        )),
    }
    return summary


def plot_dual(log, title: str, path: Path) -> None:
    df = log.to_frame()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(df["time"], df["x_next"], label="position", lw=2)
    axes[0].plot(df["time"], df["target"], "k--", label="target")
    axes[0].set_ylabel("position")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(title)

    axes[1].plot(df["time"], df["applied_a"], label="applied a")
    axes[1].plot(df["time"], df["applied_b"], label="applied b")
    axes[1].plot(df["time"], df["applied_total"], "k:", label="total")
    axes[1].set_ylabel("force")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    uncertain = (
        df.get("a_in_flight", 0) + df.get("a_acked", 0)
        + df.get("a_uncertain", 0) + df.get("b_in_flight", 0)
        + df.get("b_acked", 0) + df.get("b_uncertain", 0)
    )
    axes[2].plot(df["time"], uncertain, color="tab:red",
                 label="unconfirmed commands (both actuators)")
    axes[2].plot(df["time"], df["cancellation_first_step"],
                 color="tab:purple", label="confirmed cancellation")
    axes[2].set_ylabel("count / force")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    summaries = {}

    d06 = OUT / "06_dual_uncertain_receipt"
    summaries["06_dual_uncertain_receipt"] = scenario_06(d06)
    plot_dual(
        _load_logger_like(d06),
        "Scenario 06: dual actuators with uncertain/lost/partial receipts",
        d06 / "plot.png",
    )

    d07 = OUT / "07_dual_capability_manual"
    summaries["07_dual_capability_manual"] = scenario_07(d07)
    plot_dual(
        _load_logger_like(d07),
        "Scenario 07: capability reduction + manual confirmation + reschedule",
        d07 / "plot.png",
    )

    with open(OUT / "dual_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def _load_logger_like(directory: Path):
    class _Log:
        def __init__(self, directory: Path):
            self.to_frame_value = pd.read_csv(directory / "steps.csv")
            with open(directory / "plans.json", encoding="utf-8") as handle:
                self.plans = json.load(handle)

        def to_frame(self):
            return self.to_frame_value

    return _Log(directory)


if __name__ == "__main__":
    main()
