"""双执行器命令回执不确定时的协同控制测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from control_lab.actuator_group import ActuatorGroup
from control_lab.capability import Capability
from control_lab.command_ledger import (
    ACK,
    ACKED,
    EXEC_FULL,
    EXEC_PARTIAL,
    EXECUTED,
    IN_FLIGHT,
    MEASURE_CUMULATIVE,
    MEASURE_INCREMENTAL,
    NOT_APPLIED,
    QUEUED,
    UNCERTAIN,
    CommandLedger,
    Receipt,
)
from control_lab.config import CartConfig, MPCConfig
from control_lab.device_channel import ReceiptBus
from control_lab.dual_controller import COMPENSATE, WAIT, DualMPCController
from control_lab.dual_experiment import run_dual_experiment
from control_lab.plants import CartPlant


def make_caps():
    return {
        "a": Capability(-2.0, 2.0, 1.0, 1.0, delay_steps=1),
        "b": Capability(0.0, 1.2, 0.4, 0.4, delay_steps=2),
    }


def make_components(policy=COMPENSATE, bus=None, channel_options=None):
    plant = CartPlant(CartConfig())
    caps = make_caps()
    controller = DualMPCController(
        plant, MPCConfig(horizon=12), capabilities=caps, policy=policy
    )
    group = ActuatorGroup(
        plant, capabilities=caps, bus=bus, channel_options=channel_options
    )
    return plant, controller, group, caps


def run_n(n, group, controller, plant, **kwargs):
    defaults = dict(
        initial_state=np.array([-1.0, 0.0]),
        target_fn=lambda s: 1.0,
        disturbance_fn=lambda s, o: 0.05,
        group=group,
    )
    defaults.update(kwargs)
    return run_dual_experiment(plant, controller, n, **defaults)


def test_duplicate_and_late_receipt_keep_original_command_identity():
    def delay(step, receipt):
        return 4 if receipt.actuator == "b" and receipt.kind == EXEC_FULL else 0

    def duplicate(step, receipt):
        return 1 if receipt.actuator == "a" and receipt.kind == EXEC_FULL else 0

    bus = ReceiptBus(delay_fn=delay, duplicate_fn=duplicate)
    plant, controller, group, _ = make_components(bus=bus)
    run_n(12, group, controller, plant)

    ledger_a = group.ledgers["a"]
    a_receipts = ledger_a.receipts()
    assert sum(1 for r in a_receipts if r.duplicate) >= 1
    executed_a = ledger_a.confirmed_occurred()
    assert executed_a and all(r.status == EXECUTED for r in executed_a)
    # 重复回执不会重复计数：每条命令只留下一条非重复事实。
    ids = [r.command_id for r in a_receipts if r.kind == EXEC_FULL
           and not r.duplicate]
    assert len(ids) == len(set(ids))

    # b 的迟到回执仍归属原命令，且晚到后把该命令从不确定变为已发生。
    ledger_b = group.ledgers["b"]
    for receipt in ledger_b.receipts():
        assert ledger_b.has(receipt.command_id)
    assert ledger_b.confirmed_occurred()


def test_lost_receipt_keeps_command_unconfirmed_without_resend():
    # a 在第 3 拍生效的执行回执永久丢失。
    seen = {"n": 0}

    def loss(step, receipt):
        if receipt.actuator == "a" and receipt.kind == EXEC_FULL:
            seen["n"] += 1
            return seen["n"] == 2
        return False

    bus = ReceiptBus(loss_fn=loss)
    plant, controller, group, _ = make_components(policy=COMPENSATE, bus=bus)
    run_n(10, group, controller, plant)

    ledger = group.ledgers["a"]
    unconfirmed = ledger.unconfirmed()
    assert len(unconfirmed) == 1
    lost = unconfirmed[0]
    assert lost.status in (IN_FLIGHT, ACKED, UNCERTAIN)
    # 设备实际已经作用（真实侧），只是控制器侧没有回执。
    assert lost.command_id in group.channels["a"].ground_truth_applied
    # 之后又提交了新命令：它们是独立的新 ID，没有复用未确认命令的 ID。
    later = [r for r in ledger.records(QUEUED) + ledger.confirmed_occurred()
             if r.submitted_step > lost.submitted_step]
    assert later and all(r.command_id != lost.command_id for r in later)


def test_partial_execution_is_uncertain_interval_not_point_estimate():
    def execution(step, command):
        return 0.4 if step == 3 else 1.0

    bus = ReceiptBus()
    plant, controller, group, _ = make_components(
        bus=bus, channel_options={"b": {"execution_fn": execution}}
    )
    run_n(5, group, controller, plant)

    partial = [r for r in group.ledgers["b"].records()
               if r.status == UNCERTAIN]
    assert len(partial) == 1
    record = partial[0]
    assert record.confirmed_value is not None
    assert record.remaining_value is not None
    assert abs(record.confirmed_value - record.action) > 1e-9
    lo, hi = group.ledgers["b"].possible_effect_interval(record)
    assert lo < hi  # 区间，而不是一个精确预测值

    plan = controller.plan(
        plant.get_observation(), 1.0, group.ledgers, current_step=5
    )
    assert plan.uncertain_commands["b"] == [record.command_id]
    proj = plan.channels["b"]
    assert 0 in proj["uncertain_steps"]
    assert proj["possible_min"][0] < proj["possible_max"][0]


def test_manual_confirmation_gates_reschedule():
    seen = {"n": 0}

    def loss(step, receipt):
        if receipt.actuator == "a" and receipt.kind == EXEC_FULL:
            seen["n"] += 1
            return seen["n"] == 1
        return False

    bus = ReceiptBus(loss_fn=loss)
    plant, controller, group, _ = make_components(policy=WAIT, bus=bus)
    run_n(6, group, controller, plant)
    lost = group.ledgers["a"].unconfirmed()[0]
    with pytest.raises(ValueError):
        group.reschedule("a", lost.command_id, 0.5, 6)

    group.manual_confirm(6, "a", lost.command_id, occurred=False)
    assert group.ledgers["a"].get(lost.command_id).status == NOT_APPLIED
    new_id = group.reschedule("a", lost.command_id, 0.5, 6)
    assert group.ledgers["a"].get(lost.command_id).planned_reschedule_id == new_id

    # 人工确认发生后，迟到的冲突回执被标记 superseded，不重复补偿。
    group2_plant, group2_controller, group2, _ = make_components(bus=ReceiptBus())
    run_n(4, group2, group2_controller, group2_plant)
    # 人工核实一条本拍尚未到确认时刻的命令已发生。
    record = group2.ledgers["a"].records(QUEUED)[0]
    group2.manual_confirm(4, "a", record.command_id, occurred=True,
                          applied_value=record.action)
    late = group2.deliver_receipt(Receipt(
        command_id=record.command_id, actuator="a", kind=EXEC_FULL,
        arrived_step=8, applied_value=record.action,
    ))
    assert late.superseded
    assert sum(1 for r in group2.ledgers["a"].confirmed_occurred()
               if r.command_id == record.command_id) == 1


def test_wait_freezes_only_uncertain_channel():
    def loss(step, receipt):
        return receipt.actuator == "a" and receipt.kind == EXEC_FULL and step == 2

    bus = ReceiptBus(loss_fn=loss)
    plant, controller, group, _ = make_components(policy=WAIT, bus=bus)
    run_n(6, group, controller, plant)
    plan = controller.plan(plant.get_observation(), 1.0, group.ledgers,
                           current_step=6)
    assert plan.policy == WAIT
    assert "a" in plan.policies["wait"]["frozen_channels"]
    assert "b" not in plan.policies["wait"]["frozen_channels"]
    # 冻结通道提交保持命令；另一台在自身能力内继续调节（仍为正向）。
    assert plan.actions["a"] == pytest.approx(controller.last_confirmed["a"])
    assert plan.actions["b"] >= 0.0


def test_other_actuator_effect_not_attributed_to_unconfirmed_command():
    def loss(step, receipt):
        return receipt.actuator == "a" and receipt.kind == EXEC_FULL and step == 2

    bus = ReceiptBus(loss_fn=loss)
    plant, controller, group, _ = make_components(bus=bus)
    run_n(8, group, controller, plant)
    unconfirmed_a = group.ledgers["a"].unconfirmed()
    assert unconfirmed_a
    # b 的已发生作用不能消除 a 的未确认状态。
    assert group.ledgers["b"].confirmed_occurred()
    for record in unconfirmed_a:
        assert record.status != EXECUTED
    # 不确定命令在计划里给出的是可能作用区间。
    plan = controller.plan(plant.get_observation(), 1.0, group.ledgers, 8)
    assert plan.channels["a"]["possible_min"] != plan.channels["a"]["possible_max"] or \
        plan.uncertain_commands["a"]


def test_capability_reduction_rechecks_pending_but_keeps_facts():
    plant, controller, group, caps = make_components()
    run_n(4, group, controller, plant)
    occurred = group.ledgers["a"].confirmed_occurred()
    assert occurred
    confirmed_value = occurred[-1].confirmed_value
    new_cap = {"a": Capability(-0.8, 0.8, 0.25, 0.25, delay_steps=2)}
    group.update_capability(new_cap)
    controller.update_capabilities(new_cap, group.ledgers)
    # 已发生事实不回退、不被新限幅改写。
    assert group.ledgers["a"].confirmed_occurred()[-1].confirmed_value == \
        pytest.approx(confirmed_value)
    for record in group.ledgers["a"].records(QUEUED):
        assert -0.8 - 1e-9 <= record.action <= 0.8 + 1e-9


def test_save_resume_preserves_uncertain_commands_and_decisions(tmp_path):
    seen = {"n": 0}

    def loss(step, receipt):
        if receipt.actuator == "b" and receipt.kind == EXEC_FULL:
            seen["n"] += 1
            return seen["n"] == 2
        return False

    bus = ReceiptBus(loss_fn=loss)
    plant, controller, group, caps = make_components(bus=bus)
    run_n(6, group, controller, plant)
    unconfirmed = group.ledgers["b"].unconfirmed()
    assert unconfirmed
    group.manual_confirm(6, "b", unconfirmed[0].command_id, occurred=True,
                         applied_value=unconfirmed[0].action)
    path = group.save_state(tmp_path / "dual_state.json")

    plant2, controller2, group2, _ = make_components(bus=ReceiptBus())
    group2.load_state_file(path)
    controller2.update_capabilities(group2.capabilities(), group2.ledgers)
    record = group2.ledgers["b"].get(unconfirmed[0].command_id)
    assert record.status == EXECUTED and record.manually_decided
    # 队列中尚未生效的命令仍保留原 ID 与生效时刻，不会再次施加已生效命令。
    queued_ids = {c.command_id for c in group2.channels["b"].pending_commands()}
    for old in group.ledgers["b"].confirmed_occurred():
        assert old.command_id not in queued_ids

    run_n(8, group2, controller2, plant2, initial_step=6)
    assert group2.ledgers["b"].get(unconfirmed[0].command_id).status == EXECUTED


def test_disconnect_reconnect_reports_not_received_without_resend():
    plant, controller, group, _ = make_components()
    group.channels["b"].connected_fn = lambda step: step != 3

    def reconnect(step):
        return ("b",) if step == 4 else ()

    run_n(8, group, controller, plant, reconnect_fn=reconnect)
    not_applied = group.ledgers["b"].confirmed_not_occurred()
    assert any(r.note == "device_reported_not_received" for r in not_applied)
    # 断连期间没有入设备队列，重连不会自动重发。
    lost = [r for r in not_applied
            if r.note == "device_reported_not_received"][0]
    # 真实侧只记录“保持原作用”，请求动作没有被施加。
    assert group.channels["b"].ground_truth_applied[lost.command_id] == \
        pytest.approx(lost.confirmed_value)
    assert abs(lost.confirmed_value - lost.action) > 1e-9


def make_ledger(action=1.0, hold=0.0):
    ledger = CommandLedger("a")
    ledger.register(1, action, submitted_step=0, due_step=0, hold_value=hold)
    return ledger


def test_partial_receipt_without_fraction_keeps_remaining_balance():
    # 只携带已执行量、没有比例的回执不能记成全部执行。
    ledger = make_ledger(action=1.0, hold=0.0)
    ledger.ingest(Receipt(
        command_id=1, actuator="a", kind=EXEC_PARTIAL, arrived_step=1,
        applied_value=0.4,
    ))
    record = ledger.get(1)
    assert record.status == UNCERTAIN
    assert record.confirmed_value == pytest.approx(0.4)
    assert record.remaining_value == pytest.approx(0.6)
    lo, hi = ledger.possible_effect_interval(record)
    assert lo < hi

    # 既无量也无比例的部分回执：全部余额保留为待核实。
    ledger2 = make_ledger(action=1.0, hold=0.0)
    ledger2.ingest(Receipt(
        command_id=1, actuator="a", kind=EXEC_PARTIAL, arrived_step=1,
    ))
    record2 = ledger2.get(1)
    assert record2.status == UNCERTAIN
    assert record2.remaining_value == pytest.approx(1.0)


def test_incremental_then_cumulative_receipts_not_double_counted():
    ledger = make_ledger(action=1.0, hold=0.0)
    # 先收到增量回执：新增 0.3。
    ledger.ingest(Receipt(
        command_id=1, actuator="a", kind=EXEC_PARTIAL, arrived_step=1,
        applied_value=0.3, measure=MEASURE_INCREMENTAL,
    ))
    record = ledger.get(1)
    assert record.confirmed_value == pytest.approx(0.3)
    assert record.remaining_value == pytest.approx(0.7)

    # 随后收到累计回执：累计 0.6 是绝对水平，不是再叠加 0.6。
    ledger.ingest(Receipt(
        command_id=1, actuator="a", kind=EXEC_PARTIAL, arrived_step=2,
        applied_value=0.6, measure=MEASURE_CUMULATIVE,
    ))
    assert record.confirmed_value == pytest.approx(0.6)
    assert record.remaining_value == pytest.approx(0.4)

    # 完全相同的增量回执再次投递：标记 duplicate，不重复累计。
    dup = ledger.ingest(Receipt(
        command_id=1, actuator="a", kind=EXEC_PARTIAL, arrived_step=3,
        applied_value=0.3, measure=MEASURE_INCREMENTAL,
    ))
    assert dup.duplicate
    assert record.confirmed_value == pytest.approx(0.6)

    # 累计到达全部请求量后转为已执行。
    ledger.ingest(Receipt(
        command_id=1, actuator="a", kind=EXEC_PARTIAL, arrived_step=4,
        applied_value=1.0, measure=MEASURE_CUMULATIVE,
    ))
    assert record.status == EXECUTED
    assert record.remaining_value == pytest.approx(0.0)


def test_quantity_fraction_contradiction_keeps_dispute():
    ledger = make_ledger(action=1.0, hold=0.0)
    ledger.ingest(Receipt(
        command_id=1, actuator="a", kind=EXEC_PARTIAL, arrived_step=1,
        applied_value=0.4, fraction=0.9,
    ))
    record = ledger.get(1)
    # 不静默选择任一数字：争议双方保留，状态仍为待核实。
    assert record.disputed
    assert "矛盾" in record.dispute_detail
    assert record.disputed_values == pytest.approx([0.4, 0.9])
    assert record.status == UNCERTAIN
    lo, hi = ledger.possible_effect_interval(record)
    assert lo <= 0.4 and hi >= 0.9


def test_cumulative_regression_keeps_dispute():
    ledger = make_ledger(action=1.0, hold=0.0)
    ledger.ingest(Receipt(
        command_id=1, actuator="a", kind=EXEC_PARTIAL, arrived_step=1,
        applied_value=0.8, measure=MEASURE_CUMULATIVE,
    ))
    record = ledger.get(1)
    assert record.confirmed_value == pytest.approx(0.8)
    # 后到的累计回执倒退：保留争议，已确认水平不被覆盖。
    ledger.ingest(Receipt(
        command_id=1, actuator="a", kind=EXEC_PARTIAL, arrived_step=2,
        applied_value=0.5, measure=MEASURE_CUMULATIVE,
    ))
    assert record.disputed
    assert "倒退" in record.dispute_detail
    assert record.disputed_values == pytest.approx([0.8, 0.5])
    assert record.confirmed_value == pytest.approx(0.8)
    assert record.status == UNCERTAIN


def test_save_resume_after_partial_receipt_preserves_execution_facts(tmp_path):
    # 中途保存恢复后继续输入：原始执行事实（含争议与余额）不被覆盖。
    def execution(step, command):
        return 0.4 if step == 3 else 1.0

    bus = ReceiptBus()
    plant, controller, group, _ = make_components(
        bus=bus, channel_options={"b": {"execution_fn": execution}}
    )
    run_n(6, group, controller, plant)

    partial = [r for r in group.ledgers["b"].records() if r.status == UNCERTAIN]
    assert len(partial) == 1
    original = partial[0]
    original_confirmed = float(original.confirmed_value)
    original_remaining = float(original.remaining_value)
    truth_value = group.channels["b"].ground_truth_applied[original.command_id]

    # 对 a 通道制造一条“数量与比例矛盾”的争议记录。
    a_record = group.ledgers["a"].confirmed_occurred()[-1]
    group.deliver_receipt(Receipt(
        command_id=a_record.command_id, actuator="a", kind=EXEC_PARTIAL,
        arrived_step=6, applied_value=a_record.action - 0.5, fraction=0.9,
    ))
    disputed = group.ledgers["a"].get(a_record.command_id)
    assert disputed.disputed
    disputed_values = list(disputed.disputed_values)

    path = group.save_state(tmp_path / "dual_state.json")

    plant2, controller2, group2, _ = make_components(bus=ReceiptBus())
    group2.load_state_file(path)
    controller2.update_capabilities(group2.capabilities(), group2.ledgers)

    restored = group2.ledgers["b"].get(original.command_id)
    assert restored.status == UNCERTAIN
    assert restored.confirmed_value == pytest.approx(original_confirmed)
    assert restored.remaining_value == pytest.approx(original_remaining)
    restored_disputed = group2.ledgers["a"].get(a_record.command_id)
    assert restored_disputed.disputed
    assert restored_disputed.disputed_values == pytest.approx(disputed_values)

    # 重开后继续输入：原部分执行事实与争议不被新命令覆盖。
    run_n(8, group2, controller2, plant2, initial_step=6)
    after = group2.ledgers["b"].get(original.command_id)
    assert after.confirmed_value == pytest.approx(original_confirmed)
    assert after.remaining_value == pytest.approx(original_remaining)
    assert group2.channels["b"].ground_truth_applied[original.command_id] == \
        pytest.approx(truth_value)
    after_disputed = group2.ledgers["a"].get(a_record.command_id)
    assert after_disputed.disputed
    assert after_disputed.disputed_values == pytest.approx(disputed_values)
    # 已生效命令不会因保存/重开再次施加。
    queued_ids = {c.command_id for c in group2.channels["b"].pending_commands()}
    assert original.command_id not in queued_ids
