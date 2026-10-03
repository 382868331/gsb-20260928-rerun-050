"""回执台账测试：重复、迟到、部分执行、重连与人工核实。"""

import numpy as np

from control_lab.receipts import (
    CONFIRMED_APPLIED, CONFIRMED_NOT_APPLIED, QUEUED, UNCERTAIN,
    ReceiptLedger,
)


def _cmd(command_id, action, due_step, actuator="a", submitted=0):
    return {"command_id": command_id, "actuator": actuator,
            "action": [action], "submitted_step": submitted,
            "due_step": due_step}


def test_due_without_receipt_becomes_uncertain_not_unexecuted():
    ledger = ReceiptLedger()
    ledger.register(_cmd(0, 1.0, due_step=3))
    ledger.mark_due(3)
    record = ledger.get("a", 0)
    assert record.status == UNCERTAIN
    # 不能当作未执行：不计入可重新安排的差额
    assert ledger.compensatable_deficit("a") == 0.0


def test_duplicate_receipt_keeps_first_attribution():
    ledger = ReceiptLedger()
    ledger.register(_cmd(0, 1.0, due_step=1))
    ledger.mark_due(1)
    first = ledger.receive_receipt(
        {"actuator": "a", "command_id": 0, "step": 2, "applied": True})
    dup = ledger.receive_receipt(
        {"actuator": "a", "command_id": 0, "step": 3, "applied": True})
    assert first["reason"] == "resolved"
    assert dup["reason"] == "duplicate"
    record = ledger.get("a", 0)
    assert record.duplicate_receipts == 1
    assert record.resolution_step == 2
    # 只有一条核实决定，不重复记账
    assert len(ledger.decisions) == 1


def test_stale_receipt_attributed_to_original_command():
    ledger = ReceiptLedger()
    ledger.register(_cmd(0, 1.0, due_step=1, submitted=0))
    ledger.register(_cmd(1, -0.5, due_step=3, submitted=2))
    ledger.mark_due(3)
    result = ledger.receive_receipt(
        {"actuator": "a", "command_id": 0, "step": 4, "applied": True})
    assert result["status"] == CONFIRMED_APPLIED
    old = ledger.get("a", 0)
    new = ledger.get("a", 1)
    assert old.status == CONFIRMED_APPLIED and old.late_receipt
    # 新命令不被旧回执错误确认
    assert new.status == UNCERTAIN


def test_partial_execution_records_deficit():
    ledger = ReceiptLedger()
    ledger.register(_cmd(0, 1.0, due_step=1))
    ledger.mark_due(1)
    result = ledger.receive_receipt(
        {"actuator": "a", "command_id": 0, "step": 1, "applied": True,
         "applied_action": [0.4]})
    assert result["partial"]
    record = ledger.get("a", 0)
    assert record.status == CONFIRMED_APPLIED
    assert record.partial
    assert abs(record.deficit - 0.6) < 1e-9
    # 差额是明确未发生的作用，允许重新安排
    assert abs(ledger.compensatable_deficit("a") - 0.6) < 1e-9


def test_reconnect_resync_preserves_attribution():
    ledger = ReceiptLedger()
    ledger.register(_cmd(0, 1.0, due_step=1, submitted=0))
    ledger.register(_cmd(1, 0.5, due_step=2, submitted=1))
    ledger.mark_due(2)
    results = ledger.receive_resync(5, [
        {"actuator": "a", "command_id": 0, "applied": True},
        {"actuator": "a", "command_id": 1, "applied": False},
    ])
    assert all(r["accepted"] for r in results)
    assert ledger.get("a", 0).status == CONFIRMED_APPLIED
    assert ledger.get("a", 1).status == CONFIRMED_NOT_APPLIED
    assert ledger.get("a", 0).confirmed_by == "resync"


def test_manual_confirmation_required_before_reschedule():
    ledger = ReceiptLedger()
    ledger.register(_cmd(0, 1.0, due_step=1))
    ledger.mark_due(1)
    # 未确认前不能重新安排
    assert ledger.compensatable_deficit("a") == 0.0
    ledger.confirm_not_applied("a", 0, step=5)
    assert ledger.compensatable_deficit("a") == 1.0


def test_manual_confirm_applied_avoids_double_compensation():
    ledger = ReceiptLedger()
    ledger.register(_cmd(0, 1.0, due_step=1))
    ledger.mark_due(1)
    ledger.confirm_applied("a", 0, step=5)
    assert ledger.compensatable_deficit("a") == 0.0
    assert ledger.get("a", 0).confirmed_by == "manual"


def test_receipt_for_other_actuator_not_misattributed():
    ledger = ReceiptLedger(("a", "b"))
    ledger.register(_cmd(0, 1.0, due_step=1, actuator="a"))
    ledger.mark_due(1)
    result = ledger.receive_receipt(
        {"actuator": "b", "command_id": 0, "step": 1, "applied": True})
    assert not result["accepted"]
    assert ledger.get("a", 0).status == UNCERTAIN
    assert len(ledger.orphan_receipts) == 1


def test_state_roundtrip_preserves_records_and_decisions():
    ledger = ReceiptLedger()
    ledger.register(_cmd(0, 1.0, due_step=1, submitted=0))
    ledger.register(_cmd(1, -0.5, due_step=2, submitted=1))
    ledger.mark_due(2)
    ledger.receive_receipt(
        {"actuator": "a", "command_id": 0, "step": 2, "applied": True,
         "applied_action": [0.7]})
    state = ledger.to_state()
    restored = ReceiptLedger()
    restored.load_state(state)
    assert restored.get("a", 0).partial
    assert restored.get("a", 1).status == UNCERTAIN
    assert restored.decisions == ledger.decisions
    assert restored.view()["uncertain"]["a"][0]["command_id"] == 1

