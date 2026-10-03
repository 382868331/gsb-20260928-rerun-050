"""双执行器协同与不确定回执的闭环实验测试。"""

import json

import numpy as np

from control_lab.capability import Capability
from control_lab.config import CartConfig, MPCConfig
from control_lab.dual_controller import DualMPCController
from control_lab.dual_experiment import run_dual_experiment
from control_lab.plants import CartPlant


def _setup(delay_a=2, delay_b=0, strategy="wait"):
    plant = CartPlant(CartConfig())
    caps = {
        "a": Capability(-2.0, 2.0, 1.0, 1.0, delay_a),
        "b": Capability(-1.0, 1.0, 0.6, 0.6, delay_b),
    }
    ctrl = DualMPCController(plant, MPCConfig(), capabilities=caps)
    return plant, ctrl, caps, strategy


def _drop_receipts(drop_steps):
    def receipt_fn(step, events):
        if step in drop_steps:
            return []  # 回执丢失
        return [{**e, "step": step, "applied": True,
                 "applied_action": [e["applied_action"]]} for e in events]
    return receipt_fn


def test_lost_receipt_waits_without_resending():
    plant, ctrl, caps, strategy = _setup()
    log = run_dual_experiment(
        plant, ctrl, capabilities=caps, n_steps=30,
        initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 1.5, strategy=strategy,
        receipt_fn=_drop_receipts({5, 6, 7}),
    )
    df = log.to_frame()
    uncertain_rows = df[df["n_uncertain"] > 0]
    assert len(uncertain_rows) > 0
    # 不确定性在命令到期当拍结束时才显现；之后的每一拍都必须保守等待
    first_uncertain_step = int(uncertain_rows["step"].iloc[0])
    later = uncertain_rows[uncertain_rows["step"] > first_uncertain_step]
    assert len(later) > 0
    for _, row in later.iterrows():
        assert row["holding"]
        assert row["reason"] == "waiting_uncertain_confirmation"
    assert df["n_confirmed_not_applied"].max() == 0


def test_compensate_strategy_shrinks_remaining_capability():
    plant, ctrl, caps, _ = _setup()
    log = run_dual_experiment(
        plant, ctrl, capabilities=caps, n_steps=30,
        initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 1.5, strategy="compensate",
        receipt_fn=_drop_receipts({5, 6, 7}),
    )
    plans_with_override = [p for p in log.plans
                           if p["uncertainty"]["capability_overrides"]]
    assert plans_with_override
    for plan in plans_with_override:
        overrides = plan["uncertainty"]["capability_overrides"]
        for name, cap in overrides.items():
            original = caps[name]
            assert cap["du_up"] <= original.du_up + 1e-12
            assert cap["du_down"] <= original.du_down + 1e-12
        unc = plan["uncertainty"]
        assert unc["uncertain_commands"]
        lo = unc["planned_total_min"]
        hi = unc["planned_total_max"]
        mid = np.asarray(plan["planned_actions"])[:, 2]
        assert np.all(np.asarray(lo) <= mid + 1e-9)
        assert np.all(np.asarray(hi) >= mid - 1e-9)


def test_manual_confirm_not_applied_allows_reschedule():
    plant, ctrl, caps, _ = _setup()
    decisions = []

    def receipt_fn(step, events):
        if 5 <= step <= 8:
            return []
        return [{**e, "step": step, "applied": True,
                 "applied_action": [e["applied_action"]]} for e in events]

    def confirm_fn(step, ledger):
        if step == 9:
            for record in ledger.records():
                if record.status == "uncertain":
                    decisions.append({"actuator": record.actuator,
                                      "command_id": record.command_id,
                                      "applied": False})
            return decisions
        return []

    log = run_dual_experiment(
        plant, ctrl, capabilities=caps, n_steps=30,
        initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 1.5, strategy="wait",
        receipt_fn=receipt_fn, manual_confirmation_fn=confirm_fn,
    )
    df = log.to_frame()
    after = df[df["step"] > 9]
    assert not after["holding"].any()
    assert df["n_confirmed_not_applied"].max() >= 1


def test_cancellation_metric_matches_applied_actions():
    plant, ctrl, caps, _ = _setup(delay_a=3, delay_b=0)
    log = run_dual_experiment(
        plant, ctrl, capabilities=caps, n_steps=40,
        initial_state=np.array([0.0, 0.0]),
        target_fn=lambda s: 2.0 if s < 20 else -2.0,
        strategy="compensate",
    )
    df = log.to_frame()
    expected = np.where(
        df["applied_a"] * df["applied_b"] < 0,
        np.minimum(df["applied_a"].abs(), df["applied_b"].abs()), 0.0)
    assert np.allclose(df["cancellation"], expected)
    assert "residual_error" in df.columns
    assert "unmet_reason" in df.columns


def test_capability_reduction_rechecks_pending_keeps_applied():
    plant, ctrl, caps, _ = _setup()

    def capability_change(step):
        if step == 15:
            return {"a": Capability(-1.0, 1.0, 0.3, 0.3, 2)}
        return None

    log = run_dual_experiment(
        plant, ctrl, capabilities=caps, n_steps=40,
        initial_state=np.array([-1.0, 0.0]),
        target_fn=lambda s: 2.0, strategy="compensate",
        capability_change_fn=capability_change,
    )
    event = next(p for p in log.plans if p["capability_event"])
    recheck = event["capability_event"]["recheck"]
    assert recheck["a"]["applied_facts_kept"]
    df = log.to_frame()
    after = df[df["step"] > 15]
    assert (after["applied_a"].abs() <= 1.0 + 1e-9).all()


def test_snapshot_resume_matches_continuous_run(tmp_path):
    snapshot = tmp_path / "snapshot.json"

    plant = CartPlant(CartConfig())
    caps = {
        "a": Capability(-2.0, 2.0, 1.0, 1.0, 2),
        "b": Capability(-1.0, 1.0, 0.6, 0.6, 0),
    }
    ctrl = DualMPCController(plant, MPCConfig(), capabilities=caps)
    continuous = run_dual_experiment(
        plant, ctrl, capabilities=caps, n_steps=40,
        initial_state=np.array([0.5, 0.0]),
        target_fn=lambda s: 1.0, strategy="compensate",
        receipt_fn=_drop_receipts({10, 11, 12}),
        snapshot_step=20, snapshot_path=snapshot,
    )
    assert snapshot.exists()
    state = json.loads(snapshot.read_text(encoding="utf-8"))
    assert any(r["status"] == "uncertain" for r in state["ledger"]["records"])
    assert all("command_id" in d for d in state["ledger"]["decisions"])

    plant2 = CartPlant(CartConfig())
    ctrl2 = DualMPCController(plant2, MPCConfig(), capabilities=caps)
    resumed = run_dual_experiment(
        plant2, ctrl2, capabilities=caps, n_steps=19,
        initial_state=np.zeros(2),
        target_fn=lambda s: 1.0, strategy="compensate",
        receipt_fn=_drop_receipts({10, 11, 12}),
        resume_state=state,
    )
    cont = continuous.to_frame().set_index("step")
    res = resumed.to_frame().set_index("step")
    assert res.index[0] == 21
    for step in res.index:
        assert abs(cont.loc[step, "x"] - res.loc[step, "x"]) < 1e-9
        assert abs(cont.loc[step, "applied_total"]
                   - res.loc[step, "applied_total"]) < 1e-9

