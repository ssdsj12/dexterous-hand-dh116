"""Real exchange sequencing with CPU adapters; no native qualification."""
import copy
import numpy as np
import pytest
import torch
from test_dh116_flexible_physical_exchange import setup

def test_waiting_preserves_every_physics_rod_reaction_and_actual_action(monkeypatch):
    e,sim,base,calls=setup(monkeypatch)
    action=torch.zeros((1,24),dtype=torch.float64);action[0,17]=.4
    rows=[];clocks=[]
    def choose(proposed,clock):
        clocks.append(clock)
        paused=clock["physics_step"]%2==0
        # Mutating callback copies cannot corrupt the original proposal/clock.
        returned=torch.zeros_like(proposed) if paused else proposed.clone()
        proposed[:]=.9
        return returned,dict(paused=paused,reason="cpu_measured_wait" if paused else "cpu_ready")
    result=e.control(action,observe=lambda row:rows.append(row),select_substep_action=choose)
    assert sim.current_time_step_index==10 and base.rope.steps==8
    assert len(rows)==len(clocks)==8
    assert len([c for c in calls if c[0]=="reaction"])==16
    for row,clock in zip(rows,clocks):
        evidence=row["substep_selection"]
        assert evidence["source_clock"]["physics_step"]==row["physics_step"]-1
        assert evidence["source_clock"]==clock
        actual=row["action_readback"]["request"]["action24"]
        np.testing.assert_array_equal(actual,evidence["issued_action24"])
        np.testing.assert_array_equal(evidence["proposed_action24"],action)
        assert actual[0,17]==(0. if evidence["decision"]["paused"] else .4)
    assert result["physical_samples"]==8 and result["teacher_admission"] is False

@pytest.mark.parametrize("bad",["other_action","wrong_shape","nan","non_bool_pause","missing_reason","nonfinite_evidence","unserializable_evidence"])
def test_invalid_selection_rejects_before_native_writes(monkeypatch,bad):
    e,sim,base,calls=setup(monkeypatch)
    def choose(a,clock):
        info=dict(paused=False,reason="ready")
        if bad=="other_action":a[0,7]=.1
        if bad=="wrong_shape":a=a[0]
        if bad=="nan":a[0,7]=float("nan")
        if bad=="non_bool_pause":info["paused"]=1
        if bad=="missing_reason":info["reason"]=""
        if bad=="nonfinite_evidence":info["error_m"]=float("nan")
        if bad=="unserializable_evidence":info["owner"]=object()
        return a,info
    with pytest.raises((ValueError,TypeError)):
        e.control(torch.zeros(1,24),observe=lambda row:None,select_substep_action=choose)
    assert sim.current_time_step_index==2 and base.rope.steps==0 and not calls

def test_selector_cannot_advance_sdk_before_owned_step(monkeypatch):
    e,sim,base,calls=setup(monkeypatch)
    def choose(a,clock):
        sim.current_time_step_index+=1
        return a,dict(paused=False,reason="invalid_external_advance")
    with pytest.raises(ValueError,match="clock"):
        e.control(torch.zeros(1,24),observe=lambda row:None,select_substep_action=choose)
    assert base.rope.steps==0 and not calls

def test_original_guard_still_stops_wait_step_after_persisting_it(monkeypatch):
    e,sim,base,calls=setup(monkeypatch)
    hand=e._bindings[1].hand;old=hand.update
    def fail(dt):
        old(dt);hand.measured[0,hand.joint_names.index("finger12")]=-.05
    hand.update=fail;rows=[]
    result=e.control(torch.zeros(1,24),observe=lambda row:rows.append(row),
        select_substep_action=lambda a,c:(torch.zeros_like(a),dict(paused=True,reason="wait")))
    assert result["termination_reason"]=="right:joint_limit"
    assert len(rows)==1 and rows[0]["substep_selection"]["decision"]["paused"]
    assert base.rope.steps==1


def test_authorized_left_index_after_reseat_only_cancels_wrist_axes(monkeypatch):
    e, sim, base, _ = setup(monkeypatch)
    action = torch.zeros((1, 24), dtype=torch.float64)
    action[0, [2, 12, 14]] = torch.tensor([.2, .12, -.07], dtype=action.dtype)
    rows = []

    def choose(proposed, clock):
        if clock["physics_step"] < 4:
            return proposed, dict(paused=False, reason="paired")
        proposed[:, 12:15] = 0.
        return proposed, dict(paused=False, reason="restored", index_only_after_reseat=True)

    result = e.control(action, observe=rows.append, select_substep_action=choose,
                       allow_left_index_after_reseat=True)
    assert result["physical_samples"] == 8
    assert sim.current_time_step_index == 10 and base.rope.steps == 8
    for i, row in enumerate(rows):
        selection = row["substep_selection"]
        expected = action.clone()
        if i >= 2:
            expected[:, 12:15] = 0.
        np.testing.assert_array_equal(selection["proposed_action24"], action)
        np.testing.assert_array_equal(selection["issued_action24"], expected)
        np.testing.assert_array_equal(row["action_readback"]["request"]["action24"], expected)
        assert selection["source_clock"]["physics_step"] == row["physics_step"] - 1


@pytest.mark.parametrize("bad", ["disabled", "more_index", "other_hand", "paused", "wrong_window_axes"])
def test_index_after_reseat_rejects_unapproved_commands_before_physics(monkeypatch, bad):
    e, sim, base, calls = setup(monkeypatch)
    action = torch.zeros((1, 24), dtype=torch.float64)
    action[0, [2, 12]] = torch.tensor([.2, .12], dtype=action.dtype)
    if bad == "wrong_window_axes":
        action[0, 7] = .1

    def choose(proposed, _):
        proposed[:, 12:15] = 0.
        if bad == "more_index":
            proposed[0, 2] += .01
        if bad == "other_hand":
            proposed[0, 7] = -.01
        return proposed, dict(paused=bad == "paused", reason="invalid", index_only_after_reseat=True)

    with pytest.raises(ValueError):
        e.control(action, observe=lambda row: None, select_substep_action=choose,
                  allow_left_index_after_reseat=bad != "disabled")
    assert sim.current_time_step_index == 2 and base.rope.steps == 0 and not calls
