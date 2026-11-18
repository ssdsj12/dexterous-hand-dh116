"""Opt-in post-step source clearance guard; CPU doubles, no native launch."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from test_dh116_flexible_physical_exchange import setup as original_setup
from test_dh116_dual_pinch_builder import prepared,geometry

MODULE="go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard"
def module():
    return __import__(MODULE,fromlist=["ClearanceGuardedExchange"])

def guarded(monkeypatch,*,breach_at=None,fault=None):
    old,sim,base,calls=original_setup(monkeypatch)
    def after(previous):
        row=old._read_geometry()
        row.update(rod_source_physics_step=previous["physics_step"],
            geometry_phase="sdk_complete_rod_pending",source=module().GEOMETRY_SOURCE,
            lowest_shape_index=np.array([0],dtype=np.int64))
        if breach_at is not None and sim.current_time_step_index>=breach_at:
            row["clearance_m"][0,1]=.002099
        if fault=="shape":row["clearance_m"]=torch.zeros(2)
        if fault=="nan":row["clearance_m"][0,0]=float("nan")
        if fault=="phase":row["geometry_phase"]="synchronized"
        if fault=="clock":row["rod_source_physics_step"]-=1
        if fault=="source":row["source"]="configuration_default"
        return row
    cls=module().ClearanceGuardedExchange
    value=cls(sim,old._rod,old._bindings,root_seed_w=old._seed,
        read_geometry=old._read_geometry,read_geometry_after_sdk=after,physics_budget_end=18)
    return value,sim,base,calls

def test_safe_eight_rows_have_owned_same_boundary_measured_clearance(monkeypatch):
    e,sim,base,calls=guarded(monkeypatch)
    rows=[];result=e.control(torch.zeros(1,24),observe=rows.append)
    assert result["termination_reason"] is None and len(rows)==8
    assert sim.current_time_step_index==10 and base.rope.steps==8
    for row in rows:
        packet=row["post_step_clearance"]
        assert packet["clearance_m"]==[[.012,.012]]
        assert packet["floor_m"]==.0021 and packet["passed"] is True
        assert packet["identity"]==row["action_readback"]["post"]["identity"][0]
        assert packet["native_clearance_qualified"] is False and packet["hand_table_contacts"] is None
        assert module().validate_post_step_clearance(row)["passed"] is True

@pytest.mark.parametrize("step",[3,10])
def test_breach_delivers_terminal_row_once_without_next_action(monkeypatch,step):
    e,sim,base,calls=guarded(monkeypatch,breach_at=step)
    rows=[];result=e.control(torch.zeros(1,24),observe=rows.append)
    assert result["termination_reason"]=="post_step_source_clearance:right"
    assert result["physical_samples"]==step-2 and len(rows)==step-2
    assert sim.current_time_step_index==step and base.rope.steps==step-2
    assert rows[-1]["post_step_clearance"]["failed_sides"]==["right"]
    assert module().validate_post_step_clearance(rows[-1])["passed"] is False
    count=len(calls)
    with pytest.raises(RuntimeError,match="closed"):e.control(torch.zeros(1,24),observe=rows.append)
    assert len(calls)==count and len(rows)==step-2

def test_existing_joint_guard_keeps_priority_and_clearance_failure_still_recorded(monkeypatch):
    e,sim,base,calls=guarded(monkeypatch,breach_at=3)
    right=e._bindings[1].hand
    right.update=lambda dt:right.measured.__setitem__((0,right.joint_names.index("finger12")),-.05)
    rows=[];result=e.control(torch.zeros(1,24),observe=rows.append)
    assert result["termination_reason"]=="right:joint_limit"
    assert rows[0]["post_step_clearance"]["passed"] is False
    assert module().validate_post_step_clearance(rows[0])["passed"] is False

@pytest.mark.parametrize("fault",["shape","nan","phase","clock","source"])
def test_invalid_after_sdk_source_poisoned_before_rod_or_subsequent_action(monkeypatch,fault):
    e,sim,base,calls=guarded(monkeypatch,fault=fault)
    with pytest.raises(ValueError):e.control(torch.zeros(1,24),observe=lambda row:None)
    assert sim.current_time_step_index==3 and base.rope.steps==0
    count=len(calls)
    with pytest.raises(RuntimeError,match="closed"):e.control(torch.zeros(1,24),observe=lambda row:None)
    assert len(calls)==count

@pytest.mark.parametrize("fault",["floor","identity","pass","source","time","reason"])
def test_independent_row_validation_rejects_tamper(monkeypatch,fault):
    e,_,_,_=guarded(monkeypatch,breach_at=3)
    rows=[];e.control(torch.zeros(1,24),observe=rows.append)
    row=deepcopy(rows[0]);packet=row["post_step_clearance"]
    if fault=="floor":packet["floor_m"]=.0019
    elif fault=="identity":packet["identity"]["physics_step"]+=1
    elif fault=="pass":packet["passed"]=True
    elif fault=="source":packet["source_sha256"]={}
    elif fault=="time":packet["sdk_time_s"]+=.001
    else:row["guard_failure"]=None
    with pytest.raises(ValueError):module().validate_post_step_clearance(row)

def test_actual_builder_live_geometry_reader_control_path(monkeypatch,tmp_path,prepared,geometry):
    from go2_pvcnn.control.dh116 import tabletop_deformable_dual_hand_episode as episode
    from test_dh116_dual_pinch_interval import interval
    monkeypatch.setattr(episode,"FlexiblePhysicalExchange",module().ClearanceGuardedExchange)
    result,rows,events,trace,refs=interval(monkeypatch,tmp_path,prepared,geometry)
    physical=[r for r in rows if r.get("kind")=="physical"]
    assert len(physical)==8 and result["sdk_step"]==10
    assert all(module().validate_post_step_clearance(r["exchange"])["passed"] for r in physical)
    assert result["pre"]["common"]["valid"].sum()==66
    assert result["result"]["teacher_admission"] is False

@pytest.mark.parametrize("gap,passed",[(.0021,True),(np.nextafter(.0021,-np.inf),False)])
def test_exact_original_floor_is_inclusive_without_relaxation(monkeypatch,gap,passed):
    e,sim,base,calls=guarded(monkeypatch)
    previous=e._read_geometry_after_sdk
    def after(clock):
        row=previous(clock);row["clearance_m"][:]=gap;return row
    e._read_geometry_after_sdk=after
    rows=[];result=e.control(torch.zeros(1,24),observe=rows.append)
    assert rows[-1]["post_step_clearance"]["passed"] is passed
    assert len(rows)==(8 if passed else 1)
    if not passed:assert result["termination_reason"]=="post_step_source_clearance:left,right"

def test_terminal_clearance_is_durable_in_existing_journal_before_settlement(monkeypatch,tmp_path):
    import json
    from go2_pvcnn.control.dh116 import tabletop_deformable_dual_hand_episode as owner
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import DualHandEvidenceJournal,inspect_evidence
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from test_deformable_r0_replay import setup
    from test_dh116_dual_hand_evidence_journal import connect
    monkeypatch.setattr(owner,"FlexiblePhysicalExchange",module().ClearanceGuardedExchange)
    episode,rod,hands,sim,rows,_=setup(monkeypatch,tmp_path,mode="empty")
    old=episode._exchange._read_geometry_after_sdk
    def measured(clock):
        row=old(clock);row["clearance_m"][0,1]=.002099;return row
    episode._exchange._read_geometry_after_sdk=measured
    path=tmp_path/"terminal-clearance.jsonl"
    with DualHandEvidenceJournal(path,physics_budget_end=18,solver_substeps=12) as journal:
        connect(journal,episode,rows[0])
        pre=episode.pre_action_state();action=torch.zeros((1,24))
        journal.begin(pre,action);result=episode.control(action)
        prefix=inspect_evidence(path)
        assert prefix["terminal_reason"]=="post_step_source_clearance:right"
        assert prefix["measured_physics_rows"]==1 and prefix["unsettled_control_window"]
        raw=json.loads(path.read_text().splitlines()[-1])
        assert module().validate_post_step_clearance(decode_owned(raw["payload"])["exchange"])["passed"] is False
        journal.end(result)
        with pytest.raises(RuntimeError,match="active"):episode.control(action)
    assert sim.current_time_step_index==3 and rod.rope.steps==1
    assert inspect_evidence(path)["settled_control_windows"]==1
