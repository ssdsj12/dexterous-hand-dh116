"""Actual completed native ledger replay; CPU evidence, never new physics."""
from copy import deepcopy
import importlib, importlib.util, json
from pathlib import Path
import pytest
from scripts.validate_dh116_complete_pair_r0 import events
from scripts.profile_dh116_complete_pair_r0_ledger import sha
from scripts import replay_dh116_native_parameter_r0 as radius
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task,R0_TERMS
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope
ROOT=Path(__file__).resolve().parents[1]
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-sealed-r0-native/c8bd6cc7a5d18a3d65cc3d0143023353ca291f181c1c79209e5704a81947b8aa/attempt-vm7alevv/receipt.json")
RECEIPT_SHA="5481fb005ccc3d59197c7489a795f7623b1fbe36dc0bc8978a3760d48636b41b"
PHYSICAL_SHA="f00c545d2464a4ba0776025a8a3fe6c3c1e2187ee81c6c46a8ef4fb67c0e32cc"
MODULE="scripts.prepare_dh116_live_measured_approach"

def api():
    assert importlib.util.find_spec(MODULE) is not None,"live measured approach source is not connected"
    return importlib.import_module(MODULE)

@pytest.fixture(scope="module")
def actual():
    assert sha(RECEIPT)==RECEIPT_SHA
    receipt=json.loads(RECEIPT.read_bytes())
    assert receipt["native_qualification"] is True and receipt["outer_exit_code"]==0
    output=Path(receipt["output"])
    assert sha(output/"physical.jsonl")==PHYSICAL_SHA
    rows=[]
    for row in events(output/"physical.jsonl"):
        rows.append(row)
        if row.get("operation")=="end":break
    packet=next(events(output/"director_reset.json"))["payload"]
    return rows,packet

def construct(cls,actual,**extra):
    rows,packet=actual
    reset=next(r["payload"] for r in rows if r.get("operation")=="reset")
    measured=radius.check_radius_packet(packet,reset["reset"]["rod_reset"])
    task=load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",ROOT)
    budget=json.loads((ROOT/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_bytes())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"]=2
    budget["physics"]["solver_substeps_per_physics_step"]=12
    obj=cls(task,"pinch",env_id=0,budget_config=budget,
        diagnostic_coefficients={k:float(i+1) for i,k in enumerate(R0_TERMS)},
        backend_artifact=radius.BACKEND,backend_sha256=radius.BACKEND_SHA,
        radius_sample=dict(value=measured["value"],source="native_parameter_snapshot:"+_digest(measured)),**extra)
    return obj,reset

def run_first(obj,rows,reset):
    init=obj.complete_physical_reset(reset,reset_approach=None)
    for r in rows:
        op,p=r.get("operation"),r.get("payload")
        if op=="begin":obj.begin(**p)
        elif op=="physical":obj.record(p)
        elif op=="end":return init,obj.end(p)
    raise AssertionError("missing complete control")

def test_actual_first_control_connects_reset_and_all_eight_distances(actual):
    m=api()
    with codec_scope():
        obj,reset=construct(m.MeasuredApproachObserver,actual,project=ROOT,parameter_packet=actual[1])
        init,out=run_first(obj,actual[0],reset)
    assert init["measured_approach_contract"]["reset_baseline_available"]
    assert out["control_receipt"]["terms"]["approach_progress"]["valid_samples"]==8
    assert set(out["control_receipt"]["blocked_terms"])=={"hold_progress","material_anchor_slip","instability"}
    assert out["training_reward"] is None and not out["teacher_admission"]
    assert out["approach_first_sample_is_unavailable"] is False
    assert obj.report()["controls"][0]["approach_first_sample_is_unavailable"] is False
    for row in out["physical_rows"]:
        a=row["pair_diagnostics"]["measured_approach"]
        assert a["source"]["geometry_contract_version"]==m.GEOMETRY_VERSION
        assert not a["contact_inferred_from_geometry"] and a["contact_observed"] is None
        assert a["new_physics_steps"]==0 and not a["teacher_admission"]
        assert a["reset_baseline_available"] is False
    assert not init["native_qualification"]

def test_old_observer_remains_default_and_reports_approach_missing(actual):
    m=api()
    from scripts import prepare_dh116_live_r0_native as old
    from go2_pvcnn.control.dh116.tabletop_r0_live_observer import LiveR0Observer
    from go2_pvcnn.control.dh116.tabletop_r0_measurement_origin import OriginLiveR0Observer
    assert old.LiveR0Observer is LiveR0Observer
    with codec_scope():
        obj,reset=construct(OriginLiveR0Observer,actual)
        _,out=run_first(obj,actual[0],reset)
    assert "approach_progress" in out["control_receipt"]["blocked_terms"]
    assert m.MeasuredApproachSession is not old.LiveSession

@pytest.mark.parametrize("kind",["provided_reset","bad_parameters","provided_auxiliary","geometry_tamper"])
def test_no_unchecked_sources_or_caller_override(actual,kind):
    m=api();rows,packet=deepcopy(actual)
    if kind=="bad_parameters":packet["native_parameters"]["contact_radius_m"]=.1
    with codec_scope():
        obj,reset=construct(m.MeasuredApproachObserver,(rows,actual[1]),project=ROOT,parameter_packet=packet)
        if kind in ("provided_reset","bad_parameters"):
            with pytest.raises((ValueError,TypeError,KeyError)):
                obj.complete_physical_reset(reset,reset_approach={} if kind=="provided_reset" else None)
        else:
            obj.complete_physical_reset(reset,reset_approach=None)
            begin=next(r["payload"] for r in rows if r.get("operation")=="begin")
            obj.begin(**begin)
            event=next(r["payload"] for r in rows if r.get("operation")=="physical")
            if kind=="geometry_tamper":event["exchange"]["closure_geometry"]["raw"]["rod_nodes_m"][0,2]+=.01
            with pytest.raises((ValueError,TypeError,KeyError)):
                obj.record(event,auxiliary={} if kind=="provided_auxiliary" else None)
        assert obj._state=="failed"

@pytest.mark.parametrize("fail",[False,True])
def test_live_session_factory_binds_actual_packet_and_restores_owner(tmp_path,monkeypatch,fail):
    m=api();old=m.operation.LiveR0Observer
    sample={"actual_owned":"packet"}
    captured=[]
    def observer(*args,**kwargs):
        captured.append(kwargs)
        return object()
    def initialize(self,packet,error,identity,write_original):
        assert packet is sample
        m.operation.LiveR0Observer("task",env_id=0)
        if fail:raise RuntimeError("existing initialization failed")
        return "unchanged"
    monkeypatch.setattr(m,"MeasuredApproachObserver",observer)
    monkeypatch.setattr(m.operation.LiveSession,"initialize",initialize)
    session=m.MeasuredApproachSession(ROOT,tmp_path)
    if fail:
        with pytest.raises(RuntimeError,match="existing initialization"):
            session.initialize(sample,None,{},None)
    else:assert session.initialize(sample,None,{},None)=="unchanged"
    assert captured==[dict(env_id=0,project=ROOT,parameter_packet=sample)]
    assert m.operation.LiveR0Observer is old

def test_replay_comparison_rejects_changes_outside_approach():
    from scripts.replay_dh116_live_measured_approach import compare_row
    old=dict(measurements=dict(approach_distance_m=dict(value=None,source=None),left=dict(value=False,source="real")),
        unavailable_sources=dict(approach_distance_m="missing",stable="missing"),
        pair_diagnostics=dict(original=True),physical_guard_failure=None,
        r0=dict(terms=dict(approach_progress=dict(raw=None),joint_limit=dict(raw=0)),
            term_stats=dict(approach_progress=dict(count=0),joint_limit=dict(count=1)),
            diagnostic_reward=None,blocked_terms=["approach_progress"],success_candidate=False))
    new=deepcopy(old)
    new["pair_diagnostics"]["measured_approach"]={"evidence":True}
    new["measurements"]["approach_distance_m"]=dict(value=[.001,.002],source="measured")
    new["unavailable_sources"].pop("approach_distance_m")
    new["r0"]["terms"]["approach_progress"]={"raw":.001}
    new["r0"]["term_stats"]["approach_progress"]={"count":1}
    new["r0"]["blocked_terms"]=[];new["r0"]["diagnostic_reward"]=.001
    assert compare_row(old,new)=={"evidence":True}
    for key in ("guard","contact","term","success"):
        bad=deepcopy(new)
        if key=="guard":bad["physical_guard_failure"]="joint_limit"
        elif key=="contact":bad["measurements"]["left"]["value"]=True
        elif key=="term":bad["r0"]["terms"]["joint_limit"]["raw"]=1
        else:bad["r0"]["success_candidate"]=True
        with pytest.raises(ValueError,match="non-approach"):compare_row(old,bad)
