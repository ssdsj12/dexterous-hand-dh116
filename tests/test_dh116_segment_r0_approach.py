from copy import deepcopy
import hashlib,json
from pathlib import Path
import pytest
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from scripts import audit_dh116_pinch_segment_postexit_v2 as locked

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def actual():
    locked.load_locked()
    first=last=None
    for line in (locked.NATIVE/"physical.jsonl").open():
        row=json.loads(line)
        if row.get("operation")=="physical":
            if first is None:first=decode_owned(row["payload"]["exchange"])
            last=row["payload"]["exchange"]
    return first,decode_owned(last)

@pytest.fixture(scope="module")
def reader():
    from go2_pvcnn.control.dh116.tabletop_segment_r0_approach import SegmentMeasuredPinchApproach
    return SegmentMeasuredPinchApproach(ROOT)

def test_original_default_rejects_new_segment_contract(actual):
    from go2_pvcnn.control.dh116.tabletop_r0_measured_approach import MeasuredPinchApproach
    with pytest.raises(ValueError):MeasuredPinchApproach(ROOT).capture(actual[0])

def test_actual_segment_approach_retains_missing_stability_slip_and_native_gate(reader,actual):
    r=reader.capture(actual[1])
    assert r["auxiliary"]["identity"]["physics_step"]==234
    assert r["auxiliary"]["samples"]["approach_distance_m"]["value"]==pytest.approx([0.,.0021613144781440496],abs=1e-10)
    assert set(r["auxiliary"]["samples"])=={"approach_distance_m"}
    assert r["source"]["geometry_contract_version"]=="dh116-measured-segment-closure-geometry-v1"
    assert r["source"]["uncertainty_m"] is None
    assert r["contact_observed"] is None and r["contact_inferred_from_geometry"] is False
    assert not r["stability_measured"] and not r["native_qualification"] and not r["teacher_admission"]
    assert r["new_physics_steps"]==r["ppo_updates"]==0

@pytest.mark.parametrize("fault",["version","identity","pose","node","radius","certificate","source"])
def test_real_segment_evidence_cannot_be_relabelled_or_tampered(reader,actual,fault):
    e=deepcopy(actual[1])
    if fault=="version":e["closure_geometry"]["version"]="dh116-measured-closure-geometry-v1"
    elif fault=="identity":e["material_director"]["identity"]["physics_step"]+=1
    elif fault=="pose":e["closure_geometry"]["raw"]["hands"]["right"]["link_poses_xyzw"][0,0,0]+=.01
    elif fault=="node":e["rod"]["node_positions"][0,0,0]+=.01
    elif fault=="radius":e["material_director"]["reward_radius_sample"]["value"]*=2
    elif fault=="certificate":e["closure_geometry"]["support_certificate_version"]="invented"
    else:e["closure_geometry"]["sources"]["implementation"]={}
    with pytest.raises(ValueError):reader.capture(e)

def test_reset_is_actual_zero_step_readback_not_first_step(reader):
    rows=(locked.NATIVE/"physical.jsonl").open()
    reset=next(decode_owned(r["payload"]) for r in map(json.loads,rows) if r.get("operation")=="reset")
    rows.close()
    p0=decode_owned(json.loads((locked.NATIVE/"director_reset.json").read_bytes())["payload"])
    r=reader.capture_reset(reset["reset"],p0)
    assert r["reset_baseline_available"] and r["auxiliary"]["identity"]["physics_step"]==2
    assert r["source"]["reset_body_pose_is_not_authored_seed"]
    assert not r["r0_native_reset_complete"] and not r["native_qualification"]

def test_owned_output_never_mutates_native_record_or_later_readback(reader,actual):
    before=reader.capture(actual[1]);expected=before["auxiliary"]["samples"]["approach_distance_m"]["value"][1]
    before["auxiliary"]["samples"]["approach_distance_m"]["value"][1]=0
    assert reader.capture(actual[1])["auxiliary"]["samples"]["approach_distance_m"]["value"][1]==expected
