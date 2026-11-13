"""Read-only native evidence tests; no Isaac imports or new physical steps."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import pytest
from scripts import audit_dh116_bilateral_native_evidence as native
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def actual():
    assert native.sha(native.ACTUAL_RECEIPT)==native.ACTUAL_RECEIPT_SHA
    receipt=json.loads(native.ACTUAL_RECEIPT.read_bytes())
    native.validate_receipt_scope(receipt)
    folder=Path(receipt["output"]);events=folder/"physical.jsonl"
    assert native.sha(events)==receipt["artifact_sha256"]["physical.jsonl"]
    first=last=reset=None
    for line in events.open():
        row=json.loads(line);op=row.get("operation")
        if op=="reset":reset=decode_owned(row["payload"]["reset"])
        elif op=="physical":
            if first is None:first=decode_owned(row["payload"]["exchange"])
            last=row["payload"]["exchange"]
    parameter=decode_owned(json.loads((folder/"director_reset.json").read_bytes())["payload"])
    return first,decode_owned(last),reset,parameter

@pytest.fixture(scope="module")
def reader():
    from go2_pvcnn.control.dh116.tabletop_bilateral_r0_approach import BilateralMeasuredPinchApproach
    return BilateralMeasuredPinchApproach(ROOT)

def test_old_default_does_not_silently_accept_bilateral_scope(actual):
    from go2_pvcnn.control.dh116.tabletop_segment_r0_approach import SegmentMeasuredPinchApproach
    with pytest.raises(ValueError):SegmentMeasuredPinchApproach(ROOT).capture(actual[1])

def test_bilateral_actual_metric_is_only_geometry_and_owned(reader,actual):
    result=reader.capture(actual[1])
    assert result["auxiliary"]["identity"]["physics_step"]==234
    assert result["auxiliary"]["samples"]["approach_distance_m"]["value"]==pytest.approx([0.,0.],abs=1e-10)
    assert set(result["auxiliary"]["samples"])=={"approach_distance_m"}
    assert result["source"]["geometry_contract_version"]=="dh116-bilateral-owned-geometry-v1"
    assert result["source"]["uncertainty_m"] is None
    assert result["contact_observed"] is None
    assert not result["contact_inferred_from_geometry"] and not result["stability_measured"]
    assert not result["teacher_admission"] and not result["native_qualification"]
    assert result["new_physics_steps"]==result["ppo_updates"]==0
    result["auxiliary"]["samples"]["approach_distance_m"]["value"][0]=9
    assert reader.capture(actual[1])["auxiliary"]["samples"]["approach_distance_m"]["value"][0]==0

def test_reset_baseline_uses_actual_sdk2_not_sdk3(reader,actual):
    result=reader.capture_reset(actual[2],actual[3])
    assert result["reset_baseline_available"]
    assert result["auxiliary"]["identity"]["physics_step"]==2
    assert result["source"]["reset_body_pose_is_not_authored_seed"]
    assert not result["r0_native_reset_complete"] and not result["teacher_admission"]
    assert result["auxiliary"]["samples"]["approach_distance_m"]["value"]!=reader.capture(actual[1])["auxiliary"]["samples"]["approach_distance_m"]["value"]

@pytest.mark.parametrize("fault",["version","identity","left_action","right_action","node","radius","source"])
def test_bilateral_actual_measurements_cannot_be_relabelled(reader,actual,fault):
    e=deepcopy(actual[1])
    if fault=="version":e["closure_geometry"]["version"]="dh116-measured-segment-closure-geometry-v1"
    elif fault=="identity":e["material_director"]["identity"]["physics_step"]+=1
    elif fault=="left_action":e["action_readback"]["request"]["action24"][0,0]+=.01
    elif fault=="right_action":e["action_readback"]["request"]["action24"][0,17]+=.01
    elif fault=="node":e["rod"]["node_positions"][0,0,0]+=.01
    elif fault=="radius":e["material_director"]["reward_radius_sample"]["value"]*=2
    else:e["closure_geometry"]["sources"]["bilateral_owned"]={}
    with pytest.raises(ValueError):reader.capture(e)

def test_pinch_reader_does_not_qualify_other_grasps(reader,actual):
    with pytest.raises(ValueError):reader.capture(actual[1],grasp="tripod")
    with pytest.raises(ValueError):reader.capture_reset(actual[2],actual[3],grasp="power")
