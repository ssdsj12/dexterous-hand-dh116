"""Recorded native Reset is replayed on CPU; no reset/SDK call is performed."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_r0_measured_approach import MeasuredPinchApproach
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from scripts import replay_dh116_native_parameter_r0 as baseline

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def case():
    receipt=json.loads(baseline.read_pinned(baseline.RECEIPT,baseline.RECEIPT_SHA))
    root=Path(receipt["output"])
    rows=[decode_owned(json.loads(l)) for l in baseline.read_pinned(
        root/"physical.jsonl",receipt["artifact_sha256"]["physical.jsonl"]).splitlines()]
    reset=next(r["payload"]["reset"] for r in rows if r.get("operation")=="reset")
    p=decode_owned(json.loads(baseline.read_pinned(root/"director_reset.json",
        receipt["artifact_sha256"]["director_reset.json"]))["payload"])
    return MeasuredPinchApproach(ROOT),reset,p

def test_actual_reset_source_computes_distance_at_tick_two_without_steps(case):
    reader,reset,p=case
    result=reader.capture_reset(reset,p)
    assert result["reset_baseline_available"]
    assert result["reset_approach"]["sample_identity"]["physics_step"]==2
    assert result["reset_approach"]["sample_identity"]["rod_tick"]==0
    assert all(v>0 for v in result["reset_approach"]["approach_distance_m"]["value"])
    assert result["new_physics_steps"]==result["ppo_updates"]==0
    assert result["contact_observed"] is None
    assert not result["native_qualification"] and not result["teacher_admission"]
    assert result["source"]["boundary"]=="actual_reset_readback"
    assert result["source"]["pose_api"]=="PhysX get_link_transforms"
    assert result["source"]["reset_body_pose_is_not_authored_seed"]
    assert result["r0_native_reset_complete"] is False

@pytest.mark.parametrize("fault",[
    "status","steps","clock","solver","episode","q","qd","sdk_target","effort","pose",
    "pad_copy","pad_index","pad_path","pad_sdf","hand_names","hand_path","quaternion","radius",
])
def test_modified_actual_reset_readbacks_refused(case,fault):
    reader,r,p=case;r=deepcopy(r);p=deepcopy(p)
    h=r["hands"]["left"];ix=h["joint_names"].index("finger12");bi=h["body_names"].index("finger14_link")
    if fault=="status":r["status"]="failed"
    elif fault=="steps":r["new_sdk_steps"]=1
    elif fault=="clock":r["after_actual_clocks"]["sdk_physics_step"]+=1
    elif fault=="solver":r["after_clock"]["solver_time_s"]+=.01
    elif fault=="episode":r["after_clock"]["episode_uid"]="foreign"
    elif fault=="q":h["q"][0,ix]+=.1
    elif fault=="qd":h["qd"][0,ix]+=.1
    elif fault=="sdk_target":h["backend_targets"][0,ix]+=.1
    elif fault=="effort":h["effort_limits"][0,ix]+=.1
    elif fault=="pose":h["body_poses_xyzw"][0,bi,0]+=.01
    elif fault=="pad_copy":r["pad_poses_xyzw"]["left.finger14_link"][0][0]+=.01
    elif fault=="pad_index":r["rod_reset"]["mesh_bindings"]["left.finger14_link"]["body_index"]+=1
    elif fault=="pad_path":r["rod_reset"]["mesh_bindings"]["left.finger14_link"]["prim_path"]="/foreign"
    elif fault=="pad_sdf":r["rod_reset"]["mesh_bindings"]["left.finger14_link"]["sdf_sha256"]="0"*64
    elif fault=="hand_names":h["body_names"][0]=h["body_names"][1]
    elif fault=="hand_path":h["link_paths"][0][bi]="/foreign"
    elif fault=="quaternion":
        h["body_poses_xyzw"][0,bi,3:]*=2
        r["pad_poses_xyzw"]["left.finger14_link"][1]*=2
    elif fault=="radius":p["reward_radius_sample"]["value"]*=2
    with pytest.raises(ValueError):reader.capture_reset(r,p)

def test_reset_metric_does_not_take_authored_root_pose_as_measured(case):
    reader,r,p=case;r=deepcopy(r)
    # Mutating the authored source also breaks its own locked seed hash.
    r["seed"]["reset_root_pose_w"][0][2]+=.1
    with pytest.raises(ValueError):reader.capture_reset(r,p)

def test_reset_metric_is_owned_and_cannot_be_relabelled_as_other_grasp(case):
    reader,r,p=case
    result=reader.capture_reset(r,p)
    result["reset_approach"]["approach_distance_m"]["value"][0]=0
    assert reader.capture_reset(r,p)["reset_approach"]["approach_distance_m"]["value"][0]>0
    with pytest.raises(ValueError,match="Pinch"):reader.capture_reset(r,p,grasp="tripod")
    with pytest.raises(ValueError,match="Pinch"):reader.capture_reset(r,p,grasp="power")

def test_reset_and_post_source_remove_all_approach_missing_samples(case):
    reader,_,_=case
    result=baseline.replay(approach_reader=reader,include_reset_approach=True)
    mode=result["controls"]["applicability_v2_with_measured_reset_approach"]
    assert [r["terms"]["approach_progress"]["valid_samples"] for r in mode]==[8,8]
    assert all(set(r["blocked_terms"])=={"hold_progress","instability"} for r in mode)
    assert len(result["approach_samples"])==16
    assert result["reset_approach_sample"]["reset_approach"]["sample_identity"]["physics_step"]==2
    assert "measured_reset_approach_distance" not in result["missing_sources"]
    first=result["per_step"][0]["terms"]["approach_progress"]["raw"]
    initial=np.mean(result["reset_approach_sample"]["reset_approach"]["approach_distance_m"]["value"])
    next_value=np.mean(result["approach_samples"][0]["auxiliary"]["samples"]["approach_distance_m"]["value"])
    assert first==pytest.approx(initial-next_value,abs=1e-15)
    assert result["teacher_admission"] is False and result["new_physics_steps"]==0

def test_reset_option_requires_actual_geometry_source():
    with pytest.raises(ValueError):baseline.replay(include_reset_approach=True)
