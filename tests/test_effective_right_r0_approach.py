"""Read-only, pinned failed-budget native samples; no new native credit."""
from pathlib import Path
from copy import deepcopy
import importlib,importlib.util,json
import pytest
from scripts.profile_dh116_complete_pair_r0_ledger import RECEIPT,RECEIPT_SHA,sha
from scripts.validate_dh116_complete_pair_r0 import events
from scripts.replay_dh116_measurement_origin import PHYSICAL_SHA
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope
ROOT=Path(__file__).resolve().parents[1]
MODULE="go2_pvcnn.control.dh116.tabletop_effective_right_r0_approach"
def api():
    assert importlib.util.find_spec(MODULE) is not None,"current geometry approach source missing"
    return importlib.import_module(MODULE)

@pytest.fixture(scope="module")
def actual():
    assert sha(RECEIPT)==RECEIPT_SHA
    p=Path(json.loads(RECEIPT.read_text())["output"])
    assert sha(p/"physical.jsonl")==PHYSICAL_SHA
    reset=first=None
    for r in events(p/"physical.jsonl"):
        if r.get("operation")=="reset":reset=r["payload"]["reset"]
        elif r.get("operation")=="physical":
            first=r["payload"]["exchange"];break
    packet=next(events(p/"director_reset.json"))["payload"]
    return reset,packet,first

def test_legacy_segment_reader_does_not_accept_new_geometry(actual):
    from go2_pvcnn.control.dh116.tabletop_segment_r0_approach import SegmentMeasuredPinchApproach
    with pytest.raises(ValueError,match="source|version|segment"):
        SegmentMeasuredPinchApproach(ROOT).capture(actual[2])

def test_current_geometry_and_actual_reset_can_be_read(actual):
    mod=api()
    with codec_scope():
        reader=mod.EffectiveRightMeasuredApproach(ROOT)
        reset=reader.capture_reset(actual[0],actual[1])
        sample=reader.capture(actual[2])
    assert reset["reset_baseline_available"]
    assert sample["auxiliary"]["identity"]["physics_step"]==actual[2]["physics_step"]
    assert sample["source"]["coordinate_frame"]=="world_to_actual_pad_local"
    assert sample["source"]["geometry_contract_version"]==mod.GEOMETRY_VERSION
    assert sample["contact_observed"] is None and not sample["contact_inferred_from_geometry"]
    assert sample["new_physics_steps"]==0 and not sample["teacher_admission"]

@pytest.mark.parametrize("field",["raw_pose","raw_nodes","radius","source"])
def test_actual_geometry_tamper_is_rejected(actual,field):
    mod=api();x=deepcopy(actual[2])
    if field=="raw_pose":x["closure_geometry"]["raw"]["hands"]["right"]["link_poses_xyzw"][0,0,0]+=.01
    elif field=="raw_nodes":x["closure_geometry"]["raw"]["rod_nodes_m"][0,2]+=.01
    elif field=="radius":x["material_director"]["native_parameters"]["contact_radius_m"]=.1
    else:x["closure_geometry"]["sources"]["effective_right"]["static_path_sha256"]="0"*64
    with codec_scope(),pytest.raises((ValueError,TypeError,KeyError)):
        mod.EffectiveRightMeasuredApproach(ROOT).capture(x)

@pytest.mark.parametrize("grasp",["tripod","power"])
def test_other_grasps_need_their_own_source(actual,grasp):
    mod=api()
    with pytest.raises(ValueError,match="Pinch"):
        mod.EffectiveRightMeasuredApproach(ROOT).capture(actual[2],grasp=grasp)

def test_actual_reset_and_first_control_feed_unchanged_R0(actual):
    import numpy as np
    from scripts import replay_dh116_native_parameter_r0 as radius
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task,R0_TERMS
    from go2_pvcnn.control.dh116.tabletop_r0_measurement_origin import OriginLiveR0Observer
    p=Path(json.loads(RECEIPT.read_text())["output"])
    budget=json.loads((ROOT/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_text())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"]=2
    budget["physics"]["solver_substeps_per_physics_step"]=12
    reader=api().EffectiveRightMeasuredApproach(ROOT)
    measured=radius.check_radius_packet(actual[1],actual[0]["rod_reset"])
    obs=OriginLiveR0Observer(load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",ROOT),
        "pinch",env_id=0,budget_config=budget,diagnostic_coefficients={k:float(i+1) for i,k in enumerate(R0_TERMS)},
        backend_artifact=radius.BACKEND,backend_sha256=radius.BACKEND_SHA,
        radius_sample=dict(value=measured["value"],source="native_parameter_snapshot:"+_digest(measured)))
    with codec_scope():
        initial=reader.capture_reset(actual[0],actual[1])
        first=None
        for entry in events(p/"physical.jsonl"):
            op=entry.get("operation");payload=entry.get("payload")
            if op=="reset":obs.complete_physical_reset(payload,reset_approach=initial["reset_approach"])
            elif op=="begin":obs.begin(**payload)
            elif op=="physical":
                s=reader.capture(payload["exchange"])
                obs.record(payload,auxiliary=s["auxiliary"])
                if first is None:first=s
            elif op=="end":
                result=obs.end(payload);break
    receipt=result["control_receipt"]
    assert receipt["terms"]["approach_progress"]["valid_samples"]==8
    assert "approach_progress" not in receipt["blocked_terms"]
    assert set(receipt["blocked_terms"])=={"hold_progress","material_anchor_slip","instability"}
    assert receipt["training_reward"] is None and not result["teacher_admission"]
    difference=np.mean(initial["reset_approach"]["approach_distance_m"]["value"])-np.mean(first["auxiliary"]["samples"]["approach_distance_m"]["value"])
    assert result["physical_rows"][0]["r0"]["terms"]["approach_progress"]["raw"]==pytest.approx(difference,abs=1e-15)
