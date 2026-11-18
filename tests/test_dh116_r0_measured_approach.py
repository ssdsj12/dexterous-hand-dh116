"""Actual saved observations replayed on CPU; never live Isaac stepping."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from go2_pvcnn.control.dh116.tabletop_r0_measured_approach import MeasuredPinchApproach,positive_pad_mean
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from scripts import replay_dh116_native_parameter_r0 as baseline

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def reader():
    return MeasuredPinchApproach(ROOT)

@pytest.fixture(scope="module")
def actual():
    receipt=json.loads(baseline.read_pinned(baseline.RECEIPT,baseline.RECEIPT_SHA))
    root=Path(receipt["output"])
    rows=[decode_owned(json.loads(l)) for l in baseline.read_pinned(
        root/"physical.jsonl",receipt["artifact_sha256"]["physical.jsonl"]).splitlines()]
    return next(r["payload"]["exchange"] for r in rows if r.get("operation")=="physical")

def test_actual_source_only_adds_approach_and_never_contact(reader,actual):
    result=reader.capture(actual)
    assert set(result["auxiliary"]["samples"])=={"approach_distance_m"}
    values=result["auxiliary"]["samples"]["approach_distance_m"]["value"]
    assert values==pytest.approx([.0018339527305,.0021606807131],abs=1e-10)
    assert result["source"]["uncertainty_m"] is None
    assert result["contact_observed"] is None and not result["native_qualification"]
    assert result["contact_inferred_from_geometry"] is False
    assert result["new_physics_steps"]==result["ppo_updates"]==0
    assert result["auxiliary"]["identity"]["physics_step"]==3
    assert result["metric"]=="mean_positive_gap_over_required_sampled_nodes"

@pytest.mark.parametrize("fault",["saved_gap","identity","pose","nodes","radius_source","radius_value","derived"])
def test_stale_or_changed_evidence_is_refused(reader,actual,fault):
    e=deepcopy(actual)
    if fault=="saved_gap":e["closure_geometry"]["sides"]["left"]["required_pad_sdf"]["finger14_link"]["minimum_gap_m"]=0
    elif fault=="identity":e["material_director"]["identity"]["physics_step"]+=1
    elif fault=="pose":e["closure_geometry"]["raw"]["hands"]["left"]["link_poses_xyzw"][0,0,0]+=.01
    elif fault=="nodes":e["rod"]["node_positions"][0,0,0]+=.01
    elif fault=="radius_source":e["material_director"]["reward_radius_sample"]["source"]["hardware_measured"]=True
    elif fault=="radius_value":e["material_director"]["reward_radius_sample"]["value"]*=2
    elif fault=="derived":e["material_director"]["native_parameters"]["fixed_node_indices"]=[0]
    with pytest.raises(ValueError):reader.capture(e)

def row(g):
    return dict(minimum_gap_m=g,gradient_norm_at_minimum=1. if g is not None else None,
                in_lattice_nodes=[2] if g is not None else [])

def test_negative_gap_is_only_zero_approach_not_true_contact():
    assert positive_pad_mean([row(-.002),row(.004)])==pytest.approx(.002)

@pytest.mark.parametrize("rows",[[row(.002),row(None)],[],[row(float("nan")),row(.002)]])
def test_missing_pad_coverage_never_averages_only_remaining_pads(rows):
    assert positive_pad_mean(rows) is None

def test_no_blind_tripod_or_power_use(reader,actual):
    with pytest.raises(ValueError,match="Pinch"):reader.capture(actual,grasp="tripod")
    with pytest.raises(ValueError,match="Pinch"):reader.capture(actual,grasp="power")

def test_output_mutation_does_not_modify_next_readback(reader,actual):
    result=reader.capture(actual);result["auxiliary"]["samples"]["approach_distance_m"]["value"][0]=0
    assert reader.capture(actual)["auxiliary"]["samples"]["approach_distance_m"]["value"][0]>0

def test_same_native_cohort_adds_fifteen_deltas_without_borrowing_post_step_as_reset(reader):
    result=baseline.replay(approach_reader=reader)
    assert len(result["approach_samples"])==16
    original=result["controls"]["original"]
    previous=result["controls"]["applicability_v2"]
    actual=result["controls"]["applicability_v2_with_measured_approach"]
    assert [r["terms"]["approach_progress"]["valid_samples"] for r in original]==[0,0]
    assert [r["terms"]["approach_progress"]["valid_samples"] for r in previous]==[0,0]
    assert [r["terms"]["approach_progress"]["valid_samples"] for r in actual]==[7,8]
    assert actual[0]["terms"]["approach_progress"]["raw_sum"] is None
    assert set(actual[0]["blocked_terms"])=={"approach_progress","hold_progress","instability"}
    assert set(actual[1]["blocked_terms"])=={"hold_progress","instability"}
    assert all(r["diagnostic_reward"] is None for r in actual)
    assert result["new_physics_steps"]==result["optimizer_updates"]==0
    assert result["native_qualification"] is result["teacher_admission"] is False
    assert result["raw_hand_rod_pair_samples"]==0
    assert "measured_reset_approach_distance" in result["missing_sources"]
