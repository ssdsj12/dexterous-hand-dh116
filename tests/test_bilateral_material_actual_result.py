"""Fixed real endpoint evidence, not additional native trials."""
import hashlib,json
from pathlib import Path
import pytest

P=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-material-endpoints-20261114-run03/result.json")
SHA="b96648b2a712aae6ef036d2acdf63d396393bda313f20853414e84373bcc35a5"
@pytest.fixture(scope="module")
def result():
    raw=P.read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA
    return json.loads(raw)

def test_all_same_endpoint_candidates_retained_and_unaligned_substeps_explicit(result):
    assert result["status"]=="complete"
    assert result["endpoint_rows"]==232 and result["final_substep_pair_count"]==178
    assert result["pair_partition_reasons"]=={
        "table_outside_hand_material_scope":109870,"no_same_substep_director":1933}
    assert result["no_candidate_dropped_for_large_model_residual"]
    assert sum(v["above_original_0p5mm_model_residual"] for v in result["by_pad"].values())==124
    assert not result["collision_surface_model_equivalence_verified"]

def test_worst_actual_correspondence_is_same_episode_right_index_and_action_boundary(result):
    row=result["worst_model_residual"];identity=row["identity"]
    assert row["pad"]=="right.finger23_link" and row["node_index"]==3
    assert identity["env_id"]==0 and identity["physics_step"]==234
    assert identity["policy_step"]==28 and identity["substep"]==7
    assert identity["timestamp_s"]==pytest.approx(.4875)
    assert row["mapping"]["surface_model_candidate_residual_m"]==pytest.approx(.0013391644216668237)
    assert identity["episode_uid"]=="dual-pinch-76ef2a55c15642c8a40a405096485601:one"

def test_fixed_right_thumb_drift_cannot_be_called_qualified_slip(result):
    row=result["tracks"]["right.finger14_link:4"]
    assert row["samples"]==45 and row["maximum"]["value_m"]==pytest.approx(.0006406565177586129)
    assert row["maximum"]["identity"]["physics_step"]==234
    assert not result["reference_reselected_on_contact_loss"]
    assert result["new_physics_steps"]==result["ppo_updates"]==0
    assert result["r0_tangential_slip_m"] is None and result["physical_error_bound_m"] is None
    assert not result["teacher_admission"] and not result["native_qualification"]
