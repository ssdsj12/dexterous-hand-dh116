import copy,json
import numpy as np
import pytest
from scripts import certify_dh116_902_joint_contact_path as path
from scripts.verify_dh116_supported_distal_transfer import SupportedDistalProbe
from scripts.search_dh116_902_joint_contact import ROOT,sha

CANDIDATE=ROOT/"configs/dh116_tasks/flexible_902_tripod_joint_contact_candidate_v1.json"

def test_execution_identity_cannot_rewrite_the_original_search_source():
    candidate=json.loads(CANDIDATE.read_bytes());source=candidate["receipt_sha256"]
    assert path.checked_execution_binding({"version":"dh116-902-fixed-wrist-open-reset-v1"},
        CANDIDATE,source) is False
    with pytest.raises(ValueError):
        path.checked_execution_binding({"version":"dh116-902-fixed-wrist-open-reset-v1"},CANDIDATE,"b"*64)
    frozen=dict(version="dh116-902-joint-contact-open-reset-v1",
        joint_candidate=dict(path=str(CANDIDATE),sha256=sha(CANDIDATE)),
        screen=dict(receipt_sha256=source),
        reset_config=dict(hands=dict(right=dict(pose_wxyz=candidate["pose_wxyz"]))))
    assert path.checked_execution_binding(frozen,CANDIDATE,"b"*64) is True
    frozen["screen"]["receipt_sha256"]="b"*64
    with pytest.raises(ValueError):path.checked_execution_binding(frozen,CANDIDATE,"b"*64)

def boundary():
    clock=dict(run_uuid="new",env_id=0,episode_uid="new:0",reset_id=1,rod_uid="rod0",
        physics_step=240,rod_tick=2856,timestamp_s=.5,clock_valid=True)
    rod=dict(clock,pair_capture_status="validated_solver_samples",
        hand_wrenches_com_world={s:dict(forces=np.zeros((1,3,3)),torques=np.zeros((1,3,3))) for s in ("left","right")})
    row=dict(rod=rod,preceding_reaction_sdk_step=239,native_joint_state=dict(left={},right={}))
    probe=SupportedDistalProbe.__new__(SupportedDistalProbe);probe.boundary_sdk=240;probe.previous=None
    return probe,row,clock

def test_actual_settled_boundary_primes_first_reaction_without_fake_control_row():
    probe,row,clock=boundary()
    probe.prime_supported_boundary(row,expected_clock=clock)
    assert probe.previous["physics_step"]==240
    assert probe.previous["rod"]["episode_uid"]=="new:0"
    row["rod"]["hand_wrenches_com_world"]["right"]["forces"][0,0,0]=1.
    assert not probe.previous["rod"]["hand_wrenches_com_world"]["right"]["forces"].any()
    with pytest.raises(ValueError):probe.prime_supported_boundary(row,expected_clock=clock)

def test_stale_or_different_episode_cannot_prime_probe():
    probe,row,clock=boundary();row["rod"]["episode_uid"]="old:0"
    with pytest.raises(ValueError):probe.prime_supported_boundary(row,expected_clock=clock)


def test_new_contract_is_strict_and_feedback_tail_cannot_issue_actions():
    from scripts.screen_dh116_902_flexible_contact import independent_entry
    with pytest.raises(ValueError):
        independent_entry("tripod",dict(entry_contract="dh116-902-joint-contact-transfer-v1"))
    from scripts.dh116_902_joint_closure_contract import bounded_schedule
    a=np.zeros((80,24));a[:64,6]=.5
    np.testing.assert_array_equal(bounded_schedule(a),a)
    a[-1,6]=.1
    with pytest.raises(ValueError):bounded_schedule(a)

def test_current_sdk_and_no_hand_contact_are_required_for_priming():
    probe,row,clock=boundary();clock["physics_step"]=2
    with pytest.raises(ValueError):probe.prime_supported_boundary(row,expected_clock=clock)
    probe,row,clock=boundary()
    row["rod"]["rod_mesh_contact_pairs"]={"pairs":[{"mesh_name":"left.finger14_link"}]}
    with pytest.raises(ValueError):probe.prime_supported_boundary(row,expected_clock=clock)


def test_incomplete_or_guarded_native_run_cannot_report_completion(tmp_path):
    from scripts.verify_dh116_902_joint_closure import diagnostic_complete
    assert not diagnostic_complete(tmp_path)
    (tmp_path/"tripod_failure.json").write_text("{}")
    assert not diagnostic_complete(tmp_path)
