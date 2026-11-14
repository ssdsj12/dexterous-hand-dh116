from copy import deepcopy
import numpy as np
import pytest
from scripts import refine_dh116_right_pinch_geometry as target


def seed():
    return dict(candidate_id=target.PARENT_ID,side="right",active_q_rad=[.48,.037,.68,.275,.56,.25],
        base_pose_wxyz=[.54,-.235,.39,1.,0.,0.,0.],independent_seed_sha256="a"*64)


def geometry():
    return dict(side="right",reference=dict(reference_layout=dict(
        table_center=[.6,0,.25],table_size=[.8,1.2,.1],wire_points=[[.43,-.25,.308],[.43,.25,.308]])),
        old_candidate=dict(rod=dict(radius_m=.005,pinned_node_ids=[],gravity_world_m_s2=[0,0,-9.81])))


def task():
    return dict(scene=dict(table_top_m=.3,table_center_m=[.6,0,.25],table_size_m=[.8,1.2,.1]),
        rod=dict(pinned_node_ids=[],radius_m=None,gravity_world_m_s2=None),
        grasps=dict(pinch=dict(native_reset_source_sha256=None)),reward=dict(coefficient_source=None))


def test_targeted_grid_is_120_right_only_small_local_changes_and_keeps_seed():
    parent=seed();saved=deepcopy(parent)
    rows=list(target.candidates(parent))
    assert len(rows)==120 and len({r["candidate_id"] for r in rows})==120
    assert rows[0]["base_pose_wxyz"]==parent["base_pose_wxyz"]
    assert rows[0]["active_q_rad"]==parent["active_q_rad"]
    for row in rows:
        delta=np.array(row["base_pose_wxyz"])-np.array(parent["base_pose_wxyz"])
        assert delta[1]==0 and np.max(abs(delta[:3]))<=.000100000001
        assert np.array_equal(delta[3:],np.zeros(4))
        assert np.max(abs(np.array(row["active_q_rad"])-parent["active_q_rad"]))<=.000500000001
        assert row["side"]=="right" and row["parent_candidate_id"]==target.PARENT_ID
    assert parent==saved


def test_supported_seed_is_explicit_free_proposal_not_silent_reset_alignment():
    g=geometry();old=deepcopy(g)
    prepared,record=target.supported_geometry(g,task())
    assert g==old
    assert np.allclose(np.array(prepared["reference"]["reference_layout"]["wire_points"])[:,2],.305)
    assert record["original_bottom_gap_m"]==pytest.approx(.003)
    assert record["nominal_bottom_gap_m"]==pytest.approx(0)
    assert record["pinned_node_ids"]==[] and record["fix_root_orientation"] is False
    assert record["current_registered_reset_seed_matches"] is False
    assert record["task_native_reset_source_sha256"] is None
    assert record["native_table_support_verified"] is False


@pytest.mark.parametrize("edit",[
    lambda g,t:g["old_candidate"]["rod"].update(pinned_node_ids=[0]),
    lambda g,t:t["rod"].update(pinned_node_ids=[0]),
    lambda g,t:t["scene"].update(table_top_m=.31),
])
def test_unsupported_task_or_pinned_rod_is_rejected(edit):
    g,t=geometry(),task();edit(g,t)
    with pytest.raises(ValueError):target.supported_geometry(g,t)


def test_strict_endpoint_gate_cannot_accept_existing_coarse_half_millimetre(monkeypatch):
    monkeypatch.setattr(target,"measure_closure_grid",lambda g,c,ch:dict(c,
        coarse_geometry_passed=True,max_required_endpoint_abs_gap_m=.0001))
    result=target.measure_strict({},next(target.candidates(seed())))
    assert result["strict_endpoint_passed"] is False
    assert result["strict_endpoint_tolerance_m"]==.00005


def test_full_path_failure_or_missing_check_never_grants_geometry(monkeypatch):
    rows=[dict(candidate_id="a",strict_endpoint_passed=True,finger13_endpoint_gap_m=.0002)]
    monkeypatch.setattr(target,"verify_candidate_path",lambda g,c:dict(
        candidate_id=c["candidate_id"],geometry_preparation_passed=False,blockers=["fixture"]))
    checks,passed=target.check_paths({},rows,maximum=1)
    assert len(checks)==1 and passed is None
    assert target.check_paths({},[],maximum=1)==([],None)


def test_path_budget_stops_at_first_full_pass_without_testing_extra_candidates(monkeypatch):
    rows=[dict(candidate_id=str(i),strict_endpoint_passed=True,finger13_endpoint_gap_m=.0002-i*.00001)
          for i in range(5)]
    calls=[]
    def check(g,c):
        calls.append(c["candidate_id"])
        return dict(candidate_id=c["candidate_id"],geometry_preparation_passed=True,blockers=[])
    monkeypatch.setattr(target,"verify_candidate_path",check)
    checks,passed=target.check_paths({},rows,maximum=3)
    assert calls==["0"] and passed=="0"


@pytest.mark.parametrize("maximum",[0,-1,4,True])
def test_complete_path_budget_is_bounded(maximum):
    with pytest.raises(ValueError,match="path"):
        target.check_paths({},[],maximum=maximum)


def test_tampered_parent_receipt_is_rejected_before_geometry(tmp_path):
    receipt=tmp_path/"parent.json";receipt.write_text("{}")
    with pytest.raises(ValueError):target.load_parent(receipt,tmp_path,{})


def test_target_never_uses_a_left_hand_parent():
    parent=seed();parent["side"]="left"
    with pytest.raises(ValueError,match="right"):list(target.candidates(parent))
