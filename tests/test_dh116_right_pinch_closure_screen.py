"""Bounded right-side source geometry screen tests; no actual geometry run."""
from copy import deepcopy
import numpy as np
import pytest
from scripts import screen_dh116_right_pinch_closure as screen


def inputs():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    low,high=active_limits()
    seeds=[dict(seed_id=name,source_sha256=str(i)*64,base_pose_wxyz=[.4,-.2,.4,1,0,0,0],
                active_q_rad=((low+high)/2).tolist())
           for i,name in enumerate(("original_right","derived_right"),1)]
    geometry=dict(side="right")
    return geometry,seeds


def correction(geometry,pose,q):
    return dict(base_pose_wxyz=list(pose),vertical_shift_m=.003,
                original_minimum_clearance_m=-.001,minimum_clearance_m=.002)


def test_exact_fixed_450_grid_interleaves_independent_right_seeds_without_mirroring(monkeypatch):
    geometry,seeds=inputs();saved=deepcopy(seeds)
    monkeypatch.setattr(screen,"clearance_corrected_pose",correction)
    rows=list(screen.candidates(geometry,seeds))
    assert len(rows)==450 and len({r["candidate_id"] for r in rows})==450
    assert [r["seed_id"] for r in rows[:2]]==["original_right","derived_right"]
    assert all(r["side"]=="right" and r["base_pose_wxyz"][1]<0 for r in rows)
    assert rows[0]["active_q_rad"]==seeds[0]["active_q_rad"]
    assert rows[0]["base_pose_wxyz"]==seeds[0]["base_pose_wxyz"]
    assert saved==seeds
    with pytest.raises(ValueError,match="right"):
        list(screen.candidates(dict(side="left"),seeds))


def test_partial_budget_preserves_completed_identity_and_never_claims_full_path(monkeypatch):
    geometry,seeds=inputs()
    monkeypatch.setattr(screen,"clearance_corrected_pose",correction)
    calls=[]
    def measure(g,c,channels):
        assert channels==(1,2)
        calls.append(c["candidate_id"])
        if len(calls)==3:raise screen.BudgetExpired()
        return dict(c,coarse_geometry_passed=False,coarse_blockers=["fixture"],
                    max_required_endpoint_abs_gap_m=.004,selected_cooked_endpoint={})
    monkeypatch.setattr(screen,"measure_closure_grid",measure)
    saved=[]
    result=screen.screen_grid(geometry,seeds,max_candidates=4,progress=lambda r:saved.append(r))
    assert len(result["rows"])==2 and len(saved)==2
    assert result["status"]=="partial_budget"
    assert result["attempted_candidates"]==3 and result["completed_candidates"]==2
    assert result["remaining_candidate_ids"]==result["candidate_ids"][2:]
    assert result["path_checks"]==[] and result["geometry_preparation_passed"] is False
    assert result["native_ready"] is False and result["optimizer_updates"]==0


def test_only_best_coarse_candidate_gets_existing_path_gate_and_cannot_grant_native(monkeypatch):
    geometry,seeds=inputs()
    monkeypatch.setattr(screen,"clearance_corrected_pose",correction)
    def measure(g,c,channels):
        return dict(c,coarse_geometry_passed=True,coarse_blockers=[],
                    max_required_endpoint_abs_gap_m=.0001,
                    selected_cooked_endpoint={"finger14_link":{"capsule_surface_gap_m":.0001}})
    calls=[]
    def path(g,c):
        calls.append(c["candidate_id"])
        return dict(candidate_id=c["candidate_id"],geometry_preparation_passed=True,
                    native_ready=False,teacher_admitted=False,blockers=[])
    monkeypatch.setattr(screen,"measure_closure_grid",measure)
    monkeypatch.setattr(screen,"verify_candidate_path",path)
    result=screen.screen_grid(geometry,seeds,max_candidates=2)
    assert len(calls)==1 and result["geometry_preparation_passed"] is True
    assert result["native_ready"] is False and result["teacher_admitted"] is False
    assert result["new_physics_steps"]==0 and result["optimizer_updates"]==0


@pytest.mark.parametrize("count",[0,-1,451,True])
def test_candidate_budget_is_strict(count):
    geometry,seeds=inputs()
    with pytest.raises(ValueError,match="budget"):screen.screen_grid(geometry,seeds,max_candidates=count)


def test_path_timeout_keeps_coarse_rows_but_no_partial_path_success(monkeypatch):
    geometry,seeds=inputs()
    monkeypatch.setattr(screen,"clearance_corrected_pose",correction)
    monkeypatch.setattr(screen,"measure_closure_grid",lambda g,c,channels:dict(c,
        coarse_geometry_passed=True,coarse_blockers=[],max_required_endpoint_abs_gap_m=.0001,
        selected_cooked_endpoint={"finger14_link":{"capsule_surface_gap_m":.0001}}))
    def expire(*args):raise screen.BudgetExpired()
    monkeypatch.setattr(screen,"verify_candidate_path",expire)
    result=screen.screen_grid(geometry,seeds,max_candidates=2)
    assert result["completed_candidates"]==2 and result["status"]=="partial_budget"
    assert result["path_checks"]==[] and result["geometry_preparation_passed"] is False
    assert result["unfinished_stage"]=="continuous_path_check"


def test_geometry_failure_is_not_silently_dropped(monkeypatch):
    geometry,seeds=inputs()
    monkeypatch.setattr(screen,"clearance_corrected_pose",correction)
    def fail(*args):raise ValueError("QP did not converge")
    monkeypatch.setattr(screen,"measure_closure_grid",fail)
    result=screen.screen_grid(geometry,seeds,max_candidates=2)
    assert len(result["errors"])==2 and result["completed_candidates"]==0
    assert result["status"]=="complete_with_errors"
    assert result["remaining_candidate_ids"]==[]


def test_interrupt_after_a_completed_row_cannot_put_that_identity_back_in_remaining(monkeypatch):
    geometry,seeds=inputs()
    monkeypatch.setattr(screen,"clearance_corrected_pose",correction)
    monkeypatch.setattr(screen,"measure_closure_grid",lambda g,c,channels:dict(c,
        coarse_geometry_passed=False,coarse_blockers=["fixture"],
        max_required_endpoint_abs_gap_m=.004,selected_cooked_endpoint={}))
    def stop_after_row(row):raise screen.BudgetExpired()
    result=screen.screen_grid(geometry,seeds,max_candidates=2,progress=stop_after_row)
    assert result["completed_candidates"]==1
    assert result["remaining_candidate_ids"]==result["candidate_ids"][1:]


def test_real_deadline_signal_waits_until_row_journal_and_identity_commit(monkeypatch):
    import signal
    geometry,seeds=inputs()
    monkeypatch.setattr(screen,"clearance_corrected_pose",correction)
    monkeypatch.setattr(screen,"measure_closure_grid",lambda g,c,channels:dict(c,
        coarse_geometry_passed=False,coarse_blockers=["fixture"],
        max_required_endpoint_abs_gap_m=.004,selected_cooked_endpoint={}))
    journal=[]
    def deadline(signum,frame):raise screen.BudgetExpired()
    def progress(row):
        signal.raise_signal(signal.SIGALRM)
        journal.append(row["candidate_id"])
    previous=signal.signal(signal.SIGALRM,deadline)
    try:
        result=screen.screen_grid(geometry,seeds,max_candidates=2,progress=progress)
    finally:
        signal.signal(signal.SIGALRM,previous)
    assert result["status"]=="partial_budget"
    assert result["completed_candidates"]==1
    assert journal==result["candidate_ids"][:1]
    assert result["remaining_candidate_ids"]==result["candidate_ids"][1:]
