"""Analytic CPU fixtures only; no real source/FK/SDF search in these tests."""
from copy import deepcopy
import json
import numpy as np
import pytest


def functions():
    from scripts import solve_dh116_pinch_trust_region as m
    return m


def query(x):
    return dict(labels=["initial:safe", "endpoint:pad:contact_minus0p1mm"],
                values=[1., float(x[3]-1.)])


def test_two_scale_fd_uses21_queries_and_exact_current_point():
    m=functions(); calls=[]
    x=np.array([.2,1.,1.,.1,.2]); lo=np.array([0.,-1.,-1.,-.5,-.5]); hi=np.ones(5)
    def q(p):
        assert np.all(p>=lo) and np.all(p<=hi)
        calls.append(p.copy())
        return dict(labels=["linear"],values=[float(np.dot(p,np.arange(1,6)))])
    result=m.local_model(q,x,lo,hi,.2)
    assert len(calls)==21
    np.testing.assert_array_equal(calls[0],x)
    np.testing.assert_allclose(result["jacobians"][1],[np.arange(1,6)],atol=1e-9)
    assert result["supported"]
    assert [s["mode"] for s in result["stencils"][0]]==["central","backward","backward","central","central"]


@pytest.mark.parametrize("x", [np.array([0.,-1.,-1.,-.5,-.5]),np.ones(5)])
def test_fd_never_steps_outside_exact_hard_boundaries(x):
    m=functions(); lo=np.array([0.,-1.,-1.,-.5,-.5]); hi=np.ones(5); seen=[]
    def q(p):
        assert np.all(p>=lo) and np.all(p<=hi);seen.append(p.copy())
        return dict(labels=["quadratic"],values=[float(np.dot(p,p))])
    d=m.local_model(q,x,lo,hi,.2)
    assert len(seen)==21
    np.testing.assert_allclose(d["jacobians"][1],[2*x],atol=1e-10)
    assert all(s["mode"]!="central" for s in d["stencils"][0])


def test_narrow_bounds_shrink_stencil_instead_of_clipping_outside():
    m=functions();lo=np.zeros(5);hi=np.full(5,.0003);x=np.zeros(5)
    def q(p):
        assert np.all(p>=lo) and np.all(p<=hi)
        return dict(labels=["linear"],values=[float(p.sum())])
    d=m.local_model(q,x,lo,hi,.2)
    np.testing.assert_allclose(d["jacobians"][1],np.ones((1,5)))
    assert max(s["step"] for s in d["stencils"][0])<=.00015


def test_two_scale_disagreement_refuses_lp():
    m=functions()
    q=lambda x:dict(labels=["endpoint:pad:contact_minus0p1mm"],values=[1e6*x[0]**3-1])
    model=m.local_model(q,np.zeros(5),-np.ones(5),np.ones(5),.2)
    assert not model["supported"]


def test_actual_improvement_and_hard_constraints_required():
    m=functions(); labels=["initial:safe","endpoint:pad:contact_minus0p1mm"]
    assert m.assess(labels,[1,-1],[1,-.5],[1,-.6])["accepted"]
    for actual in ([1,-1.1],[-.01,-.2],[1,-.99]):
        assert not m.assess(labels,[1,-1],[1,-.5],actual)["accepted"]
    assert not m.assess(labels,[1,-1],[1,-1],[1,-.5])["accepted"]


def test_nonlinear_rejection_shrinks_trust_and_keeps_start():
    m=functions()
    q=lambda x:dict(labels=["initial:safe","endpoint:pad:contact_minus0p1mm"],
                    values=[1.,float(x[3]-1-100*x[3]**2)])
    trace=[]
    out=m.refine(q,np.zeros(5),-np.ones(5),np.ones(5),trace)
    assert len(trace)==3 and out["accepted_directions"]==0
    np.testing.assert_array_equal(out["selected_x"],np.zeros(5))
    assert [t["trust_radius_before"] for t in trace]==[.2,.1,.05]
    assert all(t["solver_calls"]==1 and "actual" in t and not t["accepted"] for t in trace)


def test_acceptance_uses_nonlinear_trial_then_advances_center():
    m=functions();trace=[];calls=[]
    def tracked(x):
        calls.append(x.copy());return query(x)
    out=m.refine(tracked,np.zeros(5),-np.ones(5),np.ones(5),trace)
    assert len(calls)==3*(21+1)
    assert out["accepted_directions"]==3 and len(trace)==3
    assert out["selected_x"][3]==pytest.approx(.6)
    assert all(t["assessment"]["ratio"]==pytest.approx(1.) for t in trace)
    assert trace[1]["center_x"][3]==pytest.approx(.2)


def test_candidate_hard_constraint_regression_cannot_replace_safe_start():
    m=functions();trace=[]
    q=lambda x:dict(labels=["initial:safe","endpoint:pad:contact_minus0p1mm"],
                    values=[float(.001-100*x[3]**2),float(x[3]-1)])
    out=m.refine(q,np.zeros(5),-np.ones(5),np.ones(5),trace)
    assert out["accepted_directions"]==0
    assert all(not t["assessment"]["hard_constraints_preserved"] for t in trace)


def test_constraint_identity_change_is_rejected():
    m=functions();n=[0]
    def q(x):
        n[0]+=1;return dict(labels=["a" if n[0]==1 else "b"],values=[0.])
    with pytest.raises(ValueError,match="order"):
        m.local_model(q,np.zeros(5),-np.ones(5),np.ones(5),.2)


def test_shared_geometry_cap_counts_failed_attempts_and_refuses353():
    m=functions();calls=[];records=[]
    budget=m.StateBudget(records,maximum=352)
    def measure(x):calls.append(x);return {"x":x}
    for i in range(351):budget.call(measure,i,side="left",phase="fixture")
    with pytest.raises(ValueError):
        budget.call(lambda x:(_ for _ in ()).throw(ValueError("invalid SDF")),351,side="right",phase="trial")
    with pytest.raises(m.BudgetLimit):
        budget.call(measure,352,side="right",phase="path")
    assert budget.count==len(records)==352 and len(calls)==351
    assert records[-1]["status"]=="error" and "invalid SDF" in records[-1]["error"]


def test_declared_geometry_budget_and_one_lp_per_direction():
    m=functions()
    assert m.MAX_DIRECTIONS==3 and m.FD_QUERY_POINTS==21 and m.MAX_STATE_ATTEMPTS==352
    assert m.WORST_CASE_STATES==2*(3*(21*2+2)+33)==330


def test_partial_timeout_preserves_in_progress_attempt_and_has_no_native(tmp_path,monkeypatch):
    m=functions()
    def interrupted(project,result):
        result.update(state_records=[dict(status="interrupted")],sides={"left":{"directions":[{"status":"evaluating"}]}})
        raise m.BudgetExpired()
    monkeypatch.setattr(m,"run",interrupted)
    out=tmp_path/"partial";assert m.main(["--output",str(out)])==0
    saved=json.loads((out/"result.json").read_text())
    assert saved["status"]=="partial_timeout" and saved["state_records"][0]["status"]=="interrupted"
    assert not saved["reset_proposal_created"] and not saved["native_qualification"]
    assert saved["physics_steps"]==saved["training_updates"]==0


def test_output_is_exclusive_before_any_geometry(tmp_path,monkeypatch):
    m=functions();monkeypatch.setattr(m,"run",lambda *a:pytest.fail("must not run"))
    with pytest.raises(FileExistsError):m.main(["--output",str(tmp_path)])


def test_seed_coordinates_are_relative_to_original_parent():
    m=functions()
    parent=dict(side="right",base_pose_wxyz=[.5,-.2,.4,1.,0.,0.,0.],
                active_q_rad=[.3,.4,.6,1.,1.,1.],active_delta_rad=[0,.025,.025,0,0,0])
    seed=np.array([.1,1,1,.1,.2])
    a=m.previous.decode(parent,seed)
    assert a["base_pose_wxyz"][1]==pytest.approx(-.197)
    assert a["active_q_rad"][0]==pytest.approx(.36)
    lo,hi=m.previous.bounds(parent)
    assert hi[1]==hi[2]==1
    assert not m.previous.decode(parent,seed)["native_qualification"]



def test_actual_budget_retains_interrupted_state():
    m=functions();records=[];budget=m.StateBudget(records)
    with pytest.raises(m.BudgetExpired):
        budget.call(lambda x:(_ for _ in ()).throw(m.BudgetExpired()),None,
                    side="left",phase="trial",parameter_x=[0]*5)
    assert budget.count==1 and records[0]["status"]=="interrupted"
    assert records[0]["parameter_x"]==[0]*5


def test_failed_lp_does_not_evaluate_or_accept_a_candidate(monkeypatch):
    from types import SimpleNamespace
    m=functions();calls=[];solves=[]
    def failed(*args,**kwargs):
        solves.append(1);return SimpleNamespace(success=False,status=2,message="fixture infeasible")
    monkeypatch.setattr(m,"linprog",failed)
    def tracked(x):
        calls.append(x.copy());return query(x)
    trace=[];out=m.refine(tracked,np.zeros(5),-np.ones(5),np.ones(5),trace)
    assert len(solves)==3 and len(calls)==3*21
    assert out["accepted_directions"]==0 and all("actual" not in t for t in trace)
