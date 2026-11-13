"""Direct constrained solve tests are analytic fixtures, not DH116 qualification."""
import json
import numpy as np
import pytest
from types import SimpleNamespace
from scripts import solve_dh116_pinch_direct_sqp as m

def query(x):
    return dict(labels=["initial:clear","endpoint:thumb:contact_minus0p1mm","endpoint:index:contact_minus0p1mm"],
                values=[float(.8-x[0]),float(x[0]-.4),float(x[1]-.3)])

def test_unique_query_cache_and_budget_keep_full_evidence():
    seen=[];rows=[]
    c=m.Queries(lambda x:(seen.append(x.copy()) or query(x)),rows,maximum=2)
    a=c(np.zeros(7));b=c(np.zeros(7))
    assert a is b and len(seen)==len(rows)==1
    c(np.ones(7)*.1)
    with pytest.raises(m.QueryBudget):c(np.ones(7)*.2)
    assert len(seen)==len(rows)==2 and all(x["status"]=="complete" for x in rows)

def test_actual_nonlinear_candidate_meets_constraints_and_contact():
    trace={};out=m.solve(query,np.zeros(7),-np.ones(7),np.ones(7),trace)
    assert out["selected_feasible"] and out["selected_contact_deficit_mm"]<.0001
    assert out["unique_queries"]<=m.MAX_QUERIES
    assert out["selected_x"][0]<=.8 and out["selected_x"][0]>=.3999

def test_optimizer_success_flag_cannot_override_real_hard_failure(monkeypatch):
    def fake(fun,x0,**kw):
        x=np.array(x0);x[0]=.9
        kw["callback"](x)
        return SimpleNamespace(x=x,success=True,status=0,message="fixture lies",nit=1)
    monkeypatch.setattr(m,"minimize",fake)
    out=m.solve(query,np.zeros(7),-np.ones(7),np.ones(7),{})
    assert out["selected_x"]==[0.]*7
    assert out["selected_contact_deficit_mm"]==pytest.approx(.4)

def test_query_error_retained_not_zero_filled():
    rows=[];c=m.Queries(lambda x:(_ for _ in ()).throw(ValueError("missing SDF")),rows,maximum=2)
    with pytest.raises(ValueError):c(np.zeros(7))
    assert len(rows)==1 and rows[0]["status"]=="error"
    assert "values" not in rows[0]

def test_budget_exit_preserves_verified_parent_no_automatic_restart(monkeypatch):
    calls=[]
    def fake(fun,x0,**kw):
        for i in range(4):
            x=np.array(x0);x[5]=i*.01;calls.append(1);fun(x)
        raise AssertionError("cannot reach")
    monkeypatch.setattr(m,"minimize",fake)
    out=m.solve(query,np.zeros(7),-np.ones(7),np.ones(7),{},maximum=2)
    assert out["status"]=="query_budget_exhausted" and len(calls)==3
    assert out["selected_x"]==[0.]*7

def test_failed_state_counts_and_hard_cap():
    rows=[];b=m.StateBudget(rows,maximum=2)
    assert b.call(lambda x:x,1)==1
    with pytest.raises(ValueError):b.call(lambda x:(_ for _ in ()).throw(ValueError("q")),2)
    with pytest.raises(m.QueryBudget):b.call(lambda x:x,3)
    assert b.count==len(rows)==2 and rows[-1]["status"]=="error"

def test_exact_reservations_and_no_hidden_expansion():
    assert m.MAX_QUERIES==200 and m.MAX_ITERATIONS==16
    assert m.MAX_STATES==2*(2*200+33)==866

def test_timeout_preserves_no_native_claim(tmp_path,monkeypatch):
    def run(root,result):
        result["sides"]={"left":{"solver":{"status":"in_progress"}}}
        raise m.BudgetExpired()
    monkeypatch.setattr(m,"run",run)
    assert m.main(["--output",str(tmp_path/"out")])==0
    r=json.loads((tmp_path/"out"/"result.json").read_text())
    assert r["status"]=="partial_timeout" and not r["native_qualification"]
