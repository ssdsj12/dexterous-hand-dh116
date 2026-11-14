import numpy as np
import pytest
from scripts import solve_dh116_tripod_sdf_constrained_direction as m


def parent(side="left"):
    return dict(side=side,independent_style="tripod",independent_seed_sha256=m.TRIPOD_SHA,
        base_pose_wxyz=[.5,.2 if side=="left" else -.2,.4,1,0,0,0],
        active_q_rad=[.3,.4,.6,.8,1.1,1.1])


def test_source_center_never_adds_old_closure_twice_or_uses_pinch():
    p=parent();r=m.decode(p,np.zeros(10))
    assert r["active_q_rad"]==p["active_q_rad"] and r["independent_style"]=="tripod"
    p["independent_style"]="pinch"
    with pytest.raises(ValueError):m.decode(p,np.zeros(10))


def test_two_scale_constrained_model_finds_a_feasible_direction():
    calls=[]
    def evaluate(x):
        calls.append(x.copy())
        return {"constraint_values":[x[0]-.2,.8-x[0]],"constraint_labels":["lower","upper"]}
    r=m.constrained_direction(evaluate,-np.ones(10),np.ones(10))
    assert len(calls)==45
    assert r["jacobian_relative_difference"]<1e-9
    assert r["predicted_common_violation"]==pytest.approx(0)
    assert r["best"]["constraint_values"][0]>=-1e-9


def test_incompatible_linear_bounds_report_slack_not_global_impossibility():
    def evaluate(x):return {"constraint_values":[x[0]-2,-x[0]],"constraint_labels":["a","b"]}
    r=m.constrained_direction(evaluate,-np.ones(10),np.ones(10))
    assert r["predicted_common_violation"]==pytest.approx(1)
    assert r["global_infeasibility_proven"] is False


def test_constraint_identity_change_is_rejected():
    count=0
    def evaluate(x):
        nonlocal count
        count+=1
        return {"constraint_values":[0.],"constraint_labels":["a" if count==1 else "b"]}
    with pytest.raises(ValueError):m.constrained_direction(evaluate,-np.ones(10),np.ones(10))


def test_nonfinite_constraints_rejected_before_solver():
    with pytest.raises(ValueError):
        m.constrained_direction(lambda x:{"constraint_values":[np.nan],"constraint_labels":["a"]},-np.ones(10),np.ones(10))


def test_runtime_floor_cannot_be_replaced_by_pinch_floor():
    assert m.RUNTIME_FLOOR==.0021
    assert m.qualified_constraints([.002-.0021]) is False
    assert m.qualified_constraints([0.,.001]) is True


@pytest.mark.parametrize("x",[[np.nan]*10,[1.01]+[0.]*9])
def test_unbounded_or_nonfinite_pose_delta_rejected(x):
    with pytest.raises(ValueError):m.decode(parent(),x)


def test_missing_original_parent_sha_refused(tmp_path):
    p=tmp_path/"r.json";p.write_text("{}")
    with pytest.raises(ValueError):m.load_parent(p)


def test_output_exclusive(tmp_path):
    with pytest.raises(FileExistsError):m.main(["--output",str(tmp_path)])


def test_unstable_two_scale_model_never_calls_lp_or_backtracks():
    calls=[]
    def measure(x):
        calls.append(1)
        return {"constraint_values":[x[0]**3-1],"constraint_labels":["cubic"]}
    r=m.constrained_direction(measure,-np.ones(10),np.ones(10))
    assert len(calls)==41 and r["linear_solver_calls"]==0 and not r["trials"]


def test_partial_setup_preserves_denied_status(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(m,"load_parent",lambda:{})
    monkeypatch.setattr(m,"checked_json",lambda *a:{})
    monkeypatch.setattr(m,"load_geometry",lambda *a,**k:(_ for _ in ()).throw(m.BudgetExpired()))
    out=tmp_path/"partial"
    assert m.main(["--output",str(out)])==0
    r=json.loads((out/"result.json").read_text())
    assert r["status"]=="partial_timeout" and not r["native_ready"]
    assert not r["reset_proposal_created"] and r["evaluation_records"]==[]
