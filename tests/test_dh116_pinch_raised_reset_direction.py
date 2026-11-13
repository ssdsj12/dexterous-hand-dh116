import numpy as np
import pytest
from scripts import audit_dh116_pinch_raised_reset_direction as m


def parent(side="left"):
    return dict(side=side,base_pose_wxyz=[.5,.2 if side=="left" else -.2,.4,1.,0.,0.,0.],
        active_q_rad=[.3,.4,.6,1.,1.,1.],active_delta_rad=[0,.025,.025,0,0,0])


def test_only_positive_static_z_and_two_independent_fingers_move():
    p=parent("right");r=m.decode(p,[.5,.1,.2])
    assert r["side"]=="right" and r["base_pose_wxyz"][2]==pytest.approx(.401)
    assert r["base_pose_wxyz"][:2]==p["base_pose_wxyz"][:2]
    np.testing.assert_allclose(r["endpoint_q_rad"],[.3,.43,.635,1.,1.,1.])
    assert p==parent("right") and not r["native_qualification"]


@pytest.mark.parametrize("x",[[-.01,0,0],[1.01,0,0],[0,2,0],[np.nan,0,0]])
def test_pose_and_original_speed_bounds_unchanged(x):
    with pytest.raises(ValueError):m.decode(parent(),x)


def test_global_state_budget_counts_attempt_before_query():
    calls=[]
    b=m.StateBudget(lambda x:calls.append(x),maximum=2)
    b(0);b(1)
    with pytest.raises(m.BudgetLimit):b(2)
    assert calls==[0,1] and b.count==2


def test_two_scale_one_linear_direction_only():
    calls=[]
    def q(x):
        calls.append(x.copy())
        return dict(labels=["z","thumb","index"],values=(x-np.array([.4,.2,.3])).tolist())
    r=m.direction(q,np.zeros(3),np.ones(3))
    assert len(calls)==13 and r["solver_calls"]==1 and r["jacobian_relative_difference"]<1e-8


def test_unstable_two_scale_model_stops_without_candidate():
    r=m.direction(lambda x:dict(labels=["cube"],values=[1e6*x[0]**3-1]),np.zeros(3),np.ones(3))
    assert not r["supported"] and r["solver_calls"]==0


def test_path_lower_bound_uses_nearest_sample_cover():
    result=m.path_lower([.003]*17,.0032)
    assert result==pytest.approx(.0029)


def test_incomplete_path_cannot_claim_clearance():
    with pytest.raises(ValueError):m.path_lower([.003]*16,.0032)


def test_output_exclusive(tmp_path):
    with pytest.raises(FileExistsError):m.main(["--output",str(tmp_path)])


def test_diagnostic_contact_target_is_not_qualification():
    assert m.RUNTIME_FLOOR==.0021
    assert m.CONTACT_TARGET==-0.0001
    assert m.decode(parent(),[0,0,0])["native_qualification"] is False


def test_positive_root_probe_stays_in_static_lift_range():
    seen=[]
    def query(x):
        seen.append(x.copy())
        return dict(labels=["a"],values=[x.sum()-1])
    m.direction(query,np.array([0,-.1,-.1]),np.ones(3))
    assert all(x[0]>=0 for x in seen)


def test_timeout_persists_partial_without_native_or_reset(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(m,"run",lambda *a:(_ for _ in ()).throw(m.BudgetExpired()))
    out=tmp_path/"partial";assert m.main(["--output",str(out)])==0
    r=json.loads((out/"result.json").read_text())
    assert r["status"]=="partial_timeout" and not r["native_qualification"]
    assert not r["reset_proposal_created"] and r["physics_steps"]==r["training_updates"]==0


def test_changed_constraint_identity_refused_before_solver():
    count=0
    def query(x):
        nonlocal count
        count+=1
        return dict(labels=["a" if count==1 else "b"],values=[0.])
    with pytest.raises(ValueError):m.direction(query,np.zeros(3),np.ones(3))
