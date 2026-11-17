import json
import numpy as np
import pytest
from scripts import audit_dh116_pinch_spread_axial_direction as m


def parent(side="left"):
    return dict(side=side,base_pose_wxyz=[.5,.2 if side=="left" else -.2,.4,1.,0.,0.,0.],
        active_q_rad=[.3,.4,.6,1.,1.,1.],active_delta_rad=[0,.025,.025,0,0,0])


def test_independent_static_reset_variables_and_original_motion():
    p=parent("right");r=m.decode(p,[.5,-.5,.5,.1,.2])
    assert p==parent("right") and r["side"]=="right"
    np.testing.assert_allclose(r["base_pose_wxyz"],[.5,-.2015,.401,1,0,0,0])
    np.testing.assert_allclose(r["active_q_rad"],[.33,.4,.6,1,1,1])
    np.testing.assert_allclose(r["active_delta_rad"],[0,.03,.035,0,0,0])
    assert not r["native_qualification"] and r["static_reset_only"]


@pytest.mark.parametrize("x",[[-.1,0,0,0,0],[1.1,0,0,0,0],
    [0,1.01,0,0,0],[0,0,-1.01,0,0],[0,0,0,2,0],[0,0,0,0,np.nan]])
def test_bounded_static_reset_and_original_joint_velocity_guard(x):
    with pytest.raises(ValueError):m.decode(parent(),x)


def test_joint_limits_consider_shifted_initial_spread():
    p=parent();p["active_q_rad"][0]=m.active_limits()[1][0]-.001
    with pytest.raises(ValueError):m.decode(p,[0,0,1,0,0])
    lo,hi=m.bounds(p)
    assert hi[2]==pytest.approx(.001/.06)


def test_single_direction_exact_two_scale_probe_count():
    calls=[]
    def query(x):
        calls.append(x.copy())
        return dict(labels=["q"+str(i) for i in range(5)],values=(x-.1).tolist())
    r=m.direction(query,np.array([0,-1,-1,-.5,-.5]),np.ones(5))
    assert len(calls)==21 and r["solver_calls"]==1
    assert r["jacobian_relative_difference"]<1e-7
    assert all(x[0]>=0 for x in calls)


def test_unstable_local_model_does_not_generate_candidate():
    r=m.direction(lambda x:dict(labels=["cubic"],values=[1e6*x[0]**3-1]),
        np.array([0,-1,-1,-1,-1]),np.ones(5))
    assert not r["supported"] and r["solver_calls"]==0


def test_constraint_order_change_refused():
    calls=[]
    def q(x):
        calls.append(1)
        return dict(labels=["a" if len(calls)==1 else "b"],values=[0])
    with pytest.raises(ValueError):m.direction(q,np.zeros(5),np.ones(5))


def test_every_initial_and_endpoint_state_consumes_shared_budget():
    seen=[];b=m.StateBudget(lambda x:seen.append(x),maximum=3)
    b(1);b(2);b(3)
    with pytest.raises(m.BudgetLimit):b(4)
    assert seen==[1,2,3] and b.count==3


@pytest.mark.parametrize("n",[0,161,1.5,True])
def test_budget_bounds(n):
    with pytest.raises(ValueError):m.StateBudget(lambda x:x,maximum=n)


def test_initial_constraints_include_required_pads_and_table():
    state=dict(table_m={"source:finger14_link":.0026},
        all_shape_rod_lower_m={"source:finger14_link":-.0001})
    labels,values=m.initial_constraints(state)
    assert labels==["initial:source:finger14_link:table_design_2p5mm",
                   "initial:source:finger14_link:rod_clear_0p1mm"]
    np.testing.assert_allclose(values,[.1,-.2])


def test_continuous_cover_uses_33_complete_states():
    assert m.path_lower([.003]*33,.0032)==pytest.approx(.00295)
    with pytest.raises(ValueError):m.path_lower([.003]*32,.0032)


def test_failed_bound_is_not_proven_collision_or_contact():
    result=m.gate(.00209,.0002,True,True)
    assert not result["static_candidate_verified"]
    assert not result["native_qualification"] and not result["global_infeasibility_proven"]


def test_contact_requires_both_sdfs_not_one():
    assert not m.gate(.003,.001,True,False)["static_candidate_verified"]
    assert m.gate(.003,.001,True,True)["static_candidate_verified"]
    assert not m.gate(.003,.001,True,True)["teacher_admission"]


def test_exclusive_output(tmp_path):
    with pytest.raises(FileExistsError):m.main(["--output",str(tmp_path)])


def test_timeout_keeps_partial_and_never_outputs_native_recipe(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"run",lambda *a:(_ for _ in ()).throw(m.BudgetExpired()))
    p=tmp_path/"partial";assert m.main(["--output",str(p)])==0
    r=json.loads((p/"result.json").read_text())
    assert r["status"]=="partial_timeout" and not r["reset_proposal_created"]
    assert r["physics_steps"]==r["training_updates"]==0


def test_original_clearance_and_speed_are_preserved():
    assert m.RUNTIME_FLOOR==.0021
    d=m.decode(parent(),[0,0,0,1.5,1.5])
    assert max(d["active_delta_rad"])==pytest.approx(.1)
