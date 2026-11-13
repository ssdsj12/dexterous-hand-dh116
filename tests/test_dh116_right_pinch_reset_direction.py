import numpy as np
import pytest

def m():
    from scripts import solve_dh116_right_pinch_reset_direction as s
    return s

def test_zero_reset_has_no_unrequested_joint_or_pose_change():
    s=m();q=np.array([.4,.2,.6,.3,.5,.3]);base=np.array([.04,.02])
    out=s.decode(q,base,np.zeros(7))
    np.testing.assert_array_equal(out["q0"],q)
    np.testing.assert_allclose(out["delta"],[0,.04,.02,0,0,0])
    assert out["roll_rad"]==0

@pytest.mark.parametrize("x",[np.ones(7)*1.001,np.array([np.nan,0,0,0,0,0,0]),np.zeros(3)])
def test_decode_refuses_outside_finite_trust_box(x):
    with pytest.raises(ValueError):m().decode(np.array([.4,.2,.6,.3,.5,.3]),np.array([.04,.02]),x)

def test_cumulative_motion_limits_are_not_clipped():
    with pytest.raises(ValueError):m().decode(np.array([.4,.2,.6,.3,.5,.3]),np.array([.09,.02]),np.array([0,1,0,0,0,0,0]))

def test_rotation_is_fixed_at_rod_axis_and_preserves_distances():
    s=m();axis=np.array([.43,0,.305]);t=s.reset_rotation(axis,.1)
    np.testing.assert_allclose(t@np.r_[axis,1],np.r_[axis,1],atol=1e-15)
    np.testing.assert_allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-15)
    assert np.linalg.det(t[:3,:3])==pytest.approx(1)

def test_constraint_direction_respects_bounds_and_rechecks_nonlinear_values():
    s=m();calls=[]
    def f(x):
        calls.append(x.copy());return np.array([x[0]-.2,.5-x[0]])
    r=s.local_pass(f,np.zeros(4),-np.ones(4),np.ones(4))
    assert r["best_violation"]==0 and 0.2<=r["best_x"][0]<=.5
    assert len(calls)<=21


def test_infeasible_local_linear_model_does_not_report_feasible_geometry():
    s=m()
    r=s.local_pass(lambda x:np.array([-.3-x[0],x[0]-.3]),np.zeros(4),-np.ones(4),np.ones(4))
    assert r["best_violation"]>0 and r["global_infeasibility_proven"] is False


def test_no_native_execution_or_teacher_qualification_in_cpu_gate():
    s=m();r=s.final_gate([.0022],[.00001],[-.0002,-.0002],-.8)
    assert r["static_path_passed"] and not r["native_ready"] and not r["teacher_admission"]
    assert not s.final_gate([.002099],[.00001],[-.0002,-.0002],-.8)["static_path_passed"]
    assert not s.final_gate([.0022],[-1e-8],[-.0002,-.0002],-.8)["static_path_passed"]
    assert not s.final_gate([.0022],[.00001],[-.00031,-.0002],-.8)["static_path_passed"]


def test_static_reset_translation_is_explicit_and_never_a_closure_action():
    s=m();q=np.array([.4,.2,.6,.3,.5,.3]);base=np.array([.04,.02])
    x=np.array([0,0,0,0,.5,-.25,0]);out=s.decode(q,base,x)
    np.testing.assert_allclose(out["reset_translation_world_m"],[.0005,0,-.00025])
    np.testing.assert_array_equal(out["q0"],q)
    np.testing.assert_allclose(out["delta"],[0,.04,.02,0,0,0])
    t=s.reset_transform(np.array([.43,0,.305]),out)
    np.testing.assert_allclose(t[:3,:3],np.eye(3))
    np.testing.assert_allclose(t[:3,3],[.0005,0,-.00025])

def test_six_variable_fd_records_actual_measurement_budget():
    s=m();calls=[]
    def f(x):
        calls.append(x.copy());return np.r_[x-.1,.4-x]
    r=s.local_pass(f,np.zeros(6),-np.ones(6),np.ones(6))
    assert r["best_violation"]==0
    assert len(calls)==29 and r["measurements"]==29
    assert r["finite_difference_steps"]==[.002,.001]

def test_longitudinal_reset_uses_half_node_spacing_without_resampling_rod():
    s=m();q=np.array([.4,.2,.6,.3,.5,.3]);base=np.array([.04,.02])
    x=np.array([0,0,0,0,.5,-.25,1.]);out=s.decode(q,base,x)
    np.testing.assert_allclose(out["reset_translation_world_m"],[.0005,.00625,-.00025])
    t=s.reset_transform(np.array([.43,0,.305]),out)
    np.testing.assert_allclose(t[:3,3],out["reset_translation_world_m"])
    assert "rod_nodes" not in out
    with pytest.raises(ValueError):
        s.reset_transform(np.zeros(3),dict(roll_rad=0,reset_translation_world_m=[0,.006251,0]))

def test_local_trust_radius_bounds_every_nonlinear_trial():
    s=m();calls=[]
    def f(x):
        calls.append(x.copy());return np.array([x[0]-.1,.8-x[0]])
    r=s.local_pass(f,np.zeros(7),-np.ones(7),np.ones(7),trust_radius=.05)
    assert r["trust_radius"]==.05
    assert all(np.max(np.abs(x))<=.05+1e-12 for x in calls)
    assert r["best_violation"]>=.05-1e-10
    assert r["global_infeasibility_proven"] is False
