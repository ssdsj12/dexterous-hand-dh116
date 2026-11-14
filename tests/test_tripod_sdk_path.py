"""Canonical 24D reference compiler must respect original frames and limits."""
import importlib.util
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

def module():
    name="scripts.audit_dh116_tripod_sdk_path"
    assert importlib.util.find_spec(name) is not None, "original SDK path audit is missing"
    return __import__(name,fromlist=["compile_actions"])

def poses():
    r=Rotation.from_euler("XYZ",[.3,-.2,.7])
    a=np.r_[ [.5,.1,.4], np.roll(r.as_quat(),1)]
    b=a.copy();b[:3]+=[.001,-.0002,.0003]
    b[3:]=np.roll((r*Rotation.from_euler("XYZ",[.012,-.017,.02])).as_quat(),1)
    return a,b

def test_original_intrinsic_order_round_trips_and_is_not_world_euler():
    m=module();a,b=poses()
    angles=m.relative_intrinsic(a,b)
    assert angles==pytest.approx([.012,-.017,.02],abs=1e-13)
    world=(Rotation.from_quat(b[[4,5,6,3]])*Rotation.from_quat(a[[4,5,6,3]]).inv()).as_euler("XYZ")
    assert np.linalg.norm(world-angles)>.001

@pytest.mark.parametrize("side,other",[("left",list(range(6,12))+list(range(15,18))+list(range(21,24))),
                                        ("right",list(range(6))+list(range(12,15))+list(range(18,21)))])
def test_action_is_24d_with_inactive_hand_zero_and_original_rates(side,other):
    m=module();a,b=poses();q=np.array([.3,.4,.5,.6,.7,.8]);q1=q+.01
    p=m.compile_actions(a,b,q,q1,side=side,physics_steps=80,mode="intrinsic_linear")
    actions=np.array(p["actions"])
    assert actions.shape==(10,24)
    assert np.array_equal(actions[:,other],np.zeros((10,12)))
    i=0 if side=="left" else 1
    assert actions[:,i*6:i*6+6].mean(0)==pytest.approx((q1-q)/(80/480*.3))
    assert actions[:,12+i*3:15+i*3].mean(0)==pytest.approx((b[:3]-a[:3])/(80/480*.01))

def test_translation_vector_norm_budget_rejected_not_silently_scaled():
    m=module();a,b=poses();b[:3]=a[:3]+[.002,.002,0]
    q=np.array([.3,.4,.5,.6,.7,.8])
    with pytest.raises(ValueError,match="rate"):
        m.compile_actions(a,b,q,q,side="left",physics_steps=80,mode="intrinsic_linear")

def test_eight_substeps_policy_hold_is_mandatory():
    m=module();a,b=poses();q=np.array([.3,.4,.5,.6,.7,.8])
    with pytest.raises(ValueError,match="whole"):
        m.compile_actions(a,b,q,q,side="left",physics_steps=81,mode="intrinsic_linear")

def test_geodesic_knots_are_reproduced_by_intrinsic_action_segments():
    m=module();a,b=poses();q=np.array([.3,.4,.5,.6,.7,.8])
    p=m.compile_actions(a,b,q,q+.01,side="left",physics_steps=80,mode="geodesic_knots")
    for k in range(11):
        pose,_=m.target_at(p,k/10)
        expected=m.pose_at(a,b,k/10)
        assert pose[:3]==pytest.approx(expected[:3],abs=1e-14)
        assert abs(np.dot(pose[3:],expected[3:]))==pytest.approx(1,abs=1e-13)

def test_near_gimbal_and_nonunit_quaternion_rejected():
    m=module();a,b=poses()
    b[3:]=np.roll((Rotation.from_quat(a[[4,5,6,3]])*Rotation.from_euler("XYZ",[0,np.pi/2,0])).as_quat(),1)
    with pytest.raises(ValueError,match="gimbal"):
        m.relative_intrinsic(a,b)
    a[3:]*=2
    with pytest.raises(ValueError,match="unit"):
        m.relative_intrinsic(a,b)

def test_budget_counts_setup_quiet_move_hold_without_expansion():
    m=module()
    r=m.budget_audit(setup=2,quiet=48,motion=200,hold=24,maximum=240)
    assert r["total_sdk_steps"]==274 and r["within_budget"] is False
    assert r["required_extra_steps"]==34
    assert r["native_ready"] is False

def test_unknown_mode_or_wrong_side_is_rejected():
    m=module();a,b=poses();q=np.array([.3,.4,.5,.6,.7,.8])
    for kwargs in [dict(side="mirror",mode="intrinsic_linear"),dict(side="left",mode="unknown")]:
        with pytest.raises(ValueError):
            m.compile_actions(a,b,q,q,physics_steps=80,**kwargs)
