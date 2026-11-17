import numpy as np
import pytest
from scripts.solve_dh116_pinch_effective_targets import predict_region,decode,admit,validate_config,CONFIG,rod_guard_applies

def plane_grid():
    values=np.broadcast_to(np.linspace(-.01,.01,21)[None,None,:],(21,21,21)).copy()
    return dict(values=values,origin=np.array([-.01,-.01,-.01]),spacing=.001)
def patch():
    return dict(center_local_m=[0,0,0],tangent_u=[1,0,0],tangent_v=[0,1,0],
        half_widths_m=[.004,.0035],planes_local=[[0,0,1,0]],
        surface_tolerance_m=.001,normal_cosine_min=.8)
def test_proposed_sdf_collision_outside_ellipse_is_not_endpoint_ready():
    r=predict_region(plane_grid(),np.eye(4),np.array([[.0045,0,.0049]]),patch())
    assert r["sdf_gap_m"]<0
    assert r["ellipse_value"]>1
    assert not r["original_region_passed"] and not r["actual_contact_observed"]
def test_inside_ellipse_cpu_target_never_becomes_native_contact():
    r=predict_region(plane_grid(),np.eye(4),np.array([[.001,0,.0049]]),patch())
    assert r["original_region_passed"] and r["sdf_gap_m"]==pytest.approx(-.0001,abs=1e-8)
    assert not r["actual_contact_observed"]
def test_joint_and_original_quaternion_contract_preserved():
    c=dict(root_pose_wxyz=[.4,.1,.3,1,0,0,0],q_rad=[.5,.4,.6,1.15,1.15,1.15])
    p,q=decode(c,np.zeros(9));np.testing.assert_array_equal(p,c["root_pose_wxyz"])
    np.testing.assert_array_equal(q,c["q_rad"])
    with pytest.raises(ValueError):decode(c,np.full(9,2))
def test_planning_cannot_expand_native_guards():
    with pytest.raises(ValueError):validate_config(dict(CONFIG,native_required_gap_m=-.001))
    assert not admit({"table":1.,"region":-.1})["static_constraints_passed"]
    a=admit({"table":1.,"region":1.})
    assert a["static_constraints_passed"] and not a["native_ready"]
    assert not a["continuous_path_proved"] and not a["teacher_admission"]

def test_rod_constraints_match_actual_native_guard_without_extra_required_hull_penalty():
    for kind in ("source","cooked"):
        assert not rod_guard_applies(kind+":finger14_link")
        assert not rod_guard_applies(kind+":finger23_link")
        assert rod_guard_applies(kind+":finger13_link")
        assert rod_guard_applies(kind+":finger33_link")

def test_original_action_schedule_retained_in_shifted_endpoint_path():
    from scripts.solve_dh116_pinch_effective_targets import original_path
    end=dict(root_pose_wxyz=[.4,-.1,.3,1,0,0,0],q_rad=[.5,.4,.6,1.15,1.15,1.15])
    old=dict(active_delta_rad=[.01,.03,.04,0,0,0],root_translation_delta_m=[0,0,-.0005])
    start,delta=original_path(end,old,0.)
    final,again=original_path(end,old,1.)
    np.testing.assert_allclose(np.array(final["q_rad"])-start["q_rad"],old["active_delta_rad"])
    np.testing.assert_allclose(np.array(final["root_pose_wxyz"])[:3]-np.array(start["root_pose_wxyz"])[:3],[0,0,-.0005])
    np.testing.assert_array_equal(delta,again)
    with pytest.raises(ValueError):original_path(end,old,1.1)

def test_one_hand_failure_does_not_block_other_independent_cpu_search():
    from scripts.solve_dh116_pinch_effective_targets import isolated_sides
    called=[]
    def solve(index,side):
        called.append(side)
        if side=="left":raise ValueError("no safe geometry")
        return {"status":"complete","native_ready":False}
    rows=isolated_sides(solve)
    assert called==["left","right"]
    assert rows["left"]["status"]=="blocked" and rows["right"]["status"]=="complete"

def test_bounded_bow_is_exactly_reproducible_by_twenty_60hz_actions():
    from scripts.audit_dh116_effective_pinch_target_path import bow
    values=np.array([bow(i/20,.0001) for i in range(21)])
    assert values[0]==values[-1]==0 and max(values)==pytest.approx(.0001)
    rates=np.diff(values)/(8/480)
    assert max(abs(rates))<.03
    for i in range(20):
        assert bow((i+.3)/20,.0001)==pytest.approx(values[i]+.3*(values[i+1]-values[i]))
    with pytest.raises(ValueError):bow(.5,.001)

def test_piecewise_path_bound_uses_maximum_segment_rate_not_average_arc_length():
    from scripts.audit_dh116_effective_pinch_target_path import wrist_path_lipschitz
    points=np.zeros((21,3));points[1,0]=.0001
    assert wrist_path_lipschitz(points)==pytest.approx(.002)
    assert wrist_path_lipschitz(points)>np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
