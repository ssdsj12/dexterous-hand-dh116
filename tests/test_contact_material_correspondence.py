from copy import deepcopy
import numpy as np
import pytest

def mod():
    from go2_pvcnn.control.dh116 import tabletop_contact_material_correspondence as m
    return m

def inputs(node=1,normal=(0,1.,0)):
    nodes=np.array([[0.,0,0],[.1,0,0],[.2,0,0]])
    q=np.tile([0.,0,0,1.],(2,1));rest=np.array([.1,.1])
    n=np.asarray(normal,dtype=float);n=n/np.linalg.norm(n)
    pair=dict(substep=11,node_index=node,mesh_name="right.finger14_link",
        outward_normal_world=n,contact_point_world_m=nodes[node]-.005*n,
        normal_force_on_rod_n=.1)
    return pair,rest,nodes,q

def candidate(*args):
    p,r,n,q=args or inputs()
    return mod().contact_candidate(p,r,n,q,radius_m=.005,material_reference_uid="episode-one")

def test_exact_radial_node_contact_has_reconstructible_candidate_not_slip_qualification():
    result=candidate()
    assert result["mapping"]["sphere_center_residual_m"]<1e-15
    assert result["mapping"]["axial_cross_section_residual_m"]<1e-15
    assert result["mapping"]["surface_model_candidate_residual_m"]<1e-15
    assert result["marker"].segment_index==1 and result["marker"].alpha==0
    assert result["marker"].rest_s_m==pytest.approx(.1)
    assert result["r0_tangential_slip_m"] is None and result["physical_error_bound_m"] is None
    assert not result["native_qualification"] and not result["teacher_admission"]

def test_axial_sphere_contact_is_not_forced_into_cylinder_cross_section():
    result=candidate(*inputs(normal=(.6,.8,0)))
    assert result["mapping"]["axial_cross_section_residual_m"]==pytest.approx(.003)
    assert result["mapping"]["radial_surface_residual_m"]==pytest.approx(.001)
    assert result["mapping"]["surface_model_candidate_residual_m"]==pytest.approx(.003)
    assert result["mapping"]["exceeds_original_slip_threshold_as_model_residual"]
    assert result["contact_marker_correspondence_verified"] is False

def test_last_node_uses_last_real_director_not_padding():
    result=candidate(*inputs(node=2))
    assert result["marker"].segment_index==1 and result["marker"].alpha==1.
    assert result["marker"].rest_s_m==pytest.approx(.2)

def test_sign_of_director_quaternion_does_not_change_mapping():
    p,r,n,q=inputs()
    first=candidate(p,r,n,q);second=candidate(p,r,n,-q)
    assert first["marker"]==second["marker"]
    assert first["mapping"]==second["mapping"]

@pytest.mark.parametrize("fault",["substep","node","normal","point","q","rest","radius","material"])
def test_missing_or_untimed_source_is_rejected(fault):
    p,r,n,q=inputs();radius=.005;uid="real-episode"
    if fault=="substep":p["substep"]=10
    elif fault=="node":p["node_index"]=True
    elif fault=="normal":p["outward_normal_world"]=np.zeros(3)
    elif fault=="point":p["contact_point_world_m"]=n[1]+.05
    elif fault=="q":q[0]=0
    elif fault=="rest":r[0]=0
    elif fault=="radius":radius=float("nan")
    else:uid=""
    with pytest.raises(ValueError):
        mod().contact_candidate(p,r,n,q,radius_m=radius,material_reference_uid=uid)

def test_fixed_reference_persists_across_contact_loss_and_other_nodes():
    p,r,n,q=inputs();ref=mod().bind_candidate(candidate(p,r,n,q),r,n,q,np.eye(4))
    changed=n+[.001,0,0]
    a=mod().transport(ref,r,changed,q,np.eye(4),endpoint_contact_present=False)
    b=mod().transport(ref,r,changed+[.001,0,0],q,np.eye(4),endpoint_contact_present=True)
    assert a["marker_tangent_displacement_m"]==pytest.approx(.001)
    assert b["marker_tangent_displacement_m"]==pytest.approx(.002)
    assert a["fixed_rest_s_m"]==b["fixed_rest_s_m"]==pytest.approx(.1)
    assert a["r0_tangential_slip_m"] is b["r0_tangential_slip_m"] is None
    assert not a["endpoint_contact_present"] and b["endpoint_contact_present"]
    assert not b["substep_path_bound_available"]

def test_joint_world_motion_cancels_and_material_spin_does_not():
    p,r,n,q=inputs();c=candidate(p,r,n,q)
    ref=mod().bind_candidate(c,r,n,q,np.eye(4))
    pad=np.eye(4);pad[:3,3]=[.2,.3,.4]
    result=mod().transport(ref,r,n+pad[:3,3],q,pad,endpoint_contact_present=True)
    assert result["marker_tangent_displacement_m"]<1e-15
    angle=.2;q[:]=[np.sin(angle/2),0,0,np.cos(angle/2)]
    result=mod().transport(ref,r,n,q,np.eye(4),endpoint_contact_present=True)
    assert result["marker_tangent_displacement_m"]==pytest.approx(.005*np.sin(angle))

def test_changed_material_rest_does_not_reset_reference():
    p,r,n,q=inputs();ref=mod().bind_candidate(candidate(p,r,n,q),r,n,q,np.eye(4))
    r[0]+=.01
    with pytest.raises(ValueError):mod().transport(ref,r,n,q,np.eye(4),endpoint_contact_present=True)
