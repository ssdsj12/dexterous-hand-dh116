from copy import deepcopy
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_deformable_pad_region import classify_pair,bind_geometry,independent_gates

def atlas():
    return dict(center_local_m=[0,0,0],tangent_u=[1,0,0],tangent_v=[0,1,0],
        half_widths_m=[.004,.0035],planes_local=[[0,0,1,0]],
        surface_tolerance_m=.001,normal_cosine_min=.8)
def pair(point=(0,0,0),normal=(0,0,1)):
    return dict(contact_point_world_m=list(point),outward_normal_world=list(normal),
        normal_force_on_rod_n=.1,substep=2,node_index=3,mesh_name='left.finger14_link')
def test_original_normal_convention_is_explicitly_reversed_to_on_hand():
    r=classify_pair(pair(),np.eye(4),atlas())
    assert r['accepted_by_original_patch'] and r['normal_cosine_on_hand']==-1.
    assert r['normal_on_hand_local']==[0.,0.,-1.]
def test_raw_collision_outside_original_ellipse_is_rejected():
    r=classify_pair(pair((.0035,.003,0)),np.eye(4),atlas())
    assert not r['accepted_by_original_patch']
    assert r['first_rejection']=='outside_inner_ellipse'
    assert r['overlap_rejections']==['outside_inner_ellipse']
def test_overlap_reasons_not_hidden_and_precedence_stable():
    r=classify_pair(pair((.005,.004,.002),(0,0,-1)),np.eye(4),atlas())
    assert r['first_rejection']=='outside_inner_ellipse'
    assert set(r['overlap_rejections'])=={'outside_inner_ellipse','outside_source_surface','wrong_normal'}
def test_world_rotation_translation_leave_link_local_qualification_unchanged():
    from scipy.spatial.transform import Rotation
    R=Rotation.from_rotvec([.4,-.3,.2]).as_matrix();t=np.array([.5,.1,.3])
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
    p=np.array([.001,.001,.0001]);n=np.array([0,0,1])
    a=classify_pair(pair(p,n),np.eye(4),atlas())
    b=classify_pair(pair(R@p+t,R@n),T,atlas())
    assert a['accepted_by_original_patch']==b['accepted_by_original_patch']
    np.testing.assert_allclose(a['point_link_m'],b['point_link_m'],atol=1e-14)
def test_invalid_pose_or_normal_cannot_be_qualified():
    for normal in [[0,0,0],[0,0,2],[0,float('nan'),1]]:
        with pytest.raises(ValueError):classify_pair(pair(normal=normal),np.eye(4),atlas())
    T=np.eye(4);T[0,0]=-1
    with pytest.raises(ValueError):classify_pair(pair(),T,atlas())
def test_region_thresholds_cannot_silently_expand():
    for key,value in [('half_widths_m',[.005,.004]),('surface_tolerance_m',.002),('normal_cosine_min',.5)]:
        a=atlas();a[key]=value
        with pytest.raises(ValueError):classify_pair(pair(),np.eye(4),a)
def test_patch_acceptance_does_not_supply_missing_depth_slip_or_native_success():
    r=independent_gates(True)
    assert r['patch_membership_checked']
    assert r['contact_depth_m'] is None and r['fixed_material_slip_m'] is None
    assert not r['teacher_admission'] and not r['success_qualified']
def clock():
    return dict(run_uuid='r',env_id=0,episode_uid='e',reset_id=1,rod_uid='rod',
        physics_step=3,rod_tick=12,policy_step=0,substep=0,timestamp_s=3/480,
        solver_time_s=12/5760,clock_valid=True)
def hand(side):
    names=['finger14_link']
    return dict(body_names=names,link_paths=[['/World/X/'+side+'Hand/finger14_link']],
        prim_paths=['/World/X/'+side+'Hand/wrist_fixed'],
        link_poses_xyzw=np.array([[[.4,.1,.3,0,0,0,1]]]))
def test_binding_requires_same_episode_clock_and_own_side_link():
    raw=dict(identity=clock(),hands={'left':hand('left'),'right':hand('right')})
    binding=dict(side='left',body_name='finger14_link',frame='PhysX_link_origin',unit='m',
        body_index=0,prim_path='/World/X/leftHand/finger14_link')
    T=bind_geometry(raw,clock(),'left.finger14_link',binding)
    assert T.shape==(4,4)
    for change in ['clock','side','path']:
        a=deepcopy(raw);b=dict(binding)
        if change=='clock':a['identity']['physics_step']=4
        if change=='side':b['side']='right'
        if change=='path':b['prim_path']='/World/X/rightHand/finger14_link'
        with pytest.raises(ValueError):bind_geometry(a,clock(),'left.finger14_link',b)
