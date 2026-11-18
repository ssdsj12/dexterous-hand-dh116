"""Pure NumPy full17D geometry hypotheses; no SDK or Native qualification."""
from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pytest

from test_bimanual_mpc_model_descriptor import fixture,attr


def api():
    p=Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_mounted_kinematics.py'
    assert p.is_file(), 'pure mounted kinematics module is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_mounted_kinematics')


def sample_fixture():
    f=fixture()
    def geometry(g):
        for i,j in enumerate(g['joints']):
            j['attributes']['physics:localPos0']=attr([.004*i,.003,.002])
            j['attributes']['physics:localPos1']=attr([-.001,.002,0.])
            j['attributes']['physics:localRot0']=attr([np.cos(.1),0.,0.,np.sin(.1)])
            j['attributes']['physics:localRot1']=attr([np.cos(.05),np.sin(.05),0.,0.])
        # Only finger12 differs between sides; no mirrored copying of axes.
        for j in g['joints']:
            j['attributes']['physics:axis']=attr('Y' if '/rightHand/' in j['path'] and j['path'].endswith('/finger12') else 'X')
    f.facade.mutate=geometry
    for side,h in f.hands.items():
        h.root_physx_view.arrays['get_link_transforms'][:,0,:3]=[1.,2.,3.] if side=='left' else [-1.,.5,2.]
        h.root_physx_view.arrays['get_link_transforms'][:,0,3:]=[0.,0.,np.sin(.2),np.cos(.2)]
    return f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]


def hypothesis(m,s,side):
    b=s.descriptor['hands'][side]['binding']
    return dict(version='dh116-mounted-geometry-hypothesis-v1',descriptor_sha256=m.descriptor_digest(s),side=side,
        root_pose_source='existing_sdk_link_transform_readback',root_link_row=s.descriptor['hands'][side]['graph']['root_link_row'],
        root_frame='world_link',root_quaternion_order='xyzw',
        quaternion_policy='unit_tolerance_1e-5_normalize_roundoff',
        edge_direction='authored_body0_to_body1',
        joint_coordinates=[dict(dof_name=n,dof_path=p,sign=1,zero_offset=0.,
            unit='m' if t==1 else 'rad') for n,p,t in zip(b['dof_names'],b['dof_paths'],b['dof_types'])],
        basis='explicit USD-joint/SDK-reference hypothesis; not Native calibration')


def build(side='left'):
    m=api(); s=sample_fixture(); h=hypothesis(m,s,side)
    return m,s,h,m.MountedTreeKinematics.from_descriptor(s,side=side,coordinate_hypothesis=h)


def test_full_tree_root_mapping_uses_actual_owned_sdk_root_readback():
    m,s,h,k=build(); q=np.zeros(17); result=k.fk(q)
    root=s.descriptor['hands']['left']['graph']['root_link_row']
    assert np.allclose(np.asarray(result.link_poses_world)[root][:3,3],[1.,2.,3.])
    assert len(result.link_poses_world)==23 and len(result.dof_names)==17
    assert result.kind=='geometry_hypothesis' and not any(result.qualification.values())
    assert result.assumption_receipt['descriptor_sha256']==m.descriptor_digest(s)


def test_both_sides_actual_finger12_axes_are_independent_not_reflected():
    m=api(); s=sample_fixture(); q=np.linspace(.001,.017,17)
    left=m.MountedTreeKinematics.from_descriptor(s,side='left',coordinate_hypothesis=hypothesis(m,s,'left'))
    right=m.MountedTreeKinematics.from_descriptor(s,side='right',coordinate_hypothesis=hypothesis(m,s,'right'))
    jl=np.asarray(left.point_jacobian(q,link_row=22,point_link_xyz=[.01,.02,-.01]).jacobian_world)
    jr=np.asarray(right.point_jacobian(q,link_row=22,point_link_xyz=[.01,.02,-.01]).jacobian_world)
    row=list(s.descriptor['hands']['left']['binding']['dof_names']).index('finger12')
    assert not np.allclose(jl[:,row],jr[:,row])
    for k in (left,right):
        audit=k.audit_point_jacobian(q,link_row=22,point_link_xyz=[.01,.02,-.01],epsilon_by_dof=np.full(17,1e-6))
        assert np.max(np.abs(audit['central_error']))<1e-8
        assert np.max(np.abs(audit['forward_error']))<2e-6
        assert np.max(np.abs(audit['backward_error']))<2e-6


def test_fixed_links_and_active_prismatic_columns_have_correct_analytic_jacobian():
    _,s,_,k=build(); q=np.linspace(-.03,.03,17)
    out=k.point_jacobian(q,link_row=22,point_link_xyz=[0.,0.,0.])
    b=s.descriptor['hands']['left']['binding']; j=np.asarray(out.jacobian_world)
    root_rotation=np.asarray(k.fk(q).link_poses_world)[0,:3,:3]
    # First source moving joint is wrist_yaw (revolute); every pure translation
    # column has unit length, while downstream fixed links still inherit motion.
    for i,t in enumerate(b['dof_types']):
        if t==1: assert np.isclose(np.linalg.norm(j[:,i]),1.,atol=1e-10)
    assert root_rotation.shape==(3,3)
    assert out.coordinate_units==tuple('m' if t==1 else 'rad' for t in b['dof_types'])


def test_passive_follower_is_an_independent_actual_coordinate_not_affine_filled():
    _,s,_,k=build(); q=np.zeros(17)
    i=list(s.descriptor['hands']['left']['binding']['dof_names']).index('finger13')
    a=np.asarray(k.fk(q).link_poses_world); q[i]=.04; b=np.asarray(k.fk(q).link_poses_world)
    assert not np.allclose(a[-1],b[-1])


@pytest.mark.parametrize('fault',['missing_sign','missing_frame','wrong_sha','wrong_side','wrong_row','unknown_unit','gain_not_sign','reverse_edge','floating'])
def test_missing_or_unbound_coordinate_contract_is_rejected(fault):
    m,s,h,_=build()
    if fault=='missing_sign': del h['joint_coordinates'][0]['sign']
    elif fault=='missing_frame': del h['root_frame']
    elif fault=='wrong_sha': h['descriptor_sha256']='b'*64
    elif fault=='wrong_side': h['side']='right'
    elif fault=='wrong_row': h['joint_coordinates'].reverse()
    elif fault=='unknown_unit': h['joint_coordinates'][0]['unit']='degree'
    elif fault=='gain_not_sign': h['joint_coordinates'][0]['sign']=.8
    elif fault=='reverse_edge': h['edge_direction']='reoriented_unknown'
    else: h['root_frame']='floating_world_guess'
    with pytest.raises(ValueError): m.MountedTreeKinematics.from_descriptor(s,side='left',coordinate_hypothesis=h)


def test_sign_convention_is_explicit_and_chain_rule_applies_to_jacobian():
    m,s,h,_=build(); h['joint_coordinates'][2]['sign']=-1; h['joint_coordinates'][2]['zero_offset']=.02
    k=m.MountedTreeKinematics.from_descriptor(s,side='left',coordinate_hypothesis=h)
    a=k.audit_point_jacobian(np.full(17,.01),link_row=22,point_link_xyz=[.01,0,0],epsilon_by_dof=np.full(17,1e-6))
    assert np.max(np.abs(a['central_error']))<1e-8
    assert a['kind']=='geometry_hypothesis'


def test_inputs_and_results_are_owned_and_invalid_q_or_link_is_rejected():
    _,_,h,k=build(); h['joint_coordinates'][0]['sign']=-1
    a=k.fk(np.zeros(17)); assert a.assumption_receipt['joint_coordinates'][0]['sign']==1
    with pytest.raises(TypeError): a.assumption_receipt['side']='right'
    for q in (np.zeros(16),np.full(17,np.nan),np.ones((17,1))):
        with pytest.raises(ValueError): k.fk(q)
    for row in (-1,23,True):
        with pytest.raises(ValueError): k.point_jacobian(np.zeros(17),link_row=row,point_link_xyz=[0,0,0])
    with pytest.raises(ValueError): k.audit_point_jacobian(np.zeros(17),link_row=1,point_link_xyz=[0,0,0],epsilon_by_dof=np.zeros(17))


def test_analytical_golden_rotation_translation_and_local_frame_inverse():
    m=api(); f=fixture(); s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    k=m.MountedTreeKinematics.from_descriptor(s,side='left',coordinate_hypothesis=hypothesis(m,s,'left'))
    q=np.zeros(17); q[0]=np.pi/2; q[3]=.03
    p=k.point_jacobian(q,link_row=22,point_link_xyz=[0,1,0])
    assert np.allclose(p.point_world_m,[.03,0,1],atol=1e-12)
    assert np.allclose(np.asarray(p.jacobian_world)[:,0],[0,-1,0],atol=1e-12)
    f=fixture()
    def frames(g):
        g['joints'][0]['attributes']['physics:localPos0']=attr([.2,0,0])
        g['joints'][0]['attributes']['physics:localPos1']=attr([.1,0,0])
        g['joints'][0]['attributes']['physics:localRot1']=attr([np.sqrt(.5),0,0,np.sqrt(.5)])
    f.facade.mutate=frames; s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    k=m.MountedTreeKinematics.from_descriptor(s,side='left',coordinate_hypothesis=hypothesis(m,s,'left'))
    # T0 * inverse(T1), not T0 * T1.  A local point [1,0,0] is moved to
    # Rz(-pi/2) * ([1,0,0]-[.1,0,0]) + [.2,0,0].
    assert np.allclose(k.point_jacobian(np.zeros(17),link_row=22,point_link_xyz=[1,0,0]).point_world_m,[.2,-.9,0],atol=1e-12)


def test_world_attachment_body_order_is_preserved_without_guessing_rest_root():
    m=api(); f=fixture()
    def reversed_attachment(g):
        w=g['world_attachments'][0]; w['body0'],w['body1']=w['body1'],w['body0']
    f.facade.mutate=reversed_attachment; s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    k=m.MountedTreeKinematics.from_descriptor(s,side='left',coordinate_hypothesis=hypothesis(m,s,'left'))
    assert k.assumption_receipt['world_attachment']['body1']==()
    assert np.allclose(np.asarray(k.fk(np.zeros(17)).link_poses_world)[0],np.eye(4))


def test_failed_descriptor_cannot_be_bypassed_and_missing_root_readback_is_rejected():
    m,s,h,_=build()
    with pytest.raises(ValueError,match='structure'):
        m.MountedTreeKinematics.from_descriptor(replace(s,structure_valid=False),side='left',coordinate_hypothesis=h)
    d=m._thaw(s.descriptor); d['hands']['left']['readbacks']['get_link_transforms']['selected_valid']=False
    changed=replace(s,descriptor=m._freeze(d)); h['descriptor_sha256']=m.descriptor_digest(changed)
    with pytest.raises(ValueError,match='readback'):
        m.MountedTreeKinematics.from_descriptor(changed,side='left',coordinate_hypothesis=h)


def test_finite_input_that_overflows_geometry_is_rejected_not_published_as_nan():
    _,s,_,k=build(); q=np.zeros(17)
    indices=[i for i,t in enumerate(s.descriptor['hands']['left']['binding']['dof_types']) if t==1]
    q[indices]=np.finfo(float).max
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(ValueError,match='finite|overflow'):
            k.fk(q)
