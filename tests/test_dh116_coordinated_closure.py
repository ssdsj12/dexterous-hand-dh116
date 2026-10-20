from fixture_paths import ARCHIVE, ASSET, BACKEND, CONTACT, CONTACT_SHA, SURFACE_REFS
import numpy as np
import pytest
from go2_pvcnn.control.dh116.coordinated_closure import coordinated_closure_increment


def inputs():
    J=np.zeros((2,3,10));J[0,:,3:6]=.03*np.eye(3);J[1,:,6:9]=.03*np.eye(3)
    W=np.zeros((3,10));W[:,:3]=np.eye(3)
    return J,np.array([[1.,0,0],[-1.,0,0]]),W


@pytest.mark.parametrize('actual',[False,True])
def test_touch_first_holds_index_world_point_and_palm(actual):
    from go2_pvcnn.control.dh116.coordinated_closure import coordinated_bimanual_increment
    pairs=[actual_same_section_inputs(s) if actual else inputs() for s in ('left','right')]
    J,n,W=(np.asarray(x) for x in zip(*pairs))
    dq=coordinated_bimanual_increment(J,n,W,np.zeros((2,2)),1e-4,
        target_loads_n=np.full((2,2),.05),transport_velocity_w=np.zeros(3),
        normal_speeds_m_s=np.array([[.0005,0.],[.0005,0.]]))
    v=np.einsum('shij,sj->shi',J,dq/1e-4)
    np.testing.assert_allclose(v[:,1],0.,atol=1e-10)
    assert np.all(np.sum(v[:,0]*n[:,0],axis=1)>0)
    np.testing.assert_allclose(np.einsum('sij,sj->si',W,dq/1e-4),0.,atol=1e-10)


@pytest.mark.parametrize('speeds', [np.full((2,2),np.nan),np.full((2,2),.0021),np.zeros(4)])
def test_explicit_pad_speed_rejects_invalid_or_excessive_commands(speeds):
    from go2_pvcnn.control.dh116.coordinated_closure import coordinated_bimanual_increment
    J,n,W=inputs()
    with pytest.raises(ValueError):
        coordinated_bimanual_increment(np.stack((J,J)),np.stack((n,n)),np.stack((W,W)),
            np.zeros((2,2)),1e-4,target_loads_n=np.full((2,2),.05),transport_velocity_w=np.zeros(3),
            normal_speeds_m_s=speeds)


def test_tangent_and_palm_rotation_cancelled():
    J,n,W=inputs()
    dq=coordinated_closure_increment(J,n,W,[.1,1.5],1e-4,target_loads_n=[1.,1.])
    v=np.einsum('ijk,k->ij',J,dq/1e-4)
    assert v[0,0]>0 and v[1,0]>0
    np.testing.assert_allclose(v[:,1:],0,atol=1e-12)
    np.testing.assert_allclose(W@(dq/1e-4),0,atol=1e-12)
    assert np.max(np.abs(dq[:7]/1e-4))<=.05+1e-12
    assert np.max(np.abs(dq[7:]/1e-4))<=.02+1e-12


def test_deadband_dt_scaling_and_frame_covariance():
    from scipy.spatial.transform import Rotation
    J,n,W=inputs();R=Rotation.from_rotvec([.3,-.2,.4]).as_matrix()
    call=lambda j,normal,w,dt:coordinated_closure_increment(j,normal,w,[.1,.2],dt,target_loads_n=[1.,1.])
    a=call(J,n,W,1e-4)
    np.testing.assert_allclose(a,2*call(J,n,W,5e-5),atol=1e-14)
    np.testing.assert_allclose(a,call(np.einsum('ij,kjl->kil',R,J),n@R.T,R@W,1e-4),atol=1e-14)
    np.testing.assert_array_equal(coordinated_closure_increment(J,n,W,[.98,1.02],1e-4,
        target_loads_n=[1.,1.]),np.zeros(10))


def test_common_transport_preserves_orientation_and_default_exactly():
    J,n,W=inputs();kw=dict(target_loads_n=[1.,1.])
    old=coordinated_closure_increment(J,n,W,[.1,.2],1e-4,**kw)
    explicit=coordinated_closure_increment(J,n,W,[.1,.2],1e-4,
        transport_velocity_w=np.zeros(3),**kw)
    np.testing.assert_array_equal(old,explicit)
    dq=coordinated_closure_increment(J,n,W,[1.,1.],1e-4,
        transport_velocity_w=[0.,0.,.0003],**kw)
    v=np.einsum('ijk,k->ij',J,dq/1e-4)
    np.testing.assert_allclose(v,[[0,0,.0003],[0,0,.0003]],atol=1e-12)
    np.testing.assert_allclose(W@(dq/1e-4),0,atol=1e-12)


def test_bimanual_transport_has_one_common_saturation_scale():
    from go2_pvcnn.control.dh116.coordinated_closure import coordinated_bimanual_increment
    J,n,W=inputs();both_J=np.stack((J,J*.1));both_n=np.stack((n,n));both_W=np.stack((W,W))
    dq=coordinated_bimanual_increment(both_J,both_n,both_W,np.ones((2,2)),1e-4,
        target_loads_n=np.ones((2,2)),transport_velocity_w=[0.,0.,.003])
    v=np.einsum('shij,sj->shi',both_J,dq/1e-4)
    np.testing.assert_allclose(v,np.broadcast_to([0.,0.,.00006],(2,2,3)),atol=1e-12)
    assert np.max(np.abs(dq[:,:7]/1e-4))<=.05+1e-12
    assert np.max(np.abs(dq[:,7:]/1e-4))<=.02+1e-12
    np.testing.assert_allclose(np.einsum('sij,sj->si',both_W,dq/1e-4),0.,atol=1e-12)


@pytest.mark.parametrize('actual',[False,True])
def test_bilateral_transport_preserves_velocity_ratio_and_palm_orientation(actual):
    from go2_pvcnn.control.dh116.coordinated_closure import coordinated_bimanual_increment
    pairs=[actual_same_section_inputs(s) if actual else inputs() for s in ('left','right')]
    J,n,W=(np.asarray(x) for x in zip(*pairs))
    requested=np.array([[0.,0.,.001],[0.,0.,.003]])
    dq=coordinated_bimanual_increment(J,n,W,np.ones((2,2)),1e-4,
        target_loads_n=np.ones((2,2)),transport_velocity_w=requested)
    v=np.einsum('shij,sj->shi',J,dq/1e-4)
    np.testing.assert_allclose(v[:,0],v[:,1],atol=1e-10)
    np.testing.assert_allclose(v[:,:,:2],0.,atol=1e-10)
    np.testing.assert_allclose(v[1,0,2],3*v[0,0,2],atol=1e-10)
    assert 0<v[0,0,2]<=.001+1e-12
    np.testing.assert_allclose(np.einsum('sij,sj->si',W,dq/1e-4),0.,atol=1e-10)
    assert np.max(np.abs(dq[:,:7]/1e-4))<=.05+1e-12
    assert np.max(np.abs(dq[:,7:]/1e-4))<=.02+1e-12


@pytest.mark.parametrize('v',[np.zeros((1,3)),np.zeros((2,2)),
    [[0,0,.001],[0,0,np.nan]],[[0,0,.001],[0,0,.01001]]])
def test_invalid_bilateral_transport_rejected(v):
    from go2_pvcnn.control.dh116.coordinated_closure import coordinated_bimanual_increment
    J,n,W=inputs()
    with pytest.raises(ValueError):
        coordinated_bimanual_increment(np.stack((J,J)),np.stack((n,n)),np.stack((W,W)),
            np.ones((2,2)),1e-4,target_loads_n=np.ones((2,2)),transport_velocity_w=v)


@pytest.mark.parametrize('requested',[
    [[0.,0.,0.],[.0001,-.0002,.0003]],
    [[.0001,.0002,.0003],[-.0002,.0001,-.0001]],
])
def test_generic_allocator_does_not_force_mirror_or_equal_height(requested):
    from go2_pvcnn.control.dh116.coordinated_closure import coordinated_bimanual_increment
    J,n,W=inputs();both_J=np.stack((J,J))
    dq=coordinated_bimanual_increment(both_J,np.stack((n,n)),np.stack((W,W)),
        np.ones((2,2)),1e-4,target_loads_n=np.ones((2,2)),transport_velocity_w=requested)
    measured=np.einsum('shij,sj->shi',both_J,dq/1e-4)
    np.testing.assert_allclose(measured,np.repeat(np.asarray(requested)[:,None,:],2,axis=1),atol=1e-12)


@pytest.mark.parametrize('bad',['shape','singular','nonfinite'])
def test_bimanual_rejects_bad_evidence_on_either_side(bad):
    from go2_pvcnn.control.dh116.coordinated_closure import coordinated_bimanual_increment
    J,n,W=inputs();j=np.stack((J,J));load=np.ones((2,2))
    if bad=='shape':j=j[:1]
    if bad=='singular':j[1]=0
    if bad=='nonfinite':load[1,1]=np.nan
    with pytest.raises(ValueError):coordinated_bimanual_increment(j,np.stack((n,n)),np.stack((W,W)),load,1e-4,
        target_loads_n=np.ones((2,2)),transport_velocity_w=[0,0,.001])


@pytest.mark.parametrize('transport',[[0.,0.],[0.,0.,np.nan],[0.,0.,.01001]])
def test_invalid_transport_is_rejected(transport):
    J,n,W=inputs()
    with pytest.raises(ValueError):
        coordinated_closure_increment(J,n,W,[1.,1.],1e-4,
            target_loads_n=[1.,1.],transport_velocity_w=transport)


def actual_same_section_inputs(side):
    import json
    from pathlib import Path
    pytest.importorskip('pxr.Usd')
    from pxr import Usd
    from go2_pvcnn.control.dh116.usd_kinematics import SideKinematics
    root=ARCHIVE
    if not (root/'zero_axial_touch_v1.json').exists():pytest.skip('local candidate unavailable')
    q=json.loads((root/'zero_axial_touch_v1.json').read_text())['states'][side]
    refs=[r for r in json.loads((root/'zero_axial_point_references_v1.json').read_text())['references'] if r['side']==side]
    model=SideKinematics(Usd.Stage.Open(str(ASSET)),side)
    poses,axes,anc=model.fk(q)
    J=np.array([model.point_jacobian(q,r['body_name'],r['point'])[:,:10] for r in refs])
    n=np.array([poses[model.bodies[r['body_name']]][:3,:3]@r['normal'] for r in refs])
    W=np.zeros((3,10))
    for name in anc[model.bodies['base_link']]:W[:,model.names.index(name)]=axes[name][1]
    return J,n,W


@pytest.mark.parametrize('side',['left','right'])
def test_actual_same_section_transport_preserves_common_velocity(side):
    from scipy.spatial.transform import Rotation
    J,n,W=actual_same_section_inputs(side)
    transport=np.array([0.,0.,.01]);R=Rotation.from_rotvec([.2,-.1,.3]).as_matrix()
    call=lambda j,normal,w,v:coordinated_closure_increment(j,normal,w,[1.,1.],1e-4,
        target_loads_n=[1.,1.],transport_velocity_w=v)
    dq=call(J,n,W,transport)
    v=np.einsum('ijk,k->ij',J,dq/1e-4)
    np.testing.assert_allclose(v[0],v[1],atol=1e-10)
    np.testing.assert_allclose(v[:,:2],0.,atol=1e-10)
    assert 0<v[0,2]<=.01+1e-12
    np.testing.assert_allclose(W@(dq/1e-4),0.,atol=1e-10)
    assert np.max(np.abs(dq[:7]/1e-4))<=.05+1e-12
    assert np.max(np.abs(dq[7:]/1e-4))<=.02+1e-12
    np.testing.assert_allclose(dq,call(np.einsum('ij,kjl->kil',R,J),n@R.T,R@W,R@transport),atol=1e-12)


def test_actual_bimanual_transport_has_equal_speed_with_one_shared_scale():
    from go2_pvcnn.control.dh116.coordinated_closure import coordinated_bimanual_increment
    J,n,W=(np.array(x) for x in zip(actual_same_section_inputs('left'),actual_same_section_inputs('right')))
    dq=coordinated_bimanual_increment(J,n,W,np.ones((2,2)),1e-4,
        target_loads_n=np.ones((2,2)),transport_velocity_w=[0.,0.,.01])
    v=np.einsum('shij,sj->shi',J,dq/1e-4)
    np.testing.assert_allclose(v,np.broadcast_to(v[0,0],(2,2,3)),atol=1e-10)
    np.testing.assert_allclose(v[:,:,:2],0.,atol=1e-10)
    assert 0<v[0,0,2]<=.01+1e-12
    np.testing.assert_allclose(np.einsum('sij,sj->si',W,dq/1e-4),0.,atol=1e-10)
    assert np.max(np.abs(dq[:,:7]/1e-4))<=.05+1e-12
    assert np.max(np.abs(dq[:,7:]/1e-4))<=.02+1e-12


@pytest.mark.parametrize('change',['singular','normals','dt','loads','target','palm'])
def test_rejects_invalid_or_unachievable_input(change):
    J,n,W=inputs();dt=1e-4;load=[.1,.2];target=[1.,1.]
    if change=='singular':J[:]=0
    if change=='normals':n[0]*=2
    if change=='dt':dt=.001
    if change=='loads':load=[np.nan,.2]
    if change=='target':target=[0.,1.]
    if change=='palm':W[0,7]=1
    with pytest.raises(ValueError):coordinated_closure_increment(J,n,W,load,dt,target_loads_n=target)


@pytest.mark.parametrize('side',['left','right'])
def test_actual_nominal_usd_has_feasible_normal_only_closure(side):
    import json
    from pathlib import Path
    pytest.importorskip('pxr.Usd')
    from pxr import Usd
    from go2_pvcnn.control.dh116.usd_kinematics import SideKinematics
    root=ARCHIVE
    asset=ASSET
    if not asset.exists():pytest.skip('local mounted asset unavailable')
    q=json.loads((root/'grasp_nominal_touch_pose_v1.json').read_text())['states'][side]
    refs=[r for r in json.loads((root/'grasp_point_references_v1.json').read_text())['references'] if r['side']==side]
    model=SideKinematics(Usd.Stage.Open(str(asset)),side);poses,axes,anc=model.fk(q)
    J=np.array([model.point_jacobian(q,r['body_name'],r['point'])[:,:10] for r in refs])
    n=np.array([poses[model.bodies[r['body_name']]][:3,:3]@r['normal'] for r in refs])
    W=np.zeros((3,10))
    for name in anc[model.bodies['base_link']]:W[:,model.names.index(name)]=axes[name][1]
    dq=coordinated_closure_increment(J,n,W,[0.,0.],1e-4,target_loads_n=[1.,1.])
    v=np.einsum('ijk,k->ij',J,dq/1e-4);normal_speed=np.sum(v*n,axis=1)
    assert np.all(normal_speed>0)
    np.testing.assert_allclose(v-normal_speed[:,None]*n,0,atol=1e-10)
    np.testing.assert_allclose(W@(dq/1e-4),0,atol=1e-10)
    assert np.max(np.abs(dq[:7]/1e-4))<=.05+1e-12
    assert np.max(np.abs(dq[7:]/1e-4))<=.02+1e-12
