"""CPU-only end-support candidate contracts."""
import json
from pathlib import Path
import numpy as np
import pytest
from go2_pvcnn.control.dh116 import tabletop_deformx_end_supported_scene as api

def layout():
    return dict(wire_center=[.6,0,.305], wire_length_m=.5, wire_radius_m=.005,
                table_center=[.6,0,.25], table_size=[.8,1.2,.1],
                hand_roots=dict(left=[.6,-.2,.3,1,0,0,0],right=[.6,.2,.3,1,0,0,0]))

def seeds():
    nodes=np.array([[.6,y,.305] for y in np.linspace(-.25,.25,41)])
    return {style:dict(nodes=nodes.copy(), roots=np.array(list(layout()['hand_roots'].values())),
                       joints=np.arange(22).reshape(2,11),source={'style':style,'sha256':'original'})
            for style in ('pinch','tripod','power')}

def plan():
    return api.plan_flexible_end_supported_scene(layout(),seeds(),.06)

def test_seed_translation_preserves_grasp_geometry_sources_and_joints():
    original=seeds()
    result=api.plan_flexible_end_supported_scene(layout(),original,.06)
    for style,before in original.items():
        after=result['seeds_by_style'][style]
        np.testing.assert_allclose(after['nodes']-before['nodes'],[[0,0,.06]]*41)
        np.testing.assert_allclose(after['roots'][:,:3]-before['roots'][:,:3],[[0,0,.06]]*2)
        np.testing.assert_array_equal(after['roots'][:,3:],before['roots'][:,3:])
        np.testing.assert_array_equal(after['joints'],before['joints'])
        assert after['source']==before['source'] and after['source'] is not before['source']
        np.testing.assert_array_equal(before['nodes'],seeds()[style]['nodes'])
    assert result['mesh_contract']['support_indices']==[11,12]

def test_invalid_clearance_and_seed_shapes_rejected():
    for c in (0,-.01,float('nan')):
        with pytest.raises(ValueError):
            api.plan_flexible_end_supported_scene(layout(),seeds(),c)
    broken=seeds(); broken['power']['roots']=np.zeros((1,7))
    with pytest.raises(ValueError):
        api.plan_flexible_end_supported_scene(layout(),broken,.06)

def test_extends_numpy_buffers_preserving_native_indices_and_env_origins():
    p=np.arange(66,dtype=np.float32).reshape(2,11,3)
    q=np.zeros((2,11,4),np.float32);q[:,:,3]=1
    pe,qe=api.extend_mesh_pose_buffers(p,q,plan()['scene'],np.array([[0,0,0],[3,4,0]]))
    assert pe.shape==(2,13,3) and qe.shape==(2,13,4)
    np.testing.assert_array_equal(pe[:,:11],p)
    np.testing.assert_array_equal(qe[:,:11],q)
    np.testing.assert_allclose(pe[1,11:]-pe[0,11:],[[3,4,0]]*2,atol=1e-6)
    np.testing.assert_array_equal(qe[:,11:,3],np.ones((2,2)))
    with pytest.raises(ValueError):
        api.extend_mesh_pose_buffers(pe,qe,plan()['scene'],np.zeros((2,3)))

def test_torch_cpu_buffer_device_dtype_preserved():
    torch=pytest.importorskip('torch')
    p=torch.zeros((2,11,3),dtype=torch.float64)
    q=torch.zeros((2,11,4),dtype=torch.float64);q[:,:,3]=1
    pe,qe=api.extend_mesh_pose_buffers(p,q,plan()['scene'],torch.zeros((2,3)))
    assert pe.device==p.device and pe.dtype==p.dtype and qe.dtype==q.dtype

def records(valid=True):
    b=dict(mesh_indices=[11,12],mesh_count=13,num_envs=2,
           binding_records=[dict(env_id=i,support_mesh_indices=[11,12],physx_paths=['a','b'],
                                 deformx_bound=True,physx_bound=True) for i in range(2)])
    w=np.zeros((2,64,6));w[1,11,2]=-1
    return api.support_load_records(w,b,contact_data_valid=valid,step_id=5)

def test_actual_support_wrench_slots_and_per_environment_loads():
    rows=records()
    assert rows[0]['total_support_force_norm_n']==0
    assert rows[1]['total_support_force_norm_n']==1
    assert rows[1]['support_wrenches_world'][0][2]==-1
    assert rows[1]['mesh_indices']==[11,12]
    assert rows[1]['wrench_convention']=='rope_on_support_world_force_torque_about_mesh_origin'
    assert not records(False)[0]['contact_data_valid']

def test_no_valid_contacts_means_na_even_for_zero_force():
    nodes=plan()['seeds_by_style']['pinch']['nodes']+[0,0,.04]
    for contact in (None,records(False)[0]):
        r=api.evaluate_support_clearance(nodes,.005,plan()['scene'],contact,initial_min_z_m=.365)
        assert r['status']=='NA' and not r['eligible'] and r['net_lift_m'] is None

def test_clearance_tests_full_segments_and_load_before_net_lift():
    scene=plan()['scene'];nodes=plan()['seeds_by_style']['pinch']['nodes']
    assert not api.evaluate_support_clearance(nodes,.005,scene,records()[0],initial_min_z_m=.365)['eligible']
    r=api.evaluate_support_clearance(nodes+[0,0,.04],.005,scene,records()[0],initial_min_z_m=.365)
    assert r['eligible'] and r['status']=='clear' and r['net_lift_m']==pytest.approx(.04)
    assert not api.evaluate_support_clearance(nodes+[0,0,.04],.005,scene,records()[1],initial_min_z_m=.365)['eligible']
    crossing=np.array([[.6,-.4,.355],[.6,.4,.355]])
    assert not api.evaluate_support_clearance(crossing,.005,scene,records()[0],initial_min_z_m=.365)['eligible']

def test_power_five_pads_is_blocked_without_palm_proximal_native_surfaces():
    assert api.power_surface_qualification({})['status']=='blocked'
    pads={s:dict(five_fingers_closed=True,palm_enclosure=True,pad_count=5) for s in ('left','right')}
    assert api.power_surface_qualification(pads)['status']=='blocked'

def test_config_preserves_table_and_action_contract():
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/dh116_tasks/flexible_teacher_end_supported_v1.json').read_text())
    assert c['clearance_m']==.06 and c['calibration_status']=='candidate_not_measured'
    assert c['action_size']==24 and c['table']['mu_s']==.5 and c['table']['mu_k']==.4
    assert c['power']['status']=='blocked' and not c['native_execution_validated']

def test_real_static_usd_and_box_sdf_if_dependencies_present():
    Usd=pytest.importorskip('pxr.Usd')
    from pxr import UsdPhysics
    pytest.importorskip('ropesim.sdf')
    scene=plan()['scene'];stage=Usd.Stage.CreateInMemory()
    paths=api.author_end_supports(stage,'/World/envs/env_0',scene)
    assert len(paths)==2
    for path in paths:
        prim=stage.GetPrimAtPath(path)
        assert prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    grids=api.make_support_sdf_grids(scene,spacing_m=.005,margin_m=.025)
    for s,g in zip(scene['supports'],grids):
        half=np.asarray(s['size_m'])/2
        phi,inside=g.sample([[0,0,0],[0,0,half[2]],[0,0,half[2]+.01]])
        assert inside.all() and phi[0]<0
        assert abs(phi[1])<1e-7 and phi[2]==pytest.approx(.01,abs=1e-7)
def test_native_binding_protocol_uses_real_usd_sdf_and_checks_indices():
    Usd=pytest.importorskip('pxr.Usd')
    pytest.importorskip('ropesim.sdf')
    class RecordingRope:
        """CPU call capture, not a native solver or simulation substitute."""
        def __init__(self,start=11):
            self.start=start;self.calls=[];self.enabled=False
        def add_mesh(self,*args,**kwargs):
            self.calls.append((args,kwargs))
            return self.start+len(self.calls)-1
        def enable_contact_forces(self,enabled):
            self.enabled=enabled
    stage=Usd.Stage.CreateInMemory();rope=RecordingRope()
    origins=np.array([[0,0,0],[3,4,0]],dtype=np.float32)
    result=api.bind_native_end_supports(rope,stage,'/World/envs',plan()['scene'],origins)
    assert result['mesh_indices']==[11,12] and result['mesh_count']==13 and rope.enabled
    assert len(result['binding_records'])==2 and len(result['usd_paths'])==4
    np.testing.assert_allclose(result['positions'][1]-result['positions'][0],[[3,4,0]]*2,atol=1e-6)
    for j,(args,kwargs) in enumerate(rope.calls):
        assert args[0].dtype==np.float32 and args[2]==.005
        assert kwargs['mu_s']==.5 and kwargs['mu_k']==.4
        np.testing.assert_array_equal(kwargs['pos'],result['positions'][:,j])
        np.testing.assert_array_equal(kwargs['quat'],result['quat_xyzw'][:,j])
    from pxr import UsdGeom
    p0=UsdGeom.Xformable(stage.GetPrimAtPath(result['usd_paths'][0])).GetOrderedXformOps()[0].Get()
    p1=UsdGeom.Xformable(stage.GetPrimAtPath(result['usd_paths'][2])).GetOrderedXformOps()[0].Get()
    np.testing.assert_allclose(np.asarray(p1)-np.asarray(p0),origins[1])
    with pytest.raises(RuntimeError,match='expected 11'):
        api.bind_native_end_supports(RecordingRope(12),Usd.Stage.CreateInMemory(),
                                     '/World/envs',plan()['scene'],origins)

def test_stale_nan_and_missing_binding_evidence_never_qualify():
    scene=plan()['scene'];nodes=plan()['seeds_by_style']['pinch']['nodes']+[0,0,.04]
    row=records()[0]
    assert api.evaluate_support_clearance(nodes,.005,scene,row,initial_min_z_m=.365,
                                          expected_step_id=6)['status']=='NA'
    row=dict(row,total_support_force_norm_n=float('nan'))
    assert api.evaluate_support_clearance(nodes,.005,scene,row,initial_min_z_m=.365)['status']=='NA'
    b=dict(mesh_indices=[11,12],num_envs=1,
           binding_records=[dict(env_id=0,support_mesh_indices=[11,12],physx_paths=['a','b'],
                                 deformx_bound=False,physx_bound=True)])
    row=api.support_load_records(np.zeros((1,64,6)),b,contact_data_valid=True,step_id=7)[0]
    assert not row['contact_data_valid']
    assert api.evaluate_support_clearance(nodes,.005,scene,row,initial_min_z_m=.365)['status']=='NA'

def test_geometry_only_power_evidence_does_not_assert_grasp_success():
    surfaces={side:{surface:dict(physx_collision=True,deformx_sdf_registered=True,
                                 geometry_source='qualified-native-source')
                    for surface in ('palm','proximal')} for side in ('left','right')}
    q=api.power_surface_qualification(surfaces)
    assert q['status']=='geometry_ready' and not q['grasp_execution_qualified']
def test_support_material_matches_deformx_friction_without_touching_table():
    Usd=pytest.importorskip('pxr.Usd')
    pytest.importorskip('ropesim.sdf')
    from pxr import UsdShade,UsdPhysics,UsdGeom
    class RecordingRope:
        def __init__(self):
            self.calls=[]
        def add_mesh(self,*args,**kwargs):
            self.calls.append(kwargs)
            return 10+len(self.calls)
        def enable_contact_forces(self,enabled):
            pass
    stage=Usd.Stage.CreateInMemory()
    table=UsdGeom.Cube.Define(stage,'/World/ExistingTable').GetPrim()
    oldmat=UsdShade.Material.Define(stage,'/World/OriginalTableMaterial')
    oldapi=UsdPhysics.MaterialAPI.Apply(oldmat.GetPrim())
    oldapi.CreateStaticFrictionAttr(.5);oldapi.CreateDynamicFrictionAttr(.4)
    UsdShade.MaterialBindingAPI.Apply(table).Bind(oldmat,materialPurpose='physics')
    original_table_layer=stage.GetRootLayer().ExportToString()
    rope=RecordingRope()
    result=api.bind_native_end_supports(rope,stage,'/World/envs',plan()['scene'],
                                       np.array([[0,0,0],[3,4,0]]),mu_s=.7,mu_k=.3)
    material_paths=set()
    for row in result['binding_records']:
        for path in row['physx_paths']:
            material,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial('physics')
            assert material, 'support must resolve an explicit physics material'
            assert str(material.GetPath()).startswith('/World/envs/env_'+str(row['env_id'])+'/EndSupports/')
            material_paths.add(str(material.GetPath()))
            physical=UsdPhysics.MaterialAPI(material.GetPrim())
            assert physical.GetStaticFrictionAttr().Get()==pytest.approx(.7)
            assert physical.GetDynamicFrictionAttr().Get()==pytest.approx(.3)
            assert physical.GetRestitutionAttr().Get()==0
            assert material.GetPrim().GetAttribute('physxMaterial:frictionCombineMode').Get()=='average'
            assert material.GetPrim().GetAttribute('dh116:calibrationStatus').Get()=='candidate_not_measured'
    assert len(material_paths)==2
    assert all(c['mu_s']==.7 and c['mu_k']==.3 for c in rope.calls)
    resolved,_=UsdShade.MaterialBindingAPI(table).ComputeBoundMaterial('physics')
    assert resolved.GetPath()==oldmat.GetPath()
    assert oldapi.GetStaticFrictionAttr().Get()==pytest.approx(.5)
    assert oldapi.GetDynamicFrictionAttr().Get()==pytest.approx(.4)

def test_config_settled_reference_and_original_slip_limit():
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/dh116_tasks/flexible_teacher_end_supported_v1.json').read_text())
    assert c['success']['lift_reference']=='settled_initial_min_node_z'
    assert c['success']['slip_m']==.0005