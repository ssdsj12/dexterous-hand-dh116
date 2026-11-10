"""CPU USD/getter fixtures: never native attestation or a grasp trial."""
import copy
import hashlib
import importlib
import importlib.util
from types import SimpleNamespace as NS

import numpy as np
import pytest

NAME = 'go2_pvcnn.control.dh116.tabletop_tripod_cooked_geometry'


def api():
    assert importlib.util.find_spec(NAME) is not None, 'read-only cooked geometry collector missing'
    return importlib.import_module(NAME)


def cube():
    vertices = (np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                         [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)*.001).tolist()
    faces = [[0,3,2,1],[4,5,6,7],[0,4,7,3],[1,2,6,5],[0,1,5,4],[3,7,6,2]]
    normals = [[0,0,-1],[0,0,1],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]]
    return dict(num_vertices=8, num_polygons=6, vertices=vertices,
        polygons=[dict(plane=n+[-.001],num_vertices=4,index_base=4*i) for i,n in enumerate(normals)],
        indices=sum(faces, [])), faces


class StoredCookedGetter:
    """Only the installed read methods exist; mutation APIs deliberately absent."""
    def __init__(self):
        self.data = cube()[0]; self.count = 1; self.mesh_count = 0; self.calls = []

    def get_nb_convex_mesh_data(self, path):
        self.calls.append(('count', path))
        return self.count if path.endswith('/owner') else self.mesh_count

    def get_convex_mesh_data(self, path, index):
        self.calls.append(('data', path, index)); return self.data


def scene(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
    from go2_pvcnn.control.dh116.usd_surface import read_surface, layer_pins
    tmp_path.mkdir(parents=True,exist_ok=True)
    stage = Usd.Stage.CreateInMemory(); UsdGeom.SetStageMetersPerUnit(stage, 1.)
    models=[]; roots=[]; body_names=[]; poses={}
    for hand, side in enumerate(('left','right')):
        path=tmp_path/(side+'.usda'); root='/DH_'+side
        source=Usd.Stage.CreateNew(str(path)); UsdGeom.SetStageMetersPerUnit(source, 1.)
        UsdGeom.Xform.Define(source,root)
        for i in range(17):
            body=UsdGeom.Xform.Define(source,f'{root}/body{i}').GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            owner=UsdGeom.Xform.Define(source,f'{root}/body{i}/collisions/owner').GetPrim()
            UsdPhysics.CollisionAPI.Apply(owner)
            UsdPhysics.MeshCollisionAPI.Apply(owner).CreateApproximationAttr('convexHull')
            mesh=UsdGeom.Mesh.Define(source,str(owner.GetPath())+'/mesh')
            data, faces=cube(); triangles=[]
            for a,b,c,d in faces: triangles.extend((a,b,c,a,c,d))
            mesh.CreatePointsAttr(data['vertices']); mesh.CreateFaceVertexCountsAttr([3]*12)
            mesh.CreateFaceVertexIndicesAttr(triangles)
        source.GetRootLayer().Save()
        records=[read_surface(source,f'{root}/body{i}') for i in range(17)]
        manifest=dict(verified=True,source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            layer_sha256=layer_pins(source),meters_per_unit=1.,source_collision_count=17,
            collision_meshes=[{k:r[k] for k in ('body_path','mesh_path','mesh_to_body','collision_owner',
                'collision_approximation','orientation','geometry_sha256')} for r in records])
        models.append(NS(spec=dict(side=side,root=root,source_manifest=manifest)))
        actual=f'/World/envs/env_4/{side}Hand'; roots.append(actual)
        UsdGeom.Xform.Define(stage,actual).GetPrim().GetReferences().AddReference(str(path),root)
        names=[f'body{i}' for i in range(17)]; body_names.append(names)
        poses[side]={f'{root}/{name}':np.eye(4).tolist() for name in names}
    snapshot=dict(env_id=4,episode_id=0,tensor_world_physics_steps=12,sdk_step_index=14,
        control_step=2,control_substep=4,simulation_time_s=14/480,
        source_body_poses_w=poses,provenance={'fixture':True,'native_attested':True})
    return stage,models,roots,body_names,snapshot,StoredCookedGetter()


def collect(fixture, **kwargs):
    m=api(); stage,models,roots,names,snapshot,getter=fixture
    bindings=m.prepare_cooked_bindings(models,roots,names)
    return m.collect_cooked_geometry(stage,getter,bindings,snapshot,simulation_running=kwargs.get('running',True))


def test_complete_34_original_colliders_are_owned_readonly_and_never_native_attested(tmp_path):
    fixture=scene(tmp_path); stage,_,_,_,snapshot,getter=fixture
    before=stage.GetRootLayer().ExportToString()
    snapshot['source_body_poses_w']['left']['/DH_left/body0'][0][3]=.123
    result=collect(fixture)
    assert result['complete'] and not result['censored'],result['reasons']
    assert len(result['colliders'])==34 and len(getter.calls)==68
    assert result['native_attested'] is False and result['live_actor_shape_attachment_proven'] is False
    assert stage.GetRootLayer().ExportToString()==before
    row=result['colliders'][0]
    assert row['selected_query_role']=='actual_collision_owner'
    assert len(row['cooked_query_attempts'])==1
    assert row['cooked_query_attempts'][0]['path']==row['binding']['actual_collision_owner_path']
    assert row['cooked_hulls'][0]['validation']['closed_convex']
    assert row['cooked_hulls'][0]['world_vertices_from_actual_body_pose'][0][0]==pytest.approx(.122)
    np.testing.assert_allclose(row['source_geometry']['original_mesh_points'],cube()[0]['vertices'])
    assert row['capture_clock']['tensor_world_physics_steps']==12
    getter.data['vertices'][0][0]=99; snapshot['env_id']=8
    assert result['colliders'][0]['cooked_hulls'][0]['raw']['vertices'][0][0]==-.001
    assert result['capture_clock']['env_id']==4


def test_kinematic_root_may_be_below_common_source_asset_root(tmp_path):
    fixture=scene(tmp_path)
    # SourceHandGeometry.spec['root'] is the kinematic base_link. Collision
    # meshes also include sibling links under the hand asset root.
    for model in fixture[1]:
        model.spec['root']=model.spec['root']+'/body0'
    result=collect(fixture)
    assert result['complete'], result['reasons']


@pytest.mark.parametrize('running',[False,None,1,np.bool_(True)])
def test_unknown_or_stopped_simulation_is_censored_without_getter_calls(tmp_path,running):
    fixture=scene(tmp_path); result=collect(fixture,running=running)
    assert not result['complete'] and result['censored'] and not fixture[-1].calls
    assert 'SIMULATION_RUNNING_NOT_CONFIRMED' in result['reasons']


def test_missing_duplicate_and_wrong_source_ownership_are_rejected(tmp_path):
    m=api(); stage,models,roots,names,snapshot,getter=scene(tmp_path)
    bindings=m.prepare_cooked_bindings(models,roots,names)
    for bad in (bindings[:-1],bindings[:-1]+[bindings[0]],
                [dict(bindings[0],actual_collision_owner_path=roots[1]+'/body0/collisions/owner')]+bindings[1:]):
        result=m.collect_cooked_geometry(stage,getter,bad,snapshot,simulation_running=True)
        assert not result['complete'] and result['censored']
    assert not getter.calls


def test_enabled_actual_scene_colliders_must_equal_all_34_bindings(tmp_path):
    from pxr import UsdGeom, UsdPhysics
    fixture=scene(tmp_path); stage,_,roots,*_=fixture
    extra=UsdGeom.Cube.Define(stage,roots[0]+'/unexpected').GetPrim(); UsdPhysics.CollisionAPI.Apply(extra)
    result=collect(fixture)
    assert not result['complete'] and any('COLLIDER_SET' in r for r in result['reasons'])
    assert not fixture[-1].calls


def test_disabled_actual_owner_cannot_be_treated_as_measured_empty_geometry(tmp_path):
    from pxr import UsdPhysics
    fixture=scene(tmp_path); stage,_,roots,*_=fixture
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath(roots[0]+'/body0/collisions/owner')).GetCollisionEnabledAttr().Set(False)
    result=collect(fixture)
    assert not result['complete'] and result['censored'] and not fixture[-1].calls


@pytest.mark.parametrize('count',[0,2,True,-1])
def test_hull_count_must_be_valid_and_match_actual_convex_hull_approximation(tmp_path,count):
    fixture=scene(tmp_path); fixture[-1].count=count
    result=collect(fixture)
    assert not result['complete'] and result['censored']
    assert all(not row['complete'] for row in result['colliders'])


def test_owner_zero_then_mesh_zero_is_censored_with_both_exact_queries(tmp_path):
    fixture=scene(tmp_path);fixture[-1].count=0;fixture[-1].mesh_count=0
    result=collect(fixture)
    assert not result['complete'] and result['censored']
    row=result['colliders'][0]
    assert row['selected_query_path'] is None and row['selected_query_role'] is None
    assert [attempt['role'] for attempt in row['cooked_query_attempts']]==[
        'actual_collision_owner','actual_mesh']
    assert all(attempt['count']==0 for attempt in row['cooked_query_attempts'])


def test_owner_zero_then_mesh_positive_selects_mesh_and_records_both_queries(tmp_path):
    fixture=scene(tmp_path);fixture[-1].count=0;fixture[-1].mesh_count=1
    result=collect(fixture)
    assert result['complete']
    row=result['colliders'][0]
    assert row['selected_query_role']=='actual_mesh'
    assert [attempt['count'] for attempt in row['cooked_query_attempts']]==[0,1]


def test_owner_query_is_primary_even_when_mesh_query_would_be_zero(tmp_path):
    fixture=scene(tmp_path);fixture[-1].count=1;fixture[-1].mesh_count=0
    result=collect(fixture)
    assert result['complete']
    assert all(row['selected_query_role']=='actual_collision_owner' for row in result['colliders'])
    assert all(len(row['cooked_query_attempts'])==1 for row in result['colliders'])


@pytest.mark.parametrize('damage',['index','open','plane','nonfinite','count','duplicate_vertex'])
def test_malformed_or_nonclosed_convex_data_is_retained_but_censored(tmp_path,damage):
    fixture=scene(tmp_path); data=fixture[-1].data
    if damage=='index':data['indices'][0]=999
    if damage=='open':data['indices'][0]=1
    if damage=='plane':data['polygons'][0]['plane'][3]=0
    if damage=='nonfinite':data['vertices'][0][0]=float('nan')
    if damage=='count':data['num_vertices']=True
    if damage=='duplicate_vertex':data['vertices'][0]=data['vertices'][1][:]
    result=collect(fixture)
    assert not result['complete'] and result['censored']
    assert result['colliders'][0]['cooked_hulls'][0]['raw']
    assert not result['colliders'][0]['cooked_hulls'][0]['validation']['closed_convex']


def test_unknown_actual_frame_or_clock_is_censored_before_query(tmp_path):
    fixture=scene(tmp_path); fixture[-2]['source_body_poses_w']['right']['/DH_right/body0'][0][0]=2
    result=collect(fixture)
    assert not result['complete'] and not fixture[-1].calls
    fixture=scene(tmp_path/'second'); fixture[-2]['tensor_world_physics_steps']=True
    assert not collect(fixture)['complete'] and not fixture[-1].calls


def test_original_source_pin_or_live_mesh_change_is_censored(tmp_path):
    from pxr import UsdGeom
    fixture=scene(tmp_path); stage,models,roots,*_=fixture
    points=UsdGeom.Mesh(stage.GetPrimAtPath(roots[0]+'/body0/collisions/owner/mesh')).GetPointsAttr()
    changed=points.Get(); vertex=list(changed[0]); vertex[0]=.02; changed[0]=vertex; points.Set(changed)
    result=collect(fixture)
    assert not result['complete'] and any('SOURCE' in r for r in result['colliders'][0]['reasons'])


def test_failed_sdk_getter_does_not_invent_vertices_or_native_success(tmp_path):
    fixture=scene(tmp_path)
    def fail(*args):raise RuntimeError('fixture API unavailable')
    fixture[-1].get_convex_mesh_data=fail
    result=collect(fixture)
    assert not result['complete'] and result['censored']
    assert all(row['cooked_hulls']==[] for row in result['colliders'])
    assert not result['native_attested'] and not result['live_actor_shape_attachment_proven']


def test_mesh_to_body_transform_mismatch_is_not_silently_absorbed(tmp_path):
    from pxr import UsdGeom, Gf
    fixture=scene(tmp_path); stage,_,roots,*_=fixture
    UsdGeom.Xformable(stage.GetPrimAtPath(roots[0]+'/body0/collisions/owner')).AddTranslateOp().Set(Gf.Vec3d(.001,0,0))
    result=collect(fixture)
    assert not result['complete'] and any('TRANSFORM' in r for r in result['colliders'][0]['reasons'])


def test_snapshot_env_identity_must_match_actual_hand_roots(tmp_path):
    fixture=scene(tmp_path);fixture[-2]['env_id']=5
    result=collect(fixture)
    assert not result['complete'] and not fixture[-1].calls


@pytest.mark.parametrize('field,value',[('simulation_time_s',True),('control_substep',9)])
def test_clock_cannot_accept_boolean_time_or_impossible_substep(tmp_path,field,value):
    fixture=scene(tmp_path);fixture[-2][field]=value
    result=collect(fixture)
    assert not result['complete'] and not fixture[-1].calls


def test_source_file_pin_is_checked_not_just_caller_verified_flag(tmp_path):
    fixture=scene(tmp_path)
    source=fixture[1][0].spec['source_manifest']['source_path']
    with open(source,'a') as stream:stream.write('\n# changed after source model freeze\n')
    result=collect(fixture)
    assert not result['complete'] and any('SOURCE_ASSET_SHA' in r for r in result['colliders'][0]['reasons'])


def test_finite_raw_values_with_nonfinite_derived_geometry_are_censored():
    from scipy.spatial.transform import Rotation
    m=api();data=cube()[0]
    rotation=Rotation.from_euler('XYZ',[.3,.4,.5]).as_matrix()
    data['vertices']=(np.asarray(data['vertices'])@rotation.T*1e300).tolist()
    for polygon in data['polygons']:
        polygon['plane'][:3]=(rotation@polygon['plane'][:3]).tolist()
        polygon['plane'][3]*=1e300
    with np.errstate(all='ignore'):
        result=m._validate_hull(data)
    assert not result['closed_convex'] and result['reasons']
