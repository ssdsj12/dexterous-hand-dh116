import hashlib
import json
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def asset(tmp_path, enabled=True, links=None):
    path = tmp_path / 'hand.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for name in (links or ('base_link', 'finger11_link', 'finger14_link')):
        body = UsdGeom.Xform.Define(stage, '/Hand/' + name)
        body.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        for branch in ('collisions', 'visuals'):
            owner = UsdGeom.Xform.Define(stage, str(body.GetPath()) + '/' + branch + '/owner')
            if branch == 'collisions':
                UsdPhysics.CollisionAPI.Apply(owner.GetPrim()).CreateCollisionEnabledAttr(enabled)
                UsdPhysics.MeshCollisionAPI.Apply(owner.GetPrim()).CreateApproximationAttr('convexHull')
            mesh = UsdGeom.Mesh.Define(stage, str(owner.GetPath()) + '/mesh')
            mesh.CreatePointsAttr([(0,0,0), (1,0,0), (0,1,0)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0,1,2])
            mesh.CreateOrientationAttr('leftHanded')
    stage.GetRootLayer().Save()
    return path


def test_inventory_reads_triangles_owner_winding_and_transforms_without_modification(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import audit_asset
    path = asset(tmp_path)
    before = path.read_bytes()
    report = audit_asset(path, 'left')
    assert path.read_bytes() == before
    assert report['counts']['collision_owners'] == 3
    assert report['counts']['enabled_collision_owners'] == 3
    assert report['counts']['visual_meshes'] == 3
    assert report['counts']['locked_cooked_surfaces'] == 0
    row = next(r for r in report['surfaces'] if r['link'] == 'base_link' and r['branch'] == 'collisions')
    assert row['role'] == 'palm'
    assert row['collision_enabled'] is True
    assert row['triangles_ccw'] == [[0,2,1]]
    assert row['mesh_world'][0][3] == 1.
    assert row['geometry_sha256']
    assert row['blockers'] == ['LOCKED_COOKED_SOURCE_MISSING', 'DEFORMX_REGISTRATION_NOT_OBSERVED']
    assert not row['qualified_for_power']


def test_disabled_collision_and_visual_never_qualify(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import audit_asset
    report = audit_asset(asset(tmp_path, False), 'right')
    assert report['counts']['enabled_collision_owners'] == 0
    for row in report['surfaces']:
        assert row['points_mesh'] and row['triangles']
        assert not row['qualified_for_power']
        if row['branch'] == 'collisions':
            assert 'PHYSX_COLLISION_DISABLED' in row['blockers']
        else:
            assert 'VISUAL_GEOMETRY_ONLY' in row['blockers']


def test_capture_inventory_has_exact_nonempty_bounded_owner_mesh_targets(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import audit_asset, prepare_capture_inventory
    path = asset(tmp_path)
    reports = [audit_asset(path, side) for side in ('left', 'right')]
    plan = prepare_capture_inventory(reports)
    assert len(plan['targets']) == 6
    assert plan['max_owner_queries'] == 6
    assert plan['max_total_count_queries'] == 12
    assert all(t['source_geometry_sha256'] and t['query_roles'] == ['actual_collision_owner', 'actual_mesh_fallback'] for t in plan['targets'])
    assert not plan['execute_native']
    with pytest.raises(ValueError, match='both distinct hands'):
        prepare_capture_inventory([reports[0], reports[0]])


def test_lock_tamper_fails_before_any_native_import(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import load_locked_pads
    lock = tmp_path / 'lock.json'
    lock.write_text('{}')
    with pytest.raises(ValueError, match='SHA256 differs'):
        load_locked_pads(tmp_path, lock, '0'*64)


def test_locked_source_cannot_be_reused_on_different_geometry(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import audit_asset
    path = asset(tmp_path)
    with pytest.raises(ValueError, match='locked cooked source identity differs'):
        audit_asset(path, 'left', locked_pads={'left.base_link': {'source': {'geometry_sha256': 'wrong'}}})


def test_scene_prefix_remap_preserves_same_source_link_and_mesh_identity(tmp_path):
    from copy import deepcopy
    from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import audit_asset
    path = asset(tmp_path)
    original = audit_asset(path, 'left')
    source = next(r for r in original['surfaces'] if r['link']=='base_link' and r['branch']=='collisions')
    captured = deepcopy(source)
    for key in ('body_path','mesh_path','collision_owner'):
        captured[key] = captured[key].replace('/Hand/', '/World/Capture/Left/')
    captured['mesh_to_body'][0][3] += 1e-15
    cooked = {'source': captured, 'source_asset_sha256': original['source_sha256']}
    report = audit_asset(path, 'left', locked_pads={'left.base_link': cooked})
    assert report['counts']['locked_cooked_surfaces'] == 1
    assert not any(r['qualified_for_power'] for r in report['surfaces'])


def test_inventory_captures_owner_frame_and_authored_cooking_settings(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import audit_asset
    report = audit_asset(asset(tmp_path), 'left')
    row = next(r for r in report['surfaces'] if r['branch']=='collisions')
    assert row['owner_world'][0][3] == 1.
    assert row['owner_to_body'][0][3] == 0.
    assert row['cooking_metadata']['prim_path'] == row['collision_owner']
    assert row['deformx_registration_status'] == 'not_queried_by_USD_audit'



def test_native_bindings_map_all_34_actual_owners_without_advancing(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import audit_asset, prepare_power_cooked_bindings
    links = ['base_link'] + ['finger1%d_link'%i for i in (1,2,3,4)] + ['finger%d%d_link'%(f,i) for f in (2,3,4,5) for i in (1,2,3)]
    path = asset(tmp_path, links=links)
    reports = [audit_asset(path, side) for side in ('left','right')]
    names = list(reversed(links))
    bindings = prepare_power_cooked_bindings(reports,
        ['/World/envs/env_0/leftHand', '/World/envs/env_0/rightHand'], [names, names])
    assert len(bindings) == 34
    assert len({b['actual_collision_owner_path'] for b in bindings}) == 34
    for b in bindings:
        link = b['source_body_path'].rsplit('/', 1)[-1]
        assert b['body_id'] == names.index(link)
        assert b['actual_body_path'] == b['hand_root']+'/'+link
        assert b['source_manifest']['verified'] is True
        assert len(b['source_manifest']['collision_meshes']) == 17


def test_unobserved_deformx_registration_count_is_null(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import audit_asset
    report=audit_asset(asset(tmp_path),'left')
    assert report['counts']['deformx_registered_surfaces'] is None
    assert all(r['deformx_registered'] is None for r in report['surfaces'])
