from fixture_paths import ARCHIVE, ASSET, BACKEND, CONTACT, CONTACT_SHA, SURFACE_REFS
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('pxr.Usd')
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from go2_pvcnn.control.dh116.usd_surface import read_surface, layer_pins


def scene():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    body = UsdGeom.Xform.Define(stage, '/body')
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    body.AddTranslateOp().Set((1., 2., 3.))
    owner = UsdGeom.Xform.Define(stage, '/body/collisions/owner')
    owner.AddTranslateOp().Set((.1, .2, .3))
    UsdPhysics.CollisionAPI.Apply(owner.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, '/body/collisions/owner/mesh')
    mesh.CreatePointsAttr([(0., 0., 0.), (.01, 0., 0.), (0., .01, 0.)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    return stage, mesh


def test_surface_uses_body_frame_and_ancestor_collision():
    stage, _ = scene()
    result = read_surface(stage, '/body')
    np.testing.assert_allclose(result['points_body'],
                               [[.1, .2, .3], [.11, .2, .3], [.1, .21, .3]], atol=1e-9)
    assert result['collision_owner'] == '/body/collisions/owner'
    assert len(result['geometry_sha256']) == 64
    assert result['triangles'] == [[0, 1, 2]]


@pytest.mark.parametrize('problem', ['body', 'branch', 'multiple', 'indices', 'quad',
                                    'collision', 'disabled', 'units', 'scale', 'nan'])
def test_invalid_surface_rejected(problem):
    stage, mesh = scene()
    body, branch = '/body', 'collisions'
    if problem == 'body':
        body = '/missing'
    elif problem == 'branch':
        branch = 'anything'
    elif problem == 'multiple':
        UsdGeom.Mesh.Define(stage, '/body/collisions/second')
    elif problem == 'indices':
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 10])
    elif problem == 'quad':
        mesh.GetFaceVertexCountsAttr().Set([4])
    elif problem == 'collision':
        stage.GetPrimAtPath('/body/collisions/owner').RemoveAPI(UsdPhysics.CollisionAPI)
    elif problem == 'disabled':
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/body/collisions/owner')).CreateCollisionEnabledAttr(False)
    elif problem == 'units':
        UsdGeom.SetStageMetersPerUnit(stage, .01)
    elif problem == 'scale':
        UsdGeom.Xformable(mesh).AddScaleOp().Set((2., 1., 1.))
    elif problem == 'nan':
        mesh.GetPointsAttr().Set([(float('nan'), 0., 0.)] * 3)
    with pytest.raises(ValueError):
        read_surface(stage, body, branch)


def test_pins_reject_anonymous_and_pin_actual_bytes(tmp_path):
    import hashlib
    stage, _ = scene()
    with pytest.raises(ValueError):
        layer_pins(stage)
    path = tmp_path / 'scene.usda'
    stage.GetRootLayer().Export(str(path))
    loaded = Usd.Stage.Open(str(path))
    assert layer_pins(loaded) == {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}


def test_actual_instance_proxy_surfaces():
    path = ASSET
    if not path.exists():
        pytest.skip('mounted asset not installed')
    stage = Usd.Stage.Open(str(path))
    assert len(layer_pins(stage)) >= 5
    for side in ('left', 'right'):
        for body in ('base_link', 'finger14_link', 'finger23_link', 'finger33_link', 'finger43_link', 'finger53_link'):
            prefix = f'/FixedDualPandaDH116/{side}_arm/dh116/{body}'
            collision = read_surface(stage, prefix)
            visual = read_surface(stage, prefix, 'visuals')
            assert collision['geometry_sha256'] == visual['geometry_sha256']
            np.testing.assert_allclose(collision['points_body'], visual['points_body'], atol=1e-12)
            assert len(collision['triangles']) > 100


def test_audit_refuses_existing_output_before_loading_asset(tmp_path):
    from scripts.audit_dh116_surfaces import audit
    with pytest.raises(ValueError, match='existing'):
        audit(tmp_path / 'absent.usd', tmp_path)
