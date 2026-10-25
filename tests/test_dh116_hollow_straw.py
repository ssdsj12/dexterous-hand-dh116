"""Source geometry checks; these do not claim cooked PhysX qualification."""
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.hollow_straw')


def test_default_mass_and_y_axis_inertia_are_hollow_cylinder_values():
    p = module().straw_parameters()
    expected = 900*np.pi*(.008**2-.0075**2)*.5
    assert p['mass_kg'] == pytest.approx(expected)
    assert p['mass_kg'] < .011
    assert p['axis'] == 'Y'
    ix = expected*(3*(.008**2+.0075**2)+.5**2)/12
    iy = expected*(.008**2+.0075**2)/2
    assert p['diagonal_inertia_kg_m2'] == pytest.approx([ix, iy, ix])
    assert p['inner_polygon_inradius_m'] == pytest.approx(.0075*np.cos(np.pi/32))


@pytest.mark.parametrize('change', [dict(outer_radius_m=0),dict(wall_thickness_m=.008),
    dict(wall_thickness_m=-.001),dict(length_m=0),dict(density_kg_m3=float('nan')),
    dict(density_kg_m3=-1),dict(collision_segments=3),dict(collision_segments=32.5)])
def test_invalid_geometry_is_rejected(change):
    with pytest.raises(ValueError):
        module().straw_parameters(**change)


def test_each_convex_wedge_stays_outside_open_bore():
    p = module().straw_parameters()
    wedges = module().collision_wedges(p)
    assert len(wedges) == 32
    for i, (points, faces) in enumerate(wedges):
        points = np.asarray(points)
        assert points.shape == (8,3)
        assert len(faces) == 6 and all(len(face) == 4 for face in faces)
        radius = np.linalg.norm(points[:,[0,2]],axis=1)
        assert np.count_nonzero(np.isclose(radius,.008)) == 4
        assert np.count_nonzero(np.isclose(radius,.0075)) == 4
        angle = (i+.5)*2*np.pi/32
        # Every point in this convex hull satisfies this half-space, so its
        # interior cannot fill the bore, unlike one convex hull of the tube.
        radial_projection = points[:,[0,2]] @ [np.cos(angle),np.sin(angle)]
        assert radial_projection.min() >= p['inner_polygon_inradius_m']-1e-12
        assert set(points[:,1]) == {-.25,.25}


def test_visual_mesh_is_closed_annular_wall_with_open_bore():
    points, faces = module().visual_tube(module().straw_parameters())
    points = np.asarray(points)
    assert points.shape == (128,3)
    assert len(faces) == 128
    edges = Counter()
    for face in faces:
        assert len(face) == 4
        for a,b in zip(face,face[1:]+face[:1]):
            edges[tuple(sorted((a,b)))] += 1
        face_points = points[face]
        if np.ptp(face_points[:,1]) == 0:
            # End faces are annular quads with exactly two inner and outer
            # vertices; there is no cap triangle fan through the center.
            assert np.sort(np.linalg.norm(face_points[:,[0,2]],axis=1)) == pytest.approx([.0075,.0075,.008,.008])
    assert set(edges.values()) == {2}
    assert len(points)-len(edges)+len(faces) == 0  # torus topology


def test_usd_has_one_body_and_32_separate_convex_colliders(tmp_path):
    report = module().build_straw(tmp_path/'asset')
    stage = Usd.Stage.Open(str(tmp_path/'asset/straw.usda'))
    root = stage.GetDefaultPrim()
    assert str(root.GetPath()) == '/Straw'
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not root.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(bodies) == 1 and len(colliders) == 32
    for prim in colliders:
        assert prim.IsA(UsdGeom.Mesh)
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'convexHull'
        assert len(UsdGeom.Mesh(prim).GetPointsAttr().Get()) == 8
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    visual = stage.GetPrimAtPath('/Straw/Visual')
    assert visual.IsA(UsdGeom.Mesh) and not visual.HasAPI(UsdPhysics.CollisionAPI)
    mass = UsdPhysics.MassAPI(root)
    assert mass.GetMassAttr().Get() == pytest.approx(report['mass_kg'])
    assert list(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx(report['diagonal_inertia_kg_m2'])
    assert list(mass.GetCenterOfMassAttr().Get()) == pytest.approx([0,0,0])
    assert root.GetAttribute('straw:axis').Get() == 'Y'
    assert not [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.MaterialAPI)]


def test_manifest_pins_parameters_asset_and_generator_and_refuses_overwrite(tmp_path):
    out = tmp_path/'asset'
    report = module().build_straw(out)
    assert report == json.loads((out/'manifest.json').read_text())
    assert report['usd_sha256'] == hashlib.sha256((out/'straw.usda').read_bytes()).hexdigest()
    for record in report['generators']:
        assert record['sha256'] == hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()
    assert report['physics_verified'] is False
    assert report['wall_thickness_m'] == .0005
    with pytest.raises(ValueError,match='overwrite'):
        module().build_straw(out)
