import importlib.util
import numpy as np
import pytest
from pxr import Usd, UsdGeom, Tf
from go2_pvcnn.control.fixed_dual_panda_o6.wire_visual import rod_surface


def publisher(mesh):
    name = 'go2_pvcnn.control.dh116.wire_mesh_publisher'
    assert importlib.util.find_spec(name) is not None, 'visual publisher is missing'
    return importlib.import_module(name).WireMeshPublisher(mesh, .008)


def snapshot(nodes=5):
    return {'rod_position': np.vstack((np.zeros(nodes), np.linspace(0, .5, nodes), np.ones(nodes))),
            'rod_director': np.repeat(np.eye(3)[:, :, None], nodes - 1, axis=2)}


def test_points_follow_rod_without_republishing_topology():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, '/Wire')
    publish = publisher(mesh)
    state = snapshot()
    publish(state)
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    assert publish.topology_arrays() == (tuple(counts), tuple(indices))
    changes = []
    notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged,
        lambda event, sender: changes.extend(str(p) for p in event.GetChangedInfoOnlyPaths()), stage)
    state['rod_position'][2] += .02
    publish(state)
    notice.Revoke()
    expected = rod_surface(state['rod_position'], state['rod_director'], .008)[0]
    np.testing.assert_allclose(mesh.GetPointsAttr().Get(), expected, atol=1e-7)
    assert list(mesh.GetFaceVertexCountsAttr().Get()) == counts
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == indices
    assert '/Wire.points' in changes
    assert '/Wire.faceVertexCounts' not in changes
    assert '/Wire.faceVertexIndices' not in changes
    np.testing.assert_allclose(mesh.GetExtentAttr().Get(), [expected.min(0), expected.max(0)], atol=1e-7)


def test_rejects_changed_topology_without_partial_mesh_write():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, '/Wire')
    publish = publisher(mesh)
    publish(snapshot())
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='topology'):
        publish(snapshot(6))
    assert stage.GetRootLayer().ExportToString() == before


def test_rejects_invalid_state_without_partial_mesh_write():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, '/Wire')
    publish = publisher(mesh)
    publish(snapshot())
    before = stage.GetRootLayer().ExportToString()
    invalid = snapshot()
    invalid['rod_position'][0, 0] = np.nan
    with pytest.raises(ValueError):
        publish(invalid)
    assert stage.GetRootLayer().ExportToString() == before
