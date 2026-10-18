"""Read-only, explicit-branch USD surface extraction for DH116 calibration.

Output is source triangle geometry, NOT the PhysX cooked collision surface.
"""
import hashlib
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def layer_pins(stage):
    """Hash all used file layers; ignore only an empty anonymous session layer."""
    pins = {}
    for layer in stage.GetUsedLayers():
        if layer == stage.GetSessionLayer() and layer.anonymous and layer.empty:
            continue
        if layer.anonymous or not layer.realPath:
            raise ValueError('cannot pin anonymous or unresolved source layer')
        path = Path(layer.realPath).resolve(strict=True)
        pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(pins.items()))


def read_surface(stage, body_path, branch='collisions'):
    """One mesh under body's explicit visuals/collisions branch, in body metres."""
    if branch not in ('visuals', 'collisions'):
        raise ValueError('branch must be visuals or collisions')
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError('expected metre USD stage')
    body = stage.GetPrimAtPath(body_path)
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('body must exist and be a rigid body')
    prefix = str(body.GetPath()) + '/' + branch + '/'
    meshes = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
              if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith(prefix)]
    if len(meshes) != 1:
        raise ValueError(f'expected one {branch} mesh, found {len(meshes)}')
    prim = meshes[0]
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
            or not np.isfinite(points).all() or counts.ndim != 1
            or len(counts) == 0 or not np.all(counts == 3)
            or indices.shape != (3 * len(counts),)
            or np.any(indices < 0) or np.any(indices >= len(points))):
        raise ValueError('expected finite triangulated geometry with valid indices')
    orientation = str(mesh.GetOrientationAttr().Get())
    if orientation not in ('rightHanded', 'leftHanded'):
        raise ValueError('unknown mesh orientation')
    cache = UsdGeom.XformCache()
    transform = np.array(cache.GetLocalToWorldTransform(prim)
                         * cache.GetLocalToWorldTransform(body).GetInverse()).T
    rotation = transform[:3, :3]
    if (not np.isfinite(transform).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.)
            or abs(np.linalg.det(rotation) - 1.) > 1e-8
            or not np.allclose(transform[3], [0., 0., 0., 1.], atol=1e-8, rtol=0.)):
        raise ValueError('mesh-to-body transform must be rigid, in metres')
    with np.errstate(over='ignore', invalid='ignore'):
        points_body = points @ rotation.T + transform[:3, 3]
    if not np.isfinite(points_body).all():
        raise ValueError('transformed points overflowed')
    owner, approximation = None, None
    ancestor = prim
    while ancestor and ancestor.GetPath().HasPrefix(body.GetPath()):
        if ancestor.HasAPI(UsdPhysics.CollisionAPI):
            if not UsdPhysics.CollisionAPI(ancestor).GetCollisionEnabledAttr().Get():
                raise ValueError('collision owner is disabled')
            owner = str(ancestor.GetPath())
            if ancestor.HasAPI(UsdPhysics.MeshCollisionAPI):
                approximation = str(UsdPhysics.MeshCollisionAPI(ancestor).GetApproximationAttr().Get())
            break
        ancestor = ancestor.GetParent()
    if branch == 'collisions' and owner is None:
        raise ValueError('collision branch has no collision owner')
    digest = hashlib.sha256()
    digest.update(b'DH116-triangle-surface-v1\0')
    digest.update(np.asarray(points.shape, dtype='<i8').tobytes())
    digest.update(points.astype('<f8').tobytes())
    digest.update(indices.astype('<i8').tobytes())
    digest.update(orientation.encode('ascii'))
    return dict(body_path=str(body.GetPath()), mesh_path=str(prim.GetPath()),
                points_mesh=points.tolist(), points_body=points_body.tolist(),
                triangles=indices.reshape(-1, 3).tolist(),
                mesh_to_body=transform.tolist(), collision_owner=owner,
                collision_approximation=approximation, orientation=orientation,
                geometry_sha256=digest.hexdigest())
