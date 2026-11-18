"""Read-only Power surface inventory; source triangles are never cooked hulls.

No Isaac app, CUDA, cooking, actuator edits, registration or contact synthesis.
Native availability below describes resolved USD CollisionAPI, not live PxShape.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .deformx_dh116_colliders import DISTAL_PAD_LINKS


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _role(link):
    if link == 'base_link':
        return 'palm'
    return 'distal' if link in DISTAL_PAD_LINKS else 'proximal_or_middle'


def load_locked_pads(capture_dir, lock_path, expected_lock_sha256):
    """Authenticate existing F4 packets on CPU using the original evidence reader."""
    from scripts.bake_dh116_deformx_pad_sdfs import checked_json, input_packets
    lock_path = Path(lock_path)
    lock = checked_json(lock_path, expected_lock_sha256)
    packets = input_packets(Path(capture_dir), lock)
    gpu = checked_json(Path(capture_dir) / 'gpu/result.json', lock['gpu_result_sha256'])
    result = {}
    for key, (row, hull, delta, digest) in packets.items():
        result[key] = dict(source=row['source'], hull=hull, basis_delta=delta,
                           source_asset_sha256=gpu['source_usds'][row['side']]['sha256'],
                           capture_sha256=digest,
                           capture_path=str(Path(capture_dir) / ('gpu/' + key.replace('.', '_') + '.json')),
                           evidence_lock_path=str(lock_path), evidence_lock_sha256=expected_lock_sha256)
    return result


def audit_asset(path, side, *, locked_pads=None):
    """Inventory every mesh/body and CollisionAPI owner, including instance proxies.

    locked_pads must be the output of load_locked_pads. A locked cooked source
    alone cannot qualify Power: registration and reaction forces remain absent.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from .usd_surface import layer_pins, read_surface
    from .tabletop_tripod_cooked_geometry import _cook_metadata
    if side not in ('left', 'right'):
        raise ValueError('left or right hand required')
    path = Path(path).resolve(strict=True)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError('source USD could not open')
    pins = layer_pins(stage)
    before = {l.identifier: l.ExportToString() for l in stage.GetLayerStack()}
    cache = UsdGeom.XformCache()
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    owners = []
    for p in prims:
        if p.HasAPI(UsdPhysics.CollisionAPI):
            attr = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr()
            owners.append(dict(path=str(p.GetPath()), enabled=attr.Get(),
                               enabled_authored=attr.HasAuthoredValueOpinion(),
                               instance_proxy=p.IsInstanceProxy(),
                               approximation=str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get())
                               if p.HasAPI(UsdPhysics.MeshCollisionAPI) else None))
    surfaces = []
    for p in prims:
        if not p.IsA(UsdGeom.Mesh):
            continue
        body = p
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            raise ValueError('mesh has no rigid body: ' + str(p.GetPath()))
        body_path = str(body.GetPath())
        branch = str(p.GetPath())[len(body_path)+1:].split('/')[0]
        owner = p
        while owner and owner.GetPath().HasPrefix(body.GetPath()) and not owner.HasAPI(UsdPhysics.CollisionAPI):
            owner = owner.GetParent()
        has_owner = bool(owner and owner.GetPath().HasPrefix(body.GetPath()) and owner.HasAPI(UsdPhysics.CollisionAPI))
        enabled = UsdPhysics.CollisionAPI(owner).GetCollisionEnabledAttr().Get() if has_owner else None
        mesh = UsdGeom.Mesh(p)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not len(counts) or not np.all(counts == 3) or indices.shape != (3*len(counts),)
            or indices.min() < 0 or indices.max() >= len(points)):
            raise ValueError('finite original triangles required: '+str(p.GetPath()))
        orientation = str(mesh.GetOrientationAttr().Get())
        if orientation not in ('leftHanded', 'rightHanded'):
            raise ValueError('unknown winding')
        transform = np.asarray(cache.GetLocalToWorldTransform(p)*cache.GetLocalToWorldTransform(body).GetInverse()).T
        digest = hashlib.sha256()
        digest.update(b'DH116-triangle-surface-v1\0')
        digest.update(np.asarray(points.shape, dtype='<i8').tobytes())
        digest.update(points.astype('<f8').tobytes())
        digest.update(indices.astype('<i8').tobytes())
        digest.update(orientation.encode('ascii'))
        triangles = indices.reshape(-1,3)
        blockers = []
        record = None
        if branch == 'collisions' and enabled is True:
            record = read_surface(stage, body_path)
            if record['mesh_path'] != str(p.GetPath()) or record['geometry_sha256'] != digest.hexdigest():
                raise ValueError('reused source reader disagrees')
        else:
            blockers.append('VISUAL_GEOMETRY_ONLY' if branch == 'visuals' else 'PHYSX_COLLISION_DISABLED')
        locked = (locked_pads or {}).get(side+'.'+body.GetName()) if record else None
        if locked:
            captured = locked['source']
            source_body = captured.get('body_path', '')
            same = (locked.get('source_asset_sha256') == _sha(path)
                    and source_body.rsplit('/', 1)[-1] == body.GetName()
                    and all(captured.get(k) == record[k] for k in
                            ('geometry_sha256', 'collision_approximation', 'orientation'))
                    and all(captured.get(k, '').startswith(source_body + '/')
                            and captured[k][len(source_body):] == record[k][len(body_path):]
                            for k in ('mesh_path', 'collision_owner'))
                    and np.allclose(captured.get('mesh_to_body'), record['mesh_to_body'], atol=1e-8, rtol=0))
            if not same:
                raise ValueError('locked cooked source identity differs: '+str(p.GetPath()))
        if not locked:
            blockers.append('LOCKED_COOKED_SOURCE_MISSING')
        blockers.append('DEFORMX_REGISTRATION_NOT_OBSERVED')
        surfaces.append(dict(side=side, link=body.GetName(), role=_role(body.GetName()),
            body_path=body_path, mesh_path=str(p.GetPath()), branch=branch,
            collision_owner=str(owner.GetPath()) if has_owner else None, collision_enabled=enabled,
            collision_approximation=record['collision_approximation'] if record else None,
            owner_world=np.asarray(cache.GetLocalToWorldTransform(owner)).T.tolist() if has_owner else None,
            owner_to_body=np.asarray(cache.GetLocalToWorldTransform(owner)*cache.GetLocalToWorldTransform(body).GetInverse()).T.tolist() if has_owner else None,
            cooking_metadata=_cook_metadata(owner) if has_owner else None,
            deformx_registration_status='not_queried_by_USD_audit',
            instance_proxy=p.IsInstanceProxy(), orientation=orientation,
            points_mesh=points.tolist(), triangles=triangles.tolist(),
            triangles_ccw=(triangles[:,[0,2,1]] if orientation == 'leftHanded' else triangles).tolist(),
            geometry_sha256=digest.hexdigest(), mesh_to_body=transform.tolist(),
            body_world=np.asarray(cache.GetLocalToWorldTransform(body)).T.tolist(),
            mesh_world=np.asarray(cache.GetLocalToWorldTransform(p)).T.tolist(),
            transform_convention='column_vectors_stage_units',
            locked_cooked=locked, deformx_registered=None, live_shape_attachment_proven=False,
            qualified_for_power=False, blockers=blockers))
    unchanged = pins == layer_pins(stage) and before == {l.identifier:l.ExportToString() for l in stage.GetLayerStack()}
    if not unchanged:
        raise ValueError('source layers changed during audit')
    return dict(schema='dh116-teacher-power-surface-audit-v1', side=side,
        source_path=str(path), source_sha256=_sha(path), layer_sha256=pins,
        meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage), up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        source_unchanged=unchanged, native_started=False, collision_owners=owners, surfaces=surfaces,
        counts=dict(rigid_bodies=sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims),
                    collision_owners=len(owners), enabled_collision_owners=sum(o['enabled'] is True for o in owners),
                    collision_meshes=sum(r['branch']=='collisions' for r in surfaces),
                    visual_meshes=sum(r['branch']=='visuals' for r in surfaces),
                    locked_cooked_surfaces=sum(r['locked_cooked'] is not None for r in surfaces),
                    deformx_registered_surfaces=None))


def prepare_capture_inventory(reports):
    """Exact nonempty capture targets, NOT live bindings or fabricated hulls.

    Consumer must map the source suffix under actual env_0 leftHand/rightHand,
    resolve SDK body IDs and clock, then use collect_cooked_geometry. Owner first,
    mesh fallback only on unavailable count. No request/cook/force-load is allowed.
    """
    if len(reports) != 2 or {r['side'] for r in reports} != {'left','right'}:
        raise ValueError('both distinct hands required')
    targets = []
    for report in reports:
        for row in report['surfaces']:
            if row['branch'] != 'collisions' or row['collision_enabled'] is not True:
                continue
            targets.append(dict(side=report['side'], link=row['link'], role=row['role'],
                source_path=report['source_path'], source_asset_sha256=report['source_sha256'],
                source_layer_sha256=report['layer_sha256'], source_body_path=row['body_path'],
                source_collision_owner=row['collision_owner'], source_mesh_path=row['mesh_path'],
                source_geometry_sha256=row['geometry_sha256'], mesh_to_body=row['mesh_to_body'],
                query_roles=['actual_collision_owner', 'actual_mesh_fallback'],
                actual_hand_root='/World/envs/env_0/'+report['side']+'Hand',
                actual_paths_and_sdk_body_id_require_runtime_binding=True,
                locked_cooked_already_available=row['locked_cooked'] is not None))
    if not targets or len({(r['side'],r['source_collision_owner']) for r in targets}) != len(targets):
        raise ValueError('nonempty unique collision owners required')
    return dict(schema='dh116-power-bounded-native-capture-input-v1', execute_native=False,
        targets=targets, max_environments=1, max_owner_queries=len(targets),
        max_total_count_queries=2*len(targets), max_physics_steps=0,
        collector='go2_pvcnn.control.dh116.tabletop_tripod_cooked_geometry.collect_cooked_geometry',
        requirements=['existing uniquely owned already-running Isaac only',
            'bind actual SDK body IDs and measured world_from_body poses at one capture clock',
            'verify exact actual 34 enabled CollisionAPI owners with prepare_cooked_bindings',
            'stored cooked getter only; no cooking, stepping, physics edits or synthetic hulls',
            'zero/error counts remain censored; separately prove live shape attachment',
            'authenticate capture exit codes, geometry, frame, device parity before SDF bake',
            'register palm and proximal/middle SDFs then measure actual reaction forces'])





def prepare_power_cooked_bindings(reports, actual_hand_roots, body_names):
    """Build exact existing collector bindings from audited all-17 source rows.

    Call once after the SAME diagnostic's two setup steps, without another
    update/step. actual_hand_roots are explicit live env_0 leftHand/rightHand
    prims; body_names are their actual SDK body-name arrays in SDK index order.
    Returns 34 owner/mesh/body path bindings; does not acquire any SDK interface.
    Pass these to collect_cooked_geometry with that diagnostic's capture clock
    and measured body-link world poses keyed by source_body_path.
    """
    from types import SimpleNamespace
    from pxr import Usd
    from .usd_surface import layer_pins
    from .tabletop_tripod_cooked_geometry import prepare_cooked_bindings
    if len(reports) != 2 or [r['side'] for r in reports] != ['left', 'right']:
        raise ValueError('both distinct hands in left/right order required')
    models = []
    for report in reports:
        source_path = report['source_path']
        if (report['source_unchanged'] is not True or _sha(source_path) != report['source_sha256']
            or layer_pins(Usd.Stage.Open(source_path)) != report['layer_sha256']
            or report['meters_per_unit'] != 1. or report['up_axis'] != 'Z'):
            raise ValueError('audited source changed or invalid units/axis')
        rows = [r for r in report['surfaces'] if r['branch']=='collisions' and r['collision_enabled'] is True]
        if len(rows) != 17 or report['counts']['collision_owners'] != 17:
            raise ValueError('exact original 17 enabled owners per hand required')
        keys = ('body_path','mesh_path','collision_owner','collision_approximation',
                'orientation','geometry_sha256','mesh_to_body')
        manifest = dict(verified=True, source_path=source_path,
            source_sha256=report['source_sha256'], layer_sha256=report['layer_sha256'],
            meters_per_unit=1., source_collision_count=17,
            collision_meshes=[{k:r[k] for k in keys} for r in rows])
        base = next(r['body_path'] for r in rows if r['link']=='base_link')
        models.append(SimpleNamespace(spec=dict(side=report['side'], root=base, source_manifest=manifest)))
    return prepare_cooked_bindings(models, actual_hand_roots, body_names)
