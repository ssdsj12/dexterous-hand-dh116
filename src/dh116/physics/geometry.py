"""Common vector/hull/rod/pad geometry and explicit CUDA measurement backend."""
import numpy as np
import torch
from scipy.optimize import minimize
import numpy as defor_dh116_colli__np
import numpy as geome_math__np
import numpy as batch_geome__np
from dataclasses import dataclass as cuda_geome__dataclass
import hashlib as polyl_suppo_certi__hashlib
from copy import deepcopy as tenso_geome__deepcopy
from copy import deepcopy as tenso_mesh_bridg__deepcopy
DISTAL_PAD_LINKS = ('finger14_link', 'finger23_link', 'finger33_link', 'finger43_link', 'finger53_link')
from scipy.spatial import ConvexHull, QhullError

def normalized_convex(points):
    """Cook a well-scaled hull, then restore identical geometry by mesh scale."""
    points = batch_geome__np.asarray(points, dtype=float)
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    scale = batch_geome__np.ptp(points, axis=0) / 2
    if batch_geome__np.any(scale <= 0) or not batch_geome__np.isfinite(points).all():
        raise ValueError('finite nondegenerate convex vertices required')
    return ((points - center) / scale, center, scale)
import hashlib as cuda_geome__hashlib
import math as polyl_suppo_certi__math
from dataclasses import dataclass as tenso_geome__dataclass
import math as tenso_mesh_bridg__math
HAND_SCENE_KEYS = {'left': 'left_hand', 'right': 'right_hand'}
from scipy.spatial.transform import Rotation

def rotated_points(poses, points):
    (poses, points) = (batch_geome__np.asarray(poses), batch_geome__np.asarray(points))
    xyz = poses[..., None, 4:]
    cross = 2 * batch_geome__np.cross(xyz, points)
    return points + poses[..., None, 3:4] * cross + batch_geome__np.cross(xyz, cross) + poses[..., None, :3]
import json
import numpy as polyl_suppo_certi__np
import math as tenso_geome__math
import torch as tenso_mesh_bridg__torch
_GRASP_LINK_COUNTS = {'pinch': 2, 'tripod': 3, 'power': 5}
geome_math__NUMERIC_GUARD = 2e-06

def rotated_heights(poses, points):
    """Exact z component of the existing quaternion transform, for every vertex.

    Avoid allocating two unused coordinates and cross-product temporaries for
    thousands of environments. Float64 matches the original hull arithmetic.
    No hull subsampling, bounding sphere or quaternion normalization is used.
    """
    (poses, points) = (batch_geome__np.asarray(poses, dtype=float), batch_geome__np.asarray(points, dtype=float))
    (w, x, y, z) = batch_geome__np.moveaxis(poses[..., 3:], -1, 0)
    row = batch_geome__np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), axis=-1)
    return batch_geome__np.einsum('...i,vi->...v', row, points) + poses[..., 2, None]
import math as cuda_geome__math
import os as tenso_geome__os

class CudaTensorView:
    """Hold a live default-stream torch tensor for the existing CAI binding."""

    def __init__(self, tensor):
        if not isinstance(tensor, tenso_mesh_bridg__torch.Tensor) or tensor.device.type != 'cuda' or tensor.dtype != tenso_mesh_bridg__torch.float32 or (not tensor.is_contiguous()):
            raise ValueError('contiguous CUDA float32 target buffer required')
        if tenso_mesh_bridg__torch.cuda.current_stream(tensor.device) != tenso_mesh_bridg__torch.cuda.default_stream(tensor.device):
            raise ValueError('existing solver ABI requires original legacy/default CUDA stream')
        self.tensor = tensor
        self.__cuda_array_interface__ = dict(shape=tuple(tensor.shape), strides=None, typestr='<f4', data=(tensor.data_ptr(), False), version=3, stream=1)

def collider_plan(mode, *, fixture_asset=None):
    """ColliderSpec identities for one continuous rod per environment."""
    if mode in ('left_thumb', 'right_thumb'):
        sides = (mode.split('_')[0],)
        bodies = DISTAL_PAD_LINKS[:1]
    elif mode in _GRASP_LINK_COUNTS:
        sides = tuple(HAND_SCENE_KEYS)
        bodies = DISTAL_PAD_LINKS[:_GRASP_LINK_COUNTS[mode]]
    else:
        raise ValueError(f'unknown DH116 collider plan: {mode}')
    specs = [{'asset': HAND_SCENE_KEYS[side], 'bodies': list(bodies), 'dynamic': True} for side in sides]
    if fixture_asset is not None:
        if not isinstance(fixture_asset, str) or not fixture_asset or fixture_asset in HAND_SCENE_KEYS.values():
            raise ValueError('fixture must name a separate rigid asset')
        specs.append({'asset': fixture_asset, 'bodies': None, 'dynamic': False})
    return {'task_kind': 'deformable_rod', 'rods_per_env': 1, 'specs': specs}

def finite_array(value, shape=None):
    a = geome_math__np.asarray(value, dtype=float)
    if not geome_math__np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError('finite array with required shape expected')
    return a

def collision_clearance(poses, shapes):
    heights = batch_geome__np.stack([rotated_heights(poses[:, hand, body], points).min(axis=-1) for (hand, body, points) in shapes], axis=1)
    per_hand = batch_geome__np.stack([heights[:, [i for (i, s) in enumerate(shapes) if s[0] == h]].min(axis=1) for h in range(2)], axis=1)
    return (per_hand - 0.3, heights.argmin(axis=1))
import os as cuda_geome__os
import numpy as tenso_geome__np

class TensorMeshBridge:

    def __init__(self, mesh_names, bindings, initial_positions, initial_quaternions):
        if initial_positions.ndim != 3 or initial_positions.shape[1:] != (len(mesh_names), 3) or initial_quaternions.shape != (*initial_positions.shape[:2], 4) or (initial_positions.device != initial_quaternions.device) or (len(set(mesh_names)) != len(mesh_names)):
            raise ValueError('complete same-device initial mesh layout required')
        if not all((bool(tenso_mesh_bridg__torch.isfinite(t).all()) for t in (initial_positions, initial_quaternions))):
            raise ValueError('finite initial mesh poses required')
        if not tenso_mesh_bridg__torch.allclose(initial_quaternions.norm(dim=-1), tenso_mesh_bridg__torch.ones_like(initial_quaternions[..., 0]), atol=1e-05, rtol=0):
            raise ValueError('unit initial XYZW mesh quaternions required')
        self.names = tuple(mesh_names)
        self.n = len(initial_positions)
        self.positions = initial_positions.detach().float().clone().contiguous()
        self.quaternions = initial_quaternions.detach().float().clone().contiguous()
        slots = []
        sides = []
        links = []
        for (name, row) in bindings.items():
            if name not in self.names or row['side'] not in ('left', 'right') or (not 0 <= row['body_index'] < 23):
                raise ValueError('actual original contact link binding required')
            slots.append(self.names.index(name))
            sides.append(0 if row['side'] == 'left' else 1)
            links.append(row['body_index'])
        dev = self.positions.device
        self.slots = tenso_mesh_bridg__torch.tensor(slots, device=dev)
        self.sides = tenso_mesh_bridg__torch.tensor(sides, device=dev)
        self.links = tenso_mesh_bridg__torch.tensor(links, device=dev)
        self._material = self._resets = None
        self._views = None

    @tenso_mesh_bridg__torch.no_grad()
    def targets(self, link_poses, *, validate_values=True):
        if not isinstance(link_poses, tenso_mesh_bridg__torch.Tensor) or link_poses.shape != (self.n, 2, 23, 7) or link_poses.device != self.positions.device or (not link_poses.is_floating_point()):
            raise ValueError('complete same-device actual link poses required')
        if validate_values and (not bool(tenso_mesh_bridg__torch.isfinite(link_poses).all()) or not tenso_mesh_bridg__torch.allclose(link_poses[..., 3:].norm(dim=-1), tenso_mesh_bridg__torch.ones_like(link_poses[..., 0]), atol=1e-05, rtol=0)):
            raise ValueError('complete finite unit same-device actual link poses required')
        selected = link_poses[:, self.sides, self.links]
        self.positions[:, self.slots] = selected[..., :3]
        self.quaternions[:, self.slots] = selected[..., 3:]
        return (self.positions, self.quaternions)

    def submit(self, solver, link_poses, *, validate_values=True):
        if solver.num_envs != self.n or solver.last_error != 0:
            raise ValueError('same healthy actual shared solver required')
        time = float(solver.time)
        (p, q) = self.targets(link_poses, validate_values=validate_values)
        self._views = (CudaTensorView(p), CudaTensorView(q))
        solver.set_mesh_poses_all(*self._views, teleport=False)
        if solver.last_error != 0 or float(solver.time) != time:
            raise ValueError('pose target submission changed actual solver boundary')

    def bind_material(self, parameters, *, reset_ids):
        """Startup/Reset verifier must supply its already-qualified actual snapshot."""
        if len(reset_ids) != self.n or any((type(i) is not int or i < 1 for i in reset_ids)) or (not parameters) or any((type(v) not in (int, float) or not tenso_mesh_bridg__math.isfinite(v) or v <= 0 for v in parameters.values())):
            raise ValueError('positive actual material snapshot and complete Reset IDs required')
        if self._material is not None and parameters != self._material:
            raise ValueError('frozen material changed without independent task requalification')
        self._material = tenso_mesh_bridg__deepcopy(parameters)
        self._resets = tuple(reset_ids)

    def material(self, *, reset_ids):
        if self._material is None or tuple(reset_ids) != self._resets:
            raise ValueError('startup or selected Reset material qualification required')
        return self._material

def read_pad_triangles(stage, body_path):
    """Read an enabled original pad mesh, including proxy-owned collision Xforms."""
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from dh116.env.scene import read_surface as read_surface
    path = Sdf.Path(body_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path.name not in DISTAL_PAD_LINKS:
        raise ValueError('expected an absolute DH116 distal pad body path')
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError('expected Z-up DH116 stage')
    source = read_surface(stage, body_path, branch='collisions')
    mesh = stage.GetPrimAtPath(source['mesh_path'])
    ancestor = mesh
    while ancestor.GetPath() != path:
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('collision shape belongs to a nested rigid body')
        ancestor = ancestor.GetParent()
    vertices = defor_dh116_colli__np.asarray(source['points_body'], dtype=defor_dh116_colli__np.float32)
    triangles = defor_dh116_colli__np.asarray(source['triangles'], dtype=defor_dh116_colli__np.int64)
    if source['orientation'] == 'leftHanded':
        triangles = triangles[:, [0, 2, 1]]
    record = {key: source[key] for key in ('body_path', 'mesh_path', 'collision_owner', 'collision_approximation', 'orientation', 'mesh_to_body', 'geometry_sha256')}
    record.update(instance_proxy=mesh.IsInstanceProxy(), frame='PhysX_link_origin', unit='m')
    return (vertices, triangles, record)

def pose_matrix(pose):
    pose = finite_array(pose, (7,))
    if abs(geome_math__np.linalg.norm(pose[3:]) - 1) > 2e-06:
        raise ValueError('unit wxyz quaternion required')
    t = geome_math__np.eye(4)
    t[:3, :3] = Rotation.from_quat(geome_math__np.roll(pose[3:], -1)).as_matrix()
    t[:3, 3] = pose[:3]
    return t
from pathlib import Path
import torch as tenso_geome__torch

def transform(vertices, matrix):
    return finite_array(vertices) @ matrix[:3, :3].T + matrix[:3, 3]
from dh116.env.contracts import checked_file
from dh116.hand.kinematics import defor_wrist_proje__CLEARANCE_M as CLEARANCE_M

def box_relation(vertices, center, size):
    """Exact convex-polyhedron / finite AABB SAT, not a vertex-inside heuristic."""
    (v, c, s) = (finite_array(vertices), finite_array(center, (3,)), finite_array(size, (3,)))
    if geome_math__np.any(s <= 0):
        raise ValueError('positive box size required')
    hull = ConvexHull(v)
    edges = geome_math__np.vstack([v[hull.simplices[:, (i + 1) % 3]] - v[hull.simplices[:, i]] for i in range(3)])
    axes = geome_math__np.vstack([geome_math__np.eye(3), hull.equations[:, :3], *[geome_math__np.cross(edges, unit) for unit in geome_math__np.eye(3)]])
    axes = axes[geome_math__np.linalg.norm(axes, axis=1) > 1e-12]
    axes /= geome_math__np.linalg.norm(axes, axis=1)[:, None]
    projections = (v - c) @ axes.T
    radii = geome_math__np.abs(axes) @ (s / 2)
    separation = geome_math__np.maximum(projections.min(0) - radii, -radii - projections.max(0))
    return {'intersects': bool(geome_math__np.max(separation) <= 1e-10), 'max_separating_axis_gap_m': float(geome_math__np.max(separation)), 'top_clearance_m': float(v[:, 2].min() - (c[2] + s[2] / 2)), 'bounds_world_m': [v.min(0).tolist(), v.max(0).tolist()]}
from dh116.env.contracts import runti_io__ROOT as ROOT

def sample_grid(grid, points, *, dtype=geome_math__np.float32):
    """contact.cuh sdfSample arithmetic order; CPU estimate, no GPU bitwise claim."""
    v = geome_math__np.asarray(grid['values'], dtype=dtype)
    o = geome_math__np.asarray(grid['origin'], dtype=dtype)
    p = geome_math__np.asarray(points, dtype=dtype)
    h = dtype(grid['spacing'])
    inv = dtype(1) / h
    if v.ndim != 3 or min(v.shape) < 2 or o.shape != (3,) or (p.ndim != 2) or (p.shape[1] != 3) or (not geome_math__np.isfinite(v).all()) or (not geome_math__np.isfinite(o).all()) or (not geome_math__np.isfinite(p).all()) or (not geome_math__np.isfinite(h)) or (h <= 0):
        raise ValueError('finite three-dimensional locked SDF and point array required')
    u = (p - o) * inv
    inside = geome_math__np.all((u >= 0) & (u < geome_math__np.array(v.shape) - 1), axis=1)
    phi = geome_math__np.full(len(p), geome_math__np.nan, dtype=dtype)
    grad = geome_math__np.full((len(p), 3), geome_math__np.nan, dtype=dtype)
    if not inside.any():
        return (phi, grad, inside)
    i = u[inside].astype(geome_math__np.int64)
    f = geome_math__np.clip((p[inside] - (o + i.astype(dtype) * h)) * inv, dtype(0), dtype(1))
    (fx, fy, fz) = f.T

    def c(a, b, c):
        return v[i[:, 0] + a, i[:, 1] + b, i[:, 2] + c]
    (c000, c001, c010, c011, c100, c101, c110, c111) = [c(*x) for x in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))]
    (dx00, dx01, dx10, dx11) = (c100 - c000, c101 - c001, c110 - c010, c111 - c011)
    (c00, c01, c10, c11) = (c000 + fx * dx00, c001 + fx * dx01, c010 + fx * dx10, c011 + fx * dx11)
    (dy0, dy1) = (c10 - c00, c11 - c01)
    (c0, c1) = (c00 + fy * dy0, c01 + fy * dy1)
    phi[inside] = c0 + fz * (c1 - c0)
    (gx0, gx1) = (dx00 + fy * (dx10 - dx00), dx01 + fy * (dx11 - dx01))
    grad[inside] = geome_math__np.stack((gx0 + fz * (gx1 - gx0), dy0 + fz * (dy1 - dy0), c1 - c0), axis=-1) * inv
    return (phi, grad, inside)
import torch as cuda_geome__torch

def sample_contact(grid, pose, nodes, *, radius=0.005, dtype=geome_math__np.float32):
    t = finite_array(pose, (4, 4))
    nodes = finite_array(nodes)
    local = (nodes.astype(dtype) - t[:3, 3].astype(dtype)) @ t[:3, :3].astype(dtype)
    (phi, gradient, valid) = sample_grid(grid, local, dtype=dtype)
    gap = phi - dtype(radius)
    norm = geome_math__np.linalg.norm(gradient, axis=1)
    candidates = geome_math__np.where(valid)[0]
    i = int(candidates[geome_math__np.argmin(gap[valid])]) if len(candidates) else None
    return dict(minimum_gap_m=None if i is None else float(gap[i]), minimum_node_id=i, gradient_norm_at_minimum=None if i is None else float(norm[i]), in_lattice_nodes=geome_math__np.flatnonzero(valid).tolist(), negative_gap_nodes=geome_math__np.flatnonzero(valid & (gap < 0)).tolist(), contact_predicate_nodes=geome_math__np.flatnonzero(valid & (gap < 0) & (norm > 0)).tolist(), node_gaps_m=[float(x) if ok else None for (x, ok) in zip(gap, valid)], local_node_at_minimum=None if i is None else local[i].astype(float).tolist(), cpu_dtype=geome_math__np.dtype(dtype).name, native_contact_observed=False)

def rod_support_lower(vertices, nodes):
    """Dual-feasible separation from ALL actual nodes, hence their whole polyline."""
    a = geome_math__np.arange(64) * 2 * geome_math__np.pi / 64
    n = geome_math__np.stack((geome_math__np.cos(a), geome_math__np.zeros(64), geome_math__np.sin(a)), axis=1)
    n /= geome_math__np.linalg.norm(n, axis=1)[:, None] + 1e-12
    distance = max(0.0, float(((vertices @ n.T).min(0) - (nodes @ n.T).max(0)).max()))
    return distance - 0.005 - geome_math__NUMERIC_GUARD
cuda_geome__NUMERIC_GUARD = geome_math__NUMERIC_GUARD
legacy_lower = rod_support_lower
tenso_geome__NUMERIC_GUARD = geome_math__NUMERIC_GUARD
original_matrix = pose_matrix
original_transform = transform
tenso_geome__sample_contact = sample_contact
tenso_geome__sample_grid = sample_grid
tenso_geome__rod_support_lower = rod_support_lower
polyl_suppo_certi__VERSION = 'dh116-polyline-support-certificate-v1'
polyl_suppo_certi__RADIUS = 0.005
polyl_suppo_certi__NUMERIC_GUARD = 2e-06

def arrays(vertices, nodes):
    v = polyl_suppo_certi__np.asarray(vertices, dtype=polyl_suppo_certi__np.float64)
    p = polyl_suppo_certi__np.asarray(nodes, dtype=polyl_suppo_certi__np.float64)
    if v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 1 or (p.ndim != 2) or (p.shape[1:] != (3,)) or (len(p) < 2) or (not polyl_suppo_certi__np.isfinite(v).all()) or (not polyl_suppo_certi__np.isfinite(p).all()):
        raise ValueError('complete finite shape vertices and original adjacent nodes required')
    return (v, p)

def sha(a):
    a = polyl_suppo_certi__np.ascontiguousarray(a, dtype='<f8')
    return polyl_suppo_certi__hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()

def directions(v, p):
    from scipy.spatial import ConvexHull
    from scipy.spatial import QhullError
    angle = polyl_suppo_certi__np.arange(64) * 2 * polyl_suppo_certi__np.pi / 64
    n = [polyl_suppo_certi__np.stack((polyl_suppo_certi__np.cos(angle), polyl_suppo_certi__np.zeros(64), polyl_suppo_certi__np.sin(angle)), axis=1), polyl_suppo_certi__np.eye(3), -polyl_suppo_certi__np.eye(3), v.mean(0)[None] - (p[:-1] + p[1:]) / 2]
    hull_faces = 0
    if len(v) >= 4:
        try:
            faces = ConvexHull(v).equations[:, :3]
            hull_faces = len(faces)
            n.extend((faces, -faces))
        except QhullError:
            pass
    n = polyl_suppo_certi__np.concatenate(n)
    length = polyl_suppo_certi__np.linalg.norm(n, axis=1)
    n = n[length > 1e-15]
    length = length[length > 1e-15]
    n = n / (length[:, None] * (1 + 1e-12))
    return (polyl_suppo_certi__np.unique(n, axis=0), hull_faces)

def certify(vertices, nodes):
    (v, p) = arrays(vertices, nodes)
    (n, hull_faces) = directions(v, p)
    projection_min = polyl_suppo_certi__np.concatenate([(v @ n[i:i + 256].T).min(0) for i in range(0, len(n), 256)])
    projected = p @ n.T
    pair_max = polyl_suppo_certi__np.maximum(projected[:-1], projected[1:])
    support = projection_min[None] - pair_max
    winner = support.argmax(1)
    gaps = polyl_suppo_certi__np.maximum(0, support[polyl_suppo_certi__np.arange(len(p) - 1), winner]) - polyl_suppo_certi__RADIUS - polyl_suppo_certi__NUMERIC_GUARD
    old = float(legacy_lower(v, p))
    segments = [dict(segment=i, normal=n[j].tolist(), shape_min_projection_m=float(projection_min[j]), segment_max_projection_m=float(pair_max[i, j]), lower_m=float(gaps[i])) for (i, j) in enumerate(winner)]
    return dict(version=polyl_suppo_certi__VERSION, vertices_sha256=sha(v), nodes_sha256=sha(p), original_vertex_count=len(v), original_node_count=len(p), rod_radius_m=polyl_suppo_certi__RADIUS, numeric_guard_m=polyl_suppo_certi__NUMERIC_GUARD, old_global_xz_lower_m=old, segment_lower_m=float(gaps.min()), lower_m=max(old, float(gaps.min())), segments=segments, candidate_direction_count=len(n), hull_face_count=hull_faces, geometry_scope='original_source_vertices_and_adjacent_node_capsules', actual_contact_measurement=False, native_qualification=False)

def validate(packet, vertices, nodes):
    (v, p) = arrays(vertices, nodes)
    if packet.get('version') != polyl_suppo_certi__VERSION or packet.get('vertices_sha256') != sha(v) or packet.get('nodes_sha256') != sha(p) or (packet.get('original_vertex_count') != len(v)) or (packet.get('original_node_count') != len(p)) or (packet.get('rod_radius_m') != polyl_suppo_certi__RADIUS) or (packet.get('numeric_guard_m') != polyl_suppo_certi__NUMERIC_GUARD) or (packet.get('actual_contact_measurement') is not False) or (packet.get('native_qualification') is not False) or (len(packet.get('segments', [])) != len(p) - 1):
        raise ValueError('certificate source, budget or semantics differ')
    lower = []
    for (i, s) in enumerate(packet['segments']):
        n = polyl_suppo_certi__np.asarray(s['normal'], dtype=float)
        if s['segment'] != i or n.shape != (3,) or (not polyl_suppo_certi__np.isfinite(n).all()) or (polyl_suppo_certi__np.linalg.norm(n) > 1.0):
            raise ValueError('ordered original segment and non-expansive finite normal required')
        vm = float((v @ n).min())
        pm = float(max(p[i] @ n, p[i + 1] @ n))
        gap = max(0.0, vm - pm) - polyl_suppo_certi__RADIUS - polyl_suppo_certi__NUMERIC_GUARD
        for (key, wanted) in (('shape_min_projection_m', vm), ('segment_max_projection_m', pm), ('lower_m', gap)):
            if not polyl_suppo_certi__math.isclose(s[key], wanted, rel_tol=0, abs_tol=1e-12):
                raise ValueError('full original vertex/segment support recomputation differs')
        lower.append(gap)
    old = float(legacy_lower(v, p))
    for (key, wanted) in (('old_global_xz_lower_m', old), ('segment_lower_m', min(lower)), ('lower_m', max(old, min(lower)))):
        if not polyl_suppo_certi__math.isclose(packet[key], wanted, rel_tol=0, abs_tol=1e-12):
            raise ValueError('aggregate conservative bound differs')
    return True

def refined_lower(vertices, nodes):
    (v, p) = arrays(vertices, nodes)
    old = float(legacy_lower(v, p))
    if old >= 0:
        return old
    result = certify(v, p)
    validate(result, v, p)
    return result['lower_m']
tenso_geome__RADIUS = polyl_suppo_certi__RADIUS
tenso_geome__certify = certify
validate_certificate = validate
tenso_geome__VERSION = 'dh116-tensor-geometry-v1'
DECISION_MARGIN_M = 2e-06
LATTICE_MARGIN = 0.0001

@tenso_geome__dataclass(frozen=True, slots=True)
class TensorGeometryResult:
    frame: object
    source_bindings: dict
    poses: tenso_geome__torch.Tensor
    nodes: tenso_geome__torch.Tensor
    radii: tenso_geome__torch.Tensor
    minimum: tenso_geome__torch.Tensor
    maximum: tenso_geome__torch.Tensor
    table_clearance_m: tenso_geome__torch.Tensor
    non_target_lower_m: tenso_geome__torch.Tensor
    hand_separation_lower_m: tenso_geome__torch.Tensor
    table_failure: tenso_geome__torch.Tensor
    non_target_failure: tenso_geome__torch.Tensor
    inter_hand_failure: tenso_geome__torch.Tensor
    sdf_node_gaps_m: tenso_geome__torch.Tensor
    sdf_node_valid: tenso_geome__torch.Tensor
    sdf_negative_nodes: tenso_geome__torch.Tensor
    sdf_contact_nodes: tenso_geome__torch.Tensor
    sdf_minimum_node_ids: tenso_geome__torch.Tensor
    sdf_minimum_gap_m: tenso_geome__torch.Tensor
    sdf_gradient_norm_at_minimum: tenso_geome__torch.Tensor
    sdf_valid: tenso_geome__torch.Tensor
    positive_pad_mean_m: tenso_geome__torch.Tensor
    source_complete: tenso_geome__torch.Tensor
    geometry_passed: tenso_geome__torch.Tensor
    fallback_rows: tenso_geome__torch.Tensor
    support_fallback: tenso_geome__torch.Tensor
    table_fallback: tenso_geome__torch.Tensor
    separation_fallback: tenso_geome__torch.Tensor
    sdf_fallback: tenso_geome__torch.Tensor
    ready: bool = True
    native_qualification: bool = False
    tripod_supported_guard_covered: bool = False

def _rotation_xyzw(poses):
    q = poses[..., 3:]
    q = q / tenso_geome__torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    (x, y, z, w) = q.unbind(-1)
    return tenso_geome__torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)

def _sample_grid_float64(values, origin, spacing, local, grid_shape=None):
    """Original ordered trilinear math, with fixed-shape masked corner reads.

    Invalid nodes read an inert index zero and are replaced with NaN. There is
    no boolean compaction/nonzero, CPU local transform, outside-cell clipping or
    dynamic output shape. Only the original fractional coordinate is clipped.
    """
    inv = 1 / spacing
    u = (local - origin) * inv
    shape = u.new_tensor(values.shape) if grid_shape is None else grid_shape
    valid = ((u >= 0) & (u < shape - 1)).all(-1)
    index = tenso_geome__torch.where(valid[..., None], u, 0).to(tenso_geome__torch.int64)
    safe_local = tenso_geome__torch.where(valid[..., None], local, origin)
    f = ((safe_local - (origin + index.to(values.dtype) * spacing)) * inv).clamp(0, 1)
    (fx, fy, fz) = f.unbind(-1)

    def corner(a, b, c):
        return values[index[..., 0] + a, index[..., 1] + b, index[..., 2] + c]
    (c000, c001, c010, c011, c100, c101, c110, c111) = [corner(*x) for x in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))]
    (dx00, dx01, dx10, dx11) = (c100 - c000, c101 - c001, c110 - c010, c111 - c011)
    (c00, c01, c10, c11) = (c000 + fx * dx00, c001 + fx * dx01, c010 + fx * dx10, c011 + fx * dx11)
    (dy0, dy1) = (c10 - c00, c11 - c01)
    (c0, c1) = (c00 + fy * dy0, c01 + fy * dy1)
    phi = c0 + fz * (c1 - c0)
    (gx0, gx1) = (dx00 + fy * (dx10 - dx00), dx01 + fy * (dx11 - dx01))
    gradient = tenso_geome__torch.stack((gx0 + fz * (gx1 - gx0), dy0 + fz * (dy1 - dy0), c1 - c0), -1) * inv
    return (phi.masked_fill(~valid, tenso_geome__torch.nan), gradient.masked_fill(~valid[..., None], tenso_geome__torch.nan), valid)

class TensorGeometry:
    """Static source inventory plus one owned [N,2,23,7]/[N,41,3] frame.

    shapes: (side_index, actual_link_index, "kind:body_name", ALL vertices).
    pads: (side_index, actual_link_index, body_name, original locked grid).
    The construction/startup owner is responsible for establishing the original
    inventory and source identity. from_policy_geometry preserves its complete
    inventory and locked bindings without loading assets or acquiring an SDK.

    The base backend is the original eager oracle. Runtime graph compilation
    is disabled. The shared make_tensor_geometry factory can select independent
    CUDA measurements while inheriting this capture and all of its guards.
    """

    def __init__(self, shapes, pads=(), *, device, bindings=None, body_names=None, sdf_callable=None, pose_matrix=None, transform=None, backend=None):
        self.backend = tenso_geome__os.environ.get('DH116_TENSOR_GEOMETRY_BACKEND', 'eager') if backend is None else backend
        if self.backend == 'compiled':
            raise ValueError('compiled tensor geometry backend is disabled; use eager or the shared cuda_cabi factory')
        if self.backend != 'eager':
            raise ValueError('base tensor geometry backend must be eager; use the shared factory for cuda_cabi')
        self.device = tenso_geome__torch.empty(0, device=device).device
        self._sdf = tenso_geome__sample_contact if sdf_callable is None else sdf_callable
        self._pose_matrix = original_matrix if pose_matrix is None else pose_matrix
        self._transform = original_transform if transform is None else transform
        (self._shapes, self.columns, self.vertex_counts) = ([], {}, [])
        (self._pads, self.pad_columns, self.pad_links) = ([], {}, {})
        self._pad_roundoff = []
        self.body_names = tenso_geome__deepcopy(body_names)
        self.frame = self.result = None
        for (side, link, key, vertices) in shapes:
            v = tenso_geome__np.array(vertices, dtype=tenso_geome__np.float64, copy=True)
            if type(side) is not int or side not in (0, 1) or type(link) is not int or (not 0 <= link < 23) or (type(key) is not str) or (':' not in key) or ((side, key) in self.columns) or (v.ndim != 2) or (v.shape[1:] != (3,)) or (not len(v)) or (not tenso_geome__np.isfinite(v).all()):
                raise ValueError('distinct original complete finite source hull/actual link required')
            self.columns[side, key] = len(self._shapes)
            v.setflags(write=False)
            self.vertex_counts.append(len(v))
            self._shapes.append((side, link, key, v, tenso_geome__torch.tensor(v, dtype=tenso_geome__torch.float64, device=self.device)))
        if not self._shapes or {s[0] for s in self._shapes} != {0, 1}:
            raise ValueError('complete bilateral original shapes required')
        for (side, link, name, grid) in pads:
            if type(side) is not int or side not in (0, 1) or type(link) is not int or (not 0 <= link < 23) or (type(name) is not str) or (not name) or ((side, name) in self.pad_columns):
                raise ValueError('distinct actual original pad mapping required')
            original = {k: tenso_geome__np.array(v, copy=True) for (k, v) in grid.items()}
            values = tenso_geome__np.array(grid['values'], dtype=tenso_geome__np.float64, copy=True)
            origin = tenso_geome__np.array(grid['origin'], dtype=tenso_geome__np.float64, copy=True)
            spacing = tenso_geome__np.array(grid['spacing'], dtype=tenso_geome__np.float64, copy=True)
            if values.ndim != 3 or min(values.shape) < 2 or origin.shape != (3,) or (spacing.shape != ()) or (not tenso_geome__np.isfinite(values).all()) or (not tenso_geome__np.isfinite(origin).all()) or (not tenso_geome__np.isfinite(spacing)) or (spacing <= 0):
                raise ValueError('finite original three-dimensional locked SDF required')
            self.pad_columns[side, name] = len(self._pads)
            self.pad_links[side, name] = link
            self._pads.append((side, link, name, original, tenso_geome__torch.tensor(values, dtype=tenso_geome__torch.float64, device=self.device), tenso_geome__torch.tensor(origin, dtype=tenso_geome__torch.float64, device=self.device), tenso_geome__torch.tensor(spacing, dtype=tenso_geome__torch.float64, device=self.device)))
            values32 = values.astype(tenso_geome__np.float32).astype(tenso_geome__np.float64)
            origin32 = origin.astype(tenso_geome__np.float32).astype(tenso_geome__np.float64)
            spacing32 = float(tenso_geome__np.float32(spacing))
            if not tenso_geome__np.isfinite(values32).all() or not tenso_geome__np.isfinite(origin32).all() or (not tenso_geome__math.isfinite(spacing32)) or (spacing32 <= 0):
                raise ValueError('original float32 SDF representation must remain finite')
            slopes = tenso_geome__np.array([tenso_geome__np.abs(tenso_geome__np.diff(values, axis=i)).max() / float(spacing) for i in range(3)])
            constants = tenso_geome__np.r_[slopes, tenso_geome__np.abs(values32 - values).max(), tenso_geome__np.abs(values).max(), tenso_geome__np.abs(origin32 - origin), abs(spacing32 - float(spacing))]
            self._pad_roundoff.append(tenso_geome__torch.tensor(constants, dtype=tenso_geome__torch.float64, device=self.device))
        self.source_bindings = tenso_geome__deepcopy(bindings) if bindings is not None else {}
        if bindings is not None and set(bindings) != set(self.pad_columns):
            raise ValueError('one exact source binding per original pad required')
        self._locked_bindings = tenso_geome__deepcopy(self.source_bindings)
        self._locked_body_names = tenso_geome__deepcopy(body_names)
        self.vertex_counts = tuple(self.vertex_counts)
        self._side_ids = [[j for ((s, _), j) in self.columns.items() if s == side] for side in (0, 1)]
        self._pad_side_ids = [[j for ((s, _), j) in self.pad_columns.items() if s == side] for side in (0, 1)]
        target = {(side, name) for (side, _, name, *_) in self._pads}
        self.non_target = tenso_geome__torch.tensor([(side, key.split(':', 1)[1]) not in target for (side, _, key, *_) in self._shapes], dtype=tenso_geome__torch.bool, device=self.device)
        a = tenso_geome__np.arange(64) * 2 * tenso_geome__np.pi / 64
        normals = tenso_geome__np.stack((tenso_geome__np.cos(a), tenso_geome__np.zeros(64), tenso_geome__np.sin(a)), 1)
        normals /= tenso_geome__np.linalg.norm(normals, axis=1)[:, None] + 1e-12
        self._normals = tenso_geome__torch.tensor(normals, dtype=tenso_geome__torch.float64, device=self.device)
        if self.device.type == 'cuda':
            self._convex = []
            self._shape_buckets = []
            self._pad_float = [tuple(t.float() for t in pad[4:7]) for pad in self._pads]
            axes = torch.eye(3, dtype=torch.float64, device=self.device)
            self._support_axes = torch.cat((self._normals, axes, -axes))
            buckets = {}
            for column, (side, link, _, vertices, points) in enumerate(self._shapes):
                try:
                    hull = ConvexHull(vertices)
                except QhullError:
                    faces = points[:0]
                    triangles = torch.empty((0, 3), dtype=torch.long, device=self.device)
                else:
                    faces = torch.tensor(hull.equations[:, :3], device=self.device)
                    triangles = torch.tensor(hull.simplices, dtype=torch.long, device=self.device)
                edges = triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
                edges = edges.sort(-1).values.unique(dim=0)
                self._convex.append((faces, triangles, edges))
                buckets.setdefault((len(points), len(faces)), []).append(column)
            self._shape_side = torch.tensor([row[0] for row in self._shapes], device=self.device)
            self._shape_link = torch.tensor([row[1] for row in self._shapes], device=self.device)
            for columns in buckets.values():
                points = torch.stack([self._shapes[c][4] for c in columns])
                faces = torch.stack([self._convex[c][0] for c in columns])
                directions = torch.cat((faces, -faces), dim=1)
                directions = directions / (directions.norm(dim=-1, keepdim=True) * (1 + 1e-12))
                projection = (points @ directions.transpose(1, 2)).amin(1)
                self._shape_buckets.append((torch.tensor(columns, device=self.device), points, directions, projection))

    @classmethod
    def from_materialized_geometry(cls, geometry, grids, sources, body_names, *, required_pads, device, backend=None):
        """Own complete original arrays and exact bindings without an SDK owner.

        geometry[side][source|cooked][body] contains every original vertex;
        grids[side][pad] contains the locked values/origin/spacing arrays;
        sources[side]["sdf_binding"][pad] is the original source binding.
        Both actual 23-link mappings and the required-pad order are explicit.
        Input mapping order is retained. This verifies inventory only and grants
        no live source identity, native physics or Teacher qualification.
        """
        names = tuple((tuple(n) for n in body_names))
        required = tuple(required_pads)
        if len(names) != 2 or any((len(n) != 23 or len(set(n)) != 23 or any((type(name) is not str or not name for name in n)) for n in names)):
            raise ValueError('both complete actual 23-link body mappings required')
        if not required or len(set(required)) != len(required) or any((type(name) is not str or not name for name in required)):
            raise ValueError('distinct explicit original required pads required')
        if not isinstance(geometry, dict) or set(geometry) != {'left', 'right'} or (not isinstance(grids, dict)) or (set(grids) != {'left', 'right'}) or (not isinstance(sources, dict)):
            raise ValueError('complete bilateral materialized geometry/SDF/sources required')
        (shapes, pads, bindings) = ([], [], {})
        for (side, label) in enumerate(('left', 'right')):
            inventory = geometry[label]
            if not isinstance(inventory, dict) or set(inventory) != {'source', 'cooked'} or any((not isinstance(inventory[kind], dict) or len(inventory[kind]) != count for (kind, count) in (('source', 17), ('cooked', 5)))):
                raise ValueError('original complete 17 source and 5 cooked hulls per side required')
            side_grids = grids[label]
            side_sources = sources.get(label)
            source_bindings = side_sources.get('sdf_binding') if isinstance(side_sources, dict) else None
            if not isinstance(side_grids, dict) or not isinstance(source_bindings, dict) or any((name not in inventory['source']
                        or (name not in inventory['cooked'] and name not in ('base_link', 'finger52_link'))
                        or name not in side_grids or not isinstance(source_bindings.get(name), dict)
                        for name in required)):
                raise ValueError('every required pad needs its original hull, locked grid and source binding')
            for (kind, entries) in inventory.items():
                for (name, vertices) in entries.items():
                    if name not in names[side]:
                        raise ValueError('every original hull must map to an actual body/link')
                    shapes.append((side, names[side].index(name), kind + ':' + name, vertices))
            for name in required:
                pads.append((side, names[side].index(name), name, side_grids[name]))
                bindings[side, name] = tenso_geome__deepcopy(source_bindings[name])
        result = cls(shapes, pads, device=device, bindings=bindings, body_names=names, backend=backend)
        result.sources = tenso_geome__deepcopy(sources)
        return result

    @classmethod
    def from_policy_geometry(cls, owner, body_names, *, device, backend=None):
        """Copy every source/cooked vertex and every REQUIRED_PADS locked grid.

        This helper does not validate an exchange's live SDK identity; that stays
        with the original caller. In particular Tripod's extra source guard has
        not been replaced by these bounds.
        """
        names = tuple((tuple(n) for n in body_names))
        if len(names) != 2 or any((len(n) != 23 or len(set(n)) != 23 for n in names)):
            raise ValueError('both complete actual 23-link body mappings required')
        (shapes, pads, bindings) = ([], [], {})
        for (side, label) in enumerate(('left', 'right')):
            for (kind, entries) in owner._geometry[label].items():
                for (name, vertices) in entries.items():
                    shapes.append((side, names[side].index(name), kind + ':' + name, vertices))
            for name in owner.REQUIRED_PADS:
                pads.append((side, names[side].index(name), name, owner._grids[label][name]))
                bindings[side, name] = tenso_geome__deepcopy(owner.sources[label]['sdf_binding'][name])
        result = cls(shapes, pads, device=device, bindings=bindings, body_names=names, sdf_callable=owner._sdf, pose_matrix=owner._pose_matrix, transform=owner._transform, backend=backend)
        result.sources = tenso_geome__deepcopy(owner.sources)
        return result

    def validate_metadata(self, *, bindings=None, body_names=None):
        if self.source_bindings != self._locked_bindings or (bindings is not None and bindings != self._locked_bindings):
            raise ValueError('original pad source binding changed')
        if self.body_names != self._locked_body_names or (body_names is not None and tuple((tuple(n) for n in body_names)) != self._locked_body_names):
            raise ValueError('actual body/link source mapping changed')

    def require_ready(self, *, frame, bindings=None, body_names=None):
        """Guard the consuming boundary before another SDK step or Reset."""
        if self.frame is None or self.result is None or frame != self.frame:
            raise ValueError('tensor geometry frame/Reset epoch unavailable or changed')
        self.validate_metadata(bindings=bindings, body_names=body_names)
        return self.result

    def policy_pad_gaps(self, poses, nodes, radii):
        """Return signed lattice gaps and continuous approach distances outside it."""
        if not hasattr(self, '_policy_pad_data'):
            self.validate_metadata()
            if not self._pads:
                raise ValueError('policy distance requires locked pad grids')
            grids = [pad[4].float() for pad in self._pads]
            shape = torch.tensor([grid.shape for grid in grids], device=self.device)
            stride = torch.stack((shape[:, 1] * shape[:, 2], shape[:, 2],
                                  torch.ones_like(shape[:, 2])), -1)
            sizes = shape.prod(-1)
            offsets = sizes.cumsum(0) - sizes
            corner = torch.tensor([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                                  device=self.device)
            self._policy_pad_data = (
                [pad[0] for pad in self._pads], [pad[1] for pad in self._pads],
                torch.cat([grid.flatten() for grid in grids]),
                torch.stack([pad[5].float() for pad in self._pads]),
                torch.stack([pad[6].float() for pad in self._pads]),
                shape, stride, offsets, (stride[:, None, :] * corner[None, :, :]).sum(-1),
            )
        sides, links, grid, origin, spacing, shape, stride, offsets, corners = self._policy_pad_data
        poses = poses[:, sides, links].float()
        displacement = nodes[:, None, :, :].float() - poses[..., None, :3]
        local = displacement @ _rotation_xyzw(poses)
        lattice = (local - origin[:, None, :]) / spacing[:, None, None]
        upper = (shape - 1)[:, None, :]
        inside = ((lattice >= 0) & (lattice < upper)).all(-1)
        finite = torch.isfinite(lattice).all(-1)
        coordinate = torch.where(finite[..., None], lattice, 0)
        safe = torch.minimum(coordinate.clamp_min(0), upper)
        index = torch.minimum(safe.long(), upper - 1)
        fraction = safe - index
        flat = (index * stride[:, None, :]).sum(-1) + offsets[:, None]
        values = grid[flat[..., None] + corners[:, None, :]].unflatten(-1, (2, 2, 2))
        x = torch.lerp(values[..., 0, :, :], values[..., 1, :, :], fraction[..., 0, None, None])
        y = torch.lerp(x[..., 0, :], x[..., 1, :], fraction[..., 1, None])
        phi = torch.lerp(y[..., 0], y[..., 1], fraction[..., 2])
        gap = phi - radii[:, None, None].to(phi.dtype)
        outside = ((coordinate - safe) * spacing[:, None, None]).norm(dim=-1)
        approach_gap = torch.where(inside, gap, gap.clamp_min(0) + outside)
        return approach_gap.masked_fill(~finite, float('inf')), inside

    def _separation(self, minimum, maximum):
        low = [minimum[:, ids].amin(1) for ids in self._side_ids]
        high = [maximum[:, ids].amax(1) for ids in self._side_ids]
        return tenso_geome__torch.maximum((low[1] - high[0]).amax(1), (low[0] - high[1]).amax(1))

    def _metrics(self, p, nodes, radii, rotations):
        return self._metrics_eager(p, nodes, radii, rotations)

    def _metrics_eager(self, p, nodes, radii, rotations):
        rod_high = (nodes @ self._normals.T).amax(1)
        (minimum, maximum, lower) = ([], [], [])
        for (side, link, _, _, vertices) in self._shapes:
            lo = hi = projected = None
            for start in range(0, len(vertices), 128):
                world = vertices[None, start:start + 128] @ rotations[:, side, link].transpose(-1, -2) + p[:, side, link, None, :3]
                (a, b, c) = (world.amin(1), world.amax(1), (world @ self._normals.T).amin(1))
                lo = a if lo is None else tenso_geome__torch.minimum(lo, a)
                hi = b if hi is None else tenso_geome__torch.maximum(hi, b)
                projected = c if projected is None else tenso_geome__torch.minimum(projected, c)
            minimum.append(lo)
            maximum.append(hi)
            lower.append((projected - rod_high).amax(1).clamp_min(0) - radii - tenso_geome__NUMERIC_GUARD)
        return (tenso_geome__torch.stack(minimum, 1), tenso_geome__torch.stack(maximum, 1), tenso_geome__torch.stack(lower, 1))

    def _pad_metrics(self, p, nodes, radii, rotations, thresholds):
        return self._pad_metrics_eager(p, nodes, radii, rotations, thresholds)

    def _pad_metrics_eager(self, p, nodes, radii, rotations, thresholds):
        (gaps, norms, validities, fallback) = ([], [], [], [])
        for ((side, link, _, _, values, origin, spacing), rounding) in zip(self._pads, self._pad_roundoff):
            local = (nodes - p[:, side, link, None, :3]) @ rotations[:, side, link]
            (phi, gradient, valid) = _sample_grid_float64(values, origin, spacing, local)
            gap = phi - radii[:, None]
            norm = tenso_geome__torch.sqrt((gradient * gradient).sum(-1))
            u = (local - origin) / spacing
            limit = u.new_tensor(values.shape) - 1
            eps = tenso_geome__torch.finfo(tenso_geome__torch.float32).eps
            relative32 = (nodes.float() - p[:, side, link, None, :3].float()).double()
            rotation32 = rotations[:, side, link].float().double()
            local32_center = relative32 @ rotation32
            local_error = (local32_center - local).abs() + 8 * eps * (relative32.abs() @ rotation32.abs()) + 1e-12
            u_error = (local_error + rounding[5:8]) / spacing + u.abs() * rounding[8] / spacing + 8 * eps * (u.abs() + 1) + 1e-10
            lattice_band = u_error.clamp_min(LATTICE_MARGIN)
            possible = ((u >= -lattice_band) & (u <= limit + lattice_band)).all(-1)
            face = possible & ((u - u.round()).abs() <= lattice_band).any(-1)
            radius_error = (radii.float().double() - radii).abs()[:, None]
            gap_error = ((u_error * spacing * rounding[:3]).sum(-1) + rounding[3] + 64 * eps * rounding[4] + radius_error).clamp_min(DECISION_MARGIN_M)
            near = tenso_geome__torch.zeros_like(valid)
            for threshold in thresholds:
                near |= valid & ((gap - threshold).abs() <= gap_error)
            norm_error = (16 * (rounding[3] + eps * rounding[4]) / spacing + 4 * rounding[:3].amax() / spacing * local_error.sum(-1)).clamp_min(DECISION_MARGIN_M)
            flat = valid & (norm <= norm_error)
            best = gap.masked_fill(~valid, tenso_geome__torch.inf).topk(2, dim=-1, largest=False).values
            tied = valid.any(-1) & ((best[:, 1] - best[:, 0]).abs() <= 2 * gap_error.amax(-1))
            fallback.append(face.any(-1) | near.any(-1) | flat.any(-1) | tied)
            gaps.append(gap)
            norms.append(norm)
            validities.append(valid)
        if not self._pads:
            shape = (len(p), 0, 41)
            return (p.new_empty(shape), p.new_empty(shape), tenso_geome__torch.empty(shape, dtype=tenso_geome__torch.bool, device=self.device), tenso_geome__torch.empty((len(p), 0), dtype=tenso_geome__torch.bool, device=self.device))
        return (tenso_geome__torch.stack(gaps, 1), tenso_geome__torch.stack(norms, 1), tenso_geome__torch.stack(validities, 1), tenso_geome__torch.stack(fallback, 1))

    def _certify_clear_segments(self, minimum, maximum, nodes, radii, lower, support_flags):
        # All 40 adjacent capsules are covered. Coordinate axes are a subset
        # of the original certificate directions and each AABB encloses its hull.
        rows, columns = support_flags.nonzero().unbind(1)
        if not len(rows):
            return support_flags
        left = nodes[rows, :-1]
        right = nodes[rows, 1:]
        seg_low, seg_high = torch.minimum(left, right), torch.maximum(left, right)
        hull_low, hull_high = minimum[rows, columns, None], maximum[rows, columns, None]
        axes = torch.maximum(hull_low - seg_high, seg_low - hull_high)
        gap = axes.amax(-1).clamp_min(0).amin(-1) - radii[rows] - tenso_geome__NUMERIC_GUARD
        # Spatial roundoff only increases the work sent to the exact oracle.
        extent = torch.maximum(hull_low.abs(), hull_high.abs()).flatten(1).amax(1)
        extent += torch.maximum(seg_low.abs(), seg_high.abs()).flatten(1).amax(1)
        error = 128 * torch.finfo(torch.float64).eps * (extent + 1)
        certified = torch.isfinite(gap) & (gap - error > DECISION_MARGIN_M)
        lower[rows, columns] = torch.where(certified, gap - error, lower[rows, columns])
        support_flags[rows, columns] = ~certified
        return support_flags

    def _cuda_segment_support(self, p, rotations, nodes, radii, lower, support_flags):
        # Match the original certificate directions, including each of the
        # forty segment-centre directions tested against all adjacent segments.
        for columns, vertices, faces, face_min in self._shape_buckets:
            rows, local = support_flags[:, columns].nonzero().unbind(1)
            for rows, local in zip(rows.split(256), local.split(256)):
                if not len(rows):
                    continue
                col = columns[local]
                side, link = self._shape_side[col], self._shape_link[col]
                rot = rotations[rows, side, link]
                points = vertices[local]
                rod = (nodes[rows] - p[rows, side, link, None, :3]) @ rot
                delta = points.mean(1)[:, None] - (rod[:, :-1] + rod[:, 1:]) / 2
                lengths = delta.norm(dim=-1, keepdim=True)
                center = torch.where(lengths > 1e-15, delta / (lengths * (1 + 1e-12)), 0)
                fixed = self._support_axes @ rot
                axes = torch.cat((fixed, center), dim=1)
                shape_min = torch.cat(((points @ axes.transpose(1, 2)).amin(1), face_min[local]), 1)
                directions = torch.cat((axes, faces[local]), 1)
                proj = rod @ directions.transpose(1, 2)
                bound = (shape_min[:, None] - torch.maximum(proj[:, :-1], proj[:, 1:])).amax(-1)
                rounding = 128 * torch.finfo(torch.float64).eps * (points.abs().amax((1, 2)) + rod.abs().amax((1, 2)) + 1)
                bound = (bound.amin(-1) - rounding).clamp_min(0) - radii[rows] - tenso_geome__NUMERIC_GUARD
                legacy = (shape_min[:, :64] - proj[:, :, :64].amax(1)).amax(1).clamp_min(0)
                legacy = legacy - radii[rows] - tenso_geome__NUMERIC_GUARD
                lower[rows, col] = torch.where(legacy >= 0, legacy, torch.maximum(legacy, bound))

    def _cuda_extrema_recheck(self, p, rotations, flags, minimum, maximum):
        # The source matrix uses a normalized quaternion. Keep the raw
        # quaternion reserved for the independent Tripod/support predicates.
        for columns, vertices, _, _ in self._shape_buckets:
            rows, local = flags[:, columns].nonzero().unbind(1)
            if not len(rows):
                continue
            col = columns[local]
            side, link = self._shape_side[col], self._shape_link[col]
            world = vertices[local] @ rotations[rows, side, link].transpose(1, 2)
            world += p[rows, side, link, None, :3]
            minimum[rows, col], maximum[rows, col] = world.amin(1), world.amax(1)

    def _cuda_pad_recheck(self, p, rotations, nodes, radii, flags, gaps, norms, valid, negative, contact):
        # Re-evaluate routed rows with the legacy lattice/dtype as well as
        # the main C ABI's FP64 result. Never extrapolate outside the lattice.
        for column, (side, link, *_) in enumerate(self._pads):
            rows = flags[:, column].nonzero().flatten()
            if not len(rows):
                continue
            values, origin, spacing = self._pad_float[column]
            offset = nodes[rows].float() - p[rows, side, link, None, :3].float()
            rotation = rotations[rows, side, link].float()
            local = (offset[..., 0:1] * rotation[:, None, 0]
                     + offset[..., 1:2] * rotation[:, None, 1]
                     + offset[..., 2:3] * rotation[:, None, 2])
            phi, gradient, inside = _sample_grid_float64(values, origin, spacing, local)
            gap = phi - radii[rows, None].float()
            norm = gradient.square().sum(-1).sqrt()
            gaps[rows, column], norms[rows, column] = gap.double(), norm.double()
            valid[rows, column] = inside
            negative[rows, column] = inside & (gap < 0)
            contact[rows, column] = inside & (gap < 0) & (norm > 0)

    def _fallback(self, p, nodes, radii, minimum, maximum, lower, separation, table_flags, support_flags, separation_flags, sdf_flags, gaps, norms, valid, negative, contact):
        rows = table_flags.any(1) | support_flags.any(1) | separation_flags | sdf_flags.any(1)
        ids = rows.nonzero().flatten()
        indices = ids.cpu().tolist()
        cpu_p = p.index_select(0, ids).cpu().numpy()
        cpu_n = nodes.index_select(0, ids).cpu().numpy()
        cpu_r = radii.index_select(0, ids).cpu().numpy()
        tf = table_flags.index_select(0, ids).cpu().numpy()
        sf = support_flags.index_select(0, ids).cpu().numpy()
        hf = separation_flags.index_select(0, ids).cpu().numpy()
        pf = sdf_flags.index_select(0, ids).cpu().numpy()
        for (sparse, lane) in enumerate(indices):
            matrices = {}

            def matrix(side, link):
                if (side, link) not in matrices:
                    pose = cpu_p[sparse, side, link]
                    matrices[side, link] = self._pose_matrix(pose[[0, 1, 2, 6, 3, 4, 5]])
                return matrices[side, link]
            whole = [[], []]
            for (column, (side, link, _, vertices, _)) in enumerate(self._shapes):
                if tf[sparse, column] or sf[sparse, column] or hf[sparse]:
                    world = self._transform(vertices, matrix(side, link))
                    minimum[lane, column] = minimum.new_tensor(world.min(0))
                    maximum[lane, column] = maximum.new_tensor(world.max(0))
                    if hf[sparse]:
                        whole[side].append(world)
                    if sf[sparse, column]:
                        original = tenso_geome__rod_support_lower(world, cpu_n[sparse]) + tenso_geome__RADIUS - cpu_r[sparse]
                        if original < 0:
                            certificate = tenso_geome__certify(world, cpu_n[sparse])
                            validate_certificate(certificate, world, cpu_n[sparse])
                            original = certificate['lower_m'] + tenso_geome__RADIUS - cpu_r[sparse]
                        lower[lane, column] = float(original)
            if hf[sparse]:
                (left, right) = [tenso_geome__np.concatenate(v) for v in whole]
                separation[lane] = float(max((right.min(0) - left.max(0)).max(), (left.min(0) - right.max(0)).max()))
            for (column, (side, link, _, grid, *_)) in enumerate(self._pads):
                if not pf[sparse, column]:
                    continue
                row = self._sdf(grid, matrix(side, link), cpu_n[sparse], radius=float(cpu_r[sparse]))
                values = [tenso_geome__math.nan if v is None else v for v in row['node_gaps_m']]
                gaps[lane, column] = gaps.new_tensor(values)
                valid[lane, column] = False
                valid[lane, column, row['in_lattice_nodes']] = True
                negative[lane, column] = False
                negative[lane, column, row['negative_gap_nodes']] = True
                contact[lane, column] = False
                contact[lane, column, row['contact_predicate_nodes']] = True
                t = matrix(side, link)
                local = (cpu_n[sparse].astype(tenso_geome__np.float32) - t[:3, 3].astype(tenso_geome__np.float32)) @ t[:3, :3].astype(tenso_geome__np.float32)
                (_, gradient, _) = tenso_geome__sample_grid(grid, local)
                norms[lane, column] = norms.new_tensor(tenso_geome__np.linalg.norm(gradient, axis=1))
        return rows

    @tenso_geome__torch.no_grad()
    def capture(self, poses, nodes, *, frame, radius=0.005, sdf_thresholds=(), active=None):
        """Resolve a completed substep before returning GPU bounds and masks.

        Zero is always an SDF decision threshold; supply every additional actual
        consumer threshold in metres. Routing within +/-2um does not widen any
        original predicate. Lattice routing uses +/-1e-4 in lattice coordinates.
        Input data must already be on this device. No SDK step or native contact
        assertion is made. Invalid captures invalidate the previous epoch.
        """
        self.frame = self.result = None
        self.validate_metadata()
        if frame is None or not isinstance(poses, tenso_geome__torch.Tensor) or poses.device != self.device or (poses.ndim != 4) or (poses.shape[1:] != (2, 23, 7)) or (not 1 <= len(poses) <= 512) or (poses.dtype not in (tenso_geome__torch.float32, tenso_geome__torch.float64)) or (not isinstance(nodes, tenso_geome__torch.Tensor)) or (nodes.device != self.device) or (nodes.shape != (len(poses), 41, 3)) or (nodes.dtype not in (tenso_geome__torch.float32, tenso_geome__torch.float64)):
            raise ValueError('complete same-device floating native poses/material41 and frame required')
        if isinstance(radius, tenso_geome__torch.Tensor):
            if radius.device != self.device or radius.dtype not in (tenso_geome__torch.float32, tenso_geome__torch.float64):
                raise ValueError('same-device floating actual radius required')
            radii = radius.detach().double().clone()
        else:
            if type(radius) not in (int, float):
                raise ValueError('finite positive scalar or same-device actual radius required')
            radii = tenso_geome__torch.tensor(float(radius), dtype=tenso_geome__torch.float64, device=self.device)
        if radii.shape == ():
            radii = radii.expand(len(poses)).clone()
        if radii.shape != (len(poses),):
            raise ValueError('one finite positive actual radius per lane required')
        thresholds = (0.0,) + tuple((float(t) for t in sdf_thresholds))
        if not all((tenso_geome__math.isfinite(t) for t in thresholds)):
            raise ValueError('finite actual SDF decision thresholds required')
        (p, n) = (poses.detach().double().clone(), nodes.detach().double().clone())
        invalid = ~tenso_geome__torch.isfinite(p).all() | ~tenso_geome__torch.isfinite(n).all() | ~tenso_geome__torch.isfinite(radii).all() | (radii <= 0).any() | ((tenso_geome__torch.linalg.vector_norm(p[..., 3:], dim=-1) - 1).abs() > 2e-06).any()
        rotations = _rotation_xyzw(p)
        (minimum, maximum, lower) = self._metrics(p, n, radii, rotations)
        separation = self._separation(minimum, maximum)
        invalid |= ~tenso_geome__torch.isfinite(minimum).all() | ~tenso_geome__torch.isfinite(maximum).all() | ~tenso_geome__torch.isfinite(lower).all() | ~tenso_geome__torch.isfinite(separation).all()
        table_flags = (minimum[..., 2] - 0.3 - CLEARANCE_M).abs() <= DECISION_MARGIN_M
        support_flags = self.non_target[None] & (lower <= DECISION_MARGIN_M)
        separation_flags = separation.abs() <= DECISION_MARGIN_M
        (gaps, norms, valid, sdf_flags) = self._pad_metrics(p, n, radii, rotations, thresholds)
        negative = valid & (gaps < 0)
        contact = negative & (norms > 0)
        if active is not None:
            if active.shape != (len(p),) or active.device != self.device or active.dtype != torch.bool:
                raise ValueError('same-device boolean geometry activity mask required')
            table_flags &= active[:, None]
            support_flags &= active[:, None]
            separation_flags &= active
            sdf_flags &= active[:, None]
            support_flags = self._certify_clear_segments(minimum, maximum, n, radii, lower, support_flags)
        fallback_rows = table_flags.any(1) | support_flags.any(1) | separation_flags | sdf_flags.any(1)
        if bool(invalid):
            raise ValueError('complete finite unit native poses/material41 and positive radius required')
        if self.device.type == 'cuda' and active is not None:
            extrema_flags = table_flags | separation_flags[:, None]
            if bool(extrema_flags.any()):
                self._cuda_extrema_recheck(p, rotations, extrema_flags, minimum, maximum)
                separation = self._separation(minimum, maximum)
            if bool(support_flags.any()):
                self._cuda_segment_support(p, rotations, n, radii, lower, support_flags)
            if bool(sdf_flags.any()):
                self._cuda_pad_recheck(p, rotations, n, radii, sdf_flags, gaps, norms, valid, negative, contact)
        elif bool(fallback_rows.any()):
            self._fallback(p, n, radii, minimum, maximum, lower, separation, table_flags, support_flags, separation_flags, sdf_flags, gaps, norms, valid, negative, contact)
        table = minimum[..., 2] - 0.3
        table_failure = table < CLEARANCE_M
        non_target_failure = self.non_target[None] & (lower < 0)
        inter_hand_failure = separation < 0
        any_valid = valid.any(-1)
        minimum_ids = gaps.masked_fill(~valid, tenso_geome__torch.inf).argmin(-1)
        minimum_gap = gaps.gather(-1, minimum_ids[..., None]).squeeze(-1)
        minimum_norm = norms.gather(-1, minimum_ids[..., None]).squeeze(-1)
        minimum_ids = tenso_geome__torch.where(any_valid, minimum_ids, -1)
        minimum_gap = minimum_gap.masked_fill(~any_valid, tenso_geome__torch.nan)
        minimum_norm = minimum_norm.masked_fill(~any_valid, tenso_geome__torch.nan)
        sdf_valid = any_valid & tenso_geome__torch.isfinite(minimum_gap) & tenso_geome__torch.isfinite(minimum_norm) & (minimum_norm > 0)
        means = []
        for ids in self._pad_side_ids:
            if ids:
                mean = minimum_gap[:, ids].clamp_min(0).sum(1) / len(ids)
                means.append(mean.masked_fill(~sdf_valid[:, ids].all(1), tenso_geome__torch.nan))
            else:
                means.append(p.new_full((len(p),), tenso_geome__torch.nan))
        source_complete = sdf_valid.all(1)
        passed = source_complete & ~(table_failure.any(1) | non_target_failure.any(1) | inter_hand_failure)
        result = TensorGeometryResult(tenso_geome__deepcopy(frame), tenso_geome__deepcopy(self.source_bindings), p, n, radii, minimum, maximum, table, lower, separation, table_failure, non_target_failure, inter_hand_failure, gaps, valid, negative, contact, minimum_ids, minimum_gap, minimum_norm, sdf_valid, tenso_geome__torch.stack(means, 1), source_complete, passed, fallback_rows, support_flags, table_flags, separation_flags, sdf_flags)
        (self.frame, self.result) = (tenso_geome__deepcopy(frame), result)
        return result


_DISTANCE_ROUNDING_M = 1e-12


def _certify_segment_distance(vertices, start, axis, planes, estimate, scale):
    """Check a feasible upper bound and a full-vertex separating lower bound."""
    point = estimate[:3].copy()
    endpoint = start + np.clip(estimate[3], 0., 1.) * axis
    normals, offsets = planes[:, :3], planes[:, 3]
    if np.any(offsets >= 0.):  # The constructed interior is the origin.
        return None
    heights = normals @ point
    positive = heights > 0.
    if positive.any():
        factor = min(1., float(np.min(-offsets[positive] / heights[positive])))
        point *= factor * (1. - 1e-12)
    if np.any(normals @ point + offsets > 0.):
        return None

    delta = point - endpoint
    norm = float(np.linalg.norm(delta))
    rounding_m = _DISTANCE_ROUNDING_M
    lower = 0.
    if norm > 0.:
        # A non-expansive direction makes support separation a distance bound.
        normal = delta / (norm * (1. + 1e-12))
        separation = (vertices @ normal).min() - max(start @ normal, (start + axis) @ normal)
        lower = max(0., float(separation) * scale - rounding_m)
    upper = norm * scale + rounding_m
    # Keep the original capsule contact comparison's nanometre precision.
    if not 0. <= upper - lower <= 1e-9:
        return None
    return point, endpoint, lower


def _polish_segment_distance(vertices, start, axis, planes, estimate, scale):
    """Try at most 64 nearby feature sets; only the original certificate accepts."""
    from itertools import chain, combinations, islice
    constraints = np.vstack([
        np.column_stack([planes[:, :3], np.zeros(len(planes))]),
        [0., 0., 0., -1.], [0., 0., 0., 1.],
    ])
    offsets = np.r_[planes[:, 3], 0., -1.]
    slack = np.abs(constraints @ estimate + offsets)
    # Candidate selection only. This does not relax feasibility or distance bounds.
    active = np.flatnonzero(slack <= 1e-6)
    active = active[np.argsort(slack[active], kind="stable")]
    features = chain.from_iterable(combinations(active, size) for size in range(1, 5))
    residual_matrix = np.column_stack([np.eye(3), -axis])
    attempts = 0
    for selected in islice(features, 64):
        attempts += 1
        equality, values = constraints[list(selected)], offsets[list(selected)]
        kkt = np.block([
            [residual_matrix.T @ residual_matrix, equality.T],
            [equality, np.zeros((len(equality), len(equality)))],
        ])
        rhs = np.r_[residual_matrix.T @ start, -values]
        try:
            candidate = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:4]
        except np.linalg.LinAlgError:
            continue
        if (not np.isfinite(candidate).all()
                or np.max(constraints @ candidate + offsets) > 1e-8):
            continue
        checked = _certify_segment_distance(vertices, start, axis, planes, candidate, scale)
        if checked is not None:
            return checked, attempts
    return None, attempts


def _tripod_segment_bounds(vertices, start, end):
    """Certify distance bounds for a convex solid and one finite centreline segment.

    Optimizer success is not a certificate. Both accepted stop statuses require
    feasible witnesses and full-vertex support bounds within one nanometre.
    These are unsigned distances, not penetration depth or multi-frame coverage.
    """
    v, a = finite_array(vertices), finite_array(start, (3,))
    axis = finite_array(end, (3,)) - a
    origin = v.mean(0)
    scale = max(float(np.ptp(v, axis=0).max()), float(np.linalg.norm(axis)), .001)
    v, a, axis = (v - origin) / scale, (a - origin) / scale, axis / scale
    planes = ConvexHull(v).equations
    normals, offsets = planes[:, :3], planes[:, 3]
    constraints = np.column_stack([normals, np.zeros(len(planes))])

    def residual(x):
        return x[:3] - a - x[3] * axis

    def objective(x):
        d = residual(x)
        return float(d @ d)

    def jacobian(x):
        d = residual(x)
        return np.r_[2 * d, -2 * axis @ d]

    result = minimize(objective, np.r_[v.mean(0), .5], jac=jacobian,
        bounds=[(None, None)] * 3 + [(0, 1)],
        constraints=dict(type="ineq", fun=lambda x: -constraints @ x - offsets,
                         jac=lambda x: -constraints),
        method="SLSQP", options=dict(ftol=1e-14, maxiter=200))
    estimate = result.x
    if (np.isfinite(estimate).all() and -.00000001 <= estimate[3] <= 1.00000001
            and np.max(constraints @ estimate + offsets) <= 1e-8):
        if (result.success and result.status == 0) or result.status == 8:
            checked = _certify_segment_distance(v, a, axis, planes, estimate, scale)
            method, attempts = "direct_support_bounds", 0
            if checked is None:
                checked, attempts = _polish_segment_distance(v, a, axis, planes, estimate, scale)
                method = "active_feature_polish"
            if checked is not None:
                point, endpoint, lower = checked
                return dict(
                    lower_m=lower,
                    upper_m=float(np.linalg.norm(point - endpoint)) * scale + _DISTANCE_ROUNDING_M,
                    hull_point_m=(point * scale + origin).tolist(),
                    segment_point_m=(endpoint * scale + origin).tolist(),
                    optimizer_status=int(result.status),
                    certificate_method=method, polish_candidates=attempts,
                )
    raise ValueError("continuous patch/convex-hull distance did not converge")


def _group_envelope_bounds(vertices, support_nodes):
    """Certify unsigned distance to the entire finite group convex hull.

    SLSQP only proposes feasible points. Full-set support separation and a
    feasible upper witness must agree within the original one-nanometre bound.
    """
    vertices, support_nodes = np.asarray(vertices), np.asarray(support_nodes)
    origin = vertices.mean(axis=0)
    scale = max(float(np.ptp(vertices, axis=0).max()),
                float(np.ptp(support_nodes, axis=0).max()), .001)
    v, w = (vertices-origin)/scale, (support_nodes-origin)/scale
    planes = ConvexHull(v).equations
    normals, offsets = planes[:, :3], planes[:, 3]
    if np.any(offsets >= 0.):
        return None
    count = len(w)
    constraints = np.column_stack([normals, np.zeros((len(planes), count))])
    total_gradient = np.r_[np.zeros(3), np.ones(count)]

    def residual(x):
        return x[:3]-x[3:]@w

    def objective(x):
        d = residual(x)
        return float(d@d)

    def jacobian(x):
        d = residual(x)
        return np.r_[2*d, -2*w@d]

    fit = minimize(objective, np.r_[np.zeros(3), np.full(count, 1/count)],
        jac=jacobian, bounds=[(None, None)]*3+[(0., 1.)]*count,
        constraints=[dict(type="ineq", fun=lambda x: -constraints@x-offsets,
                          jac=lambda x: -constraints),
                     dict(type="eq", fun=lambda x: x[3:].sum()-1,
                          jac=lambda x: total_gradient)],
        method="SLSQP", options=dict(ftol=1e-14, maxiter=200))
    if ((not fit.success or fit.status != 0) and fit.status != 8
            or not np.isfinite(fit.x).all()):
        return None
    point = fit.x[:3].copy()
    weights = np.maximum(fit.x[3:], 0.)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.:
        return None
    weights /= total
    group_point = weights@w
    heights = normals@point
    positive = heights > 0.
    if positive.any():
        factor = min(1., float(np.min(-offsets[positive]/heights[positive])))
        point *= factor*(1.-1e-12)
    if np.any(normals@point+offsets > 0.):
        return None
    delta = point-group_point
    length = float(np.linalg.norm(delta))
    if not np.isfinite(length) or length <= 0.:
        return None
    normal = delta/(length*(1.+1e-12))
    lower = max(0., float(np.min(v@normal)-np.max(w@normal))*scale
                - _DISTANCE_ROUNDING_M)
    upper = length*scale+_DISTANCE_ROUNDING_M
    if not np.isfinite([lower, upper]).all() or not 0. <= upper-lower <= 1e-9:
        return None
    # Return the independently checked full-set lower bound; the upper bound
    # exists solely to certify the proposal and is never the acceptance test.
    return dict(normal=normal.tolist(), lower_m=lower, upper_m=upper,
                hull_point_m=(point*scale+origin).tolist(),
                group_point_m=(group_point*scale+origin).tolist(),
                source_node_count=count, optimizer_status=int(fit.status))

def _tripod_tube(node_samples, normals):
    """Represent all recorded material segments by conservative support planes."""
    samples = np.asarray(node_samples, dtype=float)
    if (samples.ndim != 3 or samples.shape[1:] != (41, 3) or not len(samples)
            or not np.isfinite(samples).all()):
        raise ValueError("finite original 41-node material and 5 mm rod radius required")
    upper = np.stack([(samples[:, start:start+3].reshape(-1, 3)@normals.T).max(0)+ _TRIPOD_ROD_RADIUS_M
                      for start in range(0, 40, 2)])
    return upper


def _tripod_separation(vertices, normals, upper):
    lower = (np.asarray(vertices) @ normals.T).min(0)
    return float((lower[None, :] - upper).max(axis=-1).min())


def _tripod_box_clearance_lower(vertices, low, high):
    lower = float(np.linalg.norm(
        np.maximum(np.maximum(low-vertices.max(axis=0), vertices.min(axis=0)-high),0.)))
    if lower > 0.:
        return lower
    return box_relation(vertices, (low+high)/2, high-low)["max_separating_axis_gap_m"]




def _tripod_normals():
    azimuth = np.linspace(0., 2*np.pi, 72, endpoint=False)
    elevation = np.linspace(-np.pi/2, np.pi/2, 25)[:, None]
    normals = np.stack(np.broadcast_arrays(np.cos(elevation)*np.cos(azimuth),
        np.cos(elevation)*np.sin(azimuth), np.sin(elevation)), axis=-1).reshape(-1, 3)
    return normals


def _tripod_envelope_passes(vertices, nodes, normals):
    """Certify all 20 full convex node groups for this one measured frame."""
    origin = vertices.mean(axis=0)
    vertices, nodes = vertices - origin, nodes - origin
    upper = _tripod_tube(nodes[None], normals)
    lower = np.max(np.min(vertices @ normals.T, axis=0)[None, :] - upper, axis=1)
    lower -= _DISTANCE_ROUNDING_M
    budget = 40
    for group in np.flatnonzero(lower < -_TRIPOD_PAD_PENETRATION_M):
        start = int(group) * 2
        support_nodes = nodes[start:start + 3]
        for segment in (start, start + 1):
            if budget == 0:
                return False
            budget -= 1
            try:
                bound = _tripod_segment_bounds(vertices, nodes[segment], nodes[segment + 1])
            except ValueError:
                return False
            if bound['upper_m'] - _TRIPOD_ROD_RADIUS_M < -_TRIPOD_PAD_PENETRATION_M:
                return False
            delta = np.asarray(bound['hull_point_m']) - np.asarray(bound['segment_point_m'])
            length = float(np.linalg.norm(delta))
            if not np.isfinite(length) or length == 0.:
                return False
            normal = delta / (length * (1. + 1e-12))
            support_lower = float(np.min(vertices @ normal) - np.max(support_nodes @ normal)
                                  - _TRIPOD_ROD_RADIUS_M - _DISTANCE_ROUNDING_M)
            lower[group] = max(lower[group], support_lower)
            if lower[group] >= -_TRIPOD_PAD_PENETRATION_M:
                break

        if lower[group] < -_TRIPOD_PAD_PENETRATION_M:
            if budget == 0:
                return False
            budget -= 1
            bound = _group_envelope_bounds(vertices, support_nodes)
            if bound is None:
                return False
            normal = np.asarray(bound['normal'])
            support_lower = float(np.min(vertices @ normal) - np.max(support_nodes @ normal)
                                  - _TRIPOD_ROD_RADIUS_M - _DISTANCE_ROUNDING_M)
            lower[group] = max(lower[group], support_lower)
        if lower[group] < -_TRIPOD_PAD_PENETRATION_M:
            return False
    return bool(np.isfinite(lower).all() and lower.min() >= -_TRIPOD_PAD_PENETRATION_M)


_TRIPOD_PADS = frozenset(('finger14_link', 'finger23_link', 'finger33_link'))
_TRIPOD_ROD_RADIUS_M = .005
_TRIPOD_SUPPORT_LIMIT_M = 1e-10
_TRIPOD_NON_TARGET_LIMIT_M = .00025
_TRIPOD_PAD_PENETRATION_M = .00015


def _point_edge_delta(point, head, edge):
    norm = edge.square().sum(-1)
    t = (((point - head) * edge).sum(-1) / torch.where(norm > 0, norm, 1)).clamp(0, 1)
    return point - (head + t[..., None] * edge)


def _point_face_delta(point, face):
    head = face[..., 0, :]
    u, v = face[..., 1, :] - head, face[..., 2, :] - head
    delta = point - head
    normal = torch.linalg.cross(u, v)
    d = normal.square().sum(-1)
    regular = d > 1e-20 * u.square().sum(-1) * v.square().sum(-1)
    denom = torch.where(regular, d, 1)
    s = (torch.linalg.cross(delta, v) * normal).sum(-1) / denom
    t = (torch.linalg.cross(u, delta) * normal).sum(-1) / denom
    height = (delta * normal).sum(-1) / denom
    return height[..., None] * normal, regular & (s >= 0) & (t >= 0) & (s + t <= 1)


def _group_distance_lower(vertices, triangles, edges, group):
    # The closest separated features are vertex/face or edge/edge.
    # Boundary projections also cover vertices and a degenerate node triangle.
    head = vertices[:, edges[:, 0], None]
    arm = vertices[:, edges[:, 1], None] - head
    start = group[:, None]
    axis = group.roll(-1, dims=1)[:, None] - start
    delta = head - start
    normal = torch.linalg.cross(arm, axis)
    d = normal.square().sum(-1)
    regular = d > 1e-20 * arm.square().sum(-1) * axis.square().sum(-1)
    denom = torch.where(regular, d, 1)
    s = (torch.linalg.cross(axis, delta) * normal).sum(-1) / denom
    t = (torch.linalg.cross(arm, delta) * normal).sum(-1) / denom
    interior = delta + s[..., None] * arm - t[..., None] * axis
    admissible = regular & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    vertex_face, vertex_inside = _point_face_delta(vertices, group[:, None])
    node_face, node_inside = _point_face_delta(group[:, None], vertices[:, triangles][:, :, None])
    vectors = torch.cat((vertex_face, -node_face.flatten(1, 2),
                         interior.flatten(1, 2), _point_edge_delta(vertices[:, :, None], start, axis).flatten(1, 2),
                         -_point_edge_delta(start, head, arm).flatten(1, 2)), 1)
    norms = vectors.square().sum(-1)
    checked = torch.cat((vertex_inside, node_inside.flatten(1), admissible.flatten(1)), 1)
    norms[:, :checked.shape[1]].masked_fill_(~checked, torch.inf)
    index = norms.argmin(-1)
    closest = vectors[torch.arange(len(vertices), device=vertices.device), index]
    length = closest.norm(dim=-1)
    direction = closest / torch.where(length > 0, length * (1 + 1e-12), 1)[:, None]
    support = (vertices @ direction[..., None]).squeeze(-1).amin(-1)
    support -= (group @ direction[..., None]).squeeze(-1).amax(-1)
    return support - _DISTANCE_ROUNDING_M


class TripodSupportGuard:
    """Use every original convex vertex and both translated Tripod supports.

    Run conservative device bounds first. Resolve uncertain bounds on device
    during training and retain the original CPU proof for reference captures.
    A capture is valid only for its measured frame and Reset epoch.
    """

    def __init__(self, geometry, supports, env_origins):
        if not isinstance(geometry, TensorGeometry):
            raise ValueError('original bilateral TensorGeometry owner required')
        geometry.validate_metadata()
        self.geometry = geometry
        self.device = geometry.device
        self.origins = torch.as_tensor(env_origins, device=self.device, dtype=torch.float64).detach().clone()
        if (self.origins.ndim != 2 or self.origins.shape[1] != 3 or not 1 <= len(self.origins) <= 512
                or not bool(torch.isfinite(self.origins).all()) or bool((self.origins[:, 2] != 0).any())):
            raise ValueError('finite horizontal actual environment origins required')
        if (len(supports) != 2 or any(type(box['name']) is not str or not box['name'] for box in supports)
                or len({box['name'] for box in supports}) != 2):
            raise ValueError('both original independent external supports required')
        centers, sizes = (np.array([box[key] for box in supports], dtype=np.float64)
                          for key in ('center_m', 'size_m'))
        if centers.shape != (2, 3) or sizes.shape != (2, 3) or not np.isfinite([centers, sizes]).all() or np.any(sizes <= 0):
            raise ValueError('finite complete original support boxes required')
        self.supports = tenso_geome__deepcopy(supports)
        self._locked_supports = tenso_geome__deepcopy(self.supports)
        self._locked_origins = self.origins.clone()
        self._box_low = self.origins[:, None] + torch.tensor(centers - sizes / 2, device=self.device)
        self._box_high = self.origins[:, None] + torch.tensor(centers + sizes / 2, device=self.device)
        self._shapes = tuple((side, link, vertices, device_vertices, key.split(':', 1)[1] in _TRIPOD_PADS)
                             for side, link, key, vertices, device_vertices in geometry._shapes)
        if any(len(vertices) < 4 for _, _, vertices, _, _ in self._shapes):
            raise ValueError('complete original convex vertices required for Tripod support')
        self._extent = max(float(np.abs(vertices).max()) for _, _, vertices, _, _ in self._shapes)
        self._normals_cpu = _tripod_normals()
        self._normals = torch.tensor(self._normals_cpu, dtype=torch.float64, device=self.device)
        self._coarse_normals = self._normals.view(25, 72, 3)[::4, ::6].reshape(-1, 3)
        self._groups = torch.arange(0, 40, 2, device=self.device)[:, None] + torch.arange(3, device=self.device)
        self._thresholds = self._normals.new_tensor([
            -_TRIPOD_PAD_PENETRATION_M if target else _TRIPOD_NON_TARGET_LIMIT_M
            for *_, target in self._shapes
        ])
        self._side = torch.tensor([row[0] for row in self._shapes], device=self.device)
        self._link = torch.tensor([row[1] for row in self._shapes], device=self.device)
        self._rotation_extent = self._normals.new_tensor([
            2 * np.linalg.norm(vertices, axis=1).max() for _, _, vertices, _, _ in self._shapes
        ])
        buckets = {}
        for col, (_, _, vertices, _, _) in enumerate(self._shapes):
            buckets.setdefault((len(vertices) - 1).bit_length(), []).append(col)
        # Pad within vertex-count buckets by repeating a real vertex so the
        # exact full-vertex support minimum is independent of padding.
        self._vertex_buckets = []
        for columns in buckets.values():
            vertices = [self._shapes[col][3] for col in columns]
            width = max(map(len, vertices))
            padded = torch.stack([torch.cat((v, v[:1].expand(width - len(v), 3))) for v in vertices])
            self._vertex_buckets.append((torch.tensor(columns, device=self.device), padded))
        self.frame = self.result = None

    def require_ready(self, *, frame):
        if self.frame is None or frame != self.frame or self.result is None:
            raise ValueError('Tripod raw support frame/Reset epoch unavailable or changed')
        self.geometry.validate_metadata()
        return self.result

    def _fallback_flags(self, poses, rod, low, high, active, box_low, box_high, *, coarse=False):
        # The shared extrema use a normalized quaternion. Expand the box
        # by the full bound on the raw transform used by the exact oracle.
        q_sq = poses[:, self._side, self._link, 3:].square().sum(-1)
        drift = (q_sq - 1).abs() * self._rotation_extent
        (low, high) = (low - drift[..., None], high + drift[..., None])
        magnitude = poses[..., :3].abs().flatten(1).amax(1) + rod.abs().flatten(1).amax(1)
        margin = DECISION_MARGIN_M + 128 * torch.finfo(torch.float64).eps * (magnitude + self._extent + 1)
        support_gap = torch.maximum(box_low[:, None] - high[:, :, None],
                                    low[:, :, None] - box_high[:, None]).clamp_min(0).norm(dim=-1)
        support_near = (support_gap <= _TRIPOD_SUPPORT_LIMIT_M + margin[:, None, None]).any(-1)
        # The six axes belong to the locked normal set. A clear axis for
        # every material group proves the original directional criterion.
        groups = rod[:, self._groups]
        axis_gap = torch.maximum(low[:, :, None] - groups.amax(2)[:, None],
                                 groups.amin(2)[:, None] - high[:, :, None]).amax(-1)
        threshold = self._thresholds + margin[:, None] + _DISTANCE_ROUNDING_M
        rod_near = (axis_gap.amin(-1) - _TRIPOD_ROD_RADIUS_M <= threshold) & active[:, None]
        if not bool(torch.isfinite(support_gap).all() & torch.isfinite(axis_gap).all()):
            raise ValueError('nonfinite Tripod raw convex lower bound')
        if bool(rod_near.any()):
            normals = self._coarse_normals if coarse else self._normals
            projected = rod @ normals.T
            upper = torch.maximum(torch.maximum(projected[:, 0:40:2], projected[:, 1:41:2]),
                                  projected[:, 2:41:2]) + _TRIPOD_ROD_RADIUS_M
            for columns, vertices in self._vertex_buckets:
                rows, local_ids = rod_near[:, columns].nonzero().unbind(1)
                # Bound transient FP64 projection memory even for the palm hull.
                width = max(1, (128 << 20) // (8 * len(normals) * (vertices.shape[1] + 20)))
                for rows_part, local_part in zip(rows.split(width), local_ids.split(width)):
                    if not len(rows_part):
                        continue
                    col = columns[local_part]
                    pose = poses[rows_part, self._side[col], self._link[col]]
                    points = vertices[local_part]
                    xyz = pose[:, None, 3:6].expand_as(points)
                    cross = 2 * torch.linalg.cross(xyz, points, dim=-1)
                    world = points + pose[:, None, 6:7] * cross + torch.linalg.cross(xyz, cross, dim=-1) + pose[:, None, :3]
                    plane_low = (world @ normals.T).amin(1)
                    gap = (plane_low[:, None] - upper[rows_part]).amax(-1).amin(-1)
                    if not bool(torch.isfinite(gap).all()):
                        raise ValueError('nonfinite Tripod directional support bound')
                    rod_near[rows_part, col] = gap <= threshold[rows_part, col]
        return support_near & active[:, None], rod_near

    def _cuda_support_failures(self, points, world, triangles, edges, low, high, selected):
        # Preserve the original positive Euclidean AABB branch, including its
        # positive-but-failed interval below the SAT cutoff.
        aabb_gap = torch.maximum(low - world.amax(1)[:, None],
                                 world.amin(1)[:, None] - high).clamp_min(0).norm(dim=-1)
        failures = selected[:, None] & (aabb_gap > 0) & (aabb_gap <= _TRIPOD_SUPPORT_LIMIT_M)
        rows, boxes = (selected[:, None] & (aabb_gap == 0)).nonzero().unbind(1)
        if not len(rows):
            return failures.any(-1)

        basis_axes = torch.eye(3, dtype=points.dtype, device=points.device)
        n_axes = 3 + len(triangles) + 3 * len(edges)
        n_vertices = world.shape[1]
        max_values = (16 << 20) // world.element_size()
        pair_batch = max(1, max_values // (n_vertices * min(n_axes, 256)))
        for pairs, slots in zip(rows.split(pair_batch), boxes.split(pair_batch), strict=True):
            local = points[pairs]
            polygons = local[:, triangles]
            normals = torch.linalg.cross(polygons[:, :, 1] - polygons[:, :, 0],
                                         polygons[:, :, 2] - polygons[:, :, 0])
            directions = local[:, edges[:, 1]] - local[:, edges[:, 0]]
            cross = torch.linalg.cross(directions[:, :, None], basis_axes[None, None]).flatten(1, 2)
            axes = torch.cat((basis_axes.expand(len(pairs), -1, -1), normals, cross), 1)
            lengths = axes.norm(dim=-1, keepdim=True)
            unit = axes / torch.where(lengths > 0, lengths, 1)
            keep = lengths.squeeze(-1) > 1e-12
            keep[:, 3:3 + len(triangles)] = normals.square().sum(-1) > 0

            lo, hi = low[pairs, slots], high[pairs, slots]
            center, half = (lo + hi) / 2, (hi - lo) / 2
            shifted = world[pairs] - center[:, None]
            width = max(1, max_values // (len(pairs) * n_vertices))
            gap = world.new_full((len(pairs),), -torch.inf)
            for start in range(0, n_axes, width):
                end = start + width
                directions = unit[:, start:end].transpose(1, 2)
                proj = shifted @ directions
                radius = (half[:, None] @ directions.abs()).squeeze(1)
                bounds = torch.maximum(proj.amin(1) - radius, -radius - proj.amax(1))
                gap = torch.maximum(gap, bounds.masked_fill(~keep[:, start:end], -torch.inf).amax(-1))
            failures[pairs, slots] = gap <= _TRIPOD_SUPPORT_LIMIT_M
        return failures.any(-1)

    def _cuda_failures(self, poses, rod, support_near, rod_near, box_low, box_high, failures):
        near = support_near | rod_near
        for column in near.any(0).nonzero().flatten().tolist():
            side, link, _, vertices, target = self._shapes[column]
            rows = (near[:, column] & ~failures).nonzero().flatten()
            for batch in rows.split(128):
                if not len(batch):
                    continue
                pose = poses[batch, side, link]
                faces, triangles, edges = self.geometry._convex[column]
                if not len(faces):
                    raise ValueError('Tripod guard requires nondegenerate convex hulls')
                xyz = pose[:, None, 3:6].expand(len(batch), len(vertices), 3)
                cross = 2 * torch.linalg.cross(xyz, vertices.expand_as(xyz))
                points = vertices + pose[:, None, 6:7] * cross + torch.linalg.cross(xyz, cross)
                world = points + pose[:, None, :3]
                supports = support_near[batch, column]
                if bool(supports.any()):
                    failures[batch] |= self._cuda_support_failures(
                        points, world, triangles, edges, box_low[batch], box_high[batch], supports,
                    )
                check = rod_near[batch, column] & ~failures[batch]
                if not bool(check.any()):
                    continue
                active, points = batch[check], points[check]
                local_rod = rod[active] - poses[active, side, link, None, :3]
                # A single real hull vertex/rod-node pair supplies a feasible
                # upper witness. Below the original threshold, no direction
                # can certify that group, so do not allocate its full projection.
                limit = (_TRIPOD_ROD_RADIUS_M - _TRIPOD_PAD_PENETRATION_M if target
                         else _TRIPOD_ROD_RADIUS_M + _TRIPOD_NON_TARGET_LIMIT_M)
                square_gap = (points[:, :, None] - local_rod[:, None]).square().sum(-1)
                unsafe = square_gap.amin((1, 2)) < (limit - _DISTANCE_ROUNDING_M) ** 2
                if bool(unsafe.any()):
                    failures[active[unsafe]] = True
                    active, points, local_rod = active[~unsafe], points[~unsafe], local_rod[~unsafe]
                if not len(active):
                    continue
                plane_low = (points @ self._normals.T).amin(1)
                projected = local_rod @ self._normals.T
                plane_high = torch.maximum(torch.maximum(projected[:, 0:40:2], projected[:, 1:41:2]),
                                           projected[:, 2:41:2])
                lower = (plane_low[:, None] - plane_high).amax(-1) - _TRIPOD_ROD_RADIUS_M
                if not target:
                    failures[active] |= lower.amin(-1) <= _TRIPOD_NON_TARGET_LIMIT_M
                    continue
                row, group = (lower - _DISTANCE_ROUNDING_M < -_TRIPOD_PAD_PENETRATION_M).nonzero().unbind(1)
                for i, j in zip(row.split(128), group.split(128)):
                    if not len(i):
                        continue
                    nodes = rod[active[i, None], self._groups[j]] - poses[active[i], side, link, None, :3]
                    bound = _group_distance_lower(points[i], triangles, edges, nodes)
                    failed = bound - _TRIPOD_ROD_RADIUS_M < -_TRIPOD_PAD_PENETRATION_M
                    failures[active[i[failed]]] = True

    def _exact_failures(self, poses, rod, support_near, rod_near, selected, box_low, box_high):
        cpu_poses = poses.index_select(0, selected).cpu().numpy()
        cpu_nodes = rod.index_select(0, selected).cpu().numpy()
        cpu_support = support_near.index_select(0, selected).cpu().numpy()
        cpu_rod = rod_near.index_select(0, selected).cpu().numpy()
        lows = box_low.index_select(0, selected).cpu().numpy()
        highs = box_high.index_select(0, selected).cpu().numpy()
        failures = []
        for lane, nodes in enumerate(cpu_nodes):
            failed = False
            tube = None
            for column in np.flatnonzero(cpu_support[lane] | cpu_rod[lane]):
                side, link, vertices, _, target = self._shapes[column]
                pose = cpu_poses[lane, side, link]
                world = rotated_points(pose[[0, 1, 2, 6, 3, 4, 5]], vertices)
                if cpu_support[lane, column]:
                    failed = any(_tripod_box_clearance_lower(world, low, high) <= _TRIPOD_SUPPORT_LIMIT_M
                                 for low, high in zip(lows[lane], highs[lane], strict=True))
                if not failed and cpu_rod[lane, column]:
                    if target:
                        failed = not _tripod_envelope_passes(world, nodes, self._normals_cpu)
                    else:
                        if tube is None:
                            tube = _tripod_tube(nodes[None], self._normals_cpu)
                        failed = _tripod_separation(world, self._normals_cpu, tube) <= _TRIPOD_NON_TARGET_LIMIT_M
                if failed:
                    break
            failures.append(failed)
        return failures

    @torch.no_grad()
    def capture(self, poses_xyzw, nodes, *, frame, measurement=None, active=None, lane_ids=None):
        """Return the original raw-shape predicate for active lanes at this frame."""
        self.frame = self.result = None
        self.geometry.validate_metadata()
        if self.supports != self._locked_supports or not torch.equal(self.origins, self._locked_origins):
            raise ValueError('original Tripod support source or environment origins changed')
        box_low, box_high = self._box_low, self._box_high
        if lane_ids is not None:
            box_low, box_high = box_low[lane_ids], box_high[lane_ids]
        count = len(box_low)
        for value, shape in ((poses_xyzw, (count, 2, 23, 7)), (nodes, (count, 41, 3))):
            if (not isinstance(value, torch.Tensor) or value.device != self.device or value.shape != shape
                    or value.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(value).all())):
                raise ValueError('complete finite same-device actual raw poses/material41 required')
        poses, rod = poses_xyzw.detach().double(), nodes.detach().double()
        if frame is None or bool(((poses[..., 3:].norm(dim=-1) - 1).abs() > 1e-5).any()):
            raise ValueError('native unit-tolerance raw quaternion and frame required')
        tensor_guard = self.device.type == 'cuda' and active is not None
        if active is None:
            active = torch.ones(count, dtype=torch.bool, device=self.device)
        elif active.shape != (count,) or active.device != self.device or active.dtype != torch.bool:
            raise ValueError('same-device boolean Tripod activity mask required')
        if measurement is None:
            radius = poses.new_full((count,), _TRIPOD_ROD_RADIUS_M)
            low, high, _ = self.geometry._metrics(poses, rod, radius, _rotation_xyzw(poses))
        else:
            if (measurement is not self.geometry.result or measurement.frame != frame
                    or not torch.equal(poses, measurement.poses) or not torch.equal(rod, measurement.nodes)):
                raise ValueError('same-frame measured Tripod hull bounds required')
            low, high = measurement.minimum, measurement.maximum
        support_near, rod_near = self._fallback_flags(poses, rod, low, high, active, box_low, box_high, coarse=tensor_guard)
        failures = torch.zeros(count, dtype=torch.bool, device=self.device)
        if tensor_guard:
            self._cuda_failures(poses, rod, support_near, rod_near, box_low, box_high, failures)
        else:
            selected = (support_near | rod_near).any(1).nonzero().flatten()
            if len(selected):
                failures[selected] = failures.new_tensor(
                    self._exact_failures(poses, rod, support_near, rod_near, selected, box_low, box_high))
        self.frame, self.result = tenso_geome__deepcopy(frame), failures
        return failures


cuda_geome__TensorGeometry = TensorGeometry
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / 'configs/runtime/geometry.json'
MODE = 'cuda_cabi'
CANDIDATE_VERSION = 'dh116-full-vertex-fp64-cuda-measurement-v6-full-cabi-pad-nan-routing-guard'

def _sha(path):
    return cuda_geome__hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _checked_file(path, expected, label):
    checked_file(path, expected)
    return Path(path).resolve(strict=True)

@cuda_geome__dataclass(frozen=True, slots=True)
class CudaGeometryProfile:
    """A verified manifest identity; never a native admission receipt."""
    manifest_path: Path
    manifest_sha256: str
    binary_path: Path
    binary_sha256: str
    source_sha256: tuple
    native_ready: bool = False
    teacher_admission: bool = False

    @classmethod
    def load(cls, path=DEFAULT_MANIFEST):
        path = Path(path).resolve(strict=True)
        manifest_hash = _sha(path)
        m = json.loads(path.read_text())
        required = dict(schema='dh116-cuda-geometry-v2', mode=MODE, sm=89, precision='FP64', full_vertices=True, support_normals=64, fast_math=False, fmad=False, native_ready=False, teacher_admission=False, runtime_graph_compilation=False)
        if any((type(m.get(k)) is not type(v) or m.get(k) != v for (k, v) in required.items())):
            raise ValueError('invalid CUDA geometry precision, device or admission flags')
        prefix = 'src/dh116/'
        names = {'physics/geometry.py', 'cuda_measurements/hull_support.py',
            'cuda_measurements/pad_support.py','cuda_measurements/hull_support.cu',
            'cuda_measurements/pad_support.cu'}
        sources = m['source_sha256']
        if set(sources) != {prefix + n for n in names}:
            raise ValueError('complete current geometry source inventory required')
        for (name, digest) in sources.items():
            _checked_file(ROOT / name, digest, 'current geometry source')
        binary = _checked_file(ROOT / m['binary_path'], m['binary_sha256'], 'CUDA binary')
        if _sha(path) != manifest_hash:
            raise ValueError('CUDA geometry manifest changed during verification')
        return cls(path, manifest_hash, binary, m['binary_sha256'], tuple(sorted(sources.items())))

    def verify(self):
        if type(self.native_ready) is not bool or self.native_ready or type(self.teacher_admission) is not bool or self.teacher_admission:
            raise ValueError('CUDA measurements have no native admission')
        current = type(self).load(self.manifest_path)
        if current != self:
            raise ValueError('CUDA geometry profile identity changed')
        return self

    def identity(self):
        return dict(actual_measurement_backend=MODE, binary_sha256=self.binary_sha256, manifest_sha256=self.manifest_sha256, source_sha256=dict(self.source_sha256), binary_path=str(self.binary_path), native_ready=False, teacher_admission=False)

class CudaMeasurementGeometry(cuda_geome__TensorGeometry):
    """Only hull/pad measurements change; capture and every guard stay eager."""
    measurement_backend = MODE

    def __init__(self, shapes, pads=(), *, profile=None, **kwargs):
        backend = kwargs.pop('backend', None)
        if backend not in (None, 'eager'):
            raise ValueError('CUDA measurements require eager base guards')
        if cuda_geome__torch.device(kwargs['device']).type != 'cuda':
            raise ValueError('CUDA measurements require a CUDA device')
        profile = CudaGeometryProfile.load() if profile is None else profile.verify()
        if cuda_geome__torch.cuda.get_device_capability(kwargs['device']) != (8, 9):
            raise ValueError('validated CUDA measurements require SM89')
        super().__init__(shapes, pads, backend='eager', **kwargs)
        from dh116.cuda_measurements.hull_support import HullSupport
        from dh116.cuda_measurements.pad_support import PadSupport
        self.cuda_profile = profile
        self.cuda_hulls = HullSupport([(s, l, k, v) for (s, l, k, v, _) in self._shapes], self._normals, device=self.device, library=profile.binary_path)
        self.cuda_pads = PadSupport(self._pads, self._pad_roundoff, device=self.device, library=profile.binary_path) if self._pads else None

    def _metrics(self, p, nodes, radii, rotations):
        (low, high, projected) = self.cuda_hulls.extrema(rotations.contiguous(), p[..., :3].contiguous())
        rod_high = (nodes @ self._normals.T).amax(1)
        lower = (projected - rod_high[:, None]).amax(-1).clamp_min(0) - radii[:, None] - cuda_geome__NUMERIC_GUARD
        return (low, high, lower)

    def _pad_metrics(self, p, nodes, radii, rotations, thresholds):
        if self.cuda_pads is None:
            return self._pad_metrics_eager(p, nodes, radii, rotations, thresholds)
        return self.cuda_pads.metrics(rotations.contiguous(), p[..., :3].contiguous(), nodes.contiguous(), radii.contiguous(), thresholds)

def make_tensor_geometry(owner, body_names, *, device, backend=None):
    """Same inherited guards, either eager or independently compiled CUDA math."""
    mode = cuda_geome__os.environ.get('DH116_TENSOR_GEOMETRY_BACKEND', 'eager') if backend is None else backend
    if mode not in ('eager', MODE):
        raise ValueError('tensor geometry backend must be eager or cuda_cabi; compile disabled')
    cls = CudaMeasurementGeometry if mode == MODE else cuda_geome__TensorGeometry
    return cls.from_materialized_geometry(owner['geometry'], owner['grids'], owner['sources'], body_names, required_pads=owner['required_pads'], device=device, backend='eager')

def geometry_measurement_identity(geometry):
    """Report the real measurement backend without granting qualification."""
    if isinstance(geometry, CudaMeasurementGeometry):
        return geometry.cuda_profile.identity()
    return dict(actual_measurement_backend=getattr(geometry, 'backend', None), binary_sha256=None, manifest_sha256=None, source_sha256={}, native_ready=False, teacher_admission=False)

def verify_measurement_native_receipts(receipts, *, style):
    """Accept only current-source natural lifecycle evidence; no old receipt bypass."""
    if style not in ('pinch', 'tripod', 'power') or not receipts:
        raise ValueError('grasp and current native evidence required')
    profile = CudaGeometryProfile.load()
    records = []
    for (path, digest) in receipts:
        raw = checked_file(Path(path), digest)
        record = json.loads(raw)
        if record.get('process_exit_verified') is not True or record.get('exit_code') != 0 or record.get('grasp') != style or (record.get('scene_sha256') is None) or (record.get('geometry_manifest_sha256') != profile.manifest_sha256) or (record.get('runtime_graph_compilation') is not False):
            raise ValueError('matching current source, scene and natural exit evidence required')
        records.append(record)
    return records
