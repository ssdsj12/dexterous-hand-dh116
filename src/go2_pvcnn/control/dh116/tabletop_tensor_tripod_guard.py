"""Tripod-only raw-convex support guards and seven-route tensor adapter.

The numerical fast path evaluates every original vertex on the input device.
Ambiguous or failing lower bounds synchronously use check_supported_shape with
its unchanged SAT/full-envelope certificates. No SDF can clear a raw convex
failure. This module does not step Isaac/DeformX or grant native qualification.
The caller must retain the original joint, wrist, source, Reset and pre-rod
1um table guards, call capture at every substep, and validate_substep before
another SDK step. Startup/Reset distance coverage is distinct from admission.
"""
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import torch

from .tabletop_tensor_geometry import TensorGeometry, DECISION_MARGIN_M

REQUIRED_PADS = ("finger14_link", "finger23_link", "finger33_link")
SUPPORT_NAMES = ("table", "support_0", "support_1")
SUPPORT_LIMIT_M = 1e-10
NON_TARGET_LIMIT_M = .00025
PAD_PENETRATION_M = .00015
PAIR_CAPACITY = 41*17*12


@dataclass(frozen=True, slots=True)
class TensorTripodSupportResult:
    frame: object
    support_gap_lower_m: torch.Tensor
    rod_gap_lower_m: torch.Tensor
    failures: torch.Tensor
    passed: torch.Tensor
    fallback_rows: torch.Tensor
    fallback_shapes: torch.Tensor
    details: dict
    native_qualification: bool = False


class TensorTripodSupportGuard:
    """Numerical raw-quaternion part of PolicyOwnedTripodGeometry._evaluate.

    shapes use TensorGeometry's (side, actual_link, kind:name, ALL vertices).
    supports are the original two task boxes before horizontal env translation.
    Construction alone never authenticates a native owner; use the source-bound
    TensorTripodGuard factory in production.
    """
    def __init__(self, shapes, supports, *, env_origins, device):
        self.device = torch.device(device)
        self.origins = torch.as_tensor(env_origins, device=self.device, dtype=torch.float64).detach().clone()
        if (self.origins.ndim != 2 or self.origins.shape[1] != 3
                or not 1 <= len(self.origins) <= 512 or not bool(torch.isfinite(self.origins).all())
                or bool((self.origins[:, 2] != 0).any())):
            raise ValueError("finite horizontal actual environment origins required")
        self.supports = deepcopy(supports)
        self._locked_supports = deepcopy(self.supports)
        self._locked_origins = self.origins.clone()
        if (len(supports) != 2 or len({b["name"] for b in supports}) != 2
                or any(type(b["name"]) is not str or not b["name"] for b in supports)):
            raise ValueError("both original independent external supports required")
        centers, sizes = (np.array([b[k] for b in supports], dtype=np.float64) for k in ("center_m", "size_m"))
        if centers.shape != (2, 3) or sizes.shape != (2, 3) or not np.isfinite([centers, sizes]).all() or np.any(sizes <= 0):
            raise ValueError("finite complete original support boxes required")
        self.low = self.origins[:, None]+torch.tensor(centers-sizes/2, device=self.device)
        self.high = self.origins[:, None]+torch.tensor(centers+sizes/2, device=self.device)
        self.shapes, keys = [], set()
        for side, link, key, vertices in shapes:
            v = np.array(vertices, dtype=np.float64, copy=True)
            if (type(side) is not int or side not in (0, 1) or type(link) is not int or not 0 <= link < 23
                    or not isinstance(key, str) or ":" not in key or (side, key) in keys
                    or v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 4 or not np.isfinite(v).all()):
                raise ValueError("complete distinct original convex vertices and actual links required")
            keys.add((side, key))
            v.setflags(write=False)
            self.shapes.append((side, link, key, v, torch.tensor(v, device=self.device), key.split(":", 1)[1] in REQUIRED_PADS))
        if {s[0] for s in self.shapes} != {0, 1}:
            raise ValueError("both original hand shape inventories required")
        # Identical normal order/construction to archived_tube, not a new bound.
        azimuth = np.linspace(0., 2*np.pi, 72, endpoint=False)
        elevation = np.linspace(-np.pi/2, np.pi/2, 25)[:, None]
        normals = np.stack(np.broadcast_arrays(np.cos(elevation)*np.cos(azimuth),
            np.cos(elevation)*np.sin(azimuth), np.sin(elevation)), -1).reshape(-1, 3)
        self.normals = torch.tensor(normals, dtype=torch.float64, device=self.device)
        self.groups = torch.arange(0, 40, 2, device=self.device)[:, None]+torch.arange(3, device=self.device)
        self.frame = self.result = None

    def require_ready(self, *, frame):
        if self.frame is None or frame != self.frame or self.result is None:
            raise ValueError("Tripod raw support frame/Reset epoch unavailable or changed")
        return self.result

    @torch.no_grad()
    def capture(self, poses, nodes, *, frame):
        self.frame = self.result = None
        n = len(self.origins)
        if self.supports != self._locked_supports or not torch.equal(self.origins, self._locked_origins):
            raise ValueError("original Tripod support source or environment origins changed")
        for value, shape in ((poses, (n, 2, 23, 7)), (nodes, (n, 41, 3))):
            if (not isinstance(value, torch.Tensor) or value.device != self.device or value.shape != shape
                    or value.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(value).all())):
                raise ValueError("complete finite same-device actual raw poses/material41 required")
        p, rod = poses.detach().double().clone(), nodes.detach().double().clone()
        if frame is None or bool(((p[..., 3:].norm(dim=-1)-1).abs() > 1e-5).any()):
            raise ValueError("native unit-tolerance raw quaternion and frame required")
        upper = (rod[:, self.groups] @ self.normals.T).amax(-2)+.005
        support, gaps = [], []
        for side, link, key, original, vertices, target in self.shapes:
            pose = p[:, side, link]
            lo = hi = plane_low = None
            for start in range(0, len(vertices), 32):
                v = vertices[None, start:start+32].expand(n, -1, -1)
                xyz = pose[:, None, 3:6].expand_as(v)
                cross = 2*torch.linalg.cross(xyz, v, dim=-1)
                world = v+pose[:, None, 6:7]*cross+torch.linalg.cross(xyz, cross, dim=-1)+pose[:, None, :3]
                a, b, c = world.amin(1), world.amax(1), (world @ self.normals.T).amin(1)
                lo = a if lo is None else torch.minimum(lo, a)
                hi = b if hi is None else torch.maximum(hi, b)
                plane_low = c if plane_low is None else torch.minimum(plane_low, c)
            support.append(torch.maximum(torch.maximum(self.low-hi[:, None], lo[:, None]-self.high),
                torch.zeros_like(self.low)).norm(dim=-1))
            gap = (plane_low[:, None]-upper).amax(-1).amin(-1)
            gaps.append(gap-1e-12 if target else gap)
        support = torch.stack(support, 1)
        gaps = torch.stack(gaps, 1)
        targets = torch.tensor([s[-1] for s in self.shapes], dtype=torch.bool, device=self.device)
        thresholds = torch.where(targets, gaps.new_tensor(-PAD_PENETRATION_M), gaps.new_tensor(NON_TARGET_LIMIT_M))
        # Routing margin grows with world-coordinate cancellation; it never
        # changes the final original oracle predicate.
        magnitude = p[..., :3].abs().flatten(1).amax(1)+rod.abs().flatten(1).amax(1)
        extent = max(float(np.abs(s[3]).max()) for s in self.shapes)
        margin = DECISION_MARGIN_M+128*torch.finfo(torch.float64).eps*(magnitude+extent+1)
        fallback = (support <= SUPPORT_LIMIT_M+margin[:, None, None]).any(-1) | (gaps <= thresholds+margin[:, None])
        failures = torch.zeros_like(gaps, dtype=torch.bool)
        details = {}
        if not bool(torch.isfinite(support).all() & torch.isfinite(gaps).all()):
            raise ValueError("nonfinite Tripod raw convex lower bound")
        if bool(fallback.any()):
            from scripts.verify_dh116_power_shared_rod import check_supported_shape
            from scripts.screen_dh116_tripod_supported_reset import archived_tube, separation
            from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
            from scripts.audit_dh116_certified_segment_distance import ENVELOPE_REFINEMENT_VERSION
            from .tabletop_batch_geometry import rotated_points
            selected = fallback.any(1).nonzero().flatten()
            lanes = selected.cpu().tolist()
            cpu_p = p.index_select(0, selected).cpu().numpy()
            cpu_n = rod.index_select(0, selected).cpu().numpy()
            flags = fallback.index_select(0, selected).cpu().numpy()
            lows, highs = self.low.index_select(0, selected).cpu().numpy(), self.high.index_select(0, selected).cpu().numpy()
            for sparse, lane in enumerate(lanes):
                boxes = [(b["name"], lows[sparse, j], highs[sparse, j]) for j, b in enumerate(self.supports)]
                tube = None
                for col in np.flatnonzero(flags[sparse]):
                    side, link, key, vertices, _, target = self.shapes[col]
                    pose = cpu_p[sparse, side, link]
                    world = rotated_points(pose[[0, 1, 2, 6, 3, 4, 5]], vertices)
                    if not target and tube is None:
                        tube = archived_tube(cpu_n[sparse:sparse+1])
                    certificates = []
                    failure = check_supported_shape(world, boxes, None if target else tube,
                        contact_shape=target, contact_penetration_m=PAD_PENETRATION_M if target else None,
                        reference_nodes=cpu_n[sparse:sparse+1],
                        envelope_refinement=ENVELOPE_REFINEMENT_VERSION if target else None,
                        record_refinement=certificates.append)
                    support[lane, col] = support.new_tensor([support_clearance_lower(world, low, high) for _, low, high in boxes])
                    gaps[lane, col] = certificates[0]["lower_m"] if target else separation(world, *tube)
                    failures[lane, col] = failure is not None
                    if failure is not None:
                        details[lane, int(col)] = failure
        result = TensorTripodSupportResult(frame, support, gaps, failures, ~failures.any(1),
            fallback.any(1), fallback, details)
        self.frame, self.result = deepcopy(frame), result
        return result


@dataclass(frozen=True, slots=True)
class TensorTripodRouteResult:
    forces: torch.Tensor
    torques: torch.Tensor
    body_ids: tuple
    external_support_wrenches_world: torch.Tensor
    native_qualification: bool = False


class TensorTripodRoutes:
    """Original seventeen-slot, seven world/COM route and source allocation gate."""
    def __init__(self, bindings, body_names, mesh_names, *, device):
        from .tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES, _bindings, contact_links
        self.bindings = _bindings(bindings)
        self._locked_bindings = deepcopy(self.bindings)
        self.body_names = tuple(tuple(n) for n in body_names)
        self.mesh_names = tuple(mesh_names)
        self.device = torch.device(device)
        if (self.mesh_names != PROXIMAL_POWER_MESH_NAMES or len(self.body_names) != 2
                or any(len(n) != 23 or len(set(n)) != 23 for n in self.body_names)):
            raise ValueError("complete original Tripod17 mesh/body inventory required")
        links = contact_links(self.bindings)
        if len(links) != 7:
            raise ValueError("all seven original COM/world routes required")
        self.slots, self.body_ids = [], []
        for side, label in enumerate(("left", "right")):
            keys = [label+"."+name for name in links]
            bodies = tuple(self.bindings[k]["body_index"] for k in keys)
            if bodies != tuple(self.body_names[side].index(name) for name in links):
                raise ValueError("same actual side-specific route/body identities required")
            self.slots.append(tuple(self.mesh_names.index(k) for k in keys))
            self.body_ids.append(bodies)
        self.body_ids = tuple(self.body_ids)
        self.support_slots = tuple(self.mesh_names.index(k) for k in SUPPORT_NAMES)

    @torch.no_grad()
    def validate_substep(self, state, transaction, flat_capture, stamp):
        from .tabletop_deformable_solver_clock import solver_clock_matches
        from .tabletop_shared_teacher_source import SharedSourceIdentity
        pre = transaction.get("stamp")
        if (type(stamp) is not SharedSourceIdentity or type(pre) is not SharedSourceIdentity
                or self.bindings != self._locked_bindings):
            raise ValueError("actual unchanged typed Tripod source identity and bindings required")
        n = stamp.parent_num_envs
        if (not 2 <= n <= 512 or transaction.get("phase") != "finished" or pre is None
                or stamp.sdk_step != pre.sdk_step+1 or stamp.rod_ticks != tuple(t+12 for t in pre.rod_ticks)
                or not solver_clock_matches(pre.solver_time_s, stamp.solver_time_s, steps=1)
                or any(getattr(pre, k) != getattr(stamp, k) for k in ("parent_solver_id", "parent_num_envs",
                    "env_rows", "run_uuid", "episode_uids", "rod_uids", "reset_ids", "profile_sha256", "scope"))
                or flat_capture.get("pair_owner_time_s") != stamp.solver_time_s):
            raise ValueError("same-step complete Tripod source clock/Reset epoch required")
        # This is source-qualified data from flat_cuda_pair_capture, not a new
        # invented capacity: missing metadata must stay unavailable.
        meta = flat_capture.get("pair_lane_meta")
        if (not isinstance(meta, torch.Tensor) or meta.device != self.device or meta.dtype != torch.int32
                or meta.shape != (n, 4) or bool(((meta[:, 0] != 0) | (meta[:, 1] != 0)
                    | (meta[:, 2] != PAIR_CAPACITY) | (meta[:, 3] < 0) | (meta[:, 3] > meta[:, 2])).any())):
            raise ValueError("actual complete bounded seventeen-slot pair allocation source required")
        def floating(value, shape, label):
            if (not isinstance(value, torch.Tensor) or value.device != self.device or value.shape != shape
                    or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
                raise ValueError("finite complete actual "+label+" source required")
            return value.double()
        p = floating(state["link_pose_xyzw"], (n, 2, 23, 7), "link pose")
        com = floating(state["body_com_pose_w"], (n, 2, 23, 7), "COM pose")
        w = floating(flat_capture["mesh_wrenches"], (n, 17, 6), "named mesh wrench")
        force = floating(flat_capture["node_forces"], (n, 41, 3), "distributed force")
        if bool(((p[..., 3:].norm(dim=-1)-1).abs() > 1e-5).any()):
            raise ValueError("actual raw unit-tolerance route quaternion required")
        counts, owners, indices = (flat_capture[k] for k in ("counts", "env_ids", "indices"))
        if (any(not isinstance(v, torch.Tensor) or v.device != self.device or v.dtype != torch.int64
                for v in (counts, owners, indices)) or counts.shape != (n,) or indices.shape != (len(owners), 3)
                or owners.ndim != 1 or not torch.equal(counts, meta[:, 3].long())
                or bool(((owners < 0) | (owners >= n)).any())
                or bool(((indices < 0) | (indices >= indices.new_tensor([12, 41, 17]))).any())
                or not torch.equal(torch.bincount(owners, minlength=n), counts)):
            raise ValueError("complete same-step bounded original pair indices/allocation required")
        floating(flat_capture["vectors"], (len(owners), 12), "pair vectors")
        # Keep the original force guard before any native FP32 cast or write.
        hand = w[:, self.slots[0]+self.slots[1]]
        if bool((hand[..., :3].norm(dim=-1) > 30).any()):
            raise ValueError("original 30 N reaction guard failed")
        if bool(((force.sum(1)+w[..., :3].sum(1)).norm(dim=-1) > 1e-3).any()):
            raise ValueError("original rod/mesh force balance guard failed")
        forces, torques = [], []
        for side in (0, 1):
            values = w[:, self.slots[side]]
            # Original route_world_wrenches._poses casts SDK origin to FP32.
            origin = p[:, side, self.body_ids[side], :3].float().double()
            center = com[:, side, self.body_ids[side], :3]
            forces.append(values[..., :3])
            torques.append(values[..., 3:]+torch.linalg.cross(origin-center, values[..., :3], dim=-1))
        return TensorTripodRouteResult(torch.stack(forces, 1), torch.stack(torques, 1), self.body_ids,
            w[:, self.support_slots].clone())


@dataclass(frozen=True, slots=True)
class TensorTripodGeometryResult:
    base: object
    supported: TensorTripodSupportResult
    geometry_passed: torch.Tensor
    tripod_supported_guard_covered: bool = False
    native_qualification: bool = False

    def __getattr__(self, name):
        return getattr(self.base, name)


class TensorTripodGuard:
    """Source-bound adapter; original collector keeps its other exact guards."""
    @classmethod
    def from_policy_geometry(cls, original, body_names, *, env_origins, device):
        from .tabletop_native_teacher_policy_tripod_distance import PolicyOwnedTripodGeometry
        from .tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES
        if not isinstance(original, PolicyOwnedTripodGeometry) or tuple(original.REQUIRED_PADS) != REQUIRED_PADS:
            raise ValueError("actual original three-pad Tripod geometry owner required")
        # The legacy live _evaluate verifies all five locked SDF channels,
        # including the two non-selected distal pads. Preserve that source
        # contract when freezing static tensor inventories at construction.
        for side in ("left", "right"):
            usd_sha = original.sources[side]["source_sha256"]["usd"]
            for name in original._grids[side]:
                locked = original.sources[side]["sdf_binding"][name]
                actual = original._mesh_bindings[side+"."+name]
                if (actual.get("sdf_sha256") != locked["sha256"]
                        or actual.get("capture_sha256") != locked["capture_sha256"]):
                    raise ValueError("original locked Tripod SDF/capture source differs")
            for actual in original._mesh_bindings.values():
                if actual["side"] == side and (actual.get("frame") != "PhysX_link_origin"
                        or actual.get("unit") != "m" or actual.get("source_usd_sha256") != usd_sha):
                    raise ValueError("original side-specific Tripod source frame/unit/USD differs")
        result = cls()
        result.geometry = TensorGeometry.from_policy_geometry(original, body_names, device=device)
        result.routes = TensorTripodRoutes(original._mesh_bindings, body_names, PROXIMAL_POWER_MESH_NAMES, device=device)
        result.support = TensorTripodSupportGuard([(s, l, k, v) for s, l, k, v, _ in result.geometry._shapes],
            original._support["supports"], env_origins=env_origins, device=device)
        result.frame = result.result = result.validated_substep = None
        result.expected_radius = original._expected_radius()
        return result

    @torch.no_grad()
    def capture(self, poses, nodes, *, frame, radius):
        self.frame = self.result = self.validated_substep = None
        r = torch.as_tensor(radius, device=self.geometry.device, dtype=torch.float64)
        tolerance = torch.maximum(r.abs(), r.new_tensor(abs(self.expected_radius)))*2e-6
        if bool((~torch.isfinite(r) | ((r-self.expected_radius).abs() > tolerance.clamp_min(1e-9))).any()):
            raise ValueError("original locked Tripod physical radius source differs")
        base = self.geometry.capture(poses, nodes, frame=frame, radius=radius)
        raw = self.support.capture(poses, nodes, frame=frame)
        result = TensorTripodGeometryResult(base, raw, base.geometry_passed & raw.passed)
        self.frame, self.result = deepcopy(frame), result
        return result

    def require_ready(self, *, frame):
        if self.frame is None or frame != self.frame or self.result is None:
            raise ValueError("Tripod combined geometry frame/Reset epoch unavailable or changed")
        self.geometry.require_ready(frame=frame)
        self.support.require_ready(frame=frame)
        return self.result

    def validate_substep(self, state, transaction, flat_capture, stamp):
        result = self.require_ready(frame=stamp)
        if not torch.equal(state["link_pose_xyzw"].double(), result.poses):
            raise ValueError("Tripod raw geometry and route SDK poses differ")
        routes = self.routes.validate_substep(state, transaction, flat_capture, stamp)
        if not bool((result.source_complete & result.geometry_passed).all()):
            raise ValueError("original Tripod raw/source/cooked geometry or SDF coverage guard failed")
        self.validated_substep = deepcopy(stamp)
        self.result = replace(result, tripod_supported_guard_covered=True)
        return routes
