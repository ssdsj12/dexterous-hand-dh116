"""Owned tensor geometry over every original hull vertex and locked pad SDF.

This is the numerical part of PolicyOwnedPinchGeometry, not a replacement for
its SDK/source/wrist guards. Call capture at EVERY completed SDK step, before
another step may start. Negative or boundary-ambiguous bounds are resolved by
the original CPU routines synchronously inside capture. Ordinary rows never
leave the input device. Sparse fallback rows are the only CPU data transfers.

Tripod's additional route/support/raw-quaternion source/cooked geometry guard
is explicitly NOT implemented here; its original guard must still run.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch

from .tabletop_deformable_wrist_projection import CLEARANCE_M
from .tabletop_measured_closure_geometry import NUMERIC_GUARD, rod_support_lower
from .tabletop_polyline_support_certificate import RADIUS, certify, validate as validate_certificate

VERSION = "dh116-tensor-geometry-v1"
DECISION_MARGIN_M = 2e-6
# Dimensionless lattice-coordinate distance to ANY integer cell face, including
# both outer faces. This is a routing margin, never a changed lattice predicate.
LATTICE_MARGIN = 1e-4


@dataclass(frozen=True, slots=True)
class TensorGeometryResult:
    frame: object
    source_bindings: dict
    poses: torch.Tensor
    nodes: torch.Tensor
    radii: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    table_clearance_m: torch.Tensor
    non_target_lower_m: torch.Tensor
    hand_separation_lower_m: torch.Tensor
    table_failure: torch.Tensor
    non_target_failure: torch.Tensor
    inter_hand_failure: torch.Tensor
    sdf_node_gaps_m: torch.Tensor
    sdf_node_valid: torch.Tensor
    sdf_negative_nodes: torch.Tensor
    sdf_contact_nodes: torch.Tensor
    sdf_minimum_node_ids: torch.Tensor
    sdf_minimum_gap_m: torch.Tensor
    sdf_gradient_norm_at_minimum: torch.Tensor
    sdf_valid: torch.Tensor
    positive_pad_mean_m: torch.Tensor
    source_complete: torch.Tensor
    geometry_passed: torch.Tensor
    fallback_rows: torch.Tensor
    support_fallback: torch.Tensor
    table_fallback: torch.Tensor
    separation_fallback: torch.Tensor
    sdf_fallback: torch.Tensor
    ready: bool = True
    native_qualification: bool = False
    tripod_supported_guard_covered: bool = False


def _rotation_xyzw(poses):
    q = poses[..., 3:]
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def _sample_grid_float64(values, origin, spacing, local):
    """Original ordered trilinear math, with fixed-shape masked corner reads.

    Invalid nodes read an inert index zero and are replaced with NaN. There is
    no boolean compaction/nonzero, CPU local transform, outside-cell clipping or
    dynamic output shape. Only the original fractional coordinate is clipped.
    """
    inv = 1/spacing
    u = (local-origin)*inv
    shape = u.new_tensor(values.shape)
    valid = ((u >= 0) & (u < shape-1)).all(-1)
    index = torch.where(valid[..., None], u, 0).to(torch.int64)
    safe_local = torch.where(valid[..., None], local, origin)
    f = ((safe_local-(origin+index.to(values.dtype)*spacing))*inv).clamp(0, 1)
    fx, fy, fz = f.unbind(-1)
    def corner(a, b, c):
        return values[index[..., 0]+a, index[..., 1]+b, index[..., 2]+c]
    c000, c001, c010, c011, c100, c101, c110, c111 = [corner(*x) for x in (
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))]
    dx00, dx01, dx10, dx11 = c100-c000, c101-c001, c110-c010, c111-c011
    c00, c01, c10, c11 = c000+fx*dx00, c001+fx*dx01, c010+fx*dx10, c011+fx*dx11
    dy0, dy1 = c10-c00, c11-c01
    c0, c1 = c00+fy*dy0, c01+fy*dy1
    phi = c0+fz*(c1-c0)
    gx0, gx1 = dx00+fy*(dx10-dx00), dx01+fy*(dx11-dx01)
    gradient = torch.stack((gx0+fz*(gx1-gx0), dy0+fz*(dy1-dy0), c1-c0), -1)*inv
    return phi.masked_fill(~valid, torch.nan), gradient.masked_fill(~valid[..., None], torch.nan), valid


class TensorGeometry:
    """Static source inventory plus one owned [N,2,23,7]/[N,41,3] frame.

    shapes: (side_index, actual_link_index, "kind:body_name", ALL vertices).
    pads: (side_index, actual_link_index, body_name, original locked grid).
    The construction/startup owner is responsible for establishing the original
    inventory and source identity. from_policy_geometry preserves its complete
    inventory and locked bindings without loading assets or acquiring an SDK.
    """

    def __init__(self, shapes, pads=(), *, device, bindings=None, body_names=None,
                 sdf_callable=None, pose_matrix=None, transform=None):
        from scripts.audit_dh116_pinch_initial_pose import pose_matrix as original_matrix, transform as original_transform
        from scripts.prepare_dh116_pinch_closure_margin import sample_contact
        self.device = torch.empty(0, device=device).device
        self._sdf = sample_contact if sdf_callable is None else sdf_callable
        self._pose_matrix = original_matrix if pose_matrix is None else pose_matrix
        self._transform = original_transform if transform is None else transform
        self._shapes, self.columns, self.vertex_counts = [], {}, []
        self._pads, self.pad_columns, self.pad_links = [], {}, {}
        self._pad_roundoff = []
        self.body_names = deepcopy(body_names)
        self.frame = self.result = None
        for side, link, key, vertices in shapes:
            v = np.array(vertices, dtype=np.float64, copy=True)
            if (type(side) is not int or side not in (0, 1) or type(link) is not int or not 0 <= link < 23
                    or type(key) is not str or ":" not in key or (side, key) in self.columns
                    or v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all()):
                raise ValueError("distinct original complete finite source hull/actual link required")
            self.columns[side, key] = len(self._shapes)
            v.setflags(write=False)
            self.vertex_counts.append(len(v))
            self._shapes.append((side, link, key, v, torch.tensor(v, dtype=torch.float64, device=self.device)))
        if not self._shapes or {s[0] for s in self._shapes} != {0, 1}:
            raise ValueError("complete bilateral original shapes required")
        for side, link, name, grid in pads:
            if (type(side) is not int or side not in (0, 1) or type(link) is not int or not 0 <= link < 23
                    or type(name) is not str or not name or (side, name) in self.pad_columns):
                raise ValueError("distinct actual original pad mapping required")
            original = {k: np.array(v, copy=True) for k, v in grid.items()}
            values = np.array(grid["values"], dtype=np.float64, copy=True)
            origin = np.array(grid["origin"], dtype=np.float64, copy=True)
            spacing = np.array(grid["spacing"], dtype=np.float64, copy=True)
            if (values.ndim != 3 or min(values.shape) < 2 or origin.shape != (3,) or spacing.shape != ()
                    or not np.isfinite(values).all() or not np.isfinite(origin).all()
                    or not np.isfinite(spacing) or spacing <= 0):
                raise ValueError("finite original three-dimensional locked SDF required")
            self.pad_columns[side, name] = len(self._pads)
            self.pad_links[side, name] = link
            self._pads.append((side, link, name, original,
                torch.tensor(values, dtype=torch.float64, device=self.device),
                torch.tensor(origin, dtype=torch.float64, device=self.device),
                torch.tensor(spacing, dtype=torch.float64, device=self.device)))
            # Static conservative bounds for the original FP32 cast and
            # interpolation, including large-world-coordinate cancellation.
            values32 = values.astype(np.float32).astype(np.float64)
            origin32 = origin.astype(np.float32).astype(np.float64)
            spacing32 = float(np.float32(spacing))
            if (not np.isfinite(values32).all() or not np.isfinite(origin32).all()
                    or not math.isfinite(spacing32) or spacing32 <= 0):
                raise ValueError("original float32 SDF representation must remain finite")
            slopes = np.array([np.abs(np.diff(values, axis=i)).max()/float(spacing) for i in range(3)])
            constants = np.r_[slopes, np.abs(values32-values).max(), np.abs(values).max(),
                              np.abs(origin32-origin), abs(spacing32-float(spacing))]
            self._pad_roundoff.append(torch.tensor(constants, dtype=torch.float64, device=self.device))
        self.source_bindings = deepcopy(bindings) if bindings is not None else {}
        if bindings is not None and set(bindings) != set(self.pad_columns):
            raise ValueError("one exact source binding per original pad required")
        self._locked_bindings = deepcopy(self.source_bindings)
        self._locked_body_names = deepcopy(body_names)
        self.vertex_counts = tuple(self.vertex_counts)
        self._side_ids = [[j for (s, _), j in self.columns.items() if s == side] for side in (0, 1)]
        self._pad_side_ids = [[j for (s, _), j in self.pad_columns.items() if s == side] for side in (0, 1)]
        target = {(side, name) for side, _, name, *_ in self._pads}
        self.non_target = torch.tensor([(side, key.split(":", 1)[1]) not in target
            for side, _, key, *_ in self._shapes], dtype=torch.bool, device=self.device)
        a = np.arange(64)*2*np.pi/64
        normals = np.stack((np.cos(a), np.zeros(64), np.sin(a)), 1)
        normals /= np.linalg.norm(normals, axis=1)[:, None]+1e-12
        self._normals = torch.tensor(normals, dtype=torch.float64, device=self.device)

    @classmethod
    def from_policy_geometry(cls, owner, body_names, *, device):
        """Copy every source/cooked vertex and every REQUIRED_PADS locked grid.

        This helper does not validate an exchange's live SDK identity; that stays
        with the original caller. In particular Tripod's extra source guard has
        not been replaced by these bounds.
        """
        names = tuple(tuple(n) for n in body_names)
        if len(names) != 2 or any(len(n) != 23 or len(set(n)) != 23 for n in names):
            raise ValueError("both complete actual 23-link body mappings required")
        shapes, pads, bindings = [], [], {}
        for side, label in enumerate(("left", "right")):
            for kind, entries in owner._geometry[label].items():
                for name, vertices in entries.items():
                    shapes.append((side, names[side].index(name), kind+":"+name, vertices))
            for name in owner.REQUIRED_PADS:
                pads.append((side, names[side].index(name), name, owner._grids[label][name]))
                bindings[side, name] = deepcopy(owner.sources[label]["sdf_binding"][name])
        result = cls(shapes, pads, device=device, bindings=bindings, body_names=names,
                     sdf_callable=owner._sdf, pose_matrix=owner._pose_matrix, transform=owner._transform)
        result.sources = deepcopy(owner.sources)
        return result

    def validate_metadata(self, *, bindings=None, body_names=None):
        if self.source_bindings != self._locked_bindings or (bindings is not None and bindings != self._locked_bindings):
            raise ValueError("original pad source binding changed")
        if self.body_names != self._locked_body_names or (body_names is not None and tuple(tuple(n) for n in body_names) != self._locked_body_names):
            raise ValueError("actual body/link source mapping changed")

    def require_ready(self, *, frame, bindings=None, body_names=None):
        """Guard the consuming boundary before another SDK step or Reset."""
        if self.frame is None or self.result is None or frame != self.frame:
            raise ValueError("tensor geometry frame/Reset epoch unavailable or changed")
        self.validate_metadata(bindings=bindings, body_names=body_names)
        return self.result

    def _separation(self, minimum, maximum):
        low = [minimum[:, ids].amin(1) for ids in self._side_ids]
        high = [maximum[:, ids].amax(1) for ids in self._side_ids]
        return torch.maximum((low[1]-high[0]).amax(1), (low[0]-high[1]).amax(1))

    def _metrics(self, p, nodes, radii, rotations):
        rod_high = (nodes @ self._normals.T).amax(1)
        minimum, maximum, lower = [], [], []
        # Static source/chunk loops, independent of lane count. Every vertex is
        # evaluated; the 128-vertex chunk only limits temporary GPU allocation.
        for side, link, _, _, vertices in self._shapes:
            lo = hi = projected = None
            for start in range(0, len(vertices), 128):
                world = vertices[None, start:start+128] @ rotations[:, side, link].transpose(-1, -2)+p[:, side, link, None, :3]
                a, b, c = world.amin(1), world.amax(1), (world @ self._normals.T).amin(1)
                lo = a if lo is None else torch.minimum(lo, a)
                hi = b if hi is None else torch.maximum(hi, b)
                projected = c if projected is None else torch.minimum(projected, c)
            minimum.append(lo)
            maximum.append(hi)
            lower.append((projected-rod_high).amax(1).clamp_min(0)-radii-NUMERIC_GUARD)
        return torch.stack(minimum, 1), torch.stack(maximum, 1), torch.stack(lower, 1)

    def _pad_metrics(self, p, nodes, radii, rotations, thresholds):
        gaps, norms, validities, fallback = [], [], [], []
        for (side, link, _, _, values, origin, spacing), rounding in zip(self._pads, self._pad_roundoff):
            local = (nodes-p[:, side, link, None, :3]) @ rotations[:, side, link]
            phi, gradient, valid = _sample_grid_float64(values, origin, spacing, local)
            gap = phi-radii[:, None]
            norm = torch.sqrt((gradient*gradient).sum(-1))
            u = (local-origin)/spacing
            limit = u.new_tensor(values.shape)-1
            # Estimate original FP32 input-cast/subtraction displacement with
            # tensor operations, then bound its 3-term dot/interpolation error.
            # The bound only routes to the original CPU oracle; the primary
            # trilinear values and every final threshold stay unchanged.
            eps = torch.finfo(torch.float32).eps
            relative32 = (nodes.float()-p[:, side, link, None, :3].float()).double()
            rotation32 = rotations[:, side, link].float().double()
            local32_center = relative32 @ rotation32
            local_error = ((local32_center-local).abs()
                +8*eps*(relative32.abs() @ rotation32.abs())+1e-12)
            u_error = ((local_error+rounding[5:8])/spacing
                +u.abs()*rounding[8]/spacing+8*eps*(u.abs()+1)+1e-10)
            lattice_band = u_error.clamp_min(LATTICE_MARGIN)
            possible = ((u >= -lattice_band) & (u <= limit+lattice_band)).all(-1)
            face = possible & ((u-u.round()).abs() <= lattice_band).any(-1)
            radius_error = (radii.float().double()-radii).abs()[:, None]
            gap_error = ((u_error*spacing*rounding[:3]).sum(-1)+rounding[3]
                +64*eps*rounding[4]+radius_error).clamp_min(DECISION_MARGIN_M)
            near = torch.zeros_like(valid)
            for threshold in thresholds:
                near |= valid & ((gap-threshold).abs() <= gap_error)
            # The original source predicate includes gradient>0, and argmin
            # uses the first FP32 minimum. Flat/near-tied rows need its oracle.
            norm_error = (16*(rounding[3]+eps*rounding[4])/spacing
                +4*rounding[:3].amax()/spacing*local_error.sum(-1)).clamp_min(DECISION_MARGIN_M)
            flat = valid & (norm <= norm_error)
            best = gap.masked_fill(~valid, torch.inf).topk(2, dim=-1, largest=False).values
            tied = valid.any(-1) & ((best[:, 1]-best[:, 0]).abs() <= 2*gap_error.amax(-1))
            fallback.append(face.any(-1) | near.any(-1) | flat.any(-1) | tied)
            gaps.append(gap)
            norms.append(norm)
            validities.append(valid)
        if not self._pads:
            shape = (len(p), 0, 41)
            return (p.new_empty(shape), p.new_empty(shape), torch.empty(shape, dtype=torch.bool, device=self.device),
                    torch.empty((len(p), 0), dtype=torch.bool, device=self.device))
        return torch.stack(gaps, 1), torch.stack(norms, 1), torch.stack(validities, 1), torch.stack(fallback, 1)

    def _fallback(self, p, nodes, radii, minimum, maximum, lower, separation,
                  table_flags, support_flags, separation_flags, sdf_flags, gaps,
                  norms, valid, negative, contact):
        rows = table_flags.any(1) | support_flags.any(1) | separation_flags | sdf_flags.any(1)
        ids = rows.nonzero().flatten()
        # Only selected bad rows cross devices. Flag/index copies contain no
        # ordinary row data. No per-lane containers exist on the ordinary path.
        indices = ids.cpu().tolist()
        cpu_p = p.index_select(0, ids).cpu().numpy()
        cpu_n = nodes.index_select(0, ids).cpu().numpy()
        cpu_r = radii.index_select(0, ids).cpu().numpy()
        tf = table_flags.index_select(0, ids).cpu().numpy()
        sf = support_flags.index_select(0, ids).cpu().numpy()
        hf = separation_flags.index_select(0, ids).cpu().numpy()
        pf = sdf_flags.index_select(0, ids).cpu().numpy()
        for sparse, lane in enumerate(indices):
            matrices = {}
            def matrix(side, link):
                if (side, link) not in matrices:
                    pose = cpu_p[sparse, side, link]
                    matrices[side, link] = self._pose_matrix(pose[[0, 1, 2, 6, 3, 4, 5]])
                return matrices[side, link]
            whole = [[], []]
            for column, (side, link, _, vertices, _) in enumerate(self._shapes):
                if tf[sparse, column] or sf[sparse, column] or hf[sparse]:
                    world = self._transform(vertices, matrix(side, link))
                    minimum[lane, column] = minimum.new_tensor(world.min(0))
                    maximum[lane, column] = maximum.new_tensor(world.max(0))
                    if hf[sparse]:
                        whole[side].append(world)
                    if sf[sparse, column]:
                        original = rod_support_lower(world, cpu_n[sparse])+RADIUS-cpu_r[sparse]
                        if original < 0:
                            certificate = certify(world, cpu_n[sparse])
                            validate_certificate(certificate, world, cpu_n[sparse])
                            original = certificate["lower_m"]+RADIUS-cpu_r[sparse]
                        lower[lane, column] = float(original)
            if hf[sparse]:
                left, right = [np.concatenate(v) for v in whole]
                separation[lane] = float(max((right.min(0)-left.max(0)).max(), (left.min(0)-right.max(0)).max()))
            for column, (side, link, _, grid, *_) in enumerate(self._pads):
                if not pf[sparse, column]:
                    continue
                row = self._sdf(grid, matrix(side, link), cpu_n[sparse], radius=float(cpu_r[sparse]))
                values = [math.nan if v is None else v for v in row["node_gaps_m"]]
                gaps[lane, column] = gaps.new_tensor(values)
                valid[lane, column] = False
                valid[lane, column, row["in_lattice_nodes"]] = True
                negative[lane, column] = False
                negative[lane, column, row["negative_gap_nodes"]] = True
                contact[lane, column] = False
                contact[lane, column, row["contact_predicate_nodes"]] = True
                # Only norms at the selected minimum are consumed below.
                # Reconstruct original FP32 local/grid arithmetic for the full
                # norm array, so original first-tie and coverage remain exact.
                from scripts.prepare_dh116_pinch_closure_margin import sample_grid
                t = matrix(side, link)
                local = (cpu_n[sparse].astype(np.float32)-t[:3, 3].astype(np.float32)) @ t[:3, :3].astype(np.float32)
                _, gradient, _ = sample_grid(grid, local)
                norms[lane, column] = norms.new_tensor(np.linalg.norm(gradient, axis=1))
        return rows

    @torch.no_grad()
    def capture(self, poses, nodes, *, frame, radius=.005, sdf_thresholds=()):
        """Resolve a completed substep before returning GPU bounds and masks.

        Zero is always an SDF decision threshold; supply every additional actual
        consumer threshold in metres. Routing within +/-2um does not widen any
        original predicate. Lattice routing uses +/-1e-4 in lattice coordinates.
        Input data must already be on this device. No SDK step or native contact
        assertion is made. Invalid captures invalidate the previous epoch.
        """
        self.frame = self.result = None
        self.validate_metadata()
        if (frame is None or not isinstance(poses, torch.Tensor) or poses.device != self.device
                or poses.ndim != 4 or poses.shape[1:] != (2, 23, 7) or not 1 <= len(poses) <= 512
                or poses.dtype not in (torch.float32, torch.float64) or not isinstance(nodes, torch.Tensor)
                or nodes.device != self.device or nodes.shape != (len(poses), 41, 3)
                or nodes.dtype not in (torch.float32, torch.float64)):
            raise ValueError("complete same-device floating native poses/material41 and frame required")
        if isinstance(radius, torch.Tensor):
            if radius.device != self.device or radius.dtype not in (torch.float32, torch.float64):
                raise ValueError("same-device floating actual radius required")
            radii = radius.detach().double().clone()
        else:
            if type(radius) not in (int, float):
                raise ValueError("finite positive scalar or same-device actual radius required")
            radii = torch.tensor(float(radius), dtype=torch.float64, device=self.device)
        if radii.shape == ():
            radii = radii.expand(len(poses)).clone()
        if radii.shape != (len(poses),):
            raise ValueError("one finite positive actual radius per lane required")
        thresholds = (0.,)+tuple(float(t) for t in sdf_thresholds)
        if not all(math.isfinite(t) for t in thresholds):
            raise ValueError("finite actual SDF decision thresholds required")
        # Promote before all geometry arithmetic, matching the original SciPy
        # normalized-quaternion/float64 complete-vertex oracle.
        p, n = poses.detach().double().clone(), nodes.detach().double().clone()
        invalid = (~torch.isfinite(p).all() | ~torch.isfinite(n).all() | ~torch.isfinite(radii).all()
            | (radii <= 0).any() | ((torch.linalg.vector_norm(p[..., 3:], dim=-1)-1).abs() > 2e-6).any())
        rotations = _rotation_xyzw(p)
        minimum, maximum, lower = self._metrics(p, n, radii, rotations)
        separation = self._separation(minimum, maximum)
        invalid |= (~torch.isfinite(minimum).all() | ~torch.isfinite(maximum).all()
                    | ~torch.isfinite(lower).all() | ~torch.isfinite(separation).all())
        table_flags = (minimum[..., 2]-.3-CLEARANCE_M).abs() <= DECISION_MARGIN_M
        support_flags = self.non_target[None] & (lower <= DECISION_MARGIN_M)
        separation_flags = separation.abs() <= DECISION_MARGIN_M
        gaps, norms, valid, sdf_flags = self._pad_metrics(p, n, radii, rotations, thresholds)
        negative = valid & (gaps < 0)
        contact = negative & (norms > 0)
        fallback_rows = table_flags.any(1) | support_flags.any(1) | separation_flags | sdf_flags.any(1)
        # One aggregate host flag on the ordinary path. Sparse exceptional
        # frames may read additional validation/selection flags synchronously.
        if bool(invalid | fallback_rows.any()):
            if bool(invalid):
                raise ValueError("complete finite unit native poses/material41 and positive radius required")
            self._fallback(p, n, radii, minimum, maximum, lower, separation,
                table_flags, support_flags, separation_flags, sdf_flags, gaps, norms, valid, negative, contact)
        table = minimum[..., 2]-.3
        table_failure = table < CLEARANCE_M
        non_target_failure = self.non_target[None] & (lower < 0)
        inter_hand_failure = separation < 0
        any_valid = valid.any(-1)
        minimum_ids = gaps.masked_fill(~valid, torch.inf).argmin(-1)
        minimum_gap = gaps.gather(-1, minimum_ids[..., None]).squeeze(-1)
        minimum_norm = norms.gather(-1, minimum_ids[..., None]).squeeze(-1)
        minimum_ids = torch.where(any_valid, minimum_ids, -1)
        minimum_gap = minimum_gap.masked_fill(~any_valid, torch.nan)
        minimum_norm = minimum_norm.masked_fill(~any_valid, torch.nan)
        sdf_valid = any_valid & torch.isfinite(minimum_gap) & torch.isfinite(minimum_norm) & (minimum_norm > 0)
        means = []
        for ids in self._pad_side_ids:
            if ids:
                mean = minimum_gap[:, ids].clamp_min(0).sum(1)/len(ids)
                means.append(mean.masked_fill(~sdf_valid[:, ids].all(1), torch.nan))
            else:
                means.append(p.new_full((len(p),), torch.nan))
        source_complete = sdf_valid.all(1)
        passed = source_complete & ~(table_failure.any(1) | non_target_failure.any(1) | inter_hand_failure)
        result = TensorGeometryResult(deepcopy(frame), deepcopy(self.source_bindings), p, n, radii,
            minimum, maximum, table, lower, separation, table_failure, non_target_failure,
            inter_hand_failure, gaps, valid, negative, contact, minimum_ids, minimum_gap,
            minimum_norm, sdf_valid, torch.stack(means, 1), source_complete, passed,
            fallback_rows, support_flags, table_flags, separation_flags, sdf_flags)
        self.frame, self.result = deepcopy(frame), result
        return result
