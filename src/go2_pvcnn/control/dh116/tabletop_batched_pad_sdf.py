"""Owned batched pad-SDF cache; no stepping, handles or physical qualification.

Default float32 mode retains the original NumPy/SciPy local-coordinate arithmetic
before GPU interpolation, avoiding backend dot-product changes at lattice faces.
Float64 mode uses Torch transforms for mathematical parity diagnostics. Grids are
copied once to the requested device; capture replaces the complete frame cache.
Call validate with the actual lane state before using row in an existing guard.
"""
from copy import deepcopy

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _owned_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _sample_grid(values, origin, spacing, local):
    """Original sample_grid arithmetic order, including fractional-only clipping."""
    inv = 1 / spacing
    u = (local - origin) * inv
    shape = torch.tensor(values.shape, device=values.device, dtype=values.dtype)
    valid = ((u >= 0) & (u < shape - 1)).all(dim=-1)
    phi = torch.full(local.shape[:-1], torch.nan, dtype=values.dtype, device=values.device)
    gradient = torch.full_like(local, torch.nan)
    # Only real in-lattice nodes reach corner reads; no outside index is clamped.
    points = local[valid]
    index = u[valid].to(torch.int64)
    f = ((points - (origin + index.to(values.dtype) * spacing)) * inv).clamp(0, 1)
    fx, fy, fz = f.unbind(-1)

    def corner(a, b, c):
        return values[index[:, 0] + a, index[:, 1] + b, index[:, 2] + c]

    c000, c001, c010, c011, c100, c101, c110, c111 = [corner(*x) for x in (
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))]
    dx00, dx01, dx10, dx11 = c100-c000, c101-c001, c110-c010, c111-c011
    c00, c01, c10, c11 = c000+fx*dx00, c001+fx*dx01, c010+fx*dx10, c011+fx*dx11
    dy0, dy1 = c10-c00, c11-c01
    c0, c1 = c00+fy*dy0, c01+fy*dy1
    phi[valid] = c0+fz*(c1-c0)
    gx0, gx1 = dx00+fy*(dx10-dx00), dx01+fy*(dx11-dx01)
    gradient[valid] = torch.stack((gx0+fz*(gx1-gx0), dy0+fz*(dy1-dy0), c1-c0), dim=-1)*inv
    return phi, gradient, valid


class BatchedPadSDF:
    """Sample original grids using actual [N,2,23,7] XYZW poses and material41.

    pads contains (side_index, actual_link_index, pad_name, original_grid).
    dtype defaults to torch.float32 to match sample_contact's default contract.
    Input tensors may be on CPU or CUDA; borrowed inputs are never retained.
    frame must identify the completed same-clock boundary, including reset/episode.
    """

    def __init__(self, pads, *, device, dtype=torch.float32, bindings=None):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("float32 compatibility or float64 mathematical dtype required")
        self.device, self.dtype = torch.device(device), dtype
        self._pads, self.columns, self.links = [], {}, {}
        self.source_bindings = deepcopy(bindings) if bindings is not None else {}
        self.frame = None
        npdtype = np.float32 if dtype == torch.float32 else np.float64
        for side, link, name, grid in pads:
            if (type(side) is not int or side not in (0, 1) or type(link) is not int or not 0 <= link < 23
                    or not isinstance(name, str) or not name or (side, name) in self.columns):
                raise ValueError("distinct actual pad side/link mapping required")
            values = np.array(grid["values"], dtype=npdtype, copy=True)
            origin = np.array(grid["origin"], dtype=npdtype, copy=True)
            spacing = np.array(grid["spacing"], dtype=npdtype, copy=True)
            if (values.ndim != 3 or min(values.shape) < 2 or origin.shape != (3,)
                    or spacing.shape != () or not np.isfinite(values).all()
                    or not np.isfinite(origin).all() or not np.isfinite(spacing) or spacing <= 0):
                raise ValueError("finite three-dimensional locked SDF required")
            self.columns[side, name] = len(self._pads)
            self.links[side, name] = link
            self._pads.append((side, link,
                torch.tensor(values, dtype=dtype, device=self.device),
                torch.tensor(origin, dtype=dtype, device=self.device),
                torch.tensor(spacing, dtype=dtype, device=self.device)))
        if not self._pads:
            raise ValueError("original pad grids required")
        if bindings is not None and set(self.source_bindings) != set(self.columns):
            raise ValueError("one exact source binding per original grid required")

    @torch.no_grad()
    def capture(self, poses, nodes, *, frame, radius=.005):
        """Replace the cache atomically; failed capture invalidates previous rows."""
        self.frame = None
        poses, nodes = _owned_numpy(poses), _owned_numpy(nodes)
        if (frame is None or poses.ndim != 4 or poses.shape[1:] != (2, 23, 7)
                or not 1 <= len(poses) <= 512 or nodes.shape != (len(poses), 41, 3)
                or poses.dtype.kind != "f" or nodes.dtype.kind != "f"
                or not np.isfinite(poses).all() or not np.isfinite(nodes).all()
                or np.any(abs(np.linalg.norm(poses[..., 3:].astype(np.float64), axis=-1)-1) > 2e-6)):
            raise ValueError("complete finite actual XYZW poses and material41 nodes required")
        radii = _owned_numpy(radius).astype(np.float64)
        if radii.shape == ():
            radii = np.full(len(poses), float(radii), dtype=np.float64)
        if radii.shape != (len(poses),) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("finite positive actual radius per lane required")
        radius_tensor = torch.as_tensor(radii, dtype=self.dtype, device=self.device)[:, None]
        if self.dtype == torch.float32:
            # scipy normalizes the exact input quaternion just as pose_matrix does.
            rotation = Rotation.from_quat(poses[..., 3:].reshape(-1, 4)).as_matrix()
            rotation = rotation.reshape(*poses.shape[:-1], 3, 3).astype(np.float32)
            native_nodes, translation = nodes.astype(np.float32), poses[..., :3].astype(np.float32)
        else:
            p = torch.as_tensor(poses, dtype=torch.float64, device=self.device)
            q = p[..., 3:]
            q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
            x, y, z, w = q.unbind(-1)
            rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), dim=-1).reshape(*q.shape[:-1], 3, 3)
            native_nodes = torch.as_tensor(nodes, dtype=torch.float64, device=self.device)
            translation = p[..., :3]
        gaps, normals, validity, locations, minima = [], [], [], [], []
        for side, link, values, origin, spacing in self._pads:
            local = (native_nodes-translation[:, side, link, None]) @ rotation[:, side, link]
            if self.dtype == torch.float32:
                if not np.isfinite(local).all():
                    raise ValueError("finite float32 original local nodes required")
                local = torch.as_tensor(local, dtype=self.dtype, device=self.device)
            phi, gradient, valid = _sample_grid(values, origin, spacing, local)
            gap = phi-radius_tensor
            # Explicit sum order matches NumPy's three-component norm calculation.
            norm = torch.sqrt((gradient*gradient).sum(dim=-1))
            minimum = gap.masked_fill(~valid, torch.inf).argmin(dim=-1)
            minimum = torch.where(valid.any(dim=-1), minimum, -1)
            gaps.append(gap); normals.append(norm); validity.append(valid)
            locations.append(local); minima.append(minimum)
        self.node_gaps = torch.stack(gaps, dim=1).cpu().numpy()
        self.gradient_norms = torch.stack(normals, dim=1).cpu().numpy()
        self.valid = torch.stack(validity, dim=1).cpu().numpy()
        self.local_nodes = torch.stack(locations, dim=1).cpu().numpy()
        self.minimum_node_ids = torch.stack(minima, dim=1).cpu().numpy()
        self.poses, self.nodes, self.radii = poses, nodes, radii
        for a in (self.node_gaps, self.gradient_norms, self.valid, self.local_nodes,
                  self.minimum_node_ids, self.poses, self.nodes, self.radii):
            a.setflags(write=False)
        self.frame = deepcopy(frame)

    def _check_lane(self, lane):
        if self.frame is None:
            raise ValueError("pad SDF cache frame unavailable or invalidated")
        if type(lane) is not int or not 0 <= lane < len(self.poses):
            raise ValueError("actual pad SDF lane required")

    def validate(self, lane, poses, nodes, *, frame):
        """Reject cache reuse across boundaries, lane state changes or resets."""
        self._check_lane(lane)
        if frame != self.frame:
            raise ValueError("pad SDF cache frame changed")
        if not np.array_equal(_owned_numpy(poses), self.poses[lane]):
            raise ValueError("measured pose changed after pad SDF batch capture")
        if not np.array_equal(_owned_numpy(nodes), self.nodes[lane]):
            raise ValueError("material nodes changed after pad SDF batch capture")

    def row(self, lane, side, name):
        """Return an independent original sample_contact row after validation."""
        self._check_lane(lane)
        column = self.columns[side, name]
        valid = self.valid[lane, column]
        gap, norm = self.node_gaps[lane, column], self.gradient_norms[lane, column]
        index = int(self.minimum_node_ids[lane, column])
        index = None if index < 0 else index
        return dict(minimum_gap_m=None if index is None else float(gap[index]), minimum_node_id=index,
            gradient_norm_at_minimum=None if index is None else float(norm[index]),
            in_lattice_nodes=np.flatnonzero(valid).tolist(),
            negative_gap_nodes=np.flatnonzero(valid & (gap < 0)).tolist(),
            contact_predicate_nodes=np.flatnonzero(valid & (gap < 0) & (norm > 0)).tolist(),
            node_gaps_m=[float(x) if ok else None for x, ok in zip(gap, valid)],
            local_node_at_minimum=None if index is None else self.local_nodes[lane, column, index].astype(float).tolist(),
            cpu_dtype="float32" if self.dtype == torch.float32 else "float64", native_contact_observed=False)
