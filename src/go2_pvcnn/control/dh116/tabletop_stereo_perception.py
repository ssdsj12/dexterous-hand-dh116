"""RGB-only visible-wire proposal, topology rejection and stereo 3D tokens.

Warm-colour segmentation is an explicit baseline for the current gold wire,
not a semantic ground-truth mask. It rejects ambiguous skeletons and never
reconstructs hidden segments from simulator geometry. Native accuracy and
cross-frame identity remain separate validation tasks.
"""
import numpy as np
import torch

from .tabletop_stereo_geometry import backproject, transform_points


def _thin(mask):
    """Zhang–Suen thinning; delete-only topology operations on observed pixels."""
    image = np.pad(mask.astype(bool), 1)
    for _ in range(128):
        changed = False
        for second in (False, True):
            p = [image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:], image[2:, 2:],
                 image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2], image[:-2, :-2]]
            count = sum(x.astype(np.uint8) for x in p)
            transitions = sum((~p[i] & p[(i+1) % 8]).astype(np.uint8) for i in range(8))
            center = image[1:-1, 1:-1]
            if second:
                clear_a, clear_b = ~(p[0] & p[2] & p[6]), ~(p[0] & p[4] & p[6])
            else:
                clear_a, clear_b = ~(p[0] & p[2] & p[4]), ~(p[2] & p[4] & p[6])
            remove = center & (count >= 2) & (count <= 6) & (transitions == 1) & clear_a & clear_b
            if remove.any():
                changed = True
                center[remove] = False
        if not changed:
            break
    return image[1:-1, 1:-1]


def _visible_path(skeleton):
    pixels = set(map(tuple, np.argwhere(skeleton)))
    if len(pixels) < 2:
        return None
    neighbours = {p: [q for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        if (dy or dx) and (q := (p[0]+dy, p[1]+dx)) in pixels] for p in pixels}
    ends = [p for p in pixels if len(neighbours[p]) == 1]
    if len(ends) != 2 or any(len(v) > 2 or not v for v in neighbours.values()):
        return None
    path = [min(ends)]
    used = {path[0]}
    while len(path) < len(pixels):
        nxt = [q for q in neighbours[path[-1]] if q not in used]
        if len(nxt) != 1:
            return None
        path.append(nxt[0]); used.add(nxt[0])
    return np.asarray(path, dtype=np.int64)


class WirePointExtractor:
    """A missing/branched/occluded cable produces invalid tokens, never truth."""
    def __init__(self, *, point_count=128, identity_motion_limit_m=.005):
        if point_count != 128 or identity_motion_limit_m <= 0:
            raise ValueError('this contract uses 128 points and a positive consistency radius')
        self.point_count = point_count
        self.identity_motion_limit_m = float(identity_motion_limit_m)
        self._previous = None

    @torch.no_grad()
    def extract(self, left_rgb, depth_m, depth_valid, K_left, T_world_left, *, reset_mask=None):
        if (left_rgb.ndim != 4 or left_rgb.shape[1] != 3 or depth_m.shape != left_rgb.shape[:1] + left_rgb.shape[2:]
                or depth_valid.shape != depth_m.shape or depth_valid.dtype != torch.bool
                or not torch.isfinite(left_rgb).all() or bool(((left_rgb < 0) | (left_rgb > 1)).any())):
            raise ValueError('finite RGB and matching stereo depth validity required')
        n = len(left_rgb)
        if self._previous is None:
            self._previous = [None] * n
        if len(self._previous) != n:
            raise ValueError('point extractor batch size changed; construct a new instance')
        if reset_mask is not None:
            if reset_mask.shape != (n,) or reset_mask.dtype != torch.bool:
                raise ValueError('boolean reset mask [B] required')
            for i in reset_mask.nonzero().flatten().tolist():
                self._previous[i] = None
        rgb = left_rgb
        candidate = ((rgb[:, 0] > 1.07*rgb[:, 1]) & (rgb[:, 1] > 1.1*rgb[:, 2])
                     & (rgb[:, 0] > .12))
        candidate &= depth_valid & torch.isfinite(depth_m) & (depth_m > 0)
        camera = backproject(depth_m, K_left, depth_valid)
        world = transform_points(camera, T_world_left)
        points = world.new_full((n, 128, 3), float('nan'))
        valid = torch.zeros(n, 128, device=world.device, dtype=torch.bool)
        correspondence = torch.zeros(n, device=world.device, dtype=torch.bool)
        reasons, lengths = [], world.new_full((n,), float('nan'))
        for i in range(n):
            path = _visible_path(_thin(candidate[i].cpu().numpy()))
            if path is None:
                self._previous[i] = None
                reasons.append('ambiguous_or_branched_visible_skeleton')
                continue
            pixel = torch.as_tensor(path, device=world.device)
            xyz = world[i, pixel[:, 0], pixel[:, 1]]
            axis = int((xyz.amax(0)-xyz.amin(0)).argmax())
            if xyz[0, axis] > xyz[-1, axis]:
                xyz = xyz.flip(0)
            arc = torch.cat((xyz.new_zeros(1), (xyz[1:]-xyz[:-1]).norm(dim=-1).cumsum(0)))
            length = float(arc[-1])
            old = self._previous[i]
            if old is not None:
                endpoints, prior_length = old
                movement = (xyz[[0, -1]].cpu()-endpoints).norm(dim=-1).max()
                correspondence[i] = bool(movement <= self.identity_motion_limit_m
                    and abs(length-prior_length) <= self.identity_motion_limit_m)
            self._previous[i] = (xyz[[0, -1]].cpu().clone(), length)
            if len(xyz) > 128:
                # Actual observed samples nearest uniform visible arc-length
                # positions; no interpolation into missing image regions.
                indices = torch.searchsorted(arc, torch.linspace(0, length, 128, device=arc.device))
                xyz = xyz[indices.clamp_max(len(xyz)-1)]
            points[i, :len(xyz)] = xyz
            valid[i, :len(xyz)] = True
            lengths[i] = length
            reasons.append(None)
        return dict(points_world=points, point_valid=valid,
            temporal_identity_valid=correspondence, visible_arc_length_m=lengths,
            rejection_reason=reasons, source='RGB_colour_visible_skeleton_and_stereo_only',
            correspondence_scope='visible_endpoint_consistency_not_native_identity_qualification',
            centerline_qualification=None)


# Frozen comparator: callers must opt into the observed-component tracker.
LegacyWirePointExtractor = WirePointExtractor
_legacy_visible_path = _visible_path

from .tabletop_stereo_wire_tracking import TrackedWirePointExtractor  # noqa: E402,F401
