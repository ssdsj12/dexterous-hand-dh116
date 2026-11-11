"""Observed RGB paths with masked stereo samples and conservative temporal tracks.

Image components are never joined across an occluder. A track is a local sensor
association hypothesis, not a simulator/material-point identity qualification.
Metric arc coordinates restart after a depth hole; pixel arc retains the complete
observed image path. Neither coordinate interpolates a hidden 3D segment.
"""
import math

import numpy as np
import torch

from .tabletop_stereo_geometry import backproject, transform_points


def normalized_visible_paths(skeleton):
    """Return simple 8-connected paths, removing only redundant corner diagonals.

    A diagonal has a redundant axial route exactly when either of its shared
    axial intermediates is present. True diagonal chains retain their edges.
    Every connected component must be a simple path with at least two pixels.
    """
    pixels = set(map(tuple, np.argwhere(skeleton)))
    if not pixels:
        return [], None
    neighbours = {}
    for y, x in sorted(pixels):
        edges = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if not (dy or dx) or q not in pixels:
                    continue
                if dy and dx and ((y, x + dx) in pixels or (y + dy, x) in pixels):
                    continue
                edges.append(q)
        neighbours[y, x] = edges
    remaining = set(pixels)
    paths = []
    while remaining:
        component, stack = set(), [min(remaining)]
        while stack:
            p = stack.pop()
            if p in component:
                continue
            component.add(p)
            stack.extend(q for q in neighbours[p] if q not in component)
        remaining.difference_update(component)
        ends = sorted(p for p in component if len(neighbours[p]) == 1)
        if len(ends) != 2 or any(len(neighbours[p]) > 2 or not neighbours[p] for p in component):
            return [], 'ambiguous_or_branched_visible_skeleton'
        path, used = [ends[0]], {ends[0]}
        while len(path) < len(component):
            nexts = [q for q in neighbours[path[-1]] if q not in used]
            if len(nexts) != 1:
                return [], 'ambiguous_or_branched_visible_skeleton'
            path.append(nexts[0]); used.add(nexts[0])
        paths.append(np.asarray(path, dtype=np.int64))
    return paths, None


def _arc_pixels(path):
    return np.concatenate(([0.], np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()))


def _paths_from_mask(mask):
    """Return independently valid visible paths and rejected raw components."""
    from .tabletop_stereo_perception import _thin

    paths, reason = normalized_visible_paths(mask)
    if not reason:
        return paths, reason, []
    # Thin each raw image component independently. Thinning a thick fragment
    # must not erase a separate, already-simple digital staircase.
    remaining = set(map(tuple, np.argwhere(mask)))
    paths = []
    rejected = []
    while remaining:
        stack, component = [min(remaining)], set()
        while stack:
            p = stack.pop()
            if p in component:
                continue
            component.add(p)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (p[0] + dy, p[1] + dx)
                    if (dy or dx) and q in remaining and q not in component:
                        stack.append(q)
        remaining.difference_update(component)
        pixel = np.array(sorted(component))
        offset = pixel.min(0)
        local = pixel - offset
        patch = np.zeros(tuple(local.max(0) + 1), dtype=bool)
        patch[local[:, 0], local[:, 1]] = True
        local_paths, reason = normalized_visible_paths(patch)
        thick = bool((patch[:-1, :-1] & patch[1:, :-1]
                      & patch[:-1, 1:] & patch[1:, 1:]).any())
        if reason and thick:
            local_paths, reason = normalized_visible_paths(_thin(patch))
        if reason or len(local_paths) != 1:
            rejected.append(dict(reason='ambiguous_or_branched_visible_skeleton',
                pixel_count=len(component), min_yx=offset.tolist(), max_yx=pixel.max(0).tolist()))
            continue
        paths.extend(path + offset for path in local_paths)
    reason = None if paths else 'ambiguous_or_branched_visible_skeleton'
    return paths, reason, rejected


def _uniform_indices(arc, count):
    """Unique observed pixels nearest uniform arc targets; never interpolate."""
    if count >= len(arc):
        return np.arange(len(arc))
    if count == 1:
        return np.array([len(arc) // 2])
    targets = np.linspace(0., arc[-1], count)
    right = np.searchsorted(arc, targets).clip(0, len(arc) - 1)
    left = np.maximum(0, right - 1)
    nearest = np.where(abs(arc[left] - targets) <= abs(arc[right] - targets), left, right)
    # Digital steps lie in [1,sqrt(2)], and targets are >=1 apart, but do
    # not duplicate tokens if a highly unequal digital arc hits the same pixel.
    return np.unique(nearest)


def _budgets(paths, capacity):
    if sum(map(len, paths)) <= capacity:
        return [len(path) for path in paths]
    sizes = np.array([len(path) for path in paths])
    lengths = np.array([_arc_pixels(path)[-1] for path in paths])
    allocation = np.zeros(len(paths), dtype=int)
    # Include both endpoints when capacity permits. Excess components are
    # explicitly reported as truncated instead of being joined together.
    priority = np.argsort(-lengths, kind='stable')
    for index in priority:
        if allocation.sum() == capacity:
            break
        allocation[index] = min(2, capacity - allocation.sum())
    while allocation.sum() < capacity:
        score = np.where(allocation < sizes, lengths / np.maximum(allocation, 1), -1.)
        allocation[int(score.argmax())] += 1
    return allocation.tolist()


def _unit(vector):
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return torch.where(norm > 1e-10, vector / norm.clamp_min(1e-10), float('nan'))


def _descriptor(xyz, good):
    # A partial depth curve may retain 2D order, but is insufficient for a
    # complete endpoint/arc comparison. Its identity deliberately stays unknown.
    if len(xyz) < 2 or not bool(good.all()):
        return None
    arc = torch.cat((xyz.new_zeros(1), torch.diff(xyz, dim=0).norm(dim=-1).cumsum(0)))
    if not bool(torch.isfinite(arc[-1])) or float(arc[-1]) <= 1e-8:
        return None
    index = _uniform_indices(arc.cpu().numpy(), 17)
    curve = xyz[index].detach().cpu()
    delta = _unit(torch.diff(curve, dim=0))
    curvature = torch.diff(delta, dim=0).norm(dim=-1)
    return dict(endpoints=curve[[0, -1]], tangent=torch.stack((delta[0], delta[-1])),
                curve=curve, curvature=curvature, length=float(arc[-1]))


def _reverse_descriptor(descriptor):
    return dict(endpoints=descriptor['endpoints'].flip(0), tangent=-descriptor['tangent'].flip(0),
                curve=descriptor['curve'].flip(0), curvature=descriptor['curvature'].flip(0),
                length=descriptor['length'])


class TrackedWirePointExtractor:
    """Sensor-only visible fragments; fixed-capacity tokens with explicit masks.

    timestamp is optional monotonic seconds (scalar or [B]). Without it, calls
    use a 30 Hz logical clock. Explicit resets, nonfinite/nonincreasing times,
    and gaps above max_timestamp_gap_s clear the environment's entire history.
    point_confidence, if supplied, is calibrated/uncalibrated matcher confidence
    [B,H,W] in [0,1]; it is copied only at valid observed depth pixels.
    """
    def __init__(self, *, point_count=128, identity_motion_limit_m=.005,
                 max_timestamp_gap_s=.5, association_margin=.15):
        if (not isinstance(point_count, int) or not 2 <= point_count <= 128
                or not math.isfinite(identity_motion_limit_m) or identity_motion_limit_m <= 0
                or not math.isfinite(max_timestamp_gap_s) or max_timestamp_gap_s <= 0
                or not math.isfinite(association_margin) or association_margin <= 0):
            raise ValueError('2..128 samples and positive finite tracking limits required')
        self.point_count = point_count
        self.identity_motion_limit_m = float(identity_motion_limit_m)
        self.max_timestamp_gap_s = float(max_timestamp_gap_s)
        self.association_margin = float(association_margin)
        self._history = None
        self._times = None
        self._next_id = 0

    def _associate(self, descriptors, old, now):
        """Mutual unambiguous endpoint/tangent/curvature/arc/history matching."""
        matches = [(False, -1, None) for _ in descriptors]
        if not old:
            for index, descriptor in enumerate(descriptors):
                if descriptor is not None:
                    matches[index] = (False, self._next_id, False)
                    self._next_id += 1
            return matches
        candidates = []
        radius = self.identity_motion_limit_m
        for current, descriptor in enumerate(descriptors):
            if descriptor is None:
                continue
            for prior, track in enumerate(old):
                dt = now - track['time']
                prediction = track['endpoints'] + track['velocity'] * dt
                for reverse in (False, True):
                    oriented = _reverse_descriptor(descriptor) if reverse else descriptor
                    displacement = (oriented['endpoints'] - prediction).norm(dim=-1)
                    length_delta = abs(oriented['length'] - track['length'])
                    if float(displacement.max()) > radius or length_delta > max(radius, .2 * track['length']):
                        continue
                    tangent_cost = (1 - (oriented['tangent'] * track['tangent']).sum(-1)).mean()
                    # Descriptor count varies only for very short curves. Endpoint
                    # and arc gates remain conservative when shape sizes differ.
                    if not oriented['curvature'].numel() and not track['curvature'].numel():
                        curvature_cost = torch.tensor(0.)
                    elif oriented['curvature'].shape == track['curvature'].shape:
                        curvature_cost = (oriented['curvature'] - track['curvature']).square().mean().sqrt()
                    else:
                        curvature_cost = torch.tensor(1.)
                    shape_cost = 0.
                    if oriented['curve'].shape == track['curve'].shape:
                        translation = (track['velocity'] * dt).mean(0)
                        shape_cost = float((oriented['curve'] - track['curve'] - translation).norm(dim=-1).mean()) / radius
                    cost = (float(displacement.mean()) / radius + .15 * float(tangent_cost)
                            + .1 * float(curvature_cost) + .2 * length_delta / radius + .1 * shape_cost)
                    if math.isfinite(cost):
                        candidates.append((cost, current, prior, reverse))
        for current in range(len(descriptors)):
            options = sorted(item for item in candidates if item[1] == current)
            if not options:
                continue
            best = options[0]
            # Both component competition and orientation competition count as
            # ambiguity. Unknown is preferable to forcing an identity swap.
            if len(options) > 1 and options[1][0] - best[0] < self.association_margin:
                continue
            competitors = sorted(item for item in candidates if item[2] == best[2])
            if competitors[0] != best:
                continue
            if len(competitors) > 1 and competitors[1][0] - best[0] < self.association_margin:
                continue
            matches[current] = (True, old[best[2]]['id'], best[3])
        return matches

    @torch.no_grad()
    def extract(self, left_rgb, depth_m, depth_valid, K_left, T_world_left, *,
                reset_mask=None, timestamp=None, point_confidence=None):
        if (left_rgb.ndim != 4 or left_rgb.shape[1] != 3 or not left_rgb.is_floating_point()
                or depth_m.shape != left_rgb.shape[:1] + left_rgb.shape[2:]
                or depth_valid.shape != depth_m.shape or depth_valid.dtype != torch.bool
                or not torch.isfinite(left_rgb).all() or bool(((left_rgb < 0) | (left_rgb > 1)).any())):
            raise ValueError('finite RGB and matching stereo depth validity required')
        n, device, dtype = len(left_rgb), depth_m.device, depth_m.dtype
        if reset_mask is None:
            reset_mask = torch.zeros(n, dtype=torch.bool)
        if reset_mask.shape != (n,) or reset_mask.dtype != torch.bool:
            raise ValueError('boolean reset mask [B] required')
        if point_confidence is not None:
            point_confidence = torch.as_tensor(point_confidence, device=device, dtype=dtype)
            if (point_confidence.shape != depth_m.shape or not torch.isfinite(point_confidence).all()
                    or bool(((point_confidence < 0) | (point_confidence > 1)).any())):
                raise ValueError('point_confidence requires finite [B,H,W] in [0,1]')
        if self._history is None:
            self._history, self._times = [[] for _ in range(n)], [None] * n
        if len(self._history) != n:
            raise ValueError('point extractor batch size changed; construct a new instance')
        if timestamp is None:
            times = [0. if old is None else old + 1 / 30 for old in self._times]
        else:
            time_tensor = torch.as_tensor(timestamp, dtype=torch.float64)
            if time_tensor.ndim == 0:
                time_tensor = time_tensor.expand(n)
            if time_tensor.shape != (n,):
                raise ValueError('timestamp must be scalar or [B] seconds')
            times = time_tensor.tolist()

        # Build and validate all RGB topology before looking up any depth.
        candidate = ((left_rgb[:, 0] > 1.07 * left_rgb[:, 1])
                     & (left_rgb[:, 1] > 1.1 * left_rgb[:, 2]) & (left_rgb[:, 0] > .12))
        image_paths, reasons, rejected_components = [], [], []
        for mask in candidate.cpu().numpy():
            paths, reason, rejected = _paths_from_mask(mask)
            image_paths.append(paths)
            rejected_components.append(rejected)
            reasons.append(reason if reason else (None if paths else 'missing_visible_wire'))
        world = transform_points(backproject(depth_m, K_left, depth_valid), T_world_left)
        shape = (n, self.point_count)
        xyz = world.new_full((*shape, 3), float('nan'))
        valid = torch.zeros(shape, device=device, dtype=torch.bool)
        visibility = torch.zeros_like(valid)
        component_id = torch.full(shape, -1, device=device, dtype=torch.long)
        track_id = torch.full_like(component_id, -1)
        depth_segment_id = torch.full_like(component_id, -1)
        pixels = torch.full((*shape, 2), -1, device=device, dtype=torch.long)
        arc_length = world.new_full(shape, float('nan'))
        arc_length_px = world.new_full(shape, float('nan'))
        tangent = torch.full_like(xyz, float('nan'))
        confidence = world.new_zeros(shape)
        topology_valid = torch.tensor([bool(p) and r is None for p, r in zip(image_paths, reasons)], device=device)
        temporal_valid = torch.zeros(n, device=device, dtype=torch.bool)
        anomaly = torch.zeros_like(temporal_valid)
        metric_complete = torch.zeros_like(temporal_valid)
        truncated = torch.zeros_like(temporal_valid)
        lengths = world.new_full((n,), float('nan'))
        component_count = torch.tensor([len(p) for p in image_paths], device=device, dtype=torch.long)
        rejected_count = torch.tensor([len(rows) for rows in rejected_components], device=device, dtype=torch.long)
        sampling_arc_space = [[] for _ in range(n)]
        for env, paths in enumerate(image_paths):
            now, previous_time = times[env], self._times[env]
            bad_time = (not math.isfinite(now) or (previous_time is not None
                        and (now <= previous_time or now - previous_time > self.max_timestamp_gap_s)))
            anomaly[env] = bad_time
            if bool(reset_mask[env]) or bad_time:
                self._history[env] = []
            if not math.isfinite(now):
                now = 0.
            self._times[env] = now
            if not bool(topology_valid[env]):
                self._history[env] = []
                continue
            curves = []
            for path in paths:
                pixel = torch.as_tensor(path, device=device)
                curve = world[env, pixel[:, 0], pixel[:, 1]]
                good = torch.isfinite(curve).all(-1)
                curves.append((pixel, curve, good))
            descriptors = [_descriptor(curve, good) for _, curve, good in curves]
            old = self._history[env]
            matches = self._associate(descriptors, old, now)
            budgets = _budgets(paths, self.point_count)
            truncated[env] = any(budget < len(path) for budget, path in zip(budgets, paths))
            history, cursor, next_segment = [], 0, 0
            all_matched = bool(paths)
            lengths[env] = 0.
            metric_complete[env] = True
            for component, ((pixel, curve, good), descriptor, match, budget) in enumerate(zip(curves, descriptors, matches, budgets)):
                matched, identity, reverse = match
                all_matched &= matched
                if reverse:
                    pixel, curve, good = pixel.flip(0), curve.flip(0), good.flip(0)
                    descriptor = _reverse_descriptor(descriptor)
                if identity >= 0:
                    prior = next((track for track in old if track['id'] == identity), None)
                    velocity = torch.zeros(2, 3)
                    if prior is not None:
                        velocity = (descriptor['endpoints'] - prior['endpoints']) / (now - prior['time'])
                    history.append(dict(**descriptor, id=identity, time=now, velocity=velocity))
                metric_complete[env] &= bool(good.all())
                pixel_arc = _arc_pixels(pixel.cpu().numpy())
                metric_arc = curve.new_full((len(curve),), float('nan'))
                segments = torch.full((len(curve),), -1, device=device, dtype=torch.long)
                tangents = torch.full_like(curve, float('nan'))
                depth_flags = good.cpu().tolist()
                start = 0
                while start < len(curve):
                    if not depth_flags[start]:
                        start += 1
                        continue
                    end = start + 1
                    while end < len(curve) and depth_flags[end]:
                        end += 1
                    part = curve[start:end]
                    local_arc = torch.cat((part.new_zeros(1), torch.diff(part, dim=0).norm(dim=-1).cumsum(0)))
                    metric_arc[start:end] = local_arc
                    lengths[env] += local_arc[-1]
                    segments[start:end] = next_segment
                    next_segment += 1
                    if len(part) > 1:
                        deltas = torch.cat((part[1:2] - part[:1], part[2:] - part[:-2], part[-1:] - part[-2:-1]))
                        tangents[start:end] = _unit(deltas)
                    start = end
                complete = bool(good.all())
                sampling_arc_space[env].append('metric_3d' if complete else 'image_2d_partial_depth')
                sample_arc = metric_arc.cpu().numpy() if complete else pixel_arc
                indices = torch.as_tensor(_uniform_indices(sample_arc, budget), device=device) if budget else torch.empty(0, device=device, dtype=torch.long)
                count = len(indices)
                dest = slice(cursor, cursor + count)
                xyz[env, dest] = curve[indices]
                valid[env, dest] = good[indices]
                visibility[env, dest] = True
                component_id[env, dest] = component
                track_id[env, dest] = identity
                pixels[env, dest] = pixel[indices]
                arc_length[env, dest] = metric_arc[indices]
                arc_length_px[env, dest] = torch.as_tensor(pixel_arc, device=device, dtype=dtype)[indices]
                depth_segment_id[env, dest] = segments[indices]
                tangent[env, dest] = tangents[indices]
                if point_confidence is None:
                    confidence[env, dest] = good[indices].to(dtype)
                else:
                    chosen = pixel[indices]
                    confidence[env, dest] = point_confidence[env, chosen[:, 0], chosen[:, 1]] * good[indices]
                cursor += count
            self._history[env] = history
            temporal_valid[env] = all_matched
        return dict(points_world=xyz, point_valid=valid, xyz=xyz, valid=valid,
                    component_id=component_id, depth_segment_id=depth_segment_id,
                    arc_length=arc_length, arc_length_m=arc_length, arc_length_px=arc_length_px,
                    tangent=tangent, confidence=confidence, visibility=visibility,
                    track_id=track_id, point_pixels=pixels, topology_valid=topology_valid,
                    temporal_identity_valid=temporal_valid, timestamp_anomaly=anomaly,
                    visible_component_count=component_count, sampling_truncated=truncated,
                    all_visible_components_simple=topology_valid & (rejected_count == 0),
                    rejected_component_count=rejected_count, rejected_components=rejected_components,
                    sampling_arc_space=sampling_arc_space,
                    visible_arc_length_m=lengths, metric_arc_complete=metric_complete,
                    full_cable_observed=torch.zeros_like(temporal_valid), rejection_reason=reasons,
                    source='RGB_colour_visible_components_and_stereo_only',
                    correspondence_scope='conservative_visible_component_sensor_tracks_not_material_point_identity',
                    centerline_qualification=None)
