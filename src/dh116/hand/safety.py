"""Canonical24 transaction submission and actual finger/contact/support/clearance guards."""
import torch as lift_hold_windo__torch
import math as multi_monit__math
import torch as surfa_force_shari__torch
import math as surfa_succe__math
import math as tenso_contr__math

def hold_window_step(duration, anchor, latched, relative_samples, physical_ok, dt, *, slip_tolerance_m=0.0005):
    """Update a continuous stable-lift window from ALL raw substep positions.

    relative_samples: (substeps,environment,hand,rod-point,xyz). Caller checks
    actual height, effective opposed pads, native depth and invalid/reset state.
    Once holding begins its anchor stays fixed. A slip breaks the window and
    uses the last raw pose for a future window; no duration survives that slip.
    Settling during lifting cannot permanently veto later genuine stability.
    """
    start = lift_hold_windo__torch.where(latched[:, None, None, None], anchor, relative_samples[0])
    drift = (relative_samples - start[None]).norm(dim=-1).amax((0, 2, 3))
    restart = physical_ok & (drift > slip_tolerance_m)
    good = physical_ok & ~restart
    duration = lift_hold_windo__torch.where(good, lift_hold_windo__torch.where(latched, duration, 0.0) + dt, 0.0)
    next_anchor = lift_hold_windo__torch.where((~physical_ok | restart)[:, None, None, None], relative_samples[-1], start)
    return (duration, next_anchor, physical_ok.clone(), dict(good=good, restart=restart, drift_m=drift))
from numbers import Integral
CONTRACT = dict(version='opposed_surface_sharing_v1', force_target_range_n=None, total_force_source='current_measured_effective_pad_load;policy_explores_joint_targets', allocation='per_hand_two_opposed_surfaces_half_each;each_surface_equal_among_bearing_digits', digit_names_fixed=False, contact_threshold_n=0.05, minimum_opposition_cosine=-0.25, same_face_max_min_ratio=1.5, opposed_face_max_min_ratio=1.5, no_contacts_are_not_balanced=True, measurement_mutation=False)
import torch as surfa_succe__torch
import torch as tenso_contr__torch
from numbers import Real

def surface_targets(loads, normals, *, valid=None, previous_axis=None):
    """Allocate [...,5] real pad forces via [...,5,3] measured surface normals.

    Face labels are interchangeable. Their axis follows the most opposed
    bearing pair, with sign continuity when the previous axis is available.
    Every bearing contact must be assigned; ambiguous or single-face rows are
    returned unchanged and never count as a balanced grasp. Subthreshold loads
    stay unchanged so redistribution never creates a fixed pressure request.
    """
    if not isinstance(loads, surfa_force_shari__torch.Tensor) or not loads.is_floating_point() or loads.ndim < 1 or (loads.shape[-1] != 5) or (normals.shape != (*loads.shape, 3)) or (normals.dtype != loads.dtype) or (normals.device != loads.device) or (not surfa_force_shari__torch.isfinite(loads).all()) or (loads < 0).any() or (not surfa_force_shari__torch.isfinite(normals).all()):
        raise ValueError('finite matching nonnegative five-digit loads and normals required')
    if valid is None:
        valid = surfa_force_shari__torch.ones_like(loads, dtype=surfa_force_shari__torch.bool)
    if valid.shape != loads.shape or valid.dtype != surfa_force_shari__torch.bool or valid.device != loads.device:
        raise ValueError('matching boolean effective-contact validity required')
    length = normals.norm(dim=-1)
    unit = normals / length.clamp_min(1e-12)[..., None]
    bearing = (loads > 0.05) & valid
    usable = bearing & (length > 1e-12)
    dot = unit @ unit.transpose(-1, -2)
    pairs = usable[..., :, None] & usable[..., None, :]
    score = dot.masked_fill(~pairs, 2.0)
    selected = score.flatten(-2).argmin(-1)
    a = selected // 5
    b = selected % 5
    take = lambda ids: unit.gather(-2, ids[..., None, None].expand(*ids.shape, 1, 3)).squeeze(-2)
    axis = take(a) - take(b)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    major = axis.abs().argmax(-1, keepdim=True)
    sign = surfa_force_shari__torch.where(axis.gather(-1, major).squeeze(-1) < 0, -1.0, 1.0)
    if previous_axis is not None:
        if previous_axis.shape != axis.shape or previous_axis.dtype != axis.dtype or previous_axis.device != axis.device or (not surfa_force_shari__torch.isfinite(previous_axis).all()):
            raise ValueError('matching finite previous surface axis required')
        known = previous_axis.norm(dim=-1) > 1e-12
        sign = surfa_force_shari__torch.where(known, surfa_force_shari__torch.where((axis * previous_axis).sum(-1) < 0, -1.0, 1.0), sign)
    axis = axis * sign[..., None]
    projected = (unit * axis[..., None, :]).sum(-1)
    assigned = usable & (projected.abs() > 0.2)
    groups = surfa_force_shari__torch.where(assigned, (projected < 0).long(), -1)
    masks = surfa_force_shari__torch.stack((groups == 0, groups == 1), -2)
    counts = masks.sum(-1)
    face_loads = (masks * loads[..., None, :]).sum(-1)
    opposed = score.flatten(-2).amin(-1) <= -0.25
    paired = opposed & (counts > 0).all(-1) & (~bearing | assigned).all(-1)
    total = face_loads.sum(-1)
    face_per_digit = total[..., None] / 2 / counts.clamp_min(1)
    proposed = (masks * face_per_digit[..., :, None]).sum(-2)
    desired = surfa_force_shari__torch.where(paired[..., None] & assigned, proposed, loads)
    face_error = (face_loads[..., 0] - face_loads[..., 1]).abs() / total.clamp_min(1e-12)
    within = ((desired - loads).abs() * assigned).sum(-1) / total.clamp_min(1e-12)
    small = surfa_force_shari__torch.where(masks, loads[..., None, :], float('inf')).amin(-1)
    large = surfa_force_shari__torch.where(masks, loads[..., None, :], 0.0).amax(-1)
    face_ok = face_loads.amax(-1) <= 1.5 * face_loads.amin(-1)
    within_ok = (large <= 1.5 * small).all(-1)
    balanced = paired & face_ok & within_ok
    return dict(desired=desired, groups=groups, axis=surfa_force_shari__torch.where(paired[..., None], axis, 0.0), paired=paired, balanced=balanced, face_loads=face_loads, face_counts=counts, face_imbalance=surfa_force_shari__torch.where(paired, face_error, 0.0), within_imbalance=surfa_force_shari__torch.where(paired, within, 0.0))
surfa_succe__hold_window_step = hold_window_step
from dh116.physics.solver import solver_clock_matches as solver_clock_matches
import torch as multi_monit__torch
surfa_succe__surface_targets = surface_targets
from dh116.env.contracts import linea_coupl__ACTIVE as ACTIVE

def _tensor(name, value, shape, *, device=None, dtype=None):
    if not isinstance(value, multi_monit__torch.Tensor):
        raise TypeError(f'{name} must be a Torch tensor')
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}, got {tuple(value.shape)}')
    if dtype is None:
        if not value.is_floating_point():
            raise TypeError(f'{name} must have floating dtype')
    elif value.dtype != dtype:
        raise TypeError(f'{name} must have dtype {dtype}, got {value.dtype}')
    if device is not None and value.device != device:
        raise ValueError(f'{name} must be on {device}, got {value.device}')
from dh116.env.contracts import MIMIC

def _unit_quaternion(q):
    scale = q.abs().amax(dim=-1, keepdim=True)
    if (scale == 0).any():
        raise ValueError('pose quaternion must be nonzero')
    q = q / scale
    return q / multi_monit__torch.linalg.vector_norm(q, dim=-1, keepdim=True)
from dh116.env.contracts import active_limits

def _rotate(q, points):
    xyz = q[..., 1:].expand_as(points)
    twice_cross = 2 * multi_monit__torch.linalg.cross(xyz, points, dim=-1)
    return points + q[..., :1] * twice_cross + multi_monit__torch.linalg.cross(xyz, twice_cross, dim=-1)
from dh116.env.contracts import surfa_contr__ACTION_CONTRACT as ACTION_CONTRACT

def rod_points_in_hands(object_pose, hand_pose):
    """Return rod Y endpoints (+/-0.2 m) and X radius point (8 mm) in each hand.

    Inputs [N,7] and [N,2,7] share floating dtype/device; output is [N,2,3,3].
    hand_pose must be the actual palm/body_link_pose_w reference, not its COM
    pose. object_pose likewise refers to the rod's geometric body frame. This
    function applies precisely those world-to-link transforms, with no offsets.
    Nonunit, nonzero quaternions are normalized. Malformed/nonfinite poses raise
    before computing geometry; per-environment measurement isolation is handled
    by GraspStabilityMonitor.update.
    """
    if not isinstance(object_pose, multi_monit__torch.Tensor) or object_pose.ndim != 2:
        raise ValueError('object_pose must have shape [N,7]')
    n = object_pose.shape[0]
    _tensor('object_pose', object_pose, (n, 7))
    _tensor('hand_pose', hand_pose, (n, 2, 7), device=object_pose.device, dtype=object_pose.dtype)
    if not multi_monit__torch.isfinite(object_pose).all() or not multi_monit__torch.isfinite(hand_pose).all():
        raise ValueError('poses must be finite')
    local = object_pose.new_tensor([[0, -0.2, 0], [0, 0.2, 0], [0.008, 0, 0]])
    local = local.expand(n, 3, 3)
    object_q = _unit_quaternion(object_pose[:, 3:])[:, None]
    world = _rotate(object_q, local) + object_pose[:, None, :3]
    hand_q = _unit_quaternion(hand_pose[..., 3:]).clone()
    hand_q[..., 1:] *= -1
    return _rotate(hand_q[:, :, None], world[:, None] - hand_pose[:, :, None, :3])
from dh116.env.contracts import validate_actions

class GraspStabilityMonitor:
    """Rolling raw-physics-sample monitor, with independent history for each row.

    update returns a dict (also exposed as attributes): ready[N], loaded[N,2,5],
    force_cv[N,2,5], relative_drift_m[N], slip_rate_m_s[N]. A loaded digit exceeds
    0.05 N on *every* sample. CV is population std / mean on raw samples.
    Each hand needs >=2 loaded digits, CV <=0.25, finger means within 0.5..1.5
    of that hand's loaded-finger mean, and an opposing loaded pair on every
    sample (normalized normal dot <=-0.25). Total hand mean loads, including
    palm support when supplied, have ratio <=1.5. Any digit whose window mean
    exceeds 0.05 N must also be continuously
    loaded and stable: intermittent extra support cannot be excluded to qualify
    using only two quieter fingers. Unloaded zero normals are allowed.

    Optional hand_force_range=(min_n, max_n) additionally requires each hand's
    total normal force to stay within those inclusive limits on every raw sample
    of the full window. update's keyword hand_total_load_n[N,2] includes palm
    support; it defaults to loads.sum(-1). None preserves the old ready rules.
    Diagnostics hand_force_in_range[N], hand_force_min_n[N,2] and
    hand_force_max_n[N,2] are zero/False until the row has a full window.
    With no configured range, hand_force_in_range is True for full valid rows.

    Optional max_digit_ratio (finite, >=1) additionally limits max/min force
    across all loaded digits of each hand on every raw sample. It applies to
    any finger combination, alongside the original mean-balance and CV rules.
    None keeps the old rules. Diagnostic digit_ratio_ok[N] is False until full;
    without a ratio limit it is True for full valid rows.

    Drift is a conservative spatial envelope: maximum bounding-box diagonal
    over the six rod points across the window. This bounds every pairwise point
    displacement, including excursions that return to the start. Slip rate is
    the maximum adjacent-sample Euclidean speed, using each interval's dt.
    Limits are 0.5 mm and 1 mm/s. Incomplete rows return zero diagnostics and
    loaded=False, ready=False; only a full consecutive valid window can qualify.

    Shape/dtype/device errors and invalid scalar dt raise before state changes.
    Nonfinite measurements, negative loads, valid=False or reset=True clear
    only that row's history and anchor; the rejected/reset sample is discarded.
    reset(ids) immediately clears selected histories, anchors and diagnostics,
    including before the first update, without changing the shared ring cursor.
    Anchors change only via latch or row clearing. Use anchor_error separately
    during lift to detect cumulative drift beyond the rolling window.
    """

    def __init__(self, n, device, window_steps=144, hand_force_range=None, max_digit_ratio=None):
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError('n must be a positive integer')
        if isinstance(window_steps, bool) or not isinstance(window_steps, Integral) or window_steps < 2:
            raise ValueError('window_steps must be an integer >=2')
        if hand_force_range is not None:
            if not isinstance(hand_force_range, (tuple, list)) or len(hand_force_range) != 2:
                raise ValueError('hand_force_range must be a (min_n, max_n) pair or None')
            if any((isinstance(bound, bool) or not isinstance(bound, Real) or (not multi_monit__math.isfinite(bound)) for bound in hand_force_range)):
                raise ValueError('hand_force_range bounds must be finite real numbers')
            (lower, upper) = hand_force_range
            if not 0 <= lower < upper:
                raise ValueError('hand_force_range must satisfy 0 <= min_n < max_n')
            hand_force_range = (float(lower), float(upper))
        if max_digit_ratio is not None:
            if isinstance(max_digit_ratio, bool) or not isinstance(max_digit_ratio, Real) or (not multi_monit__math.isfinite(max_digit_ratio)) or (max_digit_ratio < 1):
                raise ValueError('max_digit_ratio must be a finite real number >=1 or None')
            max_digit_ratio = float(max_digit_ratio)
        self.n = int(n)
        self.window_steps = int(window_steps)
        self.hand_force_range = hand_force_range
        self.max_digit_ratio = max_digit_ratio
        self.device = multi_monit__torch.empty(0, device=device).device
        self._dtype = None
        self._cursor = 0
        self._count = multi_monit__torch.zeros(self.n, dtype=multi_monit__torch.long, device=self.device)
        self._anchored = multi_monit__torch.zeros(self.n, dtype=multi_monit__torch.bool, device=self.device)
        self.ready = multi_monit__torch.zeros(self.n, dtype=multi_monit__torch.bool, device=self.device)
        self.loaded = multi_monit__torch.zeros((self.n, 2, 5), dtype=multi_monit__torch.bool, device=self.device)
        self.force_cv = multi_monit__torch.zeros((self.n, 2, 5), device=self.device)
        self.relative_drift_m = multi_monit__torch.zeros(self.n, device=self.device)
        self.slip_rate_m_s = multi_monit__torch.zeros(self.n, device=self.device)
        self.hand_force_in_range = multi_monit__torch.zeros(self.n, dtype=multi_monit__torch.bool, device=self.device)
        self.hand_force_min_n = multi_monit__torch.zeros((self.n, 2), device=self.device)
        self.hand_force_max_n = multi_monit__torch.zeros((self.n, 2), device=self.device)
        self.digit_ratio_ok = multi_monit__torch.zeros(self.n, dtype=multi_monit__torch.bool, device=self.device)

    def _allocate(self, dtype):
        self._dtype = dtype
        options = dict(device=self.device, dtype=dtype)
        self._loads = multi_monit__torch.zeros((self.window_steps, self.n, 2, 5), **options)
        self._hand_total_loads = multi_monit__torch.zeros((self.window_steps, self.n, 2), **options)
        self._points = multi_monit__torch.zeros((self.window_steps, self.n, 2, 3, 3), **options)
        self._opposition = multi_monit__torch.zeros((self.window_steps, self.n, 2, 5, 5), device=self.device, dtype=multi_monit__torch.bool)
        self._dt = multi_monit__torch.ones(self.window_steps, **options)
        self._anchor = multi_monit__torch.zeros((self.n, 2, 3, 3), **options)

    def _check_points(self, points):
        _tensor('relative_points', points, (self.n, 2, 3, 3), device=self.device, dtype=self._dtype)

    def _selected_rows(self, ids):
        ids = multi_monit__torch.as_tensor(ids, device=self.device)
        if ids.dtype == multi_monit__torch.bool:
            _tensor('ids', ids, (self.n,), device=self.device, dtype=multi_monit__torch.bool)
            return ids
        if ids.ndim != 1 or (ids.numel() and ids.dtype not in (multi_monit__torch.int32, multi_monit__torch.int64)):
            raise TypeError('ids must be a 1D integer index tensor or a boolean [N] mask')
        if ((ids < 0) | (ids >= self.n)).any():
            raise ValueError('ids are out of range')
        selected = multi_monit__torch.zeros(self.n, device=self.device, dtype=multi_monit__torch.bool)
        selected[ids.long()] = True
        return selected

    def _clear(self, rows):
        self._count.masked_fill_(rows, 0)
        self._anchored.masked_fill_(rows, False)
        self.ready.masked_fill_(rows, False)
        self.loaded.masked_fill_(rows[:, None, None], False)
        self.force_cv.masked_fill_(rows[:, None, None], 0)
        self.relative_drift_m.masked_fill_(rows, 0)
        self.slip_rate_m_s.masked_fill_(rows, 0)
        self.hand_force_in_range.masked_fill_(rows, False)
        self.hand_force_min_n.masked_fill_(rows[:, None], 0)
        self.hand_force_max_n.masked_fill_(rows[:, None], 0)
        self.digit_ratio_ok.masked_fill_(rows, False)
        if self._dtype is None:
            return
        self._anchor.masked_fill_(rows[:, None, None, None], 0)
        self._loads.masked_fill_(rows[None, :, None, None], 0)
        self._hand_total_loads.masked_fill_(rows[None, :, None], 0)
        self._points.masked_fill_(rows[None, :, None, None, None], 0)
        self._opposition.masked_fill_(rows[None, :, None, None, None], False)

    @multi_monit__torch.no_grad()
    def reset(self, ids):
        """Clear selected rows after physical reset; accept indices or [N] bool mask.

        Clears history, anchors, ready/loaded and numeric diagnostics immediately.
        Safe before allocation or with empty ids. Other rows, the shared ring
        cursor, and interval history remain unchanged. A full new window is
        needed to become ready again; anchors require an explicit new latch.
        """
        self._clear(self._selected_rows(ids))

    @multi_monit__torch.no_grad()
    def update(self, loads, relative_points, normals, dt, valid, reset, *, hand_total_load_n=None):
        """Consume one step; dt is the positive scalar interval in seconds.

        hand_total_load_n is optional [N,2] actual total normal force, including
        palm support, with the same floating dtype/device as loads. If omitted,
        sum the five digit loads. These totals drive both the force range and
        cross-hand window-mean balance. Nonfinite/negative totals invalidate only their
        own row; malformed tensor structure raises before changing any history.
        """
        _tensor('loads', loads, (self.n, 2, 5), device=self.device, dtype=self._dtype)
        _tensor('relative_points', relative_points, (self.n, 2, 3, 3), device=self.device, dtype=loads.dtype)
        _tensor('normals', normals, (self.n, 2, 5, 3), device=self.device, dtype=loads.dtype)
        _tensor('valid', valid, (self.n,), device=self.device, dtype=multi_monit__torch.bool)
        _tensor('reset', reset, (self.n,), device=self.device, dtype=multi_monit__torch.bool)
        if isinstance(dt, bool) or not isinstance(dt, Real) or (not multi_monit__math.isfinite(dt)) or (dt <= 0):
            raise ValueError('dt must be a finite positive scalar')
        if hand_total_load_n is None:
            hand_total_load_n = loads.sum(dim=-1)
        else:
            _tensor('hand_total_load_n', hand_total_load_n, (self.n, 2), device=self.device, dtype=loads.dtype)
        if self._dtype is None:
            self._allocate(loads.dtype)
        good = valid & ~reset
        for measurement in (loads, relative_points, normals, hand_total_load_n):
            good = good & multi_monit__torch.isfinite(measurement).flatten(1).all(dim=1)
        good = good & (loads >= 0).flatten(1).all(dim=1)
        good = good & (hand_total_load_n >= 0).all(dim=1)
        self._clear(~good)
        clean_loads = multi_monit__torch.where(good[:, None, None], loads, 0)
        clean_points = multi_monit__torch.where(good[:, None, None, None], relative_points, 0)
        clean_normals = multi_monit__torch.where(good[:, None, None, None], normals, 0)
        scale = clean_normals.abs().amax(dim=-1, keepdim=True)
        unit = clean_normals / scale.clamp_min(multi_monit__torch.finfo(loads.dtype).tiny)
        length = multi_monit__torch.linalg.vector_norm(unit, dim=-1, keepdim=True)
        unit = unit / length.clamp_min(multi_monit__torch.finfo(loads.dtype).tiny)
        dot = unit @ unit.transpose(-1, -2)
        nonzero = scale.squeeze(-1) > 0
        opposed = (dot <= -0.25) & nonzero[..., :, None] & nonzero[..., None, :]
        self._loads[self._cursor].copy_(clean_loads)
        self._hand_total_loads[self._cursor].copy_(multi_monit__torch.where(good[:, None], hand_total_load_n, 0))
        self._points[self._cursor].copy_(clean_points)
        self._opposition[self._cursor].copy_(opposed)
        self._dt[self._cursor] = dt
        self._cursor = (self._cursor + 1) % self.window_steps
        self._count.add_(good).clamp_(max=self.window_steps)
        full = self._count == self.window_steps
        hand_min = self._hand_total_loads.amin(dim=0)
        hand_max = self._hand_total_loads.amax(dim=0)
        self.hand_force_min_n = multi_monit__torch.where(full[:, None], hand_min, 0)
        self.hand_force_max_n = multi_monit__torch.where(full[:, None], hand_max, 0)
        self.hand_force_in_range = full.clone()
        if self.hand_force_range is not None:
            (lower, upper) = self.hand_force_range
            self.hand_force_in_range &= (hand_min >= lower).all(dim=-1) & (hand_max <= upper).all(dim=-1)
        mean = self._loads.mean(dim=0)
        std = self._loads.std(dim=0, unbiased=False)
        cv = std / mean.clamp_min(multi_monit__torch.finfo(loads.dtype).tiny)
        self.force_cv = multi_monit__torch.where(full[:, None, None], cv, 0)
        self.loaded = (self._loads.amin(dim=0) > 0.05) & full[:, None, None]
        count = self.loaded.sum(dim=-1)
        self.digit_ratio_ok = full.clone()
        if self.max_digit_ratio is not None:
            digit_min = multi_monit__torch.where(self.loaded[None], self._loads, float('inf')).amin(dim=-1)
            digit_max = multi_monit__torch.where(self.loaded[None], self._loads, 0).amax(dim=-1)
            self.digit_ratio_ok &= (count > 0).all(dim=-1) & (digit_max <= self.max_digit_ratio * digit_min).all(dim=(0, 2))
        finger_mean = (mean * self.loaded).sum(dim=-1) / count.clamp_min(1)
        balanced = (mean >= 0.5 * finger_mean[..., None]) & (mean <= 1.5 * finger_mean[..., None])
        participating = mean > 0.05
        stable_digits = (~participating | self.loaded & balanced & (cv <= 0.25)).all(dim=(1, 2))
        hand_load = self._hand_total_loads.mean(dim=0)
        hand_balance = (hand_load.amin(dim=-1) > 0) & (hand_load.amax(dim=-1) <= 1.5 * hand_load.amin(dim=-1))
        pairs = self.loaded[..., :, None] & self.loaded[..., None, :]
        opposition = (self._opposition & pairs[None]).flatten(-2).any(dim=-1).all(dim=(0, 2))
        span = self._points.amax(dim=0) - self._points.amin(dim=0)
        drift = multi_monit__torch.linalg.vector_norm(span, dim=-1).amax(dim=(1, 2))
        order = (multi_monit__torch.arange(self.window_steps, device=self.device) + self._cursor) % self.window_steps
        ordered = self._points[order]
        speeds = multi_monit__torch.linalg.vector_norm(ordered[1:] - ordered[:-1], dim=-1)
        speeds = speeds / self._dt[order][1:, None, None, None]
        rate = speeds.amax(dim=(0, 2, 3))
        self.relative_drift_m = multi_monit__torch.where(full, drift, 0)
        self.slip_rate_m_s = multi_monit__torch.where(full, rate, 0)
        self.ready = full & self.hand_force_in_range & self.digit_ratio_ok & (count >= 2).all(dim=-1) & stable_digits & hand_balance & opposition & (drift <= 0.0005) & (rate <= 0.001) & multi_monit__torch.isfinite(cv).all(dim=(1, 2))
        return {name: getattr(self, name) for name in ('ready', 'loaded', 'force_cv', 'relative_drift_m', 'slip_rate_m_s', 'hand_force_in_range', 'hand_force_min_n', 'hand_force_max_n', 'digit_ratio_ok')}

    @multi_monit__torch.no_grad()
    def latch(self, ids, relative_points):
        """Copy selected rows from full [N,2,3,3] points as fixed lift references.

        ids may be integer indices or a boolean [N] mask. Invalid selected point
        rows clear their own history/reference; other rows are unaffected.
        Explicitly calling latch again replaces only the selected references.
        """
        self._check_points(relative_points)
        selected = self._selected_rows(ids)
        if self._dtype is None:
            self._allocate(relative_points.dtype)
        finite = multi_monit__torch.isfinite(relative_points).flatten(1).all(dim=1)
        self._clear(selected & ~finite)
        selected = selected & finite
        self._anchor[selected] = relative_points[selected]
        self._anchored[selected] = True

    @multi_monit__torch.no_grad()
    def anchor_error(self, relative_points):
        """Maximum Euclidean point error [N]; +inf for unlatched/nonfinite rows."""
        self._check_points(relative_points)
        if self._dtype is None:
            return relative_points.new_full((self.n,), float('inf'))
        finite = multi_monit__torch.isfinite(relative_points).flatten(1).all(dim=1)
        error = multi_monit__torch.linalg.vector_norm(relative_points - self._anchor, dim=-1).amax(dim=(1, 2))
        return multi_monit__torch.where(self._anchored & finite, error, float('inf'))
from dh116.hand.kinematics import finger_pre_guard
surfa_succe__rod_points_in_hands = rod_points_in_hands
from dh116.hand.kinematics import surface_final_guard

class SurfaceLiftSuccess:
    """Accumulate raw xyz+wxyz poses and native contact measurements.

    update takes object_pose[N,7], wrist_pose[N,2,7], loads[N,2,5],
    vectors[N,2,5,3], depth[N,2,5], ordinary[N], bottom[N],
    object_velocity[N,6], valid[N bool], and scalar dt in seconds.
    dt-weighted means have no EMA; uniform 480 Hz input gives mean8 at 60 Hz.
    dt may span periods; its constant measured sample is split at each boundary.
    duration changes only at period flush; event pulses on crossing 5 seconds.
    paired/window_good/drift_m describe the last completed period (or reset).
    reset(ids) discards selected rows of a pending period; other rows continue.
    """

    def __init__(self, n, device, reference_bottom_m, dtype=surfa_succe__torch.float64, control_period_s=1 / 60, minimum_lift_m=0.03, reward_diagnostics=False):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError('positive environment count required')
        if not surfa_succe__math.isfinite(control_period_s) or control_period_s <= 0:
            raise ValueError('finite positive control period required')
        if not surfa_succe__math.isfinite(minimum_lift_m) or minimum_lift_m <= 0:
            raise ValueError('finite positive minimum lift required')
        (self.n, self.period) = (n, float(control_period_s))
        self.minimum_lift_m = float(minimum_lift_m)
        self._duration = surfa_succe__torch.zeros(n, device=device, dtype=dtype)
        ref = surfa_succe__torch.as_tensor(reference_bottom_m, device=device, dtype=dtype)
        if ref.shape not in ((), (n,)) or not surfa_succe__torch.isfinite(ref).all():
            raise ValueError('finite scalar or [N] reference bottom required')
        self.reference_bottom_m = ref.expand(n).clone()
        self._anchor = self._duration.new_zeros(n, 2, 3, 3)
        self._latched = surfa_succe__torch.zeros(n, device=device, dtype=surfa_succe__torch.bool)
        self._paired = self._latched[:, None].expand(n, 2).clone()
        self._window_good = self._latched.clone()
        self._physical_window_valid = self._latched.clone()
        self._drift = self._duration.clone()
        (self._elapsed, self._samples) = (0.0, [])
        self._ok = surfa_succe__torch.ones_like(self._latched)
        self.reward_diagnostics = reward_diagnostics
        if reward_diagnostics:
            self.window_completed = False
            self.window_native_valid = self._latched.clone()
            self.slip_observable = self._latched.clone()
            self.reward_drift_m = self._drift.clone()
            self._native_ok = self._ok.clone()
            self._slip_contact = self._ok.clone()
            self._slip_anchor = self._anchor.clone()
            self._slip_latched = self._latched.clone()

    @property
    def duration(self):
        return self._duration

    @property
    def drift_m(self):
        return self._drift

    @property
    def window_good(self):
        return self._window_good

    @property
    def measured_drift_m(self):
        return surfa_succe__torch.where(self._physical_window_valid, self._drift, 0.0)

    @surfa_succe__torch.no_grad()
    def reset(self, ids):
        """Accept integer indices or a boolean [N] mask; clear only those rows."""
        self._duration[ids] = 0
        self._anchor[ids] = 0
        self._latched[ids] = False
        self._paired[ids] = False
        self._window_good[ids] = False
        self._physical_window_valid[ids] = False
        self._drift[ids] = 0
        if self._elapsed > 0:
            self._ok[ids] = False
        if self.reward_diagnostics:
            self.window_native_valid[ids] = False
            self.slip_observable[ids] = False
            self.reward_drift_m[ids] = 0.0
            self._slip_anchor[ids] = 0.0
            self._slip_latched[ids] = False
            if self._elapsed > 0:
                self._native_ok[ids] = False

    @surfa_succe__torch.no_grad()
    def update(self, object_pose, wrist_pose, loads, vectors, depth, ordinary, bottom, object_velocity, valid, dt):
        if not surfa_succe__math.isfinite(dt) or dt <= 0:
            raise ValueError('finite positive dt required')
        values = (object_pose, wrist_pose, loads, vectors, depth, ordinary, bottom, object_velocity, valid)
        tails = ((7,), (2, 7), (2, 5), (2, 5, 3), (2, 5), (), (), (6,), ())
        for (value, tail) in zip(values, tails):
            expected_dtype = surfa_succe__torch.bool if value is valid else self._duration.dtype
            if value.shape != (self.n, *tail) or value.dtype != expected_dtype or value.device != self._duration.device:
                raise ValueError('matching measurement shapes, dtype and device required')
        good = valid.clone()
        for value in values[:-1]:
            good &= surfa_succe__torch.isfinite(value).reshape(self.n, -1).all(-1)
        good &= object_pose[:, 3:].abs().amax(-1) > 0
        good &= (wrist_pose[..., 3:].abs().amax(-1) > 0).all(-1)
        good &= (loads >= 0).all((1, 2)) & (ordinary >= 0) & (depth >= 0).all((1, 2))
        physical = good & (bottom >= self.reference_bottom_m + self.minimum_lift_m) & (object_velocity[:, :3].norm(dim=-1) < 0.03) & (depth.amax((1, 2)) <= 0.0008)
        clean_object = surfa_succe__torch.where(good[:, None], object_pose, 0.0)
        clean_wrist = surfa_succe__torch.where(good[:, None, None], wrist_pose, 0.0)
        clean_object[~good, 3] = 1
        clean_wrist[~good, :, 3] = 1
        points = surfa_succe__rod_points_in_hands(clean_object, clean_wrist).clone()
        raw = (loads.clone(), vectors.clone(), ordinary.clone(), points)
        (event, updated) = (surfa_succe__torch.zeros_like(valid), False)
        if self.reward_diagnostics:
            self.window_completed = False
            if dt > self.period + 1e-12:
                raise ValueError('reward diagnostics require at most one control period per sample')
        remaining = float(dt)
        while remaining > 1e-12:
            weight = min(remaining, self.period - self._elapsed)
            self._samples.append((weight, raw))
            self._ok &= physical
            if self.reward_diagnostics:
                self._native_ok &= good
                self._slip_contact &= bottom >= self.reference_bottom_m + self.minimum_lift_m
                self._slip_contact &= (loads.sum(-1) > 0.0).all(-1)
            self._elapsed += weight
            remaining -= weight
            if self._elapsed < self.period - 1e-12:
                continue
            means = []
            for i in range(3):
                mean = sum((weight * sample[i] for (weight, sample) in self._samples)) / self._elapsed
                valid_mean = self._native_ok if self.reward_diagnostics else self._ok
                mask = valid_mean.reshape(self.n, *(1,) * (mean.ndim - 1))
                means.append(surfa_succe__torch.where(mask, mean, 0.0))
            (mean_load, mean_vectors, mean_ordinary) = means
            paired = surfa_succe__surface_targets(mean_load, mean_vectors)['paired']
            self._paired = paired & self._ok[:, None]
            normal = mean_load.sum((1, 2))
            physical_ok = self._ok & self._paired.all(-1) & (mean_ordinary / (mean_ordinary + normal).clamp_min(1e-12) <= 0.05 + 1e-12)
            self._physical_window_valid = physical_ok.clone()
            samples = surfa_succe__torch.stack([sample[3] for (_, sample) in self._samples])
            if self.reward_diagnostics:
                self.window_completed = True
                self.window_native_valid.copy_(self._native_ok)
                contact = self._native_ok & self._slip_contact & paired.all(-1)
                self.slip_observable.copy_(contact & self._slip_latched)
                drift = (samples - self._slip_anchor[None]).norm(dim=-1).amax((0, 2, 3))
                self.reward_drift_m.copy_(surfa_succe__torch.where(self.slip_observable, drift, 0.0))
                reanchor = (~self.slip_observable | (drift > 0.0005))[:, None, None, None]
                self._slip_anchor = surfa_succe__torch.where(reanchor, samples[-1], self._slip_anchor)
                self._slip_latched.copy_(contact)
                self._native_ok.fill_(True)
                self._slip_contact.fill_(True)
            samples = surfa_succe__torch.where(self._ok[None, :, None, None, None], samples, 0.0)
            previous = self._duration
            (self._duration, self._anchor, self._latched, diag) = surfa_succe__hold_window_step(previous, self._anchor, self._latched, samples, physical_ok, self._elapsed)
            (self._window_good, self._drift) = (diag['good'], diag['drift_m'])
            event |= (previous < 5.0 - 1e-12) & (self._duration >= 5.0 - 1e-12)
            (self._elapsed, self._samples) = (0.0, [])
            self._ok.fill_(True)
            updated = True
        return dict(duration=self._duration.clone(), event=event, updated=updated, paired=self._paired.clone(), window_good=self._window_good.clone(), drift_m=self._drift.clone())
from dh116.hand.kinematics import wrist_requested_targets
from dh116.hand.kinematics import project_tensor_policy_wrist_targets
from dh116.hand.kinematics import local_translation
from dh116.hand.kinematics import JOINT_NAMES
from dh116.hand.kinematics import ANGULAR_JOINT_NAMES
from dh116.hand.kinematics import palm_mount__CONTRACT as MOUNT
DT = 1 / 480

class TensorControl:

    def __init__(self, batch, root_seed_w, bindings, mesh_names, *, isolate_constraints=False):
        self.batch = batch
        self.roots = root_seed_w.detach().double().clone()
        self.device = self.roots.device
        self.n = len(self.roots)
        self.palms = tenso_contr__torch.zeros(self.n, 2, 3, device=self.device, dtype=tenso_contr__torch.float64)
        self.angles = tenso_contr__torch.zeros_like(self.palms)
        self.active = []
        self.write_ids = []
        self.mimic = []
        self.slots = []
        self.contact_bodies = []
        for side in ('left', 'right'):
            names = batch.sides[side].joint_names
            self.active.append([names.index(k) for k in ACTIVE])
            self.write_ids.append(self.active[-1] + [names.index(k) for k in JOINT_NAMES + ANGULAR_JOINT_NAMES])
            self.mimic.append([(names.index(f), names.index(m), ratio, offset) for (f, m, ratio, offset) in MIMIC])
            keys = [k for (k, v) in bindings.items() if v['side'] == side]
            self.slots.append([mesh_names.index(k) for k in keys])
            self.contact_bodies.append([bindings[k]['body_index'] for k in keys])
        (self.low, self.high) = (self.roots.new_tensor(v).repeat(2) for v in active_limits())
        self.rates = dict(joint_rate_rad_s=ACTION_CONTRACT['joint_rate_rad_s'], wrist_rate_m_s=ACTION_CONTRACT['translation_rate_m_s'], rotation_rate_rad_s=ACTION_CONTRACT['rotation_rate_rad_s'])
        self.pending = None
        self.epoch = 0
        self.mesh_count = len(mesh_names)
        self.isolate_constraints = isolate_constraints
        self.fixed_fingers = ()
        self._failed = False

    def invalidate(self, env_ids=None):
        if env_ids is None:
            self.palms.zero_()
            self.angles.zero_()
        else:
            self.palms[env_ids] = 0
            self.angles[env_ids] = 0
        self.pending = None
        self.epoch += 1

    def joint_failures(self, state):
        q = state['q'].double()
        active = tenso_contr__torch.cat([q[:, s, self.active[s]] for s in (0, 1)], 1)
        mimic = tenso_contr__torch.stack([q[:, s, f] - ratio * q[:, s, m] - offset for s in (0, 1) for (f, m, ratio, offset) in self.mimic[s]], 1)
        if not bool(tenso_contr__torch.isfinite(q).all() & tenso_contr__torch.isfinite(state['qd']).all()):
            raise ValueError('nonfinite native q or qd')
        return ((active < self.low - 0.02) | (active > self.high + 0.02)).any(1) | (mimic.abs() > 0.02).any(1)

    def prepare(self, actions, state, clearance_m, base_poses_wxyz, *, stamp, halted=None):
        if self._failed:
            raise ValueError('failed control owner requires a new native scope')
        if self.pending is not None:
            raise ValueError('previous SDK transaction unfinished')
        if not isinstance(actions, tenso_contr__torch.Tensor) or actions.shape != (self.n, 24) or actions.device != self.device or (not actions.is_floating_point()):
            raise ValueError('same-device full normalized floating batch24 required')
        finite_inputs = [actions]
        expected_joints = len(self.batch.sides['left'].joint_names)
        for key in ('q', 'qd', 'backend_targets'):
            value = state[key]
            if not isinstance(value, tenso_contr__torch.Tensor) or value.shape != (self.n, 2, expected_joints) or value.device != self.device or (not value.is_floating_point()):
                self._failed = True
                raise ValueError('finite complete original q/qd/backend targets before command projection required')
            finite_inputs.append(value)
        for (value, shape) in ((state['root_pose_wxyz'], (self.n, 2, 7)), (base_poses_wxyz, (self.n, 2, 7)), (clearance_m, (self.n, 2))):
            if not isinstance(value, tenso_contr__torch.Tensor) or value.shape != shape or value.device != self.device or (not value.is_floating_point()):
                self._failed = True
                raise ValueError('finite complete actual pre-action roots/base/clearance required')
            finite_inputs.append(value)
        finite = tenso_contr__torch.isfinite(tenso_contr__torch.cat([v.reshape(-1) for v in finite_inputs])).all()
        if not bool(finite & ~(actions.abs() > 1).any()):
            validate_actions(actions)
            self._failed = True
            raise ValueError('finite complete actual pre-action state/base/clearance required')
        q = state['q'].double()
        command = tenso_contr__torch.cat([state['backend_targets'][:, s, self.active[s]].double() for s in (0, 1)], 1)
        actual = tenso_contr__torch.cat([q[:, s, self.active[s]] for s in (0, 1)], 1)
        mimic = tenso_contr__torch.stack([q[:, s, f] - ratio * q[:, s, m] - offset for s in (0, 1) for (f, m, ratio, offset) in self.mimic[s]], 1)
        if halted is None:
            halted = tenso_contr__torch.zeros(self.n, device=self.device, dtype=tenso_contr__torch.bool)
        action = actions.double()
        target = finger_pre_guard(command, action, actual, mimic, self.low, self.high, DT, self.rates, policy_owned=True)
        (_, target) = surface_final_guard(command, target, self.low, self.high, DT)
        if self.fixed_fingers:
            target[:, self.fixed_fingers] = command[:, self.fixed_fingers]
        (palm, angle) = wrist_requested_targets(self.palms, self.angles, action, self.rates, DT)
        projection = project_tensor_policy_wrist_targets(root_seed_w=self.roots, measured_base_w=base_poses_wxyz.double(), common_lift_offset_m=self.roots.new_zeros(self.n), measured_clearance_m=clearance_m.double(), palm_offsets_m=self.palms, requested_palm_offsets_m=palm, angles_rad=self.angles, requested_angles_rad=angle, allowed=~halted, live=~halted)
        anchors = state['root_pose_wxyz'].double()
        local = tenso_contr__torch.stack([local_translation(projection['projected_root_poses'][:, s], anchors[:, s]) for s in (0, 1)], 1)
        bad = (local.abs() > MOUNT['limit_m']).any((1, 2)) & ~halted
        if not self.isolate_constraints and bool(bad.any()):
            raise ValueError('original mount translation hard limit')
        expected = []
        for (s, side) in enumerate(('left', 'right')):
            values = tenso_contr__torch.cat((target[:, 6 * s:6 * s + 6], local[:, s], projection['projected_angles'][:, s]), 1).float()
            expected.append(values)
        self.pending = dict(stamp=stamp, epoch=self.epoch, expected=expected, projection=projection,
                            halted=halted, constraint_failure=bad, phase='prepared')
        return self.pending

    def submit(self, transaction, *, stamp, wrenches, state, reaction_source):
        if transaction is not self.pending or stamp != transaction['stamp'] or self.epoch != transaction['epoch'] or (transaction['phase'] != 'prepared'):
            raise ValueError('one owned unsubmitted pre-action GPU transaction required')
        try:
            expected_source = dict(source_solver_id=stamp.parent_solver_id, parent_num_envs=stamp.parent_num_envs, run_uuid=stamp.run_uuid, source_physics_step=stamp.sdk_step, applied_physics_step=stamp.sdk_step + 1, rod_ticks=stamp.rod_ticks, episode_uids=stamp.episode_uids, reset_ids=stamp.reset_ids, solver_time_s=stamp.solver_time_s)
            if not isinstance(reaction_source, dict) or any((reaction_source.get(k) != v for (k, v) in expected_source.items())) or reaction_source.get('mesh_wrenches') is not wrenches:
                raise ValueError('owned immediate-next-SDK reaction and current source identity required')
            if not isinstance(wrenches, tenso_contr__torch.Tensor) or wrenches.shape != (self.n, self.mesh_count, 6) or wrenches.device != self.device or (not wrenches.is_floating_point()):
                raise ValueError('finite complete same-device actual mesh wrenches required')
            for key in ('link_pose_xyzw', 'body_com_pose_w'):
                value = state[key]
                if not isinstance(value, tenso_contr__torch.Tensor) or value.ndim != 4 or value.shape[:2] != (self.n, 2) or (value.shape[-1] != 7) or (value.device != self.device) or (not value.is_floating_point()) or any((max(ids) >= value.shape[2] for ids in self.contact_bodies)):
                    raise ValueError('finite complete actual link and COM poses required')
            for s in (0, 1):
                value = transaction['expected'][s]
                if not isinstance(value, tenso_contr__torch.Tensor) or value.shape != (self.n, len(self.write_ids[s])) or value.device != self.device or (not value.is_floating_point()):
                    raise ValueError('finite complete projected command before native write required')
            finite_inputs = [wrenches, state['link_pose_xyzw'], state['body_com_pose_w'], *transaction['expected']]
            finite = tenso_contr__torch.isfinite(tenso_contr__torch.cat([v.reshape(-1) for v in finite_inputs])).all()
            forces = []
            torques = []
            over_force = []
            for s in (0, 1):
                bodies = self.contact_bodies[s]
                w = wrenches[:, self.slots[s]].double()
                origin = state['link_pose_xyzw'][:, s, bodies, :3].double()
                com = state['body_com_pose_w'][:, s, bodies, :3].double()
                over_force.append((tenso_contr__torch.linalg.vector_norm(w[..., :3], dim=-1) > 30).any(1))
                forces.append(w[..., :3].float())
                torques.append((w[..., 3:] + tenso_contr__torch.linalg.cross(origin - com, w[..., :3], dim=-1)).float())
            representable = tenso_contr__torch.isfinite(tenso_contr__torch.cat([v.reshape(-1) for v in forces + torques])).all()
            over_force = tenso_contr__torch.stack(over_force).any(0)
            if not bool(finite & representable):
                raise ValueError('finite command/link/COM/wrench and representable SDK force/COM torque required')
            if self.isolate_constraints:
                transaction['constraint_failure'] |= over_force & ~transaction['halted']
                isolated = transaction['halted'] | transaction['constraint_failure']
                for s in (0, 1):
                    previous = state['backend_targets'][:, s, self.write_ids[s]]
                    transaction['expected'][s] = tenso_contr__torch.where(isolated[:, None], previous, transaction['expected'][s])
                    forces[s] = forces[s].masked_fill(isolated[:, None, None], 0)
                    torques[s] = torques[s].masked_fill(isolated[:, None, None], 0)
            elif bool(over_force.any()):
                raise ValueError('original 30 N reaction guard failed')
            for (s, side) in enumerate(('left', 'right')):
                owner = self.batch.sides[side]
                owner.hand.permanent_wrench_composer.reset()
                owner.set_wrenches(forces=forces[s], torques=torques[s], body_ids=self.contact_bodies[s], is_global=True)
                owner.set_joint_position_target(transaction['expected'][s], joint_ids=self.write_ids[s])
            self.batch.write_data_to_sim()
            transaction['phase'] = 'submitted'
        except BaseException:
            transaction['phase'] = 'failed'
            self.pending = None
            self._failed = True
            raise

    def finish(self, transaction, state, *, stamp):
        pre = transaction['stamp']
        if transaction is not self.pending or self.epoch != transaction['epoch'] or transaction['phase'] != 'submitted' or (stamp.sdk_step != pre.sdk_step + 1) or (stamp.rod_ticks != tuple((t + 12 for t in pre.rod_ticks))) or (not solver_clock_matches(pre.solver_time_s, stamp.solver_time_s, steps=1)) or any((getattr(stamp, k) != getattr(pre, k) for k in ('parent_solver_id', 'parent_num_envs', 'env_rows', 'run_uuid', 'episode_uids', 'rod_uids', 'reset_ids', 'profile_sha256', 'scope'))):
            self._failed = True
            self.pending = None
            raise ValueError('one actual SDK/rod step with unchanged source identity required by owned batch')
        bad = self.joint_failures(state)
        wrong_target = tenso_contr__torch.zeros_like(bad)
        for s in (0, 1):
            actual = state['backend_targets'][:, s, self.write_ids[s]]
            wrong_target |= ~tenso_contr__torch.isclose(actual, transaction['expected'][s], rtol=0, atol=1e-07).all(1)
        if bool(wrong_target.any()) or (not self.isolate_constraints and bool(bad.any())):
            self.pending = None
            self._failed = True
            raise ValueError('actual SDK target/finite/joint/MIMIC guard failed')
        active = ~(transaction['halted'] | transaction['constraint_failure'] | bad)
        self.palms = tenso_contr__torch.where(active[:, None, None], transaction['projection']['palm_offsets_m'], self.palms)
        self.angles = tenso_contr__torch.where(active[:, None, None], transaction['projection']['projected_angles'], self.angles)
        transaction['phase'] = 'finished'
        self.pending = None
        return bad
