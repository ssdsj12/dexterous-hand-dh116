"""Pure Torch multifinger support diagnostics; no simulator or control side effects.

Poses are xyz+wxyz. All five digits are interchangeable; this module neither
changes the 12 active / 10 passive mechanism nor sets forces or joint targets.
"""

import math
from numbers import Integral, Real

import torch


def _tensor(name, value, shape, *, device=None, dtype=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}")
    if dtype is None:
        if not value.is_floating_point():
            raise TypeError(f"{name} must have floating dtype")
    elif value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")


def _unit_quaternion(q):
    # Normalize scaled quaternions without overflowing their squared magnitude.
    scale = q.abs().amax(dim=-1, keepdim=True)
    if (scale == 0).any():
        raise ValueError("pose quaternion must be nonzero")
    q = q / scale
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def _rotate(q, points):
    xyz = q[..., 1:].expand_as(points)
    twice_cross = 2 * torch.linalg.cross(xyz, points, dim=-1)
    return points + q[..., :1] * twice_cross + torch.linalg.cross(xyz, twice_cross, dim=-1)


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
    if not isinstance(object_pose, torch.Tensor) or object_pose.ndim != 2:
        raise ValueError("object_pose must have shape [N,7]")
    n = object_pose.shape[0]
    _tensor("object_pose", object_pose, (n, 7))
    _tensor("hand_pose", hand_pose, (n, 2, 7), device=object_pose.device, dtype=object_pose.dtype)
    if not torch.isfinite(object_pose).all() or not torch.isfinite(hand_pose).all():
        raise ValueError("poses must be finite")
    local = object_pose.new_tensor([[0, -0.2, 0], [0, 0.2, 0], [0.008, 0, 0]])
    local = local.expand(n, 3, 3)
    object_q = _unit_quaternion(object_pose[:, 3:])[:, None]
    world = _rotate(object_q, local) + object_pose[:, None, :3]
    hand_q = _unit_quaternion(hand_pose[..., 3:]).clone()
    hand_q[..., 1:] *= -1  # Inverse rotation, wxyz.
    return _rotate(hand_q[:, :, None], world[:, None] - hand_pose[:, :, None, :3])


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
            raise ValueError("n must be a positive integer")
        if isinstance(window_steps, bool) or not isinstance(window_steps, Integral) or window_steps < 2:
            raise ValueError("window_steps must be an integer >=2")
        if hand_force_range is not None:
            if not isinstance(hand_force_range, (tuple, list)) or len(hand_force_range) != 2:
                raise ValueError("hand_force_range must be a (min_n, max_n) pair or None")
            if any(isinstance(bound, bool) or not isinstance(bound, Real) or not math.isfinite(bound)
                   for bound in hand_force_range):
                raise ValueError("hand_force_range bounds must be finite real numbers")
            lower, upper = hand_force_range
            if not 0 <= lower < upper:
                raise ValueError("hand_force_range must satisfy 0 <= min_n < max_n")
            hand_force_range = (float(lower), float(upper))
        if max_digit_ratio is not None:
            if (isinstance(max_digit_ratio, bool) or not isinstance(max_digit_ratio, Real)
                    or not math.isfinite(max_digit_ratio) or max_digit_ratio < 1):
                raise ValueError("max_digit_ratio must be a finite real number >=1 or None")
            max_digit_ratio = float(max_digit_ratio)
        self.n = int(n)
        self.window_steps = int(window_steps)
        self.hand_force_range = hand_force_range
        self.max_digit_ratio = max_digit_ratio
        self.device = torch.empty(0, device=device).device
        self._dtype = None  # Infer floating precision from the first measurements.
        self._cursor = 0
        self._count = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self._anchored = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.ready = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.loaded = torch.zeros((self.n, 2, 5), dtype=torch.bool, device=self.device)
        self.force_cv = torch.zeros((self.n, 2, 5), device=self.device)
        self.relative_drift_m = torch.zeros(self.n, device=self.device)
        self.slip_rate_m_s = torch.zeros(self.n, device=self.device)
        self.hand_force_in_range = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.hand_force_min_n = torch.zeros((self.n, 2), device=self.device)
        self.hand_force_max_n = torch.zeros((self.n, 2), device=self.device)
        self.digit_ratio_ok = torch.zeros(self.n, dtype=torch.bool, device=self.device)

    def _allocate(self, dtype):
        self._dtype = dtype
        options = dict(device=self.device, dtype=dtype)
        self._loads = torch.zeros((self.window_steps, self.n, 2, 5), **options)
        self._hand_total_loads = torch.zeros((self.window_steps, self.n, 2), **options)
        self._points = torch.zeros((self.window_steps, self.n, 2, 3, 3), **options)
        self._opposition = torch.zeros(
            (self.window_steps, self.n, 2, 5, 5), device=self.device, dtype=torch.bool
        )
        self._dt = torch.ones(self.window_steps, **options)
        self._anchor = torch.zeros((self.n, 2, 3, 3), **options)

    def _check_points(self, points):
        _tensor("relative_points", points, (self.n, 2, 3, 3), device=self.device, dtype=self._dtype)

    def _selected_rows(self, ids):
        ids = torch.as_tensor(ids, device=self.device)
        if ids.dtype == torch.bool:
            _tensor("ids", ids, (self.n,), device=self.device, dtype=torch.bool)
            return ids
        if ids.ndim != 1 or (ids.numel() and ids.dtype not in (torch.int32, torch.int64)):
            raise TypeError("ids must be a 1D integer index tensor or a boolean [N] mask")
        if ((ids < 0) | (ids >= self.n)).any():
            raise ValueError("ids are out of range")
        selected = torch.zeros(self.n, device=self.device, dtype=torch.bool)
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

    @torch.no_grad()
    def reset(self, ids):
        """Clear selected rows after physical reset; accept indices or [N] bool mask.

        Clears history, anchors, ready/loaded and numeric diagnostics immediately.
        Safe before allocation or with empty ids. Other rows, the shared ring
        cursor, and interval history remain unchanged. A full new window is
        needed to become ready again; anchors require an explicit new latch.
        """
        self._clear(self._selected_rows(ids))

    @torch.no_grad()
    def update(self, loads, relative_points, normals, dt, valid, reset, *, hand_total_load_n=None):
        """Consume one step; dt is the positive scalar interval in seconds.

        hand_total_load_n is optional [N,2] actual total normal force, including
        palm support, with the same floating dtype/device as loads. If omitted,
        sum the five digit loads. These totals drive both the force range and
        cross-hand window-mean balance. Nonfinite/negative totals invalidate only their
        own row; malformed tensor structure raises before changing any history.
        """
        _tensor("loads", loads, (self.n, 2, 5), device=self.device, dtype=self._dtype)
        _tensor("relative_points", relative_points, (self.n, 2, 3, 3), device=self.device, dtype=loads.dtype)
        _tensor("normals", normals, (self.n, 2, 5, 3), device=self.device, dtype=loads.dtype)
        _tensor("valid", valid, (self.n,), device=self.device, dtype=torch.bool)
        _tensor("reset", reset, (self.n,), device=self.device, dtype=torch.bool)
        if isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar")
        if hand_total_load_n is None:
            hand_total_load_n = loads.sum(dim=-1)
        else:
            _tensor("hand_total_load_n", hand_total_load_n, (self.n, 2), device=self.device, dtype=loads.dtype)
        if self._dtype is None:
            self._allocate(loads.dtype)

        good = valid & ~reset
        for measurement in (loads, relative_points, normals, hand_total_load_n):
            good = good & torch.isfinite(measurement).flatten(1).all(dim=1)
        good = good & (loads >= 0).flatten(1).all(dim=1)
        good = good & (hand_total_load_n >= 0).all(dim=1)
        self._clear(~good)
        # torch.where also removes NaN/Inf before any reductions or divisions.
        clean_loads = torch.where(good[:, None, None], loads, 0)
        clean_points = torch.where(good[:, None, None, None], relative_points, 0)
        clean_normals = torch.where(good[:, None, None, None], normals, 0)
        # Scale first so very large finite normal magnitudes remain usable.
        scale = clean_normals.abs().amax(dim=-1, keepdim=True)
        unit = clean_normals / scale.clamp_min(torch.finfo(loads.dtype).tiny)
        length = torch.linalg.vector_norm(unit, dim=-1, keepdim=True)
        unit = unit / length.clamp_min(torch.finfo(loads.dtype).tiny)
        dot = unit @ unit.transpose(-1, -2)
        nonzero = scale.squeeze(-1) > 0
        opposed = (dot <= -0.25) & nonzero[..., :, None] & nonzero[..., None, :]

        self._loads[self._cursor].copy_(clean_loads)
        self._hand_total_loads[self._cursor].copy_(torch.where(good[:, None], hand_total_load_n, 0))
        self._points[self._cursor].copy_(clean_points)
        self._opposition[self._cursor].copy_(opposed)
        self._dt[self._cursor] = dt
        self._cursor = (self._cursor + 1) % self.window_steps
        self._count.add_(good).clamp_(max=self.window_steps)
        full = self._count == self.window_steps

        hand_min = self._hand_total_loads.amin(dim=0)
        hand_max = self._hand_total_loads.amax(dim=0)
        self.hand_force_min_n = torch.where(full[:, None], hand_min, 0)
        self.hand_force_max_n = torch.where(full[:, None], hand_max, 0)
        self.hand_force_in_range = full.clone()
        if self.hand_force_range is not None:
            lower, upper = self.hand_force_range
            self.hand_force_in_range &= (hand_min >= lower).all(dim=-1) & (hand_max <= upper).all(dim=-1)

        mean = self._loads.mean(dim=0)
        std = self._loads.std(dim=0, unbiased=False)
        cv = std / mean.clamp_min(torch.finfo(loads.dtype).tiny)
        self.force_cv = torch.where(full[:, None, None], cv, 0)
        self.loaded = (self._loads.amin(dim=0) > 0.05) & full[:, None, None]
        count = self.loaded.sum(dim=-1)
        self.digit_ratio_ok = full.clone()
        if self.max_digit_ratio is not None:
            digit_min = torch.where(self.loaded[None], self._loads, float("inf")).amin(dim=-1)
            digit_max = torch.where(self.loaded[None], self._loads, 0).amax(dim=-1)
            self.digit_ratio_ok &= (count > 0).all(dim=-1) & (
                digit_max <= self.max_digit_ratio * digit_min
            ).all(dim=(0, 2))
        finger_mean = (mean * self.loaded).sum(dim=-1) / count.clamp_min(1)
        balanced = (mean >= 0.5 * finger_mean[..., None]) & (mean <= 1.5 * finger_mean[..., None])
        participating = mean > 0.05
        stable_digits = ((~participating) | (self.loaded & balanced & (cv <= 0.25))).all(dim=(1, 2))
        hand_load = self._hand_total_loads.mean(dim=0)
        hand_balance = (hand_load.amin(dim=-1) > 0) & (hand_load.amax(dim=-1) <= 1.5 * hand_load.amin(dim=-1))
        pairs = self.loaded[..., :, None] & self.loaded[..., None, :]
        # An opposing pair must exist in each hand on every raw sample.
        opposition = (self._opposition & pairs[None]).flatten(-2).any(dim=-1).all(dim=(0, 2))

        span = self._points.amax(dim=0) - self._points.amin(dim=0)
        drift = torch.linalg.vector_norm(span, dim=-1).amax(dim=(1, 2))
        order = (torch.arange(self.window_steps, device=self.device) + self._cursor) % self.window_steps
        ordered = self._points[order]
        speeds = torch.linalg.vector_norm(ordered[1:] - ordered[:-1], dim=-1)
        speeds = speeds / self._dt[order][1:, None, None, None]
        rate = speeds.amax(dim=(0, 2, 3))
        self.relative_drift_m = torch.where(full, drift, 0)
        self.slip_rate_m_s = torch.where(full, rate, 0)
        self.ready = (
            full & self.hand_force_in_range & self.digit_ratio_ok
            & (count >= 2).all(dim=-1) & stable_digits & hand_balance & opposition
            & (drift <= 0.0005) & (rate <= 0.001)
            & torch.isfinite(cv).all(dim=(1, 2))
        )
        return {name: getattr(self, name) for name in (
            "ready", "loaded", "force_cv", "relative_drift_m", "slip_rate_m_s",
            "hand_force_in_range", "hand_force_min_n", "hand_force_max_n", "digit_ratio_ok"
        )}

    @torch.no_grad()
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
        finite = torch.isfinite(relative_points).flatten(1).all(dim=1)
        self._clear(selected & ~finite)
        selected = selected & finite
        self._anchor[selected] = relative_points[selected]
        self._anchored[selected] = True

    @torch.no_grad()
    def anchor_error(self, relative_points):
        """Maximum Euclidean point error [N]; +inf for unlatched/nonfinite rows."""
        self._check_points(relative_points)
        if self._dtype is None:
            return relative_points.new_full((self.n,), float("inf"))
        finite = torch.isfinite(relative_points).flatten(1).all(dim=1)
        error = torch.linalg.vector_norm(relative_points - self._anchor, dim=-1).amax(dim=(1, 2))
        return torch.where(self._anchored & finite, error, float("inf"))
