"""Read-only physical feedback for the last completed native control interval.

Call begin after task.begin_step; sample after consume with the *raw* measurement
and pre-consume live mask, before any physical reset; finish before end_step.
Only finish publishes features. Invalid live feedback poisons that row's entire
interval: sample_valid immediately becomes false; finish publishes zero columns
and valid=false. The caller must abort rollout before optimization on such rows
(zeros are not a valid transition).

Means use each row's own live dt, depth/drift use interval maxima, hold/window
use the last existing native tracker state, and wrist increments use measured
end minus begin positions. The terminal physical sample is included and frozen.
No tracker is created or updated, and no controller/reward state is written.
"""
import math
from numbers import Real
from types import SimpleNamespace

import torch

from .tabletop_multifinger_monitor import rod_points_in_hands
from .tabletop_surface_exploration import verified_support
from .tabletop_surface_force_sharing import surface_targets


# (name, column count, physical divisor), in local columns 0..19. Kept
# independent of the parent's canonical contract to avoid import coupling.
FEATURES = (
    ('support_n', 2, 10.), ('support_margin', 1, 1.),
    ('support_imbalance', 1, 1.), ('paired', 2, 1.),
    ('ordinary_fraction', 1, 1.), ('pad_depth', 1, .0008),
    ('anchor_drift', 1, .0005), ('slip_speed', 1, .01),
    ('hold_duration', 1, 5.), ('window_good', 1, 1.),
    ('pressure_limited', 2, 1.), ('mimic_blocked', 2, 1.),
    ('wrist_z_request', 2, .01), ('wrist_z_increment', 2, .001),
)


class SurfaceFeedback:
    def __init__(self, n, device, dtype=torch.float64):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError('positive integer environment count required')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('float32 or float64 feedback dtype required')
        self.n = n
        self._features = torch.zeros(n, 20, device=device, dtype=dtype)
        self._valid = torch.zeros(n, device=device, dtype=torch.bool)
        self._scales = self._features.new_tensor([scale for _, size, scale in FEATURES for _ in range(size)])
        self._sum = torch.zeros_like(self._features)
        self._elapsed = self._features.new_zeros(n)
        self._depth = self._elapsed.clone()
        self._drift = self._elapsed.clone()
        self._hold = self._elapsed.clone()
        self._window_good = self._valid.clone()
        self._ok = self._valid.clone()
        self._open = self._valid.clone()
        self._previous = self._features.new_zeros(n, 2, 3, 3)
        self._start_z = self._features.new_zeros(n, 2)
        self._end_z = self._start_z.clone()
        self._started = False

    @property
    def valid(self):
        """[N] bool for the published completed interval; reset rows are false."""
        return self._valid.clone()

    @property
    def sample_valid(self):
        """Immediate pending-interval validity for the parent's substep guard.

        begin initializes live rows from their measured baseline. A bad sample
        remains false through finish, even if later samples recover. Outside
        an interval this equals the published valid mask.
        """
        return (self._ok if self._started else self._valid).clone()

    def features(self):
        """[N,20], scaled physical values with no clipping or running stats."""
        return self._features.clone()

    def _tensor(self, value, tail, name, boolean=False):
        dtype = torch.bool if boolean else self._features.dtype
        if (not isinstance(value, torch.Tensor) or value.shape != (self.n, *tail)
                or value.dtype != dtype or value.device != self._features.device):
            raise ValueError(f'{name}: matching shape {(self.n, *tail)}, dtype {dtype}, device required')
        return value

    def _field(self, measurement, name, tail, boolean=False):
        if name not in measurement:
            raise ValueError('missing raw measurement: ' + name)
        return self._tensor(measurement[name], tail, name, boolean)

    def _finite(self, *values):
        good = torch.ones_like(self._valid)
        for value in values:
            good &= torch.isfinite(value).reshape(self.n, -1).all(-1)
        return good

    @staticmethod
    def _selected(value, mask, fill=0.):
        return torch.where(mask.reshape(mask.shape + (1,) * (value.ndim - 1)), value, fill)

    def _poses(self, measurement):
        wrist = self._field(measurement, 'wrist_pose', (2, 7))
        relative = self._field(measurement, 'object_relative', (3,))
        quat = self._field(measurement, 'object_quat', (4,))
        pose = torch.cat((relative + wrist[..., :3].mean(1), quat), -1)
        good = self._finite(wrist, relative, pose)
        good &= (quat.abs().amax(-1) > 0.) & (wrist[..., 3:].abs().amax(-1) > 0.).all(-1)
        return wrist, pose, good

    def _points(self, pose, wrist, mask):
        # Helpers reject whole batches. Identity poses isolate invalid/inactive
        # rows without letting a synthetic sample enter any accumulator.
        pose = self._selected(pose, mask).clone()
        wrist = self._selected(wrist, mask).clone()
        pose[~mask, 3] = 1.
        wrist[~mask, :, 3] = 1.
        return rod_points_in_hands(pose, wrist)

    @torch.no_grad()
    def reset(self, ids):
        """Clear selected rows, including pending pre-reset samples/baselines."""
        if isinstance(ids, torch.Tensor):
            if (ids.device != self._features.device or ids.ndim != 1
                    or ids.dtype not in (torch.bool, torch.int32, torch.int64)
                    or (ids.dtype == torch.bool and ids.shape != (self.n,))):
                raise ValueError('integer indices or boolean [N] reset mask required')
            if ids.dtype != torch.bool and ((ids < 0).any() or (ids >= self.n).any()):
                raise ValueError('reset indices out of range')
        elif (not isinstance(ids, (list, tuple))
              or any(type(i) is not int or not 0 <= i < self.n for i in ids)):
            raise ValueError('integer reset indices required')
        for value in (self._features, self._valid, self._sum, self._elapsed,
                      self._depth, self._drift, self._hold, self._window_good,
                      self._ok, self._open, self._previous, self._start_z, self._end_z):
            value[ids] = 0

    @torch.no_grad()
    def begin(self, task, world):
        if self._started:
            raise RuntimeError('finish the previous feedback interval before begin')
        done = self._tensor(task.done, (), 'task.done', True)
        wrist, pose, good = self._poses(task.m)
        points = self._points(pose, wrist, good & ~done)
        good &= self._finite(points)
        for value in (self._sum, self._elapsed, self._depth, self._drift, self._hold,
                      self._window_good):
            value.zero_()
        self._open.copy_(~done)
        self._ok.copy_(good & ~done)
        self._previous.copy_(self._selected(points, self._ok))
        self._start_z.copy_(self._selected(wrist[..., 2], self._ok))
        self._end_z.copy_(self._start_z)
        self._started = True

    def _trackers(self, task):
        success = getattr(task, 'surface_lift_success', None)
        slip = getattr(task, 'surface_slip_tracking', None)
        hold = self._elapsed.new_zeros(self.n)
        window = torch.zeros_like(self._valid)
        drift = hold.clone()
        if success is not None:
            hold = self._tensor(success.duration, (), 'native hold duration')
            window = self._tensor(success._window_good, (), 'native window_good', True)
        tracker = slip if slip is not None else success
        if tracker is not None:
            drift = self._tensor(tracker.measured_drift_m, (), 'native measured fixed-anchor drift')
        return hold, window, drift

    @torch.no_grad()
    def sample(self, task, world, measurement, live, dt):
        if not self._started:
            raise RuntimeError('begin required before physical feedback sample')
        if isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('finite positive physics dt required')
        live = self._tensor(live, (), 'live', True)
        done = self._tensor(task.done, (), 'task.done', True)
        active = self._open & live
        wrist, pose, good = self._poses(measurement)
        loads = self._field(measurement, 'effective_pad_loads_n', (2, 5))
        vectors = self._field(measurement, 'effective_pad_vectors_w', (2, 5, 3))
        valid = self._field(measurement, 'effective_pad_valid', (2, 5), True)
        depth = self._field(measurement, 'effective_pad_depth_m', (2, 5))
        rejected = self._field(measurement, 'rejected_contact_loads_n', (2,))
        mimic = self._field(measurement, 'mimic_errors', (10,))
        components = self._tensor(world.vertical_components, (2, 2), 'world.vertical_components')
        weight = self._tensor(world.object_weight_n, (), 'world.object_weight_n')
        actions = self._tensor(task.actions, (24,), 'task.actions')
        pressure = self._tensor(task.pressure_limited, (2,), 'task.pressure_limited', True)
        blocked = self._tensor(getattr(task, 'f_blocked', torch.zeros_like(pressure)),
                               (2,), 'task.f_blocked', True)
        # Existing native policy diagnostics use .012 rad for severe coupling.
        # Read back actual blocking; measured error alone never creates a guard.
        blocked = blocked & (mimic.reshape(self.n, 2, 5).abs().amax(-1) >= .012)
        hold, window, drift = self._trackers(task)
        good &= self._finite(loads, vectors, depth, rejected, mimic, components, weight, actions, hold, drift)
        good &= ((loads >= 0.) & (depth >= 0.) & (valid == (loads > 0.))).all((1, 2))
        good &= (rejected >= 0.).all(-1) & (weight > 0.) & (hold >= 0.) & (drift >= 0.)
        good &= (actions.abs() <= 1.).all(-1)
        if 'effective_pad_bad_env' in measurement:
            good &= ~self._field(measurement, 'effective_pad_bad_env', (), True)
        self._ok &= ~(active & ~good)
        usable = active & self._ok

        clean_loads = self._selected(loads, usable)
        clean_valid = valid & usable[:, None, None]
        clean_vectors = torch.where(clean_valid[..., None], vectors, 0.)
        clean_rejected = self._selected(rejected, usable)
        clean_components = self._selected(components, usable)
        # Reuse exactly the verified attribution rule, but with raw m and the
        # raw object's signed normal/friction components, never task.m's cache.
        raw = SimpleNamespace(m=dict(effective_pad_vectors_w=clean_vectors,
            vertical_hand_force_n=clean_components[..., 0], rejected_contact_loads_n=clean_rejected))
        _, support = verified_support(raw, clean_components.sum(-1))
        paired = surface_targets(clean_loads, clean_vectors, valid=clean_valid)['paired']
        normal = clean_loads.sum((1, 2))
        ordinary = clean_rejected.sum(-1)
        safe_weight = self._selected(weight, usable, 1.)
        points = self._points(pose, wrist, usable)
        # Adjacent measured object-in-wrist displacement, including the initial
        # begin->first-sample segment. Max over both hands and all rod points;
        # average its speed in time, so out-and-back slip cannot cancel.
        speed = (points - self._previous).norm(dim=-1).amax((1, 2)) / dt
        row = torch.cat((support,
            ((support.sum(-1) - safe_weight) / safe_weight)[:, None],
            ((support[:, 0] - support[:, 1]).abs() / support.abs().sum(-1).clamp_min(1e-12))[:, None],
            paired.to(self._features.dtype),
            (ordinary / (normal + ordinary).clamp_min(1e-12))[:, None],
            self._features.new_zeros(self.n, 2), speed[:, None],
            self._features.new_zeros(self.n, 2), pressure.to(self._features.dtype),
            blocked.to(self._features.dtype), self._selected(actions[:, [14, 17]], usable) * .01,
            self._features.new_zeros(self.n, 2)), -1)
        # Finite raw values can still overflow derived arithmetic. They must
        # invalidate a live row, never publish NaNs or hide them with clipping.
        derived_ok = self._finite(row, points)
        self._ok &= ~(active & ~derived_ok)
        usable &= derived_ok
        self._sum += self._selected(row, usable) * dt
        self._elapsed += usable.to(self._elapsed.dtype) * dt
        accepted_depth = torch.where(clean_valid, depth, 0.).amax((1, 2))
        self._depth.copy_(torch.maximum(self._depth, self._selected(accepted_depth, usable)))
        self._drift.copy_(torch.maximum(self._drift, self._selected(drift, usable)))
        self._hold[usable] = hold[usable]
        self._window_good[usable] = window[usable]
        self._end_z[usable] = wrist[..., 2][usable]
        self._previous[usable] = points[usable]
        self._ok &= ~active | self._finite(self._sum, self._elapsed,
            self._depth / .0008, self._drift / .0005,
            (self._end_z - self._start_z) / .001)
        # done is read only after including the terminal raw sample. Later
        # post-reset states cannot enter even if the caller reuses live=True.
        self._open &= live & ~done

    @torch.no_grad()
    def finish(self, task, world):
        if not self._started:
            raise RuntimeError('begin required before feedback finish')
        row = self._sum / self._elapsed.clamp_min(torch.finfo(self._elapsed.dtype).tiny)[:, None]
        row[:, 7] = self._depth
        row[:, 8] = self._drift
        row[:, 10] = self._hold
        row[:, 11] = self._window_good.to(row.dtype)
        row[:, 18:20] = self._end_z - self._start_z
        row /= self._scales
        valid = self._ok & (self._elapsed > 0.) & self._finite(row)
        self._features.copy_(self._selected(row, valid))
        self._valid.copy_(valid)
        self._started = False
