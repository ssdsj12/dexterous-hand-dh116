"""Raw measured lift success; no policy, force-goal or actuator side effects."""
import math

import torch

from .tabletop_lift_hold_window import hold_window_step
from .tabletop_multifinger_monitor import rod_points_in_hands
from .tabletop_surface_force_sharing import surface_targets


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

    def __init__(self, n, device, reference_bottom_m, dtype=torch.float64, control_period_s=1/60,
                 minimum_lift_m=.03):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError('positive environment count required')
        if not math.isfinite(control_period_s) or control_period_s <= 0:
            raise ValueError('finite positive control period required')
        if not math.isfinite(minimum_lift_m) or minimum_lift_m <= 0:
            raise ValueError('finite positive minimum lift required')
        self.n, self.period = n, float(control_period_s)
        self.minimum_lift_m = float(minimum_lift_m)
        self._duration = torch.zeros(n, device=device, dtype=dtype)
        ref = torch.as_tensor(reference_bottom_m, device=device, dtype=dtype)
        if ref.shape not in ((), (n,)) or not torch.isfinite(ref).all():
            raise ValueError('finite scalar or [N] reference bottom required')
        self.reference_bottom_m = ref.expand(n).clone()
        self._anchor = self._duration.new_zeros(n, 2, 3, 3)
        self._latched = torch.zeros(n, device=device, dtype=torch.bool)
        self._paired = self._latched[:, None].expand(n, 2).clone()
        self._window_good = self._latched.clone()
        self._physical_window_valid = self._latched.clone()
        self._drift = self._duration.clone()
        self._elapsed, self._samples = 0., []
        self._ok = torch.ones_like(self._latched)

    @property
    def duration(self):
        return self._duration

    @property
    def drift_m(self):
        return self._drift

    @property
    def measured_drift_m(self):
        # Rejected windows use zeroed samples internally to isolate bad rows.
        # Their synthetic displacement must never become a slip reward.
        return torch.where(self._physical_window_valid, self._drift, 0.)

    @torch.no_grad()
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

    @torch.no_grad()
    def update(self, object_pose, wrist_pose, loads, vectors, depth, ordinary,
               bottom, object_velocity, valid, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('finite positive dt required')
        values = (object_pose, wrist_pose, loads, vectors, depth, ordinary, bottom, object_velocity, valid)
        tails = ((7,), (2, 7), (2, 5), (2, 5, 3), (2, 5), (), (), (6,), ())
        for value, tail in zip(values, tails):
            expected_dtype = torch.bool if value is valid else self._duration.dtype
            if (value.shape != (self.n, *tail) or value.dtype != expected_dtype
                    or value.device != self._duration.device):
                raise ValueError('matching measurement shapes, dtype and device required')
        good = valid.clone()
        for value in values[:-1]:
            good &= torch.isfinite(value).reshape(self.n, -1).all(-1)
        good &= (object_pose[:, 3:].abs().amax(-1) > 0)
        good &= (wrist_pose[..., 3:].abs().amax(-1) > 0).all(-1)
        good &= (loads >= 0).all((1, 2)) & (ordinary >= 0) & (depth >= 0).all((1, 2))
        physical = (good & (bottom >= self.reference_bottom_m + self.minimum_lift_m)
                    & (object_velocity[:, :3].norm(dim=-1) < .03)
                    & (depth.amax((1, 2)) <= .0008))
        # Invalid rows must never reach helpers that reject an entire batch.
        clean_object = torch.where(good[:, None], object_pose, 0.)
        clean_wrist = torch.where(good[:, None, None], wrist_pose, 0.)
        clean_object[~good, 3] = 1
        clean_wrist[~good, :, 3] = 1
        points = rod_points_in_hands(clean_object, clean_wrist).clone()
        raw = (loads.clone(), vectors.clone(), ordinary.clone(), points)
        event, updated = torch.zeros_like(valid), False
        remaining = float(dt)
        while remaining > 1e-12:
            weight = min(remaining, self.period - self._elapsed)
            self._samples.append((weight, raw))
            self._ok &= physical
            self._elapsed += weight
            remaining -= weight
            if self._elapsed < self.period - 1e-12:
                continue
            # Clean whole rejected rows, including samples collected before reset.
            means = []
            for i in range(3):
                mean = sum(weight * sample[i] for weight, sample in self._samples) / self._elapsed
                mask = self._ok.reshape(self.n, *((1,) * (mean.ndim - 1)))
                means.append(torch.where(mask, mean, 0.))
            mean_load, mean_vectors, mean_ordinary = means
            self._paired = surface_targets(mean_load, mean_vectors)['paired']
            normal = mean_load.sum((1, 2))
            physical_ok = (self._ok & self._paired.all(-1)
                           & (mean_ordinary / (mean_ordinary + normal).clamp_min(1e-12) <= .05 + 1e-12))
            self._physical_window_valid = physical_ok.clone()
            samples = torch.stack([sample[3] for _, sample in self._samples])
            samples = torch.where(self._ok[None, :, None, None, None], samples, 0.)
            previous = self._duration
            self._duration, self._anchor, self._latched, diag = hold_window_step(
                previous, self._anchor, self._latched, samples, physical_ok, self._elapsed)
            self._window_good, self._drift = diag['good'], diag['drift_m']
            event |= (previous < 5. - 1e-12) & (self._duration >= 5. - 1e-12)
            self._elapsed, self._samples = 0., []
            self._ok.fill_(True)
            updated = True
        return dict(duration=self._duration.clone(), event=event, updated=updated,
                    paired=self._paired.clone(), window_good=self._window_good.clone(), drift_m=self._drift.clone())
