"""Causal measured contact features, independent of native physics and success."""
import math
from numbers import Real

import torch


OBSERVATION_VERSION = 'dh116-surface-sharing-history-v3'
CONTRACT = dict(version='native_surface_contact_history_v1', feature_size=50,
                control_period_s=1 / 60, ema_alpha=.2, load_log1p=True,
                vector_scale_n=10, depth_scale_m=.001)


class NativeSurfaceHistory:
    """Integrate real dt and publish only completed control periods.

    Each row owns its clock so reset(ids) starts a fresh full period for only
    those environments. Invalid measurements clear that row's published history
    immediately and invalidate its current period, without restarting its clock.
    A constant input spanning boundaries is split across all affected periods.
    Depth is the per-pad maximum of the last completed valid period, without EMA.
    """

    def __init__(self, n, device, dtype=torch.float64, period=1 / 60):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError('positive integer environment count required')
        if isinstance(period, bool) or not isinstance(period, Real) or not math.isfinite(period) or period <= 0:
            raise ValueError('finite positive control period required')
        self.n, self.period = n, float(period)
        self._loads = torch.zeros(n, 2, 5, device=device, dtype=dtype)
        if not self._loads.is_floating_point():
            raise ValueError('floating history dtype required')
        self._vectors = self._loads.new_zeros(n, 2, 5, 3)
        self._previous_depth = torch.zeros_like(self._loads)
        self._load_integral = torch.zeros_like(self._loads)
        self._vector_integral = torch.zeros_like(self._vectors)
        self._max_depth = torch.zeros_like(self._loads)
        # Clock precision is independent of the observation/measurement dtype.
        self._elapsed = torch.zeros(n, device=device, dtype=torch.float64)
        self._valid = torch.ones(n, device=device, dtype=torch.bool)

    @torch.no_grad()
    def features(self):
        """Return an owned [N,50] tensor; partial periods are never exposed."""
        return torch.cat((self._loads.log1p().flatten(1),
                          (self._vectors / 10).flatten(1),
                          (self._previous_depth / .001).flatten(1)), dim=1)

    @torch.no_grad()
    def reset(self, ids):
        """Clear only selected integer indices or a boolean [N] mask."""
        selected = torch.zeros_like(self._valid)
        selected[ids] = True  # Validate selection before changing any history.
        for value in (self._loads, self._vectors, self._previous_depth,
                      self._load_integral, self._vector_integral, self._max_depth, self._elapsed):
            value[selected] = 0
        self._valid[selected] = True

    @torch.no_grad()
    def update(self, loads, vectors, depth, dt):
        """Accept matching measured tensors; isolate invalid numeric rows.

        dt is a finite positive scalar in seconds. Schema errors raise before
        changing any state. Numeric errors discard just their row's whole
        current period, including samples preceding the invalid measurement.
        """
        if isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('finite positive dt required')
        for value, tail in ((loads, (2, 5)), (vectors, (2, 5, 3)), (depth, (2, 5))):
            if (not isinstance(value, torch.Tensor) or value.shape != (self.n, *tail)
                    or value.dtype != self._loads.dtype or value.device != self._loads.device):
                raise ValueError('matching tensor shape, dtype and device required')
        good = (torch.isfinite(loads).all((1, 2)) & (loads >= 0).all((1, 2))
                & torch.isfinite(vectors).all((1, 2, 3))
                & torch.isfinite(depth).all((1, 2)) & (depth >= 0).all((1, 2)))
        for value in (self._loads, self._vectors, self._previous_depth,
                      self._load_integral, self._vector_integral, self._max_depth):
            value[~good] = 0
        clean_loads = torch.where(good[:, None, None], loads, 0.)
        clean_vectors = torch.where(good[:, None, None, None], vectors, 0.)
        clean_depth = torch.where(good[:, None, None], depth, 0.)
        remaining = torch.full_like(self._elapsed, float(dt))
        tolerance = 8 * math.ulp(self.period)
        while bool((remaining > 0).any()):
            active = remaining > 0
            weight = torch.minimum(remaining, (self.period - self._elapsed).clamp_min(0))
            self._valid &= good | ~active
            scaled = weight.to(self._loads.dtype)
            self._load_integral += clean_loads * scaled[:, None, None]
            self._vector_integral += clean_vectors * scaled[:, None, None, None]
            self._max_depth.copy_(torch.where(active[:, None, None],
                                             torch.maximum(self._max_depth, clean_depth), self._max_depth))
            self._elapsed += weight
            remaining = (remaining - weight).clamp_min(0)
            # Snap float64 boundary roundoff rather than carrying an old depth
            # spike into the next period with a spurious near-zero remainder.
            remaining = torch.where(remaining <= tolerance, 0., remaining)
            completed = active & (self._elapsed >= self.period - tolerance)
            publish = completed & self._valid
            denominator = self._elapsed.to(self._loads.dtype).clamp_min(torch.finfo(self._loads.dtype).tiny)
            self._loads.copy_(torch.where(publish[:, None, None],
                .8 * self._loads + .2 * self._load_integral / denominator[:, None, None], self._loads))
            self._vectors.copy_(torch.where(publish[:, None, None, None],
                .8 * self._vectors + .2 * self._vector_integral / denominator[:, None, None, None], self._vectors))
            self._previous_depth.copy_(torch.where(publish[:, None, None], self._max_depth, self._previous_depth))
            for value in (self._load_integral, self._vector_integral, self._max_depth, self._elapsed):
                value[completed] = 0
            self._valid[completed] = True


def enable_contact_history(task):
    """Explicitly attach history and rebuild the task's observation snapshot."""
    history = NativeSurfaceHistory(task.n, task.device, dtype=task.command.dtype)
    task.surface_contact_history = history
    task.last_observation = task.observe().clone()
    return history
