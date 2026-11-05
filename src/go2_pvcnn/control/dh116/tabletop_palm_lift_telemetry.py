"""Read-only left/right palm displacement from each actual physical reset.

The second axis retains the measured world's hand order (left, right). These
diagnostics do not use targets, nominal seeds, object poses, or task rewards.
"""

import torch


class PalmLiftTelemetry:
    def __init__(self, n, device):
        self.n = n
        self.device = device
        self.reference_z_m = torch.full((n, 2), float('nan'),
                                       dtype=torch.float64, device=device)
        self.last_m = torch.zeros((n, 2), dtype=torch.float64, device=device)
        self.peak_m = torch.zeros_like(self.last_m)
        self.valid = torch.zeros((n, 2), dtype=torch.bool, device=device)

    @torch.no_grad()
    def reset(self, ids, measurement):
        """Refresh selected physical-reset references, preserving tick samples."""
        poses = measurement['wrist_pose'][ids]
        finite = torch.isfinite(poses).all(dim=-1)
        z = poses[..., 2].to(dtype=self.reference_z_m.dtype)
        self.reference_z_m[ids] = torch.where(finite, z, float('nan'))

    @torch.no_grad()
    def begin_tick(self):
        self.last_m.zero_()
        self.peak_m.zero_()
        self.valid.zero_()

    @torch.no_grad()
    def observe(self, measurement, live):
        """Capture finite live palms before any terminal physical reset."""
        poses = measurement['wrist_pose']
        delta = poses[..., 2].to(dtype=self.reference_z_m.dtype) - self.reference_z_m
        eligible = (live[:, None] & torch.isfinite(poses).all(dim=-1)
                    & torch.isfinite(self.reference_z_m) & torch.isfinite(delta))
        self.last_m.copy_(torch.where(eligible, delta, self.last_m))
        # A first negative displacement remains negative, rather than becoming
        # a fabricated zero peak. Later invalid samples preserve finite history.
        peak = torch.where(self.valid, torch.maximum(self.peak_m, delta), delta)
        self.peak_m.copy_(torch.where(eligible, peak, self.peak_m))
        self.valid.logical_or_(eligible)

    @torch.no_grad()
    def snapshot(self):
        return {'last_m': self.last_m.clone(), 'peak_m': self.peak_m.clone(),
                'valid': self.valid.clone()}
