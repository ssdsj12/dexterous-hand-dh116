"""Measured overload takes precedence over a policy's closing target.

This is a target governor, not a change to measured forces or safety limits.
The caller still applies joint envelopes, mimic recovery and command slew.
"""
import torch

CONTRACT = dict(version='measured_pressure_relief_v1',load_band_ratio=1.25,
                release_rate_rad_s=.03,unmapped_flexion='block_positive_closure',
                palm_motion='pause_during_overload',mimic_recovery='retains_priority')


def pressure_limited_target(command, target, loads, requested, direction, dt, deadband):
    """Batched (N,2,6) commands, (N,2) loads; static DH116 flexion signs.

Calibrated nonzero directions release at <=.03 rad/s along the pressure
direction. Remaining DH116 flexion joints (positive closes) may open but
cannot continue closing under overload. An already faster opening target
is preserved; actual command speed is limited by the caller.
"""
    overloaded = loads > 1.25*requested+deadband
    sign = torch.where(direction==0,torch.ones_like(direction),direction.sign())
    oriented_delta = (target-command)*sign
    releasing = -.03*dt*direction.abs()
    limited = command+torch.minimum(oriented_delta,releasing)*sign
    return torch.where(overloaded[...,None],limited,target),overloaded
