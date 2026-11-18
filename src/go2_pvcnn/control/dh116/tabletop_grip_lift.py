"""Joint-target closure/lift and measured success, without force-goal gates.

This module does not create a simulator or change actuator/physics settings.
The caller executes the returned targets through the existing bounded drives.
"""
import math
from numbers import Real

import torch

from .linear_coupling import active_limits


def _scalar(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return float(value)


def _limits(reference):
    return tuple(reference.new_tensor(a).repeat(2) for a in active_limits())


def _joints(value, name, reference=None):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != 12):
        raise ValueError(f'{name} must be a floating N x 12 tensor')
    if reference is not None:
        if value.shape != reference.shape:
            raise ValueError(f'{name} must match the command batch')
        value = value.to(reference)
    low, high = _limits(value)
    if (not torch.isfinite(value).all() or (value < low).any() or (value > high).any()):
        raise ValueError(f'{name} must be finite and inside active_limits')
    return value


def _offset(value, reference, name, *, maximum=.3):
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if (value.shape != (reference.shape[0],) or not torch.isfinite(value).all()
            or (value < 0).any() or (value > maximum).any()):
        raise ValueError(f'{name} must contain N finite offsets between 0 and {maximum} rad')
    return value


def closing_targets(seed, reference, margin, *, thumb_spread=None):
    """Return feasible closure targets; positive offsets affect only pinch joints.

``reference`` may be [12] or [N,12]. It is clamped after adding the offsets;
the valid ``seed`` is used for batch, device and dtype, not as the reference.
Margin is bounded by .3rad; optional thumb opposition spread by .8rad.
"""
    seed = _joints(seed, 'seed')
    reference = torch.as_tensor(reference, device=seed.device, dtype=seed.dtype)
    if reference.shape not in ((12,), seed.shape) or not torch.isfinite(reference).all():
        raise ValueError('reference must be finite [12] or [N,12]')
    margin = _offset(margin, seed, 'margin')
    spread = seed.new_zeros(seed.shape[0]) if thumb_spread is None else _offset(thumb_spread, seed, 'thumb_spread', maximum=.8)
    closed = reference.expand_as(seed).clone()
    closed[:, [1, 2, 7, 8]] += margin[:, None]
    closed[:, [0, 6]] += spread[:, None]
    low, high = _limits(seed)
    return closed.maximum(low).minimum(high)


def advance_program(command, seed, closed, elapsed_s, dt, *, close_s=2.,
                    lift_start_s=2.5, lift_speed_m_s=.008, lift_height_m=.04):
    """Slew each joint at <=.3rad/s and return an independent wrist Z offset.

Inputs are feasible joint commands, not measured q (which may deflect outside
the compatible command bounds). After closure, the closed target stays held.
Negative finite elapsed times select the seed and zero lift.
"""
    dt = _scalar(dt, 'dt', positive=True)
    close_s = _scalar(close_s, 'close_s', positive=True)
    lift_start_s = _scalar(lift_start_s, 'lift_start_s')
    speed = _scalar(lift_speed_m_s, 'lift_speed_m_s')
    height = _scalar(lift_height_m, 'lift_height_m')
    if lift_start_s < close_s or not 0 <= speed <= .01 or not 0 <= height <= .05:
        raise ValueError('lift must start after closing, with speed<=.01m/s and height<=.05m')
    command = _joints(command, 'command')
    seed = _joints(seed, 'seed', command)
    closed = _joints(closed, 'closed', command)
    elapsed = torch.as_tensor(elapsed_s, device=command.device, dtype=command.dtype)
    if elapsed.shape != (command.shape[0],) or not torch.isfinite(elapsed).all():
        raise ValueError('elapsed_s must contain N finite times')
    fraction = (elapsed / close_s).clamp(0., 1.)
    goal = seed + (closed - seed) * fraction[:, None]
    low, high = _limits(command)
    next_command = (command + (goal - command).clamp(-.3 * dt, .3 * dt)).maximum(low).minimum(high)
    z_goal = ((elapsed - lift_start_s) * speed).clamp(0., height)
    return next_command, z_goal


def valid_readback(pose, velocity, q, qd, hand_load, full_pose, full_qd,
                   joint_failure, adapter_bits, clearance):
    """Return per-slot validity for an initial or post-reset physical readback.

Malformed shapes/dtypes/devices raise ValueError. Numeric faults return False
only for their own slot. Full link poses retain a final wxyz pose dimension7;
all intermediate link axes may be flattened. No force upper limit is imposed.
"""
    if (not isinstance(pose, torch.Tensor) or not pose.is_floating_point()
            or pose.ndim != 2 or pose.shape[1] != 7 or pose.shape[0] < 1):
        raise ValueError('pose must be a floating nonempty N x 7 tensor')
    n, device = pose.shape[0], pose.device
    fixed = ((pose, 'pose', (n, 7)), (velocity, 'velocity', (n, 6)),
             (q, 'q', (n, 12)), (qd, 'qd', (n, 12)),
             (hand_load, 'hand_load', (n, 2)),
             (joint_failure, 'joint_failure', (n,)), (adapter_bits, 'adapter_bits', (n,)),
             (clearance, 'clearance', (n, 2)))
    for value, name, shape in fixed:
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.device != device
                or value.is_complex()
                or (name not in ('joint_failure', 'adapter_bits') and not value.is_floating_point())):
            raise ValueError(f'{name} has the wrong shape, dtype or device')
    for value, name, minimum_ndim in ((full_pose, 'full_pose', 2), (full_qd, 'full_qd', 1)):
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or value.ndim < minimum_ndim or value.shape[0] != n or value.device != device
                or any(size < 1 for size in value.shape)
                or (name == 'full_pose' and value.shape[-1] != 7)):
            raise ValueError(f'{name} must be a floating nonempty batch on the pose device')
    result = torch.ones(n, device=device, dtype=torch.bool)
    for value in (pose, velocity, q, qd, hand_load, full_pose, full_qd,
                  joint_failure, adapter_bits, clearance):
        result &= torch.isfinite(value).reshape(n, -1).all(-1)
    object_unit = (pose[:, 3:].double().norm(dim=-1) - 1.).abs() <= 1e-3
    link_unit = ((full_pose[..., 3:].double().norm(dim=-1) - 1.).abs() <= 1e-3).reshape(n, -1).all(-1)
    result &= object_unit & link_unit & (hand_load >= 0).all(-1)
    result &= (joint_failure == 0) & (adapter_bits == 0) & (clearance.amin(-1) >= -.001)
    return result


class LiftDwell:
    """Current continuous physical lift qualification, plus historical metrics.

The cylinder has local-Y axis, .008m radius and .5m length, above a .30m
table. Quaternions are wxyz; a norm error greater than1e-5 invalidates a row.
Bad measurements/reset interrupt current success but preserve its history.
"""

    def __init__(self, n, device, height_m=.03, hold_s=.5):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError('positive candidate count required')
        self.height_m = _scalar(height_m, 'height_m', positive=True)
        self.required_hold_s = _scalar(hold_s, 'hold_s', positive=True)
        self.n = n
        self.hold_s = torch.zeros(n, device=device, dtype=torch.float64)
        self.device = self.hold_s.device
        self.longest_hold_s = torch.zeros_like(self.hold_s)
        self.maximum_height_m = torch.zeros_like(self.hold_s)
        self.bottom = torch.full_like(self.hold_s, float('nan'))
        self.last_good = torch.zeros(n, device=self.device, dtype=torch.bool)
        self.qualified = torch.zeros_like(self.last_good)
        self.ever_qualified = torch.zeros_like(self.last_good)

    @torch.no_grad()
    def update(self, pose, velocity, hand_load, valid, reset, dt):
        dt = _scalar(dt, 'dt', positive=True)
        for name, value, shape, boolean in (
            ('pose', pose, (self.n, 7), False),
            ('velocity', velocity, (self.n, 6), False),
            ('hand_load', hand_load, (self.n, 2), False),
            ('valid', valid, (self.n,), True),
            ('reset', reset, (self.n,), True),
        ):
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.device != self.device
                    or (value.dtype != torch.bool if boolean else not value.is_floating_point())):
                raise ValueError(f'{name} has the wrong shape, dtype or device')
        pose = pose.to(self.hold_s)
        velocity = velocity.to(self.hold_s)
        hand_load = hand_load.to(self.hold_s)
        finite = (torch.isfinite(pose).all(-1) & torch.isfinite(velocity).all(-1)
                  & torch.isfinite(hand_load).all(-1) & (hand_load >= 0).all(-1))
        quaternion = pose[:, 3:]
        norm = quaternion.norm(dim=-1)
        unit = torch.isfinite(norm) & ((norm - 1.).abs() <= 1e-5)
        quaternion = quaternion / norm[:, None].clamp_min(1e-12)
        qw, qx, qy, qz = quaternion.unbind(-1)
        axis_z = (2 * (qy * qz + qw * qx)).abs().clamp(0., 1.)
        bottom = pose[:, 2] - .25 * axis_z - .008 * (1 - axis_z.square()).clamp_min(0.).sqrt()
        speed = velocity[:, :3].norm(dim=-1)
        contact = hand_load.sum(-1)
        finite &= torch.isfinite(bottom) & torch.isfinite(speed) & torch.isfinite(contact)
        measured = finite & unit
        self.bottom.copy_(torch.where(measured, bottom, torch.full_like(bottom, float('nan'))))
        height = (bottom - .30).clamp_min(0.)
        usable = measured & valid & ~reset
        good = usable & (height >= self.height_m - 1e-12) & (speed < .03) & (contact > .05)
        self.hold_s.copy_(torch.where(good, self.hold_s + dt, 0.))
        self.last_good.copy_(good)
        self.qualified.copy_(good & (self.hold_s >= self.required_hold_s - 1e-12))
        self.ever_qualified |= self.qualified
        self.maximum_height_m.copy_(torch.maximum(self.maximum_height_m, torch.where(usable, height, 0.)))
        self.longest_hold_s.copy_(torch.maximum(self.longest_hold_s, self.hold_s))
        return self.qualified.clone()

    def summary(self):
        """Return strict JSON-compatible values, including before any update."""
        return dict(height_m=self.height_m, required_hold_s=self.required_hold_s,
                    hold_s=self.hold_s.cpu().tolist(), last_good=self.last_good.cpu().tolist(),
                    qualified=self.qualified.cpu().tolist(), ever_qualified=self.ever_qualified.cpu().tolist(),
                    maximum_height_m=self.maximum_height_m.cpu().tolist(),
                    longest_hold_s=self.longest_hold_s.cpu().tolist(),
                    bottom=[v if math.isfinite(v) else None for v in self.bottom.cpu().tolist()])
