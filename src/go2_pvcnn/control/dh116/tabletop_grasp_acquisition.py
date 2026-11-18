"""Stateless potential shaping from measured rod geometry and bilateral loads.

Pads 0:2 belong to the left hand and 2:4 to the right. Relative pad
positions are world-frame offsets from the object center; object_quat is
the actual local-to-world wxyz quaternion. No simulator or action dependency.
"""

import math
from collections.abc import Mapping

import torch


CONTRACT = dict(
    version='finite_cylinder_grasp_acquisition_v2',
    gamma=.995,
    potential='10*mean_pad_proximity+10*min_hand_proximity+20*clamped_min_hand_load',
    proximity_scale_m=.015,
    load_scale_n=1.,
    object_local_axis='Y',
    quaternion_order='wxyz',
    pad_hand_order='left_left_right_right',
    terminal_next_potential=0.,
    rollout_boundary='preserve_potential',
    reset_potential='compute_from_current_measured_state',
    changes_success_or_guard=False,
)


def _real_tensor(value, name):
    if not isinstance(value, torch.Tensor) or value.is_complex() or value.dtype == torch.bool:
        raise ValueError(f'{name} must be a real numeric tensor')
    value = value.to(dtype=torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must contain only finite values')
    return value


def _scalar(value, name, *, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite scalar') from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return result


def potential(measurement, radius_m=.008, length_m=.5):
    """Return an (N,) float64 potential on the measurement tensors' device.

    Distance is abs(SDF) to a *finite* cylinder, so penetration also costs
    proximity. The load contribution saturates at 1 N on both hands and
    cannot be earned with only one loaded hand. No temporal state is stored.
    """
    radius = _scalar(radius_m, 'radius_m', positive=True)
    length = _scalar(length_m, 'length_m', positive=True)
    names = ('pad_relative', 'object_quat', 'hand_object_loads')
    if not isinstance(measurement, Mapping) or any(k not in measurement for k in names):
        raise ValueError(f'measurement must contain {names}')
    pads, quat, loads = [_real_tensor(measurement[k], k) for k in names]
    if pads.ndim != 3 or pads.shape[1:] != (4, 3):
        raise ValueError('pad_relative must have shape (N, 4, 3)')
    n = pads.shape[0]
    if quat.shape != (n, 4) or loads.shape != (n, 2):
        raise ValueError('object_quat and hand_object_loads must have shapes (N, 4) and (N, 2)')
    if quat.device != pads.device or loads.device != pads.device:
        raise ValueError('measurement tensors must share a device')
    # Scale before normalizing: even very large/small finite nonzero quats work.
    scale = quat.abs().amax(dim=-1, keepdim=True)
    if (scale == 0).any():
        raise ValueError('object_quat must have nonzero norm')
    quat = quat / scale
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    inverse_xyz = -quat[:, None, 1:].expand(-1, 4, -1)
    twice_cross = 2 * torch.linalg.cross(inverse_xyz, pads, dim=-1)
    local = pads + quat[:, None, :1] * twice_cross + torch.linalg.cross(inverse_xyz, twice_cross, dim=-1)
    radial = torch.linalg.vector_norm(local[..., (0, 2)], dim=-1) - radius
    axial = local[..., 1].abs() - length / 2
    delta = torch.stack((radial, axial), dim=-1)
    sdf = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1) + delta.amax(dim=-1).clamp_max(0)
    proximity = torch.exp(-sdf.abs() / .015)
    hands = proximity.reshape(n, 2, 2).mean(dim=-1)
    bilateral_load = loads.amin(dim=-1).clamp(0, 1)
    return 10 * proximity.mean(dim=-1) + 10 * hands.amin(dim=-1) + 20 * bilateral_load


def reward(before, after, done, gamma=.995):
    """Return gamma * where(done, 0, after) - before, without rollout resets.

    Compute both potentials from actual states, before any automatic reset.
    Terminal next potentials are zero even when the terminal state is loaded.
    """
    gamma = _scalar(gamma, 'gamma')
    if not 0 <= gamma <= 1:
        raise ValueError('gamma must lie in [0, 1]')
    before = _real_tensor(before, 'before')
    after = _real_tensor(after, 'after')
    if before.ndim != 1 or after.shape != before.shape:
        raise ValueError('before and after must have identical (N,) shapes')
    if not isinstance(done, torch.Tensor) or done.dtype != torch.bool or done.shape != before.shape:
        raise ValueError('done must be a boolean tensor with shape (N,)')
    if after.device != before.device or done.device != before.device:
        raise ValueError('before, after and done must share a device')
    return gamma * torch.where(done, 0., after) - before
