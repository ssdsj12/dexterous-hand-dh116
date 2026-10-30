"""Pure reference-point geometry; these helpers do not qualify a grasp."""
import math
from numbers import Real

import torch

from .linear_coupling import active_limits


def _points(value, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.layout != torch.strided):
        raise ValueError(f'{name} must be a dense floating tensor')
    if value.ndim < 2 or value.shape[-1] != 3 or any(d == 0 for d in value.shape):
        raise ValueError(f'{name} must have nonempty shape [N, ..., 3]')


def _matching(first, second):
    if first.dtype != second.dtype:
        raise ValueError('inputs must have the same dtype')
    if first.device != second.device:
        raise ValueError('inputs must be on the same device')
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError('inputs must be finite')


def reflect_y(points, center):
    """Reflect [N, ..., 3] points in world Y about each [N, 3] center.

    Inputs must be finite floating tensors with matching dtype/device. X and Z
    are preserved exactly; inputs are not mutated. This is a point reflection,
    not a rotation to apply to an orientation or quaternion.
    """
    _points(points, 'points')
    _points(center, 'center')
    if center.shape != (points.shape[0], 3):
        raise ValueError('center must have shape [N, 3] matching points')
    _matching(points, center)
    cy = center[:, 1].double().reshape((points.shape[0],) + (1,) * (points.ndim - 2))
    result = points.clone()
    result[..., 1] = (cy - (points[..., 1].double() - cy)).to(points.dtype)
    if not torch.isfinite(result).all():
        raise ValueError('reflected points must be finite in the input dtype')
    return result


def proper_rigid_fit(source, target):
    """Return R[N,3,3], t[N,3] minimizing ||source @ R.T + t - target||.

    Corresponding inputs must have identical [N,K,3] shape, K >= 3, and finite
    floating values with matching dtype/device. Both centered point clouds must
    have rank >= 2 in every slot; a bad slot raises ValueError for the batch.
    Rank uses a relative tolerance max(K, 3) * input-dtype epsilon. Calculations
    use float64 and outputs retain the input dtype/device. Kabsch correction
    enforces det(R)=+1, including when target points were reflected; a mirror
    therefore need not admit an exact fit. No scale factor is fitted.
    """
    _points(source, 'source')
    _points(target, 'target')
    if source.ndim != 3 or source.shape != target.shape or source.shape[1] < 3:
        raise ValueError('source and target must have matching [N, K, 3] shapes with K >= 3')
    _matching(source, target)
    source64, target64 = source.double(), target.double()
    source_mean, target_mean = source64.mean(1), target64.mean(1)
    x, y = source64 - source_mean[:, None], target64 - target_mean[:, None]
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('centered point clouds must be finite')
    rank_tolerance = max(source.shape[1], 3) * torch.finfo(source.dtype).eps
    for centered in (x, y):
        singular = torch.linalg.svdvals(centered)
        if (singular[:, 1] <= singular[:, 0] * rank_tolerance).any():
            raise ValueError('each centered cloud must have rank >= 2 (noncollinear points)')
    covariance = x.transpose(-1, -2) @ y
    if not torch.isfinite(covariance).all():
        raise ValueError('point covariance must be finite')
    u, _, vh = torch.linalg.svd(covariance)
    v, ut = vh.transpose(-1, -2), u.transpose(-1, -2)
    correction = torch.ones_like(source_mean)
    correction[:, -1] = torch.where(torch.linalg.det(v @ ut) < 0, -1., 1.)
    rotation = (v * correction[:, None, :]) @ ut
    translation = target_mean - (rotation @ source_mean[..., None]).squeeze(-1)
    rotation, translation = rotation.to(source.dtype), translation.to(source.dtype)
    if not torch.isfinite(rotation).all() or not torch.isfinite(translation).all():
        raise ValueError('fit outputs must be finite in the input dtype')
    return rotation, translation


def reframe_transforms(transforms, source_frame, target_frame):
    """Return target_frame @ inverse(source_frame) @ transforms.

    Transforms have nonempty shape [N, ..., 4, 4]; frames have shape [N,4,4].
    The frame change broadcasts across all intermediate transform axes. Inputs
    must be finite proper rigid homogeneous matrices, with matching floating
    dtype/device. Rotation orthogonality, determinant +1 and homogeneous last
    rows use absolute tolerance max(1e-5, 8 * dtype epsilon). Reflections, scale
    and shear outside that tolerance raise ValueError, as does any invalid slot.
    Calculations use float64; outputs retain the transform shape/dtype/device.
    Inputs are not mutated. Object-relative positions AND orientations persist.
    """
    matrices = ((transforms, 'transforms'), (source_frame, 'source_frame'),
                (target_frame, 'target_frame'))
    for value, name in matrices:
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or value.layout != torch.strided):
            raise ValueError(f'{name} must be a dense floating tensor')
        if (value.ndim < 3 or value.shape[-2:] != (4, 4)
                or any(d == 0 for d in value.shape)):
            raise ValueError(f'{name} must have nonempty shape [N, ..., 4, 4]')
    frame_shape = (transforms.shape[0], 4, 4)
    if source_frame.shape != frame_shape or target_frame.shape != frame_shape:
        raise ValueError('source_frame and target_frame must have matching shape [N, 4, 4]')
    _matching(transforms, source_frame)
    _matching(transforms, target_frame)
    tolerance = max(1e-5, 8 * torch.finfo(transforms.dtype).eps)
    identity = torch.eye(3, dtype=torch.float64, device=transforms.device)
    last_row = transforms.new_tensor([0., 0., 0., 1.]).double()
    for value, name in matrices:
        value64 = value.double()
        rotation = value64[..., :3, :3]
        if ((value64[..., 3, :] - last_row).abs() > tolerance).any():
            raise ValueError(f'{name} must have homogeneous last row [0, 0, 0, 1]')
        if (((rotation.transpose(-1, -2) @ rotation - identity).abs() > tolerance).any()
                or ((torch.linalg.det(rotation) - 1).abs() > tolerance).any()):
            raise ValueError(f'{name} must contain proper rigid rotations (det=+1)')
    change = target_frame.double() @ torch.linalg.inv(source_frame.double())
    broadcast = (transforms.shape[0],) + (1,) * (transforms.ndim - 3) + (4, 4)
    result = (change.reshape(broadcast) @ transforms.double()).to(transforms.dtype)
    if not torch.isfinite(result).all():
        raise ValueError('reframed transforms must be finite in the input dtype')
    return result


def gravity_cylinder_frame(center, axis):
    """Build proper [N,4,4] cylinder frames without axial body roll.

    Center and axis are finite floating [N,3] tensors of matching dtype/device;
    each axis must be nonzero. Local y follows the normalized axis. With world
    Z as up (world X when abs(dot(y,Z)) > .99), x=normalize(cross(y,up)) and
    z=cross(x,y). Rotation columns are [x,y,z], and the origin is center.
    Computation uses float64; output retains input dtype/device. This convention
    is discontinuous at the fallback threshold and is intended for seed frames.
    """
    _points(center, 'center')
    _points(axis, 'axis')
    if center.ndim != 2 or axis.shape != center.shape:
        raise ValueError('center and axis must have matching shape [N, 3]')
    _matching(center, axis)
    axis64 = axis.double()
    scale = axis64.abs().amax(-1, keepdim=True)
    if (scale == 0).any():
        raise ValueError('each cylinder axis must be nonzero')
    y = axis64 / scale
    y = y / y.norm(dim=-1, keepdim=True)
    up = torch.zeros_like(y)
    up[:, 2] = 1.
    up = torch.where((y[:, 2].abs() > .99)[:, None], up.new_tensor([1., 0., 0.]), up)
    x = torch.linalg.cross(y, up, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.linalg.cross(x, y, dim=-1)
    result = torch.eye(4, dtype=center.dtype, device=center.device)[None].repeat(len(center), 1, 1)
    result[:, :3, :3] = torch.stack((x, y, z), dim=-1).to(center.dtype)
    result[:, :3, 3] = center
    return result


def manifold_depth(separation, count, start):
    """Per-environment penetration from valid native contact buffer slices.

    Positive separation is a gap. Empty slices return zero observed depth,
    which is not proof that the complete collision geometry is penetration-free.
    Padding is ignored, including nonfinite padding. Intended for the small
    diagnostic batches: temporary storage is N times contact buffer capacity.
    """
    if (not isinstance(separation, torch.Tensor) or not separation.is_floating_point()
            or separation.layout != torch.strided or separation.ndim != 1):
        raise ValueError('separation must be a dense floating vector')
    for value in (count, start):
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.dtype not in (torch.int32, torch.int64) or value.ndim != 1
                or value.numel() == 0 or value.device != separation.device):
            raise ValueError('contact count/start must be nonempty integer vectors on the same device')
    if (start.shape != count.shape or (count < 0).any() or (start < 0).any()
            or (start + count > separation.numel()).any()):
        raise ValueError('contact slices must lie within the separation buffer')
    if separation.numel() == 0:
        return separation.new_zeros(count.shape)
    index = torch.arange(separation.numel(), device=separation.device)[None]
    valid = (index >= start[:, None]) & (index < (start + count)[:, None])
    selected = torch.where(valid, separation[None], float('inf'))
    if (~torch.isfinite(separation)[None] & valid).any():
        raise ValueError('active contact separation must be finite')
    return (-selected.amin(-1)).clamp_min(0)


def contact_seek(command, seed, digit_loads, digit_depth, dt, *, rate_rad_s=.02,
                 min_contact_n=.1, max_depth_m=.0008):
    """Seek contact independently at active coordinates [[1,2],[7,8]].

    Command/seed are feasible finite [N,12] targets. Loads and nonnegative
    penetration depths have shape [N,2,2] in hand then thumb/index order. All
    tensors share floating dtype/device. Depth > max_depth_m always backs off;
    load < min_contact_n AND depth < max_depth_m closes; otherwise hold. There
    is no upper force threshold or force setpoint.

    Each selected target moves at most rate_rad_s * dt, within active_limits()
    and [seed-.03, seed+.12]. Selected input commands must already lie in this
    envelope, otherwise ValueError avoids an instantaneous out-of-rate clamp.
    AB and other fingers are returned unchanged, even outside that envelope.
    Inputs are not mutated; this function has no state or physical side effects.
    """
    controls = ((dt, 'dt'), (rate_rad_s, 'rate_rad_s'),
                (min_contact_n, 'min_contact_n'), (max_depth_m, 'max_depth_m'))
    for value, name in controls:
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    if rate_rad_s > .3:
        raise ValueError('rate_rad_s must not exceed .3 rad/s')
    values = ((command, 'command'), (seed, 'seed'),
              (digit_loads, 'digit_loads'), (digit_depth, 'digit_depth'))
    for value, name in values:
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or value.layout != torch.strided):
            raise ValueError(f'{name} must be a dense floating tensor')
    if command.ndim != 2 or command.shape[0] == 0 or command.shape[1] != 12:
        raise ValueError('command must have nonempty shape [N,12]')
    for value, name in values:
        expected = command.shape if name in ('command', 'seed') else (len(command), 2, 2)
        if value.shape != expected:
            raise ValueError(f'{name} must have shape {tuple(expected)}')
        _matching(command, value)
    low, high = (command.new_tensor(a).repeat(2) for a in active_limits())
    if (((command < low) | (command > high)).any()
            or ((seed < low) | (seed > high)).any()):
        raise ValueError('command and seed must lie within active_limits')
    if (digit_loads < 0).any() or (digit_depth < 0).any():
        raise ValueError('digit loads and depths must be nonnegative')
    selected = [1, 2, 7, 8]
    lower = (seed[:, selected] - .03).maximum(low[selected])
    upper = (seed[:, selected] + .12).minimum(high[selected])
    current = command[:, selected]
    if ((current < lower) | (current > upper)).any():
        raise ValueError('selected command targets must lie within the seed search envelope')
    close = (digit_loads < min_contact_n) & (digit_depth < max_depth_m)
    backoff = digit_depth > max_depth_m
    direction = close.to(torch.float64) - backoff.to(torch.float64)
    next_selected = current.double() + direction.reshape(-1, 4) * float(rate_rad_s) * float(dt)
    next_selected = next_selected.maximum(lower.double()).minimum(upper.double())
    result = command.clone()
    result[:, selected] = next_selected.to(command.dtype)
    return result
