"""Pure geometry targets from measured simulated pressure, without force prediction.

This helper does not execute actions, modify forces, acquire missing contacts or
read a time-indexed Teacher trajectory. Its current actual contact points are the
only path origin. The caller supplies independently audited per-style load goals.
"""
import math

import torch


REASON_CODES = dict(eligible=0, target_disabled=1, no_bearing_contact=2,
    zero_force_vector=3, pad_invalid=4, negative_measurement=5,
    nonfinite_measurement=6)


@torch.no_grad()
def pressure_reference_targets(points, vectors_on_hand, loads, valid, target_loads, *,
        horizon, dt=1 / 60, gain_m_per_n_s=.0004, max_speed_m_s=.001,
        effective_depth_m=None, retreat_depth_m=.0006):
    """Return desired material-point positions for future steps 1 through H.

    Numeric inputs share a floating dtype/device and have shapes [N,2,5,3] for
    points/force vectors, [N,2,5] otherwise. ``valid`` is bool [N,2,5]. For an
    observed bearing pad (load > .05 N), inward is the negative unit force ON
    HAND. Velocity is gain * (target - load), limited to +/- max_speed_m_s.
    Actual effective depth > retreat_depth_m requests the historical 1 mm/s
    retreat, also respecting a caller's lower speed limit. This is a controller
    reference rule, not a new simulator contact-depth constraint.

    ``validity`` describes usable finite measured geometry/force; ``eligible``
    additionally requires bearing load and a positive target. Bad pads cannot
    influence other hands/environments. A nonfinite point has a zero placeholder
    and zero weight; it is never an invented contact. Other ineligible points
    remain fixed. ``weights`` are binary [N,H,2,5], and ``reason_code`` [N,2,5]
    uses the returned dictionary. All returned tensors own their values.
    """
    if type(horizon) is not int or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    for name, value in (('dt', dt), ('gain_m_per_n_s', gain_m_per_n_s),
                        ('max_speed_m_s', max_speed_m_s), ('retreat_depth_m', retreat_depth_m)):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (name != 'retreat_depth_m' and value == 0)):
            raise ValueError(name + ' must be finite and positive (depth may be zero)')
    if max_speed_m_s > .001:
        raise ValueError('pressure reference speed cannot exceed .001 m/s')
    if (not isinstance(points, torch.Tensor) or not points.is_floating_point()
            or points.ndim != 4 or tuple(points.shape[1:]) != (2, 5, 3)
            or points.shape[0] < 1):
        raise ValueError('floating points [N,2,5,3] required')
    try:
        duration = horizon * dt
    except OverflowError as error:
        raise ValueError('horizon duration must be representable in the points dtype') from error
    if not math.isfinite(duration) or duration > torch.finfo(points.dtype).max:
        raise ValueError('horizon duration must be representable in the points dtype')
    shape = points.shape[:-1]
    numeric = [('vectors_on_hand', vectors_on_hand, points.shape),
        ('loads', loads, shape), ('target_loads', target_loads, shape)]
    if effective_depth_m is not None:
        numeric.append(('effective_depth_m', effective_depth_m, shape))
    for name, value, expected in numeric:
        if (not isinstance(value, torch.Tensor) or value.shape != expected
                or value.dtype != points.dtype or value.device != points.device):
            raise ValueError(name + ' requires matching shape, floating dtype and device')
    if (not isinstance(valid, torch.Tensor) or valid.shape != shape
            or valid.dtype != torch.bool or valid.device != points.device):
        raise ValueError('boolean valid [N,2,5] on the points device required')

    point_finite = torch.isfinite(points).all(-1)
    finite = (point_finite & torch.isfinite(vectors_on_hand).all(-1)
        & torch.isfinite(loads) & torch.isfinite(target_loads))
    nonnegative = (loads >= 0) & (target_loads >= 0)
    if effective_depth_m is not None:
        finite &= torch.isfinite(effective_depth_m)
        nonnegative &= effective_depth_m >= 0
    # Replacing unusable rows is purely a numerical guard. Masks stay false;
    # neither a substitute normal nor a positive acquisition weight is created.
    force = torch.where(finite[..., None], vectors_on_hand, 0.)
    norm = torch.linalg.vector_norm(force, dim=-1)
    finite &= torch.isfinite(norm)
    has_direction = norm > 0
    validity = finite & nonnegative & valid & has_direction
    eligible = validity & (loads > .05) & (target_loads > 0)
    denominator = torch.where(eligible, norm, 1.)
    inward = torch.where(eligible[..., None], -force / denominator[..., None], 0.)
    error = torch.where(eligible, target_loads - loads, 0.)
    speed = (gain_m_per_n_s * error).clamp(-max_speed_m_s, max_speed_m_s)
    if effective_depth_m is not None:
        speed = torch.where(eligible & (effective_depth_m > retreat_depth_m),
            -min(.001, max_speed_m_s), speed)
    speed = torch.where(eligible, speed, 0.)
    origin = torch.where(point_finite[..., None], points, 0.)
    time = torch.arange(1, horizon + 1, dtype=points.dtype, device=points.device) * dt
    target = origin[:, None] + time[None, :, None, None, None] * (inward * speed[..., None])[:, None]
    weights = eligible[:, None].expand(-1, horizon, -1, -1).to(points.dtype).clone()
    reason = torch.full(shape, REASON_CODES['eligible'], dtype=torch.long, device=points.device)
    # Later assignments take precedence: corrupt data is not reported as a
    # legitimate absence of contact, and a disabled target is not an error.
    for mask, name in ((target_loads <= 0, 'target_disabled'),
            (loads <= .05, 'no_bearing_contact'), (~has_direction, 'zero_force_vector'),
            (~valid, 'pad_invalid'), (~nonnegative, 'negative_measurement'),
            (~finite, 'nonfinite_measurement')):
        reason = torch.where(mask, REASON_CODES[name], reason)
    return dict(target_points=target, weights=weights, eligible=eligible,
        validity=validity, reason_code=reason, reason_codes=REASON_CODES.copy(),
        speed_m_s=speed, inward=inward, mode='geometry_velocity_from_simulated_force',
        force_prediction=False)
