"""Stateless Torch feedback from measured FK and real object normal loads.

FK reference-point proximity is geometry, never evidence of physical contact.
All rewards are rates; the caller adds ``rate * dt`` exactly once. This module
does not prescribe a finger pose, contact location, wrist orientation or stage.
"""
import math

import torch


_DIGITS = ('left_thumb', 'left_index', 'right_thumb', 'right_index')
_SHAPES = {
    'pad_relative': (4, 3), 'object_quat': (4,), 'bottom': (),
    'grouped_object_loads': (2, 6), 'hand_object_loads': (2,),
    'object_velocity': (6,),
}

CONTRACT = dict(
    version='dexhands_measured_feedback_v3', feature_size=20,
    feature_order=(
        [f'{d}_abs_finite_cylinder_sdf_div_0.008' for d in _DIGITS]
        + [f'{d}_radial_world_z_div_0.008' for d in _DIGITS]
        + [f'{d}_axial_projection_div_half_length' for d in _DIGITS]
        + [f'{d}_log1p_normal_load_n' for d in _DIGITS]
        + [f'{d}_measured_bearing_gt_0.05n' for d in _DIGITS]
    ),
    input_shapes={key: ['N', *shape] for key, shape in _SHAPES.items()},
    optional_input_shapes={'wrist_velocity': ['N', 2, 6]},
    quaternion='wxyz_max_abs_scaled_normalization_then_rotated_local_Y_axis',
    grouped_load_order=['thumb', 'index', 'middle', 'ring', 'little', 'palm'],
    pad_order=list(_DIGITS),
    geometry_evidence='real_FK_proximity_only_not_measured_contact',
    normal_load_evidence='measured_object_normal_loads_friction_excluded',
    distance_scale_m=.008, axial_scale='length_m/2',
    default_radius_m=.008, default_length_m=.5,
    bearing_threshold_n=.05, contact_saturation_n=1.,
    reach='3*mean(exp(-abs_sdf/0.008))+3*min(exp(-abs_sdf/0.008))',
    contact='3*mean(clamp(each_digit_load/1N,0,1))+min(clamp(each_digit_load/1N,0,1))',
    table_top_m=.30, lift_deadzone_m=.0005, lift_height_scale_m=.03,
    lift_load_gate='min_hands(clamp((normal_load_n-0.05)/(1-0.05),0,1))',
    lift='60*clamp((actual_bottom_world_z-0.3005)/0.03,0,1)*lift_load_gate',
    requested_force_range_n=[2., 12.], requested_force_shape=['N', 2],
    force='-4*mean_hands(clamp(abs(actual_normal_load-target)/target,0,1))',
    balance_max_min_ratio=1.5,
    balance_load_scale_n=2.,
    balance='-2*mean_hands(clamp((abs(thumb-index)/max(thumb,index)-1/3)/(2/3),0,1)*clamp(max_digit_load/2N,0,1));zero_pair=0',
    action_rate='-0.5*mean(clamp((actions-previous_actions)/2,-1,1)^2);missing_either=0',
    rate_bounds_per_s={
        'dex_reach': [0., 6.], 'dex_contact': [0., 4.], 'dex_lift': [0., 60.],
        'dex_force': [-4., 0.], 'dex_balance': [-2., 0.], 'dex_action_rate': [-.5, 0.],
    },
    dt_application='caller_multiplies_rates_by_dt_once',
    integration='preserve_existing_signed_force_terms;disable_old_lift_and_legacy_grasp_shaping_in_this_mode',
    stage_flow=None, pose_latch=False, wrist_height_reward=False,
    physics_changed=False,
)


def _tensor(value, name, shape, device=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f'{name} must be a floating Torch tensor')
    if tuple(value.shape) != shape:
        raise ValueError(f'{name} must have shape {shape}, got {tuple(value.shape)}')
    if device is not None and value.device != device:
        raise ValueError(f'{name} must be on device {device}')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')
    return value


def _validate(m):
    if 'pad_relative' not in m:
        raise ValueError('missing pad_relative')
    pads = m['pad_relative']
    if not isinstance(pads, torch.Tensor) or pads.ndim != 3:
        raise ValueError('pad_relative must have shape (N, 4, 3)')
    n = pads.shape[0]
    for key, tail in _SHAPES.items():
        if key not in m:
            raise ValueError(f'missing {key}')
        _tensor(m[key], key, (n, *tail), pads.device)
    if 'wrist_velocity' in m:
        _tensor(m['wrist_velocity'], 'wrist_velocity', (n, 2, 6), pads.device)
    for key in ('grouped_object_loads', 'hand_object_loads'):
        if (m[key] < 0.).any():
            raise ValueError(f'{key} must contain nonnegative normal loads')
    q = m['object_quat']
    scale = q.abs().amax(dim=-1, keepdim=True)
    if (scale == 0.).any():
        raise ValueError('object_quat must be nonzero')
    # Scale first: direct squared norm over/underflows for valid finite q.
    q = q / scale
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def _cylinder_geometry(m, q, radius, length):
    w, x, y, z = q.unbind(-1)
    axis = torch.stack((2.*(x*y-w*z), 1.-2.*(x*x+z*z), 2.*(y*z+w*x)), -1)
    relative = m['pad_relative']
    axial = (relative * axis[:, None, :]).sum(-1)
    radial = relative - axial[..., None] * axis[:, None, :]
    rho = torch.linalg.vector_norm(radial, dim=-1)
    d = torch.stack((rho-radius, axial.abs()-length/2.), -1)
    # Exact finite-cylinder SDF, including both endcaps and exterior corners.
    sdf = torch.linalg.vector_norm(d.clamp_min(0.), dim=-1) + d.amax(-1).clamp_max(0.)
    return sdf.abs(), radial[..., 2], axial


def _digit_loads(m):
    return m['grouped_object_loads'][:, :, :2].reshape(-1, 4)


def geometry_features(m, radius_m=.008, length_m=.5):
    """Return [N,20] measured features in the order specified in CONTRACT.

    Positions are world vectors relative to the cylinder center. Its axis is
    local Y rotated by the robustly normalized object wxyz quaternion. Distances
    and radial world-Z always use the fixed .008 m scale, even for custom sizes.
    The last four flags are *measured normal bearing*, strictly above .05 N.
    """
    for name, value in (('radius_m', radius_m), ('length_m', length_m)):
        if not math.isfinite(value) or value <= 0.:
            raise ValueError(f'{name} must be finite and positive')
    q = _validate(m)
    distance, radial_z, axial = _cylinder_geometry(m, q, radius_m, length_m)
    loads = _digit_loads(m)
    return torch.cat((distance/.008, radial_z/.008, axial/(length_m/2.),
                      torch.log1p(loads), (loads > .05).to(distance.dtype)), -1)


def reward_rates(m, requested_force, actions=None, previous_actions=None):
    """Return six finite [N] rates, without dt scaling or hidden history.

    requested_force is [N,2], 2--12 N per hand. Optional actions are floating
    [N,A] tensors with A>0; no action-change cost is charged until both current
    and previous actions are supplied. Optional wrist_velocity is validated,
    but wrist motion supplies no object-height credit. No inputs are mutated.
    """
    q = _validate(m)
    hand = m['hand_object_loads']
    n = hand.shape[0]
    target = _tensor(requested_force, 'requested_force', (n, 2), hand.device)
    if ((target < 2.) | (target > 12.)).any():
        raise ValueError('requested_force must be in the inclusive 2--12 N range')
    for name, value in (('actions', actions), ('previous_actions', previous_actions)):
        if value is not None:
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] == 0:
                raise ValueError(f'{name} must have shape (N, A) with A > 0')
            _tensor(value, name, (n, value.shape[1]), hand.device)
    if actions is not None and previous_actions is not None and actions.shape != previous_actions.shape:
        raise ValueError('actions and previous_actions must have matching shapes')

    distance, _, _ = _cylinder_geometry(m, q, .008, .5)
    proximity = torch.exp(-distance/.008)
    digit = _digit_loads(m)
    reach = 3.*proximity.mean(1) + 3.*proximity.amin(1)
    bearing=digit.clamp(0.,1.)
    contact = 3.*bearing.mean(1)+bearing.amin(1)
    gate = ((hand-.05)/.95).clamp(0., 1.).amin(1)
    lift = 60.*((m['bottom']-.3005)/.03).clamp(0., 1.)*gate
    force = -4.*((hand-target).abs()/target).clamp(0., 1.).mean(1)
    pair = digit.reshape(n, 2, 2)
    stronger = pair.amax(-1)
    # Replacing only an exactly zero denominator preserves arbitrarily small
    # positive measured loads while assigning an empty pair zero imbalance.
    denominator = torch.where(stronger > 0., stronger, torch.ones_like(stronger))
    imbalance = (pair[..., 0]-pair[..., 1]).abs()/denominator
    # abs(a-b)/max(a,b) > 1/3 is exactly max(a,b)/min(a,b) > 1.5.
    balance = -2.*(((imbalance-1./3.)/(2./3.)).clamp(0.,1.)*(stronger/2.).clamp(0.,1.)).mean(1)
    action_rate = torch.zeros_like(lift)
    if actions is not None and previous_actions is not None:
        delta = ((actions-previous_actions)/2.).clamp(-1., 1.)
        action_rate = -.5*delta.square().mean(1)
    return dict(dex_reach=reach, dex_contact=contact, dex_lift=lift,
                dex_force=force, dex_balance=balance, dex_action_rate=action_rate)
