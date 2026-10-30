"""Differentiable DH116 surface geometry costs, in world coordinates/meters.

Geometry is a sampling approximation, never evidence of measured contact,
normal load, or physical grasp success. This module has no simulator, actuator,
time-step, joint-count, or external-wrist dependencies.
"""
import math
from numbers import Integral, Real

import torch


def _tensor(value, name, shape=None, reference=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f'{name} must be a floating Torch tensor')
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}, got {tuple(value.shape)}')
    if reference is not None:
        if value.device != reference.device:
            raise ValueError(f'{name} must be on device {reference.device}')
        if value.dtype != reference.dtype:
            raise ValueError(f'{name} must have dtype {reference.dtype}')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')
    return value


def _scalar(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or (positive and value <= 0.)):
        qualifier = 'finite and positive' if positive else 'finite'
        raise ValueError(f'{name} must be a {qualifier} real scalar')


def _unit(value, name):
    # Scaling before taking a norm avoids overflow/underflow for finite inputs.
    scale = value.abs().amax(dim=-1, keepdim=True)
    if (scale == 0.).any():
        raise ValueError(f'{name} must contain nonzero vectors')
    scaled = value / scale
    return scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)


def _pose(points, center, quaternion, radius_m, length_m):
    _tensor(points, 'points')
    if points.ndim < 2 or points.shape[-1] != 3 or any(s == 0 for s in points.shape):
        raise ValueError('points must have shape (N, ..., 3) with nonempty dimensions')
    n = points.shape[0]
    _tensor(center, 'center', (n, 3), points)
    _tensor(quaternion, 'quaternion', (n, 4), points)
    _scalar(radius_m, 'radius_m', positive=True)
    _scalar(length_m, 'length_m', positive=True)
    return _unit(quaternion, 'quaternion')


def _geometry(points, center, q, radius_m, length_m, normals=False):
    """Cylinder SDF and optional analytic outward normal; local axis is Y."""
    w, x, y, z = q.unbind(-1)
    axis = torch.stack((2. * (x*y - w*z), 1. - 2. * (x*x + z*z),
                        2. * (y*z + w*x)), dim=-1)
    broadcast = (points.shape[0],) + (1,) * (points.ndim - 2) + (3,)
    axis = axis.reshape(broadcast)
    relative = points - center.reshape(broadcast)
    axial = (relative * axis).sum(-1)
    radial = relative - axial.unsqueeze(-1) * axis
    rho = torch.linalg.vector_norm(radial, dim=-1)
    side = rho - radius_m
    cap = axial.abs() - length_m / 2.
    delta = torch.stack((side, cap), dim=-1)
    positive = delta.clamp_min(0.)
    outside_distance = torch.linalg.vector_norm(positive, dim=-1)
    sdf = outside_distance + delta.amax(-1).clamp_max(0.)
    if not normals:
        return sdf

    # The SDF is nonsmooth on the medial axis and the surface rim. Choose local
    # +X on the radial axis, +Y at axial zero, and the side at side/cap ties.
    local_x = torch.stack((1. - 2. * (y*y + z*z), 2. * (x*y + w*z),
                           2. * (x*z - w*y)), dim=-1).reshape(broadcast)
    radial_denominator = torch.where(rho > 0., rho, torch.ones_like(rho))
    side_normal = torch.where((rho > 0.).unsqueeze(-1),
                              radial / radial_denominator.unsqueeze(-1), local_x)
    cap_normal = torch.where((axial >= 0.).unsqueeze(-1), axis, -axis)
    outside_denominator = torch.where(outside_distance > 0., outside_distance,
                                      torch.ones_like(outside_distance))
    outside_normal = (positive[..., :1] * side_normal + positive[..., 1:] * cap_normal)
    outside_normal = outside_normal / outside_denominator.unsqueeze(-1)
    inside_normal = torch.where((side >= cap).unsqueeze(-1), side_normal, cap_normal)
    outward = torch.where((outside_distance > 0.).unsqueeze(-1), outside_normal, inside_normal)
    return sdf, outward


def finite_cylinder_sdf(points, center, quaternion, radius_m=.008, length_m=.5):
    """Return signed distance in meters, shape ``points.shape[:-1]``.

    ``points`` is [N,...,3], ``center`` [N,3], and ``quaternion`` [N,4] wxyz,
    rotating cylinder local coordinates to world coordinates. Local Y is the
    cylinder axis; length is the full cap-to-cap length. Distance is negative
    inside, zero on the surface, positive outside (including beyond the caps).

    All tensors must be finite, floating, on one device and of one dtype, with
    nonempty dimensions. Nonzero quaternions are robustly normalized. Radius
    and length must be finite positive real scalars. Invalid inputs raise
    ValueError. Gradients are piecewise defined as for the exact cylinder SDF.
    """
    q = _pose(points, center, quaternion, radius_m, length_m)
    return _geometry(points, center, q, radius_m, length_m)


def surface_terms(patches, collision_points, center, quaternion, *,
                  patch_normals=None, table_z=.30, radius_m=.008, length_m=.5,
                  contact_points_per_patch=3, self_penetration_m=None,
                  object_hand_penetration_m=None):
    """Return nonnegative per-row geometry costs, with no dt scaling.

    Patches are [N,2,2,P,3] (hand, digit, sample); collision_points [N,K,3]
    samples the whole hand, including points used for table clearance. P,K,N
    must be positive. Pose, dtype, device and finiteness follow
    finite_cylinder_sdf. No input is changed or moved between devices.

    ``patch_distance_m`` [N,2,2] is each patch's mean absolute SDF over its
    nearest min(contact_points_per_patch,P) samples. Count is a positive
    integer. ``reach`` is the mean of all four patch distances / .008 m,
    regardless of the chosen cylinder radius. ``penetration`` is the maximum
    relu(-SDF(collision_points)) / .001 m. ``table`` is the maximum
    relu(table_z + .0003 - collision_points[...,2]) / .001 m.

    Optional patch_normals has exactly the patches shape, contains nonzero
    world-space hand-outward normals and is robustly normalized. For the same
    selected samples, ``normal_alignment`` is mean(1 + dot(hand_outward,
    object_outward)), clamped to [0,2] for rounding. Opposed normals cost zero.
    The object normal comes from the finite cylinder, including Y endcaps and
    exterior corners. At nondifferentiable locations we choose radial local
    +X on the axis, cap +Y at axial zero, and side normals at side/cap ties.

    Supplemental self_penetration_m and object_hand_penetration_m are optional
    nonnegative [N] depths, already reduced by the caller. Their respective
    ``self`` and ``object_in_hand`` costs divide by .001 m. Missing supplemental
    depths or normals yield zeros. Geometry costs do not assert real contact.
    """
    _tensor(patches, 'patches')
    if (patches.ndim != 5 or patches.shape[1:3] != (2, 2)
            or patches.shape[-1] != 3 or patches.shape[0] == 0 or patches.shape[3] == 0):
        raise ValueError('patches must have shape (N, 2, 2, P, 3), N,P > 0')
    n, _, _, p, _ = patches.shape
    _tensor(collision_points, 'collision_points', reference=patches)
    if (collision_points.ndim != 3 or collision_points.shape[0] != n
            or collision_points.shape[1] == 0 or collision_points.shape[2] != 3):
        raise ValueError('collision_points must have shape (N, K, 3), K > 0')
    if (isinstance(contact_points_per_patch, bool)
            or not isinstance(contact_points_per_patch, Integral) or contact_points_per_patch <= 0):
        raise ValueError('contact_points_per_patch must be a positive integer')
    _scalar(table_z, 'table_z')
    q = _pose(patches, center, quaternion, radius_m, length_m)

    unit_normals = None
    if patch_normals is not None:
        _tensor(patch_normals, 'patch_normals', patches.shape, patches)
        unit_normals = _unit(patch_normals, 'patch_normals')
    supplemental = {}
    for name, value, output in (
            ('self_penetration_m', self_penetration_m, 'self'),
            ('object_hand_penetration_m', object_hand_penetration_m, 'object_in_hand')):
        if value is None:
            supplemental[output] = patches.new_zeros(n)
        else:
            _tensor(value, name, (n,), patches)
            if (value < 0.).any():
                raise ValueError(f'{name} must contain nonnegative depths')
            supplemental[output] = value / .001

    if unit_normals is None:
        distance = _geometry(patches, center, q, radius_m, length_m)
    else:
        distance, outward = _geometry(patches, center, q, radius_m, length_m, normals=True)
    nearest, indices = distance.abs().topk(min(contact_points_per_patch, p), dim=-1, largest=False)
    patch_distance = nearest.mean(-1)
    normal_alignment = patches.new_zeros(n)
    if unit_normals is not None:
        alignment = (1. + (unit_normals * outward).sum(-1)).clamp(0., 2.)
        normal_alignment = alignment.gather(-1, indices).mean(dim=(1, 2, 3))
    collision_sdf = _geometry(collision_points, center, q, radius_m, length_m)
    return dict(
        reach=patch_distance.mean(dim=(1, 2)) / .008,
        penetration=(-collision_sdf).clamp_min(0.).amax(-1) / .001,
        table=(table_z + .0003 - collision_points[..., 2]).clamp_min(0.).amax(-1) / .001,
        normal_alignment=normal_alignment,
        patch_distance_m=patch_distance,
        **supplemental,
    )
