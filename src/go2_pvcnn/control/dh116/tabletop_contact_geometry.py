"""Measured PhysX contact geometry reducers for rigid-cylinder grasp probes."""

import torch


def aggregate_contact_geometry(
    contact_positions_w,
    normal_forces_w,
    object_center_w,
    *,
    force_threshold_n=0.05,
    centerline_epsilon_m=1e-6,
    object_axis_w=None,
):
    """Aggregate finite force-bearing contact reports per hand.

    ``contact_positions_w`` and ``normal_forces_w`` have shape ``(N, 2, L, 3)``;
    ``object_center_w`` has shape ``(N, 3)``.  The returned center is weighted
    by normal-force magnitude, while the returned normal is the normalized
    vector sum of the reported normal forces.
    """
    positions = torch.as_tensor(contact_positions_w)
    forces = torch.as_tensor(normal_forces_w, device=positions.device, dtype=positions.dtype)
    center = torch.as_tensor(object_center_w, device=positions.device, dtype=positions.dtype)
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError('contact positions must have shape (N, 2, L, 3)')
    if forces.shape != positions.shape:
        raise ValueError('normal forces must match contact positions')
    if center.shape != (positions.shape[0], 3):
        raise ValueError('object center must have shape (N, 3)')
    if not torch.isfinite(torch.as_tensor(force_threshold_n)) or force_threshold_n < 0:
        raise ValueError('force threshold must be finite and nonnegative')
    if not torch.isfinite(torch.as_tensor(centerline_epsilon_m)) or centerline_epsilon_m < 0:
        raise ValueError('centerline epsilon must be finite and nonnegative')

    magnitudes = torch.linalg.vector_norm(forces, dim=-1)
    valid = (magnitudes >= force_threshold_n) & torch.isfinite(positions).all(dim=-1)
    weights = torch.where(valid, magnitudes, torch.zeros_like(magnitudes))
    total = weights.sum(dim=2)
    hand_valid = total > 0.
    safe_total = total.clamp_min(torch.finfo(total.dtype).tiny)
    contact_center = (positions.nan_to_num(0.) * weights[..., None]).sum(dim=2) / safe_total[..., None]

    force_sum = torch.where(valid[..., None], forces, torch.zeros_like(forces)).sum(dim=2)
    force_norm = torch.linalg.vector_norm(force_sum, dim=-1)
    contact_normal = force_sum / force_norm.clamp_min(torch.finfo(force_norm.dtype).tiny)[..., None]
    contact_center = torch.where(hand_valid[..., None], contact_center, torch.full_like(contact_center, float('nan')))
    contact_normal = torch.where(hand_valid[..., None], contact_normal, torch.full_like(contact_normal, float('nan')))

    relative=positions-center[:,None,None,:]
    if object_axis_w is not None:
        axis=torch.as_tensor(object_axis_w,device=positions.device,dtype=positions.dtype)
        if axis.shape!=center.shape or not torch.isfinite(axis).all():
            raise ValueError('finite object axis must have shape (N,3)')
        norm=torch.linalg.vector_norm(axis,dim=1,keepdim=True)
        if (norm<=0).any():raise ValueError('object axis must be nonzero')
        axis=axis/norm
        along=(relative*axis[:,None,None,:]).sum(-1)
        radial_z=relative[...,2]-along*axis[:,None,None,2]
    else:
        radial_z=relative[...,2]
    below = radial_z < -centerline_epsilon_m
    all_below = hand_valid & ((~valid) | below).all(dim=2)
    return dict(
        contact_center_w=contact_center,
        contact_normal_w=contact_normal,
        normal_load_n=total,
        valid=hand_valid,
        all_below_centerline=all_below,
        contact_count=valid.sum(dim=2),
    )
