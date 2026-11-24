"""Original wrist arithmetic with one complete batch validation decision.

Metadata is checked on the host before tensor arithmetic. All original finite,
unit-quaternion and angle guards are reduced on the current device and observed
once, before targets are returned. The legacy function remains the oracle.
"""
import math

import torch

from .tabletop_command_kernels import palm_transport, per_hand_safe_z
from .tabletop_deformable_wrist_projection import CLEARANCE_M, DT, SOURCE
from .tabletop_free_wrist_grasp import CONTRACT as WRIST, multiply
from .tabletop_surface_contract import ACTION_CONTRACT


def project_tensor_policy_wrist_targets(*, root_seed_w, measured_base_w, common_lift_offset_m,
                                       measured_clearance_m, palm_offsets_m,
                                       requested_palm_offsets_m, angles_rad,
                                       requested_angles_rad, allowed, live):
    """Project the full same-device batch without per-field host decisions."""
    if (not isinstance(root_seed_w, torch.Tensor) or root_seed_w.ndim != 3
            or root_seed_w.shape[1:] != (2, 7) or len(root_seed_w) < 1
            or not root_seed_w.is_floating_point()):
        raise ValueError("measured and authored wrist poses must be floating [N,2,7] WXYZ")
    n = len(root_seed_w)
    values = (
        (root_seed_w, (n, 2, 7)), (measured_base_w, (n, 2, 7)),
        (common_lift_offset_m, (n,)), (measured_clearance_m, (n, 2)),
        (palm_offsets_m, (n, 2, 3)), (requested_palm_offsets_m, (n, 2, 3)),
        (angles_rad, (n, 2, 3)), (requested_angles_rad, (n, 2, 3)),
    )
    for value, shape in values:
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.device != root_seed_w.device or value.dtype != root_seed_w.dtype):
            raise ValueError("complete same-device same-dtype wrist projection inputs required")
    for mask in (allowed, live):
        if (not isinstance(mask, torch.Tensor) or mask.shape != (n,)
                or mask.dtype != torch.bool or mask.device != root_seed_w.device):
            raise ValueError("explicit same-device boolean allowed/live masks required")

    finite = torch.isfinite(torch.cat([value.reshape(-1) for value, _ in values])).all()
    norms = torch.linalg.vector_norm(torch.stack((root_seed_w[..., 3:], measured_base_w[..., 3:])), dim=-1)
    unit = torch.isclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0).all()
    angles_valid = ~(torch.cat((angles_rad.reshape(-1), requested_angles_rad.reshape(-1))).abs() > math.pi / 2).any()
    seed_scale = root_seed_w[..., 3:].abs().amax(-1, keepdim=True)
    nonzero = (seed_scale > 0).all()
    if not bool(finite & unit & angles_valid & nonzero):
        raise ValueError("finite wrist inputs, unit WXYZ quaternions and original intrinsic XYZ limits required")

    lower = root_seed_w.new_tensor(WRIST["palm_lower_m"])
    upper = root_seed_w.new_tensor(WRIST["palm_upper_m"])
    palms, pre_projection, reasons = palm_transport(
        palm_offsets_m, requested_palm_offsets_m, measured_clearance_m, CLEARANCE_M,
        ACTION_CONTRACT["translation_rate_m_s"], DT, lower, upper, allowed, live,
        vector_norm=True,
    )
    projected_z, safe_z = per_hand_safe_z(
        palms, measured_base_w, root_seed_w, common_lift_offset_m,
        measured_clearance_m, CLEARANCE_M,
    )
    clipped = live[:, None] & ((projected_z - palms[:, :, 2]).abs() > 1e-12)
    palms[:, :, 2] = torch.where(live[:, None], projected_z, palms[:, :, 2])
    angles = torch.where((allowed & live)[:, None, None], requested_angles_rad, angles_rad)
    roots = root_seed_w.clone()
    roots[..., :3] += palms
    roots[..., 2] += common_lift_offset_m[:, None]
    # compose_rotation's finite/nonzero guards were included above. Keep its
    # exact scale, normalization and intrinsic XYZ multiply order.
    q = root_seed_w[..., 3:] / seed_scale
    q = q / q.norm(dim=-1, keepdim=True)
    for axis in range(3):
        step = torch.zeros_like(q)
        step[..., 0] = torch.cos(angles[..., axis] / 2)
        step[..., axis + 1] = torch.sin(angles[..., axis] / 2)
        q = multiply(q, step)
    roots[..., 3:] = q / q.norm(dim=-1, keepdim=True)
    return dict(projected_root_poses=roots, projected_angles=angles, palm_offsets_m=palms,
                pre_projection_palm_offsets_m=pre_projection, safe_z_offset_m=safe_z,
                safe_z_clipped=clipped, projection_reasons=reasons | (2 * clipped.long()),
                projection_source=SOURCE, clearance_floor_m=CLEARANCE_M,
                native_clearance_qualified=False, teacher_admission=False)
