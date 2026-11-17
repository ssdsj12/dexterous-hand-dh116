"""Original per-hand clearance projection for the independent flexible runtime.

The caller binds measured base poses, the original reset seed, and full
hand collision clearance to the same physical clock. This module computes
targets only. It cannot certify geometry, native execution or PPO readiness.
"""
import math

import torch

from .tabletop_command_kernels import palm_transport, per_hand_safe_z
from .tabletop_free_wrist_grasp import CONTRACT as WRIST, compose_rotation
from .tabletop_surface_contract import ACTION_CONTRACT


DT = 1 / 480
CLEARANCE_M = .0021
SOURCE = "tabletop_command_kernels.palm_transport+per_hand_safe_z:policy_owned_default_clearance"


def project_policy_wrist_targets(*, root_seed_w, measured_base_w, common_lift_offset_m,
                                 measured_clearance_m, palm_offsets_m,
                                 requested_palm_offsets_m, angles_rad,
                                 requested_angles_rad, allowed, live):
    """Mirror the original policy-owned, independently projected wrist branch."""
    if (not isinstance(root_seed_w, torch.Tensor) or root_seed_w.ndim != 3
            or root_seed_w.shape[1:] != (2, 7) or len(root_seed_w) < 1
            or not root_seed_w.is_floating_point()):
        raise ValueError("measured and authored wrist poses must be floating [N,2,7] WXYZ")
    n = len(root_seed_w)
    for name, value, shape in (
            ("reset seed", root_seed_w, (n, 2, 7)),
            ("measured base", measured_base_w, (n, 2, 7)),
            ("common lift offset", common_lift_offset_m, (n,)),
            ("measured clearance", measured_clearance_m, (n, 2)),
            ("palm offsets", palm_offsets_m, (n, 2, 3)),
            ("requested palm offsets", requested_palm_offsets_m, (n, 2, 3)),
            ("angles", angles_rad, (n, 2, 3)),
            ("requested angles", requested_angles_rad, (n, 2, 3))):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.device != root_seed_w.device or value.dtype != root_seed_w.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"finite same-device {name} with shape {shape} required")
    for mask in (allowed, live):
        if (not isinstance(mask, torch.Tensor) or mask.shape != (n,)
                or mask.dtype != torch.bool or mask.device != root_seed_w.device):
            raise ValueError("explicit same-device boolean allowed/live masks required")
    for pose in (root_seed_w, measured_base_w):
        norm = torch.linalg.vector_norm(pose[..., 3:], dim=-1)
        if not torch.allclose(norm, torch.ones_like(norm), atol=1e-5, rtol=0):
            raise ValueError("unit WXYZ quaternions required")
    if any(bool((angles.abs() > math.pi / 2).any())
           for angles in (angles_rad, requested_angles_rad)):
        raise ValueError("original intrinsic XYZ limits required")

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
    roots[..., 3:] = compose_rotation(root_seed_w[..., 3:], angles)
    return dict(projected_root_poses=roots, projected_angles=angles, palm_offsets_m=palms,
                pre_projection_palm_offsets_m=pre_projection, safe_z_offset_m=safe_z,
                safe_z_clipped=clipped, projection_reasons=reasons | (2 * clipped.long()),
                projection_source=SOURCE, clearance_floor_m=CLEARANCE_M,
                native_clearance_qualified=False, teacher_admission=False)
