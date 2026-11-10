"""Authored front-v3 RGB rig and CPU pinhole field-of-view checks.

Projection proves geometric framing only. It does not prove native rendering,
occlusion-free visibility, stereo accuracy or compatibility with old models.
"""
import hashlib
import itertools
import math
from pathlib import Path

import torch

from .tabletop_stereo_calibration import StereoCalibration
from .tabletop_stereo_resolution import disparity_budget, validate_resolution

VERSION = 'dh116-stereo-camera-front-v3'
CENTER_WORLD_M = (1.02, 0., .88)
TARGET_WORLD_M = (.43, 0., .43)
BASELINE_M = .08
HFOV_DEGREES = 75.


def front_calibration(width, height):
    """Parallel +worldY stereo baseline, both looking at the centre's target."""
    validate_resolution(width, height)
    centre = torch.tensor(CENTER_WORLD_M, dtype=torch.float64)
    target = torch.tensor(TARGET_WORLD_M, dtype=torch.float64)
    right = torch.tensor([0., 1., 0.], dtype=torch.float64)
    forward = target-centre
    forward = forward/forward.norm()
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack((right, down, forward), dim=1)
    transforms = []
    for offset in (-BASELINE_M/2, BASELINE_M/2):
        transform = torch.eye(4, dtype=torch.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = centre+offset*right
        transforms.append(transform)
    focal = width/(2*math.tan(math.radians(HFOV_DEGREES)/2))
    k = [[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]]
    return StereoCalibration(k, k, *transforms, width, height, BASELINE_M)


def project_world(calibration, points_world):
    """World points to continuous pixel-centre coordinates; no clipping/padding."""
    if type(calibration) is not StereoCalibration:
        raise TypeError('StereoCalibration required')
    points = torch.as_tensor(points_world, dtype=torch.float64).detach().cpu()
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not torch.isfinite(points).all():
        raise ValueError('nonempty finite world points [P,3] required')
    result = {}
    for eye in ('left', 'right'):
        t = getattr(calibration, 'T_world_'+eye)
        k = getattr(calibration, 'K_'+eye)
        optical = (points-t[:3, 3]) @ t[:3, :3]
        z = optical[:, 2]
        positive = z > 0
        pixel = optical @ k.T
        uv = pixel[:, :2]/torch.where(positive, z, torch.ones_like(z))[:, None]
        uv = torch.where(positive[:, None], uv, float('nan'))
        visible = (positive & torch.isfinite(uv).all(-1) & (uv[:, 0] >= .5)
            & (uv[:, 0] <= calibration.width-.5) & (uv[:, 1] >= .5)
            & (uv[:, 1] <= calibration.height-.5))
        result[eye] = dict(uv_px=uv, optical_depth_m=z, in_view=visible)
    return result


def assess_framing(calibration, points_world, *, workspace_margin_m=.05):
    """Project every corner of an explicit XYZ box around each reference point."""
    if type(workspace_margin_m) not in (int, float) or not math.isfinite(workspace_margin_m) or workspace_margin_m < 0:
        raise ValueError('finite nonnegative workspace margin in m required')
    points = torch.as_tensor(points_world, dtype=torch.float64).detach().cpu()
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not torch.isfinite(points).all():
        raise ValueError('nonempty finite reference points [P,3] required')
    offsets = torch.tensor(list(itertools.product((-workspace_margin_m, workspace_margin_m), repeat=3)), dtype=torch.float64)
    expanded = (points[:, None]+offsets[None]).reshape(-1, 3)
    projection = project_world(calibration, expanded)
    borders = []
    for eye in projection.values():
        u, v = eye['uv_px'].unbind(-1)
        borders.extend((u-.5, calibration.width-.5-u, v-.5, calibration.height-.5-v))
    margins = torch.cat(borders)
    return dict(**projection, all_in_view=all(bool(eye['in_view'].all()) for eye in projection.values()),
        reference_point_count=len(points), expanded_point_count=len(expanded),
        workspace_margin_m=float(workspace_margin_m),
        minimum_border_margin_px=float(margins.min()) if bool(torch.isfinite(margins).all()) else None,
        minimum_optical_depth_m=min(float(eye['optical_depth_m'].min()) for eye in projection.values()),
        source_scope='CPU_projection_arithmetic', native_visibility_qualified=False,
        occlusion_checked=False, hidden_padding_or_resize=False)


def front_profile(width, height):
    """Versioned new rig profile; old configurations/defaults are never edited."""
    c = front_calibration(width, height)
    references = [[.502083, .188617, .487541], [.522067, -.197186, .478162],
        [.483137, .177137, .495522], [.515256, -.191129, .480714],
        [.43, -.25, .368], [.43, .25, .368]]
    framing = assess_framing(c, references, workspace_margin_m=.05)
    if not framing['all_in_view']:
        raise ValueError('front rig does not frame the declared workspace')
    summary = {key: value for key, value in framing.items() if key not in ('left', 'right')}
    budget = disparity_budget(c)
    max_disparity = max(64, budget['suggested_max_disparity']) if width <= 224 else max(112 if width <= 320 else 192, budget['suggested_max_disparity'])
    depth = dict(method='blockmatch', min_disparity=1 if width <= 224 else 4 if width <= 320 else 8,
        max_disparity=max_disparity, block_size=5, min_depth_m=.2, max_depth_m=1.2,
        lr_tolerance_px=1., texture_threshold=.001, uniqueness_threshold=.05,
        rectify=False, subpixel=True, image_processing_batch_size=1)
    source = Path(__file__).resolve()
    return dict(**c.to_dict(), calibration_sha256=c.sha256, profile_version=VERSION,
        profile_name=f'front_{width}x{height}_native_rerender', enabled_by_default=False,
        default_accuracy_profile=False, intended_accuracy_profile=(width, height)==(640, 480),
        resolution_mode='new_front_native_rig_not_image_resize', horizontal_fov_degrees=HFOV_DEGREES,
        optical_axes='X_right_Y_down_Z_forward', rgb_layout='BCHW_float_0_1_RGB',
        center_world_m=list(CENTER_WORLD_M), target_world_m=list(TARGET_WORLD_M),
        optical_right_world=[0., 1., 0.], depth_range_m=[.2, 1.2],
        disparity_range_px=[depth['min_disparity'], depth['max_disparity']],
        depth_estimator=depth, disparity_budget=budget,
        resource_limits=dict(max_width=2048, max_height=1536, max_readback_bytes=1024**3, allow_resource_override=False),
        matcher_resource_policy='stream_one_disparity_plane;image_processing_batch_size=1;never_hidden_downsample',
        model_compatibility='new_calibration_SHA_required;no_silent_reuse_of_old_rig_models',
        framing_reference_world_m=references, framing_projection=summary,
        source_metadata=dict(source_scope='authored_front_rig_from_reported_native_framing_coordinates',
            reported_primary_palm_xyz_m=references[:2], reported_secondary_palm_xyz_m=references[2:4],
            reported_wire_center_xyz_m=[.4299999475, 0., .3680000007],
            coordinate_frame='environment_local_base_rig;runtime_shifts_rig_and_points_by_same_environment_origin',
            coordinates_status='reported_actual_SDK_review_readback;50mm_XYZ_margin;raw_native_source_requires_separate_binding',
            source_artifact=dict(path='src/go2_pvcnn/control/dh116/'+source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
            native_capture_performed_for_this_rig=False, old_rig_failed_evidence_preserved=True),
        native_qualification='pending_new_front_rig_native_readback_visibility_and_accuracy;CPU_projection_only')
