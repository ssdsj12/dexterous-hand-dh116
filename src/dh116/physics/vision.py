from dh116.env.contracts import source_file
'Stereo calibration, depth/pyramid preprocessing, spatial transforms, resolution and sensor validation.'
from dataclasses import dataclass
import math as stere_depth__math
import torch as stere_geome__torch
from copy import deepcopy
import math as stere_pyram__math
import json as stere_resol__json
import copy
from dataclasses import field
import json as stere_depth__json

def backproject(depth_m, K, valid):
    z = stere_geome__torch.as_tensor(depth_m)
    if z.ndim != 3 or not z.is_floating_point() or valid.shape != z.shape or (valid.dtype != stere_geome__torch.bool):
        raise ValueError('depth/valid requires float/bool [B,H,W]')
    (n, h, w) = z.shape
    k = stere_geome__torch.as_tensor(K, device=z.device, dtype=z.dtype)
    if k.shape == (3, 3):
        k = k[None].expand(n, -1, -1)
    if k.shape != (n, 3, 3) or not stere_geome__torch.isfinite(k).all() or (stere_geome__torch.linalg.det(k).abs() < 1e-12).any():
        raise ValueError('finite invertible per-batch K required')
    (y, x) = stere_geome__torch.meshgrid(stere_geome__torch.arange(h, device=z.device, dtype=z.dtype) + 0.5, stere_geome__torch.arange(w, device=z.device, dtype=z.dtype) + 0.5, indexing='ij')
    pixel = stere_geome__torch.stack((x, y, stere_geome__torch.ones_like(x)), -1).reshape(1, h * w, 3).expand(n, -1, -1)
    rays = pixel @ stere_geome__torch.linalg.inv(k).transpose(-1, -2)
    p = (rays * z.reshape(n, -1, 1)).reshape(n, h, w, 3)
    good = valid & stere_geome__torch.isfinite(z) & (z > 0)
    return stere_geome__torch.where(good[..., None], p, float('nan'))
import hashlib as stere_prepr__hashlib
import torch as stere_pyram__torch
import math as stere_resol__math
import math as stere_valid__math
import hashlib as stere_calib__hashlib
from pathlib import Path as stere_depth__Path

def transform_points(points, T_world_camera):
    p = stere_geome__torch.as_tensor(points)
    if p.ndim < 3 or p.shape[-1] != 3 or (not p.is_floating_point()):
        raise ValueError('batched 3D points required')
    n = p.shape[0]
    t = stere_geome__torch.as_tensor(T_world_camera, device=p.device, dtype=p.dtype)
    if t.shape == (4, 4):
        t = t[None].expand(n, -1, -1)
    if t.shape != (n, 4, 4) or not stere_geome__torch.isfinite(t).all():
        raise ValueError('finite world transform required')
    if not stere_geome__torch.allclose(t[:, 3], t.new_tensor([0, 0, 0, 1]).expand(n, -1), atol=1e-06, rtol=0.0):
        raise ValueError('homogeneous transform required')
    r = t[:, :3, :3]
    if not stere_geome__torch.allclose(r.transpose(-1, -2) @ r, stere_geome__torch.eye(3, device=p.device, dtype=p.dtype).expand(n, -1, -1), atol=1e-05, rtol=0.0) or not stere_geome__torch.allclose(stere_geome__torch.linalg.det(r), stere_geome__torch.ones(n, device=p.device, dtype=p.dtype), atol=1e-05, rtol=0.0):
        raise ValueError('rigid rotation required')
    flat = p.reshape(n, -1, 3) @ t[:, :3, :3].transpose(-1, -2) + t[:, None, :3, 3]
    return flat.reshape_as(p)
import json as stere_prepr__json
import torch.nn.functional as stere_pyram__F
from pathlib import Path as stere_resol__Path
from numbers import Integral
import json as stere_calib__json
import torch as stere_depth__torch

def world_to_palms(points, palm_poses):
    p = stere_geome__torch.as_tensor(points)
    pose = stere_geome__torch.as_tensor(palm_poses, device=p.device, dtype=p.dtype)
    if p.ndim < 3 or p.shape[-1] != 3 or (not p.is_floating_point()) or (pose.shape != (p.shape[0], 2, 7)) or (not stere_geome__torch.isfinite(pose).all()):
        raise ValueError('world points and actual two-palm FK xyz+wxyz required')
    q = pose[..., 3:]
    scale = q.abs().amax(-1, keepdim=True)
    if (scale <= 0).any():
        raise ValueError('nonzero palm quaternion required')
    q = q / scale
    q = q / q.norm(dim=-1, keepdim=True)
    (w, x, y, z) = q.unbind(-1)
    rotation = stere_geome__torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(-1, 2, 3, 3)
    flat = p.reshape(p.shape[0], -1, 3)
    local = (flat[:, None] - pose[:, :, None, :3]) @ rotation
    return local.reshape(p.shape[0], 2, *p.shape[1:])
from pathlib import Path as stere_prepr__Path

def _raw_mask(image, mask):
    if mask is None:
        mask = stere_pyram__torch.ones_like(image, dtype=stere_pyram__torch.bool)
    if not isinstance(mask, stere_pyram__torch.Tensor) or mask.shape != image.shape or mask.dtype != stere_pyram__torch.bool or (mask.device != image.device):
        raise ValueError('raw validity requires bool [B,H,W] on image device')
    return mask & stere_pyram__torch.isfinite(image)
DEFAULT_MAX_WIDTH = 2048
import torch as stere_valid__torch
import math as stere_calib__math
import torch.nn.functional as stere_depth__F

def order_polyline(points, valid, adjacency):
    """Order an explicitly associated simple path; never invent hidden cable shape.

    adjacency must come from sensor segmentation/skeleton association. Branches,
    loops and disconnected observed fragments raise instead of claiming a full
    centreline. Padding stays NaN/false. No rigid-axis sorting or truth fallback.
    """
    p = stere_geome__torch.as_tensor(points)
    if p.ndim != 3 or p.shape[-1] != 3 or valid.shape != p.shape[:2] or (valid.dtype != stere_geome__torch.bool) or (adjacency.shape != (*p.shape[:2], p.shape[1])) or (adjacency.dtype != stere_geome__torch.bool):
        raise ValueError('points/mask/simple path topology shape mismatch')
    (n, k, _) = p.shape
    out = stere_geome__torch.full_like(p, float('nan'))
    mask = stere_geome__torch.zeros_like(valid)
    order = stere_geome__torch.full((n, k), -1, device=p.device, dtype=stere_geome__torch.long)
    arc = stere_geome__torch.full((n, k), float('nan'), device=p.device, dtype=p.dtype)
    for row in range(n):
        keep = valid[row].nonzero(as_tuple=True)[0]
        if not len(keep):
            continue
        if not stere_geome__torch.isfinite(p[row, keep]).all():
            raise ValueError('valid path points must be finite')
        a = adjacency[row][keep][:, keep]
        if a.diag().any() or not stere_geome__torch.equal(a, a.T) or adjacency[row][~valid[row]].any() or adjacency[row][:, ~valid[row]].any():
            raise ValueError('invalid sensor path topology')
        degree = a.sum(1)
        if len(keep) == 1:
            walk = [0]
        else:
            ends = (degree == 1).nonzero(as_tuple=True)[0]
            if len(ends) != 2 or (degree > 2).any() or (degree == 0).any():
                raise ValueError('branch/loop/disconnected path topology')
            walk = [int(ends[0])]
            while len(walk) < len(keep):
                nexts = [int(v) for v in a[walk[-1]].nonzero(as_tuple=True)[0] if int(v) not in walk]
                if len(nexts) != 1:
                    raise ValueError('disconnected path topology')
                walk.append(nexts[0])
        ids = keep[stere_geome__torch.tensor(walk, device=p.device)]
        count = len(ids)
        ordered = p[row, ids]
        out[row, :count] = ordered
        mask[row, :count] = True
        order[row, :count] = ids
        arc[row, 0] = 0.0
        if count > 1:
            arc[row, 1:count] = (ordered[1:] - ordered[:-1]).norm(dim=-1).cumsum(0)
    return dict(points=out, valid=mask, order=order, arc_length_m=arc, association_valid=mask.clone())

def _downsample(image, raw_valid, size):
    """Area resize and ALL source support, directly from the original image."""
    safe = stere_pyram__torch.where(raw_valid, image, 0.0)
    if image.shape[-2:] == size:
        return (safe, raw_valid)
    value = stere_pyram__F.interpolate(safe[:, None], size=size, mode='area')[:, 0]
    invalid = stere_pyram__F.adaptive_max_pool2d((~raw_valid).to(image.dtype)[:, None], size)[:, 0]
    return (value, invalid == 0)
DEFAULT_MAX_HEIGHT = 1536

def validate_images(left, right, calibration):
    if not isinstance(left, stere_valid__torch.Tensor) or not isinstance(right, stere_valid__torch.Tensor) or left.shape != right.shape or (left.ndim != 4) or (left.shape[1:] != (3, calibration.height, calibration.width)) or (left.shape[0] < 1) or (not left.is_floating_point()) or (not right.is_floating_point()) or (left.dtype != right.dtype) or (left.device != right.device):
        raise ValueError('RGB stereo requires matching float [B,3,H,W] tensors')
    for image in (left, right):
        if not stere_valid__torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError('RGB stereo must be finite in [0,1]')
from pathlib import Path as stere_calib__Path

def _resize_prior(disparity, valid, size):
    """Bilinear prior with exact horizontal units and fail-closed NaN support."""
    valid = valid & stere_pyram__torch.isfinite(disparity)
    safe = stere_pyram__torch.where(valid, disparity, 0.0)
    value = stere_pyram__F.interpolate(safe[:, None], size=size, mode='bilinear', align_corners=False)[:, 0]
    touched_invalid = stere_pyram__F.interpolate((~valid).to(disparity.dtype)[:, None], size=size, mode='bilinear', align_corners=False)[:, 0]
    value = value * (size[1] / disparity.shape[-1])
    supported = (touched_invalid == 0) & stere_pyram__torch.isfinite(value)
    return (stere_pyram__torch.where(supported, value, float('nan')), supported)
METRIC_KEYS = ('depth_mae_m', 'depth_rmse_m', 'depth_p95_m', 'wire_point_cloud_completeness', 'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes', 'gpu_free_bytes', 'capture_ms', 'depth_ms', 'actor_ms')

def validate_identity(identity, batch_size, timestamp):
    """Caller supplies pre-action identity; no task or simulator truth is queried."""
    required = {'run_uuid', 'env_id', 'episode_uid', 'reset_id', 'policy_tick', 'physics_tick_start', 'physics_tick_end', 'simulation_timestamp'}
    rows = identity if isinstance(identity, list) else [identity]
    if not rows or len(rows) != batch_size:
        raise ValueError('identity batch mismatch; supply per-environment records')
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError('missing stereo identity')
        for key in ('env_id', 'reset_id', 'policy_tick', 'physics_tick_start', 'physics_tick_end'):
            if isinstance(row[key], bool) or not isinstance(row[key], Integral) or row[key] < 0:
                raise ValueError('invalid integer stereo identity: ' + key)
        if not isinstance(row['run_uuid'], str) or not row['run_uuid'] or (not isinstance(row['episode_uid'], str)) or (not row['episode_uid']):
            raise ValueError('invalid run/episode stereo identity')
        if row['env_id'] in seen:
            raise ValueError('duplicate environment stereo identity')
        seen.add(row['env_id'])
        if not row['run_uuid'] or not stere_valid__math.isfinite(float(row['simulation_timestamp'])) or abs(float(row['simulation_timestamp']) - timestamp) > 1e-06 or (int(row['physics_tick_start']) < 0) or (int(row['physics_tick_end']) < int(row['physics_tick_start'])) or (int(row['policy_tick']) < 0):
            raise ValueError('stereo identity/time mismatch')
    return copy.deepcopy(identity)
import torch as stere_calib__torch

def _invalid_result(image):
    invalid = stere_pyram__torch.zeros_like(image, dtype=stere_pyram__torch.bool)
    confidence = stere_pyram__torch.zeros_like(image)
    disparity = stere_pyram__torch.full_like(image, float('nan'))
    return (disparity, confidence, invalid, dict(integer_disparity=disparity, integer_confidence=confidence, integer_good=invalid, refinement_valid=invalid, search_boundary=invalid, local_failed=~invalid))
WIRE_DEPTH_KEYS = ('wire_depth_mae_m', 'wire_depth_rmse_m', 'wire_depth_p95_m', 'wire_depth_validity_coverage', 'wire_depth_count')

def validate_pair_timing(packet, last_frame=-1):
    (l, r) = (packet['left_frame_id'], packet['right_frame_id'])
    (a, b) = (float(packet['left_simulation_time']), float(packet['right_simulation_time']))
    if type(l) is not int or type(r) is not int or l != r or (l < 0) or (not stere_valid__math.isfinite(a)) or (a < 0) or (a != b):
        raise ValueError('stereo frame/timestamp mismatch')
    if l <= last_frame:
        raise RuntimeError('stale stereo frame; fresh pair required')
    return (l, a)

def _matrix(value, shape, name):
    t = stere_calib__torch.as_tensor(value, dtype=stere_calib__torch.float64).detach().cpu().clone()
    if t.shape != shape or not stere_calib__torch.isfinite(t).all():
        raise ValueError('invalid calibration ' + name)
    return t

def validate_resolution(width, height, *, max_width=DEFAULT_MAX_WIDTH, max_height=DEFAULT_MAX_HEIGHT, allow_resource_override=False):
    if any((type(v) is not int or v < 1 for v in (width, height, max_width, max_height))):
        raise ValueError('positive integer resolution and resource limits required')
    if type(allow_resource_override) is not bool:
        raise ValueError('explicit resource override boolean required')
    if not allow_resource_override and (width > max_width or height > max_height):
        raise ValueError('resolution exceeds configured resource limit; explicit override required')
    return (width, height)

def depth_metrics(estimate, reference, valid, reference_valid):
    """Evaluation only: reference must have separate source qualification/provenance."""
    if estimate.shape != reference.shape or valid.shape != estimate.shape or reference_valid.shape != estimate.shape:
        raise ValueError('depth metric shapes must match')
    mask = valid.bool() & reference_valid.bool() & stere_valid__torch.isfinite(estimate) & stere_valid__torch.isfinite(reference) & (reference > 0)
    if not mask.any():
        return dict(mae_m=None, rmse_m=None, count=0, coverage=0.0)
    error = (estimate - reference)[mask]
    return dict(mae_m=float(error.abs().mean()), rmse_m=float(error.square().mean().sqrt()), count=int(mask.sum()), coverage=float(mask.float().mean()))

@dataclass(frozen=True)
class StereoCalibration:
    K_left: object
    K_right: object
    T_world_left: object
    T_world_right: object
    width: int
    height: int
    baseline_m: float
    distortion_left: object = field(default_factory=lambda : [0.0] * 5)
    distortion_right: object = field(default_factory=lambda : [0.0] * 5)
    rectified: bool = True
    version: str = 'dh116-stereo-optical-v1'

    def __post_init__(self):
        if type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) < 1 or (not stere_calib__math.isfinite(self.baseline_m)) or (self.baseline_m <= 0) or (type(self.rectified) is not bool) or (self.version != 'dh116-stereo-optical-v1'):
            raise ValueError('invalid calibration dimensions/version/baseline')
        for name in ('K_left', 'K_right'):
            k = _matrix(getattr(self, name), (3, 3), name)
            if k[0, 0] <= 0 or k[1, 1] <= 0 or (not stere_calib__torch.allclose(k[2], stere_calib__torch.tensor([0.0, 0.0, 1.0], dtype=k.dtype))) or (abs(float(k[0, 1])) > 1e-10) or (abs(float(k[1, 0])) > 1e-10):
                raise ValueError('invalid calibration intrinsic matrix')
            object.__setattr__(self, name, k)
        for name in ('T_world_left', 'T_world_right'):
            t = _matrix(getattr(self, name), (4, 4), name)
            r = t[:3, :3]
            if not stere_calib__torch.allclose(t[3], t.new_tensor([0, 0, 0, 1]), atol=1e-08, rtol=0.0) or not stere_calib__torch.allclose(r.T @ r, stere_calib__torch.eye(3, dtype=r.dtype), atol=1e-07, rtol=0.0) or abs(float(stere_calib__torch.det(r)) - 1) > 1e-07:
                raise ValueError('invalid calibration rigid rotation')
            object.__setattr__(self, name, t)
        for name in ('distortion_left', 'distortion_right'):
            d = stere_calib__torch.as_tensor(getattr(self, name), dtype=stere_calib__torch.float64).detach().cpu().clone()
            if d.shape not in ((4,), (5,), (8,)) or not stere_calib__torch.isfinite(d).all():
                raise ValueError('invalid distortion')
            object.__setattr__(self, name, d)
        relative = stere_calib__torch.linalg.inv(self.T_world_left) @ self.T_world_right
        if abs(float(relative[:3, 3].norm()) - self.baseline_m) > 1e-07:
            raise ValueError('calibration baseline mismatch')
        if self.rectified:
            if not stere_calib__torch.allclose(relative[:3, :3], stere_calib__torch.eye(3, dtype=relative.dtype), atol=1e-07, rtol=0.0) or not stere_calib__torch.allclose(relative[:3, 3], relative.new_tensor([self.baseline_m, 0, 0]), atol=1e-07, rtol=0.0) or (not stere_calib__torch.allclose(self.K_left, self.K_right, atol=1e-08, rtol=0.0)) or self.distortion_left.abs().any() or self.distortion_right.abs().any():
                raise ValueError('rectified calibration requires common K, parallel optical axes, positive-X baseline and zero distortion')

    def to_dict(self):
        return {key: getattr(self, key).tolist() if isinstance(getattr(self, key), stere_calib__torch.Tensor) else getattr(self, key) for key in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data):
        fields = cls.__dataclass_fields__
        result = cls(**{k: v for (k, v) in data.items() if k in fields})
        if 'calibration_sha256' in data and data['calibration_sha256'] != result.sha256:
            raise ValueError('calibration SHA digest mismatch')
        return result

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(stere_calib__json.loads(stere_calib__Path(path).read_text(encoding='utf-8')))

    @property
    def sha256(self):
        return stere_calib__hashlib.sha256(stere_calib__json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def validate_readback(self, K_left, K_right, T_world_left, T_world_right, origins=None):
        intrinsic_rtol = 2 * stere_calib__torch.finfo(stere_calib__torch.float32).eps
        diagnostics = dict(intrinsic_atol_px=1e-05, intrinsic_rtol=intrinsic_rtol, extrinsic_atol=1e-05, extrinsic_rtol=0.0, max_intrinsic_residual_px=0.0, max_translation_residual_m=0.0, max_rotation_matrix_residual=0.0, scope='numerical_SDK_lens_readback_only;not_native_accuracy_qualification')
        shift = None if origins is None else stere_calib__torch.as_tensor(origins, dtype=stere_calib__torch.float64).cpu()
        if shift is not None and (shift.ndim != 2 or shift.shape[1] != 3 or (not stere_calib__torch.isfinite(shift).all())):
            raise ValueError('invalid calibration origins')
        for (name, value) in zip(('K_left', 'K_right', 'T_world_left', 'T_world_right'), (K_left, K_right, T_world_left, T_world_right)):
            actual = stere_calib__torch.as_tensor(value, dtype=stere_calib__torch.float64).cpu()
            expected = getattr(self, name)
            if shift is not None:
                expected = expected[None].repeat(len(shift), 1, 1)
                if name.startswith('T_'):
                    expected[:, :3, 3] += shift
            elif actual.ndim == 3:
                expected = expected[None].expand_as(actual)
            relative = intrinsic_rtol if name.startswith('K_') else 0.0
            if actual.numel() == 0 or actual.shape != expected.shape or (not stere_calib__torch.allclose(actual, expected, atol=1e-05, rtol=relative)):
                raise ValueError('camera calibration readback mismatch: ' + name)
            difference = (actual - expected).abs()
            if name.startswith('K_'):
                diagnostics['max_intrinsic_residual_px'] = max(diagnostics['max_intrinsic_residual_px'], float(difference.max()))
            else:
                diagnostics['max_translation_residual_m'] = max(diagnostics['max_translation_residual_m'], float(difference[..., :3, 3].max()))
                diagnostics['max_rotation_matrix_residual'] = max(diagnostics['max_rotation_matrix_residual'], float(difference[..., :3, :3].max()))
        return diagnostics
stere_depth__validate_images = validate_images

def stere_resol__rerender_calibration(calibration, width, height, *, horizontal_fov_degrees=75.0, **resource_limits):
    implementation = stere_calib__rerender_calibration
    return implementation(calibration, width, height, horizontal_fov_degrees=horizontal_fov_degrees, **resource_limits)

def stere_calib__rerender_calibration(calibration, width, height, *, horizontal_fov_degrees=75.0, **resource_limits):
    """New camera authoring at the same optical rig, not an image resize.

    A fixed horizontal FOV with square pixels changes vertical FOV when the
    aspect ratio changes. The optical transforms and physical baseline stay
    exactly those of the input calibration.
    """
    _stere_calib_import__validate_resolution = validate_resolution
    _stere_calib_import__validate_resolution(width, height, **resource_limits)
    if not isinstance(calibration, StereoCalibration):
        raise TypeError('StereoCalibration required')
    if not stere_calib__math.isfinite(horizontal_fov_degrees) or not 0 < horizontal_fov_degrees < 180:
        raise ValueError('horizontal field of view must be in (0,180) degrees')
    fx = width / (2 * stere_calib__math.tan(stere_calib__math.radians(horizontal_fov_degrees) / 2))
    data = calibration.to_dict()
    data.update(width=width, height=height)
    k = [[fx, 0.0, width / 2], [0.0, fx, height / 2], [0.0, 0.0, 1.0]]
    data.update(K_left=k, K_right=k)
    return StereoCalibration.from_dict(data)
from dh116.physics.stereo_match import METHODS as WIRE_METHODS

def stere_resol__resize_calibration(calibration, width, height, **resource_limits):
    implementation = stere_calib__resize_calibration
    return implementation(calibration, width, height, **resource_limits)

def stere_calib__resize_calibration(calibration, width, height, **resource_limits):
    """Calibration for a real image resize: K'=diag(sx,sy,1)K, same T/B.

    Pixel centres use u+.5/v+.5, so this continuous coordinate scaling needs
    no extra half-pixel shift. This helper does not resize an image itself.
    """
    _stere_calib_import__validate_resolution = validate_resolution
    _stere_calib_import__validate_resolution(width, height, **resource_limits)
    if not isinstance(calibration, StereoCalibration):
        raise TypeError('StereoCalibration required')
    scale = stere_calib__torch.diag(stere_calib__torch.tensor([width / calibration.width, height / calibration.height, 1.0], dtype=stere_calib__torch.float64))
    data = calibration.to_dict()
    data.update(width=width, height=height, K_left=(scale @ calibration.K_left).tolist(), K_right=(scale @ calibration.K_right).tolist())
    return StereoCalibration.from_dict(data)
from dh116.physics.stereo_match import match_wire

def disparity_budget(calibration, near_m=0.2, far_m=1.2, *, padding_px=8):
    if not 0 < near_m < far_m or type(padding_px) is not int or padding_px < 0:
        raise ValueError('positive near/far and integer disparity padding required')
    fx = float(calibration.K_left[0, 0])
    baseline = calibration.baseline_m
    near = fx * baseline / near_m
    far = fx * baseline / far_m
    return dict(near_disparity_px=near, far_disparity_px=far, suggested_min_disparity=max(1, stere_resol__math.floor(far) - padding_px), suggested_max_disparity=16 * stere_resol__math.ceil((near + padding_px) / 16))

def rectify_rgb(left, right, calibration, *, return_masks=False):
    """Return RGB pair and NEW calibration; optionally both remap-valid masks.

    Use return_masks=True and pass those masks to StereoDepth.compute for a
    nonrectified rig. RGB border fill must never be treated as observed pixels.
    """
    from dh116.physics.vision import validate_images as validate_images
    validate_images(left, right, calibration)
    c = calibration
    if c.rectified:
        masks = [stere_calib__torch.ones_like(left[:, 0], dtype=stere_calib__torch.bool)] * 2
        return (left, right, c, *masks) if return_masks else (left, right, c)
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError('OpenCV cv2 is required for nonrectified stereo rectification') from error
    import numpy as np
    right_from_left = stere_calib__torch.linalg.inv(c.T_world_right) @ c.T_world_left
    kl = c.K_left.numpy().copy()
    kr = c.K_right.numpy().copy()
    for k in (kl, kr):
        k[:2, 2] -= 0.5
    (r1, r2, p1, p2, _, _, _) = cv2.stereoRectify(kl, c.distortion_left.numpy(), kr, c.distortion_right.numpy(), (c.width, c.height), right_from_left[:3, :3].numpy(), right_from_left[:3, 3].numpy(), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    outputs = []
    masks = []
    for (image, k, d, r, p) in [(left, kl, c.distortion_left, r1, p1), (right, kr, c.distortion_right, r2, p2)]:
        (mx, my) = cv2.initUndistortRectifyMap(k, d.numpy(), r, p[:3, :3], (c.width, c.height), cv2.CV_32FC1)
        array = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        out = np.stack([cv2.remap(x, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for x in array])
        outputs.append(stere_calib__torch.as_tensor(out, device=image.device, dtype=image.dtype).permute(0, 3, 1, 2).contiguous())
        supported = (mx >= 0) & (my >= 0) & (mx < c.width - 1) & (my < c.height - 1)
        masks.append(stere_calib__torch.as_tensor(supported, device=image.device)[None].expand(len(image), -1, -1).clone())
    transforms = []
    for (t, r) in [(c.T_world_left, r1), (c.T_world_right, r2)]:
        rect = stere_calib__torch.eye(4, dtype=stere_calib__torch.float64)
        rect[:3, :3] = stere_calib__torch.as_tensor(r).T
        transforms.append(t @ rect)
    k1 = p1[:3, :3].copy()
    k2 = p2[:3, :3].copy()
    k1[:2, 2] += 0.5
    k2[:2, 2] += 0.5
    result = StereoCalibration(k1, k2, *transforms, c.width, c.height, c.baseline_m)
    return (*outputs, result, *masks) if return_masks else (*outputs, result)
from dh116.physics.stereo_match import wire_colour_mask

def _number(value, *, positive=False):
    if type(value) not in (int, float) or not stere_resol__math.isfinite(value) or value < 0 or (positive and value <= 0):
        return None
    return value

def usd_aperture_offsets(k, width, height, focal_length):
    """Map optical K to USD offsets in the same lens units as focal_length.

    USD's view plane is +X right/+Y up; optical pixel coordinates are +Y down.
    """
    k = _matrix(k, (3, 3), 'USD lens K')
    focal = float(focal_length)
    if type(width) is not int or type(height) is not int or min(width, height) < 1 or (not stere_calib__math.isfinite(focal)) or (focal <= 0):
        raise ValueError('positive USD lens dimensions and scales required')
    if k[0, 0] <= 0 or k[1, 1] <= 0 or (not stere_calib__torch.allclose(k[2], k.new_tensor([0.0, 0.0, 1.0]))) or (abs(float(k[0, 1])) > 1e-10) or (abs(float(k[1, 0])) > 1e-10):
        raise ValueError('invalid USD lens intrinsic matrix')
    return ((width / 2 - float(k[0, 2])) * focal / float(k[0, 0]), (float(k[1, 2]) - height / 2) * focal / float(k[1, 1]))

class StereoDepth:

    def __init__(self, *, method='blockmatch', min_disparity=1, max_disparity=64, block_size=5, min_depth_m=0.2, max_depth_m=1.2, lr_tolerance_px=1.0, texture_threshold=0.001, uniqueness_threshold=0.05, rectify=False, subpixel=False, image_processing_batch_size=1, reliability=None, photometric_verification=None, pyramid_mode=None):
        if method not in ('blockmatch', 'sgbm', 'reliable_sad', 'census', 'zncc', 'gradient', 'coarse_to_fine') + WIRE_METHODS or type(min_disparity) is not int or type(max_disparity) is not int or (min_disparity < 1) or (max_disparity <= min_disparity) or (type(block_size) is not int) or (block_size < 3) or (block_size % 2 != 1) or (not 0 < min_depth_m < max_depth_m) or (not all((stere_depth__math.isfinite(x) for x in (min_depth_m, max_depth_m, lr_tolerance_px, texture_threshold, uniqueness_threshold)))) or (lr_tolerance_px < 0) or (texture_threshold < 0) or (not 0 <= uniqueness_threshold < 1) or (type(subpixel) is not bool) or (image_processing_batch_size is not None and (type(image_processing_batch_size) is not int or image_processing_batch_size < 1)):
            raise ValueError('invalid stereo depth configuration')
        if method == 'coarse_to_fine':
            pyramid_mode = 'audit' if pyramid_mode is None else pyramid_mode
            if pyramid_mode not in ('audit', 'accelerated'):
                raise ValueError('pyramid_mode must be audit or accelerated')
        elif pyramid_mode is not None:
            raise ValueError('pyramid_mode requires coarse_to_fine method')
        if method == 'wire_silhouette' and subpixel:
            raise ValueError('wire_silhouette requires integer observed-mask correspondence')
        self.pyramid_mode = pyramid_mode
        self.method = method
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity
        self.block_size = block_size
        self.near = min_depth_m
        self.far = max_depth_m
        self.lr = lr_tolerance_px
        self.texture = texture_threshold
        self.uniqueness = uniqueness_threshold
        self.rectify = rectify
        self.subpixel = subpixel
        self.image_processing_batch_size = image_processing_batch_size
        if method == 'reliable_sad':
            from dh116.physics.stereo_match import resolve_policy
            self.reliability = resolve_policy(reliability)
            if photometric_verification is not None and type(photometric_verification) is not bool:
                raise ValueError('photometric verification must be boolean')
            self.photometric_verification = True if photometric_verification is None else photometric_verification
        elif reliability is not None:
            raise ValueError('reliability policy requires opt-in reliable_sad method')
        else:
            if photometric_verification is not None:
                raise ValueError('photometric verification requires opt-in reliable_sad')
            self.reliability = None
            self.photometric_verification = False

    @classmethod
    def from_config(cls, value):
        value = stere_depth__json.loads(stere_depth__Path(value).read_text(encoding='utf-8')) if isinstance(value, (str, stere_depth__Path)) else value
        if not isinstance(value, dict):
            raise ValueError('stereo depth configuration dictionary required')
        if 'depth_estimator' in value:
            return cls(**value['depth_estimator'])
        if 'depth' in value:
            return cls(**value['depth'])
        if 'K_left' in value:
            from dh116.physics.vision import StereoCalibration as StereoCalibration
            _stere_depth_import__disparity_budget = disparity_budget
            c = StereoCalibration.from_dict(value)
            (near, far) = value.get('depth_range_m', [0.2, 1.2])
            budget = _stere_depth_import__disparity_budget(c, near, far)
            (low, high) = value.get('disparity_range_px', [budget['suggested_min_disparity'], budget['suggested_max_disparity']])
            return cls(min_disparity=low, max_disparity=high, min_depth_m=near, max_depth_m=far)
        return cls(**value)

    def to_config(self):
        result = dict(method=self.method, min_disparity=self.min_disparity, max_disparity=self.max_disparity, block_size=self.block_size, min_depth_m=self.near, max_depth_m=self.far, lr_tolerance_px=self.lr, texture_threshold=self.texture, uniqueness_threshold=self.uniqueness, rectify=self.rectify, subpixel=self.subpixel, image_processing_batch_size=self.image_processing_batch_size)
        if self.reliability is not None:
            result['reliability'] = dict(self.reliability)
            result['photometric_verification'] = self.photometric_verification
        if self.pyramid_mode is not None:
            result['pyramid_mode'] = self.pyramid_mode
        return result

    def _texture(self, gray):
        mean = stere_depth__F.avg_pool2d(gray[:, None], self.block_size, 1, self.block_size // 2)[:, 0]
        second = stere_depth__F.avg_pool2d(gray.square()[:, None], self.block_size, 1, self.block_size // 2)[:, 0]
        return (second - mean.square()).clamp_min(0).sqrt() >= self.texture

    def _match(self, a, b, sign, a_valid=None, b_valid=None, *, return_diagnostics=False):
        (n, h, w) = a.shape
        stop = min(self.max_disparity, w - 1)
        if stop - self.min_disparity + 1 < 2:
            raise ValueError('image width too small for stereo disparity range')
        x = stere_depth__torch.arange(w, device=a.device)[None, None, :]
        y = stere_depth__torch.arange(h, device=a.device)[None, :, None]
        best = stere_depth__torch.full_like(a, float('inf'))
        second = best.clone()
        minus = best.clone()
        plus = best.clone()
        previous = best.clone()
        disparity = stere_depth__torch.full_like(a, float(self.min_disparity))
        radius = self.block_size // 2
        for d in range(self.min_disparity, stop + 1):
            other = x + sign * d
            shifted = b.gather(2, other.clamp(0, w - 1).expand(n, h, w))
            cost = stere_depth__F.avg_pool2d((a - shifted).abs()[:, None], self.block_size, 1, radius)[:, 0]
            support = (x >= radius) & (x < w - radius) & (other >= radius) & (other < w - radius) & (y >= radius) & (y < h - radius)
            if a_valid is not None:
                support = support & a_valid
            if b_valid is not None:
                support = support & b_valid.gather(2, other.clamp(0, w - 1).expand(n, h, w))
            cost = cost.masked_fill(~support, float('inf'))
            plus = stere_depth__torch.where(disparity == d - 1, cost, plus)
            win = cost < best
            minus = stere_depth__torch.where(win, previous, minus)
            plus = stere_depth__torch.where(win, stere_depth__torch.full_like(plus, float('inf')), plus)
            second = stere_depth__torch.minimum(second, stere_depth__torch.where(win, best, cost))
            best = stere_depth__torch.where(win, cost, best)
            disparity = stere_depth__torch.where(win, disparity.new_full((), d), disparity)
            previous = cost
        good = stere_depth__torch.isfinite(best)
        next_cost = stere_depth__torch.where(stere_depth__torch.isfinite(second), second, stere_depth__torch.ones_like(second))
        confidence = ((next_cost - best) / next_cost.clamp_min(1e-08)).clamp(0, 1)
        confidence = stere_depth__torch.where(good, confidence, 0.0)
        integer_disparity = disparity
        integer_good = good
        integer_confidence = confidence
        refinable = stere_depth__torch.zeros_like(good)
        if self.subpixel:
            curvature = minus + plus - 2 * best
            refinable = good & stere_depth__torch.isfinite(minus) & stere_depth__torch.isfinite(plus) & stere_depth__torch.isfinite(curvature) & (curvature > 1e-08)
            numerator = stere_depth__torch.where(refinable, minus - plus, 0.0)
            denominator = stere_depth__torch.where(refinable, curvature, 1.0)
            offset = (0.5 * numerator / denominator).clamp(-0.5, 0.5)
            disparity = disparity + offset
            good = good & refinable
            confidence = stere_depth__torch.where(good, confidence, 0.0)
        result = (disparity, confidence, good)
        if not return_diagnostics:
            return result
        return (*result, dict(integer_disparity=integer_disparity, integer_good=integer_good, integer_confidence=integer_confidence, refinement_valid=refinable, search_boundary=(integer_disparity == self.min_disparity) | (integer_disparity == stop), best_cost=best, second_cost=second, minus_cost=minus, plus_cost=plus))

    def _matched(self, a, b, sign, a_valid, b_valid):
        """Keep the established three-tensor private API compatible."""
        result = self._match(a, b, sign, a_valid, b_valid, return_diagnostics=True)
        if len(result) == 4:
            return result
        (d, c, g) = result
        return (d, c, g, dict(integer_disparity=d, integer_confidence=c, integer_good=g, refinement_valid=g if self.subpixel else stere_depth__torch.zeros_like(g), search_boundary=(d == self.min_disparity) | (d == min(self.max_disparity, a.shape[-1] - 1)), trace_available=False))

    def _reliable_match(self, a, b, sign, a_valid, b_valid, *, raw_a_valid=None, raw_b_valid=None):
        from dh116.physics.stereo_match import refine_winner_neighborhood
        from dh116.physics.stereo_match import photometric_refine_winner
        from dh116.physics.stereo_match import repeated_texture_evidence as repeated_texture_evidence
        (_, _, _, trace) = self._matched(a, b, sign, a_valid, b_valid)
        winner = trace['integer_disparity']
        (n, h, w) = a.shape
        r = self.block_size // 2
        x = stere_depth__torch.arange(w, device=a.device)[None, None, :]
        y = stere_depth__torch.arange(h, device=a.device)[None, :, None]
        costs = [stere_depth__torch.full_like(a, float('inf')) for _ in range(5)]
        far = stere_depth__torch.full_like(a, float('inf'))
        for d in range(self.min_disparity, min(self.max_disparity, w - 1) + 1):
            other = x + sign * d
            shifted = b.gather(2, other.clamp(0, w - 1).expand(n, h, w))
            cost = stere_depth__F.avg_pool2d((a - shifted).abs()[:, None], self.block_size, 1, r)[:, 0]
            supported = (x >= r) & (x < w - r) & (other >= r) & (other < w - r) & (y >= r) & (y < h - r) & a_valid
            supported &= b_valid.gather(2, other.clamp(0, w - 1).expand(n, h, w))
            cost = cost.masked_fill(~supported, float('inf'))
            for (i, offset) in enumerate((-2, -1, 0, 1, 2)):
                costs[i] = stere_depth__torch.where(winner + offset == d, cost, costs[i])
            far = stere_depth__torch.minimum(far, stere_depth__torch.where((winner - d).abs() >= 2, cost, float('inf')))
        raw = a_valid if raw_a_valid is None else raw_a_valid
        gx = raw[:, :, 1:] & raw[:, :, :-1]
        gy = raw[:, 1:] & raw[:, :-1]
        dx = stere_depth__F.pad(stere_depth__torch.where(gx, (a[:, :, 1:] - a[:, :, :-1]).abs(), 0.0), (0, 1))
        dy = stere_depth__F.pad(stere_depth__torch.where(gy, (a[:, 1:] - a[:, :-1]).abs(), 0.0), (0, 0, 0, 1))
        gradient = dx + dy
        mean = stere_depth__F.avg_pool2d(gradient[:, None], self.block_size, 1, r)[:, 0]
        maximum = stere_depth__F.max_pool2d(gradient[:, None], self.block_size, 1, r)[:, 0]
        image_ambiguity = (maximum > 0.02) & (maximum > 8 * mean.clamp_min(1e-08))
        primary = refine_winner_neighborhood(winner, *costs, second_nonlocal_cost=far, available=stere_depth__torch.stack([stere_depth__torch.isfinite(c) for c in costs]).all(0), image_ambiguity=image_ambiguity, policy=self.reliability)
        confidence = ((far - trace['best_cost']) / far.abs().clamp_min(1e-08)).clamp(0, 1)
        confidence = stere_depth__torch.where(stere_depth__torch.isfinite(far) & stere_depth__torch.isfinite(trace['best_cost']), confidence, 0.0)
        d = primary['disparity_px']
        good = trace['integer_good'] & primary['valid_mask']
        refined_valid = primary['refinement_valid']
        model = primary['model_selected']
        photo = None
        if self.photometric_verification:
            photo = photometric_refine_winner(a, b, winner, sign, self.block_size, a_valid if raw_a_valid is None else raw_a_valid, b_valid if raw_b_valid is None else raw_b_valid, min_curvature=self.reliability['min_curvature'], mode_margin=self.reliability['mode_margin'])
            use_photo = ~refined_valid & trace['integer_good'] & ~image_ambiguity & self._texture(a) & (confidence >= self.uniqueness) & ~trace['search_boundary'] & photo['valid_mask']
            d = stere_depth__torch.where(use_photo, photo['disparity_px'], d)
            good = good | use_photo
            refined_valid = refined_valid | use_photo
            model = stere_depth__torch.where(use_photo, photo['model_selected'], model)
        periodic = repeated_texture_evidence(a, raw, self.max_disparity - self.min_disparity)
        period = periodic['period_px']
        alias_in_search = (winner - period >= self.min_disparity) | (winner + period <= min(self.max_disparity, w - 1))
        repeated = periodic['repeated_texture'] & alias_in_search
        good = good & ~repeated
        refined_valid = refined_valid & ~repeated
        model = stere_depth__torch.where(good, model, 0)
        refined = dict(primary=primary, photometric=photo, disparity_px=d, valid_mask=good, refinement_valid=refined_valid, model_selected=model, periodicity=periodic, repeated_texture_rejected=repeated, photometric_verification_enabled=self.photometric_verification)
        trace.update(refinement=refined, refinement_valid=refined_valid, rgb_boundary_proxy=image_ambiguity)
        return (d, confidence, good, trace)

    @staticmethod
    def sample_horizontal(value, x, valid):
        """Linear sample at floating pixel indices; both used neighbours valid."""
        if value.shape != x.shape or valid.shape != value.shape or valid.dtype != stere_depth__torch.bool:
            raise ValueError('matching [N,H,W] value/float x/bool validity required')
        w = value.shape[-1]
        inside = stere_depth__torch.isfinite(x) & (x >= 0) & (x <= w - 1)
        safe = stere_depth__torch.where(inside, x, 0.0)
        low = safe.floor().long()
        high = (low + 1).clamp_max(w - 1)
        fraction = safe - low
        lo = value.gather(2, low)
        hi = value.gather(2, high)
        unused_high = fraction <= 1e-06
        accepted = inside & valid.gather(2, low) & stere_depth__torch.isfinite(lo) & (unused_high | valid.gather(2, high) & stere_depth__torch.isfinite(hi))
        safe_hi = stere_depth__torch.where(unused_high & ~stere_depth__torch.isfinite(hi), lo, hi)
        return (lo + (safe_hi - lo) * fraction, accepted)

    def _sgbm(self, left, right):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError('OpenCV cv2 is required for optional StereoSGBM; use blockmatch without installing it') from error
        import numpy as np
        count = stere_depth__math.ceil((self.max_disparity - self.min_disparity + 1) / 16) * 16
        kwargs = dict(numDisparities=count, blockSize=self.block_size, P1=8 * self.block_size ** 2, P2=32 * self.block_size ** 2, uniquenessRatio=int(self.uniqueness * 100), speckleWindowSize=0)
        lmatcher = cv2.StereoSGBM_create(minDisparity=self.min_disparity, **kwargs)
        rmatcher = cv2.StereoSGBM_create(minDisparity=-self.min_disparity - count + 1, **kwargs)
        ls = []
        rs = []
        for (a, b) in zip(left.detach().cpu().numpy(), right.detach().cpu().numpy()):
            a = np.rint(a * 255).astype(np.uint8)
            b = np.rint(b * 255).astype(np.uint8)
            ls.append(lmatcher.compute(a, b).astype(np.float32) / 16)
            rs.append(-rmatcher.compute(b, a).astype(np.float32) / 16)
        dl = stere_depth__torch.as_tensor(np.stack(ls), device=left.device, dtype=left.dtype)
        dr = stere_depth__torch.as_tensor(np.stack(rs), device=left.device, dtype=left.dtype)
        return (dl, dr)

    @stere_depth__torch.no_grad()
    def _compute_batch(self, left_rgb, right_rgb, calibration, left_valid, right_valid, wire_masks=None):
        left = left_rgb.mean(1).float()
        right = right_rgb.mean(1).float()
        masks = []
        raw_masks = []
        for mask in (left_valid, right_valid):
            if mask is None:
                mask = stere_depth__torch.ones_like(left, dtype=stere_depth__torch.bool)
            if not isinstance(mask, stere_depth__torch.Tensor) or mask.shape != left.shape or mask.dtype != stere_depth__torch.bool or (mask.device != left.device):
                raise ValueError('rectification validity requires bool [B,H,W] on RGB device')
            raw_masks.append(mask)
            masks.append(stere_depth__F.avg_pool2d(mask.float()[:, None], self.block_size, 1, self.block_size // 2)[:, 0] >= 1.0)
        if min(left.shape[-2:]) < self.block_size:
            raise ValueError('stereo image smaller than block size')
        if self.method in ('blockmatch', 'reliable_sad'):
            if self.method == 'reliable_sad' and self.subpixel:
                (dl, cl, gl, left_trace) = self._reliable_match(left, right, -1, masks[0], masks[1], raw_a_valid=raw_masks[0], raw_b_valid=raw_masks[1])
                (dr, cr, gr, right_trace) = self._reliable_match(right, left, 1, masks[1], masks[0], raw_a_valid=raw_masks[1], raw_b_valid=raw_masks[0])
            else:
                (dl, cl, gl, left_trace) = self._matched(left, right, -1, masks[0], masks[1])
                (dr, cr, gr, right_trace) = self._matched(right, left, 1, masks[1], masks[0])
        elif self.method in WIRE_METHODS:
            lm = (wire_colour_mask(left_rgb) if wire_masks is None else wire_masks[0]) & raw_masks[0]
            rm = (wire_colour_mask(right_rgb) if wire_masks is None else wire_masks[1]) & raw_masks[1]
            options = dict(method=self.method, low=self.min_disparity, high=self.max_disparity, block=self.block_size, texture=self.texture, uniqueness=self.uniqueness, subpixel=self.subpixel)
            (dl, cl, gl, left_trace) = match_wire(left, right, lm, rm, -1, **options)
            (dr, cr, gr, right_trace) = match_wire(right, left, rm, lm, 1, **options)
        elif self.method in ('census', 'zncc', 'gradient', 'coarse_to_fine'):
            kwargs = dict(min_disparity=self.min_disparity, max_disparity=self.max_disparity, block_size=self.block_size, subpixel=self.subpixel, uniqueness_threshold=self.uniqueness, texture_threshold=self.texture)
            if self.method == 'coarse_to_fine':
                _stere_depth_import__match_pyramid = match_pyramid

                def run(a, b, sign, ma, mb):
                    return _stere_depth_import__match_pyramid(a, b, sign, calibration=calibration, a_valid=ma, b_valid=mb, pyramid_mode=self.pyramid_mode, **kwargs)
            else:
                from dh116.physics.stereo_match import match_cost

                def run(a, b, sign, ma, mb):
                    return match_cost(a, b, sign, method=self.method, a_valid=ma, b_valid=mb, **kwargs)
            (dl, cl, gl, left_trace) = run(left, right, -1, raw_masks[0], raw_masks[1])
            (dr, cr, gr, right_trace) = run(right, left, 1, raw_masks[1], raw_masks[0])
        else:
            (dl, dr) = self._sgbm(left, right)
            gl = (dl >= self.min_disparity) & (dl <= self.max_disparity)
            gr = (dr >= self.min_disparity) & (dr <= self.max_disparity)
            cl = gl.float()
            cr = gr.float()
            left_trace = dict(search_boundary=(dl == self.min_disparity) | (dl == self.max_disparity))
        (n, h, w) = left.shape
        x = stere_depth__torch.arange(w, device=left.device, dtype=dl.dtype)[None, None, :].expand(n, h, w)
        xr = x - dl
        right_texture = self._texture(right)
        left_texture = self._texture(left)
        (other_d, other_valid) = self.sample_horizontal(dr, xr, gr & right_texture & masks[1])
        (other_confidence, confidence_valid) = self.sample_horizontal(cr, xr, gr)
        occlusion = ~other_valid | ~confidence_valid | ((dl - other_d).abs() > self.lr)
        confidence = stere_depth__torch.minimum(cl, other_confidence)
        disparity_valid = stere_depth__torch.isfinite(dl) & (dl > 0)
        denominator = stere_depth__torch.where(disparity_valid, dl, float('nan'))
        z = float(calibration.K_left[0, 0]) * calibration.baseline_m / denominator
        valid = gl & disparity_valid & ~occlusion & left_texture & masks[0] & (confidence >= self.uniqueness) & stere_depth__torch.isfinite(z) & (z >= self.near) & (z <= self.far)
        radius = self.block_size // 2
        valid[:, :radius] = False
        valid[:, -radius:] = False
        valid[:, :, :radius] = False
        valid[:, :, -radius:] = False
        missing = stere_depth__torch.full_like(z, float('nan'))
        (right_texture_at_left, right_texture_inside) = self.sample_horizontal(right_texture.float(), xr, stere_depth__torch.ones_like(gr))
        patch_border = ~masks[0]
        patch_border[:, :radius] = True
        patch_border[:, -radius:] = True
        patch_border[:, :, :radius] = True
        patch_border[:, :, -radius:] = True
        subpixel_rejected = None if self.method == 'sgbm' else stere_depth__torch.zeros_like(gl)
        if self.subpixel and self.method not in ('sgbm',) + WIRE_METHODS:
            subpixel_rejected = ~left_trace['refinement_valid']
        reasons = dict(low_texture_left=~left_texture, low_texture_right=right_texture_inside & (right_texture_at_left < 1 - 1e-06), search_boundary=left_trace['search_boundary'], left_right_inconsistent=other_valid & confidence_valid & ((dl - other_d).abs() > self.lr), uniqueness=confidence < self.uniqueness if self.method != 'sgbm' else None, occlusion_composite=occlusion, true_occlusion=None, depth_range=~disparity_valid | ~stere_depth__torch.isfinite(z) | (z < self.near) | (z > self.far), subpixel_rejected=subpixel_rejected, patch_border=patch_border, out_of_right_view=~right_texture_inside, left_matcher_rejected=~gl, right_sample_unavailable=~other_valid)
        if 'refinement' in left_trace:
            reasons['repeated_texture'] = left_trace['refinement']['repeated_texture_rejected']
        for (name, value) in left_trace.get('reasons', {}).items():
            reasons['candidate_' + name] = value
        diagnostics = dict(version='dh116-stereo-matcher-diagnostics-v1', method=self.method, confidence_kind='compatibility_validity_indicator' if self.method == 'sgbm' else 'relative_cost_margin', confidence_is_probability=False, quality_confidence_available=self.method != 'sgbm', true_occlusion_available=False, refinement_scope='OpenCV_internal_not_exposed' if self.method == 'sgbm' else 'wrapper_rgb_cost_refinement', wrapper_refinement_applied=self.subpixel and self.method != 'sgbm', reasons=reasons, uniqueness_scope='OpenCV_internal_not_exposed' if self.method == 'sgbm' else 'separated_integer_modes' if self.method in ('census', 'zncc', 'gradient', 'coarse_to_fine') + WIRE_METHODS or (self.method == 'reliable_sad' and self.subpixel) else 'adjacent_top2_cost_margin', raw_disparity_px=dl, raw_confidence=confidence, candidate_enabled=self.method not in ('blockmatch', 'sgbm'), rgb_boundary_proxy=dict(minimum_gray_gradient=0.02, maximum_to_mean_gradient_ratio=8.0, window_px=self.block_size), rgb_boundary_proxy_applied=self.method == 'reliable_sad' and self.subpixel, periodicity_scope='single_tone_recurrence_and_separated_integer_modes;not_proof_all_repeated_patterns_absent', rgb_boundary_proxy_scope='gradient_concentration_only;not_semantic_background_or_occlusion_truth')
        if 'refinement' in left_trace:
            diagnostics['refinement'] = left_trace['refinement']
        for name in ('pyramid', 'metric', 'subpixel_policy', 'resource_metadata', 'optional_refinement_used'):
            if name in left_trace:
                diagnostics[name] = left_trace[name]
        if self.pyramid_mode is not None:
            diagnostics['pyramid_right'] = right_trace['pyramid']
        return dict(depth_m=stere_depth__torch.where(valid, z, missing), disparity_px=stere_depth__torch.where(valid, dl, missing), valid=valid, occlusion=occlusion, confidence=stere_depth__torch.where(valid, confidence, 0.0), diagnostics=diagnostics)

    @stere_depth__torch.no_grad()
    def compute(self, left_rgb, right_rgb, calibration, *, left_valid=None, right_valid=None, wire_masks=None):
        stere_depth__validate_images(left_rgb, right_rgb, calibration)
        if wire_masks is not None:
            if self.method not in WIRE_METHODS or len(wire_masks) != 2 or any((not isinstance(mask, stere_depth__torch.Tensor) or mask.shape != left_rgb[:, 0].shape or mask.dtype != stere_depth__torch.bool or (mask.device != left_rgb.device) for mask in wire_masks)):
                raise ValueError('both aligned RGB-derived segmentation masks required for wire methods')
        if not calibration.rectified:
            raise ValueError('unrectified stereo: call rectify_rgb explicitly and use its returned rectified calibration for depth AND world geometry')
        n = len(left_rgb)
        for mask in (left_valid, right_valid):
            if mask is not None and (not isinstance(mask, stere_depth__torch.Tensor) or mask.shape != left_rgb[:, 0].shape or mask.dtype != stere_depth__torch.bool or (mask.device != left_rgb.device)):
                raise ValueError('rectification validity requires bool [B,H,W] on RGB device')
        batch = min(n, self.image_processing_batch_size or n)
        chunks = []
        for start in range(0, n, batch):
            stop = min(n, start + batch)
            chunks.append(self._compute_batch(left_rgb[start:stop], right_rgb[start:stop], calibration, None if left_valid is None else left_valid[start:stop], None if right_valid is None else right_valid[start:stop], None if wire_masks is None else tuple((mask[start:stop] for mask in wire_masks))))
        result = {key: stere_depth__torch.cat([chunk[key] for chunk in chunks], 0) for key in ('depth_m', 'disparity_px', 'valid', 'occlusion', 'confidence')}

        def merge(values):
            first = values[0]
            if isinstance(first, stere_depth__torch.Tensor):
                return stere_depth__torch.cat(values, 0)
            if isinstance(first, dict):
                return {key: merge([v[key] for v in values]) for key in first}
            if any((value != first for value in values)):
                raise ValueError('image chunks changed matcher diagnostics contract')
            return first
        result['diagnostics'] = merge([chunk['diagnostics'] for chunk in chunks])
        result['valid_mask'] = result['valid']
        result['occlusion_mask'] = result['occlusion']
        self.last_resource_metadata = dict(method=self.method, subpixel=self.subpixel, image_processing_batch_size=batch, processed_images=n, input_resolution_wh=[calibration.width, calibration.height], output_resolution_wh=[calibration.width, calibration.height], hidden_image_resize=False, cost_volume_allocated=False if self.method != 'sgbm' else None, disparity_planes_in_memory=1 if self.method != 'sgbm' else None, cached_cost_state_planes=6 if self.method == 'blockmatch' else None, configured_disparity_range=[self.min_disparity, self.max_disparity], gpu_memory_measured=False, gpu_peak_allocated_bytes=None, gpu_peak_reserved_bytes=None)
        if self.pyramid_mode is not None:
            self.last_resource_metadata['pyramid_mode'] = self.pyramid_mode
        if self.method in WIRE_METHODS:
            self.last_resource_metadata.update(disparity_planes_in_memory=None, sparse_center_cost_tiles=True)
        return result

def _verified_stereo(proof):
    from dh116.rl.qualification import measure_stereo_evidence
    return measure_stereo_evidence(proof)

def usd_aperture_intrinsics(width, height, focal_length, horizontal_aperture, vertical_aperture, horizontal_offset, vertical_offset):
    """Recover optical K from finite USD lens data expressed in common units."""
    values = tuple((float(x) for x in (focal_length, horizontal_aperture, vertical_aperture, horizontal_offset, vertical_offset)))
    if type(width) is not int or type(height) is not int or min(width, height) < 1 or (not all((stere_calib__math.isfinite(x) for x in values))) or (min(values[:3]) <= 0):
        raise ValueError('finite USD lens data and positive dimensions required')
    (focal, horizontal, vertical, ox, oy) = values
    (fx, fy) = (width * focal / horizontal, height * focal / vertical)
    return stere_calib__torch.tensor([[fx, 0.0, width / 2 - ox * fx / focal], [0.0, fy, height / 2 + oy * fy / focal], [0.0, 0.0, 1.0]], dtype=stere_calib__torch.float64)
stere_prepr__StereoDepth = StereoDepth

def compare_resolution_results(reports, *, thresholds=None):
    """Only the common raw-proof verifier can supply native measured values.

    Unverified report metrics are deliberately ignored. GPU/timing values must
    be returned by that same verifier after its performance-artifact checks.
    No CPU-zero memory statistic, resolution preference or invented accuracy
    is used to choose a profile.
    """
    rows = []
    for source in reports:
        report = stere_resol__json.loads(stere_resol__Path(source).read_text(encoding='utf-8')) if isinstance(source, (str, stere_resol__Path)) else source
        if not isinstance(report, dict):
            raise ValueError('resolution report dictionary or JSON path required')
        (width, height) = validate_resolution(report.get('width'), report.get('height'), allow_resource_override=True)
        row = dict(width=width, height=height, metrics={k: None for k in METRIC_KEYS}, wire_depth_metrics={k: None for k in WIRE_DEPTH_KEYS}, stereo_pairs_per_second=None, qualified=False, measurements_verified=False, cohort_sha256=None, sample_count=None, capture_depth_pairs_per_second=None, capture_depth_throughput_scope='capture_plus_depth_only;excludes_Actor_and_physics;not_used_for_selection', rgb_candidate_point_cloud_completeness=None, gpu_device_used_bytes=None, blocked_reasons=[], accuracy_source='NA', performance_source='NA')
        binding = None
        if report.get('source_scope') == 'native' and report.get('schema_only') is False:
            try:
                binding = _verified_stereo(report.get('stereo_qualification', report.get('qualification')))
                if not isinstance(binding, dict) or binding.get('measurements_verified') is not True or binding.get('source_scope') != 'native' or ((binding.get('width'), binding.get('height')) != (width, height)):
                    raise ValueError('qualified resolution/calibration source mismatch')
            except (ValueError, FileNotFoundError, ImportError) as error:
                row['blocked_reasons'].append('native_stereo_proof_unavailable: ' + str(error))
                binding = None
        else:
            row['blocked_reasons'].append('CPU_fixture_or_non_native_source')
        if binding is not None:
            row['qualified'] = binding.get('qualified') is True
            row['measurements_verified'] = True
            row['accuracy_source'] = 'verified_native_raw_stereo_proof'
            if not row['qualified']:
                row['blocked_reasons'].append('measured_native_accuracy_did_not_pass_qualification_thresholds')
            metrics = binding.get('metrics', {})
            row['wire_depth_metrics'] = {key: _number(metrics.get(key)) for key in WIRE_DEPTH_KEYS}
            for (dst, src) in (('depth_mae_m', 'mae_m'), ('depth_rmse_m', 'rmse_m'), ('depth_p95_m', 'p95_m'), ('wire_point_cloud_completeness', 'wire_point_cloud_completeness')):
                row['metrics'][dst] = _number(metrics.get(src))
            row['cohort_sha256'] = binding.get('cohort_sha256')
            row['rgb_candidate_point_cloud_completeness'] = _number(metrics.get('rgb_candidate_point_cloud_completeness'))
            row['sample_count'] = _number(binding.get('sample_count', metrics.get('frame_count')))
            performance = binding.get('performance')
            if isinstance(performance, dict) and performance.get('source_scope') == 'native' and (performance.get('measurement_device') == 'cuda'):
                row['gpu_device_used_bytes'] = _number(performance.get('gpu_device_used_bytes'))
                for key in METRIC_KEYS[4:]:
                    row['metrics'][key] = _number(performance.get(key))
                (allocated, reserved) = (row['metrics'][key] for key in ('gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes'))
                if allocated is None or reserved is None or allocated <= 0 or (reserved < allocated):
                    for key in METRIC_KEYS[4:7]:
                        row['metrics'][key] = None
                    row['blocked_reasons'].append('GPU_measurements_missing_or_inconsistent')
                else:
                    row['performance_source'] = 'verified_native_cuda_artifact'
                times = [row['metrics'][key] for key in ('capture_ms', 'depth_ms', 'actor_ms')]
                batch = _number(performance.get('batch_size'), positive=True)
                capture_depth = [row['metrics'][key] for key in ('capture_ms', 'depth_ms')]
                if batch is not None and all((value is not None for value in capture_depth)) and (sum(capture_depth) > 0):
                    row['capture_depth_pairs_per_second'] = 1000 * batch / sum(capture_depth)
                if batch is not None and all((value is not None for value in times)) and (sum(times) > 0):
                    row['stereo_pairs_per_second'] = 1000 * batch / sum(times)
            else:
                row['blocked_reasons'].append('native_GPU_and_timing_proof_unavailable')
        rows.append(row)
    result = dict(version='dh116-stereo-resolution-comparison-v1', resolutions=rows, native_reports=sum((row['measurements_verified'] for row in rows)), qualified_native_reports=sum((row['qualified'] for row in rows)), selected_resolution=None, selection_basis='verified_same_cohort_measured_metrics_only', selection_reasons=[])
    required = ('max_depth_mae_m', 'max_depth_rmse_m', 'max_depth_p95_m', 'min_wire_point_cloud_completeness', 'min_sample_count', 'min_stereo_pairs_per_second')
    if thresholds is None:
        result['selection_reasons'].append('explicit_accuracy_coverage_sample_and_throughput_thresholds_required')
        return result
    if not isinstance(thresholds, dict) or any((_number(thresholds.get(key)) is None for key in required)):
        raise ValueError('complete finite resolution selection thresholds required')
    measured = [row for row in rows if row['measurements_verified']]
    native = [row for row in measured if row['qualified']]
    cohorts = {row['cohort_sha256'] for row in measured}
    if len(cohorts) != 1 or None in cohorts:
        result['selection_reasons'].append('verified_actual_same_cohort_required')
        return result
    eligible = []
    for row in native:
        m = row['metrics']
        if any((m[key] is None for key in METRIC_KEYS)) or row['sample_count'] is None or row['stereo_pairs_per_second'] is None:
            continue
        if m['depth_mae_m'] <= thresholds['max_depth_mae_m'] and m['depth_rmse_m'] <= thresholds['max_depth_rmse_m'] and (m['depth_p95_m'] <= thresholds['max_depth_p95_m']) and (m['wire_point_cloud_completeness'] >= thresholds['min_wire_point_cloud_completeness']) and (row['sample_count'] >= thresholds['min_sample_count']) and (row['stereo_pairs_per_second'] >= thresholds['min_stereo_pairs_per_second']) and ('max_gpu_peak_reserved_bytes' not in thresholds or m['gpu_peak_reserved_bytes'] <= thresholds['max_gpu_peak_reserved_bytes']):
            eligible.append(row)
    if eligible:
        selected = min(eligible, key=lambda row: (row['metrics']['depth_rmse_m'], row['metrics']['depth_p95_m'], -row['stereo_pairs_per_second']))
        result['selected_resolution'] = [selected['width'], selected['height']]
        result['selection_reasons'].append('minimum_measured_RMSE_then_P95_then_maximum_measured_throughput_among_threshold_passers')
    else:
        result['selection_reasons'].append('no_complete_qualified_resolution_meets_all_thresholds')
    return result
stere_prepr__StereoCalibration = StereoCalibration
stere_pyram__StereoCalibration = StereoCalibration
stere_pyram__resize_calibration = stere_resol__resize_calibration
from dh116.env.contracts import stere_contr__digest as digest
_PLANE_COUNTERS = ('integer_plane_calls', 'fractional_plane_calls', 'integer_cost_evaluations', 'fractional_cost_evaluations', 'total_cost_evaluations')
VERSION = 'dh116-stereo-processing-v1'

def _search_totals(image, *traces):
    """Keep executed plane arithmetic separate from eligible center counts."""
    empty = stere_pyram__torch.zeros(len(image), dtype=stere_pyram__torch.int64, device=image.device)
    out = {key: empty.clone() for key in _PLANE_COUNTERS}
    out.update(integer_candidate_count=empty.clone(), fractional_candidate_count=empty.clone())
    for trace in traces:
        stats = trace.get('search_statistics', {}) if trace is not None else {}
        for key in _PLANE_COUNTERS:
            if key in stats:
                out[key] += stats[key]
        for kind in ('integer', 'fractional'):
            key = kind + '_candidates'
            if key in stats:
                out[kind + '_candidate_count'] += stats[key].flatten(1).sum(1)
    return out

def _sources():
    names = ('physics/vision.py', 'physics/wire.py', 'physics/stereo_match.py', 'env/contracts.py')
    return {name: stere_prepr__hashlib.sha256(source_file(name).read_bytes()).hexdigest() for name in names}

def _choose_trace(local, global_result, use_local):
    """Choose diagnostics at the same centers as the returned disparity."""
    if global_result is None:
        return dict(local)
    selected = dict(global_result)
    for (name, global_value) in global_result.items():
        local_value = local.get(name)
        if isinstance(global_value, stere_pyram__torch.Tensor) and global_value.shape == use_local.shape:
            selected[name] = stere_pyram__torch.where(use_local, local_value, global_value)
        elif name == 'reasons':
            selected[name] = {key: stere_pyram__torch.where(use_local, local_value[key], value) for (key, value) in global_value.items()}
    return selected

def processing_contract(camera_config):
    config = stere_prepr__json.loads(stere_prepr__Path(camera_config).read_text(encoding='utf-8')) if isinstance(camera_config, (str, stere_prepr__Path)) else deepcopy(camera_config)
    if not isinstance(config, dict):
        raise ValueError('actual camera configuration dictionary/path required')
    calibration = stere_prepr__StereoCalibration.from_dict(config)
    depth = stere_prepr__StereoDepth.from_config(config).to_config()
    depth.pop('image_processing_batch_size', None)
    for key in ('min_depth_m', 'max_depth_m', 'lr_tolerance_px', 'texture_threshold', 'uniqueness_threshold'):
        depth[key] = float(depth[key])
    tracking = config.get('wire_tracking_mode', 'normalized_visible_components')
    if tracking not in ('legacy', 'normalized_visible_components'):
        raise ValueError('explicit supported wire tracking mode required')
    point_sampling = config.get('wire_point_sampling', 'centerline')
    if point_sampling not in ('centerline', 'observed_ribbon') or (tracking == 'legacy' and point_sampling != 'centerline'):
        raise ValueError('observed_ribbon requires normalized visible component tracking')
    ribbon_radius_px = config.get('wire_ribbon_radius_px', 1)
    if type(ribbon_radius_px) is not int or not 1 <= ribbon_radius_px <= 3 or (point_sampling == 'centerline' and ribbon_radius_px != 1):
        raise ValueError('ribbon radius must be 1..3 pixels in observed_ribbon mode')
    surface_capacity = config.get('wire_observed_surface_capacity', 32 if point_sampling == 'observed_ribbon' else 0)
    if type(surface_capacity) is not int or not 0 <= surface_capacity <= 126 or (point_sampling == 'centerline' and surface_capacity):
        raise ValueError('invalid observed surface capacity')
    result = dict(version=VERSION, calibration_sha256=calibration.sha256, image_resolution_wh=[calibration.width, calibration.height], depth_estimator=depth, rgb=dict(channel_order='RGB', layout='NCHW', range=[0.0, 1.0], dtype='float32', byte_conversion='integer_or_float_above_one_div_255', gray='arithmetic_mean_RGB', hidden_resize=False), geometry=dict(depth_unit='m', pixel_centres='u+0.5,v+0.5', optical_axes='+X_right,+Y_down,+Z_forward', transform='T_world_left_times_optical_point', calibration_rectified=calibration.rectified, rectification='explicit_only;no_hidden_rectify', missing='NaN_with_false_validity;no_simulator_or_rendered_depth_fallback'), perception=dict(algorithm='warm_colour_visible_simple_path_ZhangSuen_v1' if tracking == 'legacy' else 'normalized_visible_components_tracking_v2', point_count=128, point_sampling=point_sampling, ribbon_radius_px=ribbon_radius_px, observed_surface_capacity=surface_capacity, thresholds=dict(red_over_green=1.07, green_over_blue=1.1, minimum_red=0.12), thinning_max_iterations=128, identity_motion_limit_m=0.005, ordering='observed_visible_arc_length;dominant_world_axis_endpoint_orientation' if tracking == 'legacy' else 'component_metric_arc_when_complete;2D_arc_partial_depth;temporal_orientation', temporal_identity='endpoint_consistency_diagnostic;not_native_identity_qualification' if tracking == 'legacy' else 'endpoint_tangent_curvature_arc_prediction;ambiguous_unknown;not_material_point_identity', missing_or_branched='invalid_tokens;never_invent_hidden_segments' if tracking == 'legacy' else 'RGB_path_before_depth;reliable_components_separate;never_bridge_hidden_or_branched_segments'), source_sha256=_sources(), resource_fields_excluded=['image_processing_batch_size'], native_qualified=False)
    segmentation = config.get('wire_segmentation')
    if segmentation is not None:
        from dh116.env.contracts import require_sha
        from dh116.physics.wire import MASK_POLICY_VERSION
        from dh116.physics.wire import configured_segmentation_mask_policy
        if tracking == 'legacy' or depth['method'] not in ('wire_census', 'wire_zncc', 'wire_silhouette'):
            raise ValueError('RGB segmentation requires tracked wire matcher')
        policy = configured_segmentation_mask_policy(segmentation)
        result['perception']['segmentation'] = dict(checkpoint_sha256=require_sha(segmentation.get('checkpoint_sha256'), 'segmentation checkpoint'), diagnostic_only=True, source='learned_left_right_RGB', native_qualified=False, mask_policy=policy, mask_policy_version=MASK_POLICY_VERSION)
        result['perception']['thresholds'] = None
    result['processing_sha256'] = digest(result)
    return result

@stere_pyram__torch.no_grad()
def match_pyramid(a, b, sign, *, calibration, min_disparity, max_disparity, block_size, subpixel, uniqueness_threshold, texture_threshold, a_valid=None, b_valid=None, matcher=None, pyramid_mode='audit'):
    """Return disparity/confidence/validity/trace; input masks are RAW [B,H,W].

    ``matcher`` is a private dependency-injection hook for orchestration tests.
    The default backend is the bounded-plane ``match_cost(method='zncc')``.
    Requested .25/.5/1 scales use floor dimensions (minimum one pixel); every
    intrinsic and disparity conversion uses the resulting exact size ratios.
    Audit runs full-range validation at each finer level. Accelerated uses
    non-boundary coarse priors and accepts supported interior local matches
    with a cost margin. Every unresolved center is sent to full-range fallback.
    Local acceptance cannot establish uniqueness outside the searched window.
    """
    if not isinstance(a, stere_pyram__torch.Tensor) or not isinstance(b, stere_pyram__torch.Tensor) or a.ndim != 3 or (a.shape != b.shape) or (min(a.shape) < 1) or (a.device != b.device) or (a.dtype != b.dtype) or (not a.is_floating_point()):
        raise ValueError('matching floating grayscale [B,H,W] images required')
    if not isinstance(calibration, stere_pyram__StereoCalibration) or (calibration.height, calibration.width) != a.shape[-2:] or (not calibration.rectified):
        raise ValueError('matching rectified image calibration required')
    if type(sign) is not int or sign not in (-1, 1) or type(min_disparity) is not int or (type(max_disparity) is not int) or (not 1 <= min_disparity < max_disparity) or (type(block_size) is not int) or (block_size < 3) or (block_size % 2 != 1) or (type(subpixel) is not bool) or (not stere_pyram__math.isfinite(uniqueness_threshold)) or (not 0 <= uniqueness_threshold < 1) or (not stere_pyram__math.isfinite(texture_threshold)) or (texture_threshold < 0):
        raise ValueError('invalid stereo pyramid configuration')
    if pyramid_mode not in ('audit', 'accelerated'):
        raise ValueError('pyramid_mode must be audit or accelerated')
    accelerated = pyramid_mode == 'accelerated'
    trust_margin = max(uniqueness_threshold, 0.05)
    (am, bm) = (_raw_mask(a, a_valid), _raw_mask(b, b_valid))
    (height, width) = a.shape[-2:]
    levels = {}
    previous = None
    previous_width = None
    final = None
    totals = _search_totals(a)
    for (name, requested_scale) in (('quarter', 0.25), ('half', 0.5), ('full', 1.0)):
        (w, h) = (max(1, int(width * requested_scale)), max(1, int(height * requested_scale)))
        (sx, sy) = (w / width, h / height)
        (low, high) = (stere_pyram__math.floor(min_disparity * sx), min(w - 1, stere_pyram__math.ceil(max_disparity * sx)))
        resized_calibration = stere_pyram__resize_calibration(calibration, w, h, allow_resource_override=True)
        level = dict(requested_scale=requested_scale, resolution_wh=[w, h], scale_wh=[sx, sy], K_left=resized_calibration.K_left.tolist(), K_right=resized_calibration.K_right.tolist(), disparity_range_px=[low, high], disparity_scale_from_previous=None if previous_width is None else w / previous_width, search_radius_px=2, status='skipped')
        levels[name] = level
        if min(h, w) < block_size:
            level['skip_reason'] = 'image_smaller_than_block_size'
            continue
        if high < low:
            level['skip_reason'] = 'empty_disparity_range'
            continue
        if matcher is None:
            from dh116.physics.stereo_match import match_cost
            matcher = match_cost
        (ai, av) = _downsample(a, am, (h, w))
        (bi, bv) = _downsample(b, bm, (h, w))
        kwargs = dict(method='zncc', min_disparity=low, max_disparity=high, block_size=block_size, subpixel=subpixel, uniqueness_threshold=uniqueness_threshold, texture_threshold=texture_threshold, a_valid=av, b_valid=bv, count_search=True)
        prior = None
        prior_valid = stere_pyram__torch.zeros_like(ai, dtype=stere_pyram__torch.bool)
        local = None
        local_trace = None
        if previous is not None:
            (prior, prior_valid) = _resize_prior(previous[0], previous[2], (h, w))
            options = {'prune_search': True} if accelerated else {}
            local = matcher(ai, bi, sign, prior=prior, search_radius=2, **options, **kwargs)
        local_disagreement = stere_pyram__torch.zeros_like(prior_valid)
        local_failed = stere_pyram__torch.ones_like(prior_valid)
        selected_local = stere_pyram__torch.zeros_like(prior_valid)
        if local is not None:
            (ld, lc, lg, local_trace) = local
            lg = lg & stere_pyram__torch.isfinite(ld) & stere_pyram__torch.isfinite(lc)
            local_failed = ~prior_valid | ~lg | local_trace.get('local_failed', stere_pyram__torch.zeros_like(lg)) | local_trace.get('search_boundary', stere_pyram__torch.zeros_like(lg))
            if accelerated:
                local_failed |= lc < trust_margin
                local_failed |= local_trace.get('reasons', {}).get('fractional_tie', stere_pyram__torch.zeros_like(lg))
                if subpixel:
                    local_failed |= ~local_trace.get('refinement_valid', stere_pyram__torch.zeros_like(lg))
                selected_local = ~local_failed
        global_mask = ~selected_local if accelerated else stere_pyram__torch.ones_like(prior_valid)
        global_trace = None
        if not accelerated or bool(global_mask.any()):
            options = dict(search_mask=global_mask, prune_search=True) if accelerated else {}
            (gd, gc, gg, global_trace) = matcher(ai, bi, sign, **options, **kwargs)
            gg = gg & global_mask & stere_pyram__torch.isfinite(gd) & stere_pyram__torch.isfinite(gc)
        if accelerated and local is not None:
            if global_trace is None:
                (d, c, good) = (ld, lc, selected_local)
            else:
                (d, c) = (stere_pyram__torch.where(selected_local, ld, gd), stere_pyram__torch.where(selected_local, lc, gc))
                good = selected_local | gg
            final_trace = _choose_trace(local_trace, global_trace, selected_local)
        elif local is not None:
            local_disagreement = lg & gg & ((ld - gd).abs() > 1e-05)
            selected_local = prior_valid & ~local_failed & ~local_disagreement & gg
            d = stere_pyram__torch.where(selected_local, ld, gd)
            (c, good, final_trace) = (gc, gg, dict(global_trace))
        else:
            (d, c, good, final_trace) = (gd, gc, gg, dict(global_trace))
        good = good & stere_pyram__torch.isfinite(d)
        (d, c) = (stere_pyram__torch.where(good, d, float('nan')), stere_pyram__torch.where(good, c, 0.0))
        final_trace['refinement_valid'] = final_trace.get('refinement_valid', stere_pyram__torch.zeros_like(good)) & good
        counters = _search_totals(ai, local_trace, global_trace)
        for (key, value) in counters.items():
            totals[key] += value
        fallback_mask = ~selected_local
        level.update(status='processed', prior_valid=prior_valid, local_search_mask=prior_valid if local is not None else stere_pyram__torch.zeros_like(good), local_search_executed=local is not None, local_failed=local_failed, local_disagreement=local_disagreement, local_selected_mask=selected_local, global_fallback_mask=fallback_mask, fallback_rate=fallback_mask.flatten(1).float().mean(1) if previous is not None else stere_pyram__torch.zeros(len(ai), device=ai.device), global_audit_executed=previous is not None and (not accelerated), global_audit_mask=stere_pyram__torch.full_like(good, previous is not None and (not accelerated)), global_search_executed=global_mask.flatten(1).any(1) if accelerated else True, search_statistics=counters, refinement_valid=final_trace['refinement_valid'], search_boundary=final_trace.get('search_boundary', stere_pyram__torch.zeros_like(good)))
        credible_prior = good
        if accelerated:
            credible_prior = good & (c >= trust_margin) & ~level['search_boundary']
        previous = (d, c, credible_prior)
        previous_width = w
        if name == 'full':
            final = (d, c, good, final_trace)
    if final is None:
        final = _invalid_result(a)
    (d, c, good, trace) = final
    trace.pop('search_statistics', None)
    trace['pyramid'] = dict(version='dh116-stereo-pyramid-modes-v2', levels=levels, mode=pyramid_mode, global_certification=not accelerated, search_statistics=totals, search_statistics_scope='both_scans;eligible_centers_and_executed_full_plane_pixels', accelerated_trust_margin=trust_margin if accelerated else None, processed_level_count=sum((level['status'] == 'processed' for level in levels.values())), output_resolution_wh=[width, height], final_resize=False, scale_policy='floor_dimensions_exact_ratios', raw_support_policy='all_area_contributors_then_full_matching_patch', prior_support_policy='all_positive_bilinear_contributors_finite_and_valid', global_audit_policy='untrusted_centers_only;no_global_check_on_accepted_local' if accelerated else 'every_refined_level_full_range;global_rejection_overrides_local', effective_no_search_speedup=not accelerated, cost_volume_allocated=False, final_refinement_valid=trace['refinement_valid'])
    trace['metric'] = 'zncc'
    trace['resource_metadata'] = dict(trace.get('resource_metadata', {}), cost_volume_allocated=False, output_resolution_wh=[width, height], pyramid_global_audit=not accelerated, effective_no_search_speedup=not accelerated, retained_state='fixed_planes_per_level;no_disparity_axis')
    return (d, c, good, trace)

def validate_processing_contract(value, expected=None):
    if not isinstance(value, dict) or value.get('version') != VERSION:
        raise ValueError('versioned stereo processing contract required')
    if value.get('native_qualified') is not False or value.get('source_sha256') != _sources():
        raise ValueError('processing implementation source SHA mismatch or unsupported qualification claim')
    if value.get('processing_sha256') != digest({key: item for (key, item) in value.items() if key != 'processing_sha256'}):
        raise ValueError('processing contract SHA mismatch')
    if expected is not None and value != expected:
        raise ValueError('actual camera processing differs from bound preprocessor contract')
    return value

def require_raw_processing(actor_contract, camera_config):
    if not isinstance(actor_contract, dict):
        raise ValueError('actor processing binding required')
    binding = actor_contract.get('preprocessor', actor_contract)
    if not isinstance(binding, dict) or binding.get('binding') != 'bound_raw_stereo':
        raise ValueError('history-input-only/unspecified package has no raw RGB processing binding')
    actual = processing_contract(camera_config)
    validate_processing_contract(binding.get('processing'), expected=actual)
    if actor_contract.get('calibration_sha256', actual['calibration_sha256']) != actual['calibration_sha256']:
        raise ValueError('actor calibration differs from actual processing contract')
    return actual
