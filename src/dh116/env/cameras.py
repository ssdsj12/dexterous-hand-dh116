"""Owned synchronized RGB acquisition, CameraView/tiled sensor resources and audited raw readback."""
import json as depth_camer_probe__json
import numpy as depth_hand_point__np
from contextlib import contextmanager as previ_warp_readb__contextmanager
import copy as stere_camer__copy
import copy as stere_rgb_bridg__copy
import importlib as stere_rgb_owned_nativ__importlib
import uuid as stere_rgb_resou__uuid
import copy as tiled_stere_captu__copy
import importlib as tiled_stere_nativ__importlib
from contextlib import contextmanager as tiled_stere_resou__contextmanager
import re as visio_camer__re
import importlib as depth_camer_probe__importlib

def transform_world_points_to_hands(points_world, hand_poses):
    """Transform [P,3] world points to [2,P,3] using this frame's wrist poses."""
    points = depth_hand_point__np.asarray(points_world, dtype=depth_hand_point__np.float64)
    poses = depth_hand_point__np.asarray(hand_poses, dtype=depth_hand_point__np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or (not depth_hand_point__np.isfinite(points).all()):
        raise ValueError('points_world must be finite [P,3] positions in metres')
    if poses.shape != (2, 7) or not depth_hand_point__np.isfinite(poses).all():
        raise ValueError('hand_poses must be finite [2,7] xyz+wxyz link poses')
    quaternions = poses[:, 3:].copy()
    scale = depth_hand_point__np.abs(quaternions).max(axis=1, keepdims=True)
    if depth_hand_point__np.any(scale == 0.0):
        raise ValueError('hand pose quaternions must be nonzero')
    quaternions /= scale
    quaternions /= depth_hand_point__np.linalg.norm(quaternions, axis=1, keepdims=True)
    (w, x, y, z) = quaternions.T
    rotations = depth_hand_point__np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), axis=1).reshape(2, 3, 3)
    offsets = points[None] - poses[:, None, :3]
    return offsets @ rotations
import hashlib as previ_warp_readb__hashlib
import importlib as stere_camer__importlib
import math as stere_rgb_bridg__math
import numpy as stere_rgb_owned_nativ__np
stere_rgb_resou__RIG_PATH = '/World/DH116StereoRGB'
import math as tiled_stere_captu__math
import inspect as tiled_stere_nativ__inspect
import hashlib as tiled_stere_resou__hashlib
import numpy as visio_camer__np
from pathlib import Path as depth_camer_probe__Path

def depth_to_hand_points(depth, K, T_world_camera, hand_poses, *, object_mask=None, min_depth_m=0.01, max_depth_m=10.0):
    """Backproject retained pixels and transform them to both current hand frames.

    depth[H,W] is optical-axis Z in metres, rather than range along a ray.
    K[3,3] is a calibrated pinhole intrinsic matrix, including optional skew.
    object_mask[H,W], when supplied, must be boolean and selects visible pixels.

    Returns pixels_uv[P,2] in row-major pixel order, points_camera[P,3],
    points_world[P,3], points_in_hands[2,P,3], and valid_mask[H,W].
    An all-invalid frame produces empty point arrays and a false valid_mask.
    """
    image = depth_hand_point__np.asarray(depth, dtype=depth_hand_point__np.float64)
    intrinsic = depth_hand_point__np.asarray(K, dtype=depth_hand_point__np.float64)
    camera = depth_hand_point__np.asarray(T_world_camera, dtype=depth_hand_point__np.float64)
    if image.ndim != 2:
        raise ValueError('depth must be an [H,W] optical-Z image in metres')
    if not depth_hand_point__np.isfinite(min_depth_m) or not depth_hand_point__np.isfinite(max_depth_m) or (not 0.0 <= min_depth_m < max_depth_m):
        raise ValueError('depth limits must satisfy 0 <= min_depth_m < max_depth_m')
    if intrinsic.shape != (3, 3) or not depth_hand_point__np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0.0 or (intrinsic[1, 1] <= 0.0) or (not depth_hand_point__np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0.0)) or (not depth_hand_point__np.isclose(intrinsic[1, 0], 0.0, atol=1e-12, rtol=0.0)):
        raise ValueError('K must be a finite pinhole matrix with positive focal lengths')
    if camera.shape != (4, 4) or not depth_hand_point__np.isfinite(camera).all() or (not depth_hand_point__np.allclose(camera[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0)):
        raise ValueError('T_world_camera must be a finite rigid [4,4] transform')
    rotation = camera[:3, :3]
    if not depth_hand_point__np.allclose(rotation.T @ rotation, depth_hand_point__np.eye(3), atol=1e-06, rtol=0.0) or not depth_hand_point__np.isclose(depth_hand_point__np.linalg.det(rotation), 1.0, atol=1e-06, rtol=0.0):
        raise ValueError('T_world_camera must contain a proper orthonormal rotation')
    valid = depth_hand_point__np.isfinite(image) & (image > 0.0) & (image >= min_depth_m) & (image <= max_depth_m)
    if object_mask is not None:
        mask = depth_hand_point__np.asarray(object_mask)
        if mask.shape != image.shape or mask.dtype != depth_hand_point__np.bool_:
            raise ValueError('object_mask must be boolean with the same [H,W] shape as depth')
        valid &= mask
    (rows, columns) = depth_hand_point__np.nonzero(valid)
    pixels = depth_hand_point__np.column_stack((columns, rows))
    homogeneous_pixels = depth_hand_point__np.column_stack((pixels, depth_hand_point__np.ones(len(pixels))))
    rays = homogeneous_pixels @ depth_hand_point__np.linalg.inv(intrinsic).T
    points_camera = rays * image[valid, None]
    points_world = points_camera @ rotation.T + camera[:3, 3]
    points_in_hands = transform_world_points_to_hands(points_world, hand_poses)
    return dict(pixels_uv=pixels, points_camera=points_camera, points_world=points_world, points_in_hands=points_in_hands, valid_mask=valid)
import importlib as previ_warp_readb__importlib
import math as stere_camer__math
import importlib as stere_rgb_bridg__importlib
import torch as stere_rgb_owned_nativ__torch
_MARKER = 'dh116RgbCaptureOwner'
import re as tiled_stere_captu__re
from pathlib import Path as tiled_stere_nativ__Path
import json as tiled_stere_resou__json
import torch as visio_camer__torch
import numpy as depth_camer_probe__np
depth_camer_probe__depth_to_hand_points = depth_to_hand_points
import inspect as previ_warp_readb__inspect
import json as stere_camer__json
import re as stere_rgb_bridg__re

class RGBResources:

    def __init__(self):
        self.stage = None
        self.token = None
        self.view = None
        self.attachments = []
        self.unsafe = False
        self.closed = False
        self.error = None

    def define_rig(self, stage, define):
        if stage.GetPrimAtPath(stere_rgb_resou__RIG_PATH).IsValid():
            raise RuntimeError('stereo RGB rig already exists')
        define(stage, stere_rgb_resou__RIG_PATH)
        prim = stage.GetPrimAtPath(stere_rgb_resou__RIG_PATH)
        token = stere_rgb_resou__uuid.uuid4().hex
        prim.SetCustomDataByKey(_MARKER, token)
        (self.stage, self.token) = (stage, token)

    def bind_view(self, view):
        self.unsafe = True
        if getattr(view, '_tiled_render_product', object()) is not None or not isinstance(getattr(view, '_annotators', None), dict) or getattr(view, '_render_product', None) is None or (view.get_render_product_path() != view._render_product.path):
            raise RuntimeError('unsupported CameraView resource layout')
        self.view = view
        self.unsafe = False

    def attach(self, annotator, path, do_attach):
        self.attachments.append((annotator, path))
        do_attach()

    def close(self):
        if self.closed:
            if self.error:
                raise RuntimeError(self.error)
            return
        self.closed = True
        errors = []
        if self.unsafe:
            errors.append('unknown camera resources')
        for (annotator, path) in reversed(self.attachments):
            try:
                annotator.detach([path])
            except Exception as exc:
                errors.append('annotator detach: ' + repr(exc))
        if self.view is not None:
            view = self.view
            for annotator in view._annotators.values():
                try:
                    annotator.detach([view._render_product_path])
                except Exception as exc:
                    errors.append('camera annotator detach: ' + repr(exc))
            if not errors:
                try:
                    view._render_product.destroy()
                    view._render_product = None
                    view._render_product_path = None
                    view._annotators.clear()
                except Exception as exc:
                    errors.append('render product destroy: ' + repr(exc))
        if self.stage is not None and (not errors):
            try:
                prim = self.stage.GetPrimAtPath(stere_rgb_resou__RIG_PATH)
                if not prim.IsValid() or prim.GetCustomDataByKey(_MARKER) != self.token:
                    raise RuntimeError('rig ownership marker changed')
                if not self.stage.RemovePrim(stere_rgb_resou__RIG_PATH):
                    raise RuntimeError('rig removal failed')
            except Exception as exc:
                errors.append('rig removal: ' + repr(exc))
        if errors:
            self.error = '; '.join(errors)
            raise RuntimeError(self.error)
import torch as tiled_stere_captu__torch
import torch as tiled_stere_nativ__torch
from pathlib import Path as tiled_stere_resou__Path

def numeric_slot_order(prim_paths, count):
    """Return camera slots indexed by numeric environment ID, never lexicography."""
    ids = []
    for path in prim_paths:
        match = visio_camer__re.fullmatch('.*/camera_(\\d+)', str(path))
        if match is None:
            raise ValueError(f'unexpected vision camera prim path: {path}')
        ids.append(int(match.group(1)))
    if len(ids) != count or sorted(ids) != list(range(count)):
        raise ValueError('camera prim paths must contain each numeric environment ID exactly once')
    return visio_camer__torch.tensor([ids.index(i) for i in range(count)], dtype=visio_camer__torch.long)

def _load_camera_class():
    kit = depth_camer_probe__importlib.import_module('omni.kit.app')
    kit.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.sensors.camera', True)
    return depth_camer_probe__importlib.import_module('isaacsim.sensors.camera').Camera
from pathlib import Path as previ_warp_readb__Path
from pathlib import Path as stere_camer__Path
import torch as stere_rgb_bridg__torch
stere_rgb_owned_nativ__RGBResources = RGBResources
from dh116.physics.vision import StereoCalibration as tiled_stere_captu__StereoCalibration
import numpy as tiled_stere_nativ__np
import threading as tiled_stere_resou__threading

def detile(tiled, *, count, width, height, columns, channels):
    """Untile actual linear pixels; handles upstream rectangular reshape mistakes."""
    rows = (count + columns - 1) // columns
    data = visio_camer__torch.as_tensor(tiled)
    if data.numel() != rows * height * columns * width * channels:
        raise ValueError('tiled camera buffer size does not match calibrated tile layout')
    return data.reshape(rows, height, columns, width, channels).permute(0, 2, 1, 3, 4).reshape(rows * columns, height, width, channels)[:count]

def depth_camer_probe___numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return depth_camer_probe__np.asarray(value)
import threading as previ_warp_readb__threading
import numpy as stere_camer__np
from dh116.physics.vision import StereoCalibration as stere_rgb_bridg__StereoCalibration
from dh116.physics.vision import stere_calib__rerender_calibration as tiled_stere_captu__rerender_calibration

def ensure_clock_annotators(manager, registry):
    """Activate the installed time-node owner before requesting its annotator.

    Importing bindings does not run the extension's on_startup registration.
    An existing enabled extension is never restarted or disabled by this owner.
    Missing registration remains an error, never a synthetic timestamp.
    """
    extension = 'isaacsim.core.nodes'
    if not manager.is_extension_enabled(extension):
        manager.set_extension_enabled_immediate(extension, True)
    if not manager.is_extension_enabled(extension):
        raise RuntimeError('installed clock extension failed to enable: ' + extension)
    registered = set(registry.get_registered_annotators())
    missing = {'IsaacReadTimes'} - registered
    if missing:
        raise RuntimeError('installed clock annotators unavailable: ' + ', '.join(sorted(missing)))
import uuid as tiled_stere_resou__uuid

def brown_mask(rgb, *, float_rgb_is_255=None):
    """GPU equivalent of probe.brown_rod_mask, verified against that RGB-only rule."""
    color = rgb[..., :3].to(visio_camer__torch.float64)
    if not rgb.is_floating_point():
        color = color / 255.0
    elif bool(visio_camer__torch.any(color > 1.0)) if float_rgb_is_255 is None else float_rgb_is_255:
        color = color / 255.0
    (r, g, b) = color.unbind(-1)
    (maximum, minimum) = (color.amax(-1), color.amin(-1))
    saturation = (maximum - minimum) / maximum.clamp_min(1e-12)
    return visio_camer__torch.isfinite(color).all(-1) & (r > 1.2 * g) & (g > 1.2 * b) & (maximum > 35.0 / 255.0) & (saturation > 0.25)

def _rotation(quaternion):
    q = depth_camer_probe__np.asarray(quaternion, dtype=depth_camer_probe__np.float64)
    if q.shape != (4,) or not depth_camer_probe__np.isfinite(q).all() or depth_camer_probe__np.linalg.norm(q) == 0.0:
        raise ValueError('quaternion must be finite nonzero wxyz')
    (w, x, y, z) = q / depth_camer_probe__np.linalg.norm(q)
    return depth_camer_probe__np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
import types
import torch as stere_camer__torch
from dh116.physics.vision import stere_calib__rerender_calibration as stere_rgb_bridg__rerender_calibration
from dh116.physics.vision import validate_identity as tiled_stere_captu__validate_identity

def author_standard_camera_ops(camera, UsdGeom, Gf):
    """Prepare only a new private prim for the installed XformPrimView."""
    if camera.GetOrderedXformOps():
        raise ValueError('new bare camera required; existing transforms are not overwritten')
    precision = UsdGeom.XformOp.PrecisionDouble
    camera.AddTranslateOp(precision).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    camera.AddOrientOp(precision).Set(Gf.Quatd(1.0))
    camera.AddScaleOp(precision).Set(Gf.Vec3d(1.0, 1.0, 1.0))
tiled_stere_resou__RIG_PATH = '/World/DH116TiledStereo'

def visio_camer___numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, visio_camer__torch.Tensor) else visio_camer__np.asarray(value)

def look_at_optical(eye, target, up):
    """Return a column-vector optical-camera-to-world rigid transform."""
    (eye, target, up) = [depth_camer_probe__np.asarray(v, dtype=float) for v in (eye, target, up)]
    forward = target - eye
    if any((v.shape != (3,) or not depth_camer_probe__np.isfinite(v).all() for v in (eye, target, up))):
        raise ValueError('camera eye, target and up must be finite xyz')
    if depth_camer_probe__np.linalg.norm(forward) < 1e-10:
        raise ValueError('camera eye and target must differ')
    forward /= depth_camer_probe__np.linalg.norm(forward)
    right = depth_camer_probe__np.cross(forward, up)
    if depth_camer_probe__np.linalg.norm(right) < 1e-10:
        raise ValueError('camera up must not be parallel to the optical axis')
    right /= depth_camer_probe__np.linalg.norm(right)
    down = depth_camer_probe__np.cross(forward, right)
    result = depth_camer_probe__np.eye(4)
    result[:3, :3] = depth_camer_probe__np.column_stack((right, down, forward))
    result[:3, 3] = eye
    return result
ANNOTATOR_SHA256 = '1898abce52559be7befbc7b41bdd6752139df113ac69e62b5b2a97a9c5743a95'
from dh116.physics.vision import StereoCalibration as stere_camer__StereoCalibration
from dh116.physics.vision import validate_identity as stere_rgb_bridg__validate_identity
from dh116.physics.vision import validate_images as tiled_stere_captu__validate_images

def prepare_renderer_view(view):
    """Use the installed USD pose route for this owned Isaac4.5 camera only.

    The installed Fabric route calls usdrt.hierarchy during the first buffer
    read; that attribute was absent in the actual preview process. No global
    Fabric/physics setting is changed. RTX image rendering stays on CUDA.
    """
    if type(getattr(view, '_use_fabric', None)) is not bool or type(getattr(view, '_sync_usd_on_fabric_write', None)) is not bool:
        raise ValueError('unsupported installed renderer view layout')
    view._use_fabric = False
    view._sync_usd_on_fabric_write = True
MARKER = 'dh116TiledStereoOwner'

def paired_render_timestamp(reference, times, *, legacy_time):
    """Use the simulation timestamp carried by this exact rendered result."""
    if not isinstance(reference, dict) or not isinstance(times, dict) or (not {'referenceTimeNumerator', 'referenceTimeDenominator'} <= reference.keys()) or (not {'swhFrameNumber', 'frameNumber', 'simulationTime', 'rationalTimeOfSimNumerator', 'rationalTimeOfSimDenominator'} <= times.keys()):
        raise ValueError(f'render time metadata unavailable: reference={reference}, times={times}')
    frame_id = int(reference['referenceTimeNumerator'])
    denominator = int(reference['referenceTimeDenominator'])
    swh_frame = int(times['swhFrameNumber'])
    rational_pair = (int(times['rationalTimeOfSimNumerator']), int(times['rationalTimeOfSimDenominator']))
    if frame_id < 0 or denominator <= 0 or rational_pair != (frame_id, denominator):
        raise ValueError(f'render frame mismatch: ReferenceTime={reference}, IsaacReadTimes={times}')
    timestamp = float(times['simulationTime'])
    if not visio_camer__np.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError(f'invalid same-frame simulation time: {timestamp}')
    diagnostics = {key: value.item() if hasattr(value, 'item') else value for (key, value) in times.items()}
    return dict(frame_id=frame_id, simulation_time=timestamp, reference_time_numerator=frame_id, reference_time_denominator=denominator, render_frame_number=int(times['frameNumber']), swh_frame_number=swh_frame, legacy_simulation_time=float(legacy_time), isaac_read_times=diagnostics, time_mapping='same_render_product_IsaacReadTimes')

def usd_pose_to_optical(position, quaternion):
    """USD camera +Y up/-Z forward becomes optical +Y down/+Z forward."""
    result = depth_camer_probe__np.eye(4)
    result[:3, :3] = _rotation(depth_camer_probe___numpy(quaternion)) @ depth_camer_probe__np.diag([1.0, -1.0, -1.0])
    result[:3, 3] = depth_camer_probe___numpy(position)
    return result
BUNDLED_WARP_SOURCES = {'warp/__init__.py': '0a969e4fe2e5b10cd8770c2608834d42f49e20d3cd91c32206c69314b0bb8f00', 'warp/types.py': '6114a3fc38c971f1e4a341a260aa088f78543981210edc6bdc9428f3465ade1f'}
from dh116.physics.vision import usd_aperture_offsets as stere_camer__usd_aperture_offsets
from dh116.physics.vision import validate_images as stere_rgb_bridg__validate_images

def validate_usd_camera_pose(stage, path, transform, Usd, UsdGeom):
    """Cross-check independently authored USD, not only the Fabric cache."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError('owned USD camera missing')
    usd = tiled_stere_nativ__np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)).T
    optical = usd @ tiled_stere_nativ__np.diag([1.0, -1.0, -1.0, 1.0])
    actual = tiled_stere_nativ__torch.as_tensor(transform, dtype=tiled_stere_nativ__torch.float64).detach().cpu()
    expected = tiled_stere_nativ__torch.as_tensor(optical, dtype=tiled_stere_nativ__torch.float64)[None]
    if actual.shape != expected.shape or not tiled_stere_nativ__torch.allclose(actual, expected, atol=1e-05, rtol=0):
        raise ValueError('USD/camera pose mismatch')
tiled_stere_resou___LOCK = tiled_stere_resou__threading.RLock()

class _CameraAxesDeviceMixin:
    """Repair 4.5 CameraView's CPU constant during its parent constructor call."""

    def _convert_camera_axes(self, orientations, transform_matrix):
        if not isinstance(orientations, visio_camer__torch.Tensor):
            return super()._convert_camera_axes(orientations, transform_matrix)
        matrices = self._backend_utils.quats_to_rot_matrices(orientations).to(device=orientations.device, dtype=orientations.dtype)
        transform = visio_camer__torch.as_tensor(transform_matrix[:3, :3], device=orientations.device, dtype=orientations.dtype)
        converted = transform @ matrices
        result = self._backend_utils.rot_matrices_to_quats(converted, device=orientations.device)
        return result.to(device=orientations.device, dtype=orientations.dtype)

def brown_rod_mask(rgb):
    """Visible-color-only baseline; contains no pose, cylinder or segmentation GT."""
    image = depth_camer_probe__np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError('rgb must be [H,W,3+]')
    color = image[..., :3].astype(depth_camer_probe__np.float64)
    if depth_camer_probe__np.issubdtype(image.dtype, depth_camer_probe__np.integer) or (color.size and depth_camer_probe__np.nanmax(color) > 1.0):
        color /= 255.0
    (r, g, b) = depth_camer_probe__np.moveaxis(color, -1, 0)
    (maximum, minimum) = (color.max(-1), color.min(-1))
    saturation = (maximum - minimum) / depth_camer_probe__np.maximum(maximum, 1e-12)
    return depth_camer_probe__np.isfinite(color).all(-1) & (r > 1.2 * g) & (g > 1.2 * b) & (maximum > 35.0 / 255.0) & (saturation > 0.25)
POLICY = 'dh116-isaac45-warp112-rgb-nonowning-v1'
from dh116.physics.vision import usd_aperture_intrinsics
from dh116.physics.vision import validate_pair_timing as stere_rgb_bridg__validate_pair_timing

def verify_tiled_sources(root):
    data = tiled_stere_resou__json.loads((tiled_stere_resou__Path(root) / 'configs/runtime/tiled_api.json').read_text())
    if data.get('version') != 'dh116-installed-tiled-camera-api-v1':
        raise ValueError('unsupported tiled API contract')
    sources = data.get('sources')
    if not isinstance(sources, dict) or set(sources) != {'tiled', 'camera', 'sensor', 'replicator_create', 'xform_view', 'fabric_utils'}:
        raise ValueError('complete tiled API source contract required')
    for (name, item) in sources.items():
        p = tiled_stere_resou__Path(item['path'])
        if not p.is_absolute() or tiled_stere_resou__hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('installed tiled API source differs: ' + name)
    return sources

def cylinder_surface_distances(points_world, rod_pose, radius, length):
    """Absolute distance to finite local-Y cylinder; validation only, never mask."""
    pose = depth_camer_probe__np.asarray(rod_pose, dtype=float)
    local = (depth_camer_probe__np.asarray(points_world, dtype=float) - pose[:3]) @ _rotation(pose[3:])
    radial = depth_camer_probe__np.linalg.norm(local[:, [0, 2]], axis=1) - float(radius)
    axial = depth_camer_probe__np.abs(local[:, 1]) - float(length) / 2.0
    signed = depth_camer_probe__np.hypot(depth_camer_probe__np.maximum(radial, 0.0), depth_camer_probe__np.maximum(axial, 0.0))
    signed += depth_camer_probe__np.minimum(depth_camer_probe__np.maximum(radial, axial), 0.0)
    return depth_camer_probe__np.abs(signed)
previ_warp_readb___LOCK = previ_warp_readb__threading.RLock()
from dh116.physics.vision import validate_images as stere_camer__validate_images
stere_rgb_bridg___IDENTITY_KEYS = frozenset(('run_uuid', 'env_id', 'episode_uid', 'reset_id', 'policy_tick', 'physics_tick_start', 'physics_tick_end', 'rod_step', 'state_sha256', 'simulation_timestamp', 'calibration_sha256'))

class TiledResources:

    def __init__(self, *, rig_path=tiled_stere_resou__RIG_PATH):
        self.rig_path = rig_path
        self.stage = self.sensor = self.product = None
        self.token = None
        self.extra = []
        self.closed = False
        self.detached = False
        self.error = None

    def define_rig(self, stage, define):
        if stage.GetPrimAtPath(self.rig_path).IsValid():
            raise RuntimeError('tiled stereo rig already exists')
        define(stage, self.rig_path)
        token = tiled_stere_resou__uuid.uuid4().hex
        stage.GetPrimAtPath(self.rig_path).SetCustomDataByKey(MARKER, token)
        (self.stage, self.token) = (stage, token)

    @tiled_stere_resou__contextmanager
    def product_scope(self, api):
        """Track the real HydraTexture which upstream keeps only as a local."""
        with tiled_stere_resou___LOCK:
            original = api.render_product_tiled

            def tracked(*args, **kwargs):
                if self.product is not None:
                    raise RuntimeError('multiple tiled products are not permitted')
                kwargs = dict(kwargs, force_new=True)
                result = original(*args, **kwargs)
                self.product = result
                if not isinstance(getattr(result, 'path', None), str) or not callable(getattr(result, 'destroy', None)):
                    raise RuntimeError('unsupported tiled HydraTexture handle')
                return result
            api.render_product_tiled = tracked
            try:
                yield
            finally:
                if api.render_product_tiled is not tracked:
                    raise RuntimeError('tiled product factory changed concurrently')
                api.render_product_tiled = original

    def attach(self, annotator, path):
        self.extra.append((annotator, path))
        annotator.attach([path])

    def detach(self):
        """Detach all owned annotators before any paired product is destroyed."""
        if self.detached:
            if self.error:
                raise RuntimeError(self.error)
            return
        self.detached = True
        errors = []
        sensor = self.sensor
        if sensor is not None:
            try:
                sensor._clear_callbacks()
            except Exception as e:
                errors.append('callbacks: ' + repr(e))
        for (annotator, path) in reversed(self.extra):
            try:
                annotator.detach([path])
            except Exception as e:
                errors.append('extra annotator: ' + repr(e))
        self.extra.clear()
        if sensor is not None:
            paths = getattr(sensor, '_render_product_paths', None)
            annotators = getattr(sensor, '_annotators', None)
            if not isinstance(paths, list) or not isinstance(annotators, dict):
                errors.append('unknown tiled sensor resource layout')
            else:
                owned_paths = [] if self.product is None else [self.product.path]
                valid_paths = paths == owned_paths or (not paths and (not annotators))
                if not valid_paths:
                    errors.append('tiled product ownership path mismatch')
                else:
                    for (name, annotator) in list(annotators.items()):
                        try:
                            annotator.detach(paths)
                        except Exception as e:
                            errors.append(name + ' detach: ' + repr(e))
                    annotators.clear()
        if errors:
            self.error = '; '.join(errors)
            raise RuntimeError(self.error)

    def close(self):
        if self.closed:
            if self.error:
                raise RuntimeError(self.error)
            return
        self.closed = True
        errors = []
        self.detach()
        sensor = self.sensor
        if self.product is not None and (not errors):
            try:
                self.product.destroy()
            except Exception as e:
                errors.append('product destroy: ' + repr(e))
        if self.stage is not None and (not errors):
            try:
                prim = self.stage.GetPrimAtPath(self.rig_path)
                if not prim.IsValid() or prim.GetCustomDataByKey(MARKER) != self.token:
                    raise RuntimeError('tiled rig ownership marker changed')
                if not self.stage.RemovePrim(self.rig_path):
                    raise RuntimeError('tiled rig removal failed')
            except Exception as e:
                errors.append('rig removal: ' + repr(e))
        if errors:
            self.error = '; '.join(errors)
            raise RuntimeError(self.error)
        if sensor is not None:
            sensor._view = None
            sensor._sensor_prims = []
            sensor._render_product_paths = []
        self.sensor = self.product = self.stage = None

def save_frame_artifacts(output, name, rgb, depth, intrinsic, transform, hand_poses, rod_pose, rod_radius, rod_length, *, min_depth_m, max_depth_m):
    """Save reproducible optical/hand point arrays and a visible RGB/depth preview."""
    output = depth_camer_probe__Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (rgb, depth) = (depth_camer_probe__np.asarray(rgb)[..., :3], depth_camer_probe__np.asarray(depth))
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if rgb.shape[:2] != depth.shape:
        raise RuntimeError(f'RGB/depth shape mismatch: {rgb.shape}, {depth.shape}')
    mask = brown_rod_mask(rgb)
    kwargs = dict(min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    all_points = depth_camer_probe__depth_to_hand_points(depth, intrinsic, transform, hand_poses, **kwargs)
    selected = depth_camer_probe__depth_to_hand_points(depth, intrinsic, transform, hand_poses, object_mask=mask, **kwargs)
    if not len(all_points['points_world']):
        raise RuntimeError(f'{name}: no finite depth within clipping range')
    if not len(selected['points_world']):
        raise RuntimeError(f'{name}: visible RGB brown rod mask has no valid depth points')
    roundtrip = []
    for hand in range(2):
        reconstructed = selected['points_in_hands'][hand] @ _rotation(hand_poses[hand, 3:]).T + hand_poses[hand, :3]
        roundtrip.append(depth_camer_probe__np.linalg.norm(reconstructed - selected['points_world'], axis=1).max())
    camera_roundtrip = (selected['points_world'] - transform[:3, 3]) @ transform[:3, :3]
    camera_error = float(depth_camer_probe__np.linalg.norm(camera_roundtrip - selected['points_camera'], axis=1).max())
    if max(roundtrip) > 1e-06 or camera_error > 1e-06:
        raise RuntimeError('camera/hand coordinate roundtrip exceeded one micrometre')
    distances = cylinder_surface_distances(selected['points_world'], rod_pose, rod_radius, rod_length)
    archive = output / f'{name}_depth_coordinates.npz'
    depth_camer_probe__np.savez_compressed(archive, rgb=rgb, depth=depth, K=intrinsic, T_world_camera=transform, hand_poses=hand_poses, object_mask_rgb=mask, **selected, all_points_world=all_points['points_world'], all_points_camera=all_points['points_camera'], all_pixels_uv=all_points['pixels_uv'], all_valid_mask=all_points['valid_mask'])
    sample_id = len(selected['points_world']) // 2
    sample = dict(pixel_uv=selected['pixels_uv'][sample_id].tolist(), camera_xyz_m=selected['points_camera'][sample_id].tolist(), world_xyz_m=selected['points_world'][sample_id].tolist(), left_hand_xyz_m=selected['points_in_hands'][0, sample_id].tolist(), right_hand_xyz_m=selected['points_in_hands'][1, sample_id].tolist())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (figure, axes) = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title(f'{name}: rendered RGB')
    display_depth = depth_camer_probe__np.ma.masked_where(~all_points['valid_mask'], depth)
    view = axes[1].imshow(display_depth, cmap='viridis', vmin=min_depth_m, vmax=max_depth_m)
    figure.colorbar(view, ax=axes[1], label='Optical Z (m)', fraction=0.046)
    axes[1].set_title('Rendered depth, optical Z')
    axes[2].imshow(rgb)
    overlay = depth_camer_probe__np.zeros((*mask.shape, 4))
    overlay[selected['valid_mask']] = [0.0, 1.0, 0.2, 0.7]
    axes[2].imshow(overlay)
    axes[2].set_title('Visible RGB brown mask + valid depth')
    for axis in axes:
        axis.scatter(*sample['pixel_uv'], c='magenta', s=45, marker='x', linewidths=2)
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle('Measured visible points only; no segmentation GT; no cylinder roll estimate')
    figure.tight_layout()
    preview = output / f'{name}_rgb_depth.png'
    figure.savefig(preview, dpi=140)
    plt.close(figure)
    return dict(name=name, archive=str(archive), preview=str(preview), valid_pixels=int(all_points['valid_mask'].sum()), object_mask_rgb_pixels=int(mask.sum()), object_mask_points=int(len(selected['points_world'])), camera_world_roundtrip_max_m=camera_error, hand_world_roundtrip_max_m=float(max(roundtrip)), sample_point=sample, hand_poses_xyz_wxyz=depth_camer_probe__np.asarray(hand_poses).tolist(), T_world_camera=depth_camer_probe__np.asarray(transform).tolist(), K=depth_camer_probe__np.asarray(intrinsic).tolist(), depth_range_m=[float(depth[all_points['valid_mask']].min()), float(depth[all_points['valid_mask']].max())], gt_cylinder_surface_validation_only=dict(median_m=float(depth_camer_probe__np.median(distances)), p95_m=float(depth_camer_probe__np.quantile(distances, 0.95)), maximum_m=float(distances.max()), fraction_within_2mm=float(depth_camer_probe__np.mean(distances < 0.002))))

def nonowning_constructor(constructor):

    def create(*args, **kwargs):
        if 'owner' in kwargs:
            if kwargs.pop('owner') is not False:
                raise ValueError('only verified owner=False external buffers are supported')
            if kwargs.get('ptr') is None or kwargs.get('deleter') is not None:
                raise ValueError('non-owning external pointer without deleter required')
            kwargs['deleter'] = None
        return constructor(*args, **kwargs)
    return create
from dh116.physics.vision import validate_identity as stere_camer__validate_identity
_VALIDATION_KEYS = frozenset(('run_uuid', 'env_id', 'episode_uid', 'reset_id', 'policy_tick', 'physics_tick_start', 'physics_tick_end', 'simulation_timestamp'))
tiled_stere_nativ__TiledResources = TiledResources
tiled_stere_nativ__RIG_PATH = tiled_stere_resou__RIG_PATH
tiled_stere_nativ__verify_tiled_sources = verify_tiled_sources

def capture_depth_coordinates(world, device_world, task, args):
    """Capture initial pinch/tripod scenes, after mixed reset, before PPO creation.

    The caller must already own the sole Isaac process and enable cameras in its
    AppLauncher. Exactly two environments are required to bound render memory.
    """
    if device_world.n != 2:
        raise ValueError('depth coordinate probe requires exactly two environments')
    config_path = depth_camer_probe__Path(args.depth_camera_config)
    config = depth_camer_probe__json.loads(config_path.read_text())
    (width, height) = map(int, config['resolution_wh'])
    if not 1 <= width <= 640 or not 1 <= height <= 480:
        raise ValueError('depth coordinate probe resolution must not exceed 640x480')
    (near, far) = map(float, config['clipping_range_m'])
    fov = float(config['horizontal_fov_degrees'])
    if not (0.0 < near < far and 1.0 < fov < 179.0):
        raise ValueError('invalid camera clipping range or FOV')
    if config.get('depth_convention') != 'optical_axis_z_m':
        raise ValueError('camera config must specify optical_axis_z_m')
    import omni.usd
    Camera = _load_camera_class()
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdLux
    stage = omni.usd.get_context().get_stage()
    parent = '/World/DH116DepthCoordinateProbe'
    if stage.GetPrimAtPath(parent).IsValid():
        raise RuntimeError('depth camera probe already exists; refusing duplicate render products')
    UsdGeom.Xform.Define(stage, parent)
    light = UsdLux.DomeLight.Define(stage, parent + '/DomeLight')
    light.CreateIntensityAttr(1800.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    origins = depth_camer_probe___numpy(device_world.origins).copy()
    (cameras, expected) = ([], [])
    for (env_id, name) in enumerate(('pinch', 'tripod')):
        transform = look_at_optical(depth_camer_probe__np.asarray(config['position_world_m']) + origins[env_id], depth_camer_probe__np.asarray(config['target_world_m']) + origins[env_id], config['world_up'])
        camera = Camera(prim_path=f'{parent}/{name}', name=f'dh116_probe_{name}', resolution=(width, height), frequency=-1)
        camera.initialize()
        camera.set_focal_length(2.4)
        camera.set_horizontal_aperture(2.0 * 2.4 * depth_camer_probe__np.tan(depth_camer_probe__np.deg2rad(fov) / 2.0))
        camera.set_clipping_range(near_distance=near, far_distance=far)
        camera.add_distance_to_image_plane_to_frame()
        usd_transform = transform @ depth_camer_probe__np.diag([1.0, -1.0, -1.0, 1.0])
        xform = UsdGeom.Xformable(camera.prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d(*usd_transform.T.reshape(-1).tolist()))
        cameras.append(camera)
        expected.append(transform)
    before_pose = depth_camer_probe___numpy(device_world.roots).copy()
    before_time = float(world.sim.current_time)
    ready = False
    for frame in range(24):
        world.sim.render()
        if frame < 7:
            continue
        ready = all((camera.get_depth() is not None and depth_camer_probe___numpy(camera.get_depth()).size == width * height and (depth_camer_probe___numpy(camera.get_rgb()).shape[:2] == (height, width)) and depth_camer_probe__np.any(depth_camer_probe__np.isfinite(depth_camer_probe___numpy(camera.get_depth())) & (depth_camer_probe___numpy(camera.get_depth()) > near) & (depth_camer_probe___numpy(camera.get_depth()) < far)) for camera in cameras))
        if ready:
            break
    if not ready:
        raise RuntimeError('RGB/depth annotators did not produce valid frames after 24 render-only warmup frames')
    device_world.refresh(sensors=False)
    hand_poses = depth_camer_probe___numpy(device_world.roots).copy()
    if abs(float(world.sim.current_time) - before_time) > 1e-09 or not depth_camer_probe__np.allclose(hand_poses, before_pose, atol=1e-06, rtol=0.0):
        raise RuntimeError('render-only probe unexpectedly advanced physics or changed actual hand poses')
    rod_poses = depth_camer_probe__np.concatenate((depth_camer_probe___numpy(device_world.wp), depth_camer_probe___numpy(device_world.wq)), axis=1)
    from dh116.env.scene import wire_geometry_from_spawn
    (rod_length, rod_radius) = wire_geometry_from_spawn(world.source['wire'].cfg.spawn)
    output = depth_camer_probe__Path(args.output) / 'depth_camera_coordinates'
    groups = []
    for (env_id, (name, camera, expected_transform)) in enumerate(zip(('pinch', 'tripod'), cameras, expected)):
        (position, orientation) = camera.get_world_pose(camera_axes='usd')
        transform = usd_pose_to_optical(position, orientation)
        usd_matrix = depth_camer_probe__np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(camera.prim)).T
        usd_optical = usd_matrix @ depth_camer_probe__np.diag([1.0, -1.0, -1.0, 1.0])
        authored_error = float(depth_camer_probe__np.max(depth_camer_probe__np.abs(transform - expected_transform)))
        usd_error = float(depth_camer_probe__np.max(depth_camer_probe__np.abs(transform - usd_optical)))
        if max(authored_error, usd_error) > 1e-05:
            raise RuntimeError(f'{name}: camera extrinsic mismatch authored={authored_error}, usd={usd_error}')
        (rgb, depth) = (depth_camer_probe___numpy(camera.get_rgb()).copy(), depth_camer_probe___numpy(camera.get_depth()).copy())
        intrinsic = depth_camer_probe___numpy(camera.get_intrinsics_matrix())
        group = save_frame_artifacts(output, name, rgb, depth, intrinsic, transform, hand_poses[env_id], rod_poses[env_id], rod_radius, rod_length, min_depth_m=near, max_depth_m=far)
        group.update(env_id=env_id, env_origin_m=origins[env_id].tolist(), camera_authored_extrinsic_max_error=authored_error, camera_usd_extrinsic_max_error=usd_error, rendering_frame=int(camera.get_current_frame().get('rendering_frame', 0)), rendering_time_s=float(camera.get_current_frame().get('rendering_time', 0.0)))
        groups.append(group)
    summary = dict(status='depth_coordinates_captured', optimization_performed=False, simulation_advanced=False, camera_count=2, resolution_wh=[width, height], warmup_render_frames=frame + 1, simulation_time_s=before_time, config=str(config_path), camera_convention='x_right_y_down_z_forward; optical_axis_z_m', hand_pose_source='device_world.roots actual palm/base_link; xyz+wxyz', hand_pose_measurement='Simulation link-pose readback stands in for real encoder/FK wrist pose.', student_artifact_excludes='Object GT pose, cylinder geometry/distance GT, and contact/force GT.', mask_source='RGB only: r>1.2g, g>1.2b, max>35/255, saturation>.25', observations_371d_modified=False, cylinder_roll_estimated=False, limitations=['Visible surface only; RGB color mask can contain false positives.', 'Ground-truth cylinder distances are validation only and never select points.', 'Hand-local points do not replace object-centred world-axis 371D reference fields.'], groups=groups)
    summary_path = output / 'summary.json'
    summary['summary_path'] = str(summary_path)
    summary_path.write_text(depth_camer_probe__json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary

class _Proxy:

    def __init__(self, target, **overrides):
        (self._target, self._overrides) = (target, overrides)

    def __getattr__(self, name):
        return self._overrides[name] if name in self._overrides else getattr(self._target, name)
from dh116.physics.vision import validate_pair_timing as stere_camer__validate_pair_timing
stere_rgb_bridg___RAW_KEYS = frozenset(('sdk_step', 'sdk_timestamp_s', 'rod_tick', 'solver_time_s', 'material_reference_uid', 'rod_uid'))

def adapt_pointer_function(raw, warp):
    raw = previ_warp_readb__inspect.unwrap(raw)
    if not isinstance(raw, types.FunctionType):
        raise ValueError('Python annotator pointer function required')
    proxy = _Proxy(warp, types=_Proxy(warp.types, array=nonowning_constructor(warp.array)))
    namespace = dict(raw.__globals__, wp=proxy)
    adapted = types.FunctionType(raw.__code__, namespace, raw.__name__, raw.__defaults__, raw.__closure__)
    adapted.__kwdefaults__ = raw.__kwdefaults__
    return adapted

def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, stere_camer__torch.Tensor) else stere_camer__np.asarray(value)
_PACKET_KEYS = frozenset(('left_rgb', 'right_rgb', 'left_frame_id', 'right_frame_id', 'left_simulation_time', 'right_simulation_time', 'K_left', 'K_right', 'T_world_left', 'T_world_right', 'native_evidence', 'render_timing'))

@previ_warp_readb__contextmanager
def binding_scope(module, replacement):
    with previ_warp_readb___LOCK:
        previous = module._reshape_output_ptr
        module._reshape_output_ptr = replacement
        try:
            yield
        finally:
            if module._reshape_output_ptr is not replacement:
                raise RuntimeError('annotator pointer binding changed concurrently')
            module._reshape_output_ptr = previous

class _NativeRGBPairReader:
    """Lazy Isaac 4.5 adapter; no render-depth annotator or task/world-state access."""

    def __init__(self, sim, origins, calibration, device, *, max_readback_bytes=1024 ** 3, allow_resource_override=False):
        origin_values = stere_camer__torch.as_tensor(origins, dtype=stere_camer__torch.float64).detach().cpu()
        if origin_values.ndim != 2 or origin_values.shape[1] != 3 or (not stere_camer__torch.isfinite(origin_values).all()):
            raise ValueError('finite camera origins [N,3] required')
        if type(max_readback_bytes) is not int or max_readback_bytes < 1:
            raise ValueError('positive camera readback resource limit required')
        estimate = 2 * len(origin_values) * calibration.width * calibration.height * 16
        if estimate > max_readback_bytes and (not allow_resource_override):
            raise ValueError('2N full-resolution RGB readback exceeds resource limit; reduce num_envs or explicitly override')
        self.resource_metadata = dict(real_camera_views=2 * len(origin_values), resolution_wh=[calibration.width, calibration.height], minimum_rgb_readback_bytes_estimate=estimate, max_readback_bytes=max_readback_bytes, estimate_scope='RGBA8_plus_float32_RGB_only;renderer_assets_and_transient_overhead_not_measured', hidden_downsample=False, gpu_memory_measured=False)
        _stere_camer_import___CameraAxesDeviceMixin = _CameraAxesDeviceMixin
        _stere_camer_import__numeric_slot_order = numeric_slot_order
        kit = stere_camer__importlib.import_module('omni.kit.app')
        kit.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.sensors.camera', True)
        base = stere_camer__importlib.import_module('isaacsim.sensors.camera').CameraView

        class DeviceSafeRGBView(_stere_camer_import___CameraAxesDeviceMixin, base):
            pass
        omni_usd = stere_camer__importlib.import_module('omni.usd')
        rep = stere_camer__importlib.import_module('omni.replicator.core')
        core = stere_camer__importlib.import_module('isaacsim.core.nodes.bindings._isaacsim_core_nodes')
        from pxr import Gf
        from pxr import Usd
        from pxr import UsdGeom
        (self.sim, self.calibration, self.device) = (sim, calibration, stere_camer__torch.device(device))
        self.origins = origin_values
        if self.origins.ndim != 2 or self.origins.shape[1] != 3 or (not stere_camer__torch.isfinite(self.origins).all()):
            raise ValueError('finite camera origins [N,3] required')
        self.n = len(self.origins)
        self.count = 2 * self.n
        (self.Usd, self.UsdGeom) = (Usd, UsdGeom)
        self.stage = omni_usd.get_context().get_stage()
        parent = '/World/DH116StereoRGB'
        if self.stage.GetPrimAtPath(parent).IsValid():
            raise RuntimeError('stereo RGB rig already exists')
        UsdGeom.Xform.Define(self.stage, parent)
        for (env, origin) in enumerate(self.origins.numpy()):
            for (side, (k, t)) in enumerate(((calibration.K_left, calibration.T_world_left), (calibration.K_right, calibration.T_world_right))):
                camera = UsdGeom.Camera.Define(self.stage, parent + f'/camera_{2 * env + side}')
                focal = 24.0
                camera.CreateProjectionAttr('perspective')
                camera.CreateFocalLengthAttr(focal)
                camera.CreateHorizontalApertureAttr(calibration.width * focal / float(k[0, 0]))
                camera.CreateVerticalApertureAttr(calibration.height * focal / float(k[1, 1]))
                (horizontal_offset, vertical_offset) = stere_camer__usd_aperture_offsets(k, calibration.width, calibration.height, focal)
                camera.CreateHorizontalApertureOffsetAttr(horizontal_offset)
                camera.CreateVerticalApertureOffsetAttr(vertical_offset)
                camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
                optical = t.numpy().copy()
                optical[:3, 3] += origin
                usd = optical @ stere_camer__np.diag([1.0, -1.0, -1.0, 1.0])
                camera.AddTransformOp().Set(Gf.Matrix4d(*usd.T.reshape(-1).tolist()))
        self.view = DeviceSafeRGBView(prim_paths_expr=parent + '/camera_.*', name='dh116_stereo_rgb', camera_resolution=(calibration.width, calibration.height), output_annotators=['rgb'], reset_xform_properties=False)
        self.paths = list(self.view.prim_paths)
        self.order = _stere_camer_import__numeric_slot_order(self.paths, self.count).to(self.device)
        tiled_width = int(self.view.tiled_resolution[0])
        self.columns = tiled_width // calibration.width
        if tiled_width != self.columns * calibration.width:
            raise RuntimeError('invalid stereo tiled layout')
        self.reference = rep.AnnotatorRegistry.get_annotator('ReferenceTime', device='cpu')
        self.times = rep.AnnotatorRegistry.get_annotator('IsaacReadTimes', device='cpu')
        self.reference.attach([self.view.get_render_product_path()])
        self.times.attach([self.view.get_render_product_path()])
        self.core = core.acquire_interface()

    def _readback(self):
        _stere_camer_import__usd_pose_to_optical = usd_pose_to_optical
        (positions, quats) = self.view.get_world_poses(camera_axes='usd', usd=True)
        transforms = stere_camer__np.stack([_stere_camer_import__usd_pose_to_optical(p, q) for (p, q) in zip(_array(positions), _array(quats))])
        focal = stere_camer__np.asarray(self.view.get_focal_lengths()) * 10.0
        ah = stere_camer__np.asarray(self.view.get_horizontal_apertures())
        av = stere_camer__np.asarray(self.view.get_vertical_apertures())
        K = stere_camer__np.tile(stere_camer__np.eye(3), (self.count, 1, 1))
        c = self.calibration
        cache = self.UsdGeom.XformCache(self.Usd.TimeCode.Default())
        for (i, path) in enumerate(self.paths):
            camera = self.UsdGeom.Camera(self.stage.GetPrimAtPath(path))
            K[i] = usd_aperture_intrinsics(c.width, c.height, focal[i], ah[i], av[i], camera.GetHorizontalApertureOffsetAttr().Get(), camera.GetVerticalApertureOffsetAttr().Get()).numpy()
            authored = stere_camer__np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())).T @ stere_camer__np.diag([1.0, -1.0, -1.0, 1.0])
            if not stere_camer__np.allclose(authored, transforms[i], atol=1e-05, rtol=0.0):
                raise ValueError('USD/runtime stereo calibration mismatch')
        index = self.order.cpu().numpy()
        K = K[index].reshape(self.n, 2, 3, 3)
        transforms = transforms[index].reshape(self.n, 2, 4, 4)
        self.calibration.validate_readback(K[:, 0], K[:, 1], transforms[:, 0], transforms[:, 1], origins=self.origins)
        return (K, transforms)

    def read(self):
        _stere_camer_import__paired_render_timestamp = paired_render_timestamp
        _stere_camer_import__detile = detile
        ref = self.reference.get_data()
        if not isinstance(ref, dict) or 'referenceTimeNumerator' not in ref:
            return None
        legacy = float(self.core.get_sim_time_at_swh_frame(int(ref['referenceTimeNumerator'])))
        timing = _stere_camer_import__paired_render_timestamp(ref, self.times.get_data(), legacy_time=legacy)
        c = self.calibration
        rgb = _stere_camer_import__detile(self.view.get_rgb_tiled(device='cuda'), count=self.count, width=c.width, height=c.height, columns=self.columns, channels=3).to(self.device)[self.order].reshape(self.n, 2, c.height, c.width, 3)
        after = _stere_camer_import__paired_render_timestamp(self.reference.get_data(), self.times.get_data(), legacy_time=legacy)
        if (after['frame_id'], after['simulation_time'], after['render_frame_number']) != (timing['frame_id'], timing['simulation_time'], timing['render_frame_number']):
            raise ValueError('stereo render changed during RGB read')
        (K, T) = self._readback()
        integer_rgb = not rgb.is_floating_point()
        rgb = rgb.float()
        if integer_rgb or bool((rgb > 1).any()):
            rgb = rgb / 255.0
        return dict(left_rgb=rgb[:, 0].permute(0, 3, 1, 2).contiguous(), right_rgb=rgb[:, 1].permute(0, 3, 1, 2).contiguous(), left_frame_id=timing['frame_id'], right_frame_id=timing['frame_id'], left_simulation_time=timing['simulation_time'], right_simulation_time=timing['simulation_time'], K_left=K[:, 0], K_right=K[:, 1], T_world_left=T[:, 0], T_world_right=T[:, 1], native_evidence=True, render_timing=timing)

def stere_rgb_bridg___same_snapshot(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all((stere_rgb_bridg___same_snapshot(x, y) for (x, y) in zip(a, b)))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all((stere_rgb_bridg___same_snapshot(a[k], b[k]) for k in a))
    return a == b

def _file_identity(path):
    path = previ_warp_readb__Path(path).resolve()
    return dict(path=str(path), sha256=previ_warp_readb__hashlib.sha256(path.read_bytes()).hexdigest())

class _NativeRenderSetting:
    """Carb setting access only; must be constructed inside an active Kit process."""
    path = '/app/player/playSimulations'

    def __init__(self):
        self.settings = stere_rgb_bridg__importlib.import_module('carb.settings').get_settings()

    def get(self):
        return self.settings.get(self.path)

    def set(self, value):
        self.settings.set_bool(self.path, value)
stere_rgb_owned_nativ___NativeRGBPairReader = _NativeRGBPairReader

def _pointer_abi(module, warp):
    annotator = _file_identity(module.__file__)
    warp_init = _file_identity(warp.__file__)
    warp_array = _file_identity(previ_warp_readb__inspect.getfile(warp.array.__init__))
    parameters = previ_warp_readb__inspect.signature(warp.array.__init__).parameters
    raw = previ_warp_readb__inspect.unwrap(module._reshape_output_ptr)
    same_warp = isinstance(raw, types.FunctionType) and raw.__globals__.get('wp') is warp
    receipt = dict(warp_version=warp.__version__, warp_module=warp_init, warp_array=warp_array, annotator_module=annotator, annotator_uses_same_warp=same_warp, owner_argument='owner' in parameters, deleter_argument='deleter' in parameters, warp_extension_config=None, policy=None, error=None)
    if annotator['sha256'] != ANNOTATOR_SHA256 or not same_warp:
        receipt['error'] = 'unverified Isaac Replicator RGB pointer source'
    elif warp.__version__ == '1.5.0' and 'owner' in parameters and (parameters['owner'].default is False):
        root = previ_warp_readb__Path(warp_init['path']).parents[1]
        fingerprints = {str(previ_warp_readb__Path(row['path']).relative_to(root)): row['sha256'] for row in (warp_init, warp_array) if previ_warp_readb__Path(row['path']).is_relative_to(root)}
        if fingerprints == BUNDLED_WARP_SOURCES:
            receipt['policy'] = 'native_owner_false'
            config = root / 'config/extension.toml'
            if config.is_file():
                receipt['warp_extension_config'] = _file_identity(config)
        else:
            receipt['error'] = 'unverified bundled Warp RGB pointer source'
    elif warp.__version__ == '1.12.1' and 'owner' not in parameters and ('deleter' in parameters):
        receipt['policy'] = 'scoped_owner_to_deleter'
    else:
        receipt['error'] = 'unverified Isaac/Warp RGB pointer ABI'
    return receipt

class RGBPairBridge:

    def __init__(self, sim, reader, calibration, snapshot, *, render_setting, origins=None, max_render_attempts=1):
        if not isinstance(calibration, stere_rgb_bridg__StereoCalibration) or not callable(snapshot):
            raise ValueError('calibration and physical snapshot callback required')
        (self.sim, self.reader, self.calibration) = (sim, reader, calibration)
        if not callable(getattr(render_setting, 'get', None)) or not callable(getattr(render_setting, 'set', None)):
            raise ValueError('render setting get/set required')
        (self.snapshot, self.origins, self.last_frame) = (snapshot, origins, -1)
        self.render_setting = render_setting
        if type(max_render_attempts) is not int or not 1 <= max_render_attempts <= 32:
            raise ValueError('max_render_attempts must be an integer from 1 to 32')
        self.max_render_attempts = max_render_attempts
        self.last_epoch = None
        self.last_episode = None
        self.last_capture_diagnostic = None
        self._owns_reader = False
        self._closed = False
        self._close_error = None

    @classmethod
    def from_native(cls, sim, origins, calibration, snapshot, *, width=640, height=480, horizontal_fov_degrees=75.0, device='cuda', max_readback_bytes=1024 ** 3, allow_resource_override=False, render_setting=None, max_render_attempts=1):
        from dh116.env.cameras import OwnedNativeRGBPairReader as OwnedNativeRGBPairReader
        if not isinstance(calibration, stere_rgb_bridg__StereoCalibration) or not callable(snapshot):
            raise ValueError('calibration and physical snapshot callback required')
        if type(max_render_attempts) is not int or not 1 <= max_render_attempts <= 32:
            raise ValueError('max_render_attempts must be an integer from 1 to 32')
        c = stere_rgb_bridg__rerender_calibration(calibration, width, height, horizontal_fov_degrees=horizontal_fov_degrees, allow_resource_override=allow_resource_override)
        setting = _NativeRenderSetting() if render_setting is None else render_setting
        if not callable(getattr(setting, 'get', None)) or not callable(getattr(setting, 'set', None)):
            raise ValueError('render setting get/set required')
        if type(setting.get()) is not bool:
            raise ValueError('render setting must be a readable boolean')
        reader = OwnedNativeRGBPairReader(sim, origins, c, device, max_readback_bytes=max_readback_bytes, allow_resource_override=allow_resource_override)
        try:
            bridge = cls(sim, reader, c, snapshot, origins=origins, render_setting=setting, max_render_attempts=max_render_attempts)
        except BaseException:
            reader.close()
            raise
        bridge._owns_reader = True
        return bridge

    def close(self):
        if self._closed:
            if self._close_error is not None:
                raise RuntimeError(self._close_error)
            return
        self._closed = True
        try:
            if self._owns_reader:
                self.reader.close()
        except BaseException as exc:
            self._close_error = str(exc)
            raise
        finally:
            self.sim = self.reader = self.snapshot = self.origins = self.render_setting = None

    def capture(self):
        if self._closed:
            raise RuntimeError('RGB bridge is closed')
        original = self.render_setting.get()
        if type(original) is not bool:
            raise ValueError('render setting must be a readable boolean')
        try:
            return self._capture()
        finally:
            self.render_setting.set(original)
            if self.render_setting.get() is not original:
                raise RuntimeError('render setting restoration failed')

    def _capture(self):
        self.last_capture_diagnostic = None
        before = stere_rgb_bridg__copy.deepcopy(self.snapshot())
        rows = before if isinstance(before, list) else [before]
        if not rows or any((not isinstance(r, dict) or set(r) != stere_rgb_bridg___IDENTITY_KEYS | stere_rgb_bridg___RAW_KEYS for r in rows)):
            raise ValueError('physical identity fields are missing or unexpected')
        for row in rows:
            if type(row['rod_step']) is not int or row['rod_step'] < 0 or (not isinstance(row['state_sha256'], str)) or (not stere_rgb_bridg__re.fullmatch('[0-9a-f]{64}', row['state_sha256'])) or (row['calibration_sha256'] != self.calibration.sha256):
                raise ValueError('invalid rod clock, state digest or calibration SHA')
            if any((type(row[k]) is not int or row[k] < 0 for k in ('sdk_step', 'rod_tick'))) or any((isinstance(row[k], bool) or not isinstance(row[k], (int, float)) or (not stere_rgb_bridg__math.isfinite(row[k])) or (row[k] < 0) for k in ('sdk_timestamp_s', 'solver_time_s'))) or any((not isinstance(row[k], str) or not row[k] for k in ('rod_uid', 'material_reference_uid'))):
                raise ValueError('invalid raw SDK/rod clock or material identity')
        sim_time = float(self.sim.current_time)
        if not stere_rgb_bridg__math.isfinite(sim_time) or sim_time < 0:
            raise ValueError('invalid physical simulation time')
        sdk_step = self.sim.current_time_step_index
        if type(sdk_step) is not int or sdk_step < 0 or any((row['sdk_step'] != sdk_step or abs(row['sdk_timestamp_s'] - sim_time) > 1e-06 for row in rows)):
            raise ValueError('raw SDK clock and simulation clock mismatch')
        identity_rows = [{key: row[key] for key in _VALIDATION_KEYS} for row in rows]
        stere_rgb_bridg__validate_identity(identity_rows, len(rows), sim_time)
        episode = tuple(((row['run_uuid'], row['env_id'], row['episode_uid'], row['reset_id']) for row in rows))
        previous_epoch = self.last_epoch if episode == self.last_episode else None

        def check_state():
            if float(self.sim.current_time) != sim_time or type(self.sim.current_time_step_index) is not int or self.sim.current_time_step_index != sdk_step or (not stere_rgb_bridg___same_snapshot(self.snapshot(), before)):
                raise RuntimeError('physical state advanced during render or RGB read')
        packet = None
        read_clock = getattr(self.reader, 'read_clock', None)
        for attempts in range(1, self.max_render_attempts + 1):
            check_state()
            prior_clock = read_clock() if callable(read_clock) else None
            check_state()
            self.sim.render()
            check_state()
            if callable(read_clock):
                if prior_clock is None:
                    continue
                if not isinstance(prior_clock, dict) or type(prior_clock.get('render_frame_number')) is not int or prior_clock['render_frame_number'] < 0:
                    raise ValueError('invalid pre-render clock')
                rendered_clock = read_clock()
                check_state()
                if rendered_clock is None:
                    continue
                if not isinstance(rendered_clock, dict) or type(rendered_clock.get('render_frame_number')) is not int or rendered_clock['render_frame_number'] < 0:
                    raise ValueError('invalid post-render clock')
                if rendered_clock['render_frame_number'] <= prior_clock['render_frame_number']:
                    continue
            candidate = self.reader.read()
            current_clock = read_clock() if callable(read_clock) else None
            check_state()
            if candidate is None:
                continue
            if not isinstance(candidate, dict) or not set(candidate) <= _PACKET_KEYS:
                raise ValueError('RGB reader returned forbidden fields')
            (frame, timestamp) = stere_rgb_bridg__validate_pair_timing(candidate, -1)
            timing = candidate.get('render_timing')
            if not isinstance(timing, dict) or timing.get('frame_id') != frame or timing.get('simulation_time') != timestamp or (type(timing.get('render_frame_number')) is not int) or (timing['render_frame_number'] < 0) or (abs(timestamp - sim_time) > 1e-06):
                self.last_capture_diagnostic = dict(sdk_step=sdk_step, sdk_timestamp_s=sim_time, render_frame_id=frame, render_timestamp_s=timestamp, absolute_time_error_s=abs(timestamp - sim_time), timestamp_tolerance_s=1e-06, render_timing=repr(timing)[:2048], reader_clock_diagnostic=repr(getattr(self.reader, 'last_clock_diagnostic', None))[:2048])
                raise ValueError('render frame and physical timestamp mismatch; clocks=' + repr(self.last_capture_diagnostic))
            epoch = timing['render_frame_number']
            if callable(read_clock):
                expected = {key: timing[key] for key in ('frame_id', 'simulation_time', 'render_frame_number')}
                if rendered_clock != expected or current_clock != expected:
                    raise ValueError('render clock changed during RGB read')
            if previous_epoch is not None and epoch <= previous_epoch:
                continue
            packet = candidate
            break
        if packet is None:
            raise RuntimeError('fresh synchronized RGB unavailable within render attempt limit; reader_clock=' + str(getattr(self.reader, 'last_clock_diagnostic', None)))
        (left, right) = (packet['left_rgb'], packet['right_rgb'])
        stere_rgb_bridg__validate_images(left, right, self.calibration)
        if len(rows) != left.shape[0]:
            raise ValueError('identity and camera batch mismatch')
        self.calibration.validate_readback(packet['K_left'], packet['K_right'], packet['T_world_left'], packet['T_world_right'], origins=self.origins)
        if float(self.sim.current_time) != sim_time or type(self.sim.current_time_step_index) is not int or self.sim.current_time_step_index != sdk_step or (not stere_rgb_bridg___same_snapshot(self.snapshot(), before)):
            raise RuntimeError('physical state advanced during RGB read')
        self.last_frame = frame
        (self.last_episode, self.last_epoch) = (episode, epoch)
        return dict(left_rgb=left.clone(), right_rgb=right.clone(), identity=before, frame_id=frame, simulation_time=timestamp, calibration_sha256=self.calibration.sha256, render_passes=attempts, K_left=stere_rgb_bridg__torch.as_tensor(packet['K_left']).clone(), K_right=stere_rgb_bridg__torch.as_tensor(packet['K_right']).clone(), render_timing={key: timing[key] for key in ('frame_id', 'simulation_time', 'render_frame_number')})
from dh116.physics.vision import usd_aperture_offsets as stere_rgb_owned_nativ__usd_aperture_offsets

def inspect_rgb_readback_abi():
    """Read the active Isaac process source and constructor contract without reading RGB."""
    with previ_warp_readb___LOCK:
        module = previ_warp_readb__importlib.import_module('omni.replicator.core.scripts.utils.annotator_utils')
        return _pointer_abi(module, previ_warp_readb__importlib.import_module('warp'))

class OwnedNativeRGBPairReader(stere_rgb_owned_nativ___NativeRGBPairReader):

    def __init__(self, sim, origins, calibration, device, *, max_readback_bytes=1024 ** 3, allow_resource_override=False):
        self.resources = stere_rgb_owned_nativ__RGBResources()
        self._last_valid_clock = None
        try:
            origin_values = stere_rgb_owned_nativ__torch.as_tensor(origins, dtype=stere_rgb_owned_nativ__torch.float64).detach().cpu()
            if origin_values.ndim != 2 or origin_values.shape[1] != 3 or (not stere_rgb_owned_nativ__torch.isfinite(origin_values).all()):
                raise ValueError('finite camera origins [N,3] required')
            if type(max_readback_bytes) is not int or max_readback_bytes < 1:
                raise ValueError('positive camera readback resource limit required')
            estimate = 2 * len(origin_values) * calibration.width * calibration.height * 16
            if estimate > max_readback_bytes and (not allow_resource_override):
                raise ValueError('2N full-resolution RGB readback exceeds resource limit; reduce num_envs or explicitly override')
            self.resource_metadata = dict(real_camera_views=2 * len(origin_values), resolution_wh=[calibration.width, calibration.height], minimum_rgb_readback_bytes_estimate=estimate, max_readback_bytes=max_readback_bytes, estimate_scope='RGBA8_plus_float32_RGB_only;renderer_assets_and_transient_overhead_not_measured', hidden_downsample=False, gpu_memory_measured=False)
            _stere_rgb_owned_nativ_import___CameraAxesDeviceMixin = _CameraAxesDeviceMixin
            _stere_rgb_owned_nativ_import__numeric_slot_order = numeric_slot_order
            kit = stere_rgb_owned_nativ__importlib.import_module('omni.kit.app')
            kit.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.sensors.camera', True)
            base = stere_rgb_owned_nativ__importlib.import_module('isaacsim.sensors.camera').CameraView

            class DeviceSafeRGBView(_stere_rgb_owned_nativ_import___CameraAxesDeviceMixin, base):
                pass
            omni_usd = stere_rgb_owned_nativ__importlib.import_module('omni.usd')
            rep = stere_rgb_owned_nativ__importlib.import_module('omni.replicator.core')
            core = stere_rgb_owned_nativ__importlib.import_module('isaacsim.core.nodes.bindings._isaacsim_core_nodes')
            from pxr import Gf
            from pxr import Usd
            from pxr import UsdGeom
            (self.sim, self.calibration, self.device) = (sim, calibration, stere_rgb_owned_nativ__torch.device(device))
            self.origins = origin_values
            if self.origins.ndim != 2 or self.origins.shape[1] != 3 or (not stere_rgb_owned_nativ__torch.isfinite(self.origins).all()):
                raise ValueError('finite camera origins [N,3] required')
            self.n = len(self.origins)
            self.count = 2 * self.n
            (self.Usd, self.UsdGeom) = (Usd, UsdGeom)
            self.stage = omni_usd.get_context().get_stage()
            parent = '/World/DH116StereoRGB'
            if self.stage.GetPrimAtPath(parent).IsValid():
                raise RuntimeError('stereo RGB rig already exists')
            self.resources.define_rig(self.stage, UsdGeom.Xform.Define)
            for (env, origin) in enumerate(self.origins.numpy()):
                for (side, (k, t)) in enumerate(((calibration.K_left, calibration.T_world_left), (calibration.K_right, calibration.T_world_right))):
                    camera = UsdGeom.Camera.Define(self.stage, parent + f'/camera_{2 * env + side}')
                    focal = 24.0
                    camera.CreateProjectionAttr('perspective')
                    camera.CreateFocalLengthAttr(focal)
                    camera.CreateHorizontalApertureAttr(calibration.width * focal / float(k[0, 0]))
                    camera.CreateVerticalApertureAttr(calibration.height * focal / float(k[1, 1]))
                    (horizontal_offset, vertical_offset) = stere_rgb_owned_nativ__usd_aperture_offsets(k, calibration.width, calibration.height, focal)
                    camera.CreateHorizontalApertureOffsetAttr(horizontal_offset)
                    camera.CreateVerticalApertureOffsetAttr(vertical_offset)
                    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
                    optical = t.numpy().copy()
                    optical[:3, 3] += origin
                    usd = optical @ stere_rgb_owned_nativ__np.diag([1.0, -1.0, -1.0, 1.0])
                    camera.AddTransformOp().Set(Gf.Matrix4d(*usd.T.reshape(-1).tolist()))
            self.resources.unsafe = True
            self.view = DeviceSafeRGBView(prim_paths_expr=parent + '/camera_.*', name='dh116_stereo_rgb', camera_resolution=(calibration.width, calibration.height), output_annotators=['rgb'], reset_xform_properties=False)
            self.resources.bind_view(self.view)
            self.paths = list(self.view.prim_paths)
            self.order = _stere_rgb_owned_nativ_import__numeric_slot_order(self.paths, self.count).to(self.device)
            tiled_width = int(self.view.tiled_resolution[0])
            self.columns = tiled_width // calibration.width
            if tiled_width != self.columns * calibration.width:
                raise RuntimeError('invalid stereo tiled layout')
            self.reference = rep.AnnotatorRegistry.get_annotator('ReferenceTime', device='cpu')
            self.times = rep.AnnotatorRegistry.get_annotator('IsaacReadTimes', device='cpu')
            path = self.view.get_render_product_path()
            self.resources.attach(self.reference, path, lambda : self.reference.attach([path]))
            self.resources.attach(self.times, path, lambda : self.times.attach([path]))
            self.core = core.acquire_interface()
        except BaseException as exc:
            try:
                self.resources.close()
            except Exception as cleanup_error:
                raise RuntimeError('native reader initialization failed; cleanup incomplete: ' + str(cleanup_error)) from exc
            raise

    def read(self):
        _stere_rgb_owned_nativ_import__isaac45_rgb_readback_scope = isaac45_rgb_readback_scope
        with _stere_rgb_owned_nativ_import__isaac45_rgb_readback_scope():
            return super().read()

    def read_clock(self):
        _stere_rgb_owned_nativ_import__paired_render_timestamp = paired_render_timestamp
        ref = self.reference.get_data()
        times = self.times.get_data()
        startup = getattr(self, '_last_valid_clock', None) is None
        self.last_clock_diagnostic = dict(reference=repr(ref), times=repr(times), startup=startup)
        ref_zero = startup and isinstance(ref, dict) and (ref.get('referenceTimeNumerator') == 0) and (ref.get('referenceTimeDenominator') == 0)
        time_keys = ('swhFrameNumber', 'frameNumber', 'simulationTime', 'rationalTimeOfSimNumerator', 'rationalTimeOfSimDenominator')
        times_zero = startup and isinstance(times, dict) and all((k in times for k in time_keys)) and all((times[k] == 0 for k in time_keys))
        ref_empty = ref_zero or ref is None or (isinstance(ref, dict) and (not ref))
        times_empty = times_zero or times is None or (isinstance(times, dict) and (not times))
        if not ref_empty:
            if not isinstance(ref, dict) or not {'referenceTimeNumerator', 'referenceTimeDenominator'} <= ref.keys():
                raise ValueError('invalid ReferenceTime metadata')
            try:
                if int(ref['referenceTimeNumerator']) < 0 or int(ref['referenceTimeDenominator']) <= 0:
                    raise ValueError('invalid ReferenceTime clock: ' + repr(ref))
            except (TypeError, OverflowError) as exc:
                raise ValueError('invalid ReferenceTime clock: ' + repr(ref)) from exc
        if not times_empty:
            keys = {'swhFrameNumber', 'frameNumber', 'simulationTime', 'rationalTimeOfSimNumerator', 'rationalTimeOfSimDenominator'}
            if not isinstance(times, dict) or not keys <= times.keys():
                raise ValueError('invalid IsaacReadTimes metadata')
            try:
                if int(times['swhFrameNumber']) < 0 or int(times['frameNumber']) < 0 or int(times['rationalTimeOfSimNumerator']) < 0 or (int(times['rationalTimeOfSimDenominator']) <= 0) or (not stere_rgb_owned_nativ__np.isfinite(float(times['simulationTime']))) or (float(times['simulationTime']) < 0):
                    raise ValueError('invalid IsaacReadTimes clock: ' + repr(times))
            except (TypeError, OverflowError) as exc:
                raise ValueError('invalid IsaacReadTimes clock: ' + repr(times)) from exc
        if ref_empty or times_empty:
            return None
        legacy = float(self.core.get_sim_time_at_swh_frame(int(ref['referenceTimeNumerator'])))
        timing = _stere_rgb_owned_nativ_import__paired_render_timestamp(ref, times, legacy_time=legacy)
        self._last_valid_clock = dict(timing)
        return {key: timing[key] for key in ('frame_id', 'simulation_time', 'render_frame_number')}

    def close(self):
        try:
            self.resources.close()
        finally:
            self.sim = self.view = self.stage = self.reference = self.times = self.core = None
tiled_stere_captu___IDENTITY_KEYS = stere_rgb_bridg___IDENTITY_KEYS
tiled_stere_captu___RAW_KEYS = stere_rgb_bridg___RAW_KEYS
tiled_stere_captu___same_snapshot = stere_rgb_bridg___same_snapshot
tiled_stere_captu___NativeRenderSetting = _NativeRenderSetting

@previ_warp_readb__contextmanager
def isaac45_rgb_readback_scope():
    """Only use the exact audited installed combination, inside an active App."""
    with previ_warp_readb___LOCK:
        module = previ_warp_readb__importlib.import_module('omni.replicator.core.scripts.utils.annotator_utils')
        warp = previ_warp_readb__importlib.import_module('warp')
        receipt = _pointer_abi(module, warp)
        if receipt['policy'] == 'native_owner_false':
            yield
        elif receipt['policy'] == 'scoped_owner_to_deleter':
            replacement = adapt_pointer_function(module._reshape_output_ptr, warp)
            with binding_scope(module, replacement):
                yield
        else:
            raise ValueError(receipt['error'])
VERSION = 'dh116-fixed-dual-tiled-stereo-v1'
tiled_stere_nativ__OwnedNativeRGBPairReader = OwnedNativeRGBPairReader
_CLOCK_KEYS = {'frame_id', 'simulation_time', 'render_frame_number'}

class SequentialNativeStereoReader:
    pose_backend = 'usd_camera_prim_isaac45'

    def __init__(self, sim, origins, calibration, device, *, max_readback_bytes=1024 ** 3, side='left', rig_path=tiled_stere_nativ__RIG_PATH):
        try:
            self.resources = tiled_stere_nativ__TiledResources(rig_path=rig_path)
            self.sensor = None
            self.sim = sim
            self.device = tiled_stere_nativ__torch.device(device)
            self.calibration = calibration
            self._closed = False
            self._close_error = None
            self._native_initialized = False
            self._last_valid_clock = None
            self.reference = self.times = self.core = None
            try:
                if self.device.type != 'cuda':
                    raise ValueError('native tiled preview requires CUDA')
                origin = tiled_stere_nativ__torch.as_tensor(origins, dtype=tiled_stere_nativ__torch.float64).cpu()
                if origin.shape != (1, 3) or not tiled_stere_nativ__torch.isfinite(origin).all():
                    raise ValueError('this independent native preview requires one finite environment origin')
                if type(max_readback_bytes) is not int or max_readback_bytes < 1:
                    raise ValueError('positive RGB readback budget required')
                estimate = 2 * calibration.width * calibration.height * 16
                if estimate > max_readback_bytes:
                    raise ValueError('native RGB readback exceeds resource budget')
                self.origins = origin
                self.n = 1
                root = tiled_stere_nativ__Path(__file__).resolve().parents[3]
                sources = tiled_stere_nativ__verify_tiled_sources(root)
                import omni.replicator.core as rep
                import omni.usd
                import omni.kit.app
                ensure_clock_annotators(omni.kit.app.get_app().get_extension_manager(), rep.AnnotatorRegistry)
                from pxr import Gf
                from pxr import UsdGeom
                from isaaclab.sensors import TiledCamera
                from isaaclab.sensors import TiledCameraCfg
                if tiled_stere_nativ__Path(tiled_stere_nativ__inspect.getfile(TiledCamera)).resolve() != tiled_stere_nativ__Path(sources['tiled']['path']).resolve():
                    raise ValueError('imported TiledCamera differs from audited installation')
                self.stage = omni.usd.get_context().get_stage()
                self.resources.define_rig(self.stage, UsdGeom.Xform.Define)
                UsdGeom.Xform.Define(self.stage, rig_path + '/env_0')
                path = rig_path + '/env_0/Camera'
                self.path = path
                camera = UsdGeom.Camera.Define(self.stage, path)
                author_standard_camera_ops(camera, UsdGeom, Gf)
                c = calibration
                focal = 24.0
                camera.CreateProjectionAttr('perspective')
                camera.CreateFocalLengthAttr(focal)
                camera.CreateHorizontalApertureAttr(c.width * focal / float(c.K_left[0, 0]))
                camera.CreateVerticalApertureAttr(c.height * focal / float(c.K_left[1, 1]))
                camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
                cfg = TiledCameraCfg(prim_path=path, spawn=None, width=c.width, height=c.height, data_types=['rgb'], update_period=0.0, update_latest_camera_pose=True, debug_vis=False)

                class ExplicitTiledCamera(TiledCamera):

                    def _create_buffers(self):
                        prepare_renderer_view(self._view)
                        return super()._create_buffers()

                    def __del__(self):
                        return None
                sensor = ExplicitTiledCamera.__new__(ExplicitTiledCamera)
                self.sensor = self.resources.sensor = sensor
                sensor._annotators = {}
                sensor._render_product_paths = []
                sensor._sensor_prims = []
                for name in ('_prim_deletion_callback_id', '_initialize_handle', '_invalidate_initialize_handle', '_debug_vis_handle'):
                    setattr(sensor, name, None)
                ExplicitTiledCamera.__init__(sensor, cfg)
                with self.resources.product_scope(rep.create):
                    if sensor.is_initialized:
                        raise RuntimeError('unexpected tiled initialization before ownership tracking')
                    sensor._initialize_impl()
                    sensor._is_initialized = True
                if tiled_stere_nativ__Path(tiled_stere_nativ__inspect.getfile(type(sensor._view))).resolve() != tiled_stere_nativ__Path(sources['xform_view']['path']).resolve():
                    raise ValueError('imported camera view differs from audited installation')
                prepare_renderer_view(sensor._view)
                paths = sensor.render_product_paths
                if paths != [self.resources.product.path]:
                    raise ValueError('tiled render product identity differs')
                self.reference = rep.AnnotatorRegistry.get_annotator('ReferenceTime', device='cpu')
                self.times = rep.AnnotatorRegistry.get_annotator('IsaacReadTimes', device='cpu')
                self.resources.attach(self.reference, paths[0])
                self.resources.attach(self.times, paths[0])
                core = tiled_stere_nativ__importlib.import_module('isaacsim.core.nodes.bindings._isaacsim_core_nodes')
                self.core = core.acquire_interface()
                self.select(side)
                self._native_initialized = True
            except BaseException as original:
                try:
                    self.close()
                except BaseException as cleanup:
                    raise RuntimeError('tiled initialization failed; cleanup incomplete: ' + str(cleanup)) from original
                raise
        finally:
            sim = sensor = camera = self = None

    def select(self, side):
        if self._closed:
            raise RuntimeError('tiled reader is closed')
        if side not in ('left', 'right'):
            raise ValueError('left/right camera side required')
        from isaaclab.utils.math import quat_from_matrix
        transform = getattr(self.calibration, 'T_world_' + side).to(device=self.device, dtype=tiled_stere_nativ__torch.float32)
        position = transform[:3, 3][None] + self.origins.to(device=self.device, dtype=tiled_stere_nativ__torch.float32)
        rotation = quat_from_matrix(transform[:3, :3])[None]
        k = getattr(self.calibration, 'K_' + side)[None].to(device=self.device, dtype=tiled_stere_nativ__torch.float32)
        self.sensor.set_intrinsic_matrices(k, focal_length=24.0)
        self.sensor.set_world_poses(positions=position, orientations=rotation, convention='ros')
        self.side = side

    def read_clock(self):
        if self._closed:
            raise RuntimeError('tiled reader is closed')
        return tiled_stere_nativ__OwnedNativeRGBPairReader.read_clock(self)

    def _readback(self):
        from isaaclab.utils.math import matrix_from_quat
        data = self.sensor.data
        transform = tiled_stere_nativ__torch.eye(4, dtype=tiled_stere_nativ__torch.float64, device=self.device)[None].repeat(self.n, 1, 1)
        transform[:, :3, :3] = matrix_from_quat(data.quat_w_ros).to(tiled_stere_nativ__torch.float64)
        transform[:, :3, 3] = data.pos_w.to(tiled_stere_nativ__torch.float64)
        from pxr import Usd
        from pxr import UsdGeom
        validate_usd_camera_pose(self.stage, self.path, transform, Usd, UsdGeom)
        return (data.intrinsic_matrices.clone(), transform)

    def read(self):
        if self._closed:
            raise RuntimeError('tiled reader is closed')
        _tiled_stere_nativ_import__isaac45_rgb_readback_scope = isaac45_rgb_readback_scope
        before = self.read_clock()
        if before is None:
            return None
        with _tiled_stere_nativ_import__isaac45_rgb_readback_scope():
            self.sensor.update(0, force_recompute=True)
            image = self.sensor.data.output['rgb']
            c = self.calibration
            if not isinstance(image, tiled_stere_nativ__torch.Tensor) or image.shape != (self.n, c.height, c.width, 3):
                raise ValueError('unexpected native RGB shape')
            if image.dtype != tiled_stere_nativ__torch.uint8:
                raise ValueError('native RGB must retain audited uint8 source')
            rgb = image.permute(0, 3, 1, 2).contiguous().float() / 255.0
            (k, transform) = self._readback()
        after = self.read_clock()
        if before != after:
            raise ValueError('raw render clock changed during TiledCamera update')
        return dict(rgb=rgb.clone(), K=k, T_world=transform, raw_clock=before, raw_time_metadata=str(getattr(self, 'last_clock_diagnostic', None))[:4096], native_evidence=getattr(self, '_native_initialized', False) is True)

    def close(self):
        if self._closed:
            if self._close_error:
                raise RuntimeError(self._close_error)
            return
        self._closed = True
        try:
            if self.device.type == 'cuda':
                tiled_stere_nativ__torch.cuda.synchronize(self.device)
            self.resources.close()
        except BaseException as error:
            self._close_error = str(error)
            raise
        finally:
            self.sensor = self.sim = self.stage = self.reference = self.times = self.core = None
_FRAME_KEYS = {'rgb', 'K', 'T_world', 'raw_clock', 'native_evidence', 'raw_time_metadata'}

class FixedNativeCameraReader(SequentialNativeStereoReader):
    """One physical camera: its pose is authored once during construction."""

    def select(self, side):
        if self._native_initialized:
            raise RuntimeError('fixed camera pose cannot be switched after initialization')
        return super().select(side)

def _clock(value):
    if not isinstance(value, dict) or set(value) != _CLOCK_KEYS or any((type(value[k]) is not int or value[k] < 0 for k in ('frame_id', 'render_frame_number'))) or isinstance(value['simulation_time'], bool) or (not isinstance(value['simulation_time'], (int, float))) or (not tiled_stere_captu__math.isfinite(value['simulation_time'])) or (value['simulation_time'] < 0):
        raise ValueError('invalid raw render clock')
    return tiled_stere_captu__copy.deepcopy(value)

class TiledNativeStereoReader:
    """Two fixed camera prims, two render products, one external render boundary."""
    pose_backend = 'two_fixed_usd_cameras_isaac45'

    def __init__(self, sim, origins, calibration, device, *, max_readback_bytes=1024 ** 3):
        self.cameras = {}
        self._closed = False
        self._close_error = None
        try:
            if type(max_readback_bytes) is not int or max_readback_bytes < 2:
                raise ValueError('positive dual camera readback budget required')
            for side in ('left', 'right'):
                self.cameras[side] = FixedNativeCameraReader(sim, origins, calibration, device, max_readback_bytes=max_readback_bytes // 2, side=side, rig_path=tiled_stere_nativ__RIG_PATH + '_' + side.title())
        except BaseException:
            self.close()
            raise
        finally:
            sim = None

    def read_clock(self):
        if self._closed:
            raise RuntimeError('fixed stereo reader is closed')
        return self.cameras['left'].read_clock()

    def read_pair(self):
        if self._closed:
            raise RuntimeError('fixed stereo reader is closed')
        clocks = {side: camera.read_clock() for (side, camera) in self.cameras.items()}
        if any((clock is None for clock in clocks.values())):
            return None
        if clocks['left'] != clocks['right']:
            raise ValueError('fixed camera render products have different raw clocks')
        pair = {side: camera.read() for (side, camera) in self.cameras.items()}
        if any((row is None for row in pair.values())):
            return None
        if any((row['raw_clock'] != clocks[side] for (side, row) in pair.items())):
            raise ValueError('fixed camera clock changed during paired readback')
        return pair

    def close(self):
        if self._closed:
            if self._close_error:
                raise RuntimeError(self._close_error)
            return
        self._closed = True
        errors = []
        for (side, camera) in list(self.cameras.items()):
            try:
                camera.resources.detach()
            except BaseException as error:
                errors.append(side + ' detach: ' + str(error))
            finally:
                camera = None
        for (side, camera) in list(self.cameras.items()):
            try:
                camera.close()
            except BaseException as error:
                errors.append(side + ': ' + str(error))
            finally:
                camera = None
        self.cameras.clear()
        if errors:
            self._close_error = '; '.join(errors)
            raise RuntimeError(self._close_error)

class TiledStereoCapture:

    def __init__(self, sim, reader, calibration, snapshot, *, render_setting, max_render_attempts=4, origins=None):
        if not isinstance(calibration, tiled_stere_captu__StereoCalibration) or not callable(snapshot):
            raise ValueError('calibration and physical snapshot required')
        if type(max_render_attempts) is not int or not 1 <= max_render_attempts <= 16:
            raise ValueError('bounded per-view render attempts required')
        if any((not callable(getattr(reader, n, None)) for n in ('read_clock', 'read_pair', 'close'))):
            raise ValueError('owned fixed paired-camera reader required')
        if any((not callable(getattr(render_setting, n, None)) for n in ('get', 'set'))):
            raise ValueError('render setting get/set required')
        (self.sim, self.reader, self.calibration, self.snapshot) = (sim, reader, calibration, snapshot)
        (self.render_setting, self.max_render_attempts, self.origins) = (render_setting, max_render_attempts, origins)
        self.last_capture_id = 0
        self.last_epoch = None
        self.last_episode = None
        self._closed = False
        self._close_error = None

    @classmethod
    def from_native(cls, sim, origins, calibration, snapshot, *, width=640, height=480, horizontal_fov_degrees=75.0, device='cuda:0', render_setting=None, max_render_attempts=4, max_readback_bytes=1024 ** 3):
        try:
            _tiled_stere_captu_import__TiledNativeStereoReader = TiledNativeStereoReader
            if not callable(snapshot):
                raise ValueError('physical snapshot callback required')
            if type(max_render_attempts) is not int or not 1 <= max_render_attempts <= 16:
                raise ValueError('bounded per-view render attempts required')
            cal = tiled_stere_captu__rerender_calibration(calibration, width, height, horizontal_fov_degrees=horizontal_fov_degrees)
            setting = tiled_stere_captu___NativeRenderSetting() if render_setting is None else render_setting
            if any((not callable(getattr(setting, n, None)) for n in ('get', 'set'))) or type(setting.get()) is not bool:
                raise ValueError('readable boolean render setting required')
            reader = _tiled_stere_captu_import__TiledNativeStereoReader(sim, origins, cal, device, max_readback_bytes=max_readback_bytes)
            try:
                return cls(sim, reader, cal, snapshot, render_setting=setting, origins=origins, max_render_attempts=max_render_attempts)
            except BaseException:
                reader.close()
                raise
        finally:
            sim = reader = snapshot = render_setting = setting = None

    def _physical(self):
        before = tiled_stere_captu__copy.deepcopy(self.snapshot())
        rows = before if isinstance(before, list) else [before]
        if not rows or any((not isinstance(r, dict) or set(r) != tiled_stere_captu___IDENTITY_KEYS | tiled_stere_captu___RAW_KEYS for r in rows)):
            raise ValueError('physical identity fields missing or unexpected')
        step = self.sim.current_time_step_index
        stamp = float(self.sim.current_time)
        if type(step) is not int or step < 0 or (not tiled_stere_captu__math.isfinite(stamp)) or (stamp < 0):
            raise ValueError('invalid physical SDK clock')
        tiled_stere_captu__validate_identity(rows, len(rows), stamp)
        for row in rows:
            if any((type(row[k]) is not int or row[k] < 0 for k in ('sdk_step', 'rod_step', 'rod_tick'))) or any((isinstance(row[k], bool) or not isinstance(row[k], (int, float)) or (not tiled_stere_captu__math.isfinite(row[k])) or (row[k] < 0) for k in ('sdk_timestamp_s', 'solver_time_s'))) or row['sdk_step'] != step or (abs(row['sdk_timestamp_s'] - stamp) > 1e-06) or (row['calibration_sha256'] != self.calibration.sha256) or (not isinstance(row['state_sha256'], str)) or (not tiled_stere_captu__re.fullmatch('[0-9a-f]{64}', row['state_sha256'])) or any((not isinstance(row[k], str) or not row[k] for k in ('rod_uid', 'material_reference_uid'))):
                raise ValueError('invalid physical/rod state binding')
        return (before, rows, step, stamp)

    def capture(self):
        if self._closed:
            raise RuntimeError('tiled capture is closed')
        original = self.render_setting.get()
        if type(original) is not bool:
            raise ValueError('readable boolean render setting required')
        try:
            self.render_setting.set(False)
            if self.render_setting.get() is not False:
                raise RuntimeError('physics freeze setting failed')
            packet = self._capture()
        finally:
            self.render_setting.set(original)
            if self.render_setting.get() is not original:
                raise RuntimeError('render setting restore failed')
        timing = packet['render_timing']
        if type(self.sim.current_time_step_index) is not int or self.sim.current_time_step_index != timing['physical_step'] or float(self.sim.current_time) != timing['physical_timestamp_s'] or (not tiled_stere_captu___same_snapshot(self.snapshot(), packet['identity'])):
            raise RuntimeError('physical state changed during camera restoration')
        rows = packet['identity'] if isinstance(packet['identity'], list) else [packet['identity']]
        self.last_capture_id = timing['capture_id']
        self.last_epoch = max(timing['raw_left_clock']['render_frame_number'], timing['raw_right_clock']['render_frame_number'])
        self.last_episode = tuple(((r['run_uuid'], r['env_id'], r['episode_uid'], r['reset_id']) for r in rows))
        return packet

    def _capture(self):
        (before, rows, step, stamp) = self._physical()
        episode = tuple(((r['run_uuid'], r['env_id'], r['episode_uid'], r['reset_id']) for r in rows))
        previous = self.last_epoch if episode == self.last_episode else None

        def check():
            if type(self.sim.current_time_step_index) is not int or self.sim.current_time_step_index != step or float(self.sim.current_time) != stamp or (not tiled_stere_captu___same_snapshot(self.snapshot(), before)):
                raise RuntimeError('physical state changed during tiled stereo capture')
        render_calls = 0
        frames = None
        for _ in range(self.max_render_attempts):
            prior = self.reader.read_clock()
            check()
            self.sim.render()
            render_calls += 1
            self.render_setting.set(False)
            check()
            current = self.reader.read_clock()
            check()
            if prior is None or current is None:
                continue
            (prior, current) = (_clock(prior), _clock(current))
            if current['render_frame_number'] <= prior['render_frame_number']:
                continue
            if previous is not None and current['render_frame_number'] <= previous:
                continue
            pair = self.reader.read_pair()
            check()
            if pair is None:
                continue
            if not isinstance(pair, dict) or set(pair) != {'left', 'right'}:
                raise ValueError('reader must return both fixed cameras from one render')
            candidate = {}
            for side in ('left', 'right'):
                packet = pair[side]
                if not isinstance(packet, dict) or not set(packet) <= _FRAME_KEYS or (not _FRAME_KEYS - {'raw_time_metadata'} <= set(packet)):
                    raise ValueError('reader returned forbidden or missing fields')
                raw = _clock(packet['raw_clock'])
                if raw != current:
                    raise ValueError('fixed camera returned a stale render product')
                if _clock(self.reader.read_clock())['render_frame_number'] != current['render_frame_number']:
                    raise ValueError('raw render clock changed during paired image read')
                if type(packet['native_evidence']) is not bool:
                    raise ValueError('explicit reader source required')
                candidate[side] = {**packet, 'rgb': packet['rgb'].clone(), 'K': tiled_stere_captu__torch.as_tensor(packet['K']).clone(), 'T_world': tiled_stere_captu__torch.as_tensor(packet['T_world']).clone()}
            frames = candidate
            break
        if frames is None:
            raise RuntimeError('fresh fixed-camera pair unavailable within bounded renders')
        (left, right) = (frames['left'], frames['right'])
        tiled_stere_captu__validate_images(left['rgb'], right['rgb'], self.calibration)
        if len(rows) != left['rgb'].shape[0]:
            raise ValueError('identity/camera batch mismatch')
        self.calibration.validate_readback(left['K'], right['K'], left['T_world'], right['T_world'], origins=self.origins)
        check()
        capture_id = self.last_capture_id + 1
        timing = dict(capture_id=capture_id, raw_left_clock=left['raw_clock'], raw_right_clock=right['raw_clock'], camera_pose_backend=getattr(self.reader, 'pose_backend', 'unspecified'), raw_left_metadata=str(left.get('raw_time_metadata', 'unavailable'))[:4096], raw_right_metadata=str(right.get('raw_time_metadata', 'unavailable'))[:4096], physical_timestamp_s=stamp, physical_step=step, clock_mapping='single_render_frozen_physics_boundary', native_synchronization_qualified=False)
        return dict(left_rgb=left['rgb'], right_rgb=right['rgb'], identity=before, calibration_sha256=self.calibration.sha256, K_left=left['K'], K_right=right['K'], T_world_left=left['T_world'], T_world_right=right['T_world'], render_timing=timing, render_passes=render_calls, source_scope='native_tiled_reader' if left['native_evidence'] and right['native_evidence'] else 'cpu_fixture')

    def close(self):
        if self._closed:
            if self._close_error:
                raise RuntimeError(self._close_error)
            return
        self._closed = True
        try:
            self.reader.close()
        except BaseException as error:
            self._close_error = str(error)
            raise
        finally:
            self.sim = self.reader = self.snapshot = self.render_setting = self.origins = None
