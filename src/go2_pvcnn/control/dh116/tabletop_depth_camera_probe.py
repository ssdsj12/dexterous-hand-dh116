"""One-shot, two-camera RGB/depth coordinate capture in an existing Isaac process.

No SimulationApp, physics steps, policy update, or segmentation ground truth is
created here. Camera optical coordinates are +X right, +Y down, +Z forward.
The measured local point cloud does not replace the existing 371D observation.
"""

import json
import importlib
from pathlib import Path

import numpy as np

from .tabletop_depth_hand_points import depth_to_hand_points


def _load_camera_class():
    # IsaacLab's headless rendering experience need not enable this sensor API.
    # Enabling its installed extension registers the Python namespace first.
    kit = importlib.import_module('omni.kit.app')
    kit.get_app().get_extension_manager().set_extension_enabled_immediate(
        'isaacsim.sensors.camera', True)
    return importlib.import_module('isaacsim.sensors.camera').Camera


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rotation(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0.:
        raise ValueError('quaternion must be finite nonzero wxyz')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def look_at_optical(eye, target, up):
    """Return a column-vector optical-camera-to-world rigid transform."""
    eye, target, up = [np.asarray(v, dtype=float) for v in (eye, target, up)]
    forward = target-eye
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (eye, target, up)):
        raise ValueError('camera eye, target and up must be finite xyz')
    if np.linalg.norm(forward) < 1e-10:
        raise ValueError('camera eye and target must differ')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-10:
        raise ValueError('camera up must not be parallel to the optical axis')
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((right, down, forward))
    result[:3, 3] = eye
    return result


def usd_pose_to_optical(position, quaternion):
    """USD camera +Y up/-Z forward becomes optical +Y down/+Z forward."""
    result = np.eye(4)
    result[:3, :3] = _rotation(_numpy(quaternion)) @ np.diag([1., -1., -1.])
    result[:3, 3] = _numpy(position)
    return result


def brown_rod_mask(rgb):
    """Visible-color-only baseline; contains no pose, cylinder or segmentation GT."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError('rgb must be [H,W,3+]')
    color = image[..., :3].astype(np.float64)
    if np.issubdtype(image.dtype, np.integer) or (color.size and np.nanmax(color) > 1.):
        color /= 255.
    r, g, b = np.moveaxis(color, -1, 0)
    maximum, minimum = color.max(-1), color.min(-1)
    saturation = (maximum-minimum)/np.maximum(maximum, 1e-12)
    return (np.isfinite(color).all(-1) & (r > 1.2*g) & (g > 1.2*b)
            & (maximum > 35./255.) & (saturation > .25))


def cylinder_surface_distances(points_world, rod_pose, radius, length):
    """Absolute distance to finite local-Y cylinder; validation only, never mask."""
    pose = np.asarray(rod_pose, dtype=float)
    local = (np.asarray(points_world, dtype=float)-pose[:3]) @ _rotation(pose[3:])
    radial = np.linalg.norm(local[:, [0, 2]], axis=1)-float(radius)
    axial = np.abs(local[:, 1])-float(length)/2.
    signed = np.hypot(np.maximum(radial, 0.), np.maximum(axial, 0.))
    signed += np.minimum(np.maximum(radial, axial), 0.)
    return np.abs(signed)


def save_frame_artifacts(output, name, rgb, depth, intrinsic, transform, hand_poses,
                         rod_pose, rod_radius, rod_length, *, min_depth_m, max_depth_m):
    """Save reproducible optical/hand point arrays and a visible RGB/depth preview."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rgb, depth = np.asarray(rgb)[..., :3], np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if rgb.shape[:2] != depth.shape:
        raise RuntimeError(f'RGB/depth shape mismatch: {rgb.shape}, {depth.shape}')
    mask = brown_rod_mask(rgb)
    kwargs = dict(min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    all_points = depth_to_hand_points(depth, intrinsic, transform, hand_poses, **kwargs)
    selected = depth_to_hand_points(depth, intrinsic, transform, hand_poses, object_mask=mask, **kwargs)
    if not len(all_points['points_world']):
        raise RuntimeError(f'{name}: no finite depth within clipping range')
    if not len(selected['points_world']):
        raise RuntimeError(f'{name}: visible RGB brown rod mask has no valid depth points')
    roundtrip = []
    for hand in range(2):
        reconstructed = (selected['points_in_hands'][hand] @ _rotation(hand_poses[hand, 3:]).T
                         + hand_poses[hand, :3])
        roundtrip.append(np.linalg.norm(reconstructed-selected['points_world'], axis=1).max())
    camera_roundtrip = (selected['points_world']-transform[:3, 3]) @ transform[:3, :3]
    camera_error = float(np.linalg.norm(camera_roundtrip-selected['points_camera'], axis=1).max())
    if max(roundtrip) > 1e-6 or camera_error > 1e-6:
        raise RuntimeError('camera/hand coordinate roundtrip exceeded one micrometre')
    # Cylinder GT is consulted only after the color/depth points are finalized.
    distances = cylinder_surface_distances(selected['points_world'], rod_pose, rod_radius, rod_length)
    archive = output/f'{name}_depth_coordinates.npz'
    np.savez_compressed(archive, rgb=rgb, depth=depth, K=intrinsic,
        T_world_camera=transform, hand_poses=hand_poses, object_mask_rgb=mask,
        **selected, all_points_world=all_points['points_world'],
        all_points_camera=all_points['points_camera'], all_pixels_uv=all_points['pixels_uv'],
        all_valid_mask=all_points['valid_mask'])
    sample_id = len(selected['points_world'])//2
    sample = dict(pixel_uv=selected['pixels_uv'][sample_id].tolist(),
        camera_xyz_m=selected['points_camera'][sample_id].tolist(),
        world_xyz_m=selected['points_world'][sample_id].tolist(),
        left_hand_xyz_m=selected['points_in_hands'][0, sample_id].tolist(),
        right_hand_xyz_m=selected['points_in_hands'][1, sample_id].tolist())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(rgb); axes[0].set_title(f'{name}: rendered RGB')
    display_depth = np.ma.masked_where(~all_points['valid_mask'], depth)
    view = axes[1].imshow(display_depth, cmap='viridis', vmin=min_depth_m, vmax=max_depth_m)
    figure.colorbar(view, ax=axes[1], label='Optical Z (m)', fraction=.046)
    axes[1].set_title('Rendered depth, optical Z')
    axes[2].imshow(rgb)
    overlay = np.zeros((*mask.shape, 4))
    overlay[selected['valid_mask']] = [0., 1., .2, .7]
    axes[2].imshow(overlay); axes[2].set_title('Visible RGB brown mask + valid depth')
    for axis in axes:
        axis.scatter(*sample['pixel_uv'], c='magenta', s=45, marker='x', linewidths=2)
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle('Measured visible points only; no segmentation GT; no cylinder roll estimate')
    figure.tight_layout()
    preview = output/f'{name}_rgb_depth.png'
    figure.savefig(preview, dpi=140)
    plt.close(figure)
    return dict(name=name, archive=str(archive), preview=str(preview),
        valid_pixels=int(all_points['valid_mask'].sum()), object_mask_rgb_pixels=int(mask.sum()),
        object_mask_points=int(len(selected['points_world'])),
        camera_world_roundtrip_max_m=camera_error, hand_world_roundtrip_max_m=float(max(roundtrip)),
        sample_point=sample,
        hand_poses_xyz_wxyz=np.asarray(hand_poses).tolist(), T_world_camera=np.asarray(transform).tolist(),
        K=np.asarray(intrinsic).tolist(), depth_range_m=[float(depth[all_points['valid_mask']].min()),
                                                     float(depth[all_points['valid_mask']].max())],
        gt_cylinder_surface_validation_only=dict(median_m=float(np.median(distances)),
            p95_m=float(np.quantile(distances, .95)), maximum_m=float(distances.max()),
            fraction_within_2mm=float(np.mean(distances < .002))))


def capture_depth_coordinates(world, device_world, task, args):
    """Capture initial pinch/tripod scenes, after mixed reset, before PPO creation.

    The caller must already own the sole Isaac process and enable cameras in its
    AppLauncher. Exactly two environments are required to bound render memory.
    """
    if device_world.n != 2:
        raise ValueError('depth coordinate probe requires exactly two environments')
    config_path = Path(args.depth_camera_config)
    config = json.loads(config_path.read_text())
    width, height = map(int, config['resolution_wh'])
    if not 1 <= width <= 640 or not 1 <= height <= 480:
        raise ValueError('depth coordinate probe resolution must not exceed 640x480')
    near, far = map(float, config['clipping_range_m'])
    fov = float(config['horizontal_fov_degrees'])
    if not (0. < near < far and 1. < fov < 179.):
        raise ValueError('invalid camera clipping range or FOV')
    if config.get('depth_convention') != 'optical_axis_z_m':
        raise ValueError('camera config must specify optical_axis_z_m')
    # Imported only inside the already-running AppLauncher process.
    import omni.usd
    Camera = _load_camera_class()
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    parent = '/World/DH116DepthCoordinateProbe'
    if stage.GetPrimAtPath(parent).IsValid():
        raise RuntimeError('depth camera probe already exists; refusing duplicate render products')
    UsdGeom.Xform.Define(stage, parent)
    light = UsdLux.DomeLight.Define(stage, parent+'/DomeLight')
    light.CreateIntensityAttr(1800.)
    light.CreateColorAttr(Gf.Vec3f(1., 1., 1.))
    origins = _numpy(device_world.origins).copy()
    cameras, expected = [], []
    for env_id, name in enumerate(('pinch', 'tripod')):
        transform = look_at_optical(np.asarray(config['position_world_m'])+origins[env_id],
            np.asarray(config['target_world_m'])+origins[env_id], config['world_up'])
        camera = Camera(prim_path=f'{parent}/{name}', name=f'dh116_probe_{name}',
                        resolution=(width, height), frequency=-1)
        camera.initialize()
        camera.set_focal_length(2.4)
        camera.set_horizontal_aperture(2.*2.4*np.tan(np.deg2rad(fov)/2.))
        camera.set_clipping_range(near_distance=near, far_distance=far)
        camera.add_distance_to_image_plane_to_frame()
        # USD uses row-vector matrices; convert the optical basis to USD first.
        usd_transform = transform @ np.diag([1., -1., -1., 1.])
        xform = UsdGeom.Xformable(camera.prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d(*usd_transform.T.reshape(-1).tolist()))
        cameras.append(camera); expected.append(transform)

    # Render only: SimulationContext.render syncs Fabric/Hydra without physics.
    before_pose = _numpy(device_world.roots).copy()
    before_time = float(world.sim.current_time)
    ready = False
    for frame in range(24):
        world.sim.render()
        if frame < 7:
            continue
        ready = all(camera.get_depth() is not None and
            _numpy(camera.get_depth()).size == width*height and
            _numpy(camera.get_rgb()).shape[:2] == (height, width) and
            np.any(np.isfinite(_numpy(camera.get_depth())) & (_numpy(camera.get_depth()) > near) &
                   (_numpy(camera.get_depth()) < far)) for camera in cameras)
        if ready:
            break
    if not ready:
        raise RuntimeError('RGB/depth annotators did not produce valid frames after 24 render-only warmup frames')
    device_world.refresh(sensors=False)
    hand_poses = _numpy(device_world.roots).copy()
    if abs(float(world.sim.current_time)-before_time) > 1e-9 or not np.allclose(hand_poses, before_pose, atol=1e-6, rtol=0.):
        raise RuntimeError('render-only probe unexpectedly advanced physics or changed actual hand poses')
    rod_poses = np.concatenate((_numpy(device_world.wp), _numpy(device_world.wq)), axis=1)
    from .tabletop_end_supported_scene import wire_geometry_from_spawn
    rod_length, rod_radius = wire_geometry_from_spawn(world.source['wire'].cfg.spawn)
    output = Path(args.output)/'depth_camera_coordinates'
    groups = []
    for env_id, (name, camera, expected_transform) in enumerate(zip(('pinch', 'tripod'), cameras, expected)):
        # Check actual Camera readback and independent USD transform after sync.
        position, orientation = camera.get_world_pose(camera_axes='usd')
        transform = usd_pose_to_optical(position, orientation)
        usd_matrix = np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(camera.prim)).T
        usd_optical = usd_matrix @ np.diag([1., -1., -1., 1.])
        authored_error = float(np.max(np.abs(transform-expected_transform)))
        usd_error = float(np.max(np.abs(transform-usd_optical)))
        if max(authored_error, usd_error) > 1e-5:
            raise RuntimeError(f'{name}: camera extrinsic mismatch authored={authored_error}, usd={usd_error}')
        rgb, depth = _numpy(camera.get_rgb()).copy(), _numpy(camera.get_depth()).copy()
        intrinsic = _numpy(camera.get_intrinsics_matrix())
        group = save_frame_artifacts(output, name, rgb, depth, intrinsic, transform,
            hand_poses[env_id], rod_poses[env_id], rod_radius, rod_length,
            min_depth_m=near, max_depth_m=far)
        group.update(env_id=env_id, env_origin_m=origins[env_id].tolist(),
            camera_authored_extrinsic_max_error=authored_error, camera_usd_extrinsic_max_error=usd_error,
            rendering_frame=int(camera.get_current_frame().get('rendering_frame', 0)),
            rendering_time_s=float(camera.get_current_frame().get('rendering_time', 0.)))
        groups.append(group)
    summary = dict(status='depth_coordinates_captured', optimization_performed=False,
        simulation_advanced=False, camera_count=2, resolution_wh=[width,height],
        warmup_render_frames=frame+1, simulation_time_s=before_time, config=str(config_path),
        camera_convention='x_right_y_down_z_forward; optical_axis_z_m',
        hand_pose_source='device_world.roots actual palm/base_link; xyz+wxyz',
        hand_pose_measurement='Simulation link-pose readback stands in for real encoder/FK wrist pose.',
        student_artifact_excludes='Object GT pose, cylinder geometry/distance GT, and contact/force GT.',
        mask_source='RGB only: r>1.2g, g>1.2b, max>35/255, saturation>.25',
        observations_371d_modified=False, cylinder_roll_estimated=False,
        limitations=['Visible surface only; RGB color mask can contain false positives.',
            'Ground-truth cylinder distances are validation only and never select points.',
            'Hand-local points do not replace object-centred world-axis 371D reference fields.'],
        groups=groups)
    summary_path = output/'summary.json'
    summary['summary_path'] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary
