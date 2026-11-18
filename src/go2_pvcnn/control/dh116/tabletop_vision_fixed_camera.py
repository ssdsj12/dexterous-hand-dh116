"""One fixed real RGB-D camera per environment, sharing one tiled product.

The scene stays visible and camera poses are authored once. Every observation
renders the whole batch together; pixel postprocessing retains the existing
64-environment memory cap and sensor-only numerical contract.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .tabletop_vision_camera import _NativeTiledReader, _numpy, frames_to_points, numeric_slot_order


class FixedVisionCamera:
    """Fixed cameras for even N=2..2048; never move cameras or hide environments.

    The existing native reader checks the actual USD/CameraView calibration on
    its first read and caches that result. No per-observation calibration reset
    is necessary because this provider never changes the cameras.
    """

    def __init__(self, world, device_world, config_path, width=320, height=240, num_points=128,
                 *, debug_output_dir=None, debug_env_ids=()):
        self._sim, self._state = world.sim, device_world
        self.n = int(device_world.n)
        self.device = torch.device(device_world.device)
        if not 2 <= self.n <= 2048 or self.n % 2:
            raise ValueError('fixed cameras require an even environment count in [2,2048]')
        if not (type(width) is int and type(height) is int and 1 <= width <= 320 and 1 <= height <= 240):
            raise ValueError('fixed camera resolution must not exceed 320x240')
        if type(num_points) is not int or not 1 <= num_points <= 1024:
            raise ValueError('num_points must be an integer in [1,1024]')
        debug_ids = tuple(debug_env_ids)
        if (len(debug_ids) > 2 or len(set(debug_ids)) != len(debug_ids)
                or any(type(i) is not int or not 0 <= i < self.n for i in debug_ids)
                or (debug_ids and debug_output_dir is None)):
            raise ValueError('debug requires an output directory and at most two unique valid environment IDs')
        self._debug_ids = debug_ids
        self._debug_dir = Path(debug_output_dir) if debug_output_dir is not None else None
        self._debug_saved_files = []
        config = json.loads(Path(config_path).read_text())
        self.near, self.far = map(float, config['clipping_range_m'])
        fov = float(config['horizontal_fov_degrees'])
        if not (np.isfinite([self.near, self.far, fov]).all() and 0. < self.near < self.far
                and 1. < fov < 179. and config.get('depth_convention') == 'optical_axis_z_m'):
            raise ValueError('finite optical-Z pinhole camera configuration required')
        self.width, self.height, self.num_points = width, height, num_points
        origins = torch.as_tensor(device_world.origins, device=self.device)
        if origins.shape != (self.n, 3) or not torch.isfinite(origins).all():
            raise ValueError('one finite origin per physical environment required')
        self._reader = _NativeTiledReader(sim=self._sim, origins=origins, config=config,
            width=width, height=height, device=self.device)
        self.slot_order = numeric_slot_order(self._reader.prim_paths, self.n).to(self.device)
        self.K = torch.as_tensor(self._reader.K, device=self.device, dtype=torch.float32)[self.slot_order]
        self.T_world_camera = torch.as_tensor(self._reader.T_world_camera,
            device=self.device, dtype=torch.float32)[self.slot_order]
        self._last_frame = -1
        self._samples = 0
        self._raw_frame_sink = None

    def set_raw_frame_sink(self, sink):
        """Opt-in evidence tap; owned sensor copies, no extra render or step."""
        if sink is not None and not callable(sink):
            raise ValueError('raw frame sink must be callable or None')
        self._raw_frame_sink = sink

    def _check_frozen(self, expected_time, expected_step):
        if abs(float(self._sim.current_time) - expected_time) > 1e-9:
            raise RuntimeError('physics time changed during fixed camera observation')
        if expected_step is not None and self._sim.current_time_step_index != expected_step:
            raise RuntimeError('physics step index changed during fixed camera observation')

    def _check_fk(self, poses, hand_poses):
        current = self._state.roots if hand_poses is None else hand_poses
        if not torch.allclose(poses, torch.as_tensor(current, device=self.device,
                dtype=torch.float32), atol=1e-6, rtol=0.):
            raise RuntimeError('palm FK changed during fixed camera observation')

    def _read_fresh_frame(self, expected_time, expected_step):
        last_error = None
        # Each attempt renders before reading. A stale frame can never satisfy
        # the contract merely because the physics time happens to be unchanged.
        for attempt in range(8):
            self._sim.render()
            self._check_frozen(expected_time, expected_step)
            try:
                frame = self._reader.read()
            except (ValueError, IndexError) as error:
                last_error = repr(error)
                continue
            except RuntimeError as error:
                if not str(error).startswith(('camera frame changed while reading',
                        'render-product time metadata changed while reading')):
                    raise
                last_error = repr(error)
                continue
            if not isinstance(frame, dict) or not {'rgb', 'depth', 'frame_id', 'simulation_time'} <= frame.keys():
                last_error = 'same render-product timestamp or buffer unavailable'
                continue
            last_error = dict(frame=frame['frame_id'], previous=self._last_frame,
                rendered_time=frame['simulation_time'], physics_time=expected_time,
                isaac_read_times=frame.get('isaac_read_times'))
            if frame['frame_id'] <= self._last_frame:
                continue
            if not np.isfinite(frame['simulation_time']) or abs(frame['simulation_time'] - expected_time) > 1e-6:
                continue
            if (frame['rgb'].shape != (self.n, self.height, self.width, 3)
                    or frame['depth'].shape != (self.n, self.height, self.width, 1)):
                last_error = 'fixed RGB/depth buffer shape mismatch'
                continue
            return frame, attempt + 1
        raise RuntimeError(f'no fresh time-aligned fixed camera frame after 8 attempts; last={last_error}')

    @torch.no_grad()
    def sample(self, hand_poses=None):
        started = perf_counter()
        time_before = float(self._sim.current_time)
        step_before = getattr(self._sim, 'current_time_step_index', None)
        poses = torch.as_tensor(self._state.roots if hand_poses is None else hand_poses,
            device=self.device, dtype=torch.float32).clone()
        if poses.shape != (self.n, 2, 7) or not torch.isfinite(poses).all():
            raise ValueError('current palm FK must be finite [N,2,7]')
        # Seven warm-up renders plus the first read attempt make at least eight
        # initial renders. Up to eight read attempts allow time metadata to settle.
        warmup = 7 if self._samples == 0 else 0
        for _ in range(warmup):
            self._sim.render()
            self._check_frozen(time_before, step_before)
        frame, attempts = self._read_fresh_frame(time_before, step_before)
        self._check_fk(poses, hand_poses)
        # Copy/reorder native no-copy buffers once before the next render occurs.
        rgb = frame['rgb'].to(self.device)[self.slot_order]
        depth = frame['depth'].to(self.device)[self.slot_order]
        result = frames_to_points(rgb, depth, self.K, self.T_world_camera, poses,
            num_points=self.num_points, min_depth_m=self.near, max_depth_m=self.far)
        self._check_frozen(time_before, step_before)
        self._check_fk(poses, hand_poses)
        if self._raw_frame_sink is not None:
            self._raw_frame_sink(dict(rgb=rgb.detach().cpu().clone(),
                depth=depth.detach().cpu().clone(),K=self.K.detach().cpu().clone(),
                T_world_camera=self.T_world_camera.detach().cpu().clone(),
                hand_poses=poses.detach().cpu().clone(),frame_id=int(frame['frame_id']),
                simulation_time=time_before,rendered_simulation_time=float(frame['simulation_time']),
                physics_step=step_before))
            self._check_frozen(time_before, step_before)
            self._check_fk(poses, hand_poses)
        # Optional first-frame evidence only: real RGB-D/calibration plus the
        # current palm encoder/FK stand-in. No object pose, contact, or GT mask.
        if self._samples == 0 and self._debug_ids:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            for env_id in self._debug_ids:
                path = self._debug_dir / f'env_{env_id:05d}_rgbd.npz'
                np.savez(path, env_id=np.asarray(env_id),
                    rgb=_numpy(rgb[env_id]), depth=_numpy(depth[env_id]),
                    K=_numpy(self.K[env_id]), T_world_camera=_numpy(self.T_world_camera[env_id]),
                    hand_poses=_numpy(poses[env_id]),
                    hand_pose_source=np.asarray('current_palm_encoder_FK_simulation_standin'),
                    selected_pixels_uv=_numpy(result['selected_pixels_uv'][env_id]),
                    points_in_hands=_numpy(result['points'][env_id]),
                    point_valid=_numpy(result['point_valid'][env_id]),
                    image_valid=_numpy(result['image_valid'][env_id]),
                    object_valid=_numpy(result['object_valid'][env_id]),
                    frame_id=np.asarray(frame['frame_id']),
                    simulation_time=np.asarray(frame['simulation_time']))
                self._debug_saved_files.append(str(path))
        self._last_frame = int(frame['frame_id'])
        self._samples += 1
        rendered_time = float(frame['simulation_time'])
        result.update({key: value for key, value in frame.items() if key not in ('rgb', 'depth')})
        result.update(frame_id=self._last_frame, simulation_time=time_before,
            rendered_simulation_time=rendered_time, render_passes=warmup + attempts,
            camera_count=self.n, render_batches=1,
            camera_frame_first=self._last_frame, camera_frame_last=self._last_frame,
            frame_first=self._last_frame, frame_last=self._last_frame,
            batch_frame_ids=[self._last_frame], batch_rendered_simulation_times=[rendered_time],
            batch_environment_ranges=[[0, self.n]], batch_wall_seconds=[perf_counter() - started],
            physics_frozen=True, inactive_environment_visibility=False,
            pinned_source_environment=None, debug_saved_files=list(self._debug_saved_files),
            camera_mode='fixed_per_env')
        return result
