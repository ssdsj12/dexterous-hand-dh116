"""Standalone RGB-pair -> stereo depth -> sensor history -> Student action.

No Teacher, critic, task, simulator state or rendered depth is accepted. Robot
drivers remain responsible for validated encoder/FK input and the unchanged
24D controller/safety contract; this module does not issue hardware commands.
"""
import argparse
import json
from pathlib import Path

import torch

from .tabletop_stereo_calibration import StereoCalibration
from .tabletop_stereo_checkpoint import load_deployment_actor
from .tabletop_stereo_depth import StereoDepth
from .tabletop_stereo_observation import SensorBatch, StereoHistory
from .tabletop_stereo_perception import WirePointExtractor
from .tabletop_stereo_geometry import world_to_palms
from .tabletop_stereo_contract import ROOT, file_sha256, require_opt_in, write_json_new
from .tabletop_stereo_preprocessing import require_raw_processing


RAW_FIELDS = {'left_rgb', 'right_rgb', 'proprio', 'left_timestamp_s', 'right_timestamp_s',
              'frame_id', 'now_timestamp_s'}


class StereoRGBStudentRunner:
    def __init__(self, actor_package, camera_config, *, num_envs=1, device='cpu', max_image_age_s=1/60):
        self.calibration = StereoCalibration.from_json(camera_config)
        self.actor = load_deployment_actor(actor_package,
            expected_calibration_sha256=self.calibration.sha256).to(device).eval()
        self.processing = require_raw_processing(self.actor.deployment_contract, camera_config)
        self.camera_config_sha256 = file_sha256(camera_config)
        config = json.loads(Path(camera_config).read_text(encoding='utf-8'))
        self.depth = StereoDepth.from_config(config)
        self.history = StereoHistory(num_envs, self.actor.network_config['history'], device=device)
        self.extractor = WirePointExtractor()
        self.num_envs, self.device = num_envs, torch.device(device)
        if max_image_age_s <= 0:
            raise ValueError('positive image freshness limit required')
        self.max_image_age_s = float(max_image_age_s)
        self._frame, self._timestamp = -1, None
        self.last_visible_geometry = None
        self.package_sha256 = file_sha256(actor_package)

    @torch.no_grad()
    def step(self, left_rgb, right_rgb, proprio, *, left_timestamp_s, right_timestamp_s,
             frame_id, now_timestamp_s, reset_mask=None):
        import math
        if (type(frame_id) is not int or frame_id <= self._frame
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in
                    (left_timestamp_s, right_timestamp_s, now_timestamp_s))):
            raise ValueError('fresh frame ID and finite camera timestamps required')
        if abs(left_timestamp_s-right_timestamp_s) > 1e-9:
            raise ValueError('left/right RGB must represent the same physical time')
        age = now_timestamp_s-left_timestamp_s
        if age < 0 or age > self.max_image_age_s:
            raise ValueError('stale/future RGB pair; no truth fallback or silent timing change')
        dt = 1/60 if self._timestamp is None else left_timestamp_s-self._timestamp
        if dt <= 0:
            raise ValueError('camera simulation/physical timestamp must increase')
        def rgb(value):
            if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.uint8):
                raise ValueError('RGB must be declared CHW uint8 RGB or float32 RGB; no BGR guessing')
            return value.to(self.device).float()/255. if value.dtype == torch.uint8 else value.to(self.device)
        left, right = rgb(left_rgb), rgb(right_rgb)
        p = proprio.to(self.device)
        if len(left) != self.num_envs:
            raise ValueError('configured batch size mismatch')
        if reset_mask is None:
            reset_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            reset_mask = reset_mask.to(self.device)
        depth = self.depth.compute(left, right, self.calibration)
        points = self.extractor.extract(left, depth['depth_m'], depth['valid'],
            self.calibration.K_left, self.calibration.T_world_left, reset_mask=reset_mask)
        sensor = SensorBatch(left_rgb=left, right_rgb=right, depth_m=depth['depth_m'],
            depth_valid=depth['valid'], depth_confidence=depth['confidence'],
            points_world=points['points_world'], point_valid=points['point_valid'], proprio=p,
            dt=torch.full((self.num_envs,), dt, device=self.device, dtype=torch.float32),
            image_age=torch.full((self.num_envs,), age, device=self.device, dtype=torch.float32),
            image_valid=torch.ones(self.num_envs, 2, device=self.device, dtype=torch.bool))
        history = self.history.update(sensor, reset_mask=reset_mask)
        actions = self.actor.deterministic(history)
        if actions.shape != (self.num_envs, 24) or not torch.isfinite(actions).all() or (actions.abs() > 1).any():
            raise RuntimeError('Actor violated canonical normalized 24D action bounds')
        # Derived sensor geometry for consumers/diagnostics; no simulator truth
        # or fixed reset pose, and no change to the Actor's world-frame input.
        self.last_visible_geometry = dict(points_world_m=points['points_world'].detach().clone(),
            points_palms_m=world_to_palms(points['points_world'], p[:,24:38].reshape(self.num_envs,2,7)).detach().clone(),
            point_valid=points['point_valid'].detach().clone(), frame_id=frame_id,
            simulation_timestamp=float(left_timestamp_s),
            source='RGB_stereo_and_current_measured_palm_FK', calibration_sha256=self.calibration.sha256)
        self._frame, self._timestamp = frame_id, float(left_timestamp_s)
        return dict(actions24=actions.detach().clone(), frame_id=frame_id,
            simulation_timestamp=float(left_timestamp_s), image_age_s=float(age),
            depth_valid_fraction=depth['valid'].float().flatten(1).mean(1),
            visible_wire_point_count=points['point_valid'].sum(1),
            actor_package_sha256=self.package_sha256, calibration_sha256=self.calibration.sha256,
            camera_config_sha256=self.camera_config_sha256,
            inference_source_sha256=file_sha256(__file__),
            processing_sha256=self.processing['processing_sha256'],
            teacher_executed=False, critic_executed=False, simulator_input_used=False,
            depth_source='computed_from_synchronized_left_right_RGB', native_task_qualified=False)

    def from_mapping(self, raw, *, reset_mask=None):
        if not isinstance(raw, dict) or set(raw) != RAW_FIELDS:
            raise ValueError('only RGB pair, measured proprioception and timing are accepted; privileged fields forbidden')
        return self.step(**raw, reset_mask=reset_mask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor-package', type=Path, required=True)
    parser.add_argument('--rgb-frame', type=Path, required=True)
    parser.add_argument('--camera-config', type=Path, default=ROOT / 'configs/dh116_training/stereo_camera_640x480_v2.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--enable-experiment', action='store_true')
    args = parser.parse_args(argv)
    require_opt_in(args.enable_experiment)
    raw = torch.load(args.rgb_frame, map_location='cpu', weights_only=True)
    if not isinstance(raw, dict) or set(raw) != RAW_FIELDS:
        raise ValueError('versioned raw RGB/proprio frame required; depth/privileged tensors are not accepted')
    runner = StereoRGBStudentRunner(args.actor_package, args.camera_config,
        num_envs=len(raw['left_rgb']), device=args.device)
    result = runner.from_mapping(raw)
    result = {k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
    write_json_new(args.output, result)
    return 0
