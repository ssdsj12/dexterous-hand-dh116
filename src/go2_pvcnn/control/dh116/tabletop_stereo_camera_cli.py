"""Explicit stereo camera probes and offline depth-reference measurement."""
import argparse
import json
from pathlib import Path

import torch

from .tabletop_stereo_calibration import StereoCalibration
from .tabletop_stereo_contract import ROOT, file_sha256, new_run, require_opt_in, write_json_new


def depth_metrics(prediction, reference, valid, reference_valid):
    if (prediction.shape != reference.shape or valid.shape != prediction.shape
            or reference_valid.shape != prediction.shape or valid.dtype != torch.bool
            or reference_valid.dtype != torch.bool):
        raise ValueError('aligned stereo/reference depth and explicit masks required')
    mask = valid & reference_valid & torch.isfinite(prediction) & torch.isfinite(reference)
    if not mask.any():
        return dict(count=0, mae_m=None, rmse_m=None, p95_abs_m=None, validity_coverage=0.)
    error = (prediction[mask]-reference[mask]).float()
    return dict(count=int(mask.sum()), mae_m=float(error.abs().mean()),
        rmse_m=float(error.square().mean().sqrt()), p95_abs_m=float(error.abs().quantile(.95)),
        validity_coverage=float(mask.float().mean()))


def evaluate_capture(capture_path, reference_path):
    captured = torch.load(capture_path, map_location='cpu', weights_only=False)
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    if (captured.get('version') != 'dh116-stereo-rgb-probe-v1'
            or reference.get('version') != 'dh116-stereo-depth-reference-v1'):
        raise ValueError('versioned stereo probe and independent depth reference required')
    fields = ('calibration_sha256', 'frame_id', 'simulation_time')
    packet = captured['frames'][0]
    if any(reference.get(key) != packet.get(key) for key in fields):
        raise ValueError('depth reference calibration/frame/timestamp mismatch')
    if packet.get('depth_source') != 'computed_from_left_right_RGB':
        raise ValueError('rendered depth cannot be reported as stereo prediction')
    metrics = depth_metrics(packet['depth_m'], reference['depth_m'],
        packet['valid'], reference['valid'])
    return dict(version='dh116-stereo-depth-evaluation-v1', metrics=metrics,
        capture_sha256=file_sha256(capture_path), reference_sha256=file_sha256(reference_path),
        evidence_scope='offline_reference_comparison', native_calibration_qualification=False,
        reference_used_as_actor_input=False)


def main(mode='probe', argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/dh116_training/stereo_camera_v1.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=116)
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--frames', type=int, default=2)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--reference-depth', type=Path)
    parser.add_argument('--native-task-config', type=Path, default=ROOT / 'configs/dh116_training/stereo_native_task_v1.json')
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--enable-experiment', action='store_true')
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args(argv)
    calibration = StereoCalibration.from_json(args.config)
    if args.num_envs < 1 or args.frames < 1:
        parser.error('positive num-envs and frame count required')
    if args.check_config:
        print(json.dumps(dict(validated=True, native_started=False, calibration_sha256=calibration.sha256,
            width=calibration.width, height=calibration.height, baseline_m=calibration.baseline_m)))
        return 0
    require_opt_in(args.enable_experiment)
    if args.output is None:
        parser.error('new --output required')
    if mode == 'evaluate':
        if args.capture is None or args.reference_depth is None:
            parser.error('--capture and --reference-depth required; no simulator depth fallback')
        report = evaluate_capture(args.capture, args.reference_depth)
        run = new_run(args.output)
        write_json_new(run / 'depth_evaluation.json', report)
        return 0
    if not args.execute_native:
        parser.error('camera acquisition requires explicit --execute-native via scripts/python.sh')
    run = new_run(args.output)
    from .tabletop_stereo_native import launch_native
    from .tabletop_stereo_camera import StereoCamera

    def callback(bridge):
        camera = StereoCamera.from_native(bridge.world.sim, bridge.device_world.origins,
            args.config, device=bridge.device)
        frames = []
        for _ in range(args.frames):
            pair = camera.sample(identity=bridge.pre_action_identities(camera.calibration.sha256))
            pair = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in pair.items()}
            pair['depth_source'] = 'computed_from_left_right_RGB'
            pair['actual_palm_fk_pose_wxyz'] = bridge.palm_poses().cpu()
            frames.append(pair)
        with (run / 'stereo_probe.pt').open('xb') as stream:
            torch.save(dict(version='dh116-stereo-rgb-probe-v1', frames=frames), stream)
        report = dict(status='bounded_stereo_probe_completed', frames=len(frames),
            depth_source='computed_from_left_right_RGB', render_advances_physics=False,
            native_geometry_accuracy_qualification=False, grasp_success_claimed=False,
            mean_valid_fraction=[float(f['valid'].float().mean()) for f in frames],
            limitation='SAD integer-disparity baseline; native calibration/depth accuracy require reference tests')
        write_json_new(run / 'stereo_probe.json', report)
        return report

    launch_native(callback, native_arguments=args.native_task_config, output=run / 'native',
        num_envs=args.num_envs, seed=args.seed, enabled=True)
    return 0
