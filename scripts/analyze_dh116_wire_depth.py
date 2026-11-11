"""Replay bound native RGB captures with the runtime wire matcher and extractor."""
import argparse
import json
from pathlib import Path
import time

import torch

from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth
from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import _label_masks, load_capture
from go2_pvcnn.control.dh116.tabletop_stereo_wire_diagnostics import audit_wire_frame
from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import TrackedWirePointExtractor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    capture, references, calibration, config, proof = load_capture(args.proof)
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(scope='offline_replay_of_sha_bound_native_RGB_and_reference', calibration_sha256=calibration.sha256,
                  capture_sha256=proof['capture_artifact']['sha256'], proof_sha256=file_sha256(args.proof),
                  reference_sha256=[item['sha256'] for item in proof['reference_artifacts']],
                  reference_used_as_actor_input=False, frames=[])
    settings = {'baseline': config,
                'wire_census': dict(config, method='wire_census'),
                'wire_zncc': dict(config, method='wire_zncc'),
                'wire_silhouette': dict(config, method='wire_silhouette', subpixel=False)}
    trackers = {name: TrackedWirePointExtractor() for name in settings}
    ribbons = {'baseline_observed_ribbon': ('baseline', 1),
               'wire_silhouette_ribbon1': ('wire_silhouette', 1),
               'wire_silhouette_ribbon3': ('wire_silhouette', 3)}
    for name, (_, radius) in ribbons.items():
        trackers[name] = TrackedWirePointExtractor(point_sampling='observed_ribbon', ribbon_radius_px=radius)
    estimators = {name: StereoDepth(**options) for name, options in settings.items()}
    for index, (frame, reference) in enumerate(zip(capture['frames'], references)):
        record = dict(frame_id=frame['frame_id'], simulation_time=frame['simulation_time'], methods={})
        finger = _label_masks(reference, frame)['finger']
        predictions = {}
        for name, tracker in trackers.items():
            if name in ribbons:
                base, _ = ribbons[name]
                matcher = estimators[base]
                prediction, elapsed = predictions[base], None
            else:
                matcher = estimators[name]
                start = time.perf_counter()
                prediction = matcher.compute(frame['left_rgb'], frame['right_rgb'], calibration)
                elapsed = time.perf_counter() - start
                if any(base == name for base, _ in ribbons.values()):
                    predictions[name] = prediction
            extracted = tracker.extract(frame['left_rgb'], prediction['depth_m'], prediction['valid'],
                frame.get('K_left', calibration.K_left), frame.get('T_world_left', calibration.T_world_left),
                timestamp=[row['simulation_timestamp'] for row in frame['identity']],
                point_confidence=prediction['confidence'])
            maps, rows = audit_wire_frame(frame, prediction, extracted, calibration, reference,
                                          block_size=matcher.block_size, near=matcher.near, far=matcher.far, finger=finger)
            maps.update(stereo_depth_m=prediction['depth_m'], disparity_px=prediction['disparity_px'],
                        tracker_point_pixels=extracted['point_pixels'], tracker_point_valid=extracted['point_valid'],
                        tracker_point_provenance=extracted['point_provenance'],
                        tracker_xyz_m=extracted['points_world'], rejection_flags=prediction['diagnostics']['reasons'])
            if 'optional_refinement_used' in prediction['diagnostics']:
                maps['optional_refinement_used'] = prediction['diagnostics']['optional_refinement_used']
            filename = f'frame_{index:04d}_{name}.pt'
            with (args.output / filename).open('xb') as stream:
                torch.save(maps, stream)
            record['methods'][name] = dict(elapsed_s=elapsed, pixel_maps_file=filename,
                pixel_maps_sha256=file_sha256(args.output / filename), environments=rows,
                saved_depth_validity_mismatch=[int((value != old).sum()) for value, old in
                    zip(prediction['valid'], frame['valid'])] if name == 'baseline' else None)
        result['frames'].append(record)
        print(f'completed frame {index + 1}/{len(references)}', flush=True)
    result['settings'] = {name: estimator.to_config() for name, estimator in estimators.items()}
    for name, (base, radius) in ribbons.items():
        result['settings'][name] = dict(depth_reused_from=base, point_sampling='observed_ribbon', ribbon_radius_px=radius)
    with (args.output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
