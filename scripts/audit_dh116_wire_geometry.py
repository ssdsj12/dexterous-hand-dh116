"""Rebuild source-separated wire geometry on immutable Front/Roll20 evidence."""
import argparse
import json
from pathlib import Path

import torch

from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import load_capture
from go2_pvcnn.control.dh116.tabletop_stereo_wire_geometry import (
    WirePointGeometry, source_reference_metrics, source_temporal_metrics,
)
from go2_pvcnn.control.dh116.tabletop_stereo_wire_matching import wire_colour_mask
from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import TrackedWirePointExtractor


METHODS = ('legacy_colour', 'full_patch_guided')


def _same_episode(prior, current):
    return all(prior[key] == current[key] for key in ('run_uuid', 'env_id', 'episode_uid', 'reset_id')) and (
        0. < current['simulation_timestamp'] - prior['simulation_timestamp'] <= .5)


def _reference_support(reference, near, far):
    if (reference.get('wire_mask_source_scope') != 'native_resolved_instance_id'
            or reference.get('wire_instance_resolution', {}).get('wire_inventory_complete') is not True
            or reference.get('wire_mask_blocked_reasons')):
        return None
    depth = reference['depth_m']
    return reference['wire_reference_mask'] & reference['valid'] & torch.isfinite(depth) & (depth >= near) & (depth <= far)


def _process_frame(frame, archive, reference, calibration, config, trackers, previous, prior_identity):
    identities = frame['identity']
    if identities != archive['identity'] or not all(
            torch.equal(frame[key], archive[key]) for key in ('left_rgb', 'right_rgb')):
        raise ValueError('replay archive frame differs from immutable RGB or physical identity')
    support = _reference_support(reference, config['min_depth_m'], config['max_depth_m'])
    colour = wire_colour_mask(frame['left_rgb'])
    matrix = frame.get('K_left', calibration.K_left)
    transform = frame.get('T_world_left', calibration.T_world_left)
    reset = torch.tensor([prior_identity is None or not _same_episode(past, row)
                          for past, row in zip(prior_identity or identities, identities)])
    records, geometries = {}, {}
    for method, tracker in trackers.items():
        archived = archive['methods'][method]
        valid, disparity = archived['valid_depth'], archived['disparity_px']
        if valid.dtype != torch.bool or not bool((torch.isfinite(disparity[valid]) & (disparity[valid] > 0)).all()):
            raise ValueError('archive valid depth requires a positive finite disparity')
        z = torch.where(valid, float(calibration.K_left[0, 0]) * calibration.baseline_m / disparity, float('nan'))
        mask = colour if method == 'legacy_colour' else colour | archive['left_predicted_mask']
        result = tracker.extract(frame['left_rgb'], z, valid, matrix, transform,
                                 timestamp=[row['simulation_timestamp'] for row in identities],
                                 reset_mask=reset, wire_mask=mask)
        geometry = WirePointGeometry.from_extraction(result)
        if not torch.equal(geometry.point_pixels, archived['point_pixels']):
            raise ValueError(f'{method} source replay changed archived sampling pixels')
        geometries[method] = geometry.as_dict()
        rows = []
        for env, identity in enumerate(identities):
            temporal = source_temporal_metrics(geometry, env=env,
                previous=None if bool(reset[env]) else previous.get(method))
            scores = None
            if support is not None:
                index = slice(env, env + 1)
                single = WirePointGeometry(**{key: tensor[index] for key, tensor in geometry.as_dict().items()})
                K = matrix if len(torch.as_tensor(matrix).shape) == 2 else matrix[index]
                scores = source_reference_metrics(single, z[index], valid[index],
                    reference['depth_m'][index], support[index], K)
            rows.append(dict(identity=identity, temporal=temporal, reference=scores,
                native_reference_scope='left_resolved_wire_instance_id_optical_z' if scores is not None else None))
        records[method] = rows
        previous[method] = geometry
    return records, geometries


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--correspondence-summary', type=Path, required=True)
    parser.add_argument('--correspondence-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('positive thread count required')
    torch.set_num_threads(args.threads)

    capture, references, calibration, config, proof = load_capture(args.proof)
    archived = args.correspondence_summary
    archived_sha = file_sha256(archived)
    if archived_sha != args.correspondence_sha256:
        raise ValueError('correspondence replay summary SHA mismatch')
    source = json.loads(archived.read_text(encoding='utf-8'))
    capture_sha, proof_sha = proof['capture_artifact']['sha256'], file_sha256(args.proof)
    if (source['proof_sha256'] != proof_sha or source['capture_sha256'] != capture_sha
            or source['calibration_sha256'] != calibration.sha256
            or source['reference_sha256'] != [item['sha256'] for item in proof['reference_artifacts']]
            or len(source['frames']) != len(capture['frames'])):
        raise ValueError('correspondence archive and native proof differ')

    args.output.mkdir(parents=True, exist_ok=False)
    trackers = {method: TrackedWirePointExtractor(point_sampling='observed_ribbon',
                ribbon_radius_px=3, observed_surface_capacity=96) for method in METHODS}
    previous, prior_identity = {}, None
    records = []
    for frame, reference, item in zip(capture['frames'], references, source['frames']):
        path = archived.parent / item['artifact_file']
        if frame['frame_id'] != item['frame_id'] or file_sha256(path) != item['artifact_sha256']:
            raise ValueError('archived correspondence frame ID or SHA mismatch')
        archive = torch.load(path, map_location='cpu', weights_only=False)
        metrics, geometries = _process_frame(frame, archive, reference, calibration, config, trackers,
                                             previous, prior_identity)
        prior_identity = frame['identity']
        target = args.output / f'frame_{frame["frame_id"]:04d}_geometry.pt'
        with target.open('xb') as stream:
            torch.save(dict(identity=prior_identity, correspondence_frame_sha256=item['artifact_sha256'],
                            geometry=geometries), stream)
        records.append(dict(frame_id=frame['frame_id'], methods=metrics,
                            geometry_file=target.name, geometry_sha256=file_sha256(target)))
    summary = dict(scope='CPU_replay_of_saved_native_captures', native_started=False,
        proof_sha256=proof_sha, capture_sha256=capture_sha, calibration_sha256=calibration.sha256,
        correspondence_summary_sha256=archived_sha, point_geometry_code_sha256=file_sha256(
            Path(__file__).resolve().parents[1] / 'src/go2_pvcnn/control/dh116/tabletop_stereo_wire_geometry.py'),
        fixed_denominator='valid_left_optical_z_on_resolved_native_wire_id_in_configured_depth_range',
        correct_3d_rule_m=.005, metric_centerline_ground_truth_available=False,
        material_node_ground_truth_available=False, frames=records)
    with (args.output/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
