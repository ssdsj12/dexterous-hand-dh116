"""Same-cohort unique measured-pixel sampling comparison; no stereo recompute."""
import argparse
import json
from pathlib import Path
import torch
from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import load_capture, _label_masks
from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import TrackedWirePointExtractor
from go2_pvcnn.control.dh116.tabletop_stereo_wire_diagnostics import audit_wire_frame


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proof', type=Path, required=True)
    p.add_argument('--depth-replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(2)
    capture, references, calibration, config, proof = load_capture(args.proof)
    old = json.loads((args.depth_replay/'summary.json').read_text())
    if (old['capture_sha256'] != proof['capture_artifact']['sha256']
            or old['calibration_sha256'] != calibration.sha256
            or old['reference_sha256'] != [x['sha256'] for x in proof['reference_artifacts']]):
        raise ValueError('same-cohort immutable capture/reference binding mismatch')
    result = dict(scope='same_cohort_offline_sampling_of_real_native_RGB_depth',
        capture_sha256=old['capture_sha256'], proof_sha256=file_sha256(args.proof),
        depth_replay_summary_sha256=file_sha256(args.depth_replay/'summary.json'),
        matching_changed=False, native_capture_started=False, token_limit=128, frames=[])
    trackers = {n:TrackedWirePointExtractor(point_sampling='observed_ribbon', ribbon_radius_px=3,
        observed_surface_capacity=n) for n in (32, 96)}
    args.output.mkdir(parents=True, exist_ok=False)
    for i, (frame, ref) in enumerate(zip(capture['frames'], references)):
        binding = old['frames'][i]['methods']['wire_silhouette']
        source = args.depth_replay/binding['pixel_maps_file']
        if file_sha256(source) != binding['pixel_maps_sha256']:
            raise ValueError('saved native replay pixel-map SHA mismatch')
        data = torch.load(source, map_location='cpu', weights_only=False)
        z = data['stereo_depth_m']
        prediction = dict(depth_m=z, valid=torch.isfinite(z),
            diagnostics=dict(reasons=data['rejection_flags']))
        methods = {}
        for count, tracker in trackers.items():
            points = tracker.extract(frame['left_rgb'], z, torch.isfinite(z), frame.get('K_left',calibration.K_left),
                frame.get('T_world_left',calibration.T_world_left), timestamp=[x['simulation_timestamp'] for x in frame['identity']])
            maps, rows = audit_wire_frame(frame, prediction, points, calibration, ref,
                block_size=config['block_size'], near=config['min_depth_m'], far=config['max_depth_m'],
                finger=_label_masks(ref,frame)['finger'])
            name = f'frame_{i:04d}_surface{count}.pt'
            torch.save(dict(pixel_maps=maps, point_pixels=points['point_pixels'], xyz=points['xyz'],
                valid=points['point_valid'], provenance=points['point_provenance'],
                component_id=points['component_id'],track_id=points['track_id'],
                image_track_id=points['image_track_id']), args.output/name)
            methods[str(count)] = dict(environments=rows, pixel_file=name, sha256=file_sha256(args.output/name))
        result['frames'].append(dict(frame_id=frame['frame_id'], simulation_time=frame['simulation_time'],
            depth_pixel_sha256=binding['pixel_maps_sha256'], methods=methods))
    (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
