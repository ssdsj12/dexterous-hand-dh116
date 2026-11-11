"""Offline same-cohort replay of hard masks and soft full-patch wire matching."""
import argparse
import json
import time
from pathlib import Path

import torch

from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth
from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import load_capture
from go2_pvcnn.control.dh116.tabletop_stereo_wire_correspondence import (
    SoftWireCorrespondence, colour_anchor_fallback, score_native_correspondence,
    segmentation_evidence, silhouette_point_trace,
)
from go2_pvcnn.control.dh116.tabletop_stereo_wire_diagnostics import audit_wire_frame, wire_intersections
from go2_pvcnn.control.dh116.tabletop_stereo_wire_matching import wire_colour_mask
from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import StereoWireSegmentation
from go2_pvcnn.control.dh116.tabletop_stereo_wire_tail_audit import tail_geometry
from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import TrackedWirePointExtractor


def load_tail_fixture(path, capture_sha, calibration, frame_ids):
    if path is None:
        return []
    fixture = json.loads(path.read_text(encoding='utf-8'))
    positions = fixture['positions']
    if (fixture['capture_sha256'] != capture_sha or fixture['calibration_sha256'] != calibration.sha256
            or fixture['frames'] != frame_ids or len(positions) != 12
            or any(len(row) != 4 or any(type(n) is not int for n in row) for row in positions)
            or len({tuple(row[:3]) for row in positions}) != len(positions)):
        raise ValueError('fixed tail fixture does not match immutable cohort')
    return positions


def fixed_tail_rows(frame, reference, calibration, legacy, prediction, positions, traces):
    rows = []
    for env_id, y, x, expected in positions:
        slots = [i for i, row in enumerate(frame['identity']) if row['env_id'] == env_id]
        if len(slots) != 1 or not (0 <= y < calibration.height and 0 <= x < calibration.width):
            raise ValueError('fixed tail identity is absent from replay')
        index = slots[0], y, x
        if not bool(reference['wire_reference_mask'][index] & reference['valid'][index]):
            raise ValueError('fixed tail no longer has native left wire reference')
        if float(legacy['disparity_px'][index]) != expected:
            raise ValueError('legacy fixed-tail disparity regression')
        zr = float(reference['depth_m'][index])
        geometry = tail_geometry(float(legacy['depth_m'][index]), zr, calibration.K_left,
                                 y=y, x=x, disparity=expected, baseline=calibration.baseline_m)
        if not geometry['is_tail_5mm']:
            raise ValueError('legacy fixed-tail geometric regression')
        accepted = bool(prediction['valid'][index])
        row = dict(frame_id=frame['frame_id'], env_id=env_id, y=y, x=x,
                   legacy=geometry, accepted=accepted, candidate=bool(prediction['candidate_mask'][index]),
                   match_trace=traces[index],
                   rejection_flags={key: None if value is None else bool(value[index])
                                    for key, value in prediction['diagnostics']['reasons'].items()})
        row['candidate_geometry'] = tail_geometry(
            float(prediction['depth_m'][index]), zr, calibration.K_left, y=y, x=x,
            disparity=float(prediction['disparity_px'][index]), baseline=calibration.baseline_m) if accepted else None
        rows.append(row)
    return rows


def correct_point_transition_counts(baseline_correct, candidate_correct, accurate_depth, measured_depth):
    """Partition lost baseline samples on the same pixel grid; retain gains."""
    if any(mask.shape != baseline_correct.shape or mask.dtype != torch.bool
           for mask in (baseline_correct, candidate_correct, accurate_depth, measured_depth)):
        raise ValueError('aligned boolean point and depth evidence required')
    if (candidate_correct & ~accurate_depth).any() or (accurate_depth & ~measured_depth).any():
        raise ValueError('correct points require measured depth within 5 mm')
    omitted = baseline_correct & ~candidate_correct
    return [dict(baseline_correct=int(baseline_correct[i].sum()),
                 retained_correct=int((baseline_correct[i] & candidate_correct[i]).sum()),
                 lost_depth_rejected=int((omitted[i] & ~measured_depth[i]).sum()),
                 lost_depth_error_above_5mm=int((omitted[i] & measured_depth[i] & ~accurate_depth[i]).sum()),
                 lost_sampling=int((omitted[i] & accurate_depth[i]).sum()),
                 gained_correct=int((~baseline_correct[i] & candidate_correct[i]).sum()),
                 candidate_correct=int(candidate_correct[i].sum()))
            for i in range(len(baseline_correct))]


def accurate_wire_depth(prediction, reference, calibration, matcher):
    depth = prediction['depth_m']
    ref = reference['depth_m']
    fixed = (reference['wire_reference_mask'] & reference['valid'] & torch.isfinite(ref)
             & (ref >= matcher.near) & (ref <= matcher.far))
    measured = fixed & prediction['valid'] & torch.isfinite(depth) & (depth > 0)
    e, y, x = measured.nonzero(as_tuple=True)
    pix = torch.stack((x.double()+.5, y.double()+.5, torch.ones_like(x, dtype=torch.double)), -1)
    inv_k = torch.linalg.inv(torch.as_tensor(calibration.K_left, dtype=torch.float64, device=depth.device))
    rays = pix @ inv_k.T
    spatial = (depth[e, y, x].double()-ref[e, y, x].double()).abs() * rays.norm(dim=-1)
    accurate = torch.zeros_like(measured)
    accurate[e, y, x] = spatial <= .005
    return measured, accurate


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--segmentation-checkpoint', type=Path, required=True)
    parser.add_argument('--segmentation-sha256', required=True)
    parser.add_argument('--tail-fixture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--colour-anchor-fallback', action='store_true',
                        help='compare an additional opt-in colour-anchored soft fallback')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('positive thread count required')
    torch.set_num_threads(args.threads)
    capture, references, calibration, config, proof = load_capture(args.proof)
    capture_sha = proof['capture_artifact']['sha256']
    checkpoint_sha = file_sha256(args.segmentation_checkpoint)
    if checkpoint_sha != args.segmentation_sha256:
        raise ValueError('segmentation checkpoint SHA mismatch')
    payload = torch.load(args.segmentation_checkpoint, map_location='cpu', weights_only=False)
    contract = payload.get('contract', {})
    if (contract.get('version') != 'dh116-wire-segmentation-diagnostic-v1'
            or contract.get('source') != 'native_instance_id'):
        raise ValueError('native-instance diagnostic checkpoint required')
    segmenter = StereoWireSegmentation(payload['width']).eval()
    segmenter.load_state_dict(payload['model'], strict=True)
    positions = load_tail_fixture(args.tail_fixture, capture_sha, calibration,
                                  [frame['frame_id'] for frame in capture['frames']])
    legacy = StereoDepth(**dict(config, method='wire_silhouette', subpixel=False))
    soft_config = legacy.to_config() | {'method': 'blockmatch', 'subpixel': True}
    soft = SoftWireCorrespondence(**soft_config)
    unguided = SoftWireCorrespondence(mask_weight=0., **soft_config)
    methods = ['legacy_colour', 'legacy_hard', 'full_patch_unguided', 'full_patch_guided']
    if args.colour_anchor_fallback:
        methods.append('colour_anchor_fallback')
    trackers = {method: TrackedWirePointExtractor(point_sampling='observed_ribbon',
                ribbon_radius_px=3, observed_surface_capacity=96)
                for method in methods}

    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(scope='offline_replay_of_existing_native_RGB_only', native_capture_started=False,
                  capture_sha256=capture_sha, proof_sha256=file_sha256(args.proof),
                  calibration_sha256=calibration.sha256, checkpoint_sha256=checkpoint_sha,
                  tail_fixture_sha256=file_sha256(args.tail_fixture) if args.tail_fixture else None,
                  reference_sha256=[row['sha256'] for row in proof['reference_artifacts']],
                  rgb_patch_masked=False, reference_used_by_matcher=False, frames=[])
    for index, (frame, reference) in enumerate(zip(capture['frames'], references)):
        left, right = frame['left_rgb'], frame['right_rgb']
        views = segmentation_evidence(segmenter(left, right))
        ordered = tuple(views[side] for side in ('left', 'right'))
        masks = tuple(view['runtime_mask'] for view in ordered)
        probabilities = tuple(view['visible_probability'] for view in ordered)
        visible_masks = tuple(view['visible_mask'] for view in ordered)
        slots = {row['env_id']: i for i, row in enumerate(frame['identity'])}
        trace_pixels = [(slots[e], y, x) for e, y, x, _ in positions]
        frame_record = dict(frame_id=frame['frame_id'], identity=frame['identity'], methods={})
        archive = dict(identity=frame['identity'], left_rgb=left, right_rgb=right,
                       left_native_instance_ids=reference.get('raw_instance_ids'),
                       right_native_instance_ids=reference.get('right_raw_instance_ids'),
                       left_native_wire_mask=reference.get('wire_reference_mask'),
                       right_native_wire_mask=reference.get('right_wire_reference_mask'),
                       left_predicted_mask=masks[0], right_predicted_mask=masks[1],
                       left_visible_probability=probabilities[0], right_visible_probability=probabilities[1],
                       per_view_segmentation=views,
                       reference_sha256=proof['reference_artifacts'][index]['sha256'],
                       methods={})
        previous = None
        guided = None
        baseline_correct = None
        colour_masks = (wire_colour_mask(left), wire_colour_mask(right))
        for name, tracker in trackers.items():
            start = time.perf_counter()
            if name == 'legacy_colour':
                prediction = legacy.compute(left, right, calibration)
                previous = prediction
                candidate = colour_masks[0]
            elif name == 'legacy_hard':
                prediction = legacy.compute(left, right, calibration, wire_masks=masks)
                candidate = masks[0]
            elif name == 'colour_anchor_fallback':
                prediction = colour_anchor_fallback(previous, guided, colour_masks)
                candidate = colour_masks[0]
            else:
                estimator = soft if name == 'full_patch_guided' else unguided
                prediction = estimator.compute(left, right, calibration, probabilities=probabilities,
                                               left_proposal=masks[0], trace_pixels=trace_pixels)
                candidate = prediction['candidate_mask']
                if name == 'full_patch_guided':
                    guided = prediction
            elapsed = time.perf_counter() - start
            points = tracker.extract(left, prediction['depth_m'], prediction['valid'],
                frame.get('K_left', calibration.K_left), frame.get('T_world_left', calibration.T_world_left),
                timestamp=[row['simulation_timestamp'] for row in frame['identity']],
                point_confidence=prediction['confidence'], wire_mask=candidate)
            maps, metrics = audit_wire_frame(frame, prediction, points, calibration, reference,
                block_size=legacy.block_size, near=legacy.near, far=legacy.far, wire_mask=candidate)
            scores, visibility = score_native_correspondence(frame, reference, calibration, prediction, masks,
                                                               raw_visible_masks=visible_masks)
            native = reference['wire_reference_mask'] & reference['valid']
            flags = {key: None if flag is None else (flag & candidate)
                     for key, flag in prediction['diagnostics']['reasons'].items()}
            flags['outside_candidate_roi'] = ~candidate
            native_maps, native_rejections = wire_intersections(native, prediction['valid'],
                maps['wire_3d_point_valid'] & native, flags)
            measured, accurate = accurate_wire_depth(prediction, reference, calibration, legacy)
            correct = accurate & maps['wire_3d_point_valid']
            if baseline_correct is None:
                baseline_correct = correct
            transitions = correct_point_transition_counts(baseline_correct, correct, accurate, measured)
            for entry, score, native_rejection, transition in zip(metrics, scores, native_rejections, transitions):
                entry['correspondence'] = score
                entry['fixed_native_rejection_counts'] = native_rejection
                entry['correct_point_transition_from_legacy_colour'] = transition
                if transition['candidate_correct'] != entry['native_reference_metrics']['sampled_points']['correct_3d_points_5mm']:
                    raise RuntimeError('per-pixel transition counts differ from fixed-reference score')
            lr_consistent = ~prediction['occlusion'] if 'occlusion' in prediction else prediction['lr_consistency']
            archive['methods'][name] = dict(disparity_px=prediction['disparity_px'],
                raw_disparity_px=prediction['diagnostics']['raw_disparity_px'],
                lr_consistency=lr_consistent,
                valid_depth=prediction['valid'], final_3d_valid_mask=maps['wire_3d_point_valid'],
                point_pixels=points['point_pixels'], visibility=visibility,
                first_reject_code=maps['first_reject_code'], rejection_flags=prediction['diagnostics']['reasons'],
                fixed_native_first_reject_code=native_maps['first_reject_code'])
            if name == 'colour_anchor_fallback':
                archive['methods'][name]['baseline_depth_preserved'] = prediction['diagnostics']['baseline_depth_preserved']
                archive['methods'][name]['soft_fallback_depth'] = prediction['diagnostics']['soft_fallback_depth']
            frame_record['methods'][name] = dict(elapsed_s=elapsed, environments=metrics)
            if positions:
                if name.startswith('full_patch'):
                    traces = prediction['diagnostics']['point_traces']
                elif name == 'colour_anchor_fallback':
                    traces = {key: (guided['diagnostics']['point_traces'][key]
                                    if prediction['diagnostics']['soft_fallback_depth'][key]
                                    else silhouette_point_trace(left, right, colour_masks, key, legacy, previous))
                              for key in trace_pixels}
                else:
                    applied_masks = colour_masks if name == 'legacy_colour' else masks
                    traces = {key: silhouette_point_trace(left, right, applied_masks, key, legacy, prediction)
                              for key in trace_pixels}
                prediction['candidate_mask'] = candidate
                frame_record['methods'][name]['fixed_tail_cases'] = fixed_tail_rows(
                    frame, reference, calibration, previous, prediction, positions, traces)
        artifact = args.output / f'frame_{index:04d}.pt'
        with artifact.open('xb') as stream:
            torch.save(archive, stream)
        frame_record['artifact_file'] = artifact.name
        frame_record['artifact_sha256'] = file_sha256(artifact)
        result['frames'].append(frame_record)
        print(f'completed frame {frame["frame_id"]}', flush=True)
    with (args.output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
