"""Offline confidence diagnostics for the reserved stereo candidate pool."""

import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .tabletop_stereo_contract import STYLES, require_sha
from .tabletop_stereo_state_supervision import split_estimator_rows


_SOURCES = {1: 'legacy_dense_not_sampled', 2: 'rgb_soft_fallback'}
_PARTITIONS = ('train', 'validation', 'test')


def _checked_bytes(path, expected, name):
    require_sha(expected, name + ' SHA256')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(name + ' SHA mismatch')
    return raw


def load_replay_rows(summary_path, expected_sha):
    """Load candidate confidence and immutable 5 mm labels, without re-running the matcher."""
    summary_path = Path(summary_path)
    summary = json.loads(_checked_bytes(summary_path, expected_sha, 'replay summary'))
    retention = summary.get('baseline_retention', {})
    if (summary.get('scope') != 'offline_replay_of_existing_native_RGB_only'
            or summary.get('native_capture_started') is not False
            or summary.get('reference_used_by_matcher') is not False
            or retention.get('reference_used_for_selection') is not False
            or retention.get('candidate_5mm_label_is_offline_only') is not True
            or retention.get('deployable') is not False):
        raise ValueError('bound offline candidate-only replay summary required')
    capture_sha = require_sha(summary['capture_sha256'], 'capture SHA256')
    calibration_sha = require_sha(summary['calibration_sha256'], 'calibration SHA256')
    checkpoint_sha = require_sha(summary['checkpoint_sha256'], 'segmentation Checkpoint SHA256')
    proof_sha = require_sha(summary['proof_sha256'], 'proof SHA256')
    frames = summary['frames']
    references = summary['reference_sha256']
    if not frames or len(frames) != len(references):
        raise ValueError('one native reference binding per replay frame required')
    rows, baseline = [], []
    artifact_names = set()
    for record, reference_sha in zip(frames, references):
        name = record['artifact_file']
        if Path(name).name != name or name in artifact_names:
            raise ValueError('replay artifact path is not unique and local to summary')
        artifact_names.add(name)
        require_sha(reference_sha, 'native reference SHA256')
        raw = _checked_bytes(summary_path.parent / name, record['artifact_sha256'], 'replay frame artifact')
        archive = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        del raw
        identities = record['identity']
        if (archive['identity'] != identities or archive['reference_sha256'] != reference_sha
                or any(i['frame_id'] != record['frame_id']
                       or i['calibration_sha256'] != calibration_sha for i in identities)):
            raise ValueError('replay frame identity differs from summary binding')
        retained = archive['methods']['colour_anchor_retained']
        pool = retained['baseline_slot_retention']
        candidate = pool['candidate_geometry']
        mask, source = pool['candidate_pool_mask'], pool['candidate_depth_source']
        label, confidence = pool['candidate_reference_label_5mm'], candidate['confidence']
        pixels, valid = candidate['point_pixels'], candidate['valid']
        source_type = candidate['source_type']
        if (mask.dtype != torch.bool or mask.ndim != 2 or len(mask) != len(identities)
                or any(t.shape != mask.shape for t in (source, label, confidence, valid, source_type))
                or source.dtype != torch.uint8 or label.dtype != torch.int8 or valid.dtype != torch.bool
                or source_type.dtype != torch.uint8 or not (source_type[mask] == 2).all()
                or not confidence.is_floating_point() or pixels.dtype != torch.long
                or pixels.shape != (*mask.shape, 2) or (mask & ~valid).any()
                or not (((source[mask] == 1) | (source[mask] == 2)).all()
                        and ((label[mask] >= -1) & (label[mask] <= 1)).all())):
            raise ValueError('invalid bound candidate tensors')
        summaries = record['methods']['colour_anchor_retained']['environments']
        if len(summaries) != len(identities):
            raise ValueError('candidate metrics do not bind each environment')
        for env, (identity, metrics) in enumerate(zip(identities, summaries)):
            if metrics['identity'] != identity:
                raise ValueError('candidate metric identity differs from replay frame')
            counts = metrics['baseline_slot_retention']
            observed = label[env, mask[env]]
            if (counts['promoted_slots'] != 0 or counts['candidate_pool_size'] != len(observed)
                    or counts['candidate_pool_within_5mm'] != int((observed == 1).sum())
                    or counts['candidate_pool_above_5mm'] != int((observed == 0).sum())
                    or counts['candidate_pool_without_reference'] != int((observed == -1).sum())):
                raise ValueError('fixed candidate labels differ from replay summary')
            transition = metrics['correct_point_transition_from_legacy_colour']
            if transition['baseline_correct'] != transition['retained_correct']:
                raise ValueError('replay lost a correct baseline point')
            baseline.append(dict(frame_id=identity['frame_id'], env_id=identity['env_id'],
                grasp_style=identity['grasp_style'], baseline_correct=transition['baseline_correct'],
                retained_correct=transition['retained_correct']))
            native_right_ids = archive.get('right_native_instance_ids')
            visible = retained['visibility']['visible']
            right = metrics['correspondence']['right_instance_source'] == 'native_instance_id'
            if (visible.ndim != 3 or len(visible) != len(identities) or visible.dtype != torch.bool
                    or right and (not isinstance(native_right_ids, torch.Tensor)
                                  or native_right_ids.shape != visible.shape)):
                raise ValueError('invalid same-frame right identity evidence')
            height, width = visible.shape[1:]
            for slot in mask[env].nonzero().flatten().tolist():
                y, x = pixels[env, slot].tolist()
                if not (0 <= y < height and 0 <= x < width):
                    raise ValueError('candidate pixel outside bound reference frame')
                value = int(label[env, slot])
                rows.append(dict(sample_id=f'{capture_sha}:{identity["frame_id"]}:{identity["env_id"]}:{y}:{x}',
                    identity=identity, checkpoint_sha256=checkpoint_sha,
                    depth_source=_SOURCES[int(source[env, slot])], confidence=float(confidence[env, slot]),
                    reference_label_5mm=None if value == -1 else value,
                    right_evidence=right and bool(visible[env, y, x])))
    return rows, dict(summary_sha256=expected_sha, capture_sha256=capture_sha,
        calibration_sha256=calibration_sha, proof_sha256=proof_sha,
        replay_frames=len(frames), retained_baselines=baseline)


def _bin(value, count):
    return min(int(value * count), count - 1)


def _score(rows, groups, scores, bins):
    scored = [(i, probability) for i, probability in scores if probability is not None]
    if not scored:
        return None
    sizes = Counter(groups[i] for i, _ in scored)
    values = defaultdict(lambda: [0., 0., 0., 0])
    brier, accuracy = 0., 0.
    for i, probability in scored:
        label = rows[i]['reference_label_5mm']
        weight = 1 / sizes[groups[i]]
        bucket = values[_bin(probability, bins)]
        bucket[0] += weight
        bucket[1] += weight * probability
        bucket[2] += weight * label
        bucket[3] += 1
        brier += weight * (probability - label)**2
        accuracy += weight * label
    n_groups = len(sizes)
    return dict(samples=len(scored), groups=n_groups, brier=brier/n_groups,
        ece=sum(abs(v[1]-v[2]) for v in values.values())/n_groups,
        observed_accuracy=accuracy/n_groups,
        reliability_bins=[dict(index=b, samples=v[3], weight=v[0]/n_groups,
                               mean_score=v[1]/v[0], observed_accuracy=v[2]/v[0])
                          for b, v in sorted(values.items())])


def _heldout(rows, groups, by_part, bins):
    fit = by_part['train']
    if (len({groups[i] for i in fit}) < 2 or not by_part['validation'] or not by_part['test']
            or {rows[i]['reference_label_5mm'] for i in fit} != {0, 1}):
        return None
    fit_bins = defaultdict(lambda: defaultdict(list))
    for i in fit:
        b = _bin(rows[i]['confidence'], bins)
        fit_bins[b][groups[i]].append(rows[i]['reference_label_5mm'])
    means = {b: sum(sum(y)/len(y) for y in by_group.values())/len(by_group)
             for b, by_group in fit_bins.items()}
    result = dict(calibrator_bins=[dict(index=b, probability=means[b], fit_groups=len(fit_bins[b]))
                                   for b in sorted(means)])
    for part in ('validation', 'test'):
        indices = by_part[part]
        scores = [(i, means.get(_bin(rows[i]['confidence'], bins))) for i in indices]
        result[part] = dict(reference_samples=len(indices), coverage=sum(p is not None for _, p in scores)/len(indices),
            scores=_score(rows, groups, scores, bins),
            raw_scores=_score(rows, groups, [(i, rows[i]['confidence']) for i in indices], bins))
    return result


def audit_confidence(rows, *, bins=10, seed=116):
    """Use the existing source-family/physical-initial graph before assigning any fold."""
    if type(bins) is not int or not 2 <= bins <= 50:
        raise ValueError('bins must be an integer in [2, 50]')
    ids = set()
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        identity = row['identity']
        sample_id, score = row['sample_id'], row['confidence']
        label = row['reference_label_5mm']
        if (not isinstance(sample_id, str) or not sample_id or type(score) not in (int, float)
                or not math.isfinite(score) or not 0 <= score <= 1
                or label is not None and (type(label) is not int or label not in (0, 1))
                or type(row['right_evidence']) is not bool
                or row['depth_source'] not in _SOURCES.values() or identity['grasp_style'] not in STYLES):
            raise ValueError('invalid candidate evidence')
        if sample_id in ids:
            raise ValueError('duplicate candidate evidence')
        ids.add(sample_id)
        calibration = require_sha(identity['calibration_sha256'], 'calibration SHA256')
        checkpoint = require_sha(row['checkpoint_sha256'], 'segmentation Checkpoint SHA256')
        strata[(calibration, checkpoint, identity['grasp_style'], row['depth_source'])].append(i)
    split = split_estimator_rows(rows, seed=seed)
    groups = split['group_for_row']
    partitions = {part: set(split[part+'_indices']) for part in _PARTITIONS}
    result = []
    for (calibration, checkpoint, style, source), indices in sorted(strata.items()):
        resolved = [i for i in indices if rows[i]['reference_label_5mm'] is not None]
        right = {i for i in resolved if rows[i]['right_evidence']}
        by_part = {part: sorted(partitions[part] & right) for part in _PARTITIONS}
        heldout = _heldout(rows, groups, by_part, bins)
        if not right:
            status = 'blocked_missing_grouped_right_reference'
        elif heldout is None:
            status = 'blocked_insufficient_independent_group_or_label_support'
        else:
            status = 'offline_heldout_measured_not_deployable'
        result.append(dict(calibration_sha256=calibration, checkpoint_sha256=checkpoint,
            grasp_style=style, depth_source=source, candidate_samples=len(indices),
            independent_groups=len({groups[i] for i in indices}),
            unresolved_reference_samples=len(indices)-len(resolved),
            right_verified_reference_samples=len(right),
            right_verified_groups_by_partition={key:len({groups[i] for i in values})
                                                for key, values in by_part.items()},
            raw_left_reference_diagnostic=_score(rows, groups, [(i, rows[i]['confidence']) for i in resolved], bins),
            heldout_status=status, heldout=heldout))
    return dict(scope='offline_candidate_confidence_audit', deployable=False, promoted_samples=0,
        raw_confidence_semantics='bounded matcher score, not calibrated probability',
        native_reference_error_max_m=.005, independent_groups=split['independent_groups'],
        split_status=split['status'],
        group_by_sample_id={row['sample_id']: groups[i] for i, row in enumerate(rows)},
        partitions={part:[rows[i]['sample_id'] for i in sorted(indices)] for part, indices in partitions.items()},
        strata=result)
