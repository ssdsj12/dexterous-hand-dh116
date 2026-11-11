"""Offline state-only training on native synchronized Teacher candidates."""
from argparse import ArgumentParser
from collections import defaultdict
import json
import math
from pathlib import Path
import random

import torch

from .tabletop_stereo_contract import file_sha256, new_run, physical_initial_sha256, require_sha, write_json_new
from .tabletop_stereo_losses import StereoEstimationLoss, masked_bce, masked_smooth_l1, quaternion_geodesic_loss
from .tabletop_stereo_observation import StereoHistoryBatch
from .tabletop_stereo_state_estimator import HEAD_SHAPES, StereoStateEstimator
from .tabletop_stereo_supervision_dataset import identity_key, rows_from_records, validate_rows
from .tabletop_stereo_task_state_adapter import TaskStateAdapter, field_history_frames


HEADS = {
    'geometry': ('geometry_head', ('geometry',)),
    'velocity': ('velocity_head', ('velocity',)),
    'deformation': (None, ('deformation',)),
    'contact': ('contact_head', ('contact_logits',)),
    'slip': ('slip_head', ('slip_logits', 'slip_velocity')),
    'force_proxy': (None, ('force_proxy',)),
}
PARTITIONS = ('train', 'validation', 'test')
SOURCE_KEYS = ('source_valid', 'identity_aligned', 'timestamp_valid', 'visibility_valid', 'correspondence_valid')
_ENCODERS_ONLY = 'simulation_SDK_encoder_readback_only'


def audit_candidate_fields(rows=()):
    """Report semantic field eligibility separately from actual label counts."""
    adapter = TaskStateAdapter()
    loss = StereoEstimationLoss()
    counts = torch.zeros(371, dtype=torch.long)
    direct = torch.zeros_like(counts)
    for row in rows:
        for name in HEADS:
            target = _targets(row, name, loss).get('privileged_state')
            if target is not None:
                counts += target[1]
        if row['extra_labels'].get('training_label_source_scope') == _ENCODERS_ONLY:
            direct[:24] += row['privileged_valid'][:24]
    fields = []
    for index in adapter.field_indices:
        row = adapter._rows[index]
        measured = row['block'] in ('joint_position', 'joint_velocity', 'last_submitted_action24')
        head = row['estimator_head']
        # Distinguish the deployed prediction/passthrough from its candidate teacher label.
        actor_source = {
            'joint_position': 'history.proprio[0:12] with explicit joint_lower and joint_upper',
            'joint_velocity': 'history.proprio[12:24]',
            'last_submitted_action24': 'history.proprio[38:62] actually issued previous action',
        }.get(row['block'], 'StereoStateEstimator.privileged_state prediction at index ' + str(index))
        fields.append(dict(index=index, name=row['semantic_name'], unit=row['unit'],
            coordinate_frame=row['coordinate_frame'], normalization=row['normalization'],
            actor_source=actor_source, candidate_label_source=row['current_source'],
            label_loss_type=row['loss_type'], validity_keys=row['required_validity_keys'],
            validity_description=row['validity_mask'], minimum_history_frames=field_history_frames(row),
            uncertainty='unknown sentinel 1 in Actor features; no calibrated error',
            uncertainty_calibrated=False, actor_eligible=True,
            supervision_role='direct_sensor_or_command' if measured else 'vision_prediction',
            supervision_samples=int(counts[index]), direct_readback_samples=int(direct[index]),
            has_real_supervision=bool(counts[index]),
            availability='direct_input' if measured else 'qualified_labels' if counts[index] else 'schema_only'))
    return dict(state_contract_sha256=adapter.state_contract_sha256, candidate_count=len(fields),
        excluded_na_indices=[row['index'] for row in adapter._rows if row['supervision_status']=='NA'],
        fields=fields)


def split_estimator_rows(rows, *, seed=116):
    """Partition connected components of physical hashes and source families."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    first = {}
    for i, row in enumerate(rows):
        identity = row['identity']
        initial = require_sha(identity.get('physical_initial_condition_sha256'), 'actual initial SHA')
        family = identity.get('source_episode_group')
        if initial == '0'*64 or not isinstance(family, str) or not family.strip():
            raise ValueError('real physical initial identity and nonempty source episode group required')
        for key in (('physical', initial), ('source', family)):
            parents[find(i)] = find(first.setdefault(key, i))
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    components = sorted(groups.values())
    result = dict(independent_groups=len(components), group_for_row={i:k for k, rows in enumerate(components) for i in rows},
                  **{name+'_indices': [] for name in PARTITIONS})
    if len(components) < 3:
        return dict(result, status='blocked_insufficient_independent_groups')
    random.Random(seed).shuffle(components)
    holdout = max(1, len(components)//5)
    for name, chosen in zip(PARTITIONS, (components[2*holdout:], components[:holdout], components[holdout:2*holdout])):
        result[name+'_indices'] = sorted(i for group in chosen for i in group)
    return dict(result, status='ready')


def _raw_bindings(path, rows, checkpoint):
    wanted = {identity_key(row['identity']):row for row in rows}
    wanted_clocks = {(*key[:4],row['identity']['physics_tick_end']):row for key,row in wanted.items()}
    matches, same_clock, headers = {}, {}, {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            uid = (record.get('run_uuid'), record.get('env_id'), record.get('episode_uid'), record.get('reset_id'))
            if record.get('kind') == 'episode_start':
                headers[uid] = record
            clock = (*uid, record.get('physics_step'))
            if record.get('kind') == 'sample' and clock in wanted_clocks:
                if clock in same_clock:
                    raise ValueError('duplicate native physics readback for candidate clock')
                same_clock[clock] = record
            key = (*uid, record.get('policy_tick'))
            if record.get('kind') != 'sample' or key not in wanted:
                continue
            row = wanted[key]
            identity = row['identity']
            physics_step, timestamp = record.get('physics_step'), record.get('timestamp')
            last_step, last_time, terminal = matches.get(
                key, (identity['physics_tick_end'], identity['simulation_timestamp'], False))
            if (record.get('owner') != 'teacher' or record.get('action_owner') != 'teacher'
                    or record.get('checkpoint_sha256') != checkpoint
                    or type(physics_step) is not int or terminal
                    or not physics_step == last_step+1 <= row['execution']['physics_tick_end']
                    or type(timestamp) not in (int,float) or not math.isfinite(timestamp)
                    or not last_time < timestamp <= row['execution']['simulation_timestamp_end']+1e-8
                    or record.get('frame_id') != identity['frame_id']
                    or record.get('calibration_sha256') != identity['calibration_sha256']
                    or record.get('pre_action_identity') != identity):
                raise ValueError('native raw execution disagrees with synchronized candidate identity')
            issued = torch.as_tensor(record['actor_issued_action'], dtype=row['issued_actions'].dtype)
            if not torch.equal(issued, row['issued_actions']):
                raise ValueError('native raw Teacher action differs from candidate action')
            matches[key] = (physics_step,timestamp,record.get('terminal') is True)
    if matches.keys() != wanted.keys():
        raise ValueError('every stereo candidate needs matching Teacher native execution evidence')
    for key, row in wanted.items():
        step, when, terminal = matches[key]
        if terminal != bool(row['done']) or (not terminal and (
                step != row['execution']['physics_tick_end']
                or not math.isclose(when,row['execution']['simulation_timestamp_end'],rel_tol=0,abs_tol=1e-8))):
            raise ValueError('native candidate ledger does not close at the recorded step boundary')
        header = headers.get(key[:4], {})
        if (header.get('initial_measurement_valid') is not True
                or header.get('physical_initial_condition_source') != 'actual_sdk_readback'
                or header.get('source_episode_group') != row['identity']['source_episode_group']
                or physical_initial_sha256(header.get('physical_initial_condition')) != row['identity']['physical_initial_condition_sha256']):
            raise ValueError('candidate initial group does not match measured native reset state')
    return same_clock


def _verified_canonical_label_mask(row, readback):
    """Verify the implemented rigid geometry/velocity labels against raw SI readback."""
    valid = torch.zeros(371,dtype=torch.bool)
    if (readback is None or readback.get('measurement_valid') is not True
            or readback.get('terminal') is True
            or readback.get('timestamp') != row['identity']['simulation_timestamp']):
        return valid
    pose, wrist, velocity = (readback.get(key) for key in ('formal_object_pose_w','formal_wrist_pose_w','formal_object_velocity'))
    try:
        pose, wrist, velocity = (torch.as_tensor(x,dtype=torch.float64) for x in (pose,wrist,velocity))
    except (ValueError,TypeError,RuntimeError):
        return valid
    if pose.shape == (7,) and wrist.shape == (2,7):
        expected = (pose[:3]-wrist[:,:3].mean(0))/.05
        valid[61:64] = torch.isfinite(expected) & torch.isclose(row['privileged'][61:64].double(), expected, atol=1e-5,rtol=1e-5)
    if velocity.shape == (6,):
        expected = velocity/velocity.new_tensor([.1,.1,.1,1.,1.,1.])
        valid[68:74] = torch.isfinite(expected) & torch.isclose(row['privileged'][68:74].double(), expected, atol=1e-5,rtol=1e-5)
    # Orientation, arbitrary pad geometry, contact, deformation and slip
    # have no independently verified label source in this native collector.
    return valid


def _observed_wire_points(pixels, point_valid, depth, depth_valid, optical, wire):
    """Actual sampled pixels only, checked against independent optical Z."""
    result = torch.zeros_like(point_valid)
    if (not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.long
            or pixels.shape != (*point_valid.shape, 2) or depth.shape != optical.shape
            or depth.shape != wire.shape or depth.shape != depth_valid.shape):
        return result
    bounded = (pixels >= 0).all(-1) & (pixels[:,0] < depth.shape[0]) & (pixels[:,1] < depth.shape[1])
    selected = point_valid & bounded
    y, x = pixels[selected].unbind(-1)
    actual = optical[y,x]
    result[selected] = (wire[y,x] & depth_valid[y,x] & torch.isfinite(actual) & (actual > 0)
                        & ((depth[y,x]-actual).abs() <= .005))
    return result


def _verified_visible_tracks(row, root, cache):
    from .tabletop_stereo_promotion import _verified_wire_mask
    identity = row['identity']
    artifact = identity.get('state_reference_artifact')
    if not isinstance(artifact,dict):
        return False,set()
    path = Path(artifact['path']).resolve(strict=True)
    if path.parent != root or file_sha256(path) != require_sha(artifact['sha256']):
        raise ValueError('candidate state reference differs from native artifact binding')
    if path not in cache:
        ref = torch.load(path,map_location='cpu',weights_only=True)
        if (ref.get('source_scope') != 'native' or ref.get('validation_only') is not True
                or ref.get('reference_used_as_actor_input') is not False or ref.get('source_units') != 'optical_axis_z_m'):
            raise ValueError('independent native optical Z reference required for state labels')
        wire = _verified_wire_mask(ref,path.parent,ref['depth_m'].shape)
        cache.clear()
        cache[path] = ref,wire
    ref, wire = cache[path]
    if (ref['frame_id'] != identity['frame_id'] or ref['simulation_time'] != identity['simulation_timestamp']
            or ref['calibration_sha256'] != identity['calibration_sha256']):
        raise ValueError('candidate state reference frame/clock/calibration differs')
    if wire is None:
        return False,set()
    env = identity['env_id']
    history, labels = row['history'], row['extra_labels']
    point_valid = history['point_valid'][0,-1]
    verified = _observed_wire_points(labels.get('state_wire_point_pixels'),point_valid,
        history['depth_m'][0,-1],history['depth_valid'][0,-1],ref['depth_m'][env],wire[env]&ref['valid'][env])
    tracks = labels.get('state_wire_track_ids')
    visible_tracks = set(tracks[verified & (tracks>=0)].tolist()) if (
        isinstance(tracks,torch.Tensor) and tracks.shape == verified.shape) else set()
    return bool(verified.any()),visible_tracks


def _admit_rigid_labels(row, exact, *, wire_observed, temporal_observed):
    """Open a label only when exact physical readback and the independent view agree."""
    verified = torch.zeros_like(exact)
    if wire_observed:
        verified[61:64] = exact[61:64]
        if temporal_observed:
            # A symmetric cylinder does not independently expose angular motion.
            verified[68:71] = exact[68:71]
    row['verified_canonical_native'] = verified
    if bool(verified.any()):
        row['privileged_valid'] = row['privileged_valid'] | verified
        for key in SOURCE_KEYS:
            existing = row['state_sources'].get(key,torch.zeros_like(verified))
            row['state_sources'][key] = existing | verified
        row['extra_labels']['training_label_source_scope'] = 'independently_matched_rigid_state_readback'


def read_candidate(path, *, raw, sha256, raw_sha256):
    """Accept the existing stereo collector artifact; no BC pool admission."""
    for artifact, expected in ((path, sha256), (raw, raw_sha256)):
        if file_sha256(artifact) != require_sha(expected):
            raise ValueError('state-supervision source artifact SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('version') != 'dh116-stereo-teacher-candidate-records-v1':
        raise ValueError('expected synchronized stereo Teacher candidates, including RGB history and label masks')
    checkpoint = require_sha(payload['teacher_checkpoint_sha256'], 'Teacher checkpoint SHA')
    if checkpoint == '0'*64 or payload.get('candidate_only') is not True or payload.get('successful_demonstration_pool') is not False:
        raise ValueError('frozen Teacher candidates must remain outside the successful BC pool')
    rows = rows_from_records(payload['records'])
    if not rows or any(row['action_owner'] != 'teacher' for row in rows):
        raise ValueError('nonempty Teacher-owned candidate rows required')
    validate_rows(rows, contract={'calibration_sha256': rows[0]['identity']['calibration_sha256']}, schema_only=False)
    for row in rows:
        if row['extra_labels'].get('training_label_evidence_scope') != 'native_runtime_capture_unqualified':
            raise ValueError('candidate lacks measured label-source provenance')
    same_clock = _raw_bindings(raw, rows, checkpoint)
    previous, cache = {},{}
    for row in rows:
        identity = row['identity']
        episode = identity_key(identity)[:4]
        clock = (*episode,identity['physics_tick_end'])
        observed, tracks = _verified_visible_tracks(row,Path(raw).resolve().parent,cache)
        prior_tick, prior_tracks = previous.get(episode,(-2,set()))
        consecutive = prior_tick+1 == identity['policy_tick'] and bool(tracks & prior_tracks)
        exact = _verified_canonical_label_mask(row,same_clock.get(clock))
        _admit_rigid_labels(row,exact,wire_observed=observed,temporal_observed=consecutive)
        previous[episode] = identity['policy_tick'],tracks
    if file_sha256(path) != sha256 or file_sha256(raw) != raw_sha256:
        raise ValueError('state-supervision source changed while reading')
    return rows


def _targets(row, name, loss):
    """Select a head's supervised coordinates; never turn sensor estimates into truth."""
    labels = row.get('extra_labels') or {}
    scope = labels.get('training_label_source_scope')
    if not scope or scope == _ENCODERS_ONLY:
        return {}
    h = row['history']
    history_count = int((h['history_valid'] & h['image_valid'].all(-1)).sum())
    if not bool(h['history_valid'][0,-1] and h['image_valid'][0,-1].all()):
        return {}
    if not bool(h['point_valid'][0,-1].any() or h['depth_valid'][0,-1].any()):
        return {}
    head, extras = HEADS[name]
    data = {}
    if head is not None:
        target = row['privileged']
        source = {key: value[None] for key, value in row['state_sources'].items() if isinstance(value, torch.Tensor)}
        mask = loss.state_mask(target[None], {'privileged_state': row['privileged_valid'][None],
                                               'state_sources': source})[0]
        if 'verified_canonical_native' in row:
            mask &= row['verified_canonical_native']
        for i, field in enumerate(loss.state_contract['dimensions']):
            if field['estimator_head'] != head or field_history_frames(field) > history_count:
                mask[i] = False
        for group in loss.rotations:
            lo, hi = group['slice']
            symmetry = source.get('rotation_symmetry_valid')
            if group.get('symmetry_mask_required') and (
                    symmetry is None or symmetry.shape != (1,) or not bool(symmetry[0])):
                mask[lo:hi] = False
            if not bool(mask[lo:hi].all()):
                mask[lo:hi] = False
        if bool(mask.any()):
            data['privileged_state'] = (target, mask)
    for field in extras:
        if 'verified_canonical_native' in row and field != 'force_proxy':
            continue
        if field not in labels or (name in ('velocity', 'slip') and history_count < 2):
            continue
        label = labels[field]
        validity = row.get('label_validity') or {}
        mask, qualified = validity.get(field), validity.get(field+'_source_valid')
        if (not isinstance(label, torch.Tensor) or not label.is_floating_point() or tuple(label.shape) != HEAD_SHAPES[field]
                or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != label.shape
                or not isinstance(qualified, torch.Tensor) or qualified.dtype != torch.bool or qualified.shape != label.shape):
            continue
        if field == 'force_proxy':
            binding = row.get('force_qualification')
            if binding is not None:
                binding = {k:v[None] if isinstance(v, torch.Tensor) else v for k,v in binding.items()}
            verified, _ = loss._force_mask(binding, label[None])
            qualified = qualified & verified[0] & (label >= 0)
        valid = mask & qualified
        if bool((valid & ~torch.isfinite(label)).any()):
            raise ValueError('source-qualified label must be finite: '+field)
        if bool(valid.any()):
            data[field] = (label, valid)
    return data


def inspect_heads(rows, split):
    criterion = StereoEstimationLoss()
    result = {}
    for name in HEADS:
        eligible = [bool(_targets(row, name, criterion)) for row in rows]
        counts = {part: sum(eligible[i] for i in split[part+'_indices']) for part in PARTITIONS}
        enabled = all(counts.values())
        result[name] = dict(enabled=enabled, eligible_rows=sum(eligible), eligible_by_split=counts,
            reason=None if enabled else 'missing_verified_force_source_or_group' if name == 'force_proxy' else
            'no_synchronized_source_qualified_labels' if not any(eligible) else 'missing_qualified_labels_in_one_or_more_independent_splits')
        if not enabled:
            result[name].update(optimization=None, splits=None, uncertainty_calibration=None)
    return result


def calibrate_residuals(group_errors, *, coverage=.95):
    """Split-conformal radius using per-group max residuals as observations."""
    scores = sorted(max(errors) for errors in group_errors.values() if len(errors))
    rank = math.ceil((len(scores)+1)*coverage)
    calibrated = 1 <= rank <= len(scores)
    return dict(calibrated=calibrated, independent_validation_groups=len(scores),
        nominal_coverage=coverage, radius=scores[rank-1] if calibrated else None,
        method='split_conformal_group_max_abs_residual')


def _batch(rows, labels, indices, device):
    history = StereoHistoryBatch.cat([StereoHistoryBatch(**rows[i]['history']) for i in indices]).to(device)
    keys = sorted(set().union(*(labels[i] for i in indices)))
    targets = {}
    for key in keys:
        shape = HEAD_SHAPES[key]
        pairs = [labels[i].get(key, (torch.full(shape, float('nan')), torch.zeros(shape,dtype=torch.bool))) for i in indices]
        target, valid = zip(*pairs)
        targets[key] = (torch.stack(target).to(device), torch.stack(valid).to(device))
    return history, targets


def _prediction_loss(prediction, targets, criterion):
    terms, errors = [], {}
    for key, (target, mask) in targets.items():
        pred = prediction[key]
        scalar = mask.clone()
        if key == 'privileged_state':
            classification = criterion.state_classification[None].to(pred.device)
            for group in criterion.rotations:
                lo, hi = group['slice']
                valid = mask[:,lo:hi].all(-1)
                scalar[:,lo:hi] = False
                terms.append(quaternion_geodesic_loss(pred[:,lo:hi], target[:,lo:hi], valid))
            terms.append(masked_smooth_l1(pred, target, scalar & ~classification))
            terms.append(masked_bce(pred, target, scalar & classification))
            pred = torch.where(classification, pred.sigmoid(), pred)
            # Quaternion error is represented by SO(3) radians per valid group,
            # rather than subtracting arbitrary-sign quaternion components.
            angular = []
            for group in criterion.rotations:
                lo, hi = group['slice']
                active = mask[:,lo:hi].all(-1)
                a, b = pred[:,lo:hi][active], target[:,lo:hi][active]
                a = torch.nn.functional.normalize(a, dim=-1)
                b = torch.nn.functional.normalize(b, dim=-1)
                angle = 2*torch.acos((a*b).sum(-1).abs().clamp(max=1))
                values = torch.full((len(mask),1), float('nan'), device=pred.device)
                values[active,0] = angle.detach()
                angular.append(values)
            if angular:
                errors['rotation_rad'] = torch.cat(angular, dim=-1)
        elif key.endswith('_logits'):
            terms.append(masked_bce(pred, target, scalar))
            pred = pred.sigmoid()
        else:
            terms.append(masked_smooth_l1(pred, target, scalar))
        errors[key] = torch.where(scalar, (pred-target).abs(), float('nan')).detach().flatten(1)
    return sum(terms), errors


def _metrics(model, rows, labels, indices, split, batch_size, device, criterion, head):
    residuals = defaultdict(lambda: defaultdict(list))
    losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start+batch_size]
            history, targets = _batch(rows, labels, chunk, device)
            if not targets:
                continue
            value, errors = _prediction_loss(model(history), targets, criterion)
            losses.append(float(value))
            for key, values in errors.items():
                for i, row in enumerate(chunk):
                    group = split['group_for_row'][row]
                    valid = torch.isfinite(values[i])
                    residuals[key][group].extend(values[i][valid].tolist())
                    if key == 'privileged_state':
                        for col, error in zip(valid.nonzero().flatten().tolist(),values[i][valid].tolist()):
                            residuals['371:'+str(col)][group].append(error)
    canonical, extras = HEADS[head]
    widths = {key: math.prod(HEAD_SHAPES[key]) for key in extras}
    units = {'force_proxy':'N', 'rotation_rad':'rad', 'privileged_state':'normalized contract components'}
    if canonical:
        rotation_blocks = {i for group in criterion.rotations for i in range(*group['slice'])}
        fields = [i for i, row in enumerate(criterion.state_contract['dimensions'])
                  if row['estimator_head'] == canonical and bool(criterion.state_allowed[i])]
        widths['privileged_state'] = len(set(fields) - rotation_blocks)
        widths['rotation_rad'] = sum(all(i in fields for i in range(*group['slice'])) for group in criterion.rotations)
        for i in sorted(set(fields)-rotation_blocks):
            key = '371:'+str(i)
            widths[key] = 1
            field = criterion.state_contract['dimensions'][i]
            units[key] = 'normalized: '+field['normalization']
    report = {}
    for key, width in widths.items():
        if not width:
            continue
        groups = residuals.get(key, {})
        numbers = torch.tensor([x for group in groups.values() for x in group], dtype=torch.float64)
        total = width*len(indices)
        unit = units.get(key,'probability' if key.endswith('_logits') else 'source-declared units')
        if not numbers.numel():
            report[key] = dict(mae=None, rmse=None, p95=None, valid_count=0, valid_coverage=0.,
                               total_label_slots=total, unit=unit)
            continue
        report[key] = dict(mae=float(numbers.mean()), rmse=float(numbers.square().mean().sqrt()),
            p95=float(numbers.quantile(.95)), valid_count=len(numbers), valid_coverage=len(numbers)/total,
            total_label_slots=total, unit=unit)
    return dict(mean_batch_loss=sum(losses)/len(losses) if losses else None, outputs=report), residuals


def fit_state_heads(model, rows, split, *, epochs, batch_size, device='cpu', lr=3e-4):
    """Train only enabled ordered stages. Keep shared representation and learned heads."""
    criterion = StereoEstimationLoss()
    admission = inspect_heads(rows, split)
    if not any(head['enabled'] for head in admission.values()):
        return admission
    parts = [split[key+'_indices'] for key in PARTITIONS]
    groups = [{split['group_for_row'][i] for i in part} for part in parts]
    if (sorted(sum(parts, [])) != list(range(len(rows))) or
            any(a & b for n,a in enumerate(groups) for b in groups[n+1:])):
        raise ValueError('state estimator split overlaps a physical initial/source group')
    model.to(device)
    trained = {}
    for name in HEADS:
        record = admission[name]
        if not record['enabled']:
            continue
        labels = [_targets(row, name, criterion) for row in rows]
        indices = [i for i in split['train_indices'] if labels[i]]
        # Fresh optimizer state prevents prior-stage Adam momentum from
        # updating masked rows of the shared 371D projection.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trained[name] = labels
        record['optimization'] = []
        model.train()
        for _ in range(epochs):
            for start in range(0, len(indices), batch_size):
                history, targets = _batch(rows, labels, indices[start:start+batch_size], device)
                optimizer.zero_grad(set_to_none=True)
                value, _ = _prediction_loss(model(history), targets, criterion)
                value.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                output_gradients = {key: math.sqrt(sum(float(p.grad.detach().square().sum())
                    for p in model.heads[key].parameters() if p.grad is not None)) for key in targets}
                optimizer.step()
                record['optimization'].append(dict(loss=float(value.detach()), gradient_norm=float(gradient),
                                                     output_head_gradient_norm=output_gradients))
    for name, labels in trained.items():
        record = admission[name]
        record['splits'] = {}
        residuals = {}
        for part in PARTITIONS:
            metrics, residual = _metrics(model, rows, labels, split[part+'_indices'], split,
                                         batch_size, device, criterion, name)
            record['splits'][part] = metrics
            residuals[part] = residual
        record['uncertainty_calibration'] = {}
        for key, groups in residuals['validation'].items():
            result = calibrate_residuals(groups)
            result['unit'] = record['splits']['validation']['outputs'][key]['unit']
            test = [max(x) for x in residuals['test'].get(key, {}).values() if x]
            result['test_group_coverage'] = (sum(x <= result['radius'] for x in test)/len(test)
                                              if test and result['calibrated'] else None)
            record['uncertainty_calibration'][key] = result
    return admission


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--raw-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=116)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size) < 1:
        parser.error('positive epochs and batch size required')
    rows = read_candidate(args.candidate, raw=args.raw, sha256=args.candidate_sha256, raw_sha256=args.raw_sha256)
    split = split_estimator_rows(rows, seed=args.seed)
    heads = inspect_heads(rows, split)
    out = new_run(args.output)
    torch.manual_seed(args.seed)
    trained = any(value['enabled'] for value in heads.values())
    if trained:
        model = StereoStateEstimator(history=rows[0]['history']['history_valid'].shape[1], point_coordinate_mode='palms')
        heads = fit_state_heads(model, rows, split, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
        with (out/'state_estimator.pt').open('xb') as stream:
            torch.save(dict(estimator=model.cpu().state_dict(), trained_heads=[k for k,v in heads.items() if v['enabled']],
                            candidate_sha256=args.candidate_sha256, raw_sha256=args.raw_sha256), stream)
    report = dict(status='offline_state_estimator_trained' if trained else 'blocked_state_supervision',
        candidate_sha256=args.candidate_sha256, raw_sha256=args.raw_sha256, rows=len(rows), split=split,
        heads=heads, state_training_started=trained, successful_demonstration_pool=False)
    write_json_new(out/'state_supervision_report.json', report)
    return 0 if trained else 2
