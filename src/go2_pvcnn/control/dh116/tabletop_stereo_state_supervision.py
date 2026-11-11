"""Offline state-only training on native synchronized Teacher candidates."""
from argparse import ArgumentParser
from collections import defaultdict
import json
import math
from pathlib import Path
import random

import torch

from .tabletop_stereo_contract import digest, file_sha256, new_run, physical_initial_sha256, require_sha, write_json_new
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
TRAINING_MODES = ('schema_test', 'diagnostic_fit', 'formal_training')
SOURCE_KEYS = ('source_valid', 'identity_aligned', 'timestamp_valid', 'visibility_valid', 'correspondence_valid')
_ENCODERS_ONLY = 'simulation_SDK_encoder_readback_only'
DATASET_VERSION = 'dh116-stereo-state-dataset-v1'
_CENTER_XYZ = slice(61, 64)


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
        if (row['extra_labels'].get('training_label_source_scope') == _ENCODERS_ONLY
                or row['extra_labels'].get('direct_readback_source_scope') == _ENCODERS_ONLY):
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
    def component_identity(indices):
        identities = (rows[i]['identity'] for i in indices)
        return sorted({(item['physical_initial_condition_sha256'], item['source_episode_group'])
                       for item in identities})

    components = sorted(groups.values(), key=component_identity)
    result = dict(independent_groups=len(components), group_for_row={i:k for k, rows in enumerate(components) for i in rows},
                  **{name+'_indices': [] for name in PARTITIONS})
    if len(components) < 3:
        return dict(result, status='blocked_insufficient_independent_groups')
    random.Random(seed).shuffle(components)
    holdout = max(1, len(components)//5)
    for name, chosen in zip(PARTITIONS, (components[2*holdout:], components[:holdout], components[holdout:2*holdout])):
        result[name+'_indices'] = sorted(i for group in chosen for i in group)
    return dict(result, status='ready')


def _check_reset_readback_header(initial, binding):
    """Link native episode metadata to the same cold-reset object's SI state.

    Environment origins are translations, so world XYZ velocities need no
    subtraction/rotation. Root poses are not palms and full-joint vectors are
    not active12; neither is compared by positional coincidence here.
    """
    actual, tolerance = binding['physical_state'], binding['tolerance']
    pose = torch.as_tensor(initial['object_pose'],dtype=torch.float64).clone()
    velocity = torch.as_tensor(initial['object_velocity'],dtype=torch.float64)
    if initial['coordinate_frame'] == 'world':
        pose[:3] -= torch.as_tensor(initial['environment_origin'],dtype=torch.float64)
    elif initial['coordinate_frame'] != 'environment_local':
        raise ValueError('cold reset readback/header coordinate frame unknown')
    expected = torch.as_tensor(actual['wire_pose'],dtype=torch.float64)
    expected_velocity = torch.as_tensor(actual['wire_velocity'],dtype=torch.float64)
    q, target_q = pose[3:]/pose[3:].norm(), expected[3:]/expected[3:].norm()
    quaternion_error = min(float((q-target_q).abs().max()),float((q+target_q).abs().max()))
    if (float((pose[:3]-expected[:3]).abs().max()) > tolerance
            or quaternion_error > tolerance
            or float((velocity-expected_velocity).abs().max()) > tolerance):
        raise ValueError('native episode object pose/velocity differs from same-env cold reset readback')


def _raw_bindings(path, rows, checkpoint, *, cold_readbacks=None):
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
        if cold_readbacks is not None:
            if key[:4] not in cold_readbacks:
                raise ValueError('native episode lacks bound cold reset readback')
            _check_reset_readback_header(header['physical_initial_condition'],cold_readbacks[key[:4]])
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
        if row['extra_labels'].get('training_label_source_scope') == _ENCODERS_ONLY:
            row['extra_labels']['direct_readback_source_scope'] = _ENCODERS_ONLY
        row['privileged_valid'] = row['privileged_valid'] | verified
        for key in SOURCE_KEYS:
            existing = row['state_sources'].get(key,torch.zeros_like(verified))
            row['state_sources'][key] = existing | verified
        row['extra_labels']['training_label_source_scope'] = 'independently_matched_rigid_state_readback'


def _validate_cold_candidate_rows(payload, rows, raw):
    """Only diagnostic candidates may contain explicitly bounded cold resets.

    Formal BC/DAgger validate_rows and complete-success gates are unchanged.
    A boolean reset exception is insufficient: bind every epoch to the saved
    manifest, exclusive cohort report and actual reset-readback artifacts.
    """
    from .tabletop_privileged_evaluation import prepare_cold_trial_batches
    cohort = payload['cold_cohort']
    root = Path(raw).resolve().parent.parent/'cold_initials'
    paths = [root/'manifest.json', root/'cohort_report.json']
    original = {path:file_sha256(path) for path in paths}
    saved = json.loads(paths[1].read_text(encoding='utf-8'))
    if (saved != cohort or cohort.get('version') != 'dh116-stereo-cold-candidate-capture-v1'
            or cohort.get('native_execution_performed') is not True
            or any(cohort.get(key) is not False for key in ('native_qualified','successful_demonstration_pool','formal_success_evaluation'))
            or cohort.get('reset_buffers_restored') is not True):
        raise ValueError('cold candidate needs its intact, nonqualified bounded cohort report')
    batches = cohort.get('batches', [])
    if not batches or not batches[0].get('initial_identities'):
        raise ValueError('cold candidate initial identities absent')
    count = len(batches[0]['initial_identities'])
    plan = prepare_cold_trial_batches(paths[0],count,minimum_per_style=cohort['minimum_per_style'],schema_only=True)
    meta = plan['manifest']['metadata']
    checkpoint, calibration = payload['teacher_checkpoint_sha256'],rows[0]['identity']['calibration_sha256']
    if (plan['manifest_sha256'] != cohort.get('manifest_sha256') or len(plan['batches']) != len(batches)
            or meta['source_sha256'] != checkpoint or cohort.get('checkpoint_sha256') != checkpoint
            or meta['camera_sha256'] != calibration or cohort.get('calibration_sha256') != calibration
            or cohort.get('samples') != len(rows)):
        raise ValueError('cold candidate manifest, checkpoint, calibration or sample count differs')
    epochs = defaultdict(list)
    for row in rows:
        if row['mode'] != 'teacher_candidate_reference' or row['action_owner'] != 'teacher':
            raise ValueError('bounded cold resets are diagnostic Teacher candidates only')
        epochs[identity_key(row['identity'])[:4]].append(row)
    seen, initials, readbacks = set(), set(), {}
    for index,(batch,trials) in enumerate(zip(batches,plan['batches'])):
        steps = batch.get('completed_control_steps')
        if (batch.get('batch_index') != index or type(steps) is not int
                or not 1 <= steps <= cohort['steps_per_initial'] or len(batch.get('initial_identities',[])) != count):
            raise ValueError('cold candidate batch step budget/identity count differs')
        readpath = root/f'reset_readback_{index:04d}.json'
        original[readpath] = require_sha(batch['actual_readback_sha256'])
        if file_sha256(readpath) != original[readpath]:
            raise ValueError('cold candidate reset readback SHA mismatch')
        readback = json.loads(readpath.read_text(encoding='utf-8'))
        tolerance = readback.get('writer_match_tolerance')
        if (readback.get('physics_steps_advanced') != 0 or len(readback.get('actual_states',[])) != count
                or type(tolerance) not in (int,float) or not 0 < tolerance <= 2e-5
                or not 0 <= readback.get('max_abs_writer_readback_error',float('inf')) <= tolerance):
            raise ValueError('cold candidate needs complete matching reset readback without physics advance')
        last_done = False
        for slot,(initial,trial,actual) in enumerate(zip(batch['initial_identities'],trials,readback['actual_states'])):
            key = identity_key(initial)[:4]
            segment = epochs.get(key, [])
            initial_sha = require_sha(initial.get('physical_initial_condition_sha256'))
            if (initial['env_id'] != slot or key in seen or initial_sha in initials or len(segment) != steps or not segment
                    or segment[0]['identity'] != initial or initial['source_episode_group'] != trial['trial_id']
                    or initial.get('trial_id') != trial['trial_id'] or initial.get('grasp_style') != trial['grasp_style']
                    or actual.get('trial_id') != trial['trial_id'] or actual.get('authored_state_sha256') != trial['state_sha256']):
                raise ValueError('cold candidate epoch differs from unique manifest/readback initial identity')
            for name in ('joint_positions','joint_velocities','roots','root_velocities','wire_pose','wire_velocity'):
                observed = torch.as_tensor(actual['physical_state'][name],dtype=torch.float64)
                authored = torch.as_tensor(trial['physical_state'][name],dtype=torch.float64)
                if (observed.shape != authored.shape or not torch.isfinite(observed).all()
                        or float((observed-authored).abs().max()) > tolerance):
                    raise ValueError('cold candidate actual reset state differs from authored manifest')
            history = StereoHistoryBatch(**segment[0]['history']).validate()
            if int(history.history_valid.sum()) != 1 or not bool((history.proprio[0,-1,38:] == 0).all()):
                raise ValueError('cold candidate reset contains prior-episode history or action')
            validate_rows(segment,contract={'calibration_sha256':calibration},schema_only=False)
            last_done |= bool(segment[-1]['done'])
            seen.add(key);initials.add(initial_sha)
            readbacks[key] = dict(physical_state=actual['physical_state'],tolerance=tolerance)
        if bool(batch.get('stopped_on_terminal')) != last_done or (not last_done and steps != cohort['steps_per_initial']):
            raise ValueError('cold candidate batch must stop exactly at terminal or explicit bounded horizon')
    if set(epochs) != seen or len(initials) != cohort.get('independent_actual_initials'):
        raise ValueError('cold candidate has unbound episodes or duplicate initial states')
    previous = {}
    for row in rows:
        identity = row['identity'];key = (identity['run_uuid'],identity['env_id'])
        old = previous.get(key)
        if old is not None:
            before = old['identity']
            if (identity['policy_tick'] <= before['policy_tick'] or identity['frame_id'] <= before['frame_id']
                    or identity['simulation_timestamp'] <= before['simulation_timestamp']
                    or identity['physics_tick_start'] < old['execution']['physics_tick_end']
                    or (identity['episode_uid'] != before['episode_uid'] and identity['reset_id'] <= before['reset_id'])):
                raise ValueError('cold candidate global execution clocks/reset IDs are not ordered')
        previous[key] = row
    if any(file_sha256(path) != sha for path,sha in original.items()):
        raise ValueError('cold candidate identity artifacts changed during admission')
    return readbacks


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
    cold_readbacks = None
    if payload.get('cold_cohort') is not None:
        cold_readbacks = _validate_cold_candidate_rows(payload,rows,raw)
    else:
        validate_rows(rows, contract={'calibration_sha256': rows[0]['identity']['calibration_sha256']}, schema_only=False)
    for row in rows:
        if row['extra_labels'].get('training_label_evidence_scope') != 'native_runtime_capture_unqualified':
            raise ValueError('candidate lacks measured label-source provenance')
        row['source_teacher_checkpoint_sha256'] = checkpoint
    same_clock = _raw_bindings(raw, rows, checkpoint, cold_readbacks=cold_readbacks)
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


def _dataset_observation_binding(row):
    """Check archived processing identity without requiring today's source SHA."""
    from .tabletop_stereo_contract import validate_identity
    from .tabletop_stereo_depth import StereoDepth
    from .tabletop_stereo_preprocessing import VERSION as PROCESSING_VERSION

    identity = validate_identity(row['identity'])
    if row['privileged_identity'] != identity:
        raise ValueError('privileged and camera pre-action binding differs')
    if identity['physics_tick_start'] != identity['physics_tick_end']:
        raise ValueError('dataset requires one pre-action physical tick')
    calibration = identity['calibration_sha256']
    if calibration == '0'*64:
        raise ValueError('nonzero camera calibration SHA required')
    reference = identity.get('state_reference_artifact')
    if (not isinstance(reference, dict) or not reference.get('path')
            or require_sha(reference.get('sha256'), 'reference SHA') == '0'*64):
        raise ValueError('same-frame label-only reference binding required')
    processing = identity.get('observation_preprocessor')
    if not isinstance(processing, dict) or processing.get('version') != PROCESSING_VERSION:
        raise ValueError('archived RGB processing contract required')
    sha = require_sha(identity.get('observation_preprocessor_sha256'), 'RGB processing SHA')
    if processing.get('processing_sha256') != sha or digest({
            key: value for key, value in processing.items() if key != 'processing_sha256'}) != sha:
        raise ValueError('archived processing SHA mismatch')
    history = row['history']
    width, height = history['left_rgb'].shape[-1], history['left_rgb'].shape[-2]
    if (processing.get('calibration_sha256') != calibration
            or processing.get('image_resolution_wh') != [width, height]):
        raise ValueError('archived RGB processing calibration or resolution differs')
    sources = processing.get('source_sha256')
    if (processing.get('native_qualified') is not False or not isinstance(sources, dict)
            or require_sha(sources.get('tabletop_stereo_depth.py'), 'RGB depth processing source SHA') == '0'*64
            or processing.get('geometry', {}).get('missing') !=
            'NaN_with_false_validity;no_simulator_or_rendered_depth_fallback'):
        raise ValueError('RGB depth processing provenance lacks no-reference-input contract')
    depth_config = processing.get('depth_estimator')
    if not isinstance(depth_config, dict) or not isinstance(depth_config.get('method'), str):
        raise ValueError('archived RGB depth processing method required')
    try:
        StereoDepth.from_config(depth_config)
    except (TypeError, ValueError) as error:
        raise ValueError('unsupported archived RGB depth processing contract') from error
    valid = history['depth_valid'][0, -1]
    disparity = row['extra_labels'].get('stereo_disparity_px')
    recorded = row['label_validity'].get('stereo_disparity_px')
    if (not isinstance(disparity, torch.Tensor) or disparity.shape != valid.shape
            or not disparity.is_floating_point() or not isinstance(recorded, torch.Tensor)
            or recorded.dtype != torch.bool or not torch.equal(recorded, valid)
            or not bool((torch.isfinite(disparity[valid]) & (disparity[valid] > 0)).all())):
        raise ValueError('RGB disparity and depth validity binding differs')
    return calibration, sha, tuple(history['left_rgb'].shape[1:])


def read_candidate_dataset(manifest, *, expected_sha256):
    """Load immutable native candidate/ledger pairs and keep physical holdouts global."""
    manifest = Path(manifest).resolve(strict=True)
    expected = require_sha(expected_sha256, 'dataset manifest SHA')
    if file_sha256(manifest) != expected:
        raise ValueError('dataset manifest SHA mismatch')
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    if (not isinstance(payload, dict) or set(payload) != {'version', 'sources'}
            or payload['version'] != DATASET_VERSION
            or not isinstance(payload['sources'], list) or not payload['sources']):
        raise ValueError('versioned state dataset manifest with nonempty source pairs required')
    rows, source_audit = [], []
    source_digests, seen_runs = set(), set()
    dataset_contract = None
    for source in payload['sources']:
        if (not isinstance(source, dict) or set(source) != {'candidate','candidate_sha256','raw','raw_sha256'}
                or any(not isinstance(source[key], str) or not source[key] for key in ('candidate', 'raw'))):
            raise ValueError('candidate/raw paths and SHA bindings required')
        candidate_sha = require_sha(source['candidate_sha256'], 'candidate SHA')
        raw_sha = require_sha(source['raw_sha256'], 'raw ledger SHA')
        if candidate_sha in source_digests or raw_sha in source_digests or candidate_sha == raw_sha:
            raise ValueError('duplicate dataset source bytes')
        source_digests.update((candidate_sha, raw_sha))
        cohort = read_candidate(manifest.parent/source['candidate'], raw=manifest.parent/source['raw'],
                                sha256=candidate_sha, raw_sha256=raw_sha)
        runs, physical_initials = set(), set()
        for row in cohort:
            binding = _dataset_observation_binding(row)
            if dataset_contract is None:
                dataset_contract = binding
            if binding != dataset_contract:
                raise ValueError('dataset sources must share calibration, processing and history shape')
            identity = row['identity']
            runs.add(identity['run_uuid'])
            physical_initials.add(identity['physical_initial_condition_sha256'])
        if seen_runs & runs:
            raise ValueError('duplicate capture run across candidate bundles')
        seen_runs.update(runs)
        rows.extend(cohort)
        checkpoint = require_sha(cohort[0]['source_teacher_checkpoint_sha256'], 'Teacher checkpoint SHA')
        source_audit.append(dict(candidate_sha256=candidate_sha, raw_sha256=raw_sha,
            teacher_checkpoint_sha256=checkpoint, rows=len(cohort), run_uuids=sorted(runs),
            physical_initial_condition_sha256=sorted(physical_initials)))
    if file_sha256(manifest) != expected:
        raise ValueError('dataset manifest changed during admission')
    calibration, processing, _ = dataset_contract
    return rows, dict(manifest_sha256=expected, sources=source_audit, calibration_sha256=calibration,
        observation_preprocessor_sha256=processing, source_scope='verified_native_capture_replayed_offline',
        successful_demonstration_pool=False)


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
    return {name:_admission(name, [_targets(row, name, criterion) for row in rows], split) for name in HEADS}


def _admission(name, labels, split):
    eligible = [bool(label) for label in labels]
    counts = {part: sum(eligible[i] for i in split[part+'_indices']) for part in PARTITIONS}
    enabled = all(counts.values())
    result = dict(enabled=enabled, eligible_rows=sum(eligible), eligible_by_split=counts,
        reason=None if enabled else 'missing_verified_force_source_or_group' if name == 'force_proxy' else
        'no_synchronized_source_qualified_labels' if not any(eligible) else
        'missing_qualified_labels_in_one_or_more_independent_splits')
    if not enabled:
        result.update(optimization=None, splits=None, uncertainty_calibration=None)
    return result


def _geometry_stages(labels):
    center_rows, wire_rows = [], []
    for label in labels:
        center, wire = {}, {key: value for key, value in label.items() if key != 'privileged_state'}
        if 'privileged_state' in label:
            target, valid = label['privileged_state']
            center_valid = torch.zeros_like(valid)
            center_valid[_CENTER_XYZ] = valid[_CENTER_XYZ]
            wire_valid = valid & ~center_valid
            if center_valid.any():
                center['privileged_state'] = target, center_valid
            if wire_valid.any():
                wire['privileged_state'] = target, wire_valid
        center_rows.append(center)
        wire_rows.append(wire)
    return {'center_xyz':center_rows, 'wire_geometry':wire_rows}


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
        if key in ('371:61', '371:62', '371:63'):
            # These original contract coordinates are relative center / 0.05 m.
            report[key].update(physical_mae_m=report[key]['mae']*.05,
                physical_rmse_m=report[key]['rmse']*.05, physical_p95_m=report[key]['p95']*.05)
    return dict(mean_batch_loss=sum(losses)/len(losses) if losses else None, outputs=report), residuals


def _optimize_head(model, rows, labels, indices, *, epochs, batch_size, device, lr, criterion):
    """Shared masked optimization for diagnostic and grouped formal training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    updates = []
    model.train()
    for epoch in range(epochs):
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start+batch_size]
            history, targets = _batch(rows, labels, chunk, device)
            optimizer.zero_grad(set_to_none=True)
            value, _ = _prediction_loss(model(history), targets, criterion)
            value.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            output_gradients = {key: math.sqrt(sum(float(p.grad.detach().square().sum())
                for p in model.heads[key].parameters() if p.grad is not None)) for key in targets}
            encoder_gradient = math.sqrt(sum(float(p.grad.detach().square().sum())
                for p in model.encoder.parameters() if p.grad is not None))
            unselected_max = None
            if 'privileged_state' in targets:
                selected = targets['privileged_state'][1].any(0)
                grad = model.heads['privileged_state'].weight.grad
                unselected_max = float(grad[~selected].abs().max()) if (~selected).any() else 0.
            optimizer.step()
            updates.append(dict(epoch=epoch, row_indices=chunk, loss=float(value.detach()),
                gradient_norm=float(gradient), output_head_gradient_norm=output_gradients,
                encoder_gradient_norm=encoder_gradient, unsupervised_output_gradient_max=unselected_max))
    return updates


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
        if not record['enabled'] and name != 'geometry':
            continue
        labels = [_targets(row, name, criterion) for row in rows]
        if name == 'geometry':
            stages = _geometry_stages(labels)
            record['stages'] = {key:_admission(key, stage, split) for key, stage in stages.items()}
            record['enabled'] = any(stage['enabled'] for stage in record['stages'].values())
            if not record['enabled']:
                record.update(reason='no_geometry_stage_with_independent_labels', optimization=None,
                              splits=None, uncertainty_calibration=None)
                continue
            optimized = []
            merged_labels = [{} for _ in rows]
            for key, stage_labels in stages.items():
                stage = record['stages'][key]
                if not stage['enabled']:
                    continue
                indices = [i for i in split['train_indices'] if stage_labels[i]]
                stage['optimization'] = _optimize_head(model, rows, stage_labels, indices, epochs=epochs,
                    batch_size=batch_size, device=device, lr=lr, criterion=criterion)
                masks = [label['privileged_state'][1] for label in stage_labels if 'privileged_state' in label]
                stage['supervised_indices'] = torch.stack(masks).any(0).nonzero().flatten().tolist() if masks else []
                optimized.extend(stage['optimization'])
                for combined, label in zip(merged_labels, stage_labels):
                    for field, (target, mask) in label.items():
                        if field in combined:
                            mask = combined[field][1] | mask
                        combined[field] = target, mask
            trained[name] = merged_labels
            record['optimization'] = optimized
        else:
            indices = [i for i in split['train_indices'] if labels[i]]
            # Fresh Adam state prevents prior-stage momentum from updating
            # masked rows of the shared privileged-state projection.
            trained[name] = labels
            record['optimization'] = _optimize_head(model, rows, labels, indices, epochs=epochs,
                batch_size=batch_size, device=device, lr=lr, criterion=criterion)
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


def run_training_mode(model, rows, *, mode='formal_training', epochs=1, batch_size=4, device='cpu', lr=3e-4, seed=116):
    """Separate schema, in-sample real-label debugging and grouped training.

    CLI callers first reopen candidate/raw/reference SHA bindings with read_candidate.
    This lower-level tensor function is also exercised with explicit CPU fixtures.
    A diagnostic fit never opens an Actor, expert, uncertainty or deployment gate.
    """
    if mode not in TRAINING_MODES:
        raise ValueError('unknown state training mode')
    if not rows or min(epochs, batch_size) < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('nonempty rows and positive finite training settings required')
    split = split_estimator_rows(rows, seed=seed)
    heads = inspect_heads(rows, split)
    report = dict(mode=mode, rows=len(rows), split=split, heads=heads, state_training_started=False,
        generalization_qualified=False, actor_deployment_qualified=False, uncertainty_calibrated=False,
        successful_demonstration_pool=False)
    if mode == 'schema_test':
        for row in rows:
            StereoHistoryBatch(**row['history']).validate()
            if row['privileged'].shape != (371,) or row['privileged_valid'].shape != (371,):
                raise ValueError('original 371D target and mask required')
        for head in heads.values():
            head.update(enabled=False, reason='schema_test_no_optimization')
        return dict(report, status='schema_validated')
    if mode == 'formal_training':
        criterion = StereoEstimationLoss()
        geometry_labels = [_targets(row, 'geometry', criterion) for row in rows]
        stages = {**{key:_admission(key, labels, split) for key, labels in
            _geometry_stages(geometry_labels).items()},
            **{key:value for key,value in heads.items() if key != 'geometry'}}
        if split['status'] != 'ready':
            return dict(report, status=split['status'], stages=stages)
        heads = fit_state_heads(model, rows, split, epochs=epochs, batch_size=batch_size, device=device, lr=lr)
        stages.update(heads['geometry'].get('stages', {}))
        stages.update({key:value for key,value in heads.items() if key != 'geometry'})
        trained = any(x['enabled'] for x in heads.values())
        return dict(report, heads=heads, status='formal_state_fit_completed' if trained else 'blocked_state_supervision',
                    state_training_started=trained, stages=stages)
    criterion = StereoEstimationLoss()
    labels = []
    audit = []
    for i, row in enumerate(rows):
        data = {}
        if (isinstance(row.get('verified_canonical_native'), torch.Tensor)
                and row['extra_labels'].get('training_label_source_scope') == 'independently_matched_rigid_state_readback'):
            target = _targets(row, 'geometry', criterion).get('privileged_state')
            if target is not None:
                values, mask = target
                mask = mask.clone()
                mask[:61] = False
                mask[64:] = False
                if mask.any():
                    data['privileged_state'] = values, mask
                    audit.append(dict(row_index=i, identity=row['identity'], indices=[61,62,63],
                        valid=mask[61:64].tolist(), unit='m', normalization='relative_center / 0.05 m',
                        coordinate_frame='world_axes_relative_to_mean_of_left_and_right_wrist_origins',
                        target_normalized=[float(v) if ok else None for v,ok in zip(values[61:64],mask[61:64])],
                        target_m=[float(v)*.05 if ok else None for v,ok in zip(values[61:64],mask[61:64])]))
        labels.append(data)
    indices = [i for i, data in enumerate(labels) if data]
    for head in heads.values():
        head.update(enabled=False, reason='diagnostic_center_xyz_only', optimization=None, splits=None,
                    uncertainty_calibration=None)
    if not indices:
        return dict(report, status='blocked_no_verified_center_labels')
    model.to(device)
    record = heads['geometry']
    record.update(enabled=True, reason=None, eligible_rows=len(indices),
        supervised_indices=torch.stack([labels[i]['privileged_state'][1] for i in indices]).any(0).nonzero().flatten().tolist(),
        metrics_scope='in_sample_diagnostic_only', label_audit=audit,
        eligible_by_split=dict(train=len(indices),validation=0,test=0),
        uncertainty_calibration=dict(calibrated=False, radius=None, reason='no_independent_validation_in_diagnostic_fit'))
    record['before_fit'], _ = _metrics(model, rows, labels, indices, split, batch_size, device, criterion, 'geometry')
    record['optimization'] = _optimize_head(model, rows, labels, indices, epochs=epochs,
        batch_size=batch_size, device=device, lr=lr, criterion=criterion)
    metrics, _ = _metrics(model, rows, labels, indices, split, batch_size, device, criterion, 'geometry')
    record['splits'] = dict(train=metrics, validation=None, test=None)
    report['split'] = dict(split, diagnostic_fit_indices=indices, diagnostic_split='in_sample_only_no_holdouts')
    return dict(report, status='diagnostic_fit_completed', state_training_started=True)


def read_training_config(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    allowed = {'mode','epochs','batch_size','lr','seed','image_processing_batch_size'}
    settings = value.get('settings')
    if (value.get('version') != 'dh116-stereo-state-training-mode-v1' or not isinstance(settings,dict)
            or not set(settings) <= allowed or settings.get('mode') not in TRAINING_MODES):
        raise ValueError('explicit training mode and allowed configuration settings required')
    return settings


def main(argv=None):
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--candidate', type=Path)
    sources.add_argument('--dataset-manifest', type=Path)
    parser.add_argument('--dataset-manifest-sha256')
    parser.add_argument('--candidate-sha256')
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--raw-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=116)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mode', choices=TRAINING_MODES, default='formal_training')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--image-processing-batch-size', type=int, default=1)
    if config_args.config:
        parser.set_defaults(**read_training_config(config_args.config))
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size) < 1:
        parser.error('positive epochs and batch size required')
    single_source = (args.raw, args.candidate_sha256, args.raw_sha256)
    if args.dataset_manifest is not None:
        if not args.dataset_manifest_sha256 or any(single_source):
            parser.error('dataset manifest requires its SHA and excludes single-source arguments')
        rows, evidence = read_candidate_dataset(args.dataset_manifest, expected_sha256=args.dataset_manifest_sha256)
        provenance = dict(dataset_manifest_sha256=evidence['manifest_sha256'], dataset_sources=evidence['sources'],
                          calibration_sha256=evidence['calibration_sha256'],
                          observation_preprocessor_sha256=evidence['observation_preprocessor_sha256'])
    else:
        if not all(single_source) or args.dataset_manifest_sha256 is not None:
            parser.error('candidate requires --raw, --candidate-sha256 and --raw-sha256')
        rows = read_candidate(args.candidate, raw=args.raw, sha256=args.candidate_sha256, raw_sha256=args.raw_sha256)
        provenance = dict(candidate_sha256=args.candidate_sha256, raw_sha256=args.raw_sha256)
    out = new_run(args.output)
    torch.manual_seed(args.seed)
    model = StereoStateEstimator(history=rows[0]['history']['history_valid'].shape[1], point_coordinate_mode='palms',
        image_processing_batch_size=args.image_processing_batch_size)
    report = run_training_mode(model, rows, mode=args.mode, epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, seed=args.seed, lr=args.lr)
    if report['state_training_started']:
        with (out/'state_estimator.pt').open('xb') as stream:
            torch.save(dict(version='dh116-stereo-state-fit-v2', mode=args.mode,
                            estimator=model.cpu().state_dict(), trained_heads=[k for k,v in report['heads'].items() if v['enabled']],
                            generalization_qualified=False, actor_deployment_qualified=False,
                            successful_demonstration_pool=False,
                            state_contract_sha256=TaskStateAdapter().state_contract_sha256,
                            model_config=dict(embed_dim=128,num_layers=2,history=model.history,point_coordinate_mode='palms'),
                            **provenance), stream)
        report['checkpoint_sha256'] = file_sha256(out/'state_estimator.pt')
    report.update(**provenance, source_scope='verified_native_capture_replayed_offline', seed=args.seed, lr=args.lr,
        implementation_sha256=file_sha256(Path(__file__)), epochs=args.epochs, batch_size=args.batch_size,
        config_sha256=file_sha256(args.config) if args.config else None)
    write_json_new(out/'state_supervision_report.json', report)
    return 0 if report['state_training_started'] or args.mode == 'schema_test' else 2
