"""Native evidence/Teacher/Stereo/Student promotion, independent evaluation/holdouts and training provenance."""
from copy import deepcopy as stere_admis__deepcopy
import copy as stere_evalu__copy
import copy as stere_promo__copy
from copy import deepcopy as stere_train_linea__deepcopy
from dh116.physics.vision import processing_contract
import hashlib as stere_evalu__hashlib
import hashlib as stere_promo__hashlib
import argparse

def require_qualified_stereo_processing(verified_stereo, camera_config):
    actual = processing_contract(camera_config)
    if not isinstance(verified_stereo, dict) or verified_stereo.get('calibration_sha256') != actual['calibration_sha256']:
        raise ValueError('formal stereo proof calibration differs from selected camera')
    measured = stere_admis__deepcopy(verified_stereo.get('depth_estimator_config'))
    if not isinstance(measured, dict):
        raise ValueError('formal matcher differs from verified raw stereo processing')
    measured.pop('image_processing_batch_size', None)
    if measured != actual['depth_estimator']:
        raise ValueError('formal matcher differs from verified raw stereo processing')
    return actual
import json as stere_evalu__json
import json as stere_promo__json
import json as stere_train_linea__json
import math as stere_evalu__math
import math as stere_promo__math
import math as stere_train_linea__math
from pathlib import Path as stere_evalu__Path
from pathlib import Path as stere_promo__Path
from pathlib import Path as stere_train_linea__Path
import numpy as stere_evalu__np
import numpy as stere_promo__np
from dh116.env.contracts import stere_contr__digest as stere_train_linea__digest
from dh116.env.contracts import STYLES as stere_evalu__STYLES
from dh116.env.contracts import stere_contr__ROOT as ROOT
from dh116.env.contracts import stere_contr__file_sha256 as file_sha256
from dh116.env.contracts import action_sha256 as stere_evalu__action_sha256
from dh116.env.contracts import STYLES as stere_promo__STYLES
from dh116.env.contracts import physical_initial_sha256 as stere_train_linea__physical_initial_sha256
from dh116.env.contracts import stere_contr__digest as stere_evalu__digest
from dh116.env.contracts import action_sha256 as stere_promo__action_sha256
from dh116.env.contracts import require_sha as stere_train_linea__require_sha
from dh116.env.contracts import state_contract as stere_evalu__state_contract
from dh116.env.contracts import stere_contr__digest as stere_promo__digest
from dh116.env.contracts import write_json_new
from dh116.env.contracts import physical_initial_sha256 as stere_evalu__physical_initial_sha256
from dh116.env.contracts import require_sha as stere_promo__require_sha
stere_train_linea__VERSION = 'dh116-training-source-manifest-v1'
stere_evalu__VERSION = 'dh116-stereo-evaluation-v1'
from dh116.env.contracts import state_contract as stere_promo__state_contract
_FORBIDDEN = {'privileged', 'teacher_state_dict', 'teacher_weights', 'teacher_policy', 'critic', 'critic_state_dict', 'optimizer', 'policy', 'state_dict', 'weights', 'ground_truth', 'teacher_network'}
FORMAL_TRACKER = 'surface_lift_success_30mm_5s_0.5mm'
from dh116.env.contracts import physical_initial_sha256 as stere_promo__physical_initial_sha256

def validate_training_provenance(value, *, require_artifacts=False):
    """Only JSON metadata travels with an Actor; unavailable paths grant nothing."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError('training provenance metadata dictionary required')

    def visit(item):
        if isinstance(item, dict):
            if any((not isinstance(key, str) or key in _FORBIDDEN for key in item)):
                raise ValueError('training provenance must contain metadata, never Teacher/critic state')
            if 'path' in item or 'sha256' in item:
                if not isinstance(item.get('path'), str):
                    raise ValueError('provenance artifact path/SHA metadata required')
                stere_train_linea__require_sha(item.get('sha256'), 'provenance artifact SHA')
                path = stere_train_linea__Path(item['path'])
                if require_artifacts or path.exists():
                    if not path.is_file() or file_sha256(path) != item['sha256']:
                        raise ValueError('training provenance artifact SHA mismatch')
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            if any((isinstance(child, (int, float, list)) and (not isinstance(child, bool)) for child in item)):
                raise ValueError('provenance numeric arrays/tensors are not metadata')
            for child in item:
                visit(child)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError('training provenance must contain JSON metadata, not tensors/networks')
        elif isinstance(item, float) and (not stere_train_linea__math.isfinite(item)):
            raise ValueError('finite provenance metadata required')
    visit(value)
    return stere_train_linea__deepcopy(value)
stere_evalu__THRESHOLDS = dict(net_lift_m=0.03, continuous_hold_s=5.0, formal_anchor_slip_m=0.0005, physics_hz=480, control_hz=60)

def _read_object(value):
    if isinstance(value, dict):
        return (stere_promo__copy.deepcopy(value), stere_promo__Path.cwd())
    if isinstance(value, (str, stere_promo__Path)):
        path = stere_promo__Path(value).resolve()
        try:
            result = stere_promo__json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError('qualification JSON unavailable') from error
        if not isinstance(result, dict):
            raise ValueError('qualification dictionary required')
        return (result, path.parent)
    raise ValueError('qualification must be a dictionary or JSON path')

def _binding(value, base=None):
    declared = value if isinstance(value, dict) else None
    path = stere_train_linea__Path(value['path'] if declared is not None else value)
    if not path.is_absolute():
        path = (stere_train_linea__Path.cwd() if base is None else base) / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError('training artifact is missing: ' + str(path))
    sha = file_sha256(path)
    if declared is not None and sha != stere_train_linea__require_sha(value.get('sha256'), 'training artifact SHA'):
        raise ValueError('training artifact SHA mismatch: ' + str(path))
    return dict(path=str(path), sha256=sha)
OWNERS = ('teacher', 'mpc', 'bc', 'dagger', 'student')

def _artifact(binding, base, *, allow_empty=False):
    if not isinstance(binding, dict):
        raise ValueError('actual artifact path/SHA binding required')
    try:
        stere_promo__require_sha(binding.get('sha256'))
        path = stere_promo__Path(binding['path'])
        path = path if path.is_absolute() else base / path
        raw = path.read_bytes()
        if stere_promo__hashlib.sha256(raw).hexdigest() != binding['sha256']:
            raise ValueError('raw artifact SHA mismatch')
    except (OSError, KeyError, TypeError) as error:
        raise ValueError('raw artifact missing or invalid') from error
    if not raw and (not allow_empty):
        raise ValueError('empty raw artifact is not evidence')
    return (raw, path.resolve())

def _artifacts(value, prefix=''):
    result = []
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            result.append((prefix, value))
        else:
            for (key, child) in value.items():
                result.extend(_artifacts(child, prefix + '.' + key if prefix else key))
    elif isinstance(value, list):
        for child in value:
            result.extend(_artifacts(child, prefix))
    return result

def _finite(value):
    return type(value) in (int, float) and stere_evalu__math.isfinite(value)

def _jsonl(raw):

    def nonfinite(value):
        raise ValueError('nonfinite JSON raw value must be explicit NA/null: ' + value)
    try:
        rows = [stere_promo__json.loads(line, parse_constant=nonfinite) for line in raw.decode('utf-8').splitlines() if line.strip()]
    except (UnicodeError, ValueError) as error:
        raise ValueError('actual raw JSONL evidence required') from error
    if not rows or any((not isinstance(row, dict) for row in rows)):
        raise ValueError('raw evidence record dictionaries required')
    return rows

def _raw_inventory(binding):
    path = stere_train_linea__Path(binding['path'])
    before = file_sha256(path)
    try:
        rows = [stere_train_linea__json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise ValueError('raw episode JSONL cannot be read') from error
    if before != binding['sha256'] or file_sha256(path) != before:
        raise ValueError('raw artifact changed during registration; close producer first')
    episodes = []
    headers = {}
    counts = {}
    reasons = []
    for (number, row) in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError('raw episode dictionary required')
        if row.get('kind') not in ('episode_start', 'sample'):
            raise ValueError('unknown raw episode record kind')
        key = tuple((row.get(name) for name in ('run_uuid', 'env_id', 'episode_uid', 'reset_id')))
        if not isinstance(key[0], str) or not key[0] or type(key[1]) is not int or (key[1] < 0) or (not isinstance(key[2], str)) or (not key[2]) or (type(key[3]) is not int) or (key[3] < 0):
            raise ValueError('raw episode/reset identity required')
        if row['kind'] == 'episode_start':
            if key in headers:
                raise ValueError('duplicate raw episode_start identity')
            actual = stere_train_linea__physical_initial_sha256(row.get('physical_initial_condition'))
            source = row.get('source_episode_group')
            if not isinstance(source, str) or not source:
                reasons.append('raw_source_episode_group_missing:' + key[2])
                source = None
            if row.get('initial_measurement_valid') is not True or row.get('physical_initial_condition_source') != 'actual_sdk_readback':
                reasons.append('raw_initial_measurement_not_actual_readback:' + key[2])
            if row.get('source_scope') != 'native':
                reasons.append('raw_source_not_native:' + key[2])
            episode = dict(run_uuid=key[0], env_id=key[1], episode_uid=key[2], reset_id=key[3], physical_initial_condition_sha256=actual, source_episode_group=source, raw_artifact_sha256=binding['sha256'], header_line=number)
            episodes.append(episode)
            headers[key] = row
            counts[key] = 0
        else:
            if key not in headers:
                raise ValueError('raw sample has no preceding episode_start')
            counts[key] += 1
            if row.get('source_scope') != 'native':
                reasons.append('raw_source_not_native:' + key[2])
    if not episodes:
        reasons.append('raw_episode_headers_missing:' + str(path))
    for (key, count) in counts.items():
        if not count:
            reasons.append('raw_episode_has_no_physical_samples:' + key[2])
    return (episodes, reasons)

def wilson_interval(successes, total, z=1.959963984540054):
    if type(successes) is not int or type(total) is not int or (not 0 <= successes <= total):
        raise ValueError('valid binomial counts required')
    if total == 0:
        return (None, None)
    p = successes / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    radius = z * stere_evalu__math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return (max(0.0, centre - radius), min(1.0, centre + radius))

def _process_output(proof, base, binding, *, binding_base=None):
    """A child raw output must already be frozen by the external observer."""
    (raw, path) = _artifact(binding, base if binding_base is None else binding_base)
    (completion_raw, _) = _artifact(proof.get('completion_artifact'), base)
    outputs = stere_promo__json.loads(completion_raw).get('produced_artifacts')
    if not isinstance(outputs, list) or not any((isinstance(row, dict) and stere_promo__Path(row.get('path', '')).resolve() == path and (row.get('sha256') == binding['sha256']) for row in outputs)):
        raise ValueError('raw output artifact is absent from actual external process completion')
    return (raw, path)

def _model_provenance(binding):
    import torch
    try:
        payload = torch.load(binding['path'], map_location='cpu', weights_only=True)
    except Exception:
        return ({}, ['model_format_has_no_auditable_actor_training_provenance'])
    if not isinstance(payload, dict) or payload.get('version') != 'dh116-stereo-rgb-deployment-actor-v1':
        return ({}, ['model_format_has_no_auditable_actor_training_provenance'])
    from dh116.rl.student_data import load_deployment_actor
    actor = load_deployment_actor(binding['path'])
    provenance = validate_training_provenance(actor.deployment_contract.get('training_provenance'), require_artifacts=True)
    reasons = [] if provenance else ['actor_training_provenance_missing']
    return (provenance, reasons)

def _distribution(values, count):
    valid = [float(x) for x in values if _finite(x)]
    if not valid:
        return dict(p50=None, p90=None, p95=None, max=None, coverage=0.0, count=0)
    return dict(zip(('p50', 'p90', 'p95', 'max'), map(float, [*stere_evalu__np.quantile(valid, [0.5, 0.9, 0.95]), max(valid)])), coverage=len(valid) / max(count, 1), count=len(valid))

def _native_run(proof, base):
    if proof.get('source_scope') != 'native' or proof.get('schema_only') is not False:
        raise ValueError('native source scope required; CPU/schema fixture cannot qualify')
    if type(proof.get('process_exit_code')) is not int or proof['process_exit_code'] != 0:
        raise ValueError('completed exit0 native process required')
    (completion_raw, completion_path) = _artifact(proof.get('completion_artifact'), base)
    completion = stere_promo__json.loads(completion_raw)
    if completion.get('version') != 'dh116-external-native-completion-v1' or completion.get('source_scope') != 'native_process_exit_observed' or type(completion.get('returncode')) is not int or (completion['returncode'] != 0) or (completion.get('source_changed_during_run') is not False):
        raise ValueError('actual external native process exit0 evidence required')
    (_, runner) = _artifact(completion.get('runner_source_artifact'), completion_path.parent)
    command = completion.get('command')
    for name in ('stdout_artifact', 'stderr_artifact'):
        _artifact(completion.get(name), completion_path.parent, allow_empty=True)
    target = proof.get('capture_artifact') if proof.get('version') == 'dh116-native-stereo-proof-v1' else proof.get('raw_artifact')
    _process_output(proof, base, target)
    if proof.get('action_contract_sha256') != stere_promo__action_sha256() or proof.get('state_contract_sha256') != stere_promo__digest(stere_promo__state_contract()):
        raise ValueError('native source proof current action/state identity mismatch')
    producer = proof.get('producer')
    if not isinstance(producer, dict):
        raise ValueError('actual producer source/config/kwargs identity required')
    (_, source) = _artifact(producer.get('source_artifact'), base)
    stereo = proof.get('version') == 'dh116-native-stereo-proof-v1'
    if producer.get("api_contract") != "dh116-native-observation-producer-v2":
        raise ValueError("current native producer API contract required; old producer reports cannot qualify")
    entrypoint = current_producer_api(runner,source,command,stereo=stereo)
    (config_raw, _) = _process_output(proof, base, producer.get('config_artifact'))
    try:
        config = stere_promo__json.loads(config_raw)
    except ValueError as error:
        raise ValueError('native producer config must be real JSON') from error
    kwargs = producer.get('kwargs')
    if not isinstance(kwargs, dict) or kwargs.get('physics_hz') != 480 or kwargs.get('control_hz') != 60 or (producer.get('kwargs_sha256') != stere_promo__digest(kwargs)):
        raise ValueError('native 480Hz/60Hz producer kwargs contract mismatch')
    if not isinstance(config, dict):
        raise ValueError('native source config dictionary required')
    if config.get('resolved_kwargs') != kwargs or producer.get('entrypoint') != entrypoint:
        raise ValueError('producer entrypoint/resolved kwargs must match the actual config artifact')
    return producer

def _dataset_sources(binding):
    """Extract references from actual registered tensors, never submitted lists."""
    import torch
    try:
        value = torch.load(binding['path'], map_location='cpu', weights_only=True)
    except Exception:
        return ([], [], [])
    if not isinstance(value, dict) or value.get('version') != 'dh116-stereo-supervision-v1':
        return ([], [], [])
    paths = []
    identities = []
    reasons = []
    if value.get('schema_only') is not False:
        reasons.append('training_dataset_is_schema_only:' + binding['sha256'])
    for row in value.get('rows', []):
        path = row.get('info', {}).get('native_evidence_file')
        if not isinstance(path, str) or not path:
            reasons.append('training_dataset_row_has_no_raw_episode_artifact')
            continue
        paths.append(path)
        identity = row.get('identity', {})
        identities.append((tuple((identity.get(key) for key in ('run_uuid', 'env_id', 'episode_uid', 'reset_id'))), identity.get('physical_initial_condition_sha256'), identity.get('source_episode_group')))
    return (paths, identities, reasons)

def _key(row):
    return tuple((row.get(k) for k in ('run_uuid', 'env_id', 'episode_uid', 'reset_id')))

def _binding_proof(binding, version):
    if not isinstance(binding, dict) or not isinstance(binding.get('proof_path'), str):
        raise ValueError('actual immutable native proof_path/proof_sha256 required')
    (raw, path) = _artifact(dict(path=binding['proof_path'], sha256=binding.get('proof_sha256')), stere_promo__Path.cwd())
    try:
        proof = stere_promo__json.loads(raw)
    except ValueError as error:
        raise ValueError('native source proof JSON required') from error
    if not isinstance(proof, dict) or proof.get('version') != version:
        raise ValueError('unsupported native source proof version')
    _native_run(proof, path.parent)
    return (proof, path.parent)

def _unique(bindings):
    result = {}
    for binding in bindings:
        old = result.get(binding['path'])
        if old is not None and old != binding:
            raise ValueError('same training artifact path has conflicting SHA')
        result[binding['path']] = binding
    return [result[path] for path in sorted(result)]

def replay_formal_tracker(samples):
    """Replay the unchanged real 60Hz tracker from every 480Hz physical sample.

    Pending first-window flags are not completed evidence. Loads/normals are
    dt-averaged by SurfaceLiftSuccess; pose slip remains the per-window RAW max
    relative to the fixed hold anchor. This CPU arithmetic grants no native scope.
    """
    import torch
    from dh116.hand.safety import SurfaceLiftSuccess
    mapping = {'object_pose': 'formal_object_pose_w', 'wrist_pose': 'formal_wrist_pose_w', 'loads': 'formal_pad_loads_n', 'vectors': 'formal_pad_vectors_w', 'depth': 'formal_pad_depth_m', 'ordinary': 'formal_rejected_load_n', 'bottom': 'formal_object_bottom_m', 'object_velocity': 'formal_object_velocity'}
    required = set(mapping.values()) | {'formal_reference_bottom_m', 'formal_task_active'}
    if not samples or any((not required <= row.keys() for row in samples)):
        return dict(available=False, event=False, windows=[], reasons=['formal_raw_replay_fields_missing'])
    reference = samples[0]['formal_reference_bottom_m']
    if not _finite(reference):
        return dict(available=False, event=False, windows=[], reasons=['formal_reference_bottom_NA'])
    tracker = SurfaceLiftSuccess(1, 'cpu', reference)
    windows = []
    reasons = []
    event = False
    try:
        for row in samples:
            if row['formal_reference_bottom_m'] != reference or type(row['formal_task_active']) is not bool:
                raise ValueError('changing formal reference/active source')
            data = {name: torch.as_tensor(stere_evalu__np.asarray(row[source], dtype=float), dtype=torch.float64)[None] for (name, source) in mapping.items()}
            result = tracker.update(**data, valid=torch.tensor([row['formal_task_active']]), dt=1 / 480)
            event = bool(result['event'][0])
            if row.get('formal_window_completed') is not result['updated']:
                reasons.append('formal_completed_window_source_mismatch')
            if row.get('formal_window_good') is not bool(tracker.window_good[0]):
                reasons.append('formal_window_good_source_mismatch')
            if row.get('formal_anchor_latched') is not bool(tracker._latched[0]):
                reasons.append('formal_anchor_source_mismatch')
            if row.get('formal_drift_source') != 'last_completed_window_max_over_raw_pose_points':
                reasons.append('formal_drift_source_unavailable')
            if not _finite(row.get('formal_hold_s')) or not stere_evalu__math.isclose(row['formal_hold_s'], float(tracker.duration[0]), rel_tol=1e-08, abs_tol=1e-10):
                reasons.append('formal_duration_differs_from_raw_replay')
            if result['updated']:
                if row.get('formal_tracker_valid') is not bool(tracker._physical_window_valid[0]):
                    reasons.append('formal_window_physical_validity_source_mismatch')
                if not _finite(row.get('formal_drift_m')) or not stere_evalu__math.isclose(row['formal_drift_m'], float(tracker.drift_m[0]), rel_tol=1e-07, abs_tol=1e-10):
                    reasons.append('formal_drift_differs_from_raw_pose_max')
                windows.append(dict(good=bool(tracker.window_good[0]), drift_m=float(tracker.drift_m[0]), duration=float(tracker.duration[0]), paired=tracker._paired[0].tolist(), physical_valid=bool(tracker._physical_window_valid[0])))
    except (ValueError, TypeError, RuntimeError, KeyError) as error:
        return dict(available=False, event=False, windows=windows, reasons=['formal_raw_replay_invalid:' + str(error)])
    return dict(available=True, event=event, windows=windows, reasons=sorted(set(reasons)))

def _verify_force_qualification(binding):
    """No force loss/reward is authorized by a native_qualified string alone.

    A source proof must include independent raw per-contact object-side vectors,
    resolved partners and complete external-contact/force/attachment audit. The
    existing NativeTensorBridge emits unqualified telemetry and cannot satisfy
    this protocol yet. Missing source capabilities remain NA.
    """
    (proof, base) = _binding_proof(binding, 'dh116-force-source-proof-v1')
    required = dict(force_side='on_object', units='N', coordinate_frame='world', partner_attribution='resolved_raw_contact_body_ids')
    if any((proof.get(k) != v for (k, v) in required.items())):
        raise ValueError('force side/SI/frame/partner source not qualified')
    completeness = proof.get('source_completeness', {})
    for key in ('object_filtered_contacts', 'tangential_reports', 'external_object_contacts', 'authored_force_and_attachment_audit'):
        if completeness.get(key) is not True:
            raise ValueError('force source completeness missing: ' + key)
    audits = proof.get('independent_audit_artifacts')
    if not isinstance(audits, list) or len(audits) < 2:
        raise ValueError('independent raw contact and force/attachment audit artifacts required')
    for audit in audits:
        _artifact(audit, base)
    (raw, _) = _artifact(proof.get('raw_artifact'), base)
    rows = _jsonl(raw)
    if type(proof.get('sample_count')) is not int or proof['sample_count'] != len(rows):
        raise ValueError('force raw sample_count mismatch')
    objects = proof.get('object_body_ids')
    if not isinstance(objects, list) or not objects or any((not isinstance(v, str) or not v for v in objects)):
        raise ValueError('resolved actual object body IDs required')
    mapping = proof.get('per_digit_source')
    if not isinstance(mapping, list) or len(mapping) != 2 or any((not isinstance(side, list) or len(side) != 5 or any((not isinstance(value, str) or not value for value in side)) for side in mapping)):
        raise ValueError('independent 2x5 per-digit source mapping required')
    if len({value for side in mapping for value in side}) != 10:
        raise ValueError('whole-hand sensor source cannot be broadcast to digits')
    body_map = proof.get('body_id_to_prim_path')
    if not isinstance(body_map, dict) or not body_map:
        raise ValueError('actual SDK body-ID readback map required')
    if proof.get('raw_api_force_side') != 'on_hand':
        raise ValueError('explicit raw API force side conversion evidence required')
    previous = {}
    coverage = []
    observed_hands = set()
    for row in rows:
        if row.get('kind') != 'native_contact_source_sample' or row.get('source_scope') != 'native':
            raise ValueError('native raw contact source samples required')
        key = tuple((row.get(k) for k in ('run_uuid', 'env_id', 'episode_uid', 'reset_id')))
        tick = row.get('physics_step')
        timestamp = row.get('timestamp')
        if type(tick) is not int or type(timestamp) not in (int, float) or (not stere_promo__math.isfinite(timestamp)):
            raise ValueError('force source clock unavailable')
        if key in previous and (tick != previous[key][0] + 1 or abs(timestamp - previous[key][1] - 1 / 480) > 1e-07):
            raise ValueError('force source 480Hz clock hole')
        previous[key] = (tick, timestamp)
        contacts = row.get('contacts')
        if not isinstance(contacts, list):
            raise ValueError('actual raw contacts required')
        offsets = set()
        for contact in contacts:
            if not isinstance(contact, dict) or type(contact.get('hand')) is not int or contact['hand'] not in (0, 1) or (type(contact.get('digit')) is not int) or (not 0 <= contact['digit'] < 5) or (contact.get('partner_body_id') not in objects) or (type(contact.get('source_buffer_offset')) is not int) or (contact['source_buffer_offset'] < 0):
                raise ValueError('unresolved per-contact owner/digit/object partner')
            sensor = contact.get('source_sensor_path')
            if sensor != mapping[contact['hand']][contact['digit']]:
                raise ValueError('contact sensor path does not match independent digit source')
            offset = (sensor, contact['source_buffer_offset'])
            if offset in offsets:
                raise ValueError('raw contact slot cannot be broadcast across digits')
            offsets.add(offset)
            observed_hands.add(contact['hand'])
            if body_map.get(str(contact.get('raw_partner_body_id'))) != contact['partner_body_id']:
                raise ValueError('raw partner body ID does not resolve to actual object prim')
            for name in ('normal_force_on_object_w', 'friction_force_on_object_w', 'position_w', 'normal_force_on_hand_w', 'friction_force_on_hand_w'):
                value = stere_promo__np.asarray(contact.get(name), dtype=float)
                if value.shape != (3,) or not stere_promo__np.isfinite(value).all():
                    raise ValueError('missing actual contact vector: ' + name)
            for prefix in ('normal_force', 'friction_force'):
                if not stere_promo__np.allclose(contact[prefix + '_on_object_w'], -stere_promo__np.asarray(contact[prefix + '_on_hand_w']), rtol=1e-07, atol=1e-09):
                    raise ValueError('object-side force sign conversion does not match original API vectors')
        for name in ('external_support_force_n', 'external_upward_force_n'):
            if type(row.get(name)) not in (int, float) or not stere_promo__math.isfinite(row[name]):
                raise ValueError('external force source unavailable')
        if type(row.get('attachment_count')) is not int or row['attachment_count'] < 0:
            raise ValueError('attachment audit unavailable')
        coverage.append({key: row[key] for key in ('run_uuid', 'env_id', 'episode_uid', 'reset_id', 'physics_step', 'timestamp', 'external_support_force_n', 'external_upward_force_n', 'attachment_count')})
    if observed_hands != {0, 1}:
        raise ValueError('both native hand source channels need actual contact coverage')
    return dict(qualified=True, source_scope='native', raw_sha256=proof['raw_artifact']['sha256'], proof_sha256=binding['proof_sha256'], source_sample_count=len(rows), source_contract=required, sample_coverage=coverage)

def _build(model, raw, parent_bindings, training, requested, *, stack):
    (provenance, reasons) = _model_provenance(model)
    references = _artifacts(provenance)
    for (role, reference) in references:
        if 'manifest' in role:
            parent_bindings.append(_binding(reference))
        else:
            training.append(_binding(reference))
    parents = []
    parent_models = set()
    parent_episodes = []
    for binding in _unique(parent_bindings):
        parent = _load(stere_train_linea__Path(binding['path']), stack=stack)
        parents.append(binding)
        parent_models.add(parent['model_checkpoint_sha256'])
        raw.extend(parent['raw_episode_artifacts'])
        parent_episodes.extend(parent['episodes'])
        training.extend(parent['training_artifacts'])
        if not parent['complete_source_inventory']:
            reasons.append('parent_source_inventory_incomplete:' + binding['sha256'])
    for (role, reference) in references:
        if ('actor_package' in role or 'resume_checkpoint' in role) and 'teacher' not in role:
            if reference['sha256'] not in parent_models:
                reasons.append('input_model_training_manifest_missing:' + reference['sha256'])
    for path in provenance.get('raw_episode_source_paths', []):
        raw.append(_binding(path))
    expected = []
    for binding in _unique(training):
        (paths, identities, blocked) = _dataset_sources(binding)
        raw.extend((_binding(path) for path in paths))
        expected.extend(identities)
        reasons.extend(blocked)
    raw = _unique(raw)
    episodes = []
    for binding in raw:
        (found, blocked) = _raw_inventory(binding)
        episodes.extend(found)
        reasons.extend(blocked)
    indexed = {}
    for episode in episodes:
        key = tuple((episode[name] for name in ('run_uuid', 'env_id', 'episode_uid', 'reset_id')))
        if key in indexed and indexed[key] != episode:
            raise ValueError('episode identity appears in conflicting raw artifacts')
        indexed[key] = episode
    for (key, initial, group) in expected:
        actual = indexed.get(key)
        if actual is None or actual['physical_initial_condition_sha256'] != initial or actual['source_episode_group'] != group:
            raise ValueError('registered training dataset identity differs from actual raw episode_start')
    if not episodes:
        reasons.append('raw_training_episode_inventory_empty')
    if not requested:
        reasons.append('complete_source_inventory_not_explicitly_requested')
    reasons = sorted(set(reasons))
    result = dict(version=stere_train_linea__VERSION, model_checkpoint_artifact=model, model_checkpoint_sha256=model['sha256'], complete_source_inventory_requested=requested, complete_source_inventory=requested and (not reasons), source_scope='artifact_inventory_only', native_qualified=False, completeness_scope='explicitly_declared_recorded_dependency_closure;unrecorded_training_not_discoverable', raw_episode_artifacts=raw, training_artifacts=_unique(training), parents=parents, episodes=sorted(indexed.values(), key=lambda row: (row['run_uuid'], row['env_id'], row['reset_id'], row['episode_uid'])), physical_initial_condition_sha256=sorted({row['physical_initial_condition_sha256'] for row in episodes}), source_episode_groups=sorted({row['source_episode_group'] for row in episodes if row['source_episode_group'] is not None}), incomplete_reasons=reasons, source_sha256={stere_train_linea__Path(__file__).name: file_sha256(__file__)})
    result['manifest_sha256'] = stere_train_linea__digest(result)
    return result

def analyze_episode(rows):
    """Arithmetic/continuity only. Public qualification additionally reopens proofs."""
    headers = [r for r in rows if r.get('kind') == 'episode_start']
    samples = [r for r in rows if r.get('kind', 'sample') == 'sample' and r.get('live', True)]
    reasons = []
    header = headers[0] if len(headers) == 1 else {}
    if len(headers) != 1:
        reasons.append('missing_or_duplicate_episode_start')
    identity = _key(rows[0]) if rows else (None,) * 4
    if any((_key(r) != identity for r in rows)):
        reasons.append('mixed_episode_identity')
    owner = header.get('owner', rows[0].get('owner') if rows else None)
    style = header.get('grasp_style', rows[0].get('grasp_style') if rows else None)
    if owner not in OWNERS:
        reasons.append('unknown_action_owner')
    if style not in stere_evalu__STYLES:
        reasons.append('unknown_grasp_style')
    if any((r.get('owner') != owner or r.get('grasp_style') != style for r in rows)):
        reasons.append('mixed_owner_or_style')
    try:
        initial_sha = stere_evalu__physical_initial_sha256(header.get('physical_initial_condition'))
    except ValueError:
        initial_sha = None
        reasons.append('missing_measured_physical_initial_condition')
    if header.get('first_episode') is not True or header.get('resumed') is not False or header.get('recovered') is not False:
        reasons.append('not_complete_first_unrecovered_episode')
    if any((r.get('recovered', False) or r.get('resumed', False) for r in samples)):
        reasons.append('recovery_or_resume_in_raw')
    start = header.get('initial_physics_step')
    when = header.get('initial_timestamp')
    if type(start) is not int or not _finite(when):
        reasons.append('missing_initial_physics_clock')
    else:
        for (i, row) in enumerate(samples, 1):
            if type(row.get('physics_step')) is not int or row['physics_step'] != start + i:
                reasons.append('480hz_sample_hole_or_duplicate')
                break
            if not _finite(row.get('timestamp')) or abs(row['timestamp'] - (when + i / 480)) > 1e-07:
                reasons.append('nonmonotonic_or_incorrect_480hz_time')
                break
    terminal = [i for (i, r) in enumerate(samples) if r.get('terminal') is True]
    complete = bool(samples) and terminal == [len(samples) - 1]
    if not complete:
        reasons.append('truncated_or_after_first_terminal')
    final = samples[-1] if samples else {}
    task_success = complete and final.get('native_success') is True and (final.get('failure_code') == 0)
    if any((r.get('native_success') is True for r in samples[:-1])):
        reasons.append('success_before_terminal')
    scope = 'native' if rows and all((r.get('source_scope') == 'native' for r in rows)) else 'unqualified'
    if scope != 'native':
        reasons.append('non_native_or_unqualified_source_scope')
    lifts = [r.get('object_net_lift_m') if r.get('measurement_valid') is True else None for r in samples]
    completed = [r for r in samples if r.get('formal_window_completed') is True]
    holds = [r.get('formal_hold_s') if r.get('measurement_valid') is True and r.get('formal_tracker_source') == FORMAL_TRACKER else None for r in completed]
    slips = [r.get('formal_drift_m') if r.get('measurement_valid') is True and r.get('formal_tracker_source') == FORMAL_TRACKER and (r.get('formal_tracker_valid') is True) and (r.get('formal_anchor_latched') is True) and (r.get('formal_drift_source') == 'last_completed_window_max_over_raw_pose_points') else None for r in completed]
    if any((r.get('formal_tracker_source') != FORMAL_TRACKER for r in samples)):
        reasons.append('formal_30mm_tracker_source_unavailable')
    if not task_success:
        reasons.append('native_task_not_successful')
    tail = samples[-2400:]
    if len(tail) < 2400:
        reasons.append('less_than_5s_480hz_raw_hold_evidence')
    replay = replay_formal_tracker(samples)
    reasons.extend(replay['reasons'])
    if replay['available'] and task_success:
        if len(replay['windows']) < 300 or not replay['event'] or (not all((w['good'] for w in replay['windows'][-300:]))):
            reasons.append('formal_300_complete_windows_not_supported_by_raw_replay')
    for row in tail:
        raw_height = _finite(row.get('formal_object_bottom_m')) and _finite(row.get('formal_reference_bottom_m')) and (row['formal_object_bottom_m'] >= row['formal_reference_bottom_m'] + 0.03)
        if not (raw_height and row.get('measurement_valid') is True and (row.get('formal_task_active') is True)):
            reasons.append('raw_hold_lift_formal_slip_contact_not_qualified')
            break
    if not (_finite(final.get('formal_hold_s')) and final['formal_hold_s'] >= 5.0 - 1e-12):
        reasons.append('formal_hold_below_5s_or_NA')
    for row in samples:
        if not (row.get('force_source_qualified') is True and row.get('support_source_qualified') is True and _finite(row.get('external_support_force_n')) and (abs(row['external_support_force_n']) <= 1e-06) and _finite(row.get('external_upward_force_n')) and (abs(row['external_upward_force_n']) <= 1e-06) and (row.get('attachment_count') == 0)):
            reasons.append('support_external_force_or_attachment_source_NA')
            break
    contact_values = [float(r['bilateral_opposed_contact']) for r in samples if r.get('contact_valid') is True and type(r.get('bilateral_opposed_contact')) is bool]
    return dict(run_uuid=identity[0], env_id=identity[1], episode_uid=identity[2], episode_id=identity[2], reset_id=identity[3], owner=owner, grasp_style=style, source_scope=scope, physical_initial_condition_sha256=initial_sha, source_episode_group=header.get('source_episode_group', identity[2]), complete=complete, first_episode=header.get('first_episode') is True, resumed=header.get('resumed'), recovered=header.get('recovered'), sample_count=len(samples), native_task_success=task_success, failure_code=final.get('failure_code'), lift_m=_distribution(lifts, len(samples)), hold_s=_distribution(holds, len(completed)), formal_slip_m=_distribution(slips, len(completed)), formal_window_count=len(replay['windows']) if replay['available'] else len(completed), formal_replay_event=replay['event'], raw_measurement_coverage=sum((r.get('measurement_valid') is True for r in samples)) / max(len(samples), 1), contact_retention=sum(contact_values) / len(contact_values) if contact_values else None, contact_validity_coverage=len(contact_values) / max(len(samples), 1), qualified_success=False, raw_conditions_satisfied=not reasons, reasons=sorted(set(reasons)))

def verify_force_qualification(binding):
    try:
        return _verify_force_qualification(binding)
    except (OSError, KeyError, TypeError, UnicodeError, OverflowError) as error:
        raise ValueError('native force evidence unavailable or malformed') from error

def _load(path, *, stack):
    path = path.resolve()
    if path in stack:
        raise ValueError('training manifest parent cycle')
    try:
        value = stere_train_linea__json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('training manifest unavailable') from error
    if not isinstance(value, dict) or value.get('version') != stere_train_linea__VERSION:
        raise ValueError('versioned training source manifest required')
    if value.get('manifest_sha256') != stere_train_linea__digest({key: item for (key, item) in value.items() if key != 'manifest_sha256'}):
        raise ValueError('training source manifest SHA mismatch')
    if type(value.get('complete_source_inventory_requested')) is not bool:
        raise ValueError('explicit inventory completeness request required')
    model = _binding(value['model_checkpoint_artifact'], path.parent)
    raw = [_binding(item, path.parent) for item in value['raw_episode_artifacts']]
    parents = [_binding(item, path.parent) for item in value['parents']]
    training = [_binding(item, path.parent) for item in value['training_artifacts']]
    rebuilt = _build(model, raw, parents, training, value['complete_source_inventory_requested'], stack=stack | {path})
    if rebuilt != value:
        raise ValueError('training inventory derived from actual raw headers/artifacts differs from manifest')
    return rebuilt

def aggregate_episodes(episodes, *, provenance=None, evidence_artifacts=None, min_independent_initials=100):
    if type(min_independent_initials) is not int or min_independent_initials < 1:
        raise ValueError('positive initial-condition budget required')
    provenance = {} if provenance is None else stere_evalu__copy.deepcopy(provenance)
    groups = {}
    for e in episodes:
        groups.setdefault((e['owner'], e['grasp_style']), []).append(e)
    summaries = []
    blocked = []
    for ((owner, style), group) in sorted(groups.items(), key=lambda x: str(x[0])):
        complete = [e for e in group if e['complete']]
        cohort = {}
        for e in complete:
            if e['physical_initial_condition_sha256'] is not None and e['first_episode'] and (e['resumed'] is False) and (e['recovered'] is False):
                cohort.setdefault(e['physical_initial_condition_sha256'], e)
        independent_successes = sum((e['native_task_success'] for e in cohort.values()))
        unique = {e['physical_initial_condition_sha256'] for e in complete if e['physical_initial_condition_sha256'] is not None}
        native_independent = {e['physical_initial_condition_sha256'] for e in complete if e['source_scope'] == 'native' and e['first_episode'] and (e['resumed'] is False) and (e['recovered'] is False)}
        native_independent.discard(None)
        successes = sum((e['native_task_success'] for e in complete))
        qualified = sum((e['qualified_success'] for e in complete))
        failures = {}
        for e in complete:
            if not e['native_task_success']:
                failures[str(e['failure_code'])] = failures.get(str(e['failure_code']), 0) + 1
        if len(native_independent) < min_independent_initials:
            blocked.append(f'{owner}/{style}:insufficient_independent_initials')
        if qualified != len(complete):
            blocked.append(f'{owner}/{style}:unqualified_episode_evidence')
        summaries.append(dict(owner=owner, grasp_style=style, episodes=len(group), complete_episodes=len(complete), independent_initial_count=len(unique), native_independent_initial_count=len(native_independent), native_task_successes=successes, qualified_successes=qualified, raw_success_rate=successes / len(complete) if complete else None, independent_trials=len(cohort), independent_successes=independent_successes, success_rate=independent_successes / len(cohort) if cohort else None, wilson95=wilson_interval(independent_successes, len(cohort)), qualified_success_rate=qualified / len(complete) if complete else None, failure_breakdown=failures, lift_m=_distribution([e['lift_m']['max'] for e in group], len(group)), hold_s=_distribution([e['hold_s']['max'] for e in group], len(group)), formal_slip_m=dict(_distribution([e['formal_slip_m']['max'] for e in group], len(group)), coverage=sum((e['formal_slip_m']['coverage'] for e in group)) / max(len(group), 1)), contact_retention=_distribution([e['contact_retention'] for e in group], len(group))))
    if not episodes:
        blocked.append('no_complete_episodes')
    blocked.append('raw_native_proofs_must_be_verified_by_promotion_gate')
    if provenance.get('source_scope') != 'native':
        blocked.append('source_scope_not_native')
    for key in ('checkpoint_sha256', 'config_sha256', 'source_sha256'):
        if not provenance.get(key):
            blocked.append('missing_' + key)
    return dict(version=stere_evalu__VERSION, source_scope=provenance.get('source_scope', 'unqualified'), owner=provenance.get('owner'), checkpoint_sha256=provenance.get('checkpoint_sha256'), action_contract_sha256=stere_evalu__action_sha256(), state_contract_sha256=stere_evalu__digest(stere_evalu__state_contract()), provenance=provenance, validation_split=provenance.get('validation_split'), thresholds=stere_evalu__THRESHOLDS.copy(), minimum_independent_initials=min_independent_initials, episodes=stere_evalu__copy.deepcopy(episodes), summaries=summaries, evidence_artifacts=stere_evalu__copy.deepcopy(evidence_artifacts or []), qualified_for_promotion=not blocked, blocked_reasons=sorted(set(blocked)))

def _student_task_identity(header, samples):
    identity = header.get('task_identity')
    if not isinstance(identity, dict) or identity.get('version') != 'dh116-stereo-native-task-identity-v1':
        raise ValueError('raw Student task identity required')
    binding = stere_promo__require_sha(identity.get('task_identity_sha256'), 'Student task identity SHA')
    body = {key: value for (key, value) in identity.items() if key != 'task_identity_sha256'}
    profile = stere_promo__require_sha(identity.get('task_profile_sha256'), 'Student task profile SHA')
    if profile == '0' * 64 or binding != stere_promo__digest(body) or header.get('task_identity_sha256') != binding:
        raise ValueError('raw Student task identity or task profile mismatch')
    if identity.get('task_kind') != 'rigid_proxy' or identity.get('physical_object_kind') not in ('solid_cylinder', 'hollow_straw') or identity.get('state_source') != 'rigid_body_sdk_pose_velocity' or (identity.get('rod_state_available') is not False):
        raise ValueError('current native Student proof producer supports rigid_proxy only')
    if not samples or any((row.get('task_identity') != identity or row.get('task_identity_sha256') != binding for row in samples)):
        raise ValueError('raw Student task identity changed or is missing in physical samples')
    return stere_promo__copy.deepcopy(identity)

def register_training_lineage(output, model_checkpoint, raw_episode_artifacts, *, parent_manifests=(), complete_source_inventory=False, training_artifacts=()):
    if type(complete_source_inventory) is not bool:
        raise ValueError('explicit boolean completeness request required')
    path = stere_train_linea__Path(output)
    if path.exists():
        raise FileExistsError(path)
    model = _binding(model_checkpoint)
    value = _build(model, [_binding(item) for item in raw_episode_artifacts], [_binding(item) for item in parent_manifests], [_binding(item) for item in training_artifacts], complete_source_inventory, stack={path.resolve()})
    write_json_new(path, value)
    return dict(path=str(path.resolve()), sha256=file_sha256(path), model_checkpoint_sha256=model['sha256'], complete_source_inventory=value['complete_source_inventory'], native_qualified=False)

def evaluate_records(records, *, provenance=None, evidence_artifacts=None, min_independent_initials=100):
    if not isinstance(records, list) or any((not isinstance(r, dict) for r in records)):
        raise ValueError('raw record dictionaries required')
    groups = {}
    for row in records:
        groups.setdefault(_key(row), []).append(row)
    episodes = [analyze_episode(rows) for rows in groups.values()]
    return aggregate_episodes(episodes, provenance=provenance, evidence_artifacts=evidence_artifacts, min_independent_initials=min_independent_initials)

def _verify_episode(proof, *, expected_identity=None, subject=None, require_success=True):
    (p, base) = _read_object(proof)
    if p.get('version') != 'dh116-native-episode-proof-v1':
        raise ValueError('versioned native episode proof required')
    _native_run(p, base)
    if expected_identity is not None and p.get('model_identity') != expected_identity:
        raise ValueError('native episode model/contract identity mismatch')
    if p.get('action_contract_sha256') != stere_promo__action_sha256() or p.get('state_contract_sha256') != stere_promo__digest(stere_promo__state_contract()):
        raise ValueError('native episode current action/state contract mismatch')
    (raw, _) = _artifact(p.get('raw_artifact'), base)
    rows = _jsonl(raw)
    if len({tuple((row.get(k) for k in ('run_uuid', 'env_id', 'episode_uid', 'reset_id'))) for row in rows}) != 1:
        raise ValueError('native episode proof cannot splice environments or episodes')
    result = stere_promo__analyze_episode(rows)
    header = rows[0]
    if header.get('kind') != 'episode_start' or header.get('initial_measurement_valid') is not True or header.get('physical_initial_condition_source') != 'actual_sdk_readback':
        raise ValueError('actual SDK initial readback evidence required; seed/config is insufficient')
    live_samples = [row for row in rows if row.get('kind', 'sample') == 'sample' and row.get('live', True)]
    if result['owner'] == 'student':
        result['task_identity'] = _student_task_identity(header, live_samples)
    if p.get('sample_count') != result['sample_count']:
        raise ValueError('native episode raw sample_count mismatch')
    if subject is not None:
        expected = {'owner': subject} if isinstance(subject, str) else subject
        if not isinstance(expected, dict):
            raise ValueError('native proof subject must be owner or binding dictionary')
        if any((result.get(key, p.get(key)) != value for (key, value) in expected.items())):
            raise ValueError('native proof subject/checkpoint mismatch')
    force = p.get('force_qualification')
    if not isinstance(force, dict):
        raise ValueError('native force/support/external-source qualification remains NA')
    force = stere_promo__copy.deepcopy(force)
    if isinstance(force.get('proof_path'), str) and (not stere_promo__Path(force['proof_path']).is_absolute()):
        force['proof_path'] = str(base / force['proof_path'])
    verified_force = verify_force_qualification(force)
    fields = ('run_uuid', 'env_id', 'episode_uid', 'reset_id', 'physics_step')
    coverage = {tuple((row[k] for k in fields)): row for row in verified_force['sample_coverage']}
    for sample in live_samples:
        source = coverage.get(tuple((sample[k] for k in fields)))
        if source is None or any((sample.get(key) != source.get(key) for key in ('timestamp', 'external_support_force_n', 'external_upward_force_n', 'attachment_count'))):
            raise ValueError('native force evidence is not aligned to this exact episode/480Hz sample')
    if require_success:
        if result['reasons']:
            raise ValueError('native successful episode not qualified: ' + ','.join(result['reasons']))
    else:
        allowed = {'native_task_not_successful', 'less_than_5s_480hz_raw_hold_evidence', 'formal_hold_below_5s_or_NA', 'raw_hold_lift_formal_slip_contact_not_qualified'}
        failures = [reason for reason in result['reasons'] if reason not in allowed]
        if failures:
            raise ValueError('native completed episode protocol invalid: ' + ','.join(failures))
    result.update(native_evidence_verified=True, qualified_success=result['native_task_success'] and (not result['reasons']), proof_sha256=stere_promo__digest(p), checkpoint_sha256=p.get('checkpoint_sha256'), force_binding=verified_force, runtime=stere_promo__copy.deepcopy(header.get('runtime')), actor_runtime_samples_verified=isinstance(header.get('runtime'), dict) and all((row.get('runtime') == header['runtime'] for row in live_samples)))
    return result

def load_training_lineage(path, expected_model_checkpoint_sha256=None, require_complete=False):
    result = _load(stere_train_linea__Path(path), stack=set())
    if expected_model_checkpoint_sha256 is not None and result['model_checkpoint_sha256'] != stere_train_linea__require_sha(expected_model_checkpoint_sha256):
        raise ValueError('training manifest model checkpoint SHA mismatch')
    if require_complete and (not result['complete_source_inventory']):
        raise ValueError('complete training source inventory unavailable: ' + ';'.join(result['incomplete_reasons']))
    return result

def training_partition(report, *, base=None, require_complete=True):
    """Reopen registered training raw-header lineage for this actual model.

    Registration proves the contents and union of the declared inventory. It
    cannot independently prove an operator never omitted an unregistered source.
    Native qualification still separately verifies actual rollout evidence.
    """
    binding = report.get('provenance', {}).get('training_manifest_artifact')
    if not isinstance(binding, dict):
        raise ValueError('SHA-bound training source manifest required for native heldout evaluation')
    try:
        path = stere_evalu__Path(binding['path'])
        if not path.is_absolute():
            path = (stere_evalu__Path(base) if base is not None else stere_evalu__Path.cwd()) / path
        raw = path.read_bytes()
    except (KeyError, TypeError, OSError) as error:
        raise ValueError('training source manifest artifact unavailable') from error
    if stere_evalu__hashlib.sha256(raw).hexdigest() != binding.get('sha256'):
        raise ValueError('training source manifest SHA mismatch')
    try:
        _stere_evalu_import__load_training_lineage = load_training_lineage
    except ImportError as error:
        raise ValueError('training source manifest verifier unavailable') from error
    lineage = _stere_evalu_import__load_training_lineage(path, expected_model_checkpoint_sha256=report.get('checkpoint_sha256'), require_complete=require_complete)
    initials = set(lineage['physical_initial_condition_sha256'])
    groups = set(lineage['source_episode_groups'])
    for (name, actual) in (('training_initial_sha256', initials), ('training_episode_groups', groups)):
        declared = report.get('provenance', {}).get(name)
        if declared is not None and set(declared) != actual:
            raise ValueError('declared training partition differs from reopened raw-header manifest: ' + name)
    return dict(physical_initial_condition_sha256=initials, source_episode_groups=groups, manifest_sha256=binding['sha256'], complete_source_inventory=lineage['complete_source_inventory'])

def verify_native_episode(proof, *, expected_identity=None, subject=None):
    """Validate a complete first successful native episode; return its bound UID."""
    try:
        return _verify_episode(proof, expected_identity=expected_identity, subject=subject)
    except (OSError, KeyError, TypeError, UnicodeError) as error:
        raise ValueError('native episode evidence unavailable or malformed') from error

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-checkpoint', type=stere_train_linea__Path, required=True)
    parser.add_argument('--raw-episode-artifact', type=stere_train_linea__Path, action='append', default=[])
    parser.add_argument('--parent-manifest', type=stere_train_linea__Path, action='append', default=[])
    parser.add_argument('--training-artifact', type=stere_train_linea__Path, action='append', default=[])
    parser.add_argument('--declare-source-inventory-complete', action='store_true')
    parser.add_argument('--output', type=stere_train_linea__Path, required=True)
    args = parser.parse_args(argv)
    result = register_training_lineage(args.output, args.model_checkpoint, args.raw_episode_artifact, parent_manifests=args.parent_manifest, training_artifacts=args.training_artifact, complete_source_inventory=args.declare_source_inventory_complete)
    print(stere_train_linea__json.dumps(result, sort_keys=True))
    return 0

def select_best_checkpoint(reports, *, training_initial_sha256=(), training_episode_groups=()):
    if not reports:
        raise ValueError('heldout evaluation reports required')
    candidates = []
    cohort = None
    for report in reports:
        if report.get('validation_split') != 'heldout':
            raise ValueError('only independently heldout evaluation selects checkpoints')
        train_initials = set(training_initial_sha256)
        train_groups = set(training_episode_groups)
        if report.get('source_scope') == 'native':
            partition = training_partition(report)
            train_initials.update(partition['physical_initial_condition_sha256'])
            train_groups.update(partition['source_episode_groups'])
        initials = {e.get('physical_initial_condition_sha256') for e in report.get('episodes', [])}
        groups = {e.get('source_episode_group') for e in report.get('episodes', [])}
        if None in initials or initials & train_initials or groups & train_groups:
            raise ValueError('training/heldout initial or source episode leakage')
        checkpoint = report.get('checkpoint_sha256')
        if not isinstance(checkpoint, str) or len(checkpoint) != 64:
            raise ValueError('actual model checkpoint SHA required')
        artifacts = [a for a in report.get('evidence_artifacts', []) if a.get('kind') == 'checkpoint']
        if len(artifacts) != 1:
            raise ValueError('one actual checkpoint artifact required for selection')
        try:
            raw = stere_evalu__Path(artifacts[0]['path']).read_bytes()
        except (OSError, KeyError, TypeError) as error:
            raise ValueError('actual checkpoint artifact unavailable') from error
        if artifacts[0].get('sha256') != checkpoint or stere_evalu__hashlib.sha256(raw).hexdigest() != checkpoint:
            raise ValueError('actual checkpoint artifact SHA mismatch')
        if cohort is not None and initials != cohort:
            raise ValueError('checkpoint comparison requires the same heldout initial cohort')
        cohort = initials
        episodes = report['episodes']
        if report.get('source_scope') == 'native':
            _stere_evalu_import___report_episodes = _report_episodes
            (_, episodes, _) = _stere_evalu_import___report_episodes(report)
            if {e['physical_initial_condition_sha256'] for e in episodes} != initials:
                raise ValueError('heldout initial set differs from actual native raw proofs')
            if any((e['physical_initial_condition_sha256'] in train_initials or e.get('source_episode_group') in train_groups for e in episodes)):
                raise ValueError('actual native heldout raw-header training leakage')
        distinct = {}
        for episode in episodes:
            distinct.setdefault(episode['physical_initial_condition_sha256'], episode)
        success = sum((e['native_task_success'] for e in distinct.values()))
        score = success / max(len(distinct), 1)
        physical = False
        physical_reasons = ['source_scope_not_native']
        if report.get('source_scope') == 'native':
            _stere_evalu_import__require_teacher_qualification = require_teacher_qualification
            _stere_evalu_import__verify_student_qualification = verify_student_qualification
            verifier = {'teacher': _stere_evalu_import__require_teacher_qualification, 'student': _stere_evalu_import__verify_student_qualification}.get(report.get('owner'))
            if verifier is None:
                physical_reasons = ['owner_has_no_Teacher_or_pure_Student_physical_qualification']
            else:
                try:
                    qualified = verifier(report, checkpoint)
                    physical = qualified.get('qualified') is True
                    physical_reasons = [] if physical else ['physical_qualification_not_granted']
                except ValueError as error:
                    physical_reasons = [str(error)]
        candidates.append((score, checkpoint, report, physical, physical_reasons))
    (score, checkpoint, report, physical, physical_reasons) = max(candidates, key=lambda x: (x[0], x[1]))
    return dict(checkpoint_sha256=checkpoint, heldout_success_rate=score, physical_qualified=physical, physical_qualification_reasons=physical_reasons, selection_source='independent_heldout_episode_outcomes;not_peak_training_reward')

def _report_episodes(report):
    (r, base) = _read_object(report)
    if r.get('version') != stere_promo__VERSION or r.get('source_scope') != 'native':
        raise ValueError('native evaluation report required; schema/CPU cannot qualify')
    if r.get('action_contract_sha256') != stere_promo__action_sha256() or r.get('state_contract_sha256') != stere_promo__digest(stere_promo__state_contract()):
        raise ValueError('evaluation current state/action contract mismatch')
    if r.get('thresholds') != stere_promo__THRESHOLDS:
        raise ValueError('unchanged formal thresholds required')
    proofs = r.get('episode_proofs')
    if not isinstance(proofs, list) or not proofs:
        raise ValueError('actual SHA-bound per-episode raw proofs required')
    episodes = []
    for proof in proofs:
        if isinstance(proof, str) and (not stere_promo__Path(proof).is_absolute()):
            proof = str(base / proof)
        episodes.append(_verify_episode(proof, require_success=False))
    ids = [tuple((e[k] for k in ('run_uuid', 'env_id', 'episode_uid', 'reset_id'))) for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate native episode evidence')
    return (r, episodes, base)
stere_promo__VERSION = stere_evalu__VERSION
stere_promo__THRESHOLDS = stere_evalu__THRESHOLDS
stere_promo__analyze_episode = analyze_episode
stere_promo__aggregate_episodes = aggregate_episodes
stere_promo__training_partition = training_partition

def qualified_episode_identities(report):
    (_, episodes, _) = _report_episodes(report)
    return {tuple((e[k] for k in ('run_uuid', 'env_id', 'episode_uid', 'reset_id'))) for e in episodes if e['qualified_success']}

def require_teacher_qualification(report, checkpoint_sha256, styles=stere_promo__STYLES):
    stere_promo__require_sha(checkpoint_sha256, 'Teacher checkpoint SHA')
    (r, episodes, base) = _report_episodes(report)
    if r.get('owner') != 'teacher' or r.get('checkpoint_sha256') != checkpoint_sha256:
        raise ValueError('qualified Teacher checkpoint identity mismatch')
    if any((e['owner'] != 'teacher' or e.get('checkpoint_sha256') != checkpoint_sha256 for e in episodes)):
        raise ValueError('MPC/BC/Student outcomes cannot qualify this Teacher checkpoint')
    artifacts = r.get('evidence_artifacts', [])
    if not isinstance(artifacts, list):
        raise ValueError('original artifact bindings required')
    kinds = set()
    for item in artifacts:
        _artifact(item, base)
        kinds.add(item.get('kind'))
        if item.get('kind') == 'checkpoint' and item['sha256'] != checkpoint_sha256:
            raise ValueError('actual model checkpoint artifact mismatch')
    if not {'checkpoint', 'source', 'config'} <= kinds:
        raise ValueError('actual model/source/config artifacts required')
    for style in styles:
        subset = [e for e in episodes if e['grasp_style'] == style]
        independent = {e['physical_initial_condition_sha256'] for e in subset}
        independent.discard(None)
        if len(independent) < 100:
            raise ValueError('at least100 independent measured native initial conditions required per style')
        if not any((e['qualified_success'] for e in subset)):
            raise ValueError('no qualified complete native Teacher success for style ' + style)
    return dict(qualified=True, checkpoint_sha256=checkpoint_sha256, action_contract_sha256=stere_promo__action_sha256(), state_contract_sha256=stere_promo__digest(stere_promo__state_contract()), styles=list(styles), evidence_artifacts=artifacts, report_sha256=stere_promo__digest(r), episode_identities=sorted((tuple((e[k] for k in ('run_uuid', 'env_id', 'episode_uid', 'reset_id'))) for e in episodes if e['qualified_success'])))

def stereo_metric_gate(metrics, limits):
    """Pure threshold arithmetic, never an evidence/qualification assertion."""
    keys = ('mae_m', 'rmse_m', 'coverage', 'count')
    return all((type(metrics.get(k)) in (int, float) and stere_promo__math.isfinite(metrics[k]) for k in keys)) and metrics['count'] >= limits['min_valid_pixels'] and (metrics['coverage'] >= limits['min_coverage']) and (0 <= metrics['mae_m'] <= limits['max_mae_m']) and (0 <= metrics['rmse_m'] <= limits['max_rmse_m'])

def _verified_wire_mask(reference, base, shape):
    """Re-resolve each env's archived actual instance IDs against stage paths."""
    import torch
    from dh116.physics.diagnostics import resolve_wire_instance_mask
    from dh116.physics.diagnostics import AUDITED_CAMERA_VIEW_SHA256
    from dh116.physics.diagnostics import AUDITED_INSTANCE_ANNOTATOR_SHA256
    wire = reference.get('wire_reference_mask')
    if wire is None:
        return None
    if reference.get('wire_mask_source_scope') != 'native_resolved_instance_id':
        raise ValueError('wire completeness requires native resolved instance-ID readback')
    (_, path) = _artifact(reference.get('wire_mask_source_artifact'), base)
    raw = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(raw, dict) or raw.get('version') != 'dh116-wire-instance-reference-v2' or raw.get('source_scope') != 'native' or (raw.get('validation_only') is not True) or (raw.get('actual_wire_path_source') != 'RigidObject.root_physx_view.prim_paths') or any((raw.get(k) != reference.get(k) for k in ('frame_id', 'simulation_time', 'calibration_sha256', 'render_timing'))):
        raise ValueError('native instance reference source/alignment mismatch')
    sdk = raw.get('sdk_camera_view_source_artifact')
    _artifact(sdk, path.parent)
    if sdk.get('sha256') != AUDITED_CAMERA_VIEW_SHA256:
        raise ValueError('instance-ID SDK source is not audited')
    annotator = raw.get('sdk_instance_annotator_source_artifact')
    _artifact(annotator, path.parent)
    if annotator.get('sha256') != AUDITED_INSTANCE_ANNOTATOR_SHA256:
        raise ValueError('renderer-instance annotator mapping source is not audited')
    ids = raw.get('raw_instance_ids')
    paths = raw.get('stage_validated_paths')
    if not isinstance(ids, torch.Tensor) or ids.shape != shape or (not isinstance(paths, list)):
        raise ValueError('raw instance IDs/actual stage validation readback unavailable')
    optical = raw.get('reference_valid')
    if not isinstance(optical, torch.Tensor) or not torch.equal(optical, reference['valid']):
        raise ValueError('instance unresolved-pixel eligibility differs from actual optical-Z validity')
    if raw.get('wire_render_prim_paths') is None:
        raise ValueError('actual Wire render-geometry inventory unavailable')
    resolved = resolve_wire_instance_mask(ids, raw.get('raw_instance_info'), raw.get('wire_prim_paths'), path_validator=lambda value: value in paths, wire_render_prim_paths=raw['wire_render_prim_paths'], reference_valid=optical)
    truth = resolved['mask']
    if truth is None:
        raise ValueError('actual instance map cannot resolve per-environment Wire roots: ' + repr(resolved['blocked_reasons']))
    if not isinstance(wire, torch.Tensor) or wire.shape != truth.shape or wire.dtype != torch.bool or (not torch.equal(wire, truth)):
        raise ValueError('wire mask differs from archived per-environment resolved raw instance IDs')
    return truth

def _measure_stereo_evidence(proof):
    """Recompute RGB stereo from native captures and separate reference depth."""
    import torch
    from dh116.physics.vision import StereoCalibration as StereoCalibration
    from dh116.physics.vision import StereoDepth as StereoDepth
    from dh116.physics.vision import validate_identity
    (p, base) = _read_object(proof)
    if p.get('version') != 'dh116-native-stereo-proof-v1':
        raise ValueError('versioned native stereo proof required')
    _native_run(p, base)
    (_, calibration_path) = _artifact(p.get('calibration_artifact'), base)
    calibration = StereoCalibration.from_json(calibration_path)
    (_, validation_path) = _process_output(p, base, p.get('validation_config_artifact'))
    try:
        validation = stere_promo__json.loads(validation_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('stereo validation config unavailable') from error
    if validation.get('version') != 'dh116-stereo-validation-v1':
        raise ValueError('versioned stereo thresholds required')
    limits = validation.get('stereo_limits', {})
    policy = stere_promo__json.loads((ROOT / 'configs/cameras/validation.json').read_text(encoding='utf-8'))
    expected = policy['stereo_limits']
    policy_compatible = limits == expected
    if validation.get('depth_estimator', {}) != p['producer']['kwargs'].get('depth_estimator', {}):
        raise ValueError('measurement estimator profile differs from the actual producer kwargs')
    (_, capture_path) = _artifact(p.get('capture_artifact'), base)
    try:
        captured = torch.load(capture_path, map_location='cpu', weights_only=False)
    except Exception as error:
        raise ValueError('native RGB capture cannot be read') from error
    if not isinstance(captured, dict) or captured.get('version') != 'dh116-stereo-rgb-probe-v1':
        raise ValueError('actual native RGB probe capture required')
    frames = captured.get('frames')
    origins = captured.get('environment_origins')
    references = p.get('reference_artifacts')
    if not isinstance(frames, list) or not frames or (not isinstance(references, list)) or (len(references) != len(frames)) or (p.get('sample_count') != len(frames)) or (origins is None):
        raise ValueError('complete same-frame RGB/reference and native environment-origin readback required')
    errors = []
    valid_pixels = 0
    reference_pixels = 0
    last = -1
    last_time = -1.0
    candidate_total = 0
    candidate_reconstructed = 0
    wire_total = 0
    wire_reconstructed = 0
    wire_unavailable = []
    wire_frames = 0
    wire_errors = []
    estimator = StereoDepth(**validation.get('depth_estimator', {}))
    for (packet, reference_binding) in zip(frames, references):
        if packet.get('native_evidence') is not True or packet.get('depth_source') != 'computed_from_left_right_RGB':
            raise ValueError('native RGB-only stereo acquisition source required')
        frame = packet.get('frame_id')
        when = packet.get('simulation_time')
        timing = packet.get('render_timing', {})
        if type(frame) is not int or frame <= last or type(when) not in (int, float) or (not stere_promo__math.isfinite(when)) or (when < last_time) or (timing.get('frame_id') != frame) or (timing.get('simulation_time') != when):
            raise ValueError('native stereo render/frame/time synchronization failed')
        validate_identity(packet.get('identity'), len(packet['left_rgb']), when)
        for identity in packet['identity'] if isinstance(packet['identity'], list) else [packet['identity']]:
            if identity.get('frame_id') != frame or identity.get('calibration_sha256') != calibration.sha256:
                raise ValueError('native stereo pre-action identity/frame/calibration mismatch')
        calibration.validate_readback(packet['K_left'], packet['K_right'], packet['T_world_left'], packet['T_world_right'], origins=origins)
        if packet.get('calibration_sha256') != calibration.sha256:
            raise ValueError('native stereo calibration SHA mismatch')
        actual = estimator.compute(packet['left_rgb'], packet['right_rgb'], calibration)
        if not torch.equal(actual['valid'], packet['valid']) or not torch.allclose(actual['depth_m'][actual['valid']], packet['depth_m'][actual['valid']], atol=1e-06, rtol=1e-05):
            raise ValueError('saved depth/valid is not reproducible from saved stereo RGB')
        (_, reference_path) = _process_output(p, base, reference_binding)
        reference = torch.load(reference_path, map_location='cpu', weights_only=False)
        if not isinstance(reference, dict) or reference.get('version') != 'dh116-stereo-depth-reference-v1' or reference.get('source_scope') != 'native' or (reference.get('reference_used_as_actor_input') is not False) or (reference.get('validation_only') is not True) or (reference.get('source_units') != 'optical_axis_z_m') or (reference.get('reference_source') not in ('rendered_depth_validation_only', 'independent_calibrated_depth_reference')) or any((reference.get(k) != packet.get(k) for k in ('frame_id', 'simulation_time', 'calibration_sha256'))):
            raise ValueError('independent same-frame native depth reference source required')
        from dh116.physics.diagnostics import AUDITED_CAMERA_VIEW_SHA256
        from dh116.physics.diagnostics import validate_reference_timing
        validate_reference_timing(packet, reference, native=True)
        sdk = reference.get('sdk_camera_view_source_artifact')
        _process_output(p, base, sdk, binding_base=reference_path.parent)
        if sdk['sha256'] != AUDITED_CAMERA_VIEW_SHA256:
            raise ValueError('reference optical-Z SDK source mapping is not audited')
        (target, mask) = (reference['depth_m'], reference['valid'])
        if target.shape != actual['depth_m'].shape or mask.shape != target.shape or mask.dtype != torch.bool:
            raise ValueError('reference depth/mask schema mismatch')
        expected_reference = torch.isfinite(target) & (target > 0)
        if not torch.equal(mask, expected_reference):
            raise ValueError('reference validity cannot hide finite optical-depth pixels')
        if reference.get('evaluation_depth_range_m') != [estimator.near, estimator.far]:
            raise ValueError('reference evaluation ROI differs from declared stereo depth profile')
        eligible = mask & (target >= estimator.near) & (target <= estimator.far)
        joint = actual['valid'] & eligible
        errors.extend((actual['depth_m'][joint] - target[joint]).double().tolist())
        valid_pixels += int(joint.sum())
        reference_pixels += int(eligible.sum())
        candidate = reference.get('rgb_candidate_mask')
        if isinstance(candidate, torch.Tensor):
            from dh116.env.cameras import brown_mask
            recomputed = brown_mask(packet['left_rgb'].permute(0, 2, 3, 1), float_rgb_is_255=False)
            if candidate.shape != eligible.shape or candidate.dtype != torch.bool or (not torch.equal(candidate, recomputed)):
                raise ValueError('RGB candidate mask differs from saved RGB baseline rule')
            candidate_total += int((candidate & eligible).sum())
            candidate_reconstructed += int((candidate & joint).sum())
        if reference.get('wire_mask_source_artifact') is not None:
            _process_output(p, base, reference['wire_mask_source_artifact'], binding_base=reference_path.parent)
        if reference.get('sdk_instance_annotator_source_artifact') is not None:
            _process_output(p, base, reference['sdk_instance_annotator_source_artifact'], binding_base=reference_path.parent)
        wire = _verified_wire_mask(reference, reference_path.parent, eligible.shape)
        if wire is not None:
            wire_frames += 1
            from dh116.physics.vision import backproject
            true_points = backproject(target, calibration.K_left, eligible)
            estimated = backproject(actual['depth_m'], calibration.K_left, actual['valid'])
            distance = (true_points - estimated).norm(dim=-1)
            wire_total += int((wire & eligible).sum())
            wire_reconstructed += int((wire & joint & (distance <= limits.get('wire_point_tolerance_m', 0.005))).sum())
            wire_errors.extend((actual['depth_m'][wire & joint] - target[wire & joint]).double().tolist())
        else:
            wire_unavailable.extend(reference.get('wire_mask_blocked_reasons', ['instance_reference_unavailable']))
        (last, last_time) = (frame, when)
    values = stere_promo__np.asarray(errors)
    wire_values = stere_promo__np.asarray(wire_errors)
    wire_complete = wire_frames == len(frames)
    metrics = dict(mae_m=float(stere_promo__np.abs(values).mean()) if len(values) else None, rmse_m=float(stere_promo__np.sqrt(stere_promo__np.square(values).mean())) if len(values) else None, p95_m=float(stere_promo__np.quantile(stere_promo__np.abs(values), 0.95)) if len(values) else None, coverage=valid_pixels / max(reference_pixels, 1), count=valid_pixels, rgb_candidate_point_cloud_completeness=candidate_reconstructed / candidate_total if candidate_total else None, wire_point_cloud_completeness=wire_reconstructed / wire_total if wire_total and (not wire_unavailable) else None, wire_point_cloud_completeness_observed_subset=wire_reconstructed / wire_total if wire_total else None, wire_reference_frame_coverage=wire_frames / len(frames), wire_depth_mae_m=float(stere_promo__np.abs(wire_values).mean()) if len(wire_values) and wire_complete else None, wire_depth_rmse_m=float(stere_promo__np.sqrt(stere_promo__np.square(wire_values).mean())) if len(wire_values) and wire_complete else None, wire_depth_p95_m=float(stere_promo__np.quantile(stere_promo__np.abs(wire_values), 0.95)) if len(wire_values) and wire_complete else None, wire_depth_count=len(wire_values) if wire_complete else None, wire_depth_validity_coverage=len(wire_values) / wire_total if wire_total and wire_complete else None, wire_reference_visible_points=wire_total, wire_reconstructed_points=wire_reconstructed, wire_completeness_blocked_reasons=sorted(set(wire_unavailable)), wire_completeness_sampling='dense_reference_visible_wire_pixels;not_hidden_entire_object')
    qualified = policy_compatible and len(frames) >= expected['min_frames'] and stereo_metric_gate(metrics, expected)
    cohort = None
    initials = captured.get('physical_initial_conditions')
    if captured.get('initial_condition_source') == 'actual_sdk_readback' and isinstance(initials, list) and initials:
        actual_shas = sorted({stere_promo__physical_initial_sha256(initial) for initial in initials})
        rig = dict(baseline_m=calibration.baseline_m, T_world_left=calibration.T_world_left.tolist(), T_world_right=calibration.T_world_right.tolist())
        schedule = captured.get('capture_schedule')
        if not isinstance(schedule, dict):
            raise ValueError('actual policy/physics capture schedule required for resolution cohort')
        tick0 = captured.get('initial_physics_tick')
        time0 = captured.get('initial_simulation_time')
        resets = captured.get('initial_reset_ids')
        if type(tick0) is not int or type(time0) not in (float, int) or (not isinstance(resets, list)):
            raise ValueError('initial raw capture clocks required')
        actual_schedule = dict(advance_control_between_samples=p['producer']['kwargs'].get('advance_control_between_samples'), warmup_frames=p['producer']['kwargs'].get('warmup_frames'), frames=len(frames), policy_ticks=[], physics_ticks_from_initial=[], simulation_times_from_initial=[], reset_counts_from_initial=[], sampled_physical_initial_sha256=[])
        for packet in frames:
            identity = packet['identity'] if isinstance(packet['identity'], list) else [packet['identity']]
            if len(identity) != len(resets) or len({row['policy_tick'] for row in identity}) != 1:
                raise ValueError('batch actual policy/reset schedule mismatch')
            actual_schedule['policy_ticks'].append(identity[0]['policy_tick'])
            actual_schedule['physics_ticks_from_initial'].append(identity[0]['physics_tick_end'] - tick0)
            actual_schedule['simulation_times_from_initial'].append(round(packet['simulation_time'] - time0, 9))
            actual_schedule['reset_counts_from_initial'].append([row['reset_id'] - start for (row, start) in zip(identity, resets)])
            actual_schedule['sampled_physical_initial_sha256'].append([row.get('physical_initial_condition_sha256') for row in identity])
        if schedule != actual_schedule:
            raise ValueError('declared comparison schedule differs from raw policy/physics/episode records')
        cohort = stere_promo__digest(dict(physical_initial_sha256=actual_shas, rig=rig, capture_schedule=schedule))
        if captured.get('cohort_sha256') != cohort or p.get('cohort_sha256') != cohort or validation.get('cohort_sha256') != cohort:
            raise ValueError('resolution cohort does not match raw measured initial states/rig/config')
    performance = None
    if p.get('performance_artifact') is not None:
        (performance_raw, _) = _process_output(p, base, p['performance_artifact'])
        performance = stere_promo__json.loads(performance_raw)
        if performance != captured.get('performance') or performance.get('source_scope') != 'native':
            raise ValueError('performance artifact differs from raw capture measurement ledger')
        samples = performance.get('timing_samples')
        mem = performance.get('memory_samples')
        if not isinstance(samples, list) or len(samples) != len(frames) or (not isinstance(mem, list)) or (len(mem) != len(frames)):
            raise ValueError('per-frame actual timing/GPU samples required')
        for key in ('capture_ms', 'depth_ms', 'actor_ms'):
            measured = [row.get(key) for row in samples]
            if key == 'actor_ms' and all((value is None for value in measured)):
                if performance.get(key) is not None:
                    raise ValueError('missing Actor cannot report zero latency')
                continue
            if any((type(value) not in (float, int) or not stere_promo__math.isfinite(value) or value < 0 for value in measured)):
                raise ValueError('invalid actual performance timing samples')
            if not stere_promo__math.isclose(performance.get(key, -1), sum(measured) / len(measured), abs_tol=1e-08):
                raise ValueError('performance mean does not match raw samples')
        if performance.get('actor_ms') is not None:
            actor = p.get('actor_package_artifact')
            _artifact(actor, base)
            if actor['sha256'] != performance.get('actor_package_sha256'):
                raise ValueError('Actor latency package/source SHA mismatch')
        if performance.get('measurement_device') == 'cuda':
            if any((row.get('source_scope') != 'native' or row.get('measurement_device') != 'cuda' for row in mem)):
                raise ValueError('CUDA performance requires actual device measurements, not CPU zeros')
            for (key, reduce) in (('gpu_peak_allocated_bytes', max), ('gpu_peak_reserved_bytes', max), ('gpu_free_bytes', min)):
                measured = [row.get(key) for row in mem]
                if any((type(value) is not int or value <= 0 for value in measured)) or performance.get(key) != reduce(measured):
                    raise ValueError('CUDA memory summary not backed by raw device samples')
    return dict(qualified=qualified, measurements_verified=True, source_scope='native', calibration_sha256=calibration.sha256, metrics=metrics, proof_sha256=stere_promo__digest(p), validation_config_sha256=p['validation_config_artifact']['sha256'], width=calibration.width, height=calibration.height, depth_estimator_config=estimator.to_config(), cohort_sha256=cohort, sample_count=len(frames), performance=performance, conditions=dict(validation_policy_compatible=policy_compatible, minimum_frames_met=len(frames) >= expected['min_frames'], source_and_sync_verified=True, raw_calibration_verified=True), blocked_reasons=[] if qualified else ['native_stereo_depth_accuracy_coverage_framecount_or_policy_threshold_failed'])

def measure_stereo_evidence(proof):
    try:
        return _measure_stereo_evidence(proof)
    except (OSError, KeyError, TypeError, RuntimeError, UnicodeError) as error:
        raise ValueError('native stereo proof unavailable or malformed') from error

def verify_stereo_qualification(proof):
    result = measure_stereo_evidence(proof)
    if not result['qualified']:
        raise ValueError('native stereo depth error/coverage does not meet audited thresholds')
    return result

def _verify_student_qualification(report, checkpoint_sha256, styles=stere_promo__STYLES):
    """Independent heldout Student-only first episodes; no Teacher/MPC/critic input."""
    stere_promo__require_sha(checkpoint_sha256, 'Student checkpoint SHA')
    (r, episodes, base) = _report_episodes(report)
    if r.get('owner') != 'student' or r.get('checkpoint_sha256') != checkpoint_sha256 or r.get('validation_split') != 'heldout':
        raise ValueError('independent heldout Student checkpoint report required')
    if r.get('task_kind') != 'rigid_proxy':
        raise ValueError('independent Student report must declare rigid_proxy for this proof producer')
    task_sha = stere_promo__require_sha(r.get('task_identity_sha256'), 'Student report task identity SHA')
    initials = [episode['physical_initial_condition_sha256'] for episode in episodes]
    if len(set(initials)) != len(initials):
        raise ValueError('Student physical initial condition repeated across acceptance episodes')
    required = dict(teacher_execution_enabled=False, mpc_enabled=False, critic_deployed=False, privileged_actor_inputs=False, critic_training=False, actor_input_source='stereo_RGB_and_proprio_only', depth_source='computed_from_synchronized_left_right_RGB')
    partition = stere_promo__training_partition(r, base=base)
    training_initials = partition['physical_initial_condition_sha256']
    training_groups = partition['source_episode_groups']
    for episode in episodes:
        task = episode.get('task_identity') or {}
        if task.get('task_kind') != 'rigid_proxy' or task.get('task_identity_sha256') != task_sha:
            raise ValueError('Student report mixes raw task identities')
        runtime = episode.get('runtime') or {}
        if episode['owner'] != 'student' or episode.get('checkpoint_sha256') != checkpoint_sha256 or episode.get('actor_runtime_samples_verified') is not True or any((runtime.get(k) != v for (k, v) in required.items())):
            raise ValueError('raw Student runtime indicates Teacher/MPC/privileged/critic execution or missing source')
        if runtime.get('model_checkpoint_sha256') != checkpoint_sha256:
            raise ValueError('raw runtime model checkpoint identity mismatch')
        if episode['physical_initial_condition_sha256'] in training_initials or episode['source_episode_group'] in training_groups:
            raise ValueError('Student heldout initial/source-episode training leakage')
    artifacts = r.get('evidence_artifacts', [])
    checkpoints = []
    kinds = set()
    for item in artifacts:
        _artifact(item, base)
        kinds.add(item.get('kind'))
        if item.get('kind') == 'checkpoint':
            checkpoints.append(item)
    if not {'checkpoint', 'source', 'config'} <= kinds or len(checkpoints) != 1 or checkpoints[0]['sha256'] != checkpoint_sha256:
        raise ValueError('actual Student model/source/config artifacts required')
    for style in styles:
        subset = [e for e in episodes if e['grasp_style'] == style]
        if len({e['physical_initial_condition_sha256'] for e in subset}) < 100:
            raise ValueError('Student needs100 measured independent heldout initial conditions per grasp')
        if not any((e['qualified_success'] for e in subset)):
            raise ValueError('no qualified Student-only success for grasp')
    return dict(qualified=True, owner='student', checkpoint_sha256=checkpoint_sha256, task_kind='rigid_proxy', task_identity_sha256=task_sha, action_contract_sha256=stere_promo__action_sha256(), state_contract_sha256=stere_promo__digest(stere_promo__state_contract()), report_sha256=stere_promo__digest(r))

def verify_student_qualification(report, checkpoint_sha256, styles=stere_promo__STYLES):
    try:
        return _verify_student_qualification(report, checkpoint_sha256, styles)
    except (OSError, KeyError, TypeError, RuntimeError, UnicodeError) as error:
        raise ValueError('native Student proof unavailable or malformed') from error

def promotion_ledger(phases, *, teacher_report=None, teacher_checkpoint_sha256=None, stereo_native_proof=None, student_report=None, student_checkpoint_sha256=None):
    """Implementation/CPU progress is independent from pending physical gates."""
    output = {}
    for i in range(8):
        key = 'phase' + str(i)
        source = phases.get(key, {})
        output[key] = dict(implemented=source.get('implemented') is True, cpu_verified=source.get('cpu_verified') is True, native_verified=False, physical_qualified=False)
    reasons = []
    teacher = False
    if teacher_report is not None:
        try:
            require_teacher_qualification(teacher_report, teacher_checkpoint_sha256)
            output['phase1'].update(native_verified=True, physical_qualified=True)
            teacher = True
        except (ValueError, TypeError) as error:
            reasons.append('Teacher: ' + str(error))
    else:
        reasons.append('Teacher native physical qualification pending')
    stereo = False
    student = False
    if stereo_native_proof is not None:
        try:
            verify_stereo_qualification(stereo_native_proof)
            output['phase2'].update(native_verified=True, physical_qualified=True)
            stereo = True
        except (ValueError, TypeError, OSError, KeyError) as error:
            reasons.append('Stereo: ' + str(error))
    else:
        reasons.append('Stereo native calibration/accuracy pending')
    if student_report is not None:
        try:
            verify_student_qualification(student_report, student_checkpoint_sha256)
            output['phase7'].update(native_verified=True, physical_qualified=True)
            student = True
        except (ValueError, TypeError) as error:
            reasons.append('Student: ' + str(error))
    else:
        reasons.append('Final Student independent native qualification pending')
    return dict(phases=output, teacher_qualified=teacher, formal_dagger_allowed=teacher and stereo, deployment_allowed=stereo and student, isolated_cpu_implementation_allowed=True, blocked_reasons=reasons)


def current_producer_api(runner,source,command,*,stereo):
    """Bind real current APIs; path recognition alone never qualifies a sample."""
    package=ROOT/"src/dh116"
    expected=package/("env/cameras.py" if stereo else "physics/runtime.py")
    if (runner!= (package/"rl/training.py").resolve() or source!=expected.resolve()
            or not isinstance(command,list) or not all(isinstance(v,str) for v in command)
            or "--worker" not in command
            or str((ROOT/"scripts/train_teacher.py").resolve()) not in command):
        raise ValueError("current exact external observer, capture API and worker command required")
    return ("dh116.env.cameras.TiledStereoCapture.capture" if stereo
        else "dh116.physics.runtime.NativeTensorPhysicsRuntime._notify_capture")
