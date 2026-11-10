"""Explicit offline metadata binding after native T2 and pinch Teacher gates.

No policy parameter, optimizer, RNG, counter, physics or reward is changed.
V4 is deliberately rejected by the existing V3 loader. Pre-Kit structural
unwrapping requires an explicit opt-in; after native assets are loaded, callers
must match the original V4 record against the actual runtime adapter contract.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import torch

from .linear_coupling import ACTIVE, JOINTS, MIMIC
from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_bimanual_physical_action_adapter import ADAPTER_VERSION
from .tabletop_vision_action_contract import SQUASHED_VERSION, validate_squashed_checkpoint
from .tabletop_vision_experiment_checkpoint import (
    _same, _state_sha256, _validate_completed, file_sha256,
)


VERSION = 'dh116-vision-student-ppo-v4-physical-adapter'
QUALIFICATION_VERSION = 'dh116-bimanual-native-qualification-v1'
ADAPTER_KEY = 'physical_action_adapter'
QUALIFICATION_KEY = 'physical_adapter_native_qualification'


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError(f'{name} requires a lowercase SHA-256')
    return value


def _adapter(value):
    if not isinstance(value, dict):
        raise ValueError('complete physical adapter dictionary required')
    unhashed = deepcopy(value)
    claimed = _digest(unhashed.pop('sha256', None), 'adapter contract')
    if claimed != _json_hash(unhashed):
        raise ValueError('physical adapter contract self hash mismatch')
    expected = dict(version=ADAPTER_VERSION, actor_size=3844, critic_size=371,
                    action=deepcopy(ACTION_CONTRACT), active_joint_names=list(ACTIVE),
                    full_joint_names=list(JOINTS), affine_mimic=[list(row) for row in MIMIC],
                    frames='actual_runtime_base_link_and_palm_link;wxyz;column_vectors',
                    sdk_mapping='named_ACTIVE_rates;world_translation;seed_intrinsic_XYZ_rates',
                    qualification='independent_native_validation_required', hardware_calibrated=False)
    required = set(expected) | {'source_asset_sha256', 'adapter_source_sha256', 'sha256'}
    if set(value) != required or any(not _same(value.get(k), v) for k, v in expected.items()):
        raise ValueError('unsupported physical adapter dimensions, action or physical semantics')
    _digest(value['adapter_source_sha256'], 'adapter source')
    assets = value['source_asset_sha256']
    if not isinstance(assets, dict) or set(assets) != {'left', 'right'}:
        raise ValueError('independent left and right source asset SHA-256 values required')
    for side, digest in assets.items():
        _digest(digest, side + ' source asset')
    return deepcopy(value)


def _read_report(path):
    path = Path(path).absolute()
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError('native report must be a JSON object')
    return report, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _teacher_qualification(report, identity, adapter_identity):
    if (report.get('status') != 'completed' or report.get('native_runtime_verified') is not True
            or report.get('mode') not in ('teacher_reference', 'teacher_reference_mpc')
            or report.get('student_policy_trained') is not False
            or report.get('teacher_success_is_student_success') is not False
            or type(report.get('ppo_updates')) is not int or report['ppo_updates'] != 0):
        raise ValueError('completed native Teacher-only report required')
    difference = report.get('physical_sdk_adapter_max_executed_action_difference')
    if type(difference) not in (int, float) or difference != 0:
        raise ValueError('native Teacher adapter must preserve executed actions exactly')
    episodes = report.get('episodes')
    if not isinstance(episodes, dict):
        raise ValueError('individual native Teacher episodes required')
    pinch = [row for row in episodes.values() if isinstance(row, dict)
             and row.get('grasp_style') == 'pinch_thumb_index']
    if len(pinch) < 4:
        raise ValueError('at least four native pinch first episodes required')
    envs = [row.get('env_id') for row in pinch]
    if (any(type(index) is not int or index < 0 for index in envs)
            or len(set(envs)) != len(envs)):
        raise ValueError('distinct native Teacher environment identities required')
    for row in pinch:
        terminal = row.get('terminal')
        style = row.get('native_contact_style_audit')
        if (row.get('first_episode') is not True or row.get('native_physics') is not True
                or row.get('success') is not True or row.get('recovered') is not False
                or not isinstance(terminal, dict) or terminal.get('success') is not True
                or not isinstance(style, dict) or style.get('style') != 'pinch_thumb_index'):
            raise ValueError('all native pinch first episodes must have qualified success')
    accepted = sum(isinstance(row, dict) and row.get('success') is True for row in episodes.values())
    if type(report.get('successful_episodes')) is not int or report['successful_episodes'] != accepted:
        raise ValueError('Teacher accepted count disagrees with episode evidence')
    for key in ('native_successful_episodes', 'observed_terminal_successes'):
        if type(report.get(key)) is not int or report[key] < len(pinch):
            raise ValueError('Teacher native terminal count is incomplete')
    return dict(identity, verified_native=True, pinch_first_episodes=len(pinch),
                pinch_first_episode_successes=len(pinch),
                executed_action_difference=0., adapter_contract=adapter_identity,
                teacher_success_is_student_success=False)


def _mapping_qualification(report, identity, adapter):
    if (report.get('status') != 'completed' or type(report.get('cases')) is not int
            or report['cases'] != 53 or report.get('failures') != []):
        raise ValueError('completed 53-case native mapping report without failures required')
    sources = report.get('source_assets')
    if not isinstance(sources, dict) or set(sources) != {'left', 'right'}:
        raise ValueError('native mapping must name both source assets')
    source_identities = {}
    for side, path in sources.items():
        source = Path(path).absolute()
        digest = file_sha256(source)
        if digest != adapter['source_asset_sha256'][side]:
            raise ValueError('native mapping source asset differs from adapter: ' + side)
        source_identities[side] = dict(path=str(source), sha256=digest)
    return dict(identity, cases=53, failures=[], source_assets=source_identities)


def _unchanged_files(identities):
    for item in identities:
        if file_sha256(item['path']) != item['sha256']:
            raise RuntimeError('read-only source or qualification evidence changed: ' + item['path'])


def _preserved_state(saved):
    return {key: value for key, value in saved.items() if key not in ('version', 'contract')}


def _validate_qualification(qualification, adapter, core):
    if not isinstance(qualification, dict):
        raise ValueError('native physical adapter qualification required')
    unhashed = deepcopy(qualification)
    claimed = _digest(unhashed.pop('sha256', None), 'native qualification')
    if _json_hash(unhashed) != claimed:
        raise ValueError('native qualification self hash mismatch')
    if qualification.get('version') != QUALIFICATION_VERSION:
        raise ValueError('unsupported native physical adapter qualification')
    if set(qualification) != {'version', 'source_checkpoint_sha256', 'source_contract_sha256',
                              'source_training_state_sha256', 'teacher', 'mapping', 'sha256'}:
        raise ValueError('unknown native physical adapter qualification fields')
    _digest(qualification.get('source_checkpoint_sha256'), 'source checkpoint')
    if qualification.get('source_contract_sha256') != _state_sha256(core['contract']):
        raise ValueError('physical adapter binding changed original contract or physics')
    # This is provenance for the metadata derivation's source. Future legitimate
    # optimizer updates may change the derived model's training state. Exact
    # preservation is checked against that source during derivation below.
    _digest(qualification.get('source_training_state_sha256'), 'source training state')
    teacher, mapping = qualification.get('teacher'), qualification.get('mapping')
    if not isinstance(teacher, dict) or not isinstance(mapping, dict):
        raise ValueError('native mapping and Teacher qualification records required')
    for record in (teacher, mapping, teacher.get('adapter_contract')):
        if not isinstance(record, dict) or not isinstance(record.get('path'), str) or not record['path']:
            raise ValueError('qualification report path required')
        _digest(record.get('sha256'), 'qualification report')
    count = teacher.get('pinch_first_episodes')
    if (teacher.get('verified_native') is not True or type(count) is not int or count < 4
            or type(teacher.get('pinch_first_episode_successes')) is not int
            or teacher['pinch_first_episode_successes'] != count
            or type(teacher.get('executed_action_difference')) not in (int, float)
            or teacher['executed_action_difference'] != 0
            or teacher.get('teacher_success_is_student_success') is not False
            or type(mapping.get('cases')) is not int or mapping['cases'] != 53
            or mapping.get('failures') != []):
        raise ValueError('native mapping or Teacher gate is incomplete')
    assets = mapping.get('source_assets')
    if not isinstance(assets, dict) or set(assets) != {'left', 'right'}:
        raise ValueError('qualified per-hand source identities required')
    for side in ('left', 'right'):
        if (not isinstance(assets[side], dict) or not isinstance(assets[side].get('path'), str)
                or not assets[side]['path']
                or assets[side].get('sha256') != adapter['source_asset_sha256'][side]):
            raise ValueError('qualified source asset and adapter disagree')


def unwrap_physical_adapter_checkpoint(saved, *, expected_adapter_contract=None, allow_explicit=False):
    """Return a validated core V3 copy; this function never enables execution.

    For V4, ``allow_explicit=True`` permits pre-Kit structural validation.
    The caller must retain the original V4 record and call
    ``validate_physical_adapter_runtime`` once actual native assets are loaded.
    Existing V3 checkpoints remain ordinary V3 and gain no adapter field.
    """
    if not isinstance(saved, dict):
        raise ValueError('complete visual checkpoint dictionary required')
    if saved.get('version') == SQUASHED_VERSION:
        if any(key in saved.get('contract', {}) for key in (ADAPTER_KEY, QUALIFICATION_KEY)):
            raise ValueError('physical adapter metadata requires independent V4 version')
        return validate_squashed_checkpoint(saved)
    if saved.get('version') != VERSION or allow_explicit is not True:
        raise ValueError('V4 physical adapter checkpoint requires explicit opt-in')
    contract = saved.get('contract')
    if not isinstance(contract, dict) or contract.get('version') != VERSION:
        raise ValueError('complete V4 physical adapter contract required')
    adapter = _adapter(contract.get(ADAPTER_KEY))
    if expected_adapter_contract is not None and not _same(adapter, _adapter(expected_adapter_contract)):
        raise ValueError('checkpoint adapter differs from expected runtime adapter')
    core = deepcopy(saved)
    core['version'] = core['contract']['version'] = SQUASHED_VERSION
    core['contract'].pop(ADAPTER_KEY)
    qualification = core['contract'].pop(QUALIFICATION_KEY, None)
    _validate_completed(core)
    _validate_qualification(qualification, adapter, core)
    return core


def validate_physical_adapter_runtime(saved, actual_adapter_contract):
    """Post-Kit exact asset/code/semantics check, before an adapted action."""
    if not isinstance(saved, dict) or saved.get('version') != VERSION:
        raise ValueError('runtime adapter binding requires an explicit V4 checkpoint')
    unwrap_physical_adapter_checkpoint(saved, expected_adapter_contract=actual_adapter_contract,
                                       allow_explicit=True)
    return deepcopy(saved['contract'][ADAPTER_KEY])


def require_physical_adapter_runtime_match(expected_adapter_contract, actual_adapter_contract):
    """Match a previously validated V4 binding without retaining PPO state.

    Inference loaders call ``unwrap_physical_adapter_checkpoint`` first and
    retain only its original V4 adapter dictionary. This post-Kit check reads
    no external qualification reports; current native assets/code supply the
    actual contract. A matching pair returns a fresh expected contract copy.
    """
    expected, actual = _adapter(expected_adapter_contract), _adapter(actual_adapter_contract)
    if not _same(expected, actual):
        raise ValueError('checkpoint adapter differs from actual runtime adapter')
    return expected


def derive_physical_adapter_checkpoint(input_path, output_path, *, expected_source_sha256,
                                       adapter_contract, teacher_report_path, mapping_report_path):
    """Create independent V4 .pt/.json after native gates; preserve V3 state."""
    source, output = (Path(path).absolute() for path in (input_path, output_path))
    sidecar = output.with_suffix('.json')
    if output.suffix != '.pt' or source.resolve() in (output.resolve(), sidecar.resolve()):
        raise ValueError('independent new .pt/.json paths distinct from source required')
    for path in (output, sidecar):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    digest = _digest(expected_source_sha256, 'expected source checkpoint')
    if file_sha256(source) != digest:
        raise ValueError('source checkpoint SHA-256 differs from pinned input')
    saved = torch.load(source, map_location='cpu', weights_only=False)
    _validate_completed(saved)
    if any(key in saved['contract'] for key in (ADAPTER_KEY, QUALIFICATION_KEY)):
        raise ValueError('derivation requires an unbound original V3 contract')
    adapter = _adapter(adapter_contract)
    teacher, teacher_id = _read_report(teacher_report_path)
    bound_teacher, bound_teacher_id = _read_report(Path(teacher_report_path).parent / 'physical_action_adapter_contract.json')
    if not _same(bound_teacher.get('physical_action_adapter'), adapter):
        raise ValueError('native Teacher used a different physical adapter contract')
    mapping, mapping_id = _read_report(mapping_report_path)
    qualification = dict(version=QUALIFICATION_VERSION, source_checkpoint_sha256=digest,
                         source_contract_sha256=_state_sha256(saved['contract']),
                         source_training_state_sha256=_state_sha256(_preserved_state(saved)),
                         teacher=_teacher_qualification(teacher, teacher_id, bound_teacher_id),
                         mapping=_mapping_qualification(mapping, mapping_id, adapter))
    qualification['sha256'] = _json_hash(qualification)
    result = deepcopy(saved)
    result['version'] = result['contract']['version'] = VERSION
    result['contract'][ADAPTER_KEY] = adapter
    result['contract'][QUALIFICATION_KEY] = qualification
    core = unwrap_physical_adapter_checkpoint(result, expected_adapter_contract=adapter, allow_explicit=True)
    if not _same(core, saved):
        raise RuntimeError('unexpected source checkpoint state change')
    identities = [dict(path=str(source), sha256=digest), teacher_id, bound_teacher_id, mapping_id]
    identities.extend(qualification['mapping']['source_assets'].values())
    _unchanged_files(identities)
    report = dict(version=VERSION, status='derived_offline_metadata_only', source_path=str(source),
                  source_sha256=digest, checkpoint_path=str(output),
                  physical_action_adapter=deepcopy(adapter), native_qualification=deepcopy(qualification),
                  teacher_success_is_student_success=False, ppo_updates_performed=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    owned = []
    try:
        with output.open('xb') as checkpoint_stream:
            stat = os.fstat(checkpoint_stream.fileno())
            owned.append((output, stat.st_dev, stat.st_ino))
            with sidecar.open('x', encoding='utf-8') as json_stream:
                stat = os.fstat(json_stream.fileno())
                owned.append((sidecar, stat.st_dev, stat.st_ino))
                torch.save(result, checkpoint_stream)
                checkpoint_stream.flush()
                os.fsync(checkpoint_stream.fileno())
                _unchanged_files(identities)
                report['checkpoint_sha256'] = file_sha256(output)
                json_stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
                json_stream.flush()
                os.fsync(json_stream.fileno())
    except BaseException:
        for path, device, inode in reversed(owned):
            try:
                stat = path.lstat()
                if (stat.st_dev, stat.st_ino) == (device, inode):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    return report
