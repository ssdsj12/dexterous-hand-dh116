"""Independent stereo contracts and identities; no simulator or policy imports.

Schema checks never establish physical qualification. The legacy 3844D RGB-D
observation and every existing physical controller remain unchanged.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

from .tabletop_surface_contract import ACTION_CONTRACT

VERSION = 'dh116-stereo-rgb-teacher-student-v1'
STYLES = ('pinch_thumb_index', 'tripod_thumb_index_middle')
ROOT = Path(__file__).resolve().parents[4]
IDENTITY_KEYS = ('run_uuid', 'env_id', 'episode_uid', 'reset_id', 'policy_tick',
    'physics_tick_start', 'physics_tick_end', 'simulation_timestamp',
    'frame_id', 'calibration_sha256')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_sha(value, name='SHA256'):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be lowercase SHA256')
    return value


# The estimator v1 config was pinned from a CRLF checkout. Its committed LF
# counterpart is the only permitted alternate, with identical non-EOL bytes.
_ESTIMATOR_STATE_V1_CRLF = 'a7af9f3f214a646f2f95dce45aa4744f378488de34237431c914bd4137b2ee86'
_ESTIMATOR_STATE_V1_LF = '9f145cf8e68b1dfe8e2c7464ec5fc0cb00e81a2707115c2fe148ab49d77bbb62'


def verify_estimator_state_contract_file(path, expected_sha256):
    """Check the v1 config's file pin and return the actual raw-byte SHA.

    The exception applies only to the published v1 CRLF pin on its audited LF
    checkout. Checkpoint identities still bind their actual bytes.
    """
    expected_sha256 = require_sha(expected_sha256, 'state contract file SHA256')
    actual = file_sha256(path)
    pinned = expected_sha256 == _ESTIMATOR_STATE_V1_CRLF
    if not pinned or actual not in (_ESTIMATOR_STATE_V1_CRLF, _ESTIMATOR_STATE_V1_LF):
        raise ValueError('state contract file SHA256 mismatch')
    return actual


def action_sha256():
    return digest(ACTION_CONTRACT)


def physical_initial_sha256(initial):
    """Actual state identity, 10um/10urad quantization; exclude clone origin/seed.

    This is numerical distinctness at a declared tolerance, not a guarantee of
    broad task diversity. The producer must supply real reset readback.
    """
    if not isinstance(initial, dict):
        raise ValueError('actual physical initial condition required')
    sizes = {'object_pose': (7,), 'palm_pose': (2, 7), 'joint_position': (12,),
        'joint_velocity': (12,), 'object_velocity': (6,)}
    result = {}
    for name, shape in sizes.items():
        try:
            value = np.asarray(initial[name], dtype=float).copy()
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError('missing physical initial ' + name) from error
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('invalid physical initial ' + name)
        result[name] = value
    if initial.get('coordinate_frame') == 'world':
        origin = np.asarray(initial.get('environment_origin'), dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('physical initial world origin required')
        result['object_pose'][:3] -= origin
        result['palm_pose'][:, :3] -= origin
    elif initial.get('coordinate_frame') != 'environment_local':
        raise ValueError('physical initial coordinate frame required')
    for pose in [result['object_pose'], *result['palm_pose']]:
        q = pose[3:]; length = np.linalg.norm(q)
        if length <= 1e-12:
            raise ValueError('physical initial quaternion invalid')
        q /= length
        significant = np.flatnonzero(np.abs(q) > 1e-12)
        if len(significant) and q[significant[0]] < 0:
            q *= -1
    value = {key: np.rint(array/1e-5).astype(np.int64).tolist() for key, array in result.items()}
    if 'physical_parameters' in initial:
        parameters = initial['physical_parameters']
        allowed = {'mass_kg', 'friction', 'young_modulus_pa', 'damping', 'geometry_sha256', 'material_sha256'}
        if not isinstance(parameters, dict) or not set(parameters) <= allowed:
            raise ValueError('only declared physical parameters may affect independent initial-state identity')
        value['physical_parameters'] = parameters
    return digest(value)


def state_contract(path=None):
    path = ROOT / 'docs/contracts/stereo_privileged_state_contract.json' if path is None else Path(path)
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('size') != 371 or [r.get('index') for r in value.get('dimensions', [])] != list(range(371)):
        raise ValueError('audited 371D state contract required')
    return value


def make_contract(kind, config, *, state_schema=None, calibration_sha256=None):
    if kind not in ('teacher', 'student', 'bc', 'dagger'):
        raise ValueError('unknown independent model kind')
    schema = state_contract() if state_schema is None else state_schema
    if schema.get('size') != 371:
        raise ValueError('371D privileged contract required')
    value = dict(version=VERSION, kind=kind, privileged_size=371, action_size=24,
        action_contract=deepcopy(ACTION_CONTRACT), action_contract_sha256=action_sha256(),
        state_schema_sha256=digest(schema), config=deepcopy(config),
        actor_input='privileged_371' if kind == 'teacher' else 'stereo_RGB_and_proprio_only',
        depth_source='none' if kind == 'teacher' else 'computed_from_synchronized_left_right_RGB',
        old_3844_observation_reinterpreted=False,
        success_contract=dict(net_lift_m=.030, qualified_hold_s=5., formal_slip_m=.0005,
            additional_gates='unchanged_native_SurfaceLiftSuccess_and_environment_guards'))
    if calibration_sha256 is not None:
        value['calibration_sha256'] = require_sha(calibration_sha256, 'calibration SHA256')
    value['contract_sha256'] = digest(value)
    return value


def validate_contract(value, *, kind=None, expected=None):
    if not isinstance(value, dict) or value.get('version') != VERSION:
        raise ValueError('independent stereo checkpoint version required; legacy conversion is forbidden')
    if value.get('kind') not in ('teacher', 'student', 'bc', 'dagger') or kind is not None and value['kind'] != kind:
        raise ValueError('model kind mismatch')
    if value.get('action_size') != 24 or value.get('action_contract') != ACTION_CONTRACT or value.get('action_contract_sha256') != action_sha256():
        raise ValueError('canonical action contract mismatch')
    if value.get('privileged_size') != 371 or value.get('state_schema_sha256') != digest(state_contract()):
        raise ValueError('371D state schema mismatch')
    if value.get('old_3844_observation_reinterpreted') is not False:
        raise ValueError('legacy observation contract must remain unchanged')
    expected_input = 'privileged_371' if value['kind'] == 'teacher' else 'stereo_RGB_and_proprio_only'
    if value.get('actor_input') != expected_input:
        raise ValueError('actor input leakage or model contract mismatch')
    unhashed = {k: v for k, v in value.items() if k != 'contract_sha256'}
    if value.get('contract_sha256') != digest(unhashed):
        raise ValueError('contract SHA mismatch')
    if expected is not None and canonical_json(value) != canonical_json(expected):
        raise ValueError('complete experiment contract mismatch')
    return value


def validate_identity(row):
    if not isinstance(row, dict) or not set(IDENTITY_KEYS) <= row.keys():
        raise ValueError('complete pre-action identity required')
    for key in ('run_uuid', 'episode_uid'):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f'nonempty {key} required')
    for key in ('env_id', 'reset_id', 'policy_tick', 'physics_tick_start', 'physics_tick_end', 'frame_id'):
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('episode_policy_tick', 'trial_episode_index'):
        if key in row and (type(row[key]) is not int or row[key] < 0):
            raise ValueError(f'{key} must be a nonnegative integer')
    if row['physics_tick_start'] > row['physics_tick_end']:
        raise ValueError('invalid physics tick range')
    when = row['simulation_timestamp']
    if type(when) not in (int, float) or not math.isfinite(when) or when < 0:
        raise ValueError('finite nonnegative simulation timestamp required')
    require_sha(row['calibration_sha256'], 'calibration SHA256')
    return row


def require_aligned(sensor_identity, teacher_identity):
    validate_identity(sensor_identity)
    validate_identity(teacher_identity)
    if any(sensor_identity[key] != teacher_identity[key] for key in IDENTITY_KEYS):
        raise ValueError('Teacher and Student must be exactly pre-action aligned')
    return True


class EpisodeIdentities:
    """Run-local identities do not depend on resetting policy/physics clocks."""
    def __init__(self, num_envs, *, run_uuid=None):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError('positive num_envs required')
        self.num_envs = num_envs
        self.run_uuid = str(uuid.uuid4()) if run_uuid is None else run_uuid
        if not isinstance(self.run_uuid, str) or not self.run_uuid:
            raise ValueError('run UUID required')
        self.reset_ids = [0] * num_envs

    def uid(self, env_id):
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise ValueError('environment index out of range')
        return f'{self.run_uuid}:{env_id}:{self.reset_ids[env_id]}'

    def reset(self, env_ids):
        ids = list(env_ids)
        if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < self.num_envs for i in ids):
            raise ValueError('unique valid reset environment indices required')
        for i in ids:
            self.reset_ids[i] += 1


def new_run(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json_new(path, value):
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)


def require_opt_in(enabled):
    if enabled is not True:
        raise ValueError('new stereo experiment is disabled; explicit --enable-experiment required')
