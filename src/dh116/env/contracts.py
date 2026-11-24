"""Leaf project paths, execution/action/state/identity contracts and the canonical joint coupling model; no runtime or learned policy imports."""
import numpy as linea_coupl__np
import hashlib as runti_io__hashlib
import hashlib as proje_paths__hashlib
from collections.abc import Mapping
from copy import deepcopy
import copy
linea_coupl__ACTIVE = ('finger11', 'finger12', 'finger21', 'finger31', 'finger41', 'finger51')
import json as runti_io__json
import json as proje_paths__json
import os as runti_execu__os
import hashlib as stere_contr__hashlib
import hashlib as surfa_contr__hashlib
JOINTS = ('finger11', 'finger12', 'finger13', 'finger21', 'finger22', 'finger31', 'finger32', 'finger41', 'finger42', 'finger51', 'finger52')
import math as runti_io__math
import sys
runti_execu__VERSION = 'dh116-shared-eager-execution-v1'
import json as stere_contr__json
import json as surfa_contr__json
MIMIC = (('finger13', 'finger12', 0.88178789, -0.00697725445), ('finger22', 'finger21', 0.9206, 0.011072), ('finger32', 'finger31', 0.9206, 0.011072), ('finger42', 'finger41', 0.9206, 0.011072), ('finger52', 'finger51', 0.9206, 0.011072))
import os as runti_io__os
from pathlib import Path as proje_paths__Path

def validate_runtime_execution(*, config=None, model=None):
    """Reject compiler configuration and already compiled modules before binding.

    No compiler is imported, invoked, monkey-patched or globally disabled. Adam
    may import Torch's ordinary internal helpers independently; that does not
    authorize compiling policy or native-runtime graphs.
    """
    geometry = runti_execu__os.environ.get('DH116_TENSOR_GEOMETRY_BACKEND', 'eager')
    if geometry not in ('eager', 'cuda_cabi'):
        raise ValueError('torch.compile/compiled runtime is disabled; choose eager or cuda_cabi')

    def inspect_config(value):
        if isinstance(value, Mapping):
            for (name, item) in value.items():
                if name in ('compile', 'torch_compile', 'compile_backend') and item not in (None, False, ''):
                    raise ValueError('torch.compile configuration is disabled')
                inspect_config(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                inspect_config(item)
    inspect_config(config)
    if model is not None:
        for child in model.modules():
            if getattr(child, '_compiled_call_impl', None) is not None or hasattr(child, '_orig_mod') or hasattr(child, '_torchdynamo_orig_callable'):
                raise ValueError('already compiled policy module is forbidden by shared runtime')
    return dict(version=runti_execu__VERSION, torch_compile=False, geometry_backend=geometry)
import math as stere_contr__math
_SOURCE_LOWER = (0.0, 0.0, 0.0, 0.0, 0.011072, 0.0, 0.011072, 0.0, 0.011072, 0.0, 0.011072)
from pathlib import Path as runti_io__Path
proje_paths__ROOT = proje_paths__Path(__file__).resolve().parents[3]

class EagerRuntimePolicy:
    """Prevent in-place Module.compile on both Teacher and Student policies."""

    def compile(self, *args, **kwargs):
        raise RuntimeError('torch.compile is disabled for the shared physical runtime')
from pathlib import Path as stere_contr__Path
_SOURCE_UPPER = (1.588, 1.03, 1.13, 1.57, 1.950714, 1.57, 1.950714, 1.57, 1.950714, 1.57, 1.950714)
import stat

class ProjectPaths:
    """Resolve explicit, SHA-bound local copies without rewriting archived bytes."""

    def __init__(self, root=proje_paths__ROOT):
        self.root = proje_paths__Path(root).absolute()

    def _relative(self, relative):
        value = proje_paths__Path(relative)
        if value.is_absolute() or '..' in value.parts or (not value.parts):
            raise ValueError('root-relative project path required')
        path = self.root / value
        for part in (self.root, *reversed(path.parents), path):
            if part == self.root or self.root in part.parents:
                if part.is_symlink():
                    raise ValueError('project path cannot traverse a symlink')
        return path

    def file(self, value):
        value = proje_paths__Path(value)
        if not value.is_absolute():
            return self._relative(value)
        if value == self.root or self.root in value.parents:
            return self._relative(value.relative_to(self.root))
        raise ValueError('input must stay below the project root')

    def output(self, value):
        value = proje_paths__Path(value)
        if value.is_absolute():
            try:
                value = value.relative_to(self.root)
            except ValueError as error:
                raise ValueError('output must stay below project runs/ or build/') from error
        if not value.parts or value.parts[0] not in ('runs', 'build'):
            raise ValueError('generated output must stay below project runs/ or build/')
        return self._relative(value)
import uuid
_MANUAL_UPPER_DEG = (80.0, 90.0, 70.0, 70.0, 70.0, 70.0)
runti_io__ROOT = runti_io__Path(__file__).resolve().parents[3]

def runtime_file(value):
    return ProjectPaths().file(value)
import numpy as stere_contr__np

def active_limits():
    """Return fresh lower/upper rad arrays for this linear simulation model.

    Intersect V2 manual active limits, source joint limits, and all affine
    follower limits. These are not collision clearance or hardware approval.
    The Excel table domain does not apply to this independent URDF model.
    """
    indices = [JOINTS.index(name) for name in linea_coupl__ACTIVE]
    lower = linea_coupl__np.array(_SOURCE_LOWER)[indices]
    upper = linea_coupl__np.minimum(linea_coupl__np.array(_SOURCE_UPPER)[indices], linea_coupl__np.deg2rad(_MANUAL_UPPER_DEG))
    for (follower, master, ratio, offset) in MIMIC:
        (i, j) = (linea_coupl__ACTIVE.index(master), JOINTS.index(follower))
        lower[i] = max(lower[i], (_SOURCE_LOWER[j] - offset) / ratio)
        upper[i] = min(upper[i], (_SOURCE_UPPER[j] - offset) / ratio)
    return (lower, upper)

def runti_io__digest(value):
    return runti_io__hashlib.sha256(runti_io__json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def coupling_matrix(names=JOINTS):
    """dq_full / dq_active: named full-joint rows, six active columns."""
    if len(names) != 11 or len(set(names)) != 11 or set(names) != set(JOINTS):
        raise ValueError('expected exactly the eleven DH116 joints')
    matrix = linea_coupl__np.zeros((11, 6))
    for (i, name) in enumerate(names):
        if name in linea_coupl__ACTIVE:
            matrix[i, linea_coupl__ACTIVE.index(name)] = 1.0
        else:
            (_, master, ratio, _) = next((m for m in MIMIC if m[0] == name))
            matrix[i, linea_coupl__ACTIVE.index(master)] = ratio
    return matrix

def checked_file(path, expected):
    path = runti_io__Path(path)
    if not path.is_absolute():
        path = runti_io__ROOT / path
    if not isinstance(expected, str) or len(expected) != 64 or any((c not in '0123456789abcdef' for c in expected)) or path.is_symlink() or (not stat.S_ISREG(path.stat().st_mode)):
        raise ValueError('actual regular file and full SHA required: ' + str(path))
    raw = path.read_bytes()
    if runti_io__hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('actual file SHA differs: ' + str(path))
    return raw

def expand(q):
    """Expand six active radians; reject infeasible states without clamping.

    In particular the thumb master cannot be zero: its affine offset would
    violate the follower's zero lower limit. Caller chooses a feasible state.
    """
    q = linea_coupl__np.asarray(q, dtype=linea_coupl__np.float64)
    (lower, upper) = active_limits()
    if q.shape != (6,) or not linea_coupl__np.isfinite(q).all():
        raise ValueError('expected six finite active radians')
    if linea_coupl__np.any(q < lower) or linea_coupl__np.any(q > upper):
        raise ValueError('active state violates DH116 coupled limits')
    full = coupling_matrix() @ q
    for (name, _, _, offset) in MIMIC:
        full[JOINTS.index(name)] += offset
    return full

def write_new(path, value):
    """Durably create a new result; never overwrite an earlier run."""
    path = runti_io__Path(path)
    with path.open('x', encoding='utf-8') as stream:
        runti_io__json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        runti_io__os.fsync(stream.fileno())

def fold_jacobian(full_jacobian, joint_names):
    """Fold (..., spatial_rows, 11) to (..., spatial_rows, 6), same frame.

    Input must already refer to the desired link/contact point, not an
    unshifted COM Jacobian. This function does not change the point or frame.
    """
    j = linea_coupl__np.asarray(full_jacobian, dtype=linea_coupl__np.float64)
    if j.ndim < 2 or j.shape[-1] != 11 or j.shape[-2] not in (3, 6) or (not linea_coupl__np.isfinite(j).all()):
        raise ValueError('expected finite (...,3|6,11) Jacobian')
    return j @ coupling_matrix(joint_names)

def clock(sim):
    try:
        return {'sdk_step': int(sim.current_time_step_index), 'sdk_time_s': float(sim.current_time)}
    finally:
        sim = None

def runti_io__file_sha256(path):
    digest = runti_io__hashlib.sha256()
    with runti_io__Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
surfa_contr__ACTIVE = linea_coupl__ACTIVE
LEGACY_OBSERVATION_VERSION = 'dh116-surface-sharing-lift-objective-v2'
OBSERVATION_VERSION = 'dh116-surface-sharing-feedback-371-v1'
ACTION_NAME = '24D_free_fingers_worldXYZ_intrinsicXYZ_surface_sharing_v1'
surfa_contr__ACTION_CONTRACT = dict(version='dh116-surface-motion24-v1', size=24, joint_order=[hand + '.' + joint for hand in ('left', 'right') for joint in surfa_contr__ACTIVE], slices=dict(joint_velocity=[0, 12], world_translation_velocity=[12, 18], intrinsic_rotation_velocity=[18, 24]), hand_order=['left', 'right'], axis_order=['X', 'Y', 'Z'], joint_rate_rad_s=0.3, translation_rate_m_s=0.01, rotation_rate_rad_s=0.5, normalized_range=[-1.0, 1.0], wrist_vector_norm_limit=1.0, translation_frame='world;positive_Z_up', rotation='seed_wxyz_times_intrinsic_X_then_Y_then_Z', integration='existing_native_physics_dt;targets_integrated_at_each_substep', zero_action='retain_current_targets_subject_to_existing_joint_lead_and_table_safety_projection', guards='existing_joint_limits_lead_velocity_table_clearance;no_added_force_servo', force_action_dimensions=0)
_FIELDS = (('support_n', 2, 10.0, 'N', 'accepted_pad_normal_plus_attributable_native_friction'), ('support_margin', 1, 1.0, 'weight_fraction', '(sum_signed_support-object_weight)/object_weight'), ('support_imbalance', 1, 1.0, 'ratio', 'absolute_bilateral_support_difference/sum_absolute_support'), ('paired', 2, 1.0, 'boolean', 'native_effective_opposed_pad_pair_per_hand'), ('ordinary_fraction', 1, 1.0, 'ratio', 'rejected_native_load/(accepted+rejected)'), ('pad_depth', 1, 0.0008, 'm', 'maximum_accepted_native_pad_depth'), ('anchor_drift', 1, 0.0005, 'm', 'native_fixed_anchor_drift'), ('slip_speed', 1, 0.01, 'm/s', 'rod_points_in_measured_wrists_velocity'), ('hold_duration', 1, 5.0, 's', 'last_completed_native_success_window'), ('window_good', 1, 1.0, 'boolean', 'last_completed_native_success_window'), ('pressure_limited', 2, 1.0, 'boolean', 'actual_existing_controller_pressure_limited_flags'), ('mimic_blocked', 2, 1.0, 'boolean', 'actual_existing_block_flags_with_severe_measured_mimic_error'), ('wrist_z_request', 2, 0.01, 'm/s', 'submitted_policy_action_14_17_times_existing_wrist_rate'), ('wrist_z_increment', 2, 0.001, 'm', 'measured_wrist_pose_end_minus_start'))

def feedback_schema():
    start = 351
    fields = []
    for (name, size, scale, unit, source) in _FIELDS:
        fields.append(dict(name=name, slice=[start, start + size], scale=scale, unit=unit, source=source))
        start += size
    assert start == 371
    return fields

def semantic_sha256(contract):
    """Digest full semantics; only known batch-size repetitions are canonical."""
    value = copy.deepcopy(contract)
    value.pop('surface_contract_sha256', None)
    for key in ('num_envs', 'rollout_steps', 'minibatches'):
        value.pop(key, None)
    for (record, key) in ((value, 'success_bottom_m'), (value.get('surface_force_sharing', {}), 'lift_reference_bottom_m')):
        heights = record.get(key)
        if isinstance(heights, list) and heights and all((type(x) in (int, float) and x == heights[0] for x in heights)):
            record[key] = {'uniform_value': heights[0]}
    return surfa_contr__hashlib.sha256(surfa_contr__json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _schema():
    return dict(version='dh116-surface-contract-v1', action=copy.deepcopy(surfa_contr__ACTION_CONTRACT), base_observation=dict(size=351, version=LEGACY_OBSERVATION_VERSION, normalization='unchanged_legacy_351_columns;no_running_statistics', previous_action='existing_351_free_wrist_action_columns'), feedback_fields=feedback_schema(), alignment='observation_t_contains_completed_interval_t-1;action_t_executes_next_interval', aggregation='native_dt_weighted_mean;depth_and_drift_max;hold_at_last_completed_window', reset='selected_rows_zero;no_pre_reset_samples_in_next_interval', invalid='zero_diagnostic_features_and_abort_rollout_before_any_PPO_or_BC_update', clipping='none;physical_scales_only;finite_out_of_scale_values_remain_observable')

def _validate_motion(contract):
    if not isinstance(contract, dict):
        raise ValueError('surface contract dictionary required')
    dimensions = [contract[k] for k in ('action_size', 'policy_command_size') if k in contract]
    if not dimensions or any((type(x) is not int or x != 24 for x in dimensions)):
        raise ValueError('explicit 24D motion contract required; no 26D conversion')
    if contract.get('action') != ACTION_NAME:
        raise ValueError('unknown surface action order/semantics')
    expected_order = 'joint_velocity12_world_translation_velocity6_intrinsic_XYZ_velocity6'
    for key in ('policy_control', 'grasp_controller', 'free_wrist_grasp'):
        nested = contract.get(key, {})
        if 'action_size' in nested and nested['action_size'] != 24 or ('action_order' in nested and nested['action_order'] != expected_order):
            raise ValueError('contradictory 24D action order: ' + key)
        for (field, expected) in (('joint_rate_rad_s', 0.3), ('wrist_rate_m_s', 0.01), ('rotation_rate_rad_s', 0.5)):
            if field in nested and nested[field] != expected:
                raise ValueError('action scale mismatch: ' + field)
    if contract.get('control_hz', 60) != 60 or contract.get('physics_hz', 480) != 480:
        raise ValueError('audited 480Hz physics / 60Hz control required')

def bind_contract(legacy_contract):
    """Explicit metadata derivation; never converts actions or fabricates inputs."""
    _validate_motion(legacy_contract)
    if legacy_contract.get('observation_size') != 351 or legacy_contract.get('observation_version') != LEGACY_OBSERVATION_VERSION:
        raise ValueError('only the original versioned 351D input can be explicitly extended')
    if any((k in legacy_contract for k in ('surface_contract', 'surface_contract_sha256'))):
        raise ValueError('source already has surface schema metadata; ambiguous migration')
    value = copy.deepcopy(legacy_contract)
    value.update(observation_size=371, observation_version=OBSERVATION_VERSION, surface_contract=_schema())
    value['surface_contract_sha256'] = semantic_sha256(value)
    return value

def surfa_contr__validate_contract(contract):
    _validate_motion(contract)
    if contract.get('observation_size') != 371 or contract.get('observation_version') != OBSERVATION_VERSION:
        raise ValueError('versioned 371D feedback contract required')
    if contract.get('surface_contract') != _schema():
        raise ValueError('surface action/observation schema mismatch')
    if contract.get('surface_contract_sha256') != semantic_sha256(contract):
        raise ValueError('surface contract SHA mismatch')
    return contract

def validate_actions(actions):
    import torch
    if not isinstance(actions, torch.Tensor) or actions.ndim != 2 or actions.shape[-1] != 24 or (not actions.is_floating_point()) or (not torch.isfinite(actions).all()) or (actions.abs() > 1).any():
        raise ValueError('finite bounded normalized [N,24] actions required; no silent conversion')
    return actions
stere_contr__ACTION_CONTRACT = surfa_contr__ACTION_CONTRACT
stere_contr__VERSION = 'dh116-stereo-rgb-teacher-student-v1'
STYLES = ('pinch_thumb_index', 'tripod_thumb_index_middle')
stere_contr__ROOT = stere_contr__Path(__file__).resolve().parents[3]
IDENTITY_KEYS = ('run_uuid', 'env_id', 'episode_uid', 'reset_id', 'policy_tick', 'physics_tick_start', 'physics_tick_end', 'simulation_timestamp', 'frame_id', 'calibration_sha256')

def canonical_json(value):
    return stere_contr__json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def stere_contr__digest(value):
    return stere_contr__hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()

def stere_contr__file_sha256(path):
    h = stere_contr__hashlib.sha256()
    with stere_contr__Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require_sha(value, name='SHA256'):
    if not isinstance(value, str) or len(value) != 64 or any((c not in '0123456789abcdef' for c in value)):
        raise ValueError(f'{name} must be lowercase SHA256')
    return value
_ESTIMATOR_STATE_V1_CRLF = 'a7af9f3f214a646f2f95dce45aa4744f378488de34237431c914bd4137b2ee86'
_ESTIMATOR_STATE_V1_LF = '9f145cf8e68b1dfe8e2c7464ec5fc0cb00e81a2707115c2fe148ab49d77bbb62'

def verify_estimator_state_contract_file(path, expected_sha256):
    """Check the v1 config's file pin and return the actual raw-byte SHA.

    The exception applies only to the published v1 CRLF pin on its audited LF
    checkout. Checkpoint identities still bind their actual bytes.
    """
    expected_sha256 = require_sha(expected_sha256, 'state contract file SHA256')
    actual = stere_contr__file_sha256(path)
    pinned = expected_sha256 == _ESTIMATOR_STATE_V1_CRLF
    if not pinned or actual not in (_ESTIMATOR_STATE_V1_CRLF, _ESTIMATOR_STATE_V1_LF):
        raise ValueError('state contract file SHA256 mismatch')
    return actual

def action_sha256():
    return stere_contr__digest(stere_contr__ACTION_CONTRACT)

def physical_initial_sha256(initial):
    """Actual state identity, 10um/10urad quantization; exclude clone origin/seed.

    This is numerical distinctness at a declared tolerance, not a guarantee of
    broad task diversity. The producer must supply real reset readback.
    """
    if not isinstance(initial, dict):
        raise ValueError('actual physical initial condition required')
    sizes = {'object_pose': (7,), 'palm_pose': (2, 7), 'joint_position': (12,), 'joint_velocity': (12,), 'object_velocity': (6,)}
    result = {}
    for (name, shape) in sizes.items():
        try:
            value = stere_contr__np.asarray(initial[name], dtype=float).copy()
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError('missing physical initial ' + name) from error
        if value.shape != shape or not stere_contr__np.isfinite(value).all():
            raise ValueError('invalid physical initial ' + name)
        result[name] = value
    if initial.get('coordinate_frame') == 'world':
        origin = stere_contr__np.asarray(initial.get('environment_origin'), dtype=float)
        if origin.shape != (3,) or not stere_contr__np.isfinite(origin).all():
            raise ValueError('physical initial world origin required')
        result['object_pose'][:3] -= origin
        result['palm_pose'][:, :3] -= origin
    elif initial.get('coordinate_frame') != 'environment_local':
        raise ValueError('physical initial coordinate frame required')
    for pose in [result['object_pose'], *result['palm_pose']]:
        q = pose[3:]
        length = stere_contr__np.linalg.norm(q)
        if length <= 1e-12:
            raise ValueError('physical initial quaternion invalid')
        q /= length
        significant = stere_contr__np.flatnonzero(stere_contr__np.abs(q) > 1e-12)
        if len(significant) and q[significant[0]] < 0:
            q *= -1
    value = {key: stere_contr__np.rint(array / 1e-05).astype(stere_contr__np.int64).tolist() for (key, array) in result.items()}
    if 'physical_parameters' in initial:
        parameters = initial['physical_parameters']
        allowed = {'mass_kg', 'friction', 'young_modulus_pa', 'damping', 'geometry_sha256', 'material_sha256'}
        if not isinstance(parameters, dict) or not set(parameters) <= allowed:
            raise ValueError('only declared physical parameters may affect independent initial-state identity')
        value['physical_parameters'] = parameters
    return stere_contr__digest(value)

def state_contract(path=None):
    path = stere_contr__ROOT / 'configs/contracts/privileged_state.json' if path is None else stere_contr__Path(path)
    value = stere_contr__json.loads(path.read_text(encoding='utf-8'))
    if value.get('size') != 371 or [r.get('index') for r in value.get('dimensions', [])] != list(range(371)):
        raise ValueError('audited 371D state contract required')
    return value

def make_contract(kind, config, *, state_schema=None, calibration_sha256=None):
    if kind not in ('teacher', 'student', 'bc', 'dagger'):
        raise ValueError('unknown independent model kind')
    schema = state_contract() if state_schema is None else state_schema
    if schema.get('size') != 371:
        raise ValueError('371D privileged contract required')
    value = dict(version=stere_contr__VERSION, kind=kind, privileged_size=371, action_size=24, action_contract=deepcopy(stere_contr__ACTION_CONTRACT), action_contract_sha256=action_sha256(), state_schema_sha256=stere_contr__digest(schema), config=deepcopy(config), actor_input='privileged_371' if kind == 'teacher' else 'stereo_RGB_and_proprio_only', depth_source='none' if kind == 'teacher' else 'computed_from_synchronized_left_right_RGB', old_3844_observation_reinterpreted=False, success_contract=dict(net_lift_m=0.03, qualified_hold_s=5.0, formal_slip_m=0.0005, additional_gates='unchanged_native_SurfaceLiftSuccess_and_environment_guards'))
    if calibration_sha256 is not None:
        value['calibration_sha256'] = require_sha(calibration_sha256, 'calibration SHA256')
    value['contract_sha256'] = stere_contr__digest(value)
    return value

def stere_contr__validate_contract(value, *, kind=None, expected=None):
    if not isinstance(value, dict) or value.get('version') != stere_contr__VERSION:
        raise ValueError('independent stereo checkpoint version required; legacy conversion is forbidden')
    if value.get('kind') not in ('teacher', 'student', 'bc', 'dagger') or (kind is not None and value['kind'] != kind):
        raise ValueError('model kind mismatch')
    if value.get('action_size') != 24 or value.get('action_contract') != stere_contr__ACTION_CONTRACT or value.get('action_contract_sha256') != action_sha256():
        raise ValueError('canonical action contract mismatch')
    if value.get('privileged_size') != 371 or value.get('state_schema_sha256') != stere_contr__digest(state_contract()):
        raise ValueError('371D state schema mismatch')
    if value.get('old_3844_observation_reinterpreted') is not False:
        raise ValueError('legacy observation contract must remain unchanged')
    expected_input = 'privileged_371' if value['kind'] == 'teacher' else 'stereo_RGB_and_proprio_only'
    if value.get('actor_input') != expected_input:
        raise ValueError('actor input leakage or model contract mismatch')
    unhashed = {k: v for (k, v) in value.items() if k != 'contract_sha256'}
    if value.get('contract_sha256') != stere_contr__digest(unhashed):
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
    if type(when) not in (int, float) or not stere_contr__math.isfinite(when) or when < 0:
        raise ValueError('finite nonnegative simulation timestamp required')
    require_sha(row['calibration_sha256'], 'calibration SHA256')
    return row

def require_aligned(sensor_identity, teacher_identity):
    validate_identity(sensor_identity)
    validate_identity(teacher_identity)
    if any((sensor_identity[key] != teacher_identity[key] for key in IDENTITY_KEYS)):
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
        if len(set(ids)) != len(ids) or any((type(i) is not int or not 0 <= i < self.num_envs for i in ids)):
            raise ValueError('unique valid reset environment indices required')
        for i in ids:
            self.reset_ids[i] += 1

def new_run(path):
    path = stere_contr__Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path

def write_json_new(path, value):
    content = stere_contr__json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with stere_contr__Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)

def require_opt_in(enabled):
    if enabled is not True:
        raise ValueError('new stereo experiment is disabled; explicit --enable-experiment required')

from pathlib import Path as _SourcePath
PROJECT_ROOT = _SourcePath(__file__).resolve().parents[3]
def source_file(relative):
    value=_SourcePath(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("current package-relative source file required")
    path=PROJECT_ROOT/"src/dh116"/value
    if path.is_symlink() or not path.is_file():
        raise ValueError("current source missing or symlinked: "+str(value))
    return path
