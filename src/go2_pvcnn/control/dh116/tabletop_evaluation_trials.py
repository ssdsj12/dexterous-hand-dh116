"""CPU contracts for authored cold-reset cases and first-terminal telemetry.

This module neither starts physics nor restores a mid-PhysX checkpoint. State
fields mirror TensorWorld.reset: all named articulation q/qd, local roots and
root velocities, and the rigid object's local pose/velocity. Joint names also
include the external wrist drives when present (tabletop_palm_mount). Origins
are placement labels outside the physical payload; poses inside it are local.

An authored manifest remains ``authored_unverified`` even after collection.
Native reset/readback certification must be supplied by a separate native
runner/report; no user-set flag in this schema can manufacture that evidence.
"""
from copy import deepcopy
import hashlib
import json
import math
import struct


SCHEMA_VERSION = 'dh116-fixed-trials-v1'
GRASP_STYLES = ('pinch_thumb_index', 'tripod_thumb_index_middle')
_FINGERS = frozenset(('finger11', 'finger12', 'finger13', 'finger21', 'finger22',
    'finger31', 'finger32', 'finger41', 'finger42', 'finger51', 'finger52'))
_LINEAR_WRIST = frozenset(('wrist_x', 'wrist_y', 'wrist_z'))
_ANGULAR_WRIST = frozenset(('wrist_roll', 'wrist_pitch', 'wrist_yaw'))
_JOINT_SETS = (_FINGERS, _FINGERS | _LINEAR_WRIST,
               _FINGERS | _LINEAR_WRIST | _ANGULAR_WRIST)
_PROVENANCE = ('generator_sha256', 'source_sha256', 'physics_sha256',
               'camera_sha256', 'template_sha256')
_STATE_FIELDS = ('joint_names', 'joint_positions', 'joint_velocities', 'roots',
                 'root_velocities', 'wire_pose', 'wire_velocity')
_METRICS = ('lift', 'hold', 'slip', 'contact', 'torque', 'action')
_SAMPLE_FIELDS = ('dt_s', 'measurement_valid', 'lift_m', 'qualified_hold_s',
    'slip_m', 'bilateral_opposed_contact', 'estimated_torque_nm',
    'effort_limit_nm', 'action')


def _keys(value, required, optional=(), *, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an object')
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f'{name}: missing {sorted(missing)}, unknown {sorted(unknown)}')


def _number(value, name, *, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if (not math.isfinite(result) or (minimum is not None and result < minimum)
            or (positive and result <= 0)):
        raise ValueError(f'{name} must be finite and in range')
    return 0. if result == 0. else result


def _vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f'{name} must have shape ({size},)')
    return [_number(x, name) for x in value]


def _matrix(value, rows, columns, name):
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f'{name} must have {rows} rows')
    return [_vector(row, columns, name) for row in value]


def _pose(value, name):
    # Validate and choose the quaternion sign in the native writer's domain.
    # A leading component may underflow to zero on float32 conversion; choosing
    # its sign first would make equivalent writer states hash differently.
    result = [_physics_float32(x) for x in _vector(value, 7, name)]
    norm = math.hypot(*result[3:])
    if abs(norm - 1.) > 1e-6:
        raise ValueError(f'{name} requires a unit wxyz quaternion')
    sign = next((1. if x > 0 else -1. for x in result[3:] if x != 0.), 1.)
    # Check the written unit quaternion without repeatedly renormalizing its
    # bytes: division by a rounded norm is not idempotent. Sign canonicalizing
    # is exact and q/-q represent the same orientation. All other supplied
    # physical numbers subsequently use the native writer's float32 precision.
    result[3:] = [0. if x == 0. else sign * x for x in result[3:]]
    return result


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError(f'{name} must be a lowercase SHA256')
    return value


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _physics_float32(value):
    """Canonical values actually submitted by TensorWorld.reset's .float()."""
    try:
        result = struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError('physical state must be finite in native float32 precision') from exc
    if not math.isfinite(result):
        raise ValueError('physical state must be finite in native float32 precision')
    return 0. if result == 0. else result


def _physical_state(value):
    _keys(value, _STATE_FIELDS, name='physical_state')
    names = value['joint_names']
    if not isinstance(names, list) or len(names) != 2:
        raise ValueError('joint_names must name both complete articulations')
    result = dict(joint_names=[], joint_positions=[], joint_velocities=[])
    for key in ('joint_positions', 'joint_velocities'):
        if not isinstance(value[key], list) or len(value[key]) != 2:
            raise ValueError(f'{key} requires both complete articulations')
    for hi, hand_names in enumerate(names):
        if (not isinstance(hand_names, list) or not all(isinstance(n, str) for n in hand_names)
                or len(set(hand_names)) != len(hand_names) or set(hand_names) not in _JOINT_SETS):
            raise ValueError('complete DH116 and optional complete wrist joint_names required')
        ordered = sorted(hand_names)
        indices = [hand_names.index(n) for n in ordered]
        result['joint_names'].append(ordered)
        for key in ('joint_positions', 'joint_velocities'):
            q = _vector(value[key][hi], len(hand_names), key)
            result[key].append([q[i] for i in indices])
    if set(names[0]) != set(names[1]):
        raise ValueError('both hands require the same complete articulation kind')
    roots = value['roots']
    if not isinstance(roots, list) or len(roots) != 2:
        raise ValueError('roots requires two local poses')
    result.update(roots=[_pose(p, 'roots') for p in roots],
        root_velocities=_matrix(value['root_velocities'], 2, 6, 'root_velocities'),
        wire_pose=_pose(value['wire_pose'], 'wire_pose'),
        wire_velocity=_vector(value['wire_velocity'], 6, 'wire_velocity'))
    for key in ('joint_positions', 'joint_velocities', 'roots', 'root_velocities'):
        result[key] = [[_physics_float32(x) for x in row] for row in result[key]]
    for key in ('wire_pose', 'wire_velocity'):
        result[key] = [_physics_float32(x) for x in result[key]]
    return result


def state_sha256(physical_state):
    """Hash physical local state at native float32 reset write precision.

    IDs, grasp labels and environment placement origins are deliberately not
    accepted here. Joint order, quaternion sign, signed zero and sub-float32
    differences cannot create independent cases. This is authored write-state
    identity, not a settled PhysX/readback state or solver checkpoint identity.
    """
    return _digest(_physical_state(physical_state))


def validate_manifest(payload, *, min_trials_per_style=100):
    """Return an owned, normalized manifest, rejecting duplicate physical cases.

    Production ``validation`` always requires >=100 cases per style; reduced
    minimums are allowed only for an explicitly tagged ``test`` fixture.
    This is schema/identity validation, never native state certification.
    """
    _keys(payload, ('schema_version', 'metadata', 'trials'), name='manifest')
    if payload['schema_version'] != SCHEMA_VERSION:
        raise ValueError('unknown fixed-trial schema_version')
    metadata = payload['metadata']
    _keys(metadata, ('reset_kind', 'seed', 'split', 'validation_status') + _PROVENANCE,
          name='metadata')
    if (metadata['reset_kind'] != 'cold_authored_reset'
            or metadata['validation_status'] != 'authored_unverified'):
        raise ValueError('only cold_authored_reset / authored_unverified is supported')
    if metadata['split'] not in ('validation', 'test'):
        raise ValueError('split must be validation or test')
    if (isinstance(metadata['seed'], bool) or not isinstance(metadata['seed'], int)
            or metadata['seed'] < 0):
        raise ValueError('seed must be a nonnegative integer')
    for key in _PROVENANCE:
        _sha256(metadata[key], key)
    if (isinstance(min_trials_per_style, bool) or not isinstance(min_trials_per_style, int)
            or min_trials_per_style < 1):
        raise ValueError('positive min_trials_per_style required')
    if metadata['split'] == 'validation' and min_trials_per_style < 100:
        raise ValueError('production validation minimum is 100 per style')
    if not isinstance(payload['trials'], list):
        raise ValueError('trials must be an array')
    ids, states, counts, trials = set(), set(), dict.fromkeys(GRASP_STYLES, 0), []
    for trial in payload['trials']:
        _keys(trial, ('trial_id', 'grasp_style', 'physical_state', 'state_sha256'),
              ('environment_origin_m',), name='trial')
        trial_id, style = trial['trial_id'], trial['grasp_style']
        if not isinstance(trial_id, str) or not trial_id.strip() or trial_id in ids:
            raise ValueError('unique nonempty trial_id required')
        if not isinstance(style, str) or style not in GRASP_STYLES:
            raise ValueError('unsupported grasp_style')
        state = _physical_state(trial['physical_state'])
        digest = state_sha256(state)
        if _sha256(trial['state_sha256'], 'state_sha256') != digest:
            raise ValueError('state_sha256 does not match physical_state')
        if digest in states:
            raise ValueError('duplicate physical state, including across grasp styles')
        row = dict(trial_id=trial_id, grasp_style=style, physical_state=state,
                   state_sha256=digest)
        if 'environment_origin_m' in trial:
            row['environment_origin_m'] = _vector(trial['environment_origin_m'], 3,
                                                  'environment_origin_m')
        ids.add(trial_id); states.add(digest); counts[style] += 1; trials.append(row)
    if any(n < min_trials_per_style for n in counts.values()):
        raise ValueError(f'at least {min_trials_per_style} independent trials per style required')
    return dict(schema_version=SCHEMA_VERSION, metadata=deepcopy(metadata), trials=trials)


def manifest_sha256(manifest, *, min_trials_per_style=100):
    """Canonical normalized manifest identity, including labels/provenance.

    Pass the same explicit test-only minimum used for validation/collection.
    """
    return _digest(validate_manifest(manifest, min_trials_per_style=min_trials_per_style))


def _sample(value):
    _keys(value, ('dt_s',), set(_SAMPLE_FIELDS) - {'dt_s'}, name='sample')
    result = dict(dt_s=_number(value['dt_s'], 'dt_s', positive=True),
                  measurement_valid=value.get('measurement_valid', True))
    if not isinstance(result['measurement_valid'], bool):
        raise ValueError('measurement_valid must be boolean')
    for name in ('lift_m', 'qualified_hold_s', 'slip_m'):
        raw = value.get(name)
        result[name] = None if raw is None else _number(raw, name,
            minimum=None if name == 'lift_m' else 0.)
    contact = value.get('bilateral_opposed_contact')
    if contact is not None and not isinstance(contact, bool):
        raise ValueError('bilateral_opposed_contact must be boolean or null')
    result['bilateral_opposed_contact'] = contact
    for name in ('estimated_torque_nm', 'effort_limit_nm', 'action'):
        raw = value.get(name)
        if raw is None:
            result[name] = None
        else:
            if not isinstance(raw, list) or not raw:
                raise ValueError(f'{name} must be a nonempty finite array or null')
            result[name] = [_number(x, name, positive=name == 'effort_limit_nm') for x in raw]
    torque, limits = result['estimated_torque_nm'], result['effort_limit_nm']
    if ((torque is None) != (limits is None)
            or (torque is not None and len(torque) != len(limits))):
        raise ValueError('estimated torque and positive effort limits require matching arrays')
    return result


def _quantiles(values):
    ordered = sorted(v for v in values if v is not None)
    result = {}
    for name, p in (('P50', .5), ('P75', .75), ('P90', .9), ('P95', .95)):
        if not ordered:
            result[name] = None
        else:
            position = (len(ordered) - 1) * p
            lo, hi = math.floor(position), math.ceil(position)
            fraction = position - lo
            result[name] = ordered[lo] * (1 - fraction) + ordered[hi] * fraction
    result['max'] = max(ordered) if ordered else None
    return result


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


class FirstTerminalAggregator:
    """Consume one pre-reset physical substep per call, then freeze first done.

    The caller must call observe BEFORE native auto-reset, supplying that final
    substep and the native event. Post-reset samples cannot reconstruct it.
    ``native_success`` is an actual native event, never inferred from lift/hold.
    ``qualified_hold_s`` is the native qualified accumulator, not raw contact.
    ``slip_m`` is native fixed-anchor drift only when observable (else null).
    ``bilateral_opposed_contact`` is independent raw opposed contact eligibility.
    Torque counts are estimated joint-substep samples at abs(torque)>=limit;
    they are not measurements or action-bound hits. Action stability is RMS of
    consecutive valid sampled action-element deltas; missing samples break the
    chain. Repeated actions at physical substeps contribute real zero deltas.
    """

    def __init__(self, manifest, *, checkpoint_sha256, deterministic,
                 min_trials_per_style=100):
        self._manifest = validate_manifest(manifest, min_trials_per_style=min_trials_per_style)
        self._manifest_sha256 = _digest(self._manifest)
        self._checkpoint = _sha256(checkpoint_sha256, 'checkpoint_sha256')
        if not isinstance(deterministic, bool):
            raise ValueError('deterministic must be boolean')
        self._deterministic = deterministic
        self._envs, self._trial_envs, self._rows = {}, {}, {}
        self._total_duration = self._total_action_squared = 0.
        for trial in self._manifest['trials']:
            self._rows[trial['trial_id']] = dict(trial_id=trial['trial_id'],
                grasp_style=trial['grasp_style'], state_sha256=trial['state_sha256'],
                status='incomplete', native_success=False, sample_count=0,
                observed_duration_s=0., invalid_measurement_samples=0,
                missing_measurement_samples=0, peak_lift_m=None,
                max_qualified_hold_s=None, max_slip_m=None,
                contact_alive_duration_s=0., torque_saturation_samples=0,
                torque_valid_samples=0, action_samples=0, action_delta_elements=0,
                action_delta_samples=0, action_delta_squared_sum=0.,
                action_max_abs_delta=None,
                metric_valid_duration_s=dict.fromkeys(_METRICS, 0.),
                _last_action=None, _action_size=None)

    def observe(self, env_id, trial_id, sample, *, terminal=False, native_success=False):
        """Return true if consumed; false after this env/trial's first terminal.

        Malformed active samples raise ValueError before changing any state.
        Invalid finite measurements use measurement_valid=false, preserving
        their alive duration and invalid count while excluding their metrics.
        NaN/Inf are always rejected, including behind the invalid mask.
        """
        if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id < 0:
            raise ValueError('env_id must be a nonnegative integer')
        bound = self._envs.get(env_id)
        if bound is not None and self._rows[bound]['status'] == 'completed':
            return False
        if not isinstance(trial_id, str) or trial_id not in self._rows:
            raise ValueError('trial_id is absent from the manifest')
        row = self._rows[trial_id]
        if row['status'] == 'completed':
            return False
        if ((bound is not None and bound != trial_id)
                or (trial_id in self._trial_envs and self._trial_envs[trial_id] != env_id)):
            raise ValueError('active environment/trial is already bound')
        if not isinstance(terminal, bool) or not isinstance(native_success, bool):
            raise ValueError('terminal and native_success must be boolean native events')
        value = _sample(sample)
        action = value['action']
        if (action is not None and row['_action_size'] is not None
                and len(action) != row['_action_size']):
            raise ValueError('action dimension changed within a trial')
        # Reject arithmetic overflow before binding, counting or freezing a row.
        duration = _number(self._total_duration + value['dt_s'], 'total duration')
        deltas, squared = None, 0.
        if value['measurement_valid'] and action is not None and row['_last_action'] is not None:
            deltas = [_number(a - b, 'action delta') for a, b in zip(action, row['_last_action'])]
            squared = _number(sum(d * d for d in deltas), 'action squared delta')
        total_squared = _number(self._total_action_squared + squared, 'total action squared delta')
        self._envs[env_id], self._trial_envs[trial_id] = trial_id, env_id
        self._total_duration, self._total_action_squared = duration, total_squared
        dt = value['dt_s']
        row['sample_count'] += 1
        row['observed_duration_s'] += dt
        row['native_success'] |= native_success
        if not value['measurement_valid']:
            row['invalid_measurement_samples'] += 1
            row['_last_action'] = None
        else:
            valid = dict(lift=value['lift_m'] is not None,
                hold=value['qualified_hold_s'] is not None,
                slip=value['slip_m'] is not None,
                contact=value['bilateral_opposed_contact'] is not None,
                torque=value['estimated_torque_nm'] is not None,
                action=action is not None)
            if not all(valid.values()):
                row['missing_measurement_samples'] += 1
            for key, present in valid.items():
                if present:
                    row['metric_valid_duration_s'][key] += dt
            for source, target in (('lift_m', 'peak_lift_m'),
                    ('qualified_hold_s', 'max_qualified_hold_s'), ('slip_m', 'max_slip_m')):
                measured = value[source]
                if measured is not None:
                    row[target] = measured if row[target] is None else max(row[target], measured)
            if value['bilateral_opposed_contact']:
                row['contact_alive_duration_s'] += dt
            if valid['torque']:
                torque, limits = value['estimated_torque_nm'], value['effort_limit_nm']
                row['torque_valid_samples'] += len(torque)
                row['torque_saturation_samples'] += sum(abs(t) >= limit for t, limit in zip(torque, limits))
            if action is not None:
                if deltas is not None:
                    row['action_delta_squared_sum'] += squared
                    row['action_delta_elements'] += len(deltas)
                    row['action_delta_samples'] += 1
                    peak = max(abs(d) for d in deltas)
                    old = row['action_max_abs_delta']
                    row['action_max_abs_delta'] = peak if old is None else max(old, peak)
                row['action_samples'] += 1
                row['_action_size'] = len(action)
            row['_last_action'] = action
        # Terminal metrics above belong to the trial; reset episodes below it do not.
        if terminal:
            row['status'] = 'completed'
        return True

    @staticmethod
    def _public_row(raw):
        row = {k: deepcopy(v) for k, v in raw.items() if not k.startswith('_')}
        duration = row['observed_duration_s']
        for metric, valid_duration in row['metric_valid_duration_s'].items():
            row[f'{metric}_valid_coverage'] = _ratio(valid_duration, duration)
        row['contact_valid_duration_s'] = row['metric_valid_duration_s']['contact']
        row['slip_valid_duration_s'] = row['metric_valid_duration_s']['slip']
        if not row['contact_valid_duration_s']:
            row['contact_alive_duration_s'] = None
        row['height_reached'] = (None if row['lift_valid_coverage'] != 1.
                                 else row['peak_lift_m'] >= .03)
        row['hold_5s_reached'] = (None if row['hold_valid_coverage'] != 1.
                                else row['max_qualified_hold_s'] >= 5.)
        row['torque_saturation_rate'] = _ratio(row['torque_saturation_samples'], row['torque_valid_samples'])
        row['action_stability'] = _action_stability([row])
        row['telemetry_complete'] = (duration > 0 and row['invalid_measurement_samples'] == 0
            and row['missing_measurement_samples'] == 0 and row['action_delta_elements'] > 0)
        return row

    def summary(self):
        """Return JSON-safe owned trial rows, per-style statistics and aggregate.

        All outcome/quantile denominators use completed independent cases only.
        Contact rate is raw contact-alive seconds / valid contact-observed alive
        seconds; coverage exposes the latter / total completed alive seconds.
        Torque rate is saturated estimated joint samples / valid joint samples;
        torque coverage is valid torque packet duration / total alive duration.
        Missing metric rates are null, never fabricated zeros. Metric quantiles
        use linear interpolation of completed trials' available maxima.
        """
        rows = [self._public_row(row) for row in self._rows.values()]
        completed = [r for r in rows if r['status'] == 'completed']
        all_complete = len(completed) == len(rows)
        telemetry_complete = all_complete and all(r['telemetry_complete'] for r in completed)
        metadata = self._manifest['metadata']
        return dict(schema_version='dh116-first-terminal-summary-v1',
            manifest_sha256=self._manifest_sha256, checkpoint_sha256=self._checkpoint,
            split=metadata['split'], deterministic=self._deterministic,
            policy_mode='deterministic' if self._deterministic else 'stochastic',
            reset_kind=metadata['reset_kind'], validation_status=metadata['validation_status'],
            expected_count=len(rows), completed_count=len(completed),
            incomplete_count=len(rows) - len(completed),
            invalid_measurement_count=sum(r['invalid_measurement_samples'] > 0 for r in completed),
            telemetry_complete=telemetry_complete, validation_complete=telemetry_complete,
            native_validation_complete=False,
            quantile_method='linear_position_(n-1)*p',
            sample_unit='pre_reset_physical_substep',
            contact_duration_rate_unit='contact_alive_seconds/valid_contact_alive_seconds',
            torque_saturation_rate_unit='estimated_saturated_joint_samples/valid_joint_samples',
            trial_rows=rows, aggregate=_statistics(rows),
            styles={s: _statistics([r for r in rows if r['grasp_style'] == s]) for s in GRASP_STYLES})


def _action_stability(rows):
    count = sum(r['action_delta_elements'] for r in rows)
    squared = sum(r['action_delta_squared_sum'] for r in rows)
    peaks = [r['action_max_abs_delta'] for r in rows if r['action_max_abs_delta'] is not None]
    return dict(delta_rms=math.sqrt(squared / count) if count else None,
        max_abs_delta=max(peaks) if peaks else None, valid_delta_elements=count,
        valid_delta_samples=sum(r['action_delta_samples'] for r in rows),
        valid_action_samples=sum(r['action_samples'] for r in rows),
        unit='action_element_delta_per_consecutive_valid_physical_substep')


def _statistics(rows):
    done = [r for r in rows if r['status'] == 'completed']
    duration = sum(r['observed_duration_s'] for r in done)
    count = len(done)
    result = dict(expected_count=len(rows), completed_count=count,
        incomplete_count=len(rows) - count,
        observed_duration_s=duration,
        native_success_rate=_ratio(sum(r['native_success'] for r in done), count))
    for field, threshold, name, metric in (('peak_lift_m', .03, 'height_30mm_rate', 'lift'),
                                         ('max_qualified_hold_s', 5., 'hold_5s_rate', 'hold')):
        result[name] = (sum(r[field] >= threshold for r in done) / count
                        if count and all(r[f'{metric}_valid_coverage'] == 1. for r in done)
                        else None)
    for field, name in (('peak_lift_m', 'lift_quantiles_m'),
            ('max_qualified_hold_s', 'hold_quantiles_s'), ('max_slip_m', 'slip_quantiles_m'),
            ('contact_alive_duration_s', 'contact_duration_quantiles_s')):
        values = [r[field] for r in done
                  if field != 'contact_alive_duration_s' or r['contact_valid_duration_s'] > 0]
        result[name] = _quantiles(values)
    result['max_slip_m'] = result['slip_quantiles_m']['max']
    for metric in _METRICS:
        valid_duration = sum(r['metric_valid_duration_s'][metric] for r in done)
        result[f'{metric}_valid_duration_s'] = valid_duration
        result[f'{metric}_valid_coverage'] = _ratio(valid_duration, duration)
    contact_durations = [r['contact_alive_duration_s'] for r in done
                         if r['contact_alive_duration_s'] is not None]
    result['contact_alive_duration_s'] = sum(contact_durations) if contact_durations else None
    result['contact_duration_rate'] = _ratio(result['contact_alive_duration_s'], result['contact_valid_duration_s'])
    result['torque_saturation_samples'] = sum(r['torque_saturation_samples'] for r in done)
    result['torque_valid_samples'] = sum(r['torque_valid_samples'] for r in done)
    result['torque_saturation_rate'] = _ratio(result['torque_saturation_samples'], result['torque_valid_samples'])
    result['action_stability'] = _action_stability(done)
    return result
