"""Read-only tracking calibration from native targets and post-physics feedback.

Pre-control q, command caches, reconstructed targets and command-shadow outputs
are never tracking evidence. This module proposes thresholds only; it neither
installs a controller nor starts a simulator. Qualification checks recorded
native provenance and alignment, rather than performing a new native trial.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from .linear_coupling import ACTIVE
from .tabletop_joint_target_readback import BACKEND_SOURCE
from .tabletop_vision_supervision_dataset import (
    SEQUENCE_IDENTITY, _sequence_identity, _sha, join_sequence_context, validate_dataset,
)


VERSION = 'dh116-teacher-tracking-calibration-v1'
POST_SOURCE = 'original_advance_returned_refresh_before_TensorEnv_terminal_reset'
HANDS = ('left', 'right')


def _identity(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('explicit four-part sample_key required')
    return _sequence_identity(dict(zip(SEQUENCE_IDENTITY, value)))


def _evidence_table(records, name):
    if not isinstance(records, list):
        raise ValueError(f'{name} is missing; historical pre-feedback cannot replace it')
    result = {}
    for record in records:
        if not isinstance(record, dict): raise ValueError(f'{name} rows must be dictionaries')
        key = _identity(record.get('sample_key'))
        if key in result: raise ValueError(f'duplicate {name} sample_key')
        substeps = record.get('physics_substeps')
        if not isinstance(substeps, list) or len(substeps) != 8:
            raise ValueError(f'{name} requires eight explicitly recorded substeps')
        result[key] = substeps
    return result


def _clock(packet):
    if not isinstance(packet, dict) or packet.get('clock_aligned') is not True:
        raise ValueError('missing or unaligned native SDK clock')
    step, when = packet.get('sdk_physics_step'), packet.get('sdk_timestamp_s')
    if type(step) is not int or step < 0 or type(when) not in (int, float) or not math.isfinite(when):
        raise ValueError('finite actual native SDK physics index and timestamp required')
    return step, float(when)


def _roundoff_equal(actual, expected):
    # This is an arithmetic identity allowance measured in representable floats,
    # not a configurable time-alignment tolerance. SDK getter clocks themselves
    # are required to be exactly equal, including their float timestamps.
    return abs(actual - expected) <= 8 * max(math.ulp(actual), math.ulp(expected))


def _post_feedback(packet):
    clock = _clock(packet)
    if packet.get('native_runtime_verified') is not True or packet.get('source') != POST_SOURCE:
        raise ValueError('actual original post-physics native refresh required; pre-feedback is ineligible')
    if (packet.get('backend_readback_sdk_physics_step'), packet.get('backend_readback_sdk_timestamp_s')) != clock:
        raise ValueError('post-feedback and backend getter SDK clocks differ')
    valid = packet.get('source_valid')
    if not isinstance(valid, torch.Tensor) or valid.shape != (1,) or valid.dtype != torch.bool or not bool(valid[0]):
        raise ValueError('post-physics source_valid must explicitly qualify this environment')
    for name in ('measured_q_rad', 'measured_qd_rad_s'):
        value = packet.get(name)
        if (not isinstance(value, torch.Tensor) or value.shape != (1, 12)
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError(f'actual finite post-physics {name} [1,12] required')
    return packet['measured_q_rad'][0].detach().cpu().double(), clock


def _backend_fingers(hand, env_id):
    if not isinstance(hand, dict): raise ValueError('actual per-hand PhysX backend evidence required')
    names = hand.get('joint_names')
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
            or any(name not in names for name in ACTIVE)):
        raise ValueError('unique native joint_names containing all six ACTIVE names required')
    backend = hand.get('backend')
    if (not isinstance(backend, dict) or backend.get('available') is not True
            or backend.get('source') != BACKEND_SOURCE or backend.get('error') is not None):
        raise ValueError('actual PhysX backend target readback required; no cache or reconstructed fallback')
    values = torch.as_tensor(backend.get('values'))
    finite = torch.as_tensor(backend.get('finite_env'))
    original_shape = backend.get('shape')
    if (values.shape != (1, len(names)) or not values.is_floating_point()
            or not bool(torch.isfinite(values).all()) or finite.ndim != 1 or finite.dtype != torch.bool
            or env_id >= len(finite) or not bool(finite[env_id])
            or not isinstance(original_shape, (list, tuple))
            or tuple(original_shape) != (len(finite), len(names))
            or str(values.dtype).removeprefix('torch.') != backend.get('dtype')):
        raise ValueError('finite cropped backend values [1,DOF] and original bulk-read validity required')
    return values[0, [names.index(name) for name in ACTIVE]].detach().cpu().double()


def _reported_clock_mapping(data, context, joined, backend_table, post_table):
    """Audit one affine SDK index/time coordinate over the entire saved pool.

    SDK callback timestamps need not advance by the controller's float64 dt.
    The two clocks are kept separate: no feedback is synthesized and no dt is
    replaced. Getter pairs still require exact recorded index AND timestamp.
    """
    times, anchors, nominal, intervals = {}, [], [], []
    offset_step = None
    for i, row in enumerate(data['sample_metadata']):
        key = _sequence_identity(row)
        j = int(joined['context_indices'][i])
        physics = context['physics_context']
        valid = (physics['live'][j] & physics['source_valid'][j]).detach().cpu()
        elapsed = 0.
        for k in range(8):
            dt = float(physics['dt_s'][j, k]); elapsed += dt
            if not bool(valid[k]): continue
            clock = _clock(post_table[key][k])
            if _clock(backend_table[key][k]) != clock:
                raise ValueError('backend targets and post-physics feedback SDK clocks differ')
            step, when = clock
            offset = step - row['first_physics_step'] - k - 1
            if offset_step is None: offset_step = offset
            if offset != offset_step:
                raise ValueError('SDK/metadata integer index mapping changed across the pool')
            if step in times and times[step] != when:
                raise ValueError('one global SDK index has different recorded timestamps')
            times[step] = when
            nominal.append(dt); intervals.append((k + 1, elapsed))
        if bool(valid.any()):
            anchors.append((row['first_physics_step'] + offset_step, row['policy_timestamp_s']))
    if len(times) < 2:
        raise ValueError('at least two distinct SDK indices are required to identify a reported cadence')
    first, last = min(times), max(times)
    cadence = (times[last] - times[first]) / (last - first)
    if not math.isfinite(cadence) or cadence <= 0:
        raise ValueError('finite positive global SDK reported cadence required')
    origin = times[first] - first * cadence
    metadata_origin = anchors[0][1] - anchors[0][0] * cadence
    offset_time = origin - metadata_origin
    maximum_roundoff = 0.
    for step, when in times.items():
        expected = origin + step * cadence
        maximum_roundoff = max(maximum_roundoff, abs(when - expected))
        if not _roundoff_equal(when, expected):
            raise ValueError('SDK index/time does not share one constant global affine reported clock')
    for step, when in anchors:
        actual, expected = when + offset_time, origin + step * cadence
        maximum_roundoff = max(maximum_roundoff, abs(actual - expected))
        if not _roundoff_equal(actual, expected):
            raise ValueError('policy metadata timestamps do not share the single global SDK clock mapping')
    return dict(coordinate_mapping_version='sdk_reported_affine_clock_v1',
        reported_sdk_cadence_s=cadence, reported_sdk_time_origin_s=origin,
        cadence_inference='two_extreme_recorded_SDK_indices_then_validate_every_unique_index_and_policy_anchor',
        distinct_sdk_steps=len(times), first_sdk_physics_step=first, last_sdk_physics_step=last,
        sdk_minus_metadata_physics_step=offset_step, sdk_minus_metadata_timestamp_s=offset_time,
        maximum_timestamp_roundoff_s=maximum_roundoff,
        nominal_controller_dt_s_min=min(nominal), nominal_controller_dt_s_max=max(nominal),
        reported_minus_nominal_dt_s_min=min(cadence - dt for dt in nominal),
        reported_minus_nominal_dt_s_max=max(cadence - dt for dt in nominal),
        maximum_policy_interval_nominal_dt_drift_s=max(abs(count * cadence - elapsed)
            for count, elapsed in intervals),
        controller_dt_preserved=True,
        reported_cadence_is_not_a_control_or_physics_parameter_change=True)


def calibrate_teacher_tracking(data, context, *, dataset_sha256, context_sha256):
    """Return a qualified report or an explicit refusal without thresholds.

    Each native table is joined by sample_key independently of its row order.
    Within each row, position k identifies the original kth physical substep;
    its recorded SDK clock must match the post-feedback getter clocks exactly.
    A fixed SDK-minus-metadata integer mapping is audited rather than assuming
    World._physics_steps and the simulator index have the same origin. The
    latter World counter was not saved by this sidecar version. SDK timestamps
    use one globally validated reported cadence, independently of controller dt.
    """
    report = dict(version=VERSION, qualified=False, status='ineligible', reasons=[], reason_counts={},
        dataset_sha256=dataset_sha256, context_sha256=context_sha256,
        samples_per_hand=dict.fromkeys(HANDS, 0), per_joint_sample_count=[0] * 12,
        active_joint_order=[f'{hand}.{name}' for hand in HANDS for name in ACTIVE],
        joint_unit='rad;original_ACTIVE_contract', quantile=.999,
        quantile_method='torch.quantile_float64_linear_interpolation',
        target_source=BACKEND_SOURCE, feedback_source=POST_SOURCE,
        expected_live_physics_substeps=0, controller_installed=False, new_native_trial_performed=False,
        validation_scope='recorded_native_provenance_alignment_and_statistics_not_a_new_native_trial',
        clock_mapping=dict(world_physics_steps_available=False,
            world_physics_steps_scope='not_recorded;no_counter_or_clock_reconstruction',
            getter_clock_comparison='exact_SDK_index_and_float_timestamp',
            metadata_timestamp_comparison='single_global_affine_reported_clock_with_maximum_8_ULP_arithmetic_roundoff'))
    def reject(code, message, *, key=None, substep=None, hand=None):
        report['reason_counts'][code] = report['reason_counts'].get(code, 0) + 1
        if len(report['reasons']) < 64:
            report['reasons'].append(dict(code=code, message=str(message), sample_key=key,
                                          substep=substep, hand=hand))
    try:
        if not _sha(dataset_sha256) or not _sha(context_sha256):
            raise ValueError('actual dataset_sha256 and context_sha256 required')
        validate_dataset(data)
        for episode in data['episodes'].values():
            terminal = episode.get('terminal')
            if (not isinstance(terminal, dict) or terminal.get('success') is not True
                    or terminal.get('recovered') is not False or terminal.get('failure') != 0):
                raise ValueError('explicit complete native successful first terminal required for every calibration episode')
        joined = join_sequence_context(data, context, dataset_sha256=dataset_sha256)
        if not bool(joined['row_valid'].all()):
            raise ValueError('missing/invalid identity, action, pre-context or first-terminal source rows')
        backend_table = _evidence_table(context.get('native_backend_targets'), 'native_backend_targets')
        post_table = _evidence_table(context.get('post_physics_feedback'), 'post_physics_feedback')
        expected_keys = {_sequence_identity(row) for row in data['sample_metadata']}
        if set(backend_table) != expected_keys or set(post_table) != expected_keys:
            raise ValueError('backend and post-feedback sample_key sets must exactly match the qualified successful pool')
    except (ValueError, TypeError, KeyError, RuntimeError) as error:
        reject('source_eligibility', error)
        return report
    errors = {hand: [] for hand in HANDS}
    physics = context['physics_context']
    report['expected_live_physics_substeps'] = sum(int((physics['live'][int(j)]
        & physics['source_valid'][int(j)]).sum()) for j in joined['context_indices'])
    try:
        mapping = _reported_clock_mapping(data, context, joined, backend_table, post_table)
    except (ValueError, TypeError, KeyError, RuntimeError, IndexError) as error:
        reject('reported_clock_mapping', error)
        return report
    report['clock_mapping'].update(mapping,
        mapped_live_substeps=report['expected_live_physics_substeps'])
    for i, row in enumerate(data['sample_metadata']):
        key = _sequence_identity(row)
        j = int(joined['context_indices'][i])
        physics = context['physics_context']
        valid = (physics['live'][j] & physics['source_valid'][j]).detach().cpu()
        for k in range(8):
            if not bool(valid[k]): continue
            backend_packet, post_packet = backend_table[key][k], post_table[key][k]
            try:
                q, clock = _post_feedback(post_packet)
                if _clock(backend_packet) != clock:
                    raise ValueError('backend targets and post-physics q/qd have different SDK clocks')
                if clock[0] != row['first_physics_step'] + mapping['sdk_minus_metadata_physics_step'] + k + 1:
                    raise ValueError('SDK/metadata index mapping changed; reset or substep order mismatch')
                expected_time = mapping['reported_sdk_time_origin_s'] + clock[0] * mapping['reported_sdk_cadence_s']
                if not _roundoff_equal(clock[1], expected_time):
                    raise ValueError('SDK timestamp differs from the single global affine reported clock')
                hands = backend_packet.get('hands')
                if not isinstance(hands, list) or len(hands) != 2:
                    raise ValueError('exactly two real native backend hand records required')
            except (ValueError, TypeError, KeyError, RuntimeError, IndexError) as error:
                reject('post_physics_clock_or_feedback', error, key=key, substep=k)
                continue
            for h, hand in enumerate(HANDS):
                try:
                    target = _backend_fingers(hands[h], row['env_id'])
                    errors[hand].append((target - q[6 * h:6 * (h + 1)]).abs())
                except (ValueError, TypeError, KeyError, RuntimeError, IndexError) as error:
                    reject('backend_target', error, key=key, substep=k, hand=hand)
    for hand in HANDS: report['samples_per_hand'][hand] = len(errors[hand])
    report['per_joint_sample_count'] = [len(errors[hand]) for hand in HANDS for _ in ACTIVE]
    for hand in HANDS:
        if not errors[hand]: reject('no_samples', f'no fully qualified actual backend/post-feedback samples for {hand}', hand=hand)
    if report['reasons']: return report
    p999 = torch.cat([torch.quantile(torch.stack(errors[hand]), .999, dim=0, interpolation='linear') for hand in HANDS])
    trigger = torch.maximum(p999 * 1.5, torch.full_like(p999, .01))
    report.update(qualified=True, status='qualified_recorded_native_teacher_tracking',
        teacher_error_p999_rad=p999.tolist(), trigger_rad=trigger.tolist(),
        trigger_formula='max(0.01rad,1.5*successful_Teacher_abs_actual_backend_target_minus_post_q_P99.9)',
        successful_episode_count=len(data['episodes']),
        controller_source_sha256=context['controller_source_sha256'],
        action_contract_sha256=context['action_contract_sha256'])
    return report


def save_teacher_tracking_calibration(dataset_path, context_path, output_path):
    """Read actual source bytes and save qualified or rejected evidence exclusively."""
    dataset_path, context_path = Path(dataset_path).resolve(strict=True), Path(context_path).resolve(strict=True)
    output = Path(output_path).resolve()
    if output in (dataset_path, context_path): raise ValueError('calibration output must not replace an immutable input')
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    dataset_raw, context_raw = dataset_path.read_bytes(), context_path.read_bytes()
    dataset_sha, context_sha = hashlib.sha256(dataset_raw).hexdigest(), hashlib.sha256(context_raw).hexdigest()
    data = torch.load(io.BytesIO(dataset_raw), map_location='cpu', weights_only=False)
    context = torch.load(io.BytesIO(context_raw), map_location='cpu', weights_only=False)
    report = calibrate_teacher_tracking(data, context, dataset_sha256=dataset_sha, context_sha256=context_sha)
    report.update(dataset_path=str(dataset_path), context_path=str(context_path))
    if (hashlib.sha256(dataset_path.read_bytes()).hexdigest() != dataset_sha
            or hashlib.sha256(context_path.read_bytes()).hexdigest() != context_sha):
        report.update(qualified=False, status='ineligible')
        report.pop('teacher_error_p999_rad', None); report.pop('trigger_rad', None)
        report['reasons'].append(dict(code='source_changed', message='immutable calibration source bytes changed while reading'))
        report['reason_counts']['source_changed'] = 1
    serialized = json.dumps(report, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream: stream.write(serialized + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = save_teacher_tracking_calibration(args.dataset, args.context, args.output)
    print(json.dumps(report, allow_nan=False))
    return 0 if report['qualified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
