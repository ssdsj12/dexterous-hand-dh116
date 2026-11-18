"""Independent CPU oracles; these synthetic cases are not native validation."""
import copy
import importlib
import json
import math
import struct

import pytest


MODULE = 'go2_pvcnn.control.dh116.tabletop_evaluation_trials'
FINGERS = ['finger11', 'finger12', 'finger13', 'finger21', 'finger22',
           'finger31', 'finger32', 'finger41', 'finger42', 'finger51', 'finger52']
STYLES = ('pinch_thumb_index', 'tripod_thumb_index_middle')


def api():
    # A missing implementation fails this assertion, rather than collection.
    assert importlib.util.find_spec(MODULE) is not None, 'fixed-trial CPU contract is missing'
    return importlib.import_module(MODULE)


def physical_state(index=0):
    # Local scene coordinates; a known synthetic q difference makes cases unique.
    return dict(joint_names=[FINGERS.copy(), FINGERS.copy()],
        joint_positions=[[.1 + index * .00001] * 11, [.2] * 11],
        joint_velocities=[[0.] * 11, [0.] * 11],
        roots=[[0., -.1, .4, 1., 0., 0., 0.], [0., .1, .4, 1., 0., 0., 0.]],
        root_velocities=[[0.] * 6, [0.] * 6],
        wire_pose=[0., 0., .308, 1., 0., 0., 0.], wire_velocity=[0.] * 6)


def manifest(n=1, split='test'):
    m = api()
    rows = []
    for si, style in enumerate(STYLES):
        for i in range(n):
            state = physical_state(si * n + i)
            rows.append(dict(trial_id=f'{style}:{i}', grasp_style=style,
                             physical_state=state, state_sha256=m.state_sha256(state)))
    return dict(schema_version='dh116-fixed-trials-v1', metadata=dict(
        reset_kind='cold_authored_reset', seed=17, split=split,
        generator_sha256='1' * 64, source_sha256='2' * 64,
        physics_sha256='3' * 64, camera_sha256='4' * 64,
        template_sha256='5' * 64, validation_status='authored_unverified'), trials=rows)


def collector(n=1):
    m = api()
    return m.FirstTerminalAggregator(manifest(n), checkpoint_sha256='a' * 64,
                                    deterministic=True, min_trials_per_style=1)


def sample(**changes):
    value = dict(dt_s=.25, lift_m=-.01, qualified_hold_s=0., slip_m=.0002,
                 bilateral_opposed_contact=True, estimated_torque_nm=[.1, -.2],
                 effort_limit_nm=[.2, .2], action=[0., 0.])
    value.update(changes)
    return value


def test_contract_exists_and_minimum_is_per_style_in_production():
    m = api()
    with pytest.raises(ValueError, match='100'):
        m.validate_manifest(manifest())
    with pytest.raises(ValueError, match='100'):
        m.validate_manifest(manifest(split='validation'), min_trials_per_style=1)
    result = m.validate_manifest(manifest(100, 'validation'))
    assert len(result['trials']) == 200
    assert result['metadata']['validation_status'] == 'authored_unverified'


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(extra=True),
    lambda x: x['metadata'].pop('physics_sha256'),
    lambda x: x['metadata'].update(physics_sha256='bad'),
    lambda x: x['metadata'].update(validation_status='native_verified'),
    lambda x: x['metadata'].update(reset_kind='snapshot_restore'),
    lambda x: x['trials'][0].update(unknown=4),
    lambda x: x['trials'][0]['physical_state'].pop('joint_velocities'),
    lambda x: x['trials'][0]['physical_state'].update(hidden_physx_state=[]),
    lambda x: x['trials'][0].update(state_sha256='0' * 64),
    lambda x: x['trials'][0]['physical_state']['joint_positions'][0].append(.2),
    lambda x: x['trials'][0]['physical_state']['joint_names'][0].__setitem__(0, 'unknown_joint'),
    lambda x: x['trials'][0]['physical_state']['wire_velocity'].__setitem__(0, float('nan')),
    lambda x: x['trials'][0]['physical_state']['roots'][0].__setitem__(3, .9),
])
def test_strict_schema_rejects_unknown_missing_bad_provenance_and_nonfinite(mutation):
    value = manifest()
    mutation(value)
    with pytest.raises(ValueError):
        api().validate_manifest(value, min_trials_per_style=1)


def test_relabelled_and_relocated_duplicates_are_rejected_across_styles():
    value = manifest()
    value['trials'][1]['physical_state'] = copy.deepcopy(value['trials'][0]['physical_state'])
    value['trials'][1]['state_sha256'] = value['trials'][0]['state_sha256']
    value['trials'][1]['environment_origin_m'] = [100., 20., 0.]
    with pytest.raises(ValueError, match='duplicate physical'):
        api().validate_manifest(value, min_trials_per_style=1)


def test_duplicate_ids_and_within_style_duplicate_states_are_rejected():
    value = manifest(2)
    value['trials'][1]['trial_id'] = value['trials'][0]['trial_id']
    with pytest.raises(ValueError, match='trial_id'):
        api().validate_manifest(value, min_trials_per_style=1)
    value = manifest(2)
    value['trials'][1].update(physical_state=value['trials'][0]['physical_state'],
                              state_sha256=value['trials'][0]['state_sha256'])
    with pytest.raises(ValueError, match='duplicate physical'):
        api().validate_manifest(value, min_trials_per_style=1)


def test_state_hash_is_physical_and_invariant_to_joint_order_quaternion_sign_and_signed_zero():
    m = api()
    state = physical_state()
    state['joint_positions'][0][0] = .7
    reordered = copy.deepcopy(state)
    for hand in range(2):
        for key in ('joint_names', 'joint_positions', 'joint_velocities'):
            reordered[key][hand].reverse()
    reordered['roots'][0][3:] = [-1., -0., -0., -0.]
    reordered['wire_pose'][0] = -0.
    assert m.state_sha256(state) == m.state_sha256(reordered)
    changed = copy.deepcopy(state)
    changed['root_velocities'][0][0] = .0001
    assert m.state_sha256(state) != m.state_sha256(changed)


def test_quaternion_sign_is_chosen_after_float32_underflow_and_hash_is_idempotent():
    m = api()
    raw = physical_state()
    raw['wire_pose'][3:] = [1e-50, -1., 0., 0.]
    written = copy.deepcopy(raw)
    written['wire_pose'][3:] = [0., -1., 0., 0.]
    # Independently identical float32 writer bytes must identify one state.
    import struct
    assert struct.pack('<4f', *raw['wire_pose'][3:]) == struct.pack('<4f', *written['wire_pose'][3:])
    assert m.state_sha256(raw) == m.state_sha256(written)
    value = manifest()
    value['trials'][0].update(physical_state=raw, state_sha256=m.state_sha256(raw))
    normalized = m.validate_manifest(value, min_trials_per_style=1)
    assert m.validate_manifest(normalized, min_trials_per_style=1) == normalized


def test_unit_quaternion_tolerance_uses_the_actual_float32_written_values():
    m = api()
    state = physical_state()
    state['wire_pose'][3:] = [1.000000919989921, .0004000003893293227, 0., 0.]
    import struct
    raw_norm = math.hypot(*state['wire_pose'][3:])
    written = struct.unpack('<4f', struct.pack('<4f', *state['wire_pose'][3:]))
    assert abs(raw_norm - 1.) <= 1e-6 < abs(math.hypot(*written) - 1.)
    with pytest.raises(ValueError, match='unit'):
        m.state_sha256(state)


def test_manifest_hash_validation_is_idempotent_for_tolerated_unit_quaternion_roundoff():
    m = api()
    value = manifest()
    state = value['trials'][0]['physical_state']
    state['roots'][0][3:] = [0.6912178143264008, 0.3679518147175309,
                            0.21103645383148245, 0.5850593220850616]
    value['trials'][0]['state_sha256'] = m.state_sha256(state)
    normalized = m.validate_manifest(value, min_trials_per_style=1)
    assert m.validate_manifest(normalized, min_trials_per_style=1) == normalized
    assert m.manifest_sha256(value, min_trials_per_style=1) == m.manifest_sha256(
        normalized, min_trials_per_style=1)


def test_physical_identity_uses_native_float32_write_precision():
    m = api()
    state = physical_state()
    alias = copy.deepcopy(state)
    alias['joint_positions'][0][0] = .1000000001
    assert struct.pack('<f', .1) == struct.pack('<f', .1000000001)
    assert m.state_sha256(state) == m.state_sha256(alias)
    value = manifest()
    value['trials'][1].update(physical_state=alias, state_sha256=m.state_sha256(alias))
    with pytest.raises(ValueError, match='duplicate physical'):
        m.validate_manifest(value, min_trials_per_style=1)
    state['wire_velocity'][0] = 1e100
    with pytest.raises(ValueError, match='float32'):
        m.state_sha256(state)


def test_full_external_wrist_q_and_qd_are_required_when_named():
    state = physical_state()
    external = ['wrist_x', 'wrist_y', 'wrist_z', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
    for hand in range(2):
        state['joint_names'][hand].extend(external)
        state['joint_positions'][hand].extend([0.] * 6)
        state['joint_velocities'][hand].extend([0.] * 6)
    assert len(api().state_sha256(state)) == 64
    state['joint_velocities'][1].pop()
    with pytest.raises(ValueError):
        api().state_sha256(state)


def test_terminal_before_reset_is_included_and_first_terminal_freezes_environment():
    c = collector()
    trial = f'{STYLES[0]}:0'
    c.observe(0, trial, sample(lift_m=-.02))
    c.observe(0, trial, sample(lift_m=.031, qualified_hold_s=5., slip_m=.0004,
                             action=[.6, -.8]), terminal=True, native_success=True)
    # This represents an automatically reset episode, which must have no effect.
    assert c.observe(0, trial, sample(lift_m=99., slip_m=9.), terminal=True) is False
    report = c.summary()
    row = report['trial_rows'][0]
    assert row['peak_lift_m'] == pytest.approx(.031)
    assert row['max_qualified_hold_s'] == 5.
    assert row['max_slip_m'] == pytest.approx(.0004)
    assert row['native_success'] is True
    assert row['observed_duration_s'] == .5
    assert row['torque_saturation_samples'] == 2
    assert row['torque_valid_samples'] == 4
    assert row['action_stability']['delta_rms'] == pytest.approx(math.sqrt(.5))
    assert report['completed_count'] == 1
    assert report['expected_count'] == 2
    assert report['validation_complete'] is False
    assert report['styles'][STYLES[0]]['native_success_rate'] == 1.


def test_height_and_native_qualified_hold_do_not_fabricate_success_or_raw_contacts():
    c = collector()
    c.observe(0, f'{STYLES[0]}:0', sample(dt_s=5., lift_m=.04,
        qualified_hold_s=5., bilateral_opposed_contact=False), terminal=True)
    s = c.summary()['styles'][STYLES[0]]
    assert s['height_30mm_rate'] == 1.
    assert s['hold_5s_rate'] == 1.
    assert s['native_success_rate'] == 0.
    assert s['contact_duration_rate'] == 0.
    assert s['contact_alive_duration_s'] == 0.


def test_raw_contact_time_is_not_height_or_hold_gated_and_is_dt_weighted():
    c = collector()
    trial = f'{STYLES[0]}:0'
    c.observe(0, trial, sample(dt_s=.1, lift_m=-.02))
    c.observe(0, trial, sample(dt_s=.3, lift_m=-.01,
                              bilateral_opposed_contact=False), terminal=True)
    s = c.summary()['styles'][STYLES[0]]
    assert s['contact_alive_duration_s'] == pytest.approx(.1)
    assert s['contact_valid_duration_s'] == pytest.approx(.4)
    assert s['contact_duration_rate'] == pytest.approx(.25)
    assert s['lift_quantiles_m']['max'] == -.01


def test_missing_and_invalid_slip_are_null_or_partial_with_explicit_coverage():
    c = collector()
    trial = f'{STYLES[0]}:0'
    c.observe(0, trial, sample(dt_s=.1, slip_m=None))
    c.observe(0, trial, sample(dt_s=.3, slip_m=.002), terminal=True)
    row = c.summary()['trial_rows'][0]
    assert row['max_slip_m'] == .002
    assert row['slip_valid_coverage'] == pytest.approx(.75)
    assert row['missing_measurement_samples'] == 1
    c.observe(1, f'{STYLES[1]}:0', sample(slip_m=None), terminal=True)
    report = c.summary()
    assert report['styles'][STYLES[1]]['max_slip_m'] is None
    assert report['styles'][STYLES[1]]['slip_quantiles_m']['P50'] is None
    assert report['styles'][STYLES[1]]['slip_valid_coverage'] == 0.
    assert report['validation_complete'] is False


def test_incomplete_height_and_hold_coverage_cannot_report_a_definite_failure_rate():
    c = collector()
    trial = f'{STYLES[0]}:0'
    c.observe(0, trial, sample(lift_m=0., qualified_hold_s=0.))
    c.observe(0, trial, sample(lift_m=None, qualified_hold_s=None), terminal=True)
    style = c.summary()['styles'][STYLES[0]]
    assert style['height_30mm_rate'] is None
    assert style['hold_5s_rate'] is None
    assert style['lift_valid_coverage'] == .5
    assert style['hold_valid_coverage'] == .5
    assert style['lift_quantiles_m']['max'] == 0.


def test_explicit_invalid_measurement_is_separate_from_unfinished_trials():
    c = collector()
    c.observe(0, f'{STYLES[0]}:0', sample(measurement_valid=False), terminal=True)
    r = c.summary()
    assert r['completed_count'] == 1 and r['incomplete_count'] == 1
    assert r['invalid_measurement_count'] == 1
    assert r['trial_rows'][0]['max_slip_m'] is None
    assert r['trial_rows'][1]['status'] == 'incomplete'


@pytest.mark.parametrize('changes', [dict(dt_s=0.), dict(lift_m=float('nan')),
    dict(slip_m=float('inf')), dict(qualified_hold_s=-.1),
    dict(action=[float('nan')]), dict(estimated_torque_nm=[.2], effort_limit_nm=[0.]),
    dict(estimated_torque_nm=[.2], effort_limit_nm=[.2, .2]), dict(unknown=0)])
def test_bad_samples_rejected_atomically(changes):
    c = collector()
    with pytest.raises(ValueError):
        c.observe(0, f'{STYLES[0]}:0', sample(**changes), terminal=True)
    assert c.summary()['completed_count'] == 0
    assert c.summary()['trial_rows'][0]['observed_duration_s'] == 0.


def test_actions_at_one_are_not_torque_saturation_and_missing_actions_are_null():
    c = collector()
    c.observe(0, f'{STYLES[0]}:0', sample(action=[1., -1.],
        estimated_torque_nm=None, effort_limit_nm=None), terminal=True)
    s = c.summary()['styles'][STYLES[0]]
    assert s['torque_saturation_rate'] is None
    assert s['torque_valid_coverage'] == 0.
    assert s['action_stability']['delta_rms'] is None
    c.observe(1, f'{STYLES[1]}:0', sample(action=None), terminal=True)
    assert c.summary()['styles'][STYLES[1]]['action_stability']['delta_rms'] is None


def test_quantiles_use_linear_interpolation_and_completed_independent_cases_only():
    c = collector(4)
    for i, lift in enumerate([-.01, .01, .03, .09]):
        c.observe(i, f'{STYLES[0]}:{i}', sample(lift_m=lift), terminal=True)
    r = c.summary()
    s = r['styles'][STYLES[0]]
    # Sorted four values, position=(n-1)*p: independent hand calculation.
    assert s['lift_quantiles_m'] == pytest.approx(dict(
        P50=.02, P75=.045, P90=.072, P95=.081, max=.09))
    assert s['height_30mm_rate'] == .5
    assert s['completed_count'] == 4 and s['expected_count'] == 4
    assert r['styles'][STYLES[1]]['lift_quantiles_m']['P50'] is None


def test_identity_counts_and_authored_status_survive_complete_synthetic_fixture():
    c = collector()
    for i, style in enumerate(STYLES):
        c.observe(i, f'{style}:0', sample())
        c.observe(i, f'{style}:0', sample(action=[.1, .1]), terminal=True)
    r = c.summary()
    assert r['manifest_sha256'] == api().manifest_sha256(manifest(), min_trials_per_style=1)
    assert r['checkpoint_sha256'] == 'a' * 64
    assert r['split'] == 'test' and r['deterministic'] is True
    assert r['validation_complete'] is True
    assert r['native_validation_complete'] is False
    assert r['validation_status'] == 'authored_unverified'
    assert r['completed_count'] == r['expected_count'] == 2
    # Owned report and manifest snapshots cannot mutate collection state.
    r['trial_rows'][0]['peak_lift_m'] = 999.
    assert c.summary()['trial_rows'][0]['peak_lift_m'] == -.01


def test_active_environment_and_trial_bindings_cannot_be_reassigned():
    c = collector()
    c.observe(0, f'{STYLES[0]}:0', sample())
    with pytest.raises(ValueError, match='bound'):
        c.observe(0, f'{STYLES[1]}:0', sample())
    with pytest.raises(ValueError, match='bound'):
        c.observe(1, f'{STYLES[0]}:0', sample())


def test_missing_contact_duration_is_null_not_a_measured_zero():
    c = collector()
    c.observe(0, f'{STYLES[0]}:0', sample(bilateral_opposed_contact=None), terminal=True)
    r = c.summary()
    assert r['trial_rows'][0]['contact_alive_duration_s'] is None
    assert r['styles'][STYLES[0]]['contact_alive_duration_s'] is None
    assert r['styles'][STYLES[0]]['contact_duration_rate'] is None
    assert r['trial_rows'][1]['height_reached'] is None


def test_action_delta_overflow_is_rejected_atomically():
    c = collector()
    trial = f'{STYLES[0]}:0'
    c.observe(0, trial, sample(action=[1e308]))
    with pytest.raises(ValueError, match='finite'):
        c.observe(0, trial, sample(action=[-1e308]), terminal=True)
    assert c.summary()['trial_rows'][0]['sample_count'] == 1


def test_summary_is_strict_json_safe_even_when_finite_durations_would_overflow():
    c = collector()
    c.observe(0, f'{STYLES[0]}:0', sample(dt_s=1e308), terminal=True)
    with pytest.raises(ValueError, match='finite'):
        c.observe(1, f'{STYLES[1]}:0', sample(dt_s=1e308), terminal=True)
    assert c.summary()['completed_count'] == 1
    json.dumps(c.summary(), allow_nan=False)
