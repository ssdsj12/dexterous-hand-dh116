"""Numerical recorder cases; these fixtures never attest native execution."""
import copy
import importlib
import importlib.util
import math

import pytest

MODULE = 'go2_pvcnn.control.dh116.tabletop_tripod_support_diagnostics'
DT = 1 / 480
SENSOR_BINDINGS = [(h, b, b if b < 5 else None) for h in range(2) for b in range(17)]
PASSIVE_SURFACES = ['table', 'support_left', 'support_right']


def test_prefix_recovery_cannot_pass_new_fixed_wrist_stage_without_recovery_inside_stage():
    d=recorder();reset(d,'A');d.observe(packet(1,'A'))
    lost=packet(2,'A');lost['effective_pad_loads_n'][1][:3]=[0.,0.,0.];d.observe(lost)
    for step in range(3,13):out=d.observe(packet(step,'A'))
    assert out['stage_gate']['right_recovery_observed']
    d.begin_stage(packet(12,'A'))
    for step in range(13,23):out=d.observe(packet(step,'A'))
    assert not out['stage_gate']['right_recovery_observed']
    assert not out['stage_gate']['pass']
    lost=packet(23,'A');lost['effective_pad_loads_n'][1][:3]=[0.,0.,0.];d.observe(lost)
    for step in range(24,34):out=d.observe(packet(step,'A'))
    assert out['stage_gate']['right_recovery_observed'] and out['stage_gate']['pass']


def test_original_signed_compliant_release_keeps_raw_coverage_but_cannot_be_bearing():
    p=packet();p['raw_contact_buffers'][16]['count']=1
    p['raw_contacts'].append(dict(sensor_index=16,hand_index=0,body_id=16,digit_index=None,
        raw_slot=0,raw_count=1,raw_start=0,valid=True,accepted=False,
        normal_scalar_n=-.25,normal_w=[0.,0.,-1.],force_on_hand_w_n=[0.,0.,.25],
        point_w_m=[0.,0.,.32],depth_m=0.))
    p['vertical_components_n'][0][0]-=.25;p['rejected_contact_loads_n'][0]=.25
    out=recorder().observe(p)
    assert out['support_validity']['raw_contacts']
    assert out['raw']['raw_contacts'][-1]['normal_scalar_n']==-.25
    assert not out['support_bearing_attested']
    p=packet(2);p['raw_contacts'][0]['normal_scalar_n']=-1.
    p['raw_contacts'][0]['force_on_hand_w_n']=[-v for v in p['raw_contacts'][0]['normal_w']]
    assert not recorder().observe(p)['support_validity']['raw_contacts']


def api():
    assert importlib.util.find_spec(MODULE) is not None, 'missing tripod substep reducer'
    return importlib.import_module(MODULE)


def packet(step=1, stage='B', **changes):
    loads = [[1., 1., 1., 0., 0.] for _ in range(2)]
    x = math.sqrt(.96)
    vectors = [[[x, 0., -.2], [-x, 0., -.2], [-x, 0., -.2], [0., 0., 0.], [0., 0., 0.]] for _ in range(2)]
    result = dict(env_id=0, episode_id=0, stage_id=stage, event_kind='physics',
        tensor_world_physics_steps=step, simulation_step_index=step + 17,
        simulation_time_s=(step + 17) * DT, control_step=(step - 1) // 8,
        control_substep=(step - 1) % 8 + 1, dt_s=DT, live=True,
        terminal=False, fatal=False, measurement_valid=True,
        effective_pad_loads_n=loads, effective_pad_vectors_w=vectors,
        effective_pad_points_w=[[[0., 0., .32] for _ in range(5)] for _ in range(2)],
        effective_pad_valid=[[True, True, True, False, False] for _ in range(2)],
        effective_pad_depth_m=[[.0001, .0002, .0001, 0., 0.] for _ in range(2)],
        rejected_contact_loads_n=[0., 0.], raw_contacts=[dict(accepted=True, normal_scalar_n=1.)],
        raw_friction_contacts=[dict(force_on_hand_w_n=[0., 0., -.4])],
        vertical_components_n=[[.6, .4], [.6, .4]], object_weight_n=2.,
        object_table_support_n=0., object_table_measurement_valid=True,
        object_external_force_applied=False, induced_disconnect=False,
        wrist_pose=[[-.1, 0., .32, 1., 0., 0., 0.], [.1, 0., .32, 1., 0., 0., 0.]],
        object_pose_w=[0., 0., .32, 1., 0., 0., 0.], bottom_m=.312,
        estimated_torque_nm=[.02] * 12, computed_torque_nm=[.03] * 12,
        effort_limit_nm=[.2] * 12, drive_effort_limit_nm=[.15] * 12,
        explicit_effort_command_nm=[.05] * 12,
        torque_source='implicit_actuator_estimate_not_hardware_measurement',
        native_success=False, wrists_fixed=True, left_targets_fixed=True)
    result.update(changes)
    if 'raw_contacts' not in changes:
        result['raw_contacts'] = [dict(sensor_index=17*h+d, hand_index=h, body_id=d, digit_index=d,
            raw_slot=0, raw_start=0, raw_count=1, valid=True,
            force_on_hand_w_n=copy.deepcopy(vectors[h][d]), point_w_m=[0., 0., .32],
            normal_w=copy.deepcopy(vectors[h][d]),
            normal_scalar_n=1., depth_m=.0001, accepted=True)
            for h in range(2) for d in range(3)]
    if 'raw_friction_contacts' not in changes:
        result['raw_friction_contacts'] = [dict(sensor_index=17*h, hand_index=h, body_id=0, digit_index=0,
            raw_slot=0, raw_start=0, raw_count=1, valid=True,
            force_on_hand_w_n=[0., 0., -.4], point_w_m=[0., 0., .32]) for h in range(2)]
    for field, prefix in (('raw_contact_buffers', 'normal'), ('raw_friction_buffers', 'friction')):
        result.setdefault(field, [dict(sensor_index=si, hand_index=h, body_id=b,
            count=int(d is not None and (d < 3 if prefix == 'normal' else d == 0)),
            start=0, capacity=1) for si, (h, b, d) in enumerate(SENSOR_BINDINGS)])
    # Synthetic completeness assertions exercise the pure reducer. They are
    # deliberately labelled fixtures, never native sensor attestations.
    result.setdefault('contact_evidence_coverage', dict(
        schema='dh116-tripod-contact-coverage-v1', fixture=True,
        **{k: result[k] for k in ('env_id', 'episode_id', 'tensor_world_physics_steps')},
        ownership_valid=True, normal_complete=True, friction_complete=True,
        normal_sensor_count=34, friction_sensor_count=34))
    result.setdefault('passive_support_coverage', dict(
        schema='dh116-passive-support-coverage-v1', fixture=True,
        **{k: result[k] for k in ('env_id', 'episode_id', 'tensor_world_physics_steps')},
        complete=True, ownership_valid=True,
        expected_surface_ids=['table', 'support_left', 'support_right'],
        surfaces=[dict(surface_id=k, support_n=(result['object_table_support_n'] if k == 'table' else 0.),
            measurement_valid=True) for k in ('table', 'support_left', 'support_right')]))
    return result


def recorder(**kwargs):
    kwargs.setdefault('sensor_bindings', SENSOR_BINDINGS)
    kwargs.setdefault('passive_surface_ids', PASSIVE_SURFACES)
    return api().TripodSupportDiagnostics('fixture', **kwargs)


def reset(d, stage='B'):
    return d.observe(packet(0, stage, event_kind='reset', control_step=0, control_substep=0))


def test_first_missing_measurements_stay_unknown_without_fabricated_zeroes():
    d = recorder()
    out = d.observe({k: v for k, v in packet().items() if k in (
        'env_id', 'episode_id', 'stage_id', 'event_kind', 'tensor_world_physics_steps',
        'simulation_step_index', 'simulation_time_s', 'control_step', 'control_substep',
        'dt_s', 'live', 'terminal', 'fatal')})
    assert out['six_pad_contact'] is None
    assert out['palm_displacement_m'] is None
    assert out['net_lift_m'] is None
    assert out['slip_m'] is None
    assert out['torque']['applied_estimate_nm'] is None
    assert not out['support_bearing_attested']
    assert not out['stage_gate']['pass']


def test_raw_evidence_is_detached_and_nonfinite_values_have_explicit_original_tags():
    d = recorder()
    p = packet()
    p['estimated_torque_nm'][3] = math.nan
    out = d.observe(p)
    p['raw_contacts'][0]['normal_scalar_n'] = 9.
    assert out['raw']['raw_contacts'][0]['normal_scalar_n'] == 1.
    assert out['raw']['estimated_torque_nm'][3] is None
    assert out['nonfinite_originals']['estimated_torque_nm.3'] == 'nan'
    assert out['torque']['applied_valid'][3] is False
    assert out['torque']['applied_valid'][2] is True


def test_six_pad_continuity_uses_each_substep_and_unknown_censors_it():
    d = recorder(); reset(d)
    for step in range(1, 8):
        out = d.observe(packet(step))
    assert out['six_pad_contiguous_s'] == pytest.approx(7 * DT)
    lost = packet(8); lost['effective_pad_loads_n'][1][2] = 0.
    out = d.observe(lost)
    assert out['six_pad_contact'] is False and out['six_pad_contiguous_s'] == 0.
    assert not out['stage_gate']['pass']
    out = d.observe(packet(9))
    assert out['six_pad_contiguous_s'] == pytest.approx(DT)
    unknown = packet(10); unknown['effective_pad_loads_n'][0][2] = math.nan
    out = d.observe(unknown)
    assert out['six_pad_contact'] is None and out['six_pad_contiguous_s'] == 0.


def test_opposition_and_normal_support_use_actual_vector_sum_not_scalar_times_unit():
    d = recorder(); out = d.observe(packet())
    assert out['per_hand_opposed'] == [True, True]
    assert out['normal_support_n'] == pytest.approx([.6, .6])
    assert out['friction_support_n'] == pytest.approx([.4, .4])
    assert out['verified_support_n'] == pytest.approx([1., 1.])
    same = packet(2)
    same['effective_pad_vectors_w'][1] = [[1., 0., -.2]] * 5
    assert d.observe(same)['per_hand_opposed'] == [True, False]


def test_rejected_contacts_prevent_positive_friction_attribution_and_table_unknown_prevents_attestation():
    d = recorder(); p = packet(); p['rejected_contact_loads_n'] = [.1, 0.]
    out = d.observe(p)
    assert out['verified_support_n'] == pytest.approx([.6, 1.])
    assert not out['support_bearing_attested']
    p = packet(2); p.pop('object_table_support_n'); p.pop('object_table_measurement_valid')
    out = d.observe(p)
    assert out['support_bearing_attested'] is False
    assert out['support_validity']['object_table'] is False


def test_actual_reset_baselines_and_common_rigid_transport_leave_zero_relative_slip():
    d = recorder(); reset(d)
    p = packet()
    for pose in p['wrist_pose']: pose[2] += .03
    p['object_pose_w'][2] += .03; p['bottom_m'] += .03
    out = d.observe(p)
    assert [v[2] for v in out['palm_displacement_m']] == pytest.approx([.03, .03])
    assert out['net_lift_m'] == pytest.approx(.03)
    assert out['slip_m'] == pytest.approx(0.)
    p = packet(2); p['object_pose_w'][0] += .0006
    out = d.observe(p)
    assert out['slip_m'] == pytest.approx(.0006)
    assert not out['stage_gate']['pass']


def test_stage_c_needs_full_five_seconds_and_original_anchor_cannot_follow_drift():
    d = recorder(); reset(d, 'C')
    for step in range(1, 2400): out = d.observe(packet(step, 'C'))
    assert not out['stage_gate']['pass']
    out = d.observe(packet(2400, 'C'))
    assert out['stage_gate']['pass']
    p = packet(2401, 'C'); p['object_pose_w'][0] += .00051
    out = d.observe(p)
    assert not out['stage_gate']['pass'] and out['qualified_stage_hold_s'] == 0.
    out = d.observe(packet(2402, 'C'))
    assert out['slip_m'] == pytest.approx(0.)
    assert out['qualified_stage_hold_s'] == pytest.approx(DT)


def test_stage_c_table_supported_contact_hold_does_not_attest_support_bearing():
    d = recorder(); reset(d, 'C')
    for step in range(1, 2401):
        out = d.observe(packet(step, 'C', object_table_support_n=2.))
    assert out['stage_gate']['contact_hold_pass']
    assert not out['support_bearing_attested']
    assert not out['stage_gate']['pass']


def test_stage_d_requires_original_success_net_lift_and_strict_six_pad_hold():
    d = recorder(); reset(d, 'D')
    for step in range(1, 2401):
        out = d.observe(packet(step, 'D', bottom_m=.342))
    assert not out['stage_gate']['pass']
    out = d.observe(packet(2401, 'D', bottom_m=.342, native_success=True))
    assert out['stage_gate']['pass']
    p = packet(2402, 'D', bottom_m=.342, native_success=True)
    p['effective_pad_loads_n'][1][2] = 0.
    assert not d.observe(p)['stage_gate']['pass']


def test_stage_a_distinguishes_initial_acquisition_from_observed_natural_recovery():
    d = recorder(); reset(d, 'A')
    out = d.observe(packet(1, 'A'))
    assert out['stage_gate']['right_contacts_ready']
    assert not out['stage_gate']['right_recovery_observed']
    assert not out['stage_gate']['pass']
    p = packet(2, 'A'); p['effective_pad_loads_n'][1][:3] = [0., 0., 0.]
    d.observe(p)
    for step in range(3, 12):
        out = d.observe(packet(step, 'A'))
        assert not out['stage_gate']['pass']
    out = d.observe(packet(12, 'A'))
    assert out['stage_gate']['right_recovery_observed']
    assert out['stage_gate']['pass']


def test_terminal_fatal_and_selected_reset_emit_once_and_clear_only_their_episode():
    rows = []; d = recorder(sink=rows.append); reset(d)
    p = packet(1, terminal=True, fatal=True)
    out = d.observe(p)
    assert out['terminal'] and out['fatal']
    assert d.observe(copy.deepcopy(p)) is None
    assert d.observe(packet(2)) is None
    assert len(rows) == 2
    reset_packet = packet(2, episode_id=1, event_kind='reset')
    assert d.observe(reset_packet)['event_kind'] == 'reset'
    assert d.observe(copy.deepcopy(reset_packet)) is None
    out = d.observe(packet(3, episode_id=1))
    assert out['six_pad_contiguous_s'] == pytest.approx(DT)
    assert out['episode_id'] == 1


def test_clock_continuity_and_480hz_are_enforced_separately_from_warmup_offset():
    d = recorder(); d.observe(packet(1)); out = d.observe(packet(2))
    assert out['simulation_step_index'] == 19
    assert out['tensor_world_physics_steps'] == 2
    with pytest.raises(ValueError, match='480'):
        d.observe(packet(3, dt_s=1 / 60))
    with pytest.raises(ValueError, match='clock'):
        d.observe(packet(4))


def test_sim_estimate_drive_budget_and_command_have_independent_saturation_coverage():
    d = recorder(); p = packet()
    p['estimated_torque_nm'][0] = .199
    p['computed_torque_nm'][1] = .19
    p['explicit_effort_command_nm'][2] = .2
    out = d.observe(p); torque = out['torque']
    assert torque['applied_estimate_saturated'][0] is True
    assert torque['computed_drive_saturated'][1] is True
    assert torque['explicit_command_saturated'][2] is True
    assert torque['hardware_torque_nm'] is None
    assert torque['hardware_calibrated'] is False
    assert torque['applied_valid_count'] == 12
    summary = d.summary()
    assert summary['live_substeps'] == 1
    assert summary['torque_joint_samples']['applied_estimate']['valid'] == 12
    assert summary['torque_joint_samples']['applied_estimate']['saturated'] == 1


def test_inactive_and_cross_environment_data_do_not_share_baselines_or_hold():
    d = recorder(); reset(d)
    out = d.observe(packet(1, env_id=1))
    assert out['palm_displacement_m'] is None
    assert d.observe(packet(1, live=False)) is None
    assert d.summary()['live_substeps'] == 1


def test_native_computed_validity_is_independent_from_applied_estimate_validity():
    d = recorder(); p = packet()
    p['torque_valid'] = [False] + [True] * 11
    p['computed_torque_valid'] = [True, False] + [True] * 10
    torque = d.observe(p)['torque']
    assert torque['applied_valid'][:2] == [False, True]
    assert torque['computed_valid'][:2] == [True, False]


def test_supplied_native_anchor_reanchoring_cannot_hide_original_relative_slip():
    d = recorder(); baseline = reset(d, 'C')
    anchor = baseline['raw'].get('fixed_anchor_markers_m')
    if anchor is None:
        anchor = [[[.1, -.2, 0.], [.1, .2, 0.], [.108, 0., 0.]],
                  [[-.1, -.2, 0.], [-.1, .2, 0.], [-.092, 0., 0.]]]
    d.observe(packet(1, 'C', fixed_anchor_markers_m=anchor))
    shifted = copy.deepcopy(anchor)
    for hand in shifted:
        for point in hand: point[0] += .0007
    p = packet(2, 'C', fixed_anchor_markers_m=shifted)
    p['object_pose_w'][0] += .0007
    out = d.observe(p)
    assert out['slip_m'] == pytest.approx(.0007)
    assert out['supplied_anchor_changed'] is True
    assert not out['stage_gate']['pass']


def test_new_fixed_stage_uses_actual_stage_start_wrist_pose_without_rewriting_reset_lift_baseline():
    d = recorder(); reset(d)
    d.observe(packet(1))
    p = packet(2, 'C', bottom_m=.332)
    for pose in p['wrist_pose']: pose[2] += .02
    p['object_pose_w'][2] += .02
    # A new stage is an explicit reference from the last actual measurement.
    d.observe(packet(2, bottom_m=.332, wrist_pose=p['wrist_pose'], object_pose_w=p['object_pose_w']))
    p['stage_id']='C'
    d.begin_stage(p)
    p=packet(3,'C',bottom_m=.332,wrist_pose=p['wrist_pose'],object_pose_w=p['object_pose_w'])
    out = d.observe(p)
    assert out['actual_palms_fixed'] is True
    assert out['net_lift_m'] == pytest.approx(.02)
    assert [v[2] for v in out['palm_displacement_m']] == pytest.approx([.02, .02])


def test_missing_raw_friction_evidence_disables_bearing_attestation_but_keeps_measured_components():
    d = recorder(); p = packet(); p.pop('raw_friction_contacts')
    out = d.observe(p)
    assert out['friction_support_n'] == pytest.approx([.4, .4])
    assert out['support_validity']['raw_friction'] is False
    assert out['support_bearing_attested'] is False


def test_relative_markers_match_original_native_tracker_and_detect_axial_twist():
    import torch
    from go2_pvcnn.control.dh116.tabletop_multifinger_monitor import rod_points_in_hands
    p = packet(); d = recorder(); reset(d)
    out = d.observe(p)
    expected = rod_points_in_hands(torch.tensor([p['object_pose_w']], dtype=torch.float64),
                                  torch.tensor([p['wrist_pose']], dtype=torch.float64))
    assert torch.allclose(torch.tensor(out['object_relative_markers_m']), expected[0].float())
    p = packet(2)
    p['object_pose_w'][3:] = [math.cos(.1 / 2), 0., math.sin(.1 / 2), 0.]
    assert d.observe(p)['slip_m'] > .0005


def test_extra_ring_contact_cannot_supply_unattributed_tripod_bearing_credit():
    d = recorder(); p = packet()
    p['effective_pad_loads_n'][1][3] = 5.
    p['effective_pad_valid'][1][3] = True
    p['effective_pad_vectors_w'][1][3] = [0., 0., -5.]
    p['vertical_components_n'][1][0] = 5.6
    out = d.observe(p)
    assert out['six_pad_contact'] is True
    assert out['non_required_bearing'] == [False, True]
    assert out['normal_support_n'] == pytest.approx([.6, .6])
    assert not out['support_bearing_attested']


@pytest.mark.parametrize('case', ['empty', 'stale', 'missing_support', 'duplicate_support',
                                 'wrong_force', 'unaccepted', 'wrong_normal'])
def test_support_requires_complete_same_clock_consistent_owned_evidence(case):
    d = recorder(); reset(d, 'C'); p = packet(1, 'C')
    if case == 'empty': p['raw_contacts'] = []
    if case == 'stale': p['contact_evidence_coverage']['tensor_world_physics_steps'] = 0
    if case == 'missing_support': p['passive_support_coverage']['surfaces'].pop()
    if case == 'duplicate_support':
        p['passive_support_coverage']['surfaces'][1]['surface_id'] = 'table'
    if case == 'wrong_force': p['raw_contacts'][0]['force_on_hand_w_n'][2] = -.1
    if case == 'unaccepted': p['raw_contacts'][0]['accepted'] = False
    if case == 'wrong_normal': p['raw_contacts'][0]['normal_w'] = [1., 0., 0.]
    assert not d.observe(p)['support_bearing_attested']


def test_first_late_anchor_cannot_hide_drift_even_after_2399_good_samples():
    d = recorder(); reset(d, 'C')
    for step in range(1, 2400): d.observe(packet(step, 'C'))
    p = packet(2400, 'C'); p['object_pose_w'][0] += .0007
    p['fixed_anchor_markers_m'] = api()._markers(p)
    out = d.observe(p)
    assert out['supplied_anchor_changed'] and out['slip_m'] == pytest.approx(.0007)
    assert out['qualified_stage_hold_s'] == 0 and not out['stage_gate']['pass']


def test_physical_reset_requires_strictly_new_episode_and_cannot_reopen_closed_identity():
    d = recorder(); reset(d); d.observe(packet(1, terminal=True))
    with pytest.raises(ValueError, match='episode'):
        d.observe(packet(2, event_kind='reset'))
    with pytest.raises(ValueError, match='reset'):
        d.observe(packet(2, episode_id=1))


def test_induced_episode_history_cannot_be_laundered_into_natural_recovery():
    d = recorder(); reset(d, 'A'); d.observe(packet(1, 'A'))
    p = packet(2, 'A', induced_disconnect=True)
    p['effective_pad_loads_n'][1][:3] = [0., 0., 0.]
    d.observe(p)
    for step in range(3, 15): out = d.observe(packet(step, 'A'))
    assert not out['stage_gate']['right_recovery_observed']
    assert not out['stage_gate']['pass']


def test_one_explicit_stage_reference_preserves_episode_contact_history_and_lift_baseline():
    d=recorder();reset(d,'A');d.observe(packet(1,'A',wrists_fixed=False))
    p=packet(2,'A',wrists_fixed=False)
    p['effective_pad_loads_n'][1][:3]=[0.,0.,0.]
    for pose in p['wrist_pose']:pose[2]+=.02
    p['bottom_m']+=.02;p['object_pose_w'][2]+=.02
    d.observe(p)
    current=copy.deepcopy(p)
    row=d.begin_stage(current)
    assert row['event_kind']=='stage_start' and not row['physical_episode_reset']
    with pytest.raises(ValueError,match='once'):
        d.begin_stage(current)
    for step in range(3,13):
        q=packet(step,'A')
        for pose in q['wrist_pose']:pose[2]+=.02
        q['bottom_m']+=.02;q['object_pose_w'][2]+=.02
        out=d.observe(q)
    assert out['stage_gate']['pass'] and out['stage_gate']['right_recovery_observed']
    assert out['net_lift_m']==pytest.approx(.02)


@pytest.mark.parametrize('case', ['depth', 'rejected', 'zero_normal'])
def test_a_geometry_or_normal_failure_with_contact_load_never_counts_as_actual_loss(case):
    d=recorder();reset(d,'A');d.observe(packet(1,'A'))
    p=packet(2,'A')
    if case=='depth':p['effective_pad_depth_m'][1][:3]=[.0007]*3
    if case=='rejected':p['rejected_contact_loads_n'][1]=.06
    if case=='zero_normal':p['effective_pad_vectors_w'][1][:3]=[[0.,0.,0.]]*3
    d.observe(p)
    for step in range(3,14):out=d.observe(packet(step,'A'))
    assert not out['stage_gate']['right_recovery_observed']
    assert not out['stage_gate']['pass']


@pytest.mark.parametrize('case', ['unknown_body','wrong_sensor','wrong_digit','wrong_slot',
    'duplicate_row','missing_owned_row','duplicate_buffer','wrong_range','per_pad_swap',
    'wrong_scalar','omitted_manifest_surface'])
def test_strict_support_audits_source_bindings_owned_ranges_and_each_pad(case):
    d=recorder();reset(d,'C');p=packet(1,'C')
    if case=='unknown_body':p['raw_contacts'][0]['body_id']=999
    if case=='wrong_sensor':p['raw_contacts'][0]['sensor_index']=33
    if case=='wrong_digit':p['raw_contacts'][0]['digit_index']=4
    if case=='wrong_slot':p['raw_contacts'][0]['raw_slot']=1
    if case=='duplicate_row':p['raw_contacts'].append(copy.deepcopy(p['raw_contacts'][0]))
    if case=='missing_owned_row':p['raw_contacts'].pop()
    if case=='duplicate_buffer':p['raw_contact_buffers'][-1]=copy.deepcopy(p['raw_contact_buffers'][0])
    if case=='wrong_range':p['raw_contact_buffers'][0]['count']=0
    if case=='per_pad_swap':
        p['raw_contacts'][0]['force_on_hand_w_n'],p['raw_contacts'][1]['force_on_hand_w_n']=p['raw_contacts'][1]['force_on_hand_w_n'],p['raw_contacts'][0]['force_on_hand_w_n']
        p['raw_contacts'][0]['normal_w'],p['raw_contacts'][1]['normal_w']=p['raw_contacts'][1]['normal_w'],p['raw_contacts'][0]['normal_w']
    if case=='wrong_scalar':
        p['effective_pad_loads_n'][0][0]=2.
    if case=='omitted_manifest_surface':
        p['passive_support_coverage']['expected_surface_ids']=['table']
        p['passive_support_coverage']['surfaces']=p['passive_support_coverage']['surfaces'][:1]
    assert not d.observe(p)['support_bearing_attested']


def test_no_external_source_manifests_means_unknown_strict_support():
    d=api().TripodSupportDiagnostics('unbound_fixture')
    out=d.observe(packet())
    assert not out['support_bearing_attested']
    assert not out['support_validity']['raw_contacts']
    assert not out['support_validity']['all_original_passive_surfaces']


@pytest.mark.parametrize('change', ['wrist','object','bottom','clock','invalid'])
def test_begin_stage_cannot_replace_last_actual_state_or_clock(change):
    d=recorder();reset(d,'A');p=packet(1,'A');d.observe(p)
    q=copy.deepcopy(p);q['stage_id']='B'
    if change=='wrist':q['wrist_pose'][0][0]+=.001
    if change=='object':q['object_pose_w'][0]+=.001
    if change=='bottom':q['bottom_m']+=.001
    if change=='clock':q['tensor_world_physics_steps']+=1
    if change=='invalid':q['measurement_valid']=False
    with pytest.raises(ValueError):d.begin_stage(q)


def test_implicit_stage_change_is_rejected_and_stage_can_begin_only_once():
    d=recorder();reset(d,'A');p=packet(1,'A');d.observe(p)
    with pytest.raises(ValueError,match='begin_stage'):d.observe(packet(2,'B'))
    q=copy.deepcopy(p);q['stage_id']='B';d.begin_stage(q)
    out=d.observe(packet(2,'B'));assert out['stage_id']=='B'
    q=packet(2,'A');d.begin_stage(q);d.observe(packet(3,'A'))
    with pytest.raises(ValueError,match='once'):d.begin_stage(packet(3,'B'))


@pytest.mark.parametrize('case', ['short','duplicate','bool_hand','bool_body','bool_digit','missing_pad'])
def test_constructor_rejects_invalid_external_sensor_manifest(case):
    bindings=copy.deepcopy(SENSOR_BINDINGS)
    if case=='short':bindings.pop()
    if case=='duplicate':bindings[-1]=bindings[-2]
    if case=='bool_hand':bindings[0]=(False,0,0)
    if case=='bool_body':bindings[0]=(0,False,0)
    if case=='bool_digit':bindings[0]=(0,0,False)
    if case=='missing_pad':bindings[4]=(0,4,None)
    with pytest.raises(ValueError,match='bindings'):recorder(sensor_bindings=bindings)


@pytest.mark.parametrize('surfaces', [[],['table','table'],[True],['']])
def test_constructor_rejects_invalid_source_passive_surface_manifest(surfaces):
    with pytest.raises(ValueError,match='surface'):recorder(passive_surface_ids=surfaces)


def test_source_manifests_and_stage_reference_are_owned_and_not_rewritten_by_caller():
    bindings=copy.deepcopy(SENSOR_BINDINGS);surfaces=PASSIVE_SURFACES[:]
    d=recorder(sensor_bindings=bindings,passive_surface_ids=surfaces)
    bindings[0]=(0,999,0);surfaces[:]=['fake']
    reset(d,'C');p=packet(1,'C');d.observe(p);row=d.begin_stage(p)
    row['fixed_anchor_markers_m'][0][0][0]+=1.
    out=d.observe(packet(2,'C'))
    assert out['support_bearing_attested'] and out['slip_m']==0.


def test_begin_stage_cannot_use_an_invalid_or_closed_last_observation():
    d=recorder();reset(d,'A')
    p=packet(1,'A',measurement_valid=False);d.observe(p)
    with pytest.raises(ValueError,match='actual'):d.begin_stage(p)
    p=packet(2,'A',terminal=True);d.observe(p)
    with pytest.raises(ValueError,match='live'):d.begin_stage(p)


@pytest.mark.parametrize('stage', ['A','B','C'])
def test_fixed_stage_violation_is_sticky_after_palms_return_to_the_reference(stage):
    d=recorder();reset(d,stage);p=packet(1,stage);d.observe(p);d.begin_stage(p)
    p=packet(2,stage)
    if stage=='A':p['effective_pad_loads_n'][1][:3]=[0.,0.,0.]
    d.observe(p)
    p=packet(3,stage)
    for pose in p['wrist_pose']:pose[2]+=.02
    p['object_pose_w'][2]+=.02;p['bottom_m']+=.02
    assert not d.observe(p)['actual_palms_fixed']
    end=2404 if stage=='C' else 15
    for step in range(4,end):out=d.observe(packet(step,stage))
    assert out['actual_palms_fixed']
    assert not out['stage_gate']['pass'], 'Returning cannot erase20mm motion during a fixed experiment'
