"""CPU SDK-shaped fixtures; none are native Isaac validation evidence."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS


def builder(w, models=None):
    from go2_pvcnn.control.dh116.tabletop_tripod_native_evidence import TripodNativeEvidenceBuilder
    return TripodNativeEvidenceBuilder(w, source_models=models)


def fixture(n=2):
    names = ['base_link'] + [f'link{i}' for i in range(1, 17)]
    joint_names = list(reversed(JOINTS)) + ['carriage_x']
    q = torch.arange(n*2*12, dtype=torch.float64).reshape(n, 2, 12)/100
    qd = -q.clone()
    poses = torch.zeros(n, 2, 17, 7, dtype=torch.float64); poses[..., 3] = 1.
    poses[..., 2] = .3
    hands = [NS(joint_names=joint_names[:], body_names=names[:],
        data=NS(joint_pos=q[:, h].clone(), joint_vel=qd[:, h].clone(),
            body_link_pose_w=poses[:, h].clone(), applied_torque=torch.ones(n, 12)*.02,
            computed_torque=torch.ones(n, 12)*.03),
        root_physx_view=NS(get_dof_max_forces=lambda: torch.ones(n, 12)*.15)) for h in range(2)]
    raw = []; fric = []; bindings = []
    for h in range(2):
        for b in range(17):
            # Unowned padding is deliberately NaN and must not invalidate rows.
            raw.append((torch.full((8, 1), float('nan')), torch.full((8, 3), float('nan')),
                torch.full((8, 3), float('nan')), torch.full((8, 1), float('nan')),
                torch.zeros(n, 1, dtype=torch.int32), torch.zeros(n, 1, dtype=torch.int32)))
            fric.append((torch.full((8, 3), float('nan')), torch.full((8, 3), float('nan')),
                torch.zeros(n, 1, dtype=torch.int32), torch.zeros(n, 1, dtype=torch.int32)))
            bindings.append((h, b, b if b < 5 else None, None))
    patch = dict(center_local_m=[0., 0., 0.], tangent_u=[1., 0., 0.],
        tangent_v=[0., 1., 0.], half_widths_m=[.004, .004],
        planes_local=[[0., 0., 1., 0.]], surface_tolerance_m=.001, normal_cosine_min=.8)
    pads = dict(effective_pad_loads_n=torch.zeros(n, 2, 5, dtype=torch.float64),
        effective_pad_vectors_w=torch.zeros(n, 2, 5, 3, dtype=torch.float64),
        effective_pad_points_w=torch.full((n, 2, 5, 3), float('nan'), dtype=torch.float64),
        effective_pad_valid=torch.ones(n, 2, 5, dtype=torch.bool),
        effective_pad_depth_m=torch.zeros(n, 2, 5, dtype=torch.float64),
        rejected_contact_loads_n=torch.zeros(n, 2, dtype=torch.float64))
    ids = [[joint_names.index(k) for k in ACTIVE] for _ in range(2)]
    w = NS(n=n, dt=1/480, _physics_steps=9, hands=hands, q=q, qd=qd, poses=poses,
        roots=poses[:, :, 0].clone(), commands=torch.cat([q[:, h, ids[h]] for h in range(2)], -1),
        wp=torch.tensor([[0., 0., .32]]*n), wq=torch.tensor([[1., 0., 0., 0.]]*n),
        wv=torch.zeros(n, 6), bottom=torch.ones(n)*.312, object_weight_n=torch.ones(n)*2.,
        vertical_components=torch.zeros(n, 2, 2, dtype=torch.float64), pad_measurements=pads,
        parameters=NS(effort_limit_nm=.2),
        pad_sampler=NS(bindings=bindings, rows={(h, d): deepcopy(patch) for h in range(2) for d in range(5)}),
        tripod_contact_snapshot=dict(complete=True, valid_envs=torch.ones(n, dtype=torch.bool),
            tensor_world_physics_steps=9, raw_contacts=raw, raw_friction=fric),
        w=NS(ids=ids, sim=NS(current_time_step_index=26, current_time=26/480)))
    w.wire=NS(data=NS(root_pos_w=w.wp.clone(),root_quat_w=w.wq.clone(),root_vel_w=w.wv.clone()))
    return w


def capture(b, n=2, **kwargs):
    values = dict(episode_ids=list(range(10, 10+n)), stage_id='A', control_step=1,
        control_substep=1, live=[True]*n, terminal=[False]*n, fatal=[False]*n,
        native_success=[False]*n, wrists_fixed=True, left_targets_fixed=True,
        object_external_force_applied=False, induced_disconnect=False)
    values.update(kwargs)
    return b.capture(**values)


def put_contact(w, env=0, sensor=0, slot=3):
    raw = w.tripod_contact_snapshot['raw_contacts'][sensor]
    raw[0][slot] = 1.; raw[1][slot] = torch.tensor([0., 0., .3])
    raw[2][slot] = torch.tensor([0., 0., -1.]); raw[3][slot] = -.0001
    raw[4][env] = 1; raw[5][env] = slot
    f = w.tripod_contact_snapshot['raw_friction'][sensor]
    f[0][6] = torch.tensor([.1, 0., -.2]); f[1][6] = torch.tensor([0., 0., .3])
    f[2][env] = 1; f[3][env] = 6
    h, b, d, _ = w.pad_sampler.bindings[sensor]
    w.vertical_components[env, h] = torch.tensor([1., .2])
    w.pad_measurements['effective_pad_loads_n'][env, h, d] = 1.
    w.pad_measurements['effective_pad_vectors_w'][env, h, d] = torch.tensor([0., 0., -1.])
    w.pad_measurements['effective_pad_points_w'][env, h, d] = torch.tensor([0., 0., .3])
    w.pad_measurements['effective_pad_depth_m'][env, h, d] = .0001


def test_empty_owned_buffers_are_measured_zero_only_with_complete_coverage():
    w = fixture(); out = capture(builder(w)); p = out['packets'][0]
    assert p['raw_contacts'] == [] and p['raw_friction_contacts'] == []
    c = p['contact_evidence_coverage']
    assert c['normal_sensor_count'] == c['friction_sensor_count'] == 34
    assert c['normal_complete'] and c['friction_complete'] and c['ownership_valid']
    assert c['normal_vertical_match'] and c['friction_vertical_match']
    assert c['accepted_normal_matches_effective']
    assert len(p['raw_contact_buffers']) == len(p['raw_friction_buffers']) == 34
    assert p['passive_support_coverage'] is None
    assert p['object_table_support_n'] is None and p['object_table_measurement_valid'] is False
    assert out['execution'] == 'read_only_sdk_snapshot_builder_not_native_attestation'


def test_sparse_normal_friction_owners_are_independent_and_packets_owned():
    w = fixture(); put_contact(w); out = capture(builder(w)); p, other = out['packets']
    assert len(p['raw_contacts']) == len(p['raw_friction_contacts']) == 1
    r = p['raw_contacts'][0]
    assert r['raw_slot'] == 3 and r['accepted'] and r['hand_index'] == 0 and r['body_id'] == 0
    assert r['force_on_hand_w_n'] == [0., 0., -1.]
    assert r['normal_w'] == [0., 0., -1.]
    assert r['depth_m'] == pytest.approx(.0001)
    assert p['raw_friction_contacts'][0]['raw_slot'] == 6
    assert other['raw_contacts'] == []
    assert p['contact_evidence_coverage']['accepted_normal_matches_effective']
    assert p['contact_evidence_coverage']['normal_vertical_match']
    assert p['contact_evidence_coverage']['friction_vertical_match']
    w.tripod_contact_snapshot['raw_contacts'][0][0].fill_(99)
    w.q.fill_(88); w.poses.fill_(77)
    assert r['normal_scalar_n'] == 1.
    assert p['actual_kinematics']['hands'][0]['full_joint_positions_rad'][0] != 88


@pytest.mark.parametrize('corrupt', ['overlap', 'out_of_range', 'fractional', 'nan_owned'])
def test_corrupt_ownership_or_owned_nonfinite_fail_closed(corrupt):
    w = fixture(); put_contact(w)
    raw = w.tripod_contact_snapshot['raw_contacts'][0]
    if corrupt == 'overlap': raw[4][1] = 1; raw[5][1] = 3
    elif corrupt == 'out_of_range': raw[4][0] = 7; raw[5][0] = 6
    elif corrupt == 'fractional':
        values = list(raw); values[4] = values[4].float(); values[4][0] = .5
        w.tripod_contact_snapshot['raw_contacts'][0] = tuple(values)
    else: raw[0][3] = float('nan')
    p = capture(builder(w))['packets'][0]
    assert not p['contact_evidence_coverage']['normal_complete']
    assert not p['measurement_valid']


def test_raw_summary_mismatch_is_unknown_not_scalar_reconstruction_of_measurement():
    w = fixture(); put_contact(w); w.vertical_components[0, 0, 0] = 9.
    w.pad_measurements['effective_pad_vectors_w'][0, 0, 0, 2] = -.5
    p = capture(builder(w))['packets'][0]
    c = p['contact_evidence_coverage']
    assert not c['normal_vertical_match'] and not c['accepted_normal_matches_effective']
    assert not p['measurement_valid']
    assert p['vertical_components_n'][0][0] == 9.
    assert p['effective_pad_vectors_w'][0][0][2] == -.5
    assert p['raw_contacts'][0]['force_on_hand_w_n'][2] == -1.


def test_stale_or_reset_invalidated_snapshot_never_reuses_old_episode_contacts():
    w = fixture(); put_contact(w); b = builder(w)
    w.tripod_contact_snapshot['tensor_world_physics_steps'] = 8
    p = capture(b)['packets'][0]
    assert p['raw_contacts'] is None and not p['measurement_valid']
    w.tripod_contact_snapshot['tensor_world_physics_steps'] = 9
    w.tripod_contact_snapshot['complete'] = False
    w.tripod_contact_snapshot['valid_envs'][0] = False
    p = capture(b)['packets'][0]
    assert p['raw_contacts'] is None and not p['measurement_valid']


def test_actual_named_state_never_expands_mimic_or_uses_commands_as_q():
    w = fixture(); w.commands.fill_(1.)
    p = capture(builder(w))['packets'][0]; state = p['actual_kinematics']
    for h in range(2):
        hand = state['hands'][h]
        assert hand['joint_names'] == list(JOINTS)
        assert hand['full_joint_positions_rad'] == [float(w.q[0, h, w.hands[h].joint_names.index(k)]) for k in JOINTS]
        assert hand['full_joint_velocities_rad_s'] == [-v for v in hand['full_joint_positions_rad']]
        assert hand['command_active_rad'] == [1.]*6
        assert len(hand['body_link_pose_w']) == 17
    assert p['tensor_world_physics_steps'] == 9
    assert p['simulation_step_index'] == 26


def test_implicit_torque_command_and_drive_limits_remain_separate():
    w = fixture(); w.effort_feedforward = torch.full((2, 2, 6), .05)
    p = capture(builder(w))['packets'][0]
    assert p['estimated_torque_nm'] == pytest.approx([.02]*12)
    assert p['computed_torque_nm'] == pytest.approx([.03]*12)
    assert p['drive_effort_limit_nm'] == pytest.approx([.15]*12)
    assert p['effort_limit_nm'] == pytest.approx([.2]*12)
    assert p['explicit_effort_command_nm'] == pytest.approx([.05]*12)
    assert p['torque_source'] == 'implicit_actuator_estimate_not_hardware_measurement'
    del w.hands[0].data.applied_torque
    del w.hands[1].data.computed_torque
    del w.effort_feedforward
    p = capture(builder(w))['packets'][0]
    assert p['estimated_torque_nm'] is None and p['computed_torque_nm'] is None
    assert p['explicit_effort_command_nm'] is None


def test_source_bind_uses_all_actual_frames_and_independent_follower_state():
    w = fixture()
    class Model:
        bodies = {f'/DH/{k}' for k in w.hands[0].body_names}
        source_hash = 'test-source-not-native'; root = '/DH/base_link'
        def bind(self, full_q, poses):
            assert set(poses) == self.bodies
            return dict(full_q=full_q.copy(), poses=deepcopy(poses))
    models = [Model(), Model()]
    out = capture(builder(w, models)); p = out['packets'][0]
    for h in range(2):
        bound = out['geometry_snapshots'][0][h]
        assert bound['full_q'].tolist() == p['actual_kinematics']['hands'][h]['full_joint_positions_rad']
        assert len(bound['poses']) == 17
        assert bound['snapshot_identity'] == p['actual_kinematics']['snapshot_identity']
    assert not p['actual_kinematics']['native_attested']


def test_missing_source_body_or_bad_joint_names_refused_at_init():
    w = fixture(); w.hands[0].joint_names[0] = JOINTS[0]
    with pytest.raises(ValueError, match='joint'): builder(w)
    w = fixture(); model = NS(bodies={'/DH/absent'}, source_hash='x', root='/DH/absent')
    with pytest.raises(ValueError, match='body'): builder(w, [model, model])


def test_simulator_clock_change_during_capture_is_rejected():
    w = fixture()
    def bad_read():
        w.w.sim.current_time_step_index += 1
        return torch.ones(2, 12)*.15
    w.hands[0].root_physx_view.get_dof_max_forces = bad_read
    with pytest.raises(RuntimeError, match='clock'): capture(builder(w))


def test_capture_does_not_read_sensor_views_or_write_or_step_world():
    w = fixture()
    def no_call(*args, **kwargs): raise AssertionError('forbidden physics/sensor mutation')
    w.advance = w.refresh = w.reset = no_call
    w.w.sim.step = no_call
    for h in w.hands: h.update = h.write_data_to_sim = no_call
    out = capture(builder(w))
    assert len(out['packets']) == 2


def test_reset_retains_actual_pose_baseline_and_marks_contact_unknown():
    w = fixture(); put_contact(w)
    p = capture(builder(w), event_kind='reset', control_substep=0)['packets'][0]
    assert p['event_kind'] == 'reset' and p['raw_contacts'] is None
    assert p['effective_pad_loads_n'] is None
    assert p['wrist_pose'] == w.roots[0].tolist()
    assert p['bottom_m'] == float(w.bottom[0])


def test_sdk_readback_and_tensorworld_cache_mismatch_does_not_bind_as_same_clock():
    w = fixture(); w.hands[0].data.joint_pos[0, 0] = .99
    out = capture(builder(w)); p = out['packets'][0]
    assert p['actual_kinematics']['hands'][0]['full_joint_positions_rad'][-1] == .99
    assert not p['measurement_valid']
    assert not p['actual_kinematics']['hands'][0]['tensorworld_cache_matches_sdk']


def test_nonunit_actual_quaternion_without_geometry_model_invalidates_measurement():
    w = fixture(); w.poses[0, 0, 4, 3] = 0.; w.hands[0].data.body_link_pose_w[0, 4, 3] = 0.
    p = capture(builder(w))['packets'][0]
    assert not p['measurement_valid'] and not p['actual_kinematics']['hands'][0]['valid']


def test_native_hand_sensor_binding_must_cover_both_complete_hands_once():
    w = fixture(); w.pad_sampler.bindings[-1] = (0, 17, None, None)
    with pytest.raises(ValueError, match='sensor'): builder(w)


def test_finite_signed_release_is_valid_measured_nonbearing_not_sensor_failure():
    w = fixture(); put_contact(w); w.tripod_contact_snapshot['raw_contacts'][0][0][3] = -.2
    w.pad_measurements['effective_pad_loads_n'].zero_()
    w.pad_measurements['effective_pad_vectors_w'].zero_()
    w.pad_measurements['effective_pad_valid'].zero_()
    w.pad_measurements['rejected_contact_loads_n'][0,0]=.2
    w.vertical_components[0,0,0]=-.2
    p = capture(builder(w))['packets'][0]
    assert p['raw_contacts'][0]['normal_scalar_n'] == pytest.approx(-.2)
    assert p['raw_contacts'][0]['force_on_hand_w_n'][2] == pytest.approx(.2)
    assert not p['raw_contacts'][0]['accepted']
    assert p['raw_contacts'][0]['valid']
    assert p['contact_evidence_coverage']['normal_complete'] and p['measurement_valid']
    assert p['effective_pad_loads_n'][0][0]==0.
    assert p['contact_evidence_coverage']['rejected_load_matches_effective']


@pytest.mark.parametrize('signed_value',[0.,-.00001,-.25])
def test_signed_release_uses_original_absolute_rejected_total_without_positive_bearing_credit(signed_value):
    w=fixture();put_contact(w)
    raw=w.tripod_contact_snapshot['raw_contacts'][0]
    raw[0][4]=signed_value;raw[1][4]=raw[1][3].clone()
    raw[2][4]=raw[2][3].clone();raw[3][4]=raw[3][3].clone();raw[4][0]=2
    w.vertical_components[0,0,0]=1.+signed_value
    w.pad_measurements['rejected_contact_loads_n'][0,0]=abs(signed_value)
    p=capture(builder(w))['packets'][0]
    assert p['measurement_valid'] and p['contact_evidence_coverage']['normal_complete']
    assert len(p['raw_contacts'])==2
    assert p['raw_contacts'][0]['accepted'] and not p['raw_contacts'][1]['accepted']
    assert p['raw_contacts'][1]['valid']
    assert p['raw_contacts'][1]['normal_scalar_n']==pytest.approx(signed_value)
    assert p['effective_pad_loads_n'][0][0]==1.
    assert p['rejected_contact_loads_n'][0]==pytest.approx(abs(signed_value))
    assert p['contact_evidence_coverage']['rejected_load_matches_effective']


def test_missing_negative_release_in_rejected_summary_is_not_certified_complete():
    w=fixture();put_contact(w)
    raw=w.tripod_contact_snapshot['raw_contacts'][0]
    raw[0][4]=-.25;raw[1][4]=raw[1][3].clone()
    raw[2][4]=raw[2][3].clone();raw[3][4]=raw[3][3].clone();raw[4][0]=2
    w.vertical_components[0,0,0]=.75
    # Deliberately wrong: original sampler's abs(f) total requires .25N.
    w.pad_measurements['rejected_contact_loads_n'][0,0]=0.
    p=capture(builder(w))['packets'][0]
    assert not p['measurement_valid']
    assert not p['contact_evidence_coverage']['rejected_load_matches_effective']


def test_only_one_sensor_failure_does_not_erase_independent_other_sensor_evidence():
    w = fixture(); put_contact(w); put_contact(w, sensor=17)
    w.tripod_contact_snapshot['raw_friction'][0][0][6] = float('nan')
    p = capture(builder(w))['packets'][0]
    assert not p['contact_evidence_coverage']['friction_complete']
    assert p['contact_evidence_coverage']['normal_complete']
    assert len(p['raw_contacts']) == 2
    assert p['raw_friction_contacts'][1]['force_on_hand_w_n'][2] == pytest.approx(-.2)


def test_object_invalidity_does_not_allow_same_clock_planning_snapshot():
    w = fixture(); w.wq[0].zero_()
    p = capture(builder(w))['packets'][0]
    assert not p['measurement_valid'] and not p['actual_kinematics']['valid']
    assert p['actual_kinematics']['cached_object_readback']['quaternion_wxyz'] == [0.,0.,0.,0.]
    assert p['object_pose_w'][3:] == [1.,0.,0.,0.]


def test_midcontrol_physical_reset_keeps_actual_control_clock_and_next_substep():
    w = fixture(); b = builder(w)
    p = capture(b, event_kind='reset', control_substep=3, control_step=7,
        episode_ids=[12, 13])['packets'][0]
    assert p['control_substep'] == 3 and p['control_step'] == 7
    assert p['simulation_step_index'] == 26 and p['tensor_world_physics_steps'] == 9
    assert p['episode_id'] == 12 and p['raw_contacts'] is None
    w._physics_steps = 10; w.w.sim.current_time_step_index = 27; w.w.sim.current_time = 27/480
    w.tripod_contact_snapshot['tensor_world_physics_steps'] = 10
    p = capture(b, control_substep=4, control_step=7, episode_ids=[12, 13])['packets'][0]
    assert p['control_substep'] == 4 and p['simulation_step_index'] == 27


@pytest.mark.parametrize('which',['position','quaternion','velocity'])
def test_wire_sdk_and_cached_object_mismatch_invalidates_current_geometry(which):
    w=fixture()
    if which=='position':w.wire.data.root_pos_w[0,0]+=.01
    elif which=='quaternion':w.wire.data.root_quat_w[0]=torch.tensor([0.,1.,0.,0.])
    else:w.wire.data.root_vel_w[0,0]=1.
    out=capture(builder(w));p=out['packets'][0]
    assert not p['measurement_valid'] and not p['actual_kinematics']['valid']
    assert not p['actual_kinematics']['object_tensorworld_cache_matches_sdk']
    assert out['geometry_snapshots'][0]==[None,None]
    assert out['packets'][1]['measurement_valid']


@pytest.mark.parametrize('missing',['wire','data','velocity'])
def test_absent_actual_wire_readback_is_unknown_not_certified_cached_state(missing):
    w=fixture()
    if missing=='wire':del w.wire
    elif missing=='data':del w.wire.data
    else:del w.wire.data.root_vel_w
    out=capture(builder(w));p=out['packets'][0]
    assert not p['measurement_valid'] and not p['actual_kinematics']['valid']
    assert p['actual_kinematics']['object_sdk_readback_available'] is False
    assert out['geometry_snapshots'][0]==[None,None]


@pytest.mark.parametrize('value',[.9,False,True])
def test_body_binding_is_not_silently_converted_to_another_sensor_owner(value):
    w=fixture();h,b,d,s=w.pad_sampler.bindings[0]
    w.pad_sampler.bindings[0]=(h,value,d,s)
    with pytest.raises(ValueError,match='integer'):
        builder(w)


def test_cached_wrist_pose_is_not_substituted_for_actual_sdk_base_link():
    w=fixture();w.roots[0,0,0]+=.000001
    p=capture(builder(w))['packets'][0]
    assert not p['measurement_valid']
    assert p['wrist_pose'][0]==w.hands[0].data.body_link_pose_w[0,0].tolist()
    h=p['actual_kinematics']['hands'][0]
    assert h['tensorworld_cache_matches_sdk'] is False
    assert h['tensorworld_cached_base_link_pose_w'][0]==pytest.approx(.000001)


@pytest.mark.parametrize('kind',['normal','friction'])
def test_full_original_sensor_capacity_is_censored_without_dropping_observed_raw(kind):
    w=fixture()
    if kind=='normal':
        raw=w.tripod_contact_snapshot['raw_contacts'][0]
        raw[0].fill_(1.);raw[1][:]=torch.tensor([0.,0.,.3])
        raw[2][:]=torch.tensor([0.,0.,-1.]);raw[3].fill_(-.0001)
        raw[4][0]=8;raw[5][0]=0
        w.pad_measurements['effective_pad_loads_n'][0,0,0]=8.
        w.pad_measurements['effective_pad_vectors_w'][0,0,0]=torch.tensor([0.,0.,-8.])
        w.vertical_components[0,0,0]=8.
    else:
        raw=w.tripod_contact_snapshot['raw_friction'][0]
        raw[0][:]=torch.tensor([0.,0.,-.2]);raw[1][:]=torch.tensor([0.,0.,.3])
        raw[2][0]=8;raw[3][0]=0
        w.vertical_components[0,0,1]=1.6
    packets=capture(builder(w))['packets']
    assert len(packets[0]['raw_contacts' if kind=='normal' else 'raw_friction_contacts'])==8
    for p in packets:
        c=p['contact_evidence_coverage']
        assert c['capacity_saturated'] and not c[kind+'_complete']
        assert kind+'_capacity_saturated_overflow_unknown' in c['invalid_reasons']
        assert not p['measurement_valid']
