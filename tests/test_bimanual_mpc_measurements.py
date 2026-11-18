"""CPU SDK-shaped fixtures only; no native qualification or controller writes."""
from copy import deepcopy
import hashlib
import inspect
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
from go2_pvcnn.control.dh116.tabletop_tensor_world import TensorWorld


class SDKWorld(NS):
    # Source identity only; fixtures never execute native refresh or physics.
    refresh = TensorWorld.refresh


def fixture():
    n = 2
    names = ['base_link'] + [f'link{i}' for i in range(1, 17)]
    joints = list(reversed(JOINTS)) + ['carriage_x']
    q = torch.arange(n*2*12, dtype=torch.float64).reshape(n, 2, 12)/100
    poses = torch.zeros(n, 2, 17, 7, dtype=torch.float64)
    poses[..., 3] = 1.; poses[..., 2] = .3
    com = poses.clone(); com[..., 0] += .01
    velocity = torch.zeros(n, 2, 17, 6, dtype=torch.float64)
    velocity[..., 5] = 2.
    hands = [NS(joint_names=joints[:], body_names=names[:],
        data=NS(joint_pos=q[:, h].clone(), joint_vel=-q[:, h].clone(),
            body_link_pose_w=poses[:, h].clone(), body_com_pose_w=com[:, h].clone(),
            body_com_vel_w=velocity[:, h].clone(), _sim_timestamp=.05,
            _body_com_pose_w=NS(timestamp=.05), _body_com_vel_w=NS(timestamp=.05)),
        root_physx_view=NS(get_dof_max_forces=lambda: torch.ones(n, 12)*.15)) for h in range(2)]
    raw = []; friction = []; bindings = []
    for h in range(2):
        for b in range(17):
            raw.append((torch.full((8, 1), float('nan')), torch.full((8, 3), float('nan')),
                torch.full((8, 3), float('nan')), torch.full((8, 1), float('nan')),
                torch.zeros(n, 1, dtype=torch.int32), torch.zeros(n, 1, dtype=torch.int32)))
            friction.append((torch.full((8, 3), float('nan')), torch.full((8, 3), float('nan')),
                torch.zeros(n, 1, dtype=torch.int32), torch.zeros(n, 1, dtype=torch.int32)))
            bindings.append((h, b, b if b < 5 else None, None))
    patch = dict(center_local_m=[0., 0., 0.], tangent_u=[1., 0., 0.], tangent_v=[0., 1., 0.],
        half_widths_m=[.004, .004], planes_local=[[0., 0., 1., 0.]],
        surface_tolerance_m=.001, normal_cosine_min=.8)
    ids = [[joints.index(name) for name in ACTIVE] for _ in range(2)]
    pads = dict(effective_pad_loads_n=torch.zeros(n, 2, 5, dtype=torch.float64),
        effective_pad_vectors_w=torch.zeros(n, 2, 5, 3, dtype=torch.float64),
        effective_pad_points_w=torch.zeros(n, 2, 5, 3, dtype=torch.float64),
        effective_pad_valid=torch.zeros(n, 2, 5, dtype=torch.bool),
        effective_pad_depth_m=torch.zeros(n, 2, 5, dtype=torch.float64),
        rejected_contact_loads_n=torch.zeros(n, 2, dtype=torch.float64))
    w = SDKWorld(n=n, dt=1/480, _physics_steps=9, hands=hands, q=q, qd=-q.clone(), poses=poses,
        roots=poses[:, :, 0].clone(), commands=torch.cat([q[:, h, ids[h]] for h in range(2)], -1),
        wp=torch.tensor([[0., 0., .32]]*n, dtype=torch.float64),
        wq=torch.tensor([[1., 0., 0., 0.]]*n, dtype=torch.float64), wv=torch.zeros(n, 6),
        bottom=torch.ones(n)*.312, object_weight_n=torch.ones(n)*2.,
        clearances=torch.ones(n, 2)*.01, limits=torch.stack((torch.zeros_like(q), torch.ones_like(q)), -1),
        vertical_components=torch.zeros(n, 2, 2, dtype=torch.float64), pad_measurements=pads,
        parameters=NS(effort_limit_nm=.2),
        pad_sampler=NS(bindings=bindings, rows={(h, d): deepcopy(patch) for h in range(2) for d in range(5)}),
        tripod_contact_snapshot=dict(complete=True, valid_envs=torch.ones(n, dtype=torch.bool),
            tensor_world_physics_steps=9, raw_contacts=raw, raw_friction=friction),
        w=NS(ids=ids, sim=NS(current_time_step_index=26, current_time=26/480)))
    w.wire = NS(data=NS(root_pos_w=w.wp.clone(), root_quat_w=w.wq.clone(), root_vel_w=w.wv.clone(),
        root_com_pose_w=torch.cat((w.wp.clone(), w.wq.clone()), -1), root_com_vel_w=w.wv.clone(),
        _sim_timestamp=.05, _root_com_pose_w=NS(timestamp=.05), _root_com_vel_w=NS(timestamp=.05)),
        root_physx_view=NS(get_masses=lambda: torch.ones(n, 1)*.2))
    return w


def contact(w):
    r = w.tripod_contact_snapshot['raw_contacts'][0]
    for slot, normal in ((2, [.5, 0., -np.sqrt(.75)]), (3, [-.5, 0., -np.sqrt(.75)])):
        r[0][slot] = 1.; r[1][slot] = torch.tensor([0., 0., .3])
        r[2][slot] = torch.tensor(normal); r[3][slot] = -.0001
    r[4][0] = 2; r[5][0] = 2
    w.pad_measurements['effective_pad_loads_n'][0, 0, 0] = 2.
    w.pad_measurements['effective_pad_vectors_w'][0, 0, 0, 2] = -np.sqrt(3.)
    w.pad_measurements['effective_pad_points_w'][0, 0, 0, 2] = .3
    w.pad_measurements['effective_pad_valid'][0, 0, 0] = True
    w.vertical_components[0, 0, 0] = np.sqrt(3.)


def bridge(w, *, receipt_provider=None):
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_measurements import BimanualMPCMeasurementBridge
    options = {} if receipt_provider is None else dict(refresh_receipt_provider=receipt_provider)
    result = BimanualMPCMeasurementBridge(w, run_uuid='cpu_fixture_not_native', **options)
    # SDK-shaped simulation of a real physics refresh AFTER bridge baseline.
    # Only matching existing buffers advance; deliberately stale/missing cache
    # timestamps remain stale/missing for the negative tests.
    prior_step = w._physics_steps
    w._physics_steps += 1; w.w.sim.current_time_step_index += 1; w.w.sim.current_time += w.dt
    if w.tripod_contact_snapshot['tensor_world_physics_steps'] == prior_step:
        w.tripod_contact_snapshot['tensor_world_physics_steps'] += 1
    for data in [hand.data for hand in w.hands] + [w.wire.data]:
        old = data._sim_timestamp
        data._sim_timestamp += w.dt
        for name in ('_body_com_pose_w', '_body_com_vel_w', '_root_com_pose_w', '_root_com_vel_w'):
            buffer = getattr(data, name, None)
            if buffer is not None and buffer.timestamp == old:
                buffer.timestamp = data._sim_timestamp
    return result


def capture(b, **kwargs):
    values = dict(episode_uids=['fixture-episode-10', 'fixture-episode-11'], episode_ids=[10, 11],
        reset_generations=[0, 0], policy_step=1, control_substep=1, live=[True, True], terminal=[False, False])
    values.update(kwargs)
    return b.capture(**values)


def test_com_point_velocity_includes_angular_cross_offset_and_refuses_bad_shapes():
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_measurements import point_velocity
    assert point_velocity([1., 0., 0., 0., 0., 2.], [1., 0., 0.], [1., 3., 0.]) == pytest.approx([-5., 0., 0.])
    with pytest.raises(ValueError):
        point_velocity([0.]*3, [0.]*3, [0.]*3)


def test_builder_accepts_neutral_measurement_stage_without_fixed_wrist_qualification():
    from go2_pvcnn.control.dh116.tabletop_tripod_native_evidence import TripodNativeEvidenceBuilder
    w = fixture()
    result = TripodNativeEvidenceBuilder(w).capture(episode_ids=[10, 11], stage_id='MPC_MEASUREMENT',
        control_step=1, control_substep=1, live=[True, True], terminal=[False, False],
        fatal=[False, False], native_success=[False, False], wrists_fixed=None,
        left_targets_fixed=None, object_external_force_applied=None, induced_disconnect=None)
    assert result['packets'][0]['stage_id'] == 'MPC_MEASUREMENT'
    assert result['packets'][0]['actual_kinematics']['native_attested'] is False
    assert result['packets'][0]['wrists_fixed'] is None


def test_sdk_active_joint_mapping_and_real_com_origin_are_used_without_mutation():
    w = fixture(); before = w.q.clone(); b = bridge(w); s = capture(b)[0]
    expected = w.q[0, 0, w.w.ids[0]].tolist()
    assert s.fields['robot.left_q'].value == tuple(expected)
    assert s.fields['wrist.left_velocity'].value == pytest.approx([0., -.02, 0., 0., 0., 2.])
    assert s.fields['wrist.left_velocity'].origin == 'palm_origin'
    assert s.fields['wrist.left_base_pose'].value == s.fields['wrist.left_pose'].value
    assert s.fields['wrist.left_base_velocity'].origin == 'base_origin'
    assert s.fields['wrist.left_base_velocity'].value == s.fields['wrist.left_velocity'].value
    assert s.fields['fingertips.left_thumb.position'].value == pytest.approx([0., 0., .3])
    assert s.fields['fingertips.left_thumb.position'].kind == 'derived'
    assert s.fields['object.com'].value == pytest.approx([0., 0., .32])
    assert s.fields['object.mass'].value == pytest.approx(.2)
    assert torch.equal(w.q, before)
    assert b.last_evidence['native_attested'] is False


def test_accepted_scalar_is_not_resultant_norm_and_object_force_has_opposite_sign():
    w = fixture(); contact(w); s = capture(bridge(w))[0]
    scalar = np.asarray(s.fields['contact.accepted_load'].value)
    force = np.asarray(s.fields['contact.accepted_force_on_hand'].value)
    assert scalar[0, 0] == pytest.approx(2.)
    assert np.linalg.norm(force[0, 0]) == pytest.approx(np.sqrt(3.))
    assert np.asarray(s.fields['contact.accepted_force_on_object'].value) == pytest.approx(-force)
    assert s.fields['contact.normal_impulse_derived'].kind == 'derived'
    assert np.asarray(s.fields['contact.normal_impulse_derived'].value)[0, 0] == pytest.approx(2/480)
    assert s.fields['contact.accepted_force_on_hand'].force_side == 'ON_HAND'
    assert s.fields['contact.accepted_force_on_object'].force_side == 'ON_OBJECT'


def test_wholehand_rejection_friction_partner_and_passive_support_are_not_fabricated():
    w = fixture(); contact(w); s = capture(bridge(w))[0]
    for key in ('contact.rejected_load', 'contact.partner_body_id', 'support.support_margin',
                'safety.force_limit', 'safety.collision_clearance', 'object.centerline'):
        assert s.fields[key].value is None and not s.fields[key].valid
        assert s.fields[key].missing_reason
    assert s.fields['contact.rejected_load_hand'].value == ((0.), (0.))
    assert s.fields['contact.friction_force_hand'].valid
    assert s.fields['contact.friction_force_hand'].source.find('whole_hand') >= 0
    assert not s.fields['contact.contact_age'].valid


def test_partial_contacts_have_none_and_mask_and_slip_uses_each_actual_raw_point():
    w = fixture(); contact(w); s = capture(bridge(w))[0]
    field = s.fields['contact.contact_position']
    assert field.valid_mask == ((True, False, False), (False, False, False))
    assert field.value[0][1] is None and field.value[1][0] is None
    assert s.fields['contact.hand_body_id'].value[0][0] == 0
    assert s.fields['contact.partner_body_id'].value is None
    # COM is 10mm from each real point; omega_z=2 produces vy=-.02.
    # Both different contact normals are in xz, so vy is tangent to each.
    slip = s.fields['contact.slip_speed']
    assert slip.kind == 'derived' and slip.value[0][0] == pytest.approx(.02)
    assert slip.valid_mask == field.valid_mask and slip.value[0][1] is None
    assert 'not_fixed_anchor' in slip.source
    assert s.fields['contact.contact_mask'].kind == 'derived'
    assert s.fields['contact.contact_mask'].contact_basis == 'actual_contact'


def test_absent_com_only_invalidates_that_env_and_reset_cannot_reuse_a_pre_reset_clock():
    w = fixture(); w.hands[0].data.body_com_pose_w[1, 0, 0] = float('nan')
    result = capture(bridge(w))
    assert result[0].fields['wrist.left_velocity'].valid
    assert not result[1].fields['wrist.left_velocity'].valid
    w = fixture(); b = bridge(w); capture(b)
    with pytest.raises(ValueError, match='reset|stale|advance'):
        capture(b, episode_uids=['new-left', 'new-right'], episode_ids=[12, 13], reset_generations=[1, 1])


def test_clock_and_raw_evidence_are_owned_and_runner_declarations_remain_unattested():
    w = fixture(); contact(w); b = bridge(w); s = capture(b)[0]
    assert s.fields['contact.accepted_load'].timestamp_s == s.clock.world_time_s
    assert s.clock.sdk_time_s == pytest.approx(27/480)
    assert b.last_evidence['setup_declaration_provenance'] == 'runner_self_reported_not_independently_measured'
    assert b.last_evidence['runner_setup_declarations']['wrists_fixed'] is None
    assert b.last_evidence['builder_evidence']['packets'][0]['fatal'] is None
    assert b.last_evidence['builder_evidence']['packets'][0]['native_success'] is None
    assert len(b.last_evidence['extra_sdk_reads']) == 7
    original_com = b.last_evidence['extra_sdk_reads'][0]
    assert original_com['value'][0][0][:3] == pytest.approx([.01, 0., .3])
    assert original_com['origin'] == 'body_com'
    assert original_com['valid_envs'] == [True, True]
    assert original_com['episode_uids'] == ['fixture-episode-10', 'fixture-episode-11']
    raw = b.last_evidence['builder_evidence']['packets'][0]['raw_contacts'][0]
    w.tripod_contact_snapshot['raw_contacts'][0][0].fill_(99.)
    assert raw['normal_scalar_n'] == 1.
    assert s.fields['contact.accepted_load'].value[0][0] == 2.


def test_raw_observed_hand_contribution_preserves_rejected_force_without_support_credit():
    w = fixture(); contact(w)
    f = w.tripod_contact_snapshot['raw_friction'][0]
    f[0][5] = torch.tensor([.1, 0., -.2]); f[1][5] = torch.tensor([0., 0., .3])
    f[2][0] = 1; f[3][0] = 5
    w.vertical_components[0, 0, 1] = .2
    b = bridge(w); s = capture(b)[0]
    observed = b.last_evidence['observed_filtered_hand_force_z_on_object'][0]
    assert observed['normal_n'] == pytest.approx([np.sqrt(3.), 0.])
    assert observed['friction_n'] == pytest.approx([.2, 0.])
    assert observed['total_n'] == pytest.approx([np.sqrt(3.)+.2, 0.])
    assert observed['includes_rejected_normal'] is True
    assert observed['accepted_pad_support_credit'] is False
    assert observed['airborne_proof'] is False
    assert observed['normal_complete'] and observed['friction_complete'] and observed['ownership_valid']
    assert s.fields['support.total_upward_support'].value is None
    assert 'accepted' in s.fields['support.total_upward_support'].missing_reason
    assert 'table' in s.fields['support.table_reaction'].missing_reason


@pytest.mark.parametrize('change', ['policy', 'source_episode', 'relative_clocks', 'reset_without_generation'])
def test_identity_clocks_cannot_silently_change(change):
    w = fixture(); b = bridge(w); capture(b)
    w._physics_steps += 1; w.w.sim.current_time_step_index += 1; w.w.sim.current_time += w.dt
    w.tripod_contact_snapshot['tensor_world_physics_steps'] = w._physics_steps
    kwargs = {}
    if change == 'policy': kwargs['policy_step'] = 0
    elif change == 'source_episode': kwargs['episode_ids'] = [99, 11]
    elif change == 'relative_clocks': w.w.sim.current_time_step_index += 1
    else: kwargs['reset'] = True
    with pytest.raises(ValueError, match='policy|episode|clock|reset'):
        capture(b, **kwargs)


@pytest.mark.parametrize('fault', ['missing_timestamp', 'stale_timestamp'])
def test_lazy_sdk_property_needs_cache_freshness_and_keeps_raw_value_for_audit(fault):
    w = fixture()
    if fault == 'missing_timestamp': del w.hands[0].data._body_com_vel_w
    else: w.hands[0].data._body_com_vel_w.timestamp = .01
    b = bridge(w); s = capture(b)[0]
    assert not s.fields['wrist.left_velocity'].valid
    read = b.last_evidence['extra_sdk_reads'][2]
    assert read['read_kind'] == 'SDK_data_property_cache'
    assert read['available'] and not read['source_valid']
    assert read['value'][0][0][5] == 2.
    assert read['independent_backend_read_attested'] is False


def test_outer_clock_covers_derived_state_construction_and_failed_capture_clears_previous(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_bimanual_mpc_measurements as module
    w = fixture(); original = module.make_mpc_state
    def changed_after_read(clock, fields):
        state = original(clock, fields)
        w.w.sim.current_time_step_index += 1
        return state
    monkeypatch.setattr(module, 'make_mpc_state', changed_after_read)
    b = bridge(w)
    with pytest.raises(RuntimeError, match='clock'):
        capture(b)
    assert b.last_evidence is None


def test_stale_raw_and_stale_sdk_cache_fail_closed_and_capture_clock_is_bracketed():
    w = fixture(); w.tripod_contact_snapshot['tensor_world_physics_steps'] = 8
    s = capture(bridge(w))[0]
    assert not s.fields['contact.accepted_load'].valid
    w = fixture(); w.hands[0].data.joint_pos[0, 0] += .1
    s = capture(bridge(w))[0]
    assert not s.fields['robot.left_q'].valid
    w = fixture(); original = w.wire.root_physx_view.get_masses
    def changed_clock():
        w.w.sim.current_time_step_index += 1
        return original()
    w.wire.root_physx_view.get_masses = changed_clock
    with pytest.raises(RuntimeError, match='clock'):
        capture(bridge(w))


def test_missing_com_is_na_and_terminal_is_retained_but_repeated_or_changed_episode_is_refused():
    w = fixture(); del w.hands[0].data.body_com_pose_w
    s = capture(bridge(w), terminal=[True, False], live=[True, True])[0]
    assert s.clock.terminal and s.fields['robot.left_q'].valid
    assert not s.fields['wrist.left_velocity'].valid
    w = fixture(); b = bridge(w); capture(b)
    with pytest.raises(ValueError, match='stale|repeat'):
        capture(b)
    w._physics_steps += 1; w.w.sim.current_time_step_index += 1; w.w.sim.current_time += w.dt
    with pytest.raises(ValueError, match='episode|reset'):
        capture(b, episode_uids=['silently-changed', 'fixture-episode-11'])


def test_old_equal_data_and_com_generations_cannot_be_retimestamped_as_fresh():
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import validate_planning_state
    w = fixture(); b = bridge(w); capture(b)
    # Reproduce the reviewer probe: clocks advance, no DATA update occurs.
    w._physics_steps += 1; w.w.sim.current_time_step_index += 1; w.w.sim.current_time += w.dt
    s = capture(b)[0]
    assert not s.fields['robot.left_q'].valid
    assert not s.fields['wrist.left_velocity'].valid
    assert not s.fields['object.com'].valid
    with pytest.raises(ValueError):
        validate_planning_state(s, s.clock, max_age_s=0,
            required_fields=('wrist.left_velocity', 'robot.left_q'))


def test_constructing_at_same_step_is_not_proof_of_a_fresh_data_generation():
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_measurements import BimanualMPCMeasurementBridge
    w = fixture(); b = BimanualMPCMeasurementBridge(w, run_uuid='unadvanced-fixture')
    s = capture(b)[0]
    assert not s.fields['robot.left_q'].valid
    assert not s.fields['wrist.left_velocity'].valid


def test_tracking_error_uses_actual_backend_targets_and_absent_backend_is_na():
    w = fixture()
    for hi, hand in enumerate(w.hands):
        hand.root_physx_view.get_dof_position_targets = lambda hi=hi: w.q[:, hi].clone()+.1
    s = capture(bridge(w))[0]
    assert s.fields['safety.tracking_error'].valid
    assert np.asarray(s.fields['safety.tracking_error'].value) == pytest.approx(np.full((2, 6), .1))
    assert 'backend' in s.fields['safety.tracking_error'].source
    w = fixture(); s = capture(bridge(w))[0]
    assert s.fields['safety.tracking_error'].value is None
    assert not s.fields['safety.tracking_error'].valid


def test_contact_centroid_comes_from_accepted_raw_points_instead_of_unverified_cache():
    w = fixture(); contact(w)
    w.pad_measurements['effective_pad_points_w'][0, 0, 0, 2] = 123.
    b = bridge(w); s = capture(b)[0]
    assert s.fields['contact.contact_position'].valid
    assert s.fields['contact.contact_position'].value[0][0] == pytest.approx([0., 0., .3])
    assert 'raw' in s.fields['contact.contact_position'].source


def advance_fixture_refresh(w):
    w._physics_steps += 1; w.w.sim.current_time_step_index += 1; w.w.sim.current_time += w.dt
    w.tripod_contact_snapshot['tensor_world_physics_steps'] = w._physics_steps
    for data in [hand.data for hand in w.hands] + [w.wire.data]:
        data._sim_timestamp += w.dt
        for name in ('_body_com_pose_w', '_body_com_vel_w', '_root_com_pose_w', '_root_com_vel_w'):
            buffer = getattr(data, name, None)
            if buffer is not None:
                buffer.timestamp = data._sim_timestamp


def test_closed_terminal_slot_can_be_inactive_with_new_uid_while_other_env_continues():
    w = fixture(); b = bridge(w)
    first = capture(b, terminal=[True, False])
    terminal_previous = b._previous[0]
    advance_fixture_refresh(w)
    same_uid = capture(b, live=[False, True])
    assert not same_uid[0].fields['robot.left_q'].valid
    assert b._previous[0] == terminal_previous
    advance_fixture_refresh(w)
    second = capture(b, episode_uids=['next-left-episode', 'fixture-episode-11'], episode_ids=[12, 11],
        reset_generations=[1, 0], live=[False, True])
    assert second[0].clock.episode_uid == 'next-left-episode'
    assert not second[0].clock.terminal
    assert all(not f.valid and f.value is None for f in second[0].fields.values())
    assert b._previous[0] == terminal_previous
    assert second[1].fields['robot.right_q'].valid
    assert second[1].clock.world_step == first[1].clock.world_step+2
    advance_fixture_refresh(w)
    with pytest.raises(ValueError, match='terminal|episode'):
        capture(b, episode_uids=['next-left-episode', 'fixture-episode-11'], episode_ids=[12, 11],
            reset_generations=[1, 0], live=[True, True])


def test_temporarily_inactive_nonterminal_slot_keeps_history_and_can_legally_resume():
    w = fixture(); b = bridge(w); first = capture(b)
    previous = b._previous[0]
    advance_fixture_refresh(w)
    masked = capture(b, live=[False, True])
    assert not masked[0].fields['robot.left_q'].valid
    assert b._previous[0] == previous
    advance_fixture_refresh(w)
    resumed = capture(b)
    assert resumed[0].fields['robot.left_q'].valid
    assert resumed[0].clock.world_step == first[0].clock.world_step+2


def refresh_receipt(w, sequence):
    clock = [w._physics_steps, w.w.sim.current_time_step_index, w.w.sim.current_time]
    generation = {f'hands[{h}]': hand.data._sim_timestamp for h, hand in enumerate(w.hands)}
    generation['wire'] = w.wire.data._sim_timestamp
    return dict(sequence=sequence, world_sdk_clock_before=clock[:], world_sdk_clock_after=clock[:],
        data_generation_before={key: value-w.dt for key, value in generation.items()},
        data_generation_after=generation, dt=w.dt,
        original_method_sha256=hashlib.sha256(inspect.getsource(type(w).refresh).encode('utf-8')).hexdigest(),
        source_scope='actual original TensorWorld.refresh returned;SDK_DATA generation only')


def test_actual_refresh_receipt_keeps_other_env_current_after_an_unrelated_reset_refresh():
    w = fixture(); receipt = {}; b = bridge(w, receipt_provider=lambda: receipt)
    receipt.update(refresh_receipt(w, 1))
    first = capture(b, terminal=[True, False])
    assert first[1].fields['robot.right_q'].valid
    # Original reset refresh advances all DATA objects, without a physics step.
    for data in [hand.data for hand in w.hands] + [w.wire.data]:
        data._sim_timestamp += w.dt
    advance_fixture_refresh(w)
    receipt.clear(); receipt.update(refresh_receipt(w, 3))
    second = capture(b, episode_uids=['reset-left', 'fixture-episode-11'], episode_ids=[12, 11],
        reset_generations=[1, 0], live=[False, True])
    assert not second[0].fields['robot.left_q'].valid
    assert second[1].fields['robot.right_q'].valid
    assert second[1].fields['wrist.right_velocity'].valid
    assert b.last_evidence['refresh_receipt_evidence']['source_valid'] is True


@pytest.mark.parametrize('fault', ['stale_clock', 'no_data_advance', 'bad_dt', 'bad_hash', 'reused_sequence',
                                  'source_scope', 'naked_data_update', 'absent_receipt'])
def test_invalid_refresh_receipt_never_falls_back_to_constructor_generation(fault):
    w = fixture(); receipt = {}; b = bridge(w, receipt_provider=lambda: receipt)
    receipt.update(refresh_receipt(w, 1)); capture(b)
    old_clock = receipt['world_sdk_clock_after'][:]
    advance_fixture_refresh(w)
    receipt.clear(); receipt.update(refresh_receipt(w, 2))
    if fault == 'stale_clock': receipt['world_sdk_clock_after'] = old_clock
    elif fault == 'no_data_advance': receipt['data_generation_before'] = dict(receipt['data_generation_after'])
    elif fault == 'bad_dt': receipt['dt'] = 1/60
    elif fault == 'bad_hash': receipt['original_method_sha256'] = '0'*64
    elif fault == 'reused_sequence': receipt['sequence'] = 1
    elif fault == 'source_scope': receipt['source_scope'] = 'caller says refreshed'
    elif fault == 'naked_data_update': w.hands[0].data._sim_timestamp += w.dt
    else: receipt.clear()
    s = capture(b)[0]
    assert not s.fields['robot.left_q'].valid
    assert not s.fields['wrist.left_velocity'].valid
    assert not s.fields['object.com'].valid
    assert b.last_evidence['refresh_receipt_evidence']['source_valid'] is False
