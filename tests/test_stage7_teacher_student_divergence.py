"""Stage7 offline math and original-controller fixtures; no simulator."""
import copy
import importlib
import importlib.util
import json
import subprocess
import sys

import pytest
import torch
import numpy as np


def api():
    name = 'go2_pvcnn.control.dh116.tabletop_vision_teacher_student_divergence'
    assert importlib.util.find_spec(name), 'Stage7 divergence analyzer is missing'
    return importlib.import_module(name)


def labels(n):
    return {name: torch.ones(n, dtype=torch.bool) for name in ('START', 'CONTACT', 'LIFT_ONSET', 'HOLD')}


def test_action_math_uses_absolute_errors_linear_p95_and_exact_signs():
    teacher = torch.zeros(2, 24, dtype=torch.float64)
    teacher[:, 0] = torch.tensor([1., -1.])
    predicted = teacher.clone()
    predicted[:, 0] = torch.tensor([-.5, -.5])
    report = api().action_statistics(predicted, teacher, labels(2))
    group = report['all']['groups']['left_thumb']
    assert group['mae'] == .5  # errors 1.5,0,.5,0
    assert group['p95'] == pytest.approx(1.35)
    assert group['max'] == 1.5
    assert group['direction']['nonzero_teacher_elements'] == 2
    assert group['direction']['nonzero_teacher_sign_agreement'] == .5
    assert group['direction']['zero_teacher_elements'] == 2
    assert group['direction']['zero_teacher_predicted_nonzero_elements'] == 0
    assert report['all']['by_dimension'][0]['mae'] == 1.
    assert report['event_strata']['CONTACT'] == report['all']
    assert report['event_labels_may_overlap'] is True


def test_zero_teacher_labels_and_tiny_differences_are_not_hidden_by_epsilon():
    teacher = torch.zeros(1, 24, dtype=torch.float64)
    predicted = teacher.clone(); predicted[0, 17] = 1e-300
    masks = labels(1); masks['HOLD'][:] = False
    result = api().action_statistics(predicted, teacher, masks)
    right = result['all']['groups']['right_palm_xyz']
    assert right['max'] == 1e-300
    assert right['direction']['nonzero_teacher_sign_agreement'] is None
    assert right['direction']['zero_teacher_predicted_nonzero_elements'] == 1
    assert result['event_strata']['HOLD']['rows'] == 0
    assert result['event_strata']['HOLD']['groups']['all']['mae'] is None


@pytest.mark.parametrize('bad', ['nonfinite', 'shape', 'label_dtype', 'label_shape'])
def test_actions_refuse_ambiguous_or_nonfinite_rows(bad):
    teacher = torch.zeros(2, 24); predicted = teacher.clone(); masks = labels(2)
    if bad == 'nonfinite': predicted[0, 0] = float('nan')
    if bad == 'shape': predicted = predicted[:, :23]
    if bad == 'label_dtype': masks['START'] = masks['START'].float()
    if bad == 'label_shape': masks['START'] = masks['START'][:1]
    with pytest.raises(ValueError): api().action_statistics(predicted, teacher, masks)


def command_fixture(length=17):
    """Saved targets produced by original control+advance, not the analyzer."""
    from test_vision_command_shadow import fixture
    from test_vision_sequence_dataset import fixture_data
    from go2_pvcnn.control.dh116 import tabletop_vision_command_shadow as shadow
    from go2_pvcnn.control.dh116.tabletop_vision_sequence_dataset import SequenceDataset
    t, w = fixture(); w.clearances.fill_(.02); w.palms.zero_()
    data = fixture_data((length,)); data['actions'].zero_(); data['reference_actions'].zero_()
    for row in data['sample_metadata']:
        row['previous_action'] = [0.] * 24
    data['observations'][:, -961+934:-961+958] = 0.
    starts, contexts, targets = [], [], []
    contract = shadow.controller_contract(t, w)
    for tick in range(length):
        starts.append({key: value[0].clone() for key, value in shadow.command_state(t, w).items()})
        subcontexts, subtargets = [], []
        for substep in range(8):
            live = torch.ones(3, dtype=torch.bool)
            ctx = shadow.measured_context(t, w, live)
            allowed = t.control(torch.zeros(3, 24), live, w.dt); ctx['allowed'] = allowed.clone()
            w.advance(t, allowed, live)
            subcontexts.append({key: value[0].clone() for key, value in ctx.items()})
            subtargets.append(dict(finger=t.command[0].clone(), palm=w.palms[0].clone(), wrist_angle=w.wrist_angles[0].clone()))
        contexts.append({key: torch.stack([r[key] for r in subcontexts]) for key in subcontexts[0]})
        targets.append({key: torch.stack([r[key] for r in subtargets]) for key in subtargets[0]})
    rows = data['sample_metadata']
    samples = {key: [r[key] for r in rows] for key in ('run_uuid', 'env_id', 'episode_uid', 'tick', 'policy_timestamp_s', 'first_physics_step', 'terminal')}
    samples.update(valid=torch.ones(length, dtype=torch.bool), actions=data['actions'].clone(),
        previous_action=torch.zeros(length, 24), event_id=['fixture']*length)
    stack = lambda trees: {key: torch.stack([r[key] for r in trees]) for key in trees[0]}
    context = dict(version='dh116-sequence-context-v1', dataset_sha256='d'*64,
        controller_source_sha256=shadow.controller_source_sha256(), action_contract_sha256=shadow.action_contract_sha256(),
        controller_contract=contract, samples=samples, controller_start=stack(starts),
        physics_context=stack(contexts), teacher_targets=stack(targets),
        events=dict(opposed_contact=torch.zeros(length, dtype=torch.bool),
            measured_palm_velocity_m_s=torch.zeros(length, 2, 3), native_hold_qualified=torch.zeros(length, dtype=torch.bool)))
    sequence = SequenceDataset(data, context, dataset_sha256='d'*64)
    return sequence, contract


def test_l16_keeps_own_cumulative_states_in_original_units_and_no_mutation():
    sequence, contract = command_fixture()
    prediction = sequence.data['actions'].clone().double()
    prediction[:, 0] = 1e-3; prediction[:, 14] = 1e-3; prediction[:, 18] = 1e-3
    context_before = copy.deepcopy(sequence.context)
    result = api().compare_l16_commands(sequence, {'fixture': prediction}, contract, batch_windows=2)
    assert result['teacher_exact_parity']['passed'] is True
    assert result['full_window_count'] == 2
    values = result['models']['fixture']['signed_endpoint_differences']
    # Original rate constants times16/60, independently of shadow's internals.
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    assert values['finger'][0, 0] == pytest.approx(1e-3 * ACTION_CONTRACT['joint_rate_rad_s'] * 16/60)
    assert values['palm'][0, 0, 2] == pytest.approx(1e-3 * ACTION_CONTRACT['translation_rate_m_s'] * 16/60)
    assert values['wrist_angle'][0, 0, 0] == pytest.approx(1e-3 * ACTION_CONTRACT['rotation_rate_rad_s'] * 16/60)
    assert result['normalization_scales'] == dict(finger=.01, palm=.0005, wrist_angle=.01)
    assert result['models']['fixture']['groups']['finger']['raw']['max'] == pytest.approx(float(values['finger'].abs().max()))
    assert torch.equal(sequence.context['teacher_targets']['finger'], context_before['teacher_targets']['finger'])
    assert torch.equal(sequence.context['controller_start']['joint_command_rad'], context_before['controller_start']['joint_command_rad'])


def test_l16_rejects_recorded_teacher_parity_error_before_student_projection():
    sequence, contract = command_fixture()
    sequence.context['teacher_targets']['finger'][0, 0, 0] += .0001
    with pytest.raises(ValueError, match='parity'):
        api().compare_l16_commands(sequence, {'fixture': sequence.data['actions']}, contract)


def test_l16_does_not_cross_missing_tick_or_use_padded_short_windows():
    sequence, contract = command_fixture(17)
    sequence.windows[0]['row_indices'][8] = 10
    with pytest.raises(ValueError, match='continuous|window'):
        api().compare_l16_commands(sequence, {'fixture': sequence.data['actions']}, contract)
    short, contract = command_fixture(8)
    with pytest.raises(ValueError, match='full|16'):
        api().compare_l16_commands(short, {'fixture': short.data['actions']}, contract)


def test_l16_refuses_teacher_recorded_target_reset_each_step():
    sequence, contract = command_fixture(16)
    prediction = sequence.data['actions'].clone().double(); prediction[:, 18] = .01
    out = api().compare_l16_commands(sequence, {'fixture': prediction}, contract)
    endpoint = out['models']['fixture']['signed_endpoint_differences']['wrist_angle'][0, 0, 0]
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    assert endpoint == pytest.approx(.01 * ACTION_CONTRACT['rotation_rate_rad_s'] * 16/60)
    assert abs(endpoint) > .01 * ACTION_CONTRACT['rotation_rate_rad_s'] / 60 * 15


def test_l16_requires_all_sixteen_actions_to_be_actually_active():
    sequence, contract = command_fixture(16)
    sequence.context['physics_context']['live'][8] = False
    with pytest.raises(ValueError, match='active|full|16'):
        api().compare_l16_commands(sequence, {'fixture': sequence.data['actions']}, contract)


def test_forward_has_one_final_tanh_and_no_rng_or_parameter_mutation():
    class ConstantActor(torch.nn.Module):
        def forward(self, obs):
            assert obs.dtype == torch.float32 and obs.shape[-1] == 3844
            return torch.full((len(obs), 24), .5)
    rng = torch.get_rng_state().clone()
    result = api().bounded_predictions(ConstantActor(), torch.zeros(3, 3844, dtype=torch.float64), batch_size=2)
    assert torch.equal(result, torch.full((3, 24), .5).tanh())
    assert torch.equal(rng, torch.get_rng_state())


def checkpoint_fixture():
    return dict(policy={'actor.weight': torch.ones(2), 'critic.weight': torch.zeros(2),
        'log_std': torch.zeros(24)}, optimizer={'state': {'step': torch.tensor(3)}},
        learning_rate=.0003, iteration=7, timesteps=64, episodes=8, successes=2,
        torch_rng=torch.get_rng_state().clone(), cuda_rng=[], contract={'actor': 3844, 'critic': 371})


@pytest.mark.parametrize('field', ['critic', 'log_std', 'optimizer', 'learning_rate', 'iteration', 'torch_rng', 'contract', 'unknown'])
def test_all_original_non_actor_state_is_hard_bound(field):
    source = checkpoint_fixture(); candidate = copy.deepcopy(source)
    if field in ('critic', 'log_std'):
        key = 'critic.weight' if field == 'critic' else 'log_std'; candidate['policy'][key][0] += 1
    elif field == 'optimizer': candidate['optimizer']['state']['step'] += 1
    elif field == 'torch_rng': candidate[field][0] ^= 1
    elif field == 'contract': candidate[field]['critic'] = 372
    elif field == 'unknown': candidate['unexpected'] = True
    else: candidate[field] += 1
    with pytest.raises(ValueError, match='non.Actor|original|unknown'):
        api().audit_source_preservation(source, candidate)


def test_actual_five_independent_bc_appendices_are_allowed_without_relaxing_original_keys():
    source = checkpoint_fixture(); candidate = copy.deepcopy(source)
    for field in ('stage6_sequence_bc', 'stage6_bc_optimizer', 'stage6_bc_sampling_rng',
                  'stage6_pointnet_probe', 'stage6_diagnostic_input'):
        candidate[field] = {'independent': True}
    candidate['policy']['actor.weight'] += 2
    result = api().audit_source_preservation(source, candidate)
    assert result['all_original_non_actor_state_unchanged'] is True
    assert result['actor_parameter_l2_distance'] == pytest.approx(8 ** .5)


@pytest.mark.parametrize('bad', ['partial', 'wrong_arm', 'wrong_prepared', 'ambiguous', 'model_sha'])
def test_completed_model_never_reads_partial_or_guesses_checkpoint(tmp_path, bad):
    m = api(); path = tmp_path/'final.pt'; path.write_bytes(b'actual completed model')
    model_id = m.file_identity(path); prepared = {'path': '/actual/prepared', 'sha256': 'a'*64}
    source = {'path': '/actual/c58', 'sha256': 'b'*64}
    report = dict(profile='B2', status='completed', steps=3000, requested_steps=3000,
        valid_exposures=192000, source_immutable=True, ppo_updates=0,
        checkpoint_selection='prespecified_final_update_not_train_MSE', prepared=prepared,
        source_checkpoint=source, checkpoints=[dict(**model_id, status='completed', step=3000)])
    if bad == 'partial': report['status'] = 'running'
    if bad == 'wrong_arm': report['profile'] = 'B1'
    if bad == 'wrong_prepared': report['prepared']['sha256'] = 'f'*64
    if bad == 'ambiguous': report['checkpoints'].append(report['checkpoints'][0].copy())
    if bad == 'model_sha': report['checkpoints'][0]['sha256'] = 'f'*64
    file = tmp_path/'report.json'; file.write_text(json.dumps(report))
    # The explicit expected prepared identity must be independent of mutation.
    expected = {'path': '/actual/prepared', 'sha256': 'a'*64}
    with pytest.raises(ValueError): m.completed_model(file, 'B2', expected, source)


def test_completed_report_selects_unique_prespecified_final_model(tmp_path):
    m = api(); path = tmp_path/'final.pt'; path.write_bytes(b'final')
    identity = m.file_identity(path); prepared = {'path': '/actual/prepared', 'sha256': 'a'*64}
    source = {'path': '/actual/c58', 'sha256': 'b'*64}
    report = dict(profile='B0', status='completed', steps=3000, requested_steps=3000,
        valid_exposures=192000, source_immutable=True, ppo_updates=0,
        checkpoint_selection='prespecified_final_update_not_train_MSE', prepared=prepared,
        source_checkpoint=source, checkpoints=[dict(**identity, status='completed', step=3000)])
    file = tmp_path/'report.json'; file.write_text(json.dumps(report))
    result = m.completed_model(file, 'B0', prepared, source)
    assert result['checkpoint'] == identity
    assert result['checkpoint_selection'] == 'unique_completed_update3000;not_MSE'


def test_dependency_snapshot_covers_lazy_command_kernel_import_before_forward():
    result = subprocess.run([sys.executable, '-c',
        'from go2_pvcnn.control.dh116 import tabletop_vision_teacher_student_divergence as m; '
        'before=m._dependencies(); '
        'from go2_pvcnn.control.dh116.tabletop_tensor_world import constrain_palm_z_movement; '
        'after=m._dependencies(); '
        'assert before==after, sorted(set(after)-set(before))'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr


def event_fixture():
    from test_stage6_physical_divergence import sample, run
    def samples(uid, owner):
        rows = []
        for sub in range(1, 7):
            row = sample(substep=sub, uid=f'{uid}:0:0', owner=owner, contact=True, terminal=sub==6)
            load = np.zeros((2, 5)); load[:, :2] = .1
            row.update(measurement_valid=True, prior_anchor_latched=True, slip_applicable=True,
                slip_valid=True, slip_m=0., failure_code=0,
                pad=dict(load_n=load, accepted=load>0),
                target_stages={'requested_palm': {'left.z': dict(value=0., unit='m')}},
                raw_pointer={'path':f'/{uid}/raw.gz','sha256':'a'*64,'line':sub},
                diagnostic_pointer={'path':f'/{uid}/physics.npz','sha256':'b'*64,'batch':sub-1,'env_index':0})
            rows.append(row)
        return run(uid, rows)
    candidate = samples('candidate', 'student'); references = [samples('reference1', 'teacher_replay'), samples('reference2', 'teacher_replay')]
    policy = [dict(env_id=0, policy_step=0, owner='student', run_uuid='candidate',
        episode_uid='candidate:0:0', recorded_teacher_episode_uid=candidate['samples'][0]['source_episode_uid'],
        issued_action=[.01]+[0.]*23, recorded_teacher_action=[0.]*24,
        clock_before_action=dict(physics_step=0, simulation_step_index=18, timestamp_s=0.),
        raw_pointer=dict(path='/candidate/live.gz',sha256='c'*64,line=1))]
    return candidate, references, policy


def test_six_first_events_are_independent_and_keep_real_pointers_and_units():
    candidate, references, policy = event_fixture(); rows = candidate['samples']
    rows[0]['target_stages']['requested_palm']['left.z']['value'] = 1e-15
    rows[1]['q_rad'][0] = .001
    rows[2]['pad']['load_n'][0, 1] = 0.; rows[2]['pad']['accepted'][0, 1] = False
    rows[3]['slip_m'] = .0006
    rows[4]['failure_code'] = 3
    summary = api().summarize_six_first(candidate, references, policy, np.full(12, .01))
    env = summary['per_env'][0]
    assert env['first_action_difference']['policy_step'] == 0
    assert env['first_control_target_difference']['substep'] == 1
    assert env['first_joint_tracking_difference']['substep'] == 2
    assert env['first_finger_pad_contact_loss']['substep'] == 3
    assert env['first_finger_pad_contact_loss']['hand'] == 'left'
    assert env['first_finger_pad_contact_loss']['digit'] == 'index'
    assert env['first_finger_pad_contact_loss']['aggregate_opposed_at_sample'] is True
    assert env['first_object_slip']['substep'] == 4
    assert env['first_failure']['substep'] == 5
    assert env['first_terminal']['substep'] == 6
    for key in ('first_control_target_difference','first_joint_tracking_difference','first_finger_pad_contact_loss','first_object_slip','first_failure'):
        event = env[key]
        assert event['raw_pointer']['sha256'] == 'a'*64
        assert event['diagnostic_pointer']['sha256'] == 'b'*64
        assert event['execution_episode_uid'] == 'candidate:0:0' and event['cause_claimed'] is False
    assert env['first_control_target_difference']['differences'][0]['unit'] == 'm'


def test_preaction_event_uses_saved_policy_anchor_source_never_inherits_postphysics_getter_evidence():
    candidate,references,policy=event_fixture()
    candidate['samples'][0].update(clock_quality='actual_postphysical_independent_getter_pair',
        independent_getter_clock_pair_available=True)
    result=api().summarize_six_first(candidate,references,policy,np.full(12,.01))['per_env'][0]
    event=result['first_action_difference']
    assert event['clock_quality']=='actual_saved_pre_action_World_SDK_policy_anchor'
    assert event['independent_getter_clock_pair_available'] is False
    assert event['clock_before_action']==policy[0]['clock_before_action']
    assert event['raw_pointer']==policy[0]['raw_pointer']
    assert event['world_physics_step']==0 and event['diagnostic_pointer'] is None


def test_primary_numeric_difference_is_not_hidden_inside_observed_repeat_range():
    candidate,references,policy=event_fixture()
    references[0]['samples'][0]['target_stages']['requested_palm']['left.z']['value']=0.
    references[1]['samples'][0]['target_stages']['requested_palm']['left.z']['value']=2e-6
    candidate['samples'][0]['target_stages']['requested_palm']['left.z']['value']=1e-6
    references[0]['samples'][0]['q_rad'][0]=0.
    references[1]['samples'][0]['q_rad'][0]=-2e-4
    candidate['samples'][0]['q_rad'][0]=-1e-4
    out=api().summarize_six_first(candidate,references,policy,np.full(12,.01))['per_env'][0]
    target=out['first_control_target_difference'];tracking=out['first_joint_tracking_difference']
    assert target['substep']==1 and tracking['substep']==1
    assert target['comparison_reference']['execution_run_uuid']=='reference1'
    assert target['comparison_reference']['raw_pointer']['path']=='/reference1/raw.gz'
    assert target['differences'][0]['reference_value']==0.
    assert tracking['joints'][0]['comparison_reference']['execution_run_uuid']=='reference1'
    assert out['first_control_target_repeat_range_departure'] is None
    assert out['first_joint_tracking_repeat_range_departure'] is None
    assert out['first_significant_tracking_departure'] is None


def test_explicit_original_teacher_is_primary_even_when_not_listed_first():
    candidate,references,policy=event_fixture()
    for row in references[1]['samples']: row['action_owner']='original_Teacher02'
    references[1]['samples'][0]['target_stages']['requested_palm']['left.z']['value']=2e-6
    references[1]['samples'][0]['q_rad'][0]=-2e-4
    out=api().summarize_six_first(candidate,references,policy,np.full(12,.01))['per_env'][0]
    assert out['first_control_target_difference']['comparison_reference']['execution_run_uuid']=='reference2'
    assert out['first_joint_tracking_difference']['joints'][0]['comparison_reference']['execution_run_uuid']=='reference2'


def test_every_new_event_keeps_sdk_step_with_actual_pre_vs_derived_post_source():
    candidate,references,policy=event_fixture()
    for row in candidate['samples']: row['derived_sdk_physics_step']=row['world_physics_step']+18
    candidate['samples'][0]['target_stages']['requested_palm']['left.z']['value']=.001
    candidate['samples'][0]['q_rad'][0]=.02
    out=api().summarize_six_first(candidate,references,policy,np.full(12,.01))['per_env'][0]
    action=out['first_action_difference']
    assert action['sdk_physics_step']==18 and action['sdk_physics_step_is_derived'] is False
    assert action['sdk_step_source']=='actual_clock_before_action.simulation_step_index'
    for name in ('first_control_target_difference','first_joint_tracking_difference','first_significant_tracking_departure'):
        event=out[name]
        assert event['sdk_physics_step']==19 and event['sdk_physics_step_is_derived'] is True
        assert event['sdk_step_source']=='derived_from_actual_policy_World_SDK_anchor_offset'
        assert event['independent_getter_clock_pair_available'] is False


def test_bearing_threshold_loss_is_not_accepted_contact_loss_and_can_reacquire():
    candidate, references, policy = event_fixture(); rows = candidate['samples']
    rows[1]['pad']['load_n'][0, 0] = .01
    rows[2]['pad']['load_n'][0, 0] = .1
    out = api().summarize_six_first(candidate, references, policy, np.full(12, .01))['per_env'][0]
    assert out['first_finger_pad_contact_loss'] is None
    assert out['first_bearing_pad_loss']['substep'] == 2
    assert out['pad_transitions']['left.thumb']['first_bearing_reacquired']['substep'] == 3


@pytest.mark.parametrize('bad', ['measurement_invalid','pad_missing','gap'])
def test_unknown_pad_measurement_cannot_become_contact_loss_or_bridge_a_gap(bad):
    candidate, references, policy = event_fixture(); rows = candidate['samples']
    rows[1]['pad']['accepted'][:] = False; rows[1]['pad']['load_n'][:] = 0.
    if bad == 'measurement_invalid': rows[1]['measurement_valid'] = False
    elif bad == 'pad_missing': rows[1]['pad'] = None
    else: del rows[0]
    result = api().summarize_six_first(candidate, references, policy, np.full(12, .01))
    assert result['per_env'][0]['first_finger_pad_contact_loss'] is None
    assert result['unavailable_counts']['candidate_pad_samples'] >= (0 if bad == 'gap' else 1)


def test_one_contact_reference_is_na_and_prefix_loss_is_not_student_causality():
    candidate, references, policy = event_fixture()
    for row in candidate['samples']: row['action_owner'] = 'teacher_prefix'
    policy[0]['owner'] = 'teacher_prefix'
    row = candidate['samples'][1]; row['pad']['load_n'][1, 0] = 0.; row['pad']['accepted'][1, 0] = False
    result = api().summarize_six_first(candidate, references[:1], policy, np.full(12, .01))
    out = result['per_env'][0]
    assert out['first_action_difference'] is None
    assert out['first_finger_pad_contact_loss']['owner_scope'] == 'prefix_event_not_Student_evidence'
    assert out['first_repeat_pad_contact_departure'] is None
    assert result['contact_repeat_minimum_reference_samples'] == 2


def test_qualified_repeat_pad_departure_needs_same_time_valid_bilateral_opposition():
    candidate,references,policy=event_fixture()
    candidate['samples'][1]['pad']['load_n'][0,0]=0.
    candidate['samples'][1]['pad']['accepted'][0,0]=False
    references[0]['samples'][1]['bilateral_opposed_contact']=False
    out=api().summarize_six_first(candidate,references,policy,np.full(12,.01))['per_env'][0]
    assert out['first_finger_pad_contact_loss'] is not None
    assert out['first_repeat_pad_contact_departure'] is None


def test_pad_loss_before_first_actual_bilateral_contact_is_labeled_without_inventing_support():
    candidate,references,policy=event_fixture()
    for row in candidate['samples']: row['bilateral_opposed_contact']=False
    candidate['samples'][1]['pad']['load_n'][0,0]=0.
    candidate['samples'][1]['pad']['accepted'][0,0]=False
    candidate['samples'][2].update(bilateral_opposed_contact=True,measurement_valid=False)
    candidate['samples'][3]['bilateral_opposed_contact']=True
    out=api().summarize_six_first(candidate,references,policy,np.full(12,.01))['per_env'][0]
    assert out['first_actual_bilateral_opposed_contact']['substep']==4
    loss=out['first_finger_pad_contact_loss']
    assert loss['substep']==2 and loss['aggregate_opposed_at_sample'] is False
    assert loss['had_observed_bilateral_contact_before_loss'] is False
    assert loss['was_original_bearing_before_loss'] is True


def test_success_terminal_is_not_failure_and_invalid_or_unanchored_slip_is_na():
    candidate, references, policy = event_fixture()
    candidate['samples'][1].update(slip_m=.01, prior_anchor_latched=False)
    candidate['samples'][2].update(slip_m=.01, slip_valid=False)
    out = api().summarize_six_first(candidate, references, policy, np.full(12, .01))['per_env'][0]
    assert out['first_failure'] is None and out['first_terminal'] is not None
    assert out['first_object_slip'] is None


def test_slip_without_any_valid_fixed_anchor_samples_is_explicitly_na_not_zero_slip():
    candidate,references,policy=event_fixture()
    for row in candidate['samples']: row['slip_valid']=False
    out=api().summarize_six_first(candidate,references,policy,np.full(12,.01))['per_env'][0]
    assert out['availability']['fixed_anchor_slip']['status']=='unavailable_no_valid_fixed_anchor_samples'
    assert out['availability']['fixed_anchor_slip']['valid_samples']==0
    candidate,references,policy=event_fixture()
    out=api().summarize_six_first(candidate,references,policy,np.full(12,.01))['per_env'][0]
    assert out['availability']['fixed_anchor_slip']['status']=='observed_no_original_threshold_crossing'
    assert out['availability']['fixed_anchor_slip']['valid_samples']==6


def test_target_cache_and_backend_are_separate_and_joint_tracking_never_uses_cache():
    candidate, references, policy = event_fixture()
    for run in [candidate, *references]:
        for row in run['samples']:
            row['target_stages']['command_cache'] = {'left.finger11': dict(value=0.,unit='rad')}
    candidate['samples'][0]['target_stages']['command_cache']['left.finger11']['value'] = 1.
    out = api().summarize_six_first(candidate, references, policy, np.full(12, .01))['per_env'][0]
    assert out['first_control_target_difference_by_stage']['command_cache']['substep'] == 1
    assert out['first_joint_tracking_difference'] is None
    assert out['first_significant_tracking_departure'] is None


def test_single_measured_teacher_target_supports_numeric_difference_but_not_repeat_range():
    candidate, references, policy = event_fixture()
    candidate['samples'][0]['target_stages']['requested_palm']['left.z']['value'] = 1e-15
    out=api().summarize_six_first(candidate,references[:1],policy,np.full(12,.01))['per_env'][0]
    event=out['first_control_target_difference_by_stage']['requested_palm']
    assert event['observed_reference_samples']==1 and event['repeat_range_eligible'] is False
    assert event['cause_claimed'] is False


def test_full_named_wrist_tracking_preserves_meters_separately_from_active_rad_calibration():
    candidate,references,policy=event_fixture()
    for run in [candidate,*references]:
        for row in run['samples']:
            row['named_tracking']={'values':np.zeros(3),'names':('left.wrist_x','left.finger11','left.finger22'),'units':('m','rad','rad')}
    candidate['samples'][0]['named_tracking']['values'][0]=1e-12
    candidate['samples'][0]['named_tracking']['values'][2]=.6
    result=api().summarize_six_first(candidate,references[:1],policy,np.full(12,.01))['per_env'][0]
    event=result['first_joint_tracking_difference']
    assert event['substep']==1 and event['joints'][0]['unit']=='m'
    assert event['joints'][0]['joint']=='wrist_x' and event['joints'][0]['observed_reference_samples']==1
    assert event['joints'][0]['original_control_role']=='driven_wrist'
    assert event['joints'][1]['original_control_role']=='passive_mimic_backend_diagnostic'
    assert event['joints'][1]['qualified_ACTIVE_trigger_applicable'] is False
    assert result['first_significant_tracking_departure'] is None


@pytest.mark.parametrize('bad', ['in_progress','missing_terminal','model','teacher_owner'])
def test_native_source_scope_gate_rejects_partial_or_wrong_actor_before_output(bad):
    report = dict(mode='source',status='completed',error=None,completed_count=4,ppo_updates=0,
        native_runtime_verified=True,inputs={'checkpoint':{'path':'/model','sha256':'a'*64}},
        source_rows=[dict(env_id=i,teacher_prefix_policy_steps_executed=0,teacher_replay_policy_steps_executed=0,
            terminal=dict(action_owner='student')) for i in range(4)])
    if bad == 'in_progress': report['status'] = 'not_completed'
    if bad == 'missing_terminal': report['source_rows'][0]['terminal'] = None
    if bad == 'model': report['inputs']['checkpoint']['sha256'] = 'b'*64
    if bad == 'teacher_owner': report['source_rows'][0]['terminal']['action_owner'] = 'teacher_prefix'
    with pytest.raises(ValueError):
        api().require_native_event_scope(report, {'path':'/model','sha256':'a'*64})


def test_native_event_scope_refuses_a_run_with_actual_ppo_updates():
    report=dict(mode='source',status='completed',error=None,completed_count=4,ppo_updates=1,
        native_runtime_verified=True,inputs={'checkpoint':{'path':'/model','sha256':'a'*64}},
        source_rows=[dict(env_id=i,teacher_prefix_policy_steps_executed=0,teacher_replay_policy_steps_executed=0,
            terminal=dict(action_owner='student')) for i in range(4)])
    with pytest.raises(ValueError,match='PPO|ppo|source'):
        api().require_native_event_scope(report,{'path':'/model','sha256':'a'*64})


def enriched_native_fixture(tmp_path):
    from test_stage6_physical_divergence import native_files
    from go2_pvcnn.control.dh116 import tabletop_vision_physical_divergence as physical
    files, dataset, context, calibration, arrays, path = native_files(tmp_path)
    shape = arrays['env_id'].shape
    fields = dict(effective_pad_loads_n=np.full((*shape,2,5),.1),
        effective_pad_valid=np.ones((*shape,2,5),bool), requested_palm_target_m=np.zeros((*shape,2,3)),
        projected_palm_target_m=np.zeros((*shape,2,3)), final_sdk_palm_target_m=np.zeros((*shape,2,3)),
        left_command_cache=np.ones((*shape,17)), right_command_cache=np.ones((*shape,17)))
    arrays.update(fields)
    for name in fields:
        for flag in ('available','finite','source_valid'): arrays[name+'__'+flag] = np.ones(shape,bool)
    def save():
        np.savez_compressed(path,**arrays)
        summary=json.loads(open(files['audit_summary']['path']).read());summary['files']=[physical.file_identity(path)]
        open(files['audit_summary']['path'],'w').write(json.dumps(summary));files['audit_summary']=physical.file_identity(files['audit_summary']['path'])
    save()
    return files,dataset,context,calibration,arrays,path,save


def test_enrichment_uses_bound_same_sample_sources_separate_cache_backend_and_named_mixed_units(tmp_path):
    files,data,context,calibration,arrays,path,save = enriched_native_fixture(tmp_path)
    from go2_pvcnn.control.dh116 import tabletop_vision_physical_divergence as physical
    run = physical.load_native_run(files,dataset_id=data,context_id=context,calibration=calibration)
    enriched = api().load_native_event_fields(run)
    first = enriched['samples'][0]
    assert first['pad']['accepted'].all() and first['measurement_valid'] is True
    assert first['raw_pointer']['line'] == 1
    assert first['diagnostic_pointer']['sha256'] == physical.file_identity(path)['sha256']
    backend = api().stage_values(first['target_stages']['left_physx_backend'])
    cache = api().stage_values(first['target_stages']['left_command_cache'])
    assert backend['wrist_x'] == dict(value=0.,unit='m')
    assert backend['finger11'] == dict(value=0.,unit='rad')
    assert cache['finger11'] == dict(value=1.,unit='rad')
    assert first['unavailable_contact_fields']['native_body_id'] is None


def test_pad_false_is_observed_absence_but_source_invalid_is_unavailable(tmp_path):
    files,data,context,calibration,arrays,path,save = enriched_native_fixture(tmp_path)
    arrays['effective_pad_valid'][0,0,0,0] = False; arrays['effective_pad_loads_n'][0,0,0,0] = 0.
    arrays['effective_pad_valid__source_valid'][1,0] = False; save()
    from go2_pvcnn.control.dh116 import tabletop_vision_physical_divergence as physical
    run=physical.load_native_run(files,dataset_id=data,context_id=context,calibration=calibration)
    enriched=api().load_native_event_fields(run)
    by_key={(r['env_id'],r['world_physics_step']):r for r in enriched['samples']}
    assert not by_key[0,1]['pad']['accepted'][0,0]
    assert by_key[0,2]['pad'] is None


def test_enrichment_rechecks_original_manifest_bytes_instead_of_trusting_loaded_dict(tmp_path):
    files,data,context,calibration,arrays,path,save = enriched_native_fixture(tmp_path)
    from go2_pvcnn.control.dh116 import tabletop_vision_physical_divergence as physical
    run=physical.load_native_run(files,dataset_id=data,context_id=context,calibration=calibration)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA|immutable'):
        api().load_native_event_fields(run)


def test_native_time_aligned_action_reference_is_exact_source_four_key_join_not_an_unchecked_live_label():
    from test_vision_sequence_dataset import fixture_data
    data=fixture_data((16,)); meta=data['sample_metadata'][0]
    report=dict(source_case_definitions=[dict(env_id=0,source_run_uuid=meta['run_uuid'],source_episode_uid=meta['episode_uid'])])
    policy=dict(env_id=0,policy_step=0,recorded_teacher_episode_uid=meta['episode_uid'],
        recorded_teacher_action=data['actions'][0].tolist())
    identity={'path':'/actual/data.pt','sha256':'d'*64}
    bound=api().bind_policy_teacher_actions([policy],data,report,identity)
    assert bound[0]['source_teacher_pointer']['row_index']==0
    policy['recorded_teacher_action']=[.5]*24
    with pytest.raises(ValueError,match='Teacher|reference|action'):
        api().bind_policy_teacher_actions([policy],data,report,identity)


def test_original_teacher_reference_pointer_uses_independently_keyed_actual_post_and_backend_tables():
    from test_stage6_tracking_calibration import evidence,calibrate
    from go2_pvcnn.control.dh116 import tabletop_vision_physical_divergence as physical
    data,context,_=evidence();calibration=calibrate(data,context)
    context['native_backend_targets'].reverse()
    original=physical.load_teacher_context(data,context,calibration,provenance={})
    result=api().attach_original_teacher_post_pointers(original,context,{'path':'/actual/context.pt','sha256':'c'*64})
    row=result['samples'][0];post=row['raw_pointer'];backend=row['diagnostic_pointer']
    assert post['table']=='post_physics_feedback' and backend['table']=='native_backend_targets'
    assert post['sample_key']==backend['sample_key']
    assert tuple(context[post['table']][post['entry_index']]['sample_key'])==tuple(post['sample_key'])
    assert tuple(context[backend['table']][backend['entry_index']]['sample_key'])==tuple(backend['sample_key'])
    assert post['entry_index']!=backend['entry_index']
    assert post['physics_substep_index']==0 and row['independent_getter_clock_pair_available'] is True
