"""Sequence identity/boundary/sampler checks use synthetic CPU records only."""
import copy
import hashlib
import importlib
import json

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_vision_supervision_dataset as supervision
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT


def fixture_data(lengths=(35,), groups=None):
    groups = groups or ['source-a'] * len(lengths)
    rows, episodes, action_rows, observations = [], {}, [], []
    for ep, length in enumerate(lengths):
        episodes[str(ep)] = dict(env_id=ep, first_episode=True, success=True, recovered=False,
            native_physics=True, grasp_style='pinch_thumb_index', source_trajectory_id=groups[ep],
            source_episode_id=ep, run_uuid='run', episode_uid=f'episode-{ep}', proof_path='cpu-fixture',
            terminal=dict(success=True, recovered=False, failure=0, control_steps=length))
        previous = torch.zeros(24)
        for tick in range(length):
            action = torch.full((24,), tick / 1000.)
            observation = torch.full((3844,), (ep * 100 + tick) / 10000.)
            observation[-961 + 934:-961 + 958] = previous
            rows.append(dict(run_uuid='run', env_id=ep, episode_uid=f'episode-{ep}', episode_id=ep,
                tick=tick, frame_id=tick + 1, simulation_time_s=tick / 60,
                policy_timestamp_s=tick / 60, physics_step=tick * 8, first_physics_step=tick * 8,
                terminal=tick == length - 1, previous_action=previous.tolist(),
                grasp_style='pinch_thumb_index', source_trajectory_id=groups[ep], source_episode_id=ep))
            observations.append(observation); action_rows.append(action); previous = action
    contract = dict(actor_observation_size=3844, critic_observation_size=371,
        actor_uses_privileged_state=False, action_contract=copy.deepcopy(ACTION_CONTRACT),
        camera_config_sha256='1' * 64, primary_reset_sha256='2' * 64, secondary_reset_sha256='3' * 64)
    actions = torch.stack(action_rows)
    return supervision.build_dataset(torch.stack(observations), actions, actions, rows, episodes,
        visual_contract=contract, teacher_source_sha256='a' * 64)


def fixture_context(data):
    n = len(data['observations']); rows = data['sample_metadata']
    samples = {key: [row[key] for row in rows] for key in
        ('run_uuid', 'env_id', 'episode_uid', 'tick', 'policy_timestamp_s', 'first_physics_step', 'terminal')}
    samples.update(valid=torch.ones(n, dtype=torch.bool), event_id=['recorded'] * n,
        actions=data['actions'].clone(), previous_action=torch.tensor([r['previous_action'] for r in rows]))
    contract_sha = hashlib.sha256(json.dumps(ACTION_CONTRACT, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(version='dh116-sequence-context-v1', dataset_sha256='d' * 64,
        controller_source_sha256='c' * 64, action_contract_sha256=contract_sha,
        samples=samples,
        controller_start=dict(joint_command_rad=torch.zeros(n, 12),
            palm_command_offset_m=torch.zeros(n, 2, 3), wrist_command_angle_rad=torch.zeros(n, 2, 3)),
        physics_context=dict(dt_s=torch.full((n, 8), 1 / 480, dtype=torch.float64),
            measured_q_rad=torch.zeros(n, 8, 12), measured_qd_rad_s=torch.zeros(n, 8, 12),
            measured_root_pose=torch.zeros(n, 8, 2, 7), clearance_m=torch.zeros(n, 8, 2),
            task_offset_m=torch.zeros(n, 8), live=torch.ones(n, 8, dtype=torch.bool),
            allowed=torch.ones(n, 8, dtype=torch.bool), joint_lower=torch.full((n, 8, 12), -1.),
            joint_upper=torch.ones(n, 8, 12), active_controller_flags=torch.zeros(n, 8, dtype=torch.long),
            source_valid=torch.ones(n, 8, dtype=torch.bool)),
        teacher_targets=dict(finger=torch.zeros(n, 8, 12), palm=torch.zeros(n, 8, 2, 3),
                             wrist_angle=torch.zeros(n, 8, 2, 3)),
        events=dict(opposed_contact=torch.zeros(n, dtype=torch.bool),
            measured_palm_velocity_m_s=torch.zeros(n, 2, 3), native_hold_qualified=torch.zeros(n, dtype=torch.bool)))


def make_dataset(data, context=None, **kwargs):
    module = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_sequence_dataset')
    return module.SequenceDataset(data, context, dataset_sha256='d' * 64, **kwargs)


def test_explicit_row_and_episode_source_groups_cannot_disagree():
    data = fixture_data()
    data['episodes']['0']['source_group_id'] = 'actual-parent-family'
    for row in data['sample_metadata']:
        row['source_group_id'] = 'incorrect-derived-family'
    with pytest.raises(ValueError, match='source.group.*disagree|disagree.*source.group'):
        make_dataset(data)


def test_one_physical_episode_cannot_be_partitioned_into_multiple_source_groups():
    data = fixture_data(lengths=(40,))
    for i, row in enumerate(data['sample_metadata']):
        row['source_group_id'] = 'same-episode-train' if i < 20 else 'same-episode-heldout'
    with pytest.raises(ValueError, match='episode.*source.group|source.group.*episode'):
        make_dataset(data, heldout_source_trajectory_ids=('same-episode-heldout',))


@pytest.mark.parametrize('explicit_half', ['first', 'last'])
def test_partial_explicit_lineage_cannot_split_a_physical_episode(explicit_half):
    data = fixture_data(lengths=(40,))
    for i, row in enumerate(data['sample_metadata']):
        if (i < 20) == (explicit_half == 'first'):
            row['source_group_id'] = 'different-parent-family'
    with pytest.raises(ValueError, match='episode.*source.group|source.group.*episode'):
        make_dataset(data, heldout_source_trajectory_ids=('different-parent-family',))


def test_matching_explicit_parent_lineage_overrides_local_trajectory_name():
    data = fixture_data()
    data['episodes']['0']['source_group_id'] = 'actual-parent-family'
    for row in data['sample_metadata']:
        row['source_group_id'] = 'actual-parent-family'
    indexed = make_dataset(data)
    assert indexed.report['fit_only'] is True
    assert {window['source_group'] for window in indexed.windows} == {'actual-parent-family'}


def reorder_context(context, order):
    def take(value):
        if isinstance(value, dict): return {k: take(v) for k, v in value.items()}
        if isinstance(value, torch.Tensor): return value[order]
        if isinstance(value, list): return [value[i] for i in order]
        return value
    result = copy.deepcopy(context)
    for key in ('samples', 'controller_start', 'physics_context', 'teacher_targets', 'events'):
        result[key] = take(result[key])
    return result


def test_join_uses_identity_not_row_counts_and_is_pure():
    data = fixture_data(); context = fixture_context(data)
    reordered = reorder_context(context, list(reversed(range(35))))
    joined = supervision.join_sequence_context(data, reordered, dataset_sha256='d' * 64)
    assert joined['context_indices'].tolist() == list(reversed(range(35)))
    assert bool(joined['row_valid'].all())
    assert data['sample_metadata'][0]['tick'] == 0 and reordered['samples']['tick'][0] == 34


@pytest.mark.parametrize('field', ['dataset_sha256', 'controller_source_sha256', 'action_contract_sha256'])
def test_sidecar_hashes_must_match_exact_bound_sources(field):
    data = fixture_data(); context = fixture_context(data)
    kwargs = dict(dataset_sha256='d' * 64, controller_source_sha256='c' * 64,
                  action_contract_sha256=context['action_contract_sha256'])
    context[field] = 'e' * 64
    with pytest.raises(ValueError, match='sha256'):
        supervision.join_sequence_context(data, context, **kwargs)


def test_duplicate_and_missing_identity_never_join_by_position():
    data = fixture_data(); context = fixture_context(data)
    context['samples']['tick'][1] = 0
    with pytest.raises(ValueError, match='duplicate'):
        supervision.join_sequence_context(data, context, dataset_sha256='d' * 64)
    context = reorder_context(fixture_context(data), list(range(1, 35)))
    joined = supervision.join_sequence_context(data, context, dataset_sha256='d' * 64)
    assert not joined['row_valid'][0] and joined['context_indices'][0] == -1
    assert bool(joined['row_valid'][1:].all())
    for row in data['sample_metadata']: row.pop('run_uuid')
    with pytest.raises(ValueError, match='identity|run_uuid'):
        supervision.join_sequence_context(data, context, dataset_sha256='d' * 64)


@pytest.mark.parametrize('field', ['actions', 'previous_action', 'policy_timestamp_s', 'first_physics_step', 'terminal'])
def test_context_mismatch_invalidates_record_instead_of_inventing_context(field):
    data = fixture_data(); context = fixture_context(data)
    if field in ('actions', 'previous_action'): context['samples'][field][5, 0] += .1
    elif field == 'terminal': context['samples'][field][5] = True
    else: context['samples'][field][5] += 1
    joined = supervision.join_sequence_context(data, context, dataset_sha256='d' * 64)
    assert not joined['row_valid'][5]
    assert bool(joined['row_valid'][[0, 4, 6, 34]].all())


@pytest.mark.parametrize('bad', ['missing', 'substeps', 'nonfinite', 'dt', 'source_valid'])
def test_physics_context_requires_eight_actual_finite_substeps(bad):
    data = fixture_data(); context = fixture_context(data)
    if bad == 'missing': context['physics_context'].pop('measured_q_rad')
    elif bad == 'substeps': context['physics_context']['measured_q_rad'] = torch.zeros(35, 7, 12)
    elif bad == 'nonfinite': context['physics_context']['measured_q_rad'][0, 0, 0] = float('nan')
    elif bad == 'dt': context['physics_context']['dt_s'][0, 0] = 0
    elif bad == 'source_valid': context['physics_context']['source_valid'] = torch.ones(35, 8)
    with pytest.raises(ValueError): supervision.validate_sequence_context(context)


def test_true_history_full_windows_cover_tail_and_batch_has_64_exposures():
    data = fixture_data(); ds = make_dataset(data)
    assert all(len(window['row_indices']) == 16 for window in ds.windows)
    assert set(i for window in ds.windows for i in window['row_indices']) == set(range(35))
    batch = ds.sample_batch(generator=torch.Generator().manual_seed(116))
    assert batch['observations'].shape == (4, 16, 3844)
    assert batch['actions'].shape == (4, 16, 24)
    assert batch['valid'].sum().item() == 64
    assert torch.equal(batch['observations'], data['observations'][batch['row_indices']])
    assert batch['controller_start'] is None and ds.report['fit_only']
    assert all(not w['events'] - {'START'} for w in ds.windows)


def test_short_episode_padding_never_crosses_terminal_or_other_episode():
    data = fixture_data((7, 21)); context = fixture_context(data)
    ds = make_dataset(data, context)
    batch = ds.batch([0])
    assert batch['valid'].sum().item() == 7
    assert batch['row_indices'][0, 7:].tolist() == [-1] * 9
    assert not bool(batch['physics_valid'][0, 7:].any())
    assert batch['controller_start']['joint_command_rad'].shape == (1, 16, 12)
    assert batch['physics_context']['dt_s'].shape == (1, 16, 8)
    for window in ds.windows:
        assert len({data['sample_metadata'][i]['episode_uid'] for i in window['row_indices']}) == 1


def test_missing_context_is_gap_and_no_window_spans_it():
    data = fixture_data((45,)); context = fixture_context(data)
    context = reorder_context(context, [i for i in range(45) if i != 20])
    ds = make_dataset(data, context)
    for window in ds.windows:
        ticks = [data['sample_metadata'][i]['tick'] for i in window['row_indices']]
        assert not (min(ticks) < 20 < max(ticks))
        assert 20 not in ticks
    assert ds.report['invalid_context_rows'] == 1


def test_first_terminal_reset_source_and_clock_discontinuities_are_boundaries():
    data = fixture_data((55,))
    data['sample_metadata'][18]['terminal'] = True
    data['sample_metadata'][30]['reset'] = True
    for row in data['sample_metadata'][40:]: row['source_trajectory_id'] = 'source-b'
    data['source_trajectory_ids'] = [r['source_trajectory_id'] for r in data['sample_metadata']]
    ds = make_dataset(data)
    # A terminal consumes the episode. Trailing pre-reset records cannot become new episodes.
    assert max(i for w in ds.windows for i in w['row_indices']) == 18
    data = fixture_data((55,))
    data['sample_metadata'][20]['reset'] = True
    data['sample_metadata'][40]['policy_timestamp_s'] += .1
    for row in data['sample_metadata'][30:]: row['source_trajectory_id'] = 'source-b'
    ds = make_dataset(data)
    for window in ds.windows:
        indices = window['row_indices']
        assert not any(min(indices) < boundary <= max(indices) for boundary in (20, 30, 40, 41))


def test_event_predicates_use_measured_opposed_velocity_and_native_hold():
    data = fixture_data((70,)); context = fixture_context(data)
    events = context['events']; events['opposed_contact'][15:] = True
    events['measured_palm_velocity_m_s'][30:33, :, 2] = .0006
    events['native_hold_qualified'][60:] = True
    ds = make_dataset(data, context)
    labels = ds.event_labels
    assert labels['START'].nonzero().flatten().tolist() == [0, 1, 2, 3]
    assert labels['CONTACT'].nonzero().flatten().tolist() == list(range(9, 28))
    assert labels['LIFT_ONSET'].nonzero().flatten().tolist() == list(range(18, 55))
    assert labels['HOLD'].nonzero().flatten().tolist() == list(range(60, 70))
    assert 'RECOVERY' not in labels
    context['events']['opposed_contact'][:] = False
    ds = make_dataset(data, context)
    assert not bool(ds.event_labels['CONTACT'].any()) and not bool(ds.event_labels['LIFT_ONSET'].any())


def test_sampler_is_source_then_episode_then_window_uniform_and_private():
    data = fixture_data((20, 30, 50), ['source-a', 'source-a', 'source-b'])
    ds = make_dataset(data, sampling_mode='uniform')
    probs = ds.sampling_probabilities
    for source, mass in [('source-a', .5), ('source-b', .5)]:
        assert sum(float(p) for p, w in zip(probs, ds.windows) if w['source_group'] == source) == pytest.approx(mass)
    for ep, mass in [('episode-0', .25), ('episode-1', .25), ('episode-2', .5)]:
        assert sum(float(p) for p, w in zip(probs, ds.windows) if w['episode_uid'] == ep) == pytest.approx(mass)
    state = torch.random.get_rng_state().clone()
    ds.sample_batch(generator=torch.Generator().manual_seed(3))
    ds.sample_batch()
    assert torch.equal(state, torch.random.get_rng_state())


def test_event_mix_renormalizes_missing_events_and_every_window_stays_positive():
    data = fixture_data((70,)); context = fixture_context(data)
    ds = make_dataset(data, context, sampling_mode='event')
    uniform = make_dataset(data, context, sampling_mode='uniform')
    start = [i for i, window in enumerate(ds.windows) if 'START' in window['events']]
    expected = .4 * uniform.sampling_probabilities
    expected[start] += .6 / len(start)
    assert torch.allclose(ds.sampling_probabilities, expected)
    assert bool((ds.sampling_probabilities > 0).all())


def test_source_replay_copies_bound_and_one_source_cannot_fake_heldout():
    data = fixture_data((20, 20, 20), ['source-a', 'source-a', 'source-b'])
    train = make_dataset(data, heldout_source_trajectory_ids=['source-b'])
    heldout = make_dataset(data, heldout_source_trajectory_ids=['source-b'], subset='heldout')
    assert {w['source_group'] for w in train.windows} == {'source-a'}
    assert {w['source_group'] for w in heldout.windows} == {'source-b'}
    assert not train.report['fit_only']
    with pytest.raises(ValueError): make_dataset(fixture_data(), heldout_source_trajectory_ids=['source-a'])


def test_recovery_requires_native_qualified_versioned_corrective_metadata():
    data = fixture_data(); context = fixture_context(data)
    context['events']['recovery_qualified'] = torch.ones(35, dtype=torch.bool)
    ds = make_dataset(data, context)
    assert 'RECOVERY' not in ds.event_labels
    context['recovery_native_verified'] = True
    context['recovery_schema_version'] = 'dh116-corrective-segment-v1'
    context['corrective_segments'] = [dict(parent_student_episode_uid='parent', branch_policy_step=3,
        actual_branch_state_path='native/state.pt', expert_version='expert-v1',
        issued_actions_path='native/actions.pt', native_result_path='native/result.json',
        native_success=True, start_identity=('run', 0, 'episode-0', 0), end_identity=('run', 0, 'episode-0', 34))]
    ds = make_dataset(data, context)
    assert bool(ds.event_labels['RECOVERY'].all())


def test_partial_terminal_substeps_preserve_true_prefix_mask():
    data = fixture_data((18,)); context = fixture_context(data)
    context['physics_context']['live'][-1, 3:] = False
    context['physics_context']['source_valid'][-1, 3:] = False
    ds = make_dataset(data, context)
    batch = ds.batch([len(ds) - 1])
    assert bool(batch['valid'].all())
    assert batch['row_indices'][0, -1].item() == 17
    assert batch['physics_valid'][0, -1].tolist() == [True] * 3 + [False] * 5
    context['physics_context']['source_valid'][-2, 3] = False
    ds = make_dataset(data, context)
    assert all(16 not in w['row_indices'] for w in ds.windows)


def test_long_episode_gap_fragments_do_not_silently_reduce_64_exposure_budget():
    data = fixture_data((40,)); context = fixture_context(data)
    context['samples']['valid'][5] = False
    ds = make_dataset(data, context)
    assert all(w['valid_steps'] == 16 for w in ds.windows)
    assert all(5 not in w['row_indices'] for w in ds.windows)
    assert ds.sample_batch()['valid'].sum().item() == 64
    assert ds.report['discarded_short_fragment_count'] == 1


def test_explicit_replay_lineage_groups_derived_source_ids_together():
    data = fixture_data((20, 20, 20), ['derived-a', 'derived-b', 'independent'])
    data['episodes']['0']['source_group_id'] = 'shared-origin'
    data['episodes']['1']['source_group_id'] = 'shared-origin'
    train = make_dataset(data, heldout_source_trajectory_ids=['independent'])
    assert train.report['source_trajectory_count'] == 2
    assert {w['source_group'] for w in train.windows} == {'shared-origin'}


def test_dataset_and_context_clocks_disagreeing_with_original_metadata_reject_rows():
    data = fixture_data((40,)); context = fixture_context(data)
    data['sample_metadata'][8]['policy_timestamp_s'] += .1
    context['samples']['policy_timestamp_s'][8] += .1
    # Internal metadata must agree with the original saved sensor clock too.
    joined = supervision.join_sequence_context(data, context, dataset_sha256='d' * 64)
    assert not joined['row_valid'][8]
    data = fixture_data((40,)); context = fixture_context(data)
    data['sample_metadata'][8]['first_physics_step'] += 8
    context['samples']['first_physics_step'][8] += 8
    joined = supervision.join_sequence_context(data, context, dataset_sha256='d' * 64)
    assert not joined['row_valid'][8]


def test_first_terminal_evidence_cannot_be_shifted_in_both_metadata_and_sidecar():
    data = fixture_data((40,)); context = fixture_context(data)
    data['sample_metadata'][-1]['terminal'] = False
    context['samples']['terminal'][-1] = False
    joined = supervision.join_sequence_context(data, context, dataset_sha256='d' * 64)
    assert not joined['row_valid'][-1]


def test_action_identity_comparison_does_not_round_away_a_real_mismatch():
    data = fixture_data(); context = fixture_context(data)
    context['samples']['actions'] = context['samples']['actions'].double()
    context['samples']['actions'][5, 0] += 1e-12
    joined = supervision.join_sequence_context(data, context, dataset_sha256='d' * 64)
    assert not joined['row_valid'][5]


def test_trusted_native_lift_event_precedes_fallback_and_needs_no_reinvented_contact():
    data = fixture_data((60,)); context = fixture_context(data)
    context['events'].pop('opposed_contact')
    context['events']['lift_onset'] = torch.zeros(60, dtype=torch.bool)
    context['events']['lift_onset'][25] = True
    context['lift_onset_source'] = 'native_executed_event'
    ds = make_dataset(data, context)
    assert ds.event_labels['LIFT_ONSET'].nonzero().flatten().tolist() == list(range(13, 50))
    context['lift_onset_source'] = 'unknown'
    ds = make_dataset(data, context)
    assert not bool(ds.event_labels['LIFT_ONSET'].any())


def test_live_reactivation_inside_policy_interval_is_an_invalid_reset_boundary():
    data = fixture_data((40,)); context = fixture_context(data)
    context['physics_context']['live'][10, 3] = False
    joined = supervision.join_sequence_context(data, context, dataset_sha256='d' * 64)
    assert not joined['row_valid'][10]


def test_report_exposes_uniform_fallback_when_no_recorded_events_remain():
    data = fixture_data((40,)); context = fixture_context(data)
    context['samples']['valid'][:5] = False
    ds = make_dataset(data, context)
    assert ds.report['available_event_weights'] == {}
    assert ds.report['uniform_fraction'] == 1.
    assert ds.report['event_fraction'] == 0.
