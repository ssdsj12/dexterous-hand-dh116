"""Pure CPU actor-only sequence fitting; synthetic rows are not native proof."""
import copy
import importlib

import pytest
import torch

from test_vision_sequence_dataset import fixture_data, fixture_context, make_dataset


@pytest.fixture(autouse=True)
def isolated_rng_and_threads():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(2)
        try: yield
        finally: torch.set_num_threads(threads)


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_supervised_bc')


def source_and_dataset(*, length=35, sampling='uniform', dead=False):
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(116)
        actor = UnboundedVisionStudentActor()
    if dead:
        for layer in actor.point_net:
            if isinstance(layer, torch.nn.Linear):
                with torch.no_grad(): layer.weight.zero_(); layer.bias.fill_(-1.)
    data = fixture_data((length,))
    frames = data['observations'].reshape(length, 4, 961)
    frames[..., 768:896] = 1.
    points = torch.randn(length, 4, 768, generator=torch.Generator().manual_seed(17)) * .02
    frames[..., :768] = points
    return actor.state_dict(), make_dataset(data, sampling_mode=sampling)


def test_sequences_fit_single_tanh_with_fresh_adam_exact_exposure_and_immutable_inputs():
    state, ds = source_and_dataset()
    before = copy.deepcopy(state); obs = ds.data['observations'].clone()
    rng = torch.random.get_rng_state().clone(); sampler = ds.generator.get_state().clone()
    snapshots = []
    result = module().fit_actor_sequences(state, ds, profile='B0', steps=3,
        seed=5, snapshot_callback=snapshots.append, checkpoint_interval=2, probe_interval=2)
    assert result['steps'] == 3 and result['valid_exposures'] == 192
    assert result['source_immutable'] and result['physical_success_unverified']
    assert result['actor_initialization'] == 'source_no_reset'
    assert torch.equal(rng, torch.random.get_rng_state()) and torch.equal(sampler, ds.generator.get_state())
    assert all(torch.equal(state[k], before[k]) for k in state)
    assert torch.equal(obs, ds.data['observations'])
    assert result['loss_after'] < result['loss_before']
    assert all(value.device.type == 'cpu' for value in result['actor'].values())
    assert [row['step'] for row in snapshots] == [2, 3]
    assert snapshots[-1]['input']['observations'].shape == (4, 16, 3844)
    assert result['optimized_parameters'] == sorted(state)
    assert any(not torch.equal(state[k], result['actor'][k]) for k in state if k.startswith('point_net.'))


def test_sequence_training_is_repeatable_from_private_seed():
    state, ds = source_and_dataset()
    first = module().fit_actor_sequences(state, ds, profile='B0', steps=2, seed=8)
    second = module().fit_actor_sequences(state, ds, profile='B0', steps=2, seed=8)
    assert all(torch.equal(first['actor'][k], second['actor'][k]) for k in state)
    assert torch.equal(first['sampling_rng'], second['sampling_rng'])


def test_short_real_episode_exposures_are_reported_instead_of_counted_as_64():
    state, ds = source_and_dataset(length=7)
    result = module().fit_actor_sequences(state, ds, profile='B0', steps=2)
    assert result['valid_exposures'] == 56
    assert result['logs'][0]['valid_exposures'] == 28
    assert result['padding_exposures'] == 72


@pytest.mark.parametrize('profile,sampling', [('B0', 'event'), ('B1', 'uniform'), ('B2', 'uniform')])
def test_profiles_cannot_silently_change_sampler(profile, sampling):
    state, ds = source_and_dataset(sampling=sampling)
    with pytest.raises(ValueError, match='sampling'):
        module().fit_actor_sequences(state, ds, profile=profile, steps=1)


def test_b2_refuses_missing_saved_context_even_during_command_warmup():
    state, ds = source_and_dataset(sampling='event')
    with pytest.raises(ValueError, match='context|contract'):
        module().fit_actor_sequences(state, ds, profile='B2', steps=1)


def test_command_schedule_preserves_specified_warmup_and_b3_is_explicit():
    m = module()
    assert [m.sequence_command_weight(i) for i in (0, 249, 250, 375, 500, 501)] == [0., 0., 0., .05, .1, .1]
    state, ds = source_and_dataset(sampling='event')
    result = m.fit_actor_sequences(state, ds, profile='B1', steps=1)
    assert result['logs'][0]['command_weight'] == 0.
    assert result['logs'][0]['delta_weight'] == 0.


def test_fixed_event_probes_report_layers_gradients_and_sensor_interventions():
    state, ds = source_and_dataset()
    result = module().fit_actor_sequences(state, ds, profile='B0', steps=1)
    probe = result['probes'][-1]['events']['START']
    assert probe['valid_hand_point_count'] > 0
    assert set(probe['layers']) == {'0', '1', '2', '3', '4', '5'}
    assert all('abs_max' in record and 'nonzero_fraction' in record for record in probe['layers'].values())
    assert any(row['l2'] > 0 for row in probe['pointnet_parameter_gradients'].values())
    assert probe['zero_coordinates_max_action_delta'] > 0
    assert probe['point_permutation_max_action_delta'] < 1e-7
    assert probe['small_geometry_max_action_delta'] > 0


def test_repeated_dead_pointnet_stops_candidate_without_reinitializing_source():
    state, ds = source_and_dataset(dead=True)
    snapshots = []
    result = module().fit_actor_sequences(state, ds, profile='B0', steps=5,
        probe_interval=1, dead_patience=2, snapshot_callback=snapshots.append)
    assert result['status'] == 'stopped_dead_pointnet'
    assert result['steps'] < 5
    assert all(torch.equal(result['actor'][k], state[k]) for k in state if k.startswith('point_net.'))
    assert snapshots[-1]['status'] == 'stopped_dead_pointnet'
    assert snapshots[-1]['input']['row_indices'].shape == (4, 16)


def test_bounded_legacy_actor_is_explicitly_unsupported():
    state, ds = source_and_dataset()
    with pytest.raises(ValueError, match='unbounded|bounded'):
        module().fit_actor_sequences(state, ds, profile='B0', steps=1, source_actor_kind='bounded')


def command_dataset():
    from test_vision_command_shadow import fixture, shadow
    sh = shadow(); task, world = fixture()
    state, ds0 = source_and_dataset(length=16, sampling='event')
    data = ds0.data; data['actions'].zero_()
    for row in data['sample_metadata']: row['previous_action'] = [0.] * 24
    data['observations'].reshape(16, 4, 961)[:, -1, 934:958] = 0
    context = fixture_context(data)
    current = sh.command_state(task, world)
    measured = sh.measured_context(task, world, torch.ones(3, dtype=torch.bool))
    # One deterministic synthetic episode; each record owns actual Tensor
    # controller feedback, without invoking SDK or advancing native physics.
    context['controller_start'] = {k: v[:1].expand(16, *v.shape[1:]).clone() for k, v in current.items()}
    context['physics_context'] = {k: v[:1, None].expand(16, 8, *v.shape[1:]).clone() for k, v in measured.items()}
    contract = sh.controller_contract(task, world)
    context['controller_source_sha256'] = contract['controller_source_sha256']
    batch_context = {k: v[None] for k, v in context['physics_context'].items()}
    teacher = sh.shadow_sequence({k: v[:1] for k, v in current.items()}, data['actions'][None], batch_context, contract)
    context['teacher_targets'] = {k: v[0].detach() for k, v in teacher.items()}
    # A zero issued action keeps the synthetic recorded starts constant only
    # when safety projections do not move them; capture true starts per tick.
    for name, key in (('finger', 'joint_command_rad'), ('palm', 'palm_command_offset_m'), ('wrist_angle', 'wrist_command_angle_rad')):
        context['controller_start'][key][1:] = teacher[name][0, :-1, -1]
    return state, make_dataset(data, context), contract


def test_b2_replays_teacher_against_saved_targets_before_enabling_commands():
    state, ds, contract = command_dataset()
    result = module().fit_actor_sequences(state, ds, profile='B2', steps=1, controller_contract=contract)
    assert result['teacher_shadow_parity']['passed']
    assert result['teacher_shadow_parity']['max_absolute_error'] == 0.
    assert result['logs'][0]['command_weight'] == 0.
    ds.context['teacher_targets']['finger'][0, 0, 0] += .001
    with pytest.raises(ValueError, match='parity|target'):
        module().fit_actor_sequences(state, ds, profile='B2', steps=1, controller_contract=contract)


def test_command_schedule_receives_one_based_completed_update_count(monkeypatch):
    state, ds, contract = command_dataset()
    seen = []; original = module().sequence_command_weight
    def observe(update):
        seen.append(update)
        return original(update)
    monkeypatch.setattr(module(), 'sequence_command_weight', observe)
    module().fit_actor_sequences(state, ds, profile='B2', steps=2, controller_contract=contract)
    assert seen == [1, 2]


@pytest.mark.parametrize('profile,delta', [('B2', 0.), ('B3', .02)])
def test_active_command_branch_runs_real_shadow_and_backpropagates_without_mutating_feedback(monkeypatch, profile, delta):
    state, ds, contract = command_dataset()
    feedback = ds.context['physics_context']['measured_q_rad'].clone()
    # Exercise the active loss branch with real shared controller kernels;
    # the production warmup schedule is independently checked above.
    monkeypatch.setattr(module(), 'sequence_command_weight', lambda update: .1)
    result = module().fit_actor_sequences(state, ds, profile=profile, steps=1,
        controller_contract=contract)
    assert result['logs'][0]['command'] > 0.
    assert result['logs'][0]['command_weight'] == .1
    assert result['logs'][0]['delta_weight'] == delta
    assert result['logs'][0]['gradient_norm_before_clip'] > 0.
    assert result['teacher_shadow_parity']['physics_samples'] >= 5 * 16 * 8
    assert torch.equal(ds.context['physics_context']['measured_q_rad'], feedback)
    assert all(torch.isfinite(value).all() for value in result['actor'].values())
