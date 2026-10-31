"""CPU-only surface BC tests; no simulator, collector or CLI is started."""
import copy
import importlib
import importlib.util
import math

import pytest
import torch
from torch import nn


@pytest.fixture
def bc():
    name = 'go2_pvcnn.control.dh116.tabletop_surface_imitation'
    return importlib.import_module(name)


def test_surface_imitation_module_is_available():
    name = 'go2_pvcnn.control.dh116.tabletop_surface_imitation'
    assert importlib.util.find_spec(name) is not None, 'surface imitation module is missing'


def contract():
    return dict(observation_size=351, action_size=24, policy_command_size=24,
                observation_version='dh116-surface-sharing-lift-objective-v2',
                num_envs=2, rollout_steps=24, minibatches=2,
                physics_hz=480, control_hz=60, active_effort_limit_nm=.2,
                initialization=dict(pose='static_pregrasp'),
                surface_force_sharing=dict(success_hold_s=5., stage_flow=None),
                success_slip_m=.0005)


def dataset(n=48):
    generator = torch.Generator().manual_seed(7)
    obs = torch.randn(n, 351, generator=generator) * .2
    actions = torch.tanh(obs[:, :24] * .6 + .15)
    return dict(version='dh116-surface-demonstrations-v1', contract=contract(),
                observations=obs, actions=actions,
                episode_ids=torch.zeros(n, dtype=torch.long),
                qualification=dict(qualified=True, successful_replays=1,
                                   proof_path='native/surface_replay.json',
                                   native_physics_path='native/surface_replay_physics.pt'))


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.fixture
def algorithm():
    from tensordict import TensorDict
    from go2_pvcnn.control.dh116.tabletop_tensor_training import make_algorithm
    obs = TensorDict({'policy': torch.zeros(2, 351)}, batch_size=[2], device='cpu')
    algo = make_algorithm(obs, steps=2, action_size=24)
    # Real nonempty Adam history, including critic and exploration parameters.
    sum(p.square().sum() for p in algo.policy.parameters()).backward()
    algo.optimizer.step()
    return algo


def test_actor_matches_real_surface_rsl_actor(bc, algorithm):
    actor = bc.make_actor()
    assert isinstance(actor, nn.Sequential)
    assert [type(layer) for layer in actor] == [nn.Linear, nn.Tanh] * 3
    assert [(layer.in_features, layer.out_features) for layer in actor
            if isinstance(layer, nn.Linear)] == [(351, 128), (128, 128), (128, 24)]
    algorithm.policy.actor.load_state_dict(actor.state_dict(), strict=True)
    obs = dataset()['observations']
    assert_same(actor(obs), algorithm.policy.actor(obs))
    assert actor(obs).abs().max() <= 1


def test_valid_dataset_accepts_fresh_success_and_batch_only_changes(bc):
    data = dataset()
    current = dict(contract(), num_envs=8, rollout_steps=64, minibatches=4)
    before = copy.deepcopy(data)
    bc.validate_dataset(data, current)
    bc.validate_dataset(data)
    assert_same(data, before)
    # Current surface contracts can declare the action dimension this way.
    del data['contract']['action_size']
    bc.validate_dataset(data)


@pytest.mark.parametrize('field', ['version', 'contract', 'observations', 'actions',
                                  'episode_ids', 'qualification'])
def test_missing_required_dataset_fields_are_rejected(bc, field):
    data = dataset()
    del data[field]
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


@pytest.mark.parametrize('bad', [None, [], 'demonstrations.pt'])
def test_dataset_must_be_a_dictionary(bc, bad):
    with pytest.raises(ValueError):
        bc.validate_dataset(bad)


@pytest.mark.parametrize('field,value', [
    ('version', 'dh116-bc-v1'), ('contract', None),
    ('observations', torch.zeros(2, 156)), ('actions', torch.zeros(48, 20)),
    ('observations', torch.zeros(0, 351)), ('actions', torch.zeros(47, 24)),
    ('observations', torch.zeros(48, 351, dtype=torch.long)),
    ('actions', torch.zeros(48, 24, dtype=torch.long)),
    ('episode_ids', torch.zeros(48)),
    ('episode_ids', torch.zeros(48, 1, dtype=torch.long)),
    ('episode_ids', torch.zeros(47, dtype=torch.long)),
    ('episode_ids', torch.full((48,), -1, dtype=torch.long)),
])
def test_invalid_samples_are_rejected(bc, field, value):
    data = dataset()
    data[field] = value
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


@pytest.mark.parametrize('field,value', [
    ('observations', float('nan')), ('observations', float('inf')),
    ('actions', float('nan')), ('actions', float('-inf')),
    ('actions', 1.0001), ('actions', -1.0001),
])
def test_nonfinite_and_out_of_range_samples_are_rejected(bc, field, value):
    data = dataset()
    data[field][0, 0] = value
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


@pytest.mark.parametrize('field,value', [
    ('qualified', False), ('qualified', 1), ('successful_replays', 0),
    ('successful_replays', True), ('successful_replays', 1.5),
    ('proof_path', ''), ('proof_path', '  '), ('proof_path', None),
])
def test_unqualified_demonstrations_are_rejected(bc, field, value):
    data = dataset()
    data['qualification'][field] = value
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


@pytest.mark.parametrize('field,value', [
    ('observation_size', 156), ('action_size', 20), ('policy_command_size', 26),
    ('observation_version', 'dh116-force-progress-milestones-v1'),
])
def test_contract_must_describe_current_surface_observations_and_actions(bc, field, value):
    data = dataset()
    data['contract'][field] = value
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


@pytest.mark.parametrize('field,value', [
    ('physics_hz', 240), ('active_effort_limit_nm', .3),
    ('initialization', dict(pose='other')),
    ('surface_force_sharing', dict(success_hold_s=1., stage_flow=None)),
    ('success_slip_m', .001), ('learning_rate', .001),
    ('unknown_physical_field', 'different'),
])
def test_only_explicit_top_level_batch_fields_are_ignored(bc, field, value):
    current = contract()
    current[field] = value
    with pytest.raises(ValueError, match='contract'):
        bc.validate_dataset(dataset(), current)


def test_nested_batch_named_physical_fields_are_not_stripped(bc):
    data = dataset()
    data['contract']['initialization']['num_envs'] = 2
    current = copy.deepcopy(data['contract'])
    current['initialization']['num_envs'] = 3
    with pytest.raises(ValueError, match='contract'):
        bc.validate_dataset(data, current)


def surface_height_contract(n):
    value = contract()
    value['num_envs'] = n
    value['success_bottom_m'] = [.39] * n
    value['surface_force_sharing']['lift_reference_bottom_m'] = [.36] * n
    return value


def height_values(value, path):
    if path == 'success_bottom_m':
        return value, path
    return value['surface_force_sharing'], 'lift_reference_bottom_m'


@pytest.mark.parametrize('source_envs,target_envs', [(2, 64), (2, 16), (64, 2)])
def test_uniform_height_arrays_allow_only_env_count_change(bc, source_envs, target_envs):
    data = dataset()
    data['contract'] = surface_height_contract(source_envs)
    current = surface_height_contract(target_envs)
    before = copy.deepcopy(data)
    current_before = copy.deepcopy(current)
    bc.validate_dataset(data, current)
    assert_same(data, before)
    assert_same(current, current_before)


@pytest.mark.parametrize('path', ['success_bottom_m', 'surface_force_sharing.lift_reference_bottom_m'])
@pytest.mark.parametrize('side', ['source', 'target'])
def test_uniform_height_arrays_reject_any_real_height_change(bc, path, side):
    data = dataset()
    data['contract'] = surface_height_contract(2)
    current = surface_height_contract(64)
    container, key = height_values(data['contract'] if side == 'source' else current, path)
    container[key] = [math.nextafter(container[key][0], math.inf)] * len(container[key])
    with pytest.raises(ValueError, match='contract'):
        bc.validate_dataset(data, current)


@pytest.mark.parametrize('path', ['success_bottom_m', 'surface_force_sharing.lift_reference_bottom_m'])
@pytest.mark.parametrize('side', ['source', 'target', 'both'])
def test_nonuniform_height_arrays_keep_full_env_layout(bc, path, side):
    data = dataset()
    data['contract'] = surface_height_contract(2)
    current = surface_height_contract(64)
    for name, value in [('source', data['contract']), ('target', current)]:
        if side in (name, 'both'):
            container, key = height_values(value, path)
            container[key][1] = math.nextafter(container[key][0], math.inf)
    with pytest.raises(ValueError, match='contract'):
        bc.validate_dataset(data, current)
    # Identical nonuniform contracts still match; their full values are kept.
    bc.validate_dataset(data, copy.deepcopy(data['contract']))


@pytest.mark.parametrize('path', ['success_bottom_m', 'surface_force_sharing.lift_reference_bottom_m'])
@pytest.mark.parametrize('value', [True, '0.39', None, float('nan'), float('inf')])
def test_uniform_height_arrays_require_finite_numeric_lists(bc, path, value):
    data = dataset()
    data['contract'] = surface_height_contract(2)
    current = surface_height_contract(64)
    for n, target in [(2, data['contract']), (64, current)]:
        container, key = height_values(target, path)
        container[key] = [value] * n
    with pytest.raises(ValueError, match='contract'):
        bc.validate_dataset(data, current)


@pytest.mark.parametrize('path', ['success_bottom_m', 'surface_force_sharing.lift_reference_bottom_m'])
def test_empty_height_arrays_are_kept_and_do_not_match_uniform_height(bc, path):
    data = dataset()
    data['contract'] = surface_height_contract(2)
    container, key = height_values(data['contract'], path)
    container[key] = []
    with pytest.raises(ValueError, match='contract'):
        bc.validate_dataset(data, surface_height_contract(64))
    bc.validate_dataset(data, copy.deepcopy(data['contract']))


@pytest.mark.parametrize('path', ['success_bottom_m', 'surface_force_sharing.lift_reference_bottom_m'])
@pytest.mark.parametrize('form', ['tuple', 'literal_marker'])
def test_uniform_height_arrays_do_not_accept_nonlist_marker_equivalents(bc, path, form):
    data = dataset()
    data['contract'] = surface_height_contract(2)
    container, key = height_values(data['contract'], path)
    values = container[key]
    container[key] = tuple(values) if form == 'tuple' else {'uniform_env_value': values[0]}
    with pytest.raises(ValueError, match='contract'):
        bc.validate_dataset(data, surface_height_contract(64))


@pytest.mark.parametrize('path', ['initialization.success_bottom_m',
                                  'surface_force_sharing.other_bottom_m',
                                  'lift_reference_bottom_m'])
def test_uniform_height_arrays_do_not_normalize_other_paths(bc, path):
    data = dataset()
    data['contract'] = surface_height_contract(2)
    current = surface_height_contract(64)
    for n, target in [(2, data['contract']), (64, current)]:
        components = path.split('.')
        parent = target if len(components) == 1 else target[components[0]]
        parent[components[-1]] = [.36] * n
    with pytest.raises(ValueError, match='contract'):
        bc.validate_dataset(data, current)


def test_uniform_height_arrays_preserve_actor_payload_and_dataset_contract(bc, algorithm, tmp_path):
    data = dataset()
    data['contract'] = surface_height_contract(2)
    original = copy.deepcopy(data)
    output = tmp_path / 'two_env_actor.pt'
    bc.pretrain(data, output, steps=1)
    original_bytes = output.read_bytes()
    current = surface_height_contract(64)
    metadata = bc.load_actor(algorithm, output, current)
    regularizer = bc.DemonstrationRegularizer(data, current, 'cpu')
    payload = torch.load(output, weights_only=True)
    assert_same(payload['contract'], original['contract'])
    assert_same(metadata['contract'], original['contract'])
    assert_same(regularizer.contract, original['contract'])
    assert_same(data, original)
    assert output.read_bytes() == original_bytes


def test_pretrain_reduces_real_error_and_saves_actor_only_identity(bc, tmp_path):
    data = dataset()
    path = tmp_path / 'surface_actor.pt'
    metrics = bc.pretrain(data, path, steps=100, batch_size=32, seed=42)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    actor = bc.make_actor()
    actor.load_state_dict(saved['actor'])
    with torch.no_grad():
        mse = float((actor(data['observations']) - data['actions']).square().mean())
    assert metrics['final_mse'] == pytest.approx(mse)
    assert mse < .15 * metrics['initial_mse']
    print(f"surface BC MSE: {metrics['initial_mse']:.8f} -> {mse:.8f}")
    assert saved['version'] == bc.ACTOR_VERSION
    assert saved['version'] != bc.DATASET_VERSION
    assert_same(saved['contract'], data['contract'])
    assert_same(saved['metrics'], metrics)
    assert len(saved['dataset_sha256']) == 64
    assert metrics['steps'] == 100 and metrics['samples'] == 48
    assert 'critic' not in saved and 'optimizer' not in saved


def test_pretraining_is_reproducible_and_content_identity_tracks_actions(bc, tmp_path):
    data = dataset()
    saved = []
    for index in range(3):
        if index == 2:
            data['actions'][0, 0] += .01
        path = tmp_path / f'{index}.pt'
        bc.pretrain(data, path, steps=3, batch_size=8, seed=11)
        saved.append(torch.load(path, weights_only=True))
    assert_same(saved[0]['actor'], saved[1]['actor'])
    assert saved[0]['dataset_sha256'] == saved[1]['dataset_sha256']
    assert saved[0]['dataset_sha256'] != saved[2]['dataset_sha256']


def test_load_changes_only_actor_and_reports_projected_noise(bc, algorithm, tmp_path):
    path = tmp_path / 'surface.pt'
    bc.pretrain(dataset(), path, steps=3)
    saved = torch.load(path, weights_only=True)
    critic = copy.deepcopy(algorithm.policy.critic.state_dict())
    optimizer = copy.deepcopy(algorithm.optimizer.state_dict())
    current = dict(contract(), num_envs=8, rollout_steps=64, minibatches=4)
    metadata = bc.load_actor(algorithm, path, current)
    assert_same(algorithm.policy.actor.state_dict(), saved['actor'])
    assert_same(algorithm.policy.critic.state_dict(), critic)
    assert_same(algorithm.optimizer.state_dict(), optimizer)
    from go2_pvcnn.control.dh116.tabletop_policy_noise import CONTRACT
    std = algorithm.policy.log_std.detach().exp()
    assert torch.allclose(std, torch.full_like(std, max(.03, CONTRACT['min_std'])))
    assert metadata['actual_std'] == pytest.approx(std.tolist())
    assert metadata['dataset_sha256'] == saved['dataset_sha256']
    assert metadata['metrics'] == saved['metrics']


@pytest.mark.parametrize('corruption', ['legacy', 'physics', 'actor_shape', 'actor_nan'])
def test_load_rejects_incompatible_checkpoint_before_policy_mutation(bc, algorithm, tmp_path, corruption):
    path = tmp_path / 'surface.pt'
    bc.pretrain(dataset(), path, steps=1)
    payload = torch.load(path, weights_only=True)
    if corruption == 'legacy':
        payload['version'] = 'dh116-bc-v1'
    elif corruption == 'physics':
        payload['contract']['physics_hz'] = 240
    elif corruption == 'actor_shape':
        payload['actor']['0.weight'] = torch.zeros(128, 156)
    else:
        payload['actor']['0.weight'][0, 0] = float('nan')
    torch.save(payload, path)
    policy = copy.deepcopy(algorithm.policy.state_dict())
    optimizer = copy.deepcopy(algorithm.optimizer.state_dict())
    with pytest.raises(ValueError):
        bc.load_actor(algorithm, path, contract())
    assert_same(algorithm.policy.state_dict(), policy)
    assert_same(algorithm.optimizer.state_dict(), optimizer)


def test_regularizer_uses_existing_optimizer_actor_gradients_and_exponential_decay(bc, algorithm):
    from go2_pvcnn.control.dh116.tabletop_tensor_training import observations
    data = dataset()
    regularizer = bc.DemonstrationRegularizer(data, contract(), 'cpu')
    # Preserve real rollout action / logprob and storage, with pending PPO grads.
    algorithm.act(observations(data['observations'][:2]))
    transition = {name: value.clone() for name, value in vars(algorithm.transition).items()
                  if isinstance(value, torch.Tensor)}
    critic = copy.deepcopy(algorithm.policy.critic.state_dict())
    actor = copy.deepcopy(algorithm.policy.actor.state_dict())
    nonactor = [p for p in algorithm.policy.parameters()
                if all(p is not q for q in algorithm.policy.actor.parameters())]
    moments = [copy.deepcopy(algorithm.optimizer.state[p]) for p in nonactor]
    optimizer = algorithm.optimizer
    calls = []
    hook = optimizer.register_step_post_hook(lambda *_: calls.append('step'))
    with torch.no_grad():
        before = float((algorithm.policy.actor(data['observations']) - data['actions']).square().mean())
    for iteration in range(12):
        result = regularizer.update(algorithm, iteration, initial_weight=.7, decay=.8, batch_size=256)
        assert result['weight'] == pytest.approx(.7 * .8 ** iteration)
        assert math.isfinite(result['loss']) and result['loss'] >= 0
        assert all(p.grad is None for p in nonactor)
    hook.remove()
    assert calls == ['step'] * 12 and algorithm.optimizer is optimizer
    assert any(not torch.equal(actor[k], v) for k, v in algorithm.policy.actor.state_dict().items())
    assert_same(algorithm.policy.critic.state_dict(), critic)
    for parameter, old in zip(nonactor, moments):
        assert_same(algorithm.optimizer.state[parameter], old)
    for name, old in transition.items():
        assert_same(getattr(algorithm.transition, name), old)
    assert algorithm.storage.step == 0
    with torch.no_grad():
        after = float((algorithm.policy.actor(data['observations']) - data['actions']).square().mean())
    assert after < before


def test_regularizer_rejects_incompatible_physics(bc):
    with pytest.raises(ValueError, match='contract'):
        bc.DemonstrationRegularizer(dataset(), dict(contract(), physics_hz=240), 'cpu')


@pytest.mark.parametrize('initial_weight,decay,iteration', [(0., .995, 0), (.1, 0., 1)])
def test_zero_weight_skips_adam_momentum_step(bc, algorithm, initial_weight, decay, iteration):
    regularizer = bc.DemonstrationRegularizer(dataset(), contract(), 'cpu')
    before = copy.deepcopy(algorithm.policy.state_dict())
    optimizer = copy.deepcopy(algorithm.optimizer.state_dict())
    result = regularizer.update(algorithm, iteration, initial_weight=initial_weight, decay=decay)
    assert result['weight'] == 0 and math.isfinite(result['loss'])
    assert_same(algorithm.policy.state_dict(), before)
    assert_same(algorithm.optimizer.state_dict(), optimizer)
    assert all(p.grad is None for p in algorithm.policy.parameters())


@pytest.mark.parametrize('kwargs', [dict(steps=0), dict(steps=True), dict(batch_size=0),
                                   dict(seed=-1)])
def test_invalid_pretraining_configuration_does_not_create_checkpoint(bc, tmp_path, kwargs):
    output = tmp_path / 'invalid.pt'
    arguments = dict(steps=1, batch_size=16, seed=42)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        bc.pretrain(dataset(), output, **arguments)
    assert not output.exists()


def test_legacy_bc_continues_to_reject_surface_dataset(bc, tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_behavior_cloning as legacy
    path = tmp_path / 'surface.pt'
    torch.save(dataset(), path)
    with pytest.raises(ValueError, match='version'):
        legacy.train_imitation(path, tmp_path / 'legacy_actor.pt', epochs=1)


@pytest.mark.parametrize('kwargs', [dict(iteration=-1), dict(initial_weight=-.1),
                                   dict(initial_weight=float('nan')), dict(decay=1.01),
                                   dict(decay=-.1), dict(batch_size=0)])
def test_regularizer_rejects_invalid_schedule_before_any_update(bc, algorithm, kwargs):
    regularizer = bc.DemonstrationRegularizer(dataset(), contract(), 'cpu')
    before = copy.deepcopy(algorithm.policy.state_dict())
    arguments = dict(iteration=0, initial_weight=.1, decay=.995, batch_size=16)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        regularizer.update(algorithm, **arguments)
    assert_same(algorithm.policy.state_dict(), before)


@pytest.mark.parametrize('weights', [None, [1.] * 48, torch.ones(47),
                                   torch.ones(48, dtype=torch.long), torch.zeros(48),
                                   torch.full((48,), -1.), torch.full((48,), float('nan')),
                                   torch.full((48,), float('inf'))])
def test_invalid_sample_weights_are_rejected(bc, weights):
    data = dataset()
    data['sample_weights'] = weights
    with pytest.raises(ValueError):
        bc.validate_dataset(data)


def test_family_actor_load_preserves_critic_optimizer_and_full_provenance(bc, algorithm, tmp_path):
    from test_surface_demo_library import family_data, reset_contract
    data = family_data(3)
    data['sample_weights'] = torch.tensor([.1, .1, .1, .7])
    path = tmp_path / 'family_actor.pt'
    bc.pretrain(data, path, steps=2)
    payload = torch.load(path, weights_only=True)
    assert_same(payload['reset_family'], data['reset_family'])
    assert_same(payload['sample_weights'], data['sample_weights'])
    assert payload['sample_weights_source']['dataset_sha256'] == payload['dataset_sha256']
    assert payload['sample_weights_source']['replacement'] is True
    critic = copy.deepcopy(algorithm.policy.critic.state_dict())
    optimizer = copy.deepcopy(algorithm.optimizer.state_dict())
    metadata = bc.load_actor(algorithm, path, reset_contract(2, n=64))
    assert_same(metadata['reset_family'], data['reset_family'])
    assert_same(algorithm.policy.critic.state_dict(), critic)
    assert_same(algorithm.optimizer.state_dict(), optimizer)
    before = copy.deepcopy(algorithm.policy.state_dict())
    with pytest.raises(ValueError):
        bc.load_actor(algorithm, path, reset_contract(9))
    assert_same(algorithm.policy.state_dict(), before)


def test_content_identity_covers_family_provenance_and_sample_weights(bc):
    from test_surface_demo_library import family_data
    data = family_data()
    first = bc._dataset_sha256(data)
    data['reset_family'][1]['proof_path'] = 'new/proof.json'
    assert bc._dataset_sha256(data) != first
    plain = dataset()
    without_weights = bc._dataset_sha256(plain)
    plain['sample_weights'] = torch.ones(48)
    first = bc._dataset_sha256(plain)
    assert first != without_weights
    plain['sample_weights'][0] = 2
    assert bc._dataset_sha256(plain) != first


def test_legacy_single_style_content_identity_is_unchanged(bc):
    assert bc._dataset_sha256(dataset()) == '7850b8e7c6e45c64fac180b02e0745ff7c1823a93bd88a0cef9cf271bfc865cf'


def test_merged_library_pretrain_load_and_auxiliary_step(bc, algorithm, tmp_path):
    from test_surface_demo_library import write_components, reset_contract
    from go2_pvcnn.control.dh116.tabletop_surface_demo_library import merge_datasets
    data = merge_datasets(write_components(tmp_path, 3), tmp_path / 'library.pt')
    actor_path = tmp_path / 'actor.pt'
    bc.pretrain(data, actor_path, steps=5, batch_size=32)
    critic = copy.deepcopy(algorithm.policy.critic.state_dict())
    optimizer = copy.deepcopy(algorithm.optimizer.state_dict())
    current = reset_contract(2, n=64)
    bc.load_actor(algorithm, actor_path, current)
    assert_same(algorithm.optimizer.state_dict(), optimizer)
    regularizer = bc.DemonstrationRegularizer(data, current, 'cpu')
    result = regularizer.update(algorithm, 0)
    assert math.isfinite(result['loss'])
    assert_same(algorithm.policy.critic.state_dict(), critic)


@pytest.mark.parametrize('operation', ['pretrain', 'regularizer'])
def test_weighted_sampling_uses_real_multinomial_and_balances_rare_style(bc, algorithm, monkeypatch, tmp_path, operation):
    data = dataset(n=100)
    data['sample_weights'] = torch.tensor([1 / 99] * 99 + [1.])
    weights_before = data['sample_weights'].clone()
    original = torch.multinomial
    sampled = []

    def observe(weights, n, replacement=False, *, generator=None):
        assert replacement is True
        indices = original(weights, n, replacement=replacement, generator=generator)
        sampled.append(indices.detach().cpu())
        return indices

    monkeypatch.setattr(torch, 'multinomial', observe)
    if operation == 'pretrain':
        bc.pretrain(data, tmp_path / 'balanced.pt', steps=8, batch_size=512)
    else:
        regularizer = bc.DemonstrationRegularizer(data, contract(), 'cpu')
        for i in range(8):
            regularizer.update(algorithm, i, batch_size=512)
    assert len(sampled) == 8
    rare_fraction = float((torch.cat(sampled) == 99).float().mean())
    assert .46 < rare_fraction < .54
    assert_same(data['sample_weights'], weights_before)
