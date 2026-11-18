"""CPU fixtures test actor-only fitting, never native grasp qualification."""
import copy
import hashlib
import importlib
from pathlib import Path

import pytest
import torch

from test_vision_action_derivation import assert_same, source_checkpoint, observations


@pytest.fixture(autouse=True)
def isolate_fixture_rng_and_threads():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        try: yield
        finally: torch.set_num_threads(threads)


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_supervised_bc')


def source_v3(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_action_derivation import derive_action_checkpoint
    source, _ = source_checkpoint(tmp_path)
    v3 = tmp_path/'v3.pt'
    derive_action_checkpoint(source, v3)
    return v3, torch.load(v3, map_location='cpu', weights_only=False)


def sensor_rows(n=8):
    packet = observations()['policy']
    return packet.repeat((n+1)//2, 1)[:n].clone()


def test_bounded_fit_uses_one_tanh_and_supports_endpoint_labels():
    m = module()
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    torch.set_num_threads(2)
    torch.manual_seed(11)
    actor = UnboundedVisionStudentActor()
    obs = sensor_rows()
    labels = torch.ones(len(obs), 24)
    labels[:, 1::2] = -1.
    initial = copy.deepcopy(actor.state_dict())
    before_rng = torch.get_rng_state().clone()
    result = m.fit_actor(initial, obs, labels, steps=12, batch_size=4,
                         learning_rate=.003, seed=42)
    assert_same(before_rng, torch.get_rng_state())
    assert result['loss_after'] < result['loss_before']
    assert result['steps'] == 12
    assert all(torch.isfinite(value).all() for value in result['actor'].values())
    # Both shared PointNet and action MLP are actually optimized.
    for name in ('point_net.0.weight', 'action_net.0.weight', 'action_net.4.weight'):
        assert not torch.equal(initial[name], result['actor'][name]), name
    check = UnboundedVisionStudentActor()
    check.load_state_dict(result['actor'])
    with torch.no_grad(): expected = ((check(obs).tanh()-labels).square()).mean().item()
    assert result['loss_after'] == pytest.approx(expected, abs=1e-7, rel=1e-5)
    assert result['physical_success_unverified'] is True


def test_private_bc_adam_and_sampling_rng_can_continue_exactly():
    m = module()
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    torch.set_num_threads(2)
    initial = UnboundedVisionStudentActor().state_dict()
    obs = sensor_rows()
    labels = torch.linspace(-1., 1., 24).expand(len(obs), -1).clone()
    kwargs = dict(batch_size=3, learning_rate=.001, seed=8)
    first = m.fit_actor(initial, obs, labels, steps=3, **kwargs)
    resumed = m.fit_actor(first['actor'], obs, labels, steps=4,
        optimizer_state=first['optimizer'], sampling_rng_state=first['sampling_rng'], **kwargs)
    uninterrupted = m.fit_actor(initial, obs, labels, steps=7, **kwargs)
    assert_same(uninterrupted['actor'], resumed['actor'])
    assert_same(uninterrupted['optimizer'], resumed['optimizer'])
    assert_same(uninterrupted['sampling_rng'], resumed['sampling_rng'])


@pytest.mark.parametrize('bad', ['shape', 'nonfinite_obs', 'nonfinite_actions', 'out_of_bounds'])
def test_fitting_rejects_invalid_sensor_or_issued_labels(bad):
    m = module()
    obs, labels = sensor_rows(), torch.zeros(8, 24)
    if bad == 'shape': obs = torch.zeros(8, 371)
    elif bad == 'nonfinite_obs': obs[0, 0] = float('nan')
    elif bad == 'nonfinite_actions': labels[0, 0] = float('inf')
    elif bad == 'out_of_bounds': labels[0, 0] = 1.01
    with pytest.raises(ValueError):
        m.fit_actor({}, obs, labels, steps=1)


def minimal_data(contract):
    # Synthetic metadata is explicitly confined to CPU tests, never reported
    # as a real native run. Runtime dataset validation is still exercised.
    from go2_pvcnn.control.dh116.tabletop_vision_supervision_dataset import VERSION
    styles = ['pinch_thumb_index']*4 + ['tripod_thumb_index_middle']*4
    rows = []
    episodes = {}
    for i, style in enumerate(styles):
        env = 0 if i < 4 else 1
        episode = env
        source_id = 'a'*64 if env == 0 else 'b'*64
        episodes[str(episode)] = dict(env_id=env, first_episode=True, success=True,
            recovered=False, native_physics=True, grasp_style=style,
            source_trajectory_id=source_id, source_episode_id=env,
            proof_path='CPU_SYNTHETIC_FIXTURE_NOT_NATIVE_PROOF')
        rows.append(dict(env_id=env, episode_id=episode, tick=i%4,
            frame_id=i%4, simulation_time_s=(i%4)/60.,
            source_trajectory_id=source_id, grasp_style=style, source_episode_id=env))
    return dict(version=VERSION, observations=sensor_rows(), actions=torch.ones(8,24)*.6,
        reference_actions=torch.ones(8,24)*.6, sample_metadata=rows,
        episodes=episodes, visual_contract=copy.deepcopy(contract),
        teacher_source_sha256='c'*64, validation_scope='schema_only_not_native_verification',
        source_trajectory_ids=[row['source_trajectory_id'] for row in rows],
        grasp_styles=styles)


def test_explicit_artifact_and_warmstart_preserve_all_ppo_state(tmp_path):
    m = module()
    source, before = source_v3(tmp_path)
    source_bytes = source.read_bytes()
    data = minimal_data(before['contract'])
    dataset = tmp_path/'data.pt'; torch.save(data, dataset)
    artifact = tmp_path/'bc.pt'
    report = m.fit_supervision(source, dataset, artifact, steps=3,
        batch_size=4, learning_rate=.001, allow_fit_only=True)
    assert source.read_bytes() == source_bytes
    fitted = torch.load(artifact, weights_only=False)
    assert fitted['version'] == m.VERSION
    assert_same(before, fitted['source_checkpoint_state'])
    assert report['fit_only'] is True and report['native_success_claimed'] is False
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_squashed_checkpoint
    with pytest.raises(ValueError): validate_squashed_checkpoint(fitted)
    derived_path = tmp_path/'warm.pt'
    m.derive_supervised_checkpoint(artifact, derived_path)
    after = torch.load(derived_path, weights_only=False)
    validate_squashed_checkpoint(after, before['contract'])
    for key, value in before.items():
        if key != 'policy': assert_same(value, after[key])
    for key, value in before['policy'].items():
        if not key.startswith('actor.'): assert_same(value, after['policy'][key])
    for key, value in fitted['trained_actor'].items():
        assert_same(value, after['policy']['actor.'+key])
    meta = after['visual_supervision_derivation']
    assert meta['status'] == 'warm_start_not_exact_resume'
    assert meta['source_checkpoint_sha256'] == hashlib.sha256(source_bytes).hexdigest()
    assert meta['physical_success_unverified'] is True
    assert after['visual_supervision_history'][-1] == meta


def test_fit_requires_explicit_fit_only_without_independent_source_holdout(tmp_path):
    m = module()
    source, saved = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    with pytest.raises(ValueError, match='fit.only'):
        m.fit_supervision(source, dataset, tmp_path/'bc.pt', steps=1)
    assert not (tmp_path/'bc.pt').exists()


@pytest.mark.parametrize('collision', ['source', 'artifact', 'sidecar'])
def test_no_output_can_overwrite_source_or_existing_artifacts(tmp_path, collision):
    m = module()
    source, saved = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    out = source if collision == 'source' else tmp_path/'bc.pt'
    block = out if collision in ('source', 'artifact') else out.with_suffix('.json')
    if collision != 'source': block.write_bytes(b'preserve')
    initial = block.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        m.fit_supervision(source, dataset, out, steps=1, allow_fit_only=True)
    assert block.read_bytes() == initial


def test_training_subset_excludes_whole_heldout_source(tmp_path):
    m = module()
    source, saved = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    report = m.fit_supervision(source, dataset, tmp_path/'bc.pt', steps=1,
        heldout_trajectories=['b'*64])
    assert report['train_sample_count'] == report['heldout_sample_count'] == 4
    assert report['heldout_source_trajectory_ids'] == ['b'*64]
    assert report['fit_only'] is False
    # One independent trajectory of only one style is not per-style generalization.
    assert report['per_style_independent_holdout_complete'] is False


def test_supervised_derivation_appends_history(tmp_path):
    m = module()
    source, saved = source_v3(tmp_path)
    saved['visual_supervision_history'] = [{'older': 'preserved'}]
    saved['visual_supervision_derivation'] = {'older': 'preserved'}
    torch.save(saved, source)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    m.fit_supervision(source, dataset, tmp_path/'bc.pt', steps=1, allow_fit_only=True)
    m.derive_supervised_checkpoint(tmp_path/'bc.pt', tmp_path/'warm.pt')
    after = torch.load(tmp_path/'warm.pt', weights_only=False)
    assert after['visual_supervision_history'][0] == {'older': 'preserved'}
    assert len(after['visual_supervision_history']) == 2


def test_contract_batch_projection_must_be_explicit(tmp_path):
    m = module()
    source, saved = source_v3(tmp_path)
    data = minimal_data(saved['contract'])
    data['visual_contract']['num_envs'] = 4
    dataset = tmp_path/'data.pt'; torch.save(data, dataset)
    with pytest.raises(ValueError, match='explicitly project'):
        m.fit_supervision(source, dataset, tmp_path/'bc.pt', steps=1, allow_fit_only=True)
    assert not (tmp_path/'bc.pt').exists()


@pytest.mark.parametrize('field', ['critic', 'optimizer', 'counters'])
def test_artifact_cannot_mutate_retained_source_state(tmp_path, field):
    m = module()
    source, saved = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    artifact = tmp_path/'bc.pt'
    m.fit_supervision(source, dataset, artifact, steps=1, allow_fit_only=True)
    corrupt = torch.load(artifact, weights_only=False)
    if field == 'critic': corrupt['source_checkpoint_state']['policy']['critic.0.bias'][0] += .01
    elif field == 'optimizer': corrupt['source_checkpoint_state']['optimizer']['state'][0]['exp_avg'][0] += .01
    else: corrupt['source_checkpoint_state']['timesteps'] += 1
    torch.save(corrupt, artifact)
    with pytest.raises(ValueError, match='immutable source'):
        m.derive_supervised_checkpoint(artifact, tmp_path/'warm.pt')
    assert not (tmp_path/'warm.pt').exists()


@pytest.mark.parametrize('phase', ['fit', 'derive'])
@pytest.mark.parametrize('collision', ['checkpoint', 'sidecar'])
def test_competing_writer_survives_without_partial_artifact(tmp_path, monkeypatch, phase, collision):
    m = module()
    source, saved = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    artifact = tmp_path/'bc.pt'
    if phase == 'derive':
        m.fit_supervision(source, dataset, artifact, steps=1, allow_fit_only=True)
    output = artifact if phase == 'fit' else tmp_path/'warm.pt'
    blocked = output if collision == 'checkpoint' else output.with_suffix('.json')
    original_open = Path.open
    raced = []
    def racing_open(path, mode='r', *args, **kwargs):
        if path == blocked and mode in ('xb', 'x') and not raced:
            raced.append(True)
            with original_open(path, 'wb') as stream: stream.write(b'competing owner')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', racing_open)
    with pytest.raises(FileExistsError):
        if phase == 'fit':
            m.fit_supervision(source, dataset, output, steps=1, allow_fit_only=True)
        else: m.derive_supervised_checkpoint(artifact, output)
    assert raced and blocked.read_bytes() == b'competing owner'
    other = output.with_suffix('.json') if collision == 'checkpoint' else output
    assert not other.exists()


@pytest.mark.parametrize('mutation', ['checkpoint', 'dataset'])
def test_changed_inputs_remove_only_our_new_outputs(tmp_path, monkeypatch, mutation):
    m = module()
    source, saved = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    original_save = m.torch.save
    target = source if mutation == 'checkpoint' else dataset
    out = tmp_path/'bc.pt'
    def changing_save(*args, **kwargs):
        original_save(*args, **kwargs)
        with target.open('ab') as stream: stream.write(b'external modification')
    monkeypatch.setattr(m.torch, 'save', changing_save)
    with pytest.raises(RuntimeError, match='changed during BC'):
        m.fit_supervision(source, dataset, out, steps=1, allow_fit_only=True)
    assert not out.exists() and not out.with_suffix('.json').exists()
    assert target.read_bytes().endswith(b'external modification')


def test_failed_native_episode_cannot_enter_bc_pool(tmp_path):
    m = module()
    source, saved = source_v3(tmp_path)
    data = minimal_data(saved['contract'])
    data['episodes']['0']['success'] = False
    dataset = tmp_path/'data.pt'; torch.save(data, dataset)
    with pytest.raises(ValueError, match='successful first native episodes'):
        m.fit_supervision(source, dataset, tmp_path/'bc.pt', steps=1, allow_fit_only=True)
    assert not (tmp_path/'bc.pt').exists()


def dead_source(tmp_path):
    source, saved = source_v3(tmp_path)
    saved['policy']['actor.point_net.4.weight'].zero_()
    saved['policy']['actor.point_net.4.bias'].fill_(-1.)
    torch.save(saved, source)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    return source, saved, dataset


def test_inactive_source_pointnet_is_rejected_before_fit_or_artifact_write(tmp_path, monkeypatch):
    m = module()
    source, _, dataset = dead_source(tmp_path)
    monkeypatch.setattr(m, 'fit_actor', lambda *a, **k: pytest.fail('dead source must fail before training'))
    with pytest.raises(ValueError, match='inactive PointNet.*reset_pointnet'):
        m.fit_supervision(source, dataset, tmp_path/'bad.pt', steps=3, allow_fit_only=True)
    assert not (tmp_path/'bad.pt').exists()
    assert not (tmp_path/'bad.json').exists()


def test_explicit_pointnet_reset_restarts_only_encoder_and_obtains_visual_gradients(tmp_path):
    m = module()
    source, saved, dataset = dead_source(tmp_path)
    source_bytes = source.read_bytes()
    rng = torch.get_rng_state().clone()
    report = m.fit_supervision(source, dataset, tmp_path/'reset_bc.pt', steps=5,
        learning_rate=.001, allow_fit_only=True, actor_initialization='reset_pointnet', seed=7)
    assert_same(rng, torch.get_rng_state())
    assert source.read_bytes() == source_bytes
    assert report['source_visual_branch_audit']['nonzero_valid_point_features'] == 0
    assert report['visual_branch_before']['nonzero_valid_point_features'] > 0
    assert report['visual_branch_after']['zero_points_max_action_delta'] > 0
    assert report['pointnet_gradient_norm_max'] > 0
    assert report['pointnet_gradient_nonzero_steps'] > 0
    assert report['visual_gradient_verified'] is True
    init = report['actor_initialization']
    assert init['mode'] == 'reset_pointnet' and init['seed'] == 7
    assert len(init['reinitialized_parameters']) == 6
    for name, value in saved['policy'].items():
        if not name.startswith('actor.'): continue
        key = name.removeprefix('actor.')
        source_sha = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        if key.startswith('action_net.'):
            assert init['parameter_sha256_before_BC'][key] == source_sha
    for name, coverage in report['pointnet_gradient_coverage'].items():
        assert name.startswith('point_net.')
        assert coverage['nonzero_gradient_steps'] > 0
        assert coverage['ever_nonzero_elements'] > 0
    m.derive_supervised_checkpoint(tmp_path/'reset_bc.pt', tmp_path/'reset_warm.pt')
    warm = torch.load(tmp_path/'reset_warm.pt', weights_only=False)
    for key, value in saved.items():
        if key != 'policy': assert_same(value, warm[key])
    for name, value in saved['policy'].items():
        if not name.startswith('actor.'): assert_same(value, warm['policy'][name])
    assert warm['visual_supervision_derivation']['fit_report']['actor_initialization'] == init


def test_primitive_reports_dead_visual_gradients_instead_of_claiming_visual_learning():
    m = module()
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    actor = UnboundedVisionStudentActor()
    state = actor.state_dict()
    state['point_net.4.weight'].zero_(); state['point_net.4.bias'].fill_(-1.)
    obs = sensor_rows()
    out = m.fit_actor(state, obs, torch.ones(8,24)*.6, steps=3)
    assert out['pointnet_gradient_norm_max'] == 0.
    assert out['pointnet_gradient_nonzero_steps'] == 0
    assert out['visual_branch_before']['zero_points_max_action_delta'] == 0.
    assert out['visual_branch_after']['zero_points_max_action_delta'] == 0.
    assert all(row['ever_nonzero_elements'] == 0 for row in out['pointnet_gradient_coverage'].values())
    assert out['loss_after'] < out['loss_before']  # Lower fit error alone is insufficient.


@pytest.mark.parametrize('failure', ['gradient', 'sensitivity'])
def test_public_fit_never_writes_unverified_visual_artifact(tmp_path, monkeypatch, failure):
    m = module()
    source, saved = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    original = m.fit_actor
    def invalid_fit(*args, **kwargs):
        out = original(*args, **kwargs)
        if failure == 'gradient': out['pointnet_gradient_nonzero_steps'] = 0
        else: out['visual_branch_after']['zero_points_max_action_delta'] = 0.
        return out
    monkeypatch.setattr(m, 'fit_actor', invalid_fit)
    with pytest.raises(ValueError, match='visual BC verification failed'):
        m.fit_supervision(source, dataset, tmp_path/'bad.pt', steps=1, allow_fit_only=True)
    assert not (tmp_path/'bad.pt').exists() and not (tmp_path/'bad.json').exists()


def test_legacy_proprio_only_bc_artifact_cannot_be_derived_as_visual_verified(tmp_path):
    m = module()
    source, saved = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    m.fit_supervision(source, dataset, tmp_path/'bc.pt', steps=1, allow_fit_only=True)
    artifact = torch.load(tmp_path/'bc.pt', weights_only=False)
    artifact['version'] = 'dh116-vision-supervised-bc-v1'
    torch.save(artifact, tmp_path/'legacy.pt')
    with pytest.raises(ValueError, match='artifact version'):
        m.derive_supervised_checkpoint(tmp_path/'legacy.pt', tmp_path/'bad.pt')
    assert not (tmp_path/'bad.pt').exists()


def test_pointnet_reset_is_seed_reproducible_and_rejects_unknown_modes():
    m = module()
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    initial = UnboundedVisionStudentActor().state_dict()
    obs, labels = sensor_rows(), torch.ones(8,24)*.6
    one = m.fit_actor(initial, obs, labels, steps=2, actor_initialization='reset_pointnet', seed=9)
    two = m.fit_actor(initial, obs, labels, steps=2, actor_initialization='reset_pointnet', seed=9)
    assert_same(one['actor'], two['actor'])
    assert one['actor_initialization'] == two['actor_initialization']
    with pytest.raises(ValueError, match='initialization'):
        m.fit_actor(initial, obs, labels, steps=1, actor_initialization='silently_reset_everything')


def test_pointnet_reset_never_reseeds_global_CUDA_generators(monkeypatch):
    m = module()
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    initial = UnboundedVisionStudentActor().state_dict()
    obs = sensor_rows()
    # torch.manual_seed also seeds CUDA generators. Only a private CPU
    # generator may drive encoder initialization, even in a GPU PPO process.
    monkeypatch.setattr(m.torch, 'manual_seed', lambda *a: pytest.fail('global reseed touches CUDA RNG'))
    m.fit_actor(initial, obs, torch.ones(8,24)*.6, steps=1,
                actor_initialization='reset_pointnet', seed=9)
