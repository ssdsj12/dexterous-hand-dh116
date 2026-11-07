"""CPU-only integration oracles for the explicitly selected v3 policy."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from test_vision_training import fixture, Camera
from test_vision_resume import assert_same
from go2_pvcnn.control.dh116 import tabletop_vision_training as training


V3 = 'dh116-vision-student-ppo-v3-squashed-gaussian'


def test_supervised_warmstart_provenance_survives_resume_and_save(tmp_path):
    dw,task,env,args=setup(tmp_path)
    run(dw,task,env,args)
    source=args.output/'vision_student_ppo.pt'
    saved=torch.load(source,weights_only=False)
    derivation=dict(dataset_sha256='a'*64,warm_start_not_exact_resume=True,
        physical_success_unverified=True)
    saved['visual_supervision_derivation']=derivation
    saved['visual_supervision_history']=[derivation]
    torch.save(saved,source)
    (tmp_path/'resumed').mkdir()
    dw,task,env,new_args=setup(tmp_path/'resumed')
    new_args.resume=source
    run(dw,task,env,new_args)
    result=torch.load(new_args.output/'vision_student_ppo.pt',weights_only=False)
    assert result['visual_supervision_derivation']==derivation
    assert result['visual_supervision_history']==[derivation]


def setup(tmp_path, maximum=1):
    torch.set_num_threads(1)
    dw, task, env, args = fixture(tmp_path, maximum=maximum)
    args.vision_action_distribution = 'tanh_gaussian'
    args.surface_reward_profile = 'baseline'  # The minimal CPU critic has no reward ablation.
    return dw, task, env, args


def run(dw, task, env, args, **kwargs):
    return training.run_vision_training(SimpleNamespace(), dw, task, env, args,
        {'observation_size': 371, 'physics_hz': 480, 'pose_sha256': 'authored'},
        camera=kwargs.pop('camera', Camera(2)), **kwargs)


def test_v3_contract_checkpoint_and_json_tensorboard_diagnostics(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    dw, task, env, args = setup(tmp_path)
    run(dw, task, env, args)
    saved = torch.load(args.output/'vision_student_ppo.pt', weights_only=False)
    assert saved['version'] == saved['contract']['version'] == V3
    contract = saved['contract']
    assert contract['network']['actor_output'] == 'unbounded_latent_mean'
    assert (contract['actor_observation_size'], contract['critic_observation_size']) == (3844, 371)
    assert contract['action_contract'] == training.ACTION_CONTRACT
    assert contract['action_diagnostics']['saturation_comparison'] == 'abs_action_greater_than_or_equal_to_threshold'
    assert contract['physics_reward_provenance']['privileged_critic_contract_sha256'] == hashlib.sha256(
        json.dumps(contract['privileged_critic_contract'], sort_keys=True,
                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    row = json.loads((args.output/'iterations.jsonl').read_text().splitlines()[0])
    assert row['backend'] == V3
    diagnostics = row['ppo_diagnostics']
    assert len(diagnostics['bounded_action_mean']) == len(diagnostics['bounded_action_std']) == 24
    for name in ('bounded_policy_entropy', 'latent_gaussian_entropy', 'approx_kl',
                 'analytic_kl_old_new', 'clip_fraction', 'old_log_prob_mean',
                 'new_log_prob_mean', 'frozen_log_prob_abs_error_max',
                 'action_saturation_rate', 'preclip_gradient_norm'):
        assert torch.isfinite(torch.tensor(diagnostics[name]))
    assert diagnostics['frozen_log_prob_abs_error_max'] < 2e-5
    tags = EventAccumulator(str(args.output/'tensorboard')).Reload().Tags()['scalars']
    assert 'ppo_diagnostics/analytic_kl_old_new' in tags
    assert all(f'ppo_diagnostics/bounded_action_mean/{i:02d}' in tags for i in range(24))
    assert set(row['losses']) == {'value_function', 'surrogate', 'entropy'}


def test_bounded_command_storage_and_history_survive_environment_mutation(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import make_squashed_vision_algorithm
    dw, task, env, args = setup(tmp_path)
    received, sampled, history = [], [], []
    original_step, original_observations = env.step, training._observations
    def factory(*a, **kw):
        algorithm = make_squashed_vision_algorithm(*a, **kw)
        original_act = algorithm.act
        def act(obs):
            result = original_act(obs)
            sampled.append(result.clone())
            return result
        algorithm.act = act
        original_update = algorithm.update
        def update():
            torch.testing.assert_close(algorithm.storage.actions, torch.stack(sampled), atol=0, rtol=0)
            return original_update()
        algorithm.update = update
        return algorithm
    def step(action):
        received.append(action.clone())
        result = original_step(action)
        action.fill_(.88)  # The environment owns only its input copy.
        return result
    def observations(packet, world, previous, *a, **kw):
        history.append(previous.clone())
        return original_observations(packet, world, previous, *a, **kw)
    monkeypatch.setattr(training, 'make_squashed_vision_algorithm', factory, raising=False)
    monkeypatch.setattr(training, '_observations', observations)
    env.step = step
    run(dw, task, env, args)
    assert len(received) == args.rollout_steps
    assert len(sampled) == args.rollout_steps
    assert len(history) == args.rollout_steps+1
    for actual, expected, previous in zip(received, sampled, history[1:]):
        assert bool((actual.abs() <= 1).all())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(previous, expected, atol=0, rtol=0)


@pytest.mark.parametrize('invalid', [1.35, float('nan')])
def test_invalid_new_mode_command_is_rejected_before_environment(tmp_path, monkeypatch, invalid):
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import make_squashed_vision_algorithm
    dw, task, env, args = setup(tmp_path)
    def factory(*a, **kw):
        algorithm = make_squashed_vision_algorithm(*a, **kw)
        algorithm.act = lambda obs: torch.full((2, 24), invalid)
        return algorithm
    monkeypatch.setattr(training, 'make_squashed_vision_algorithm', factory, raising=False)
    with pytest.raises((ValueError, RuntimeError), match='action'):
        run(dw, task, env, args)
    assert env.tick == 0


@pytest.mark.parametrize('corruption', ['v2', 'contract', 'type', 'nonfinite', 'downsize',
    'weight_shape', 'weight_dtype', 'weight_missing', 'weight_extra', 'moment_shape',
    'moment_dtype', 'negative_square_moment', 'negative_step', 'vector_step',
    'empty_adam', 'std_above', 'std_below', 'cpu_rng', 'maximize'])
def test_resume_rejected_before_any_metadata_reset_or_camera(tmp_path, corruption):
    dw, task, env, args = setup(tmp_path)
    run(dw, task, env, args)
    source = args.output/'vision_student_ppo.pt'
    saved = torch.load(source, weights_only=False)
    if corruption == 'v2': saved['version'] = training.VERSION
    if corruption == 'contract': saved['contract']['unknown'] = 'unsafe'
    if corruption == 'type': saved['contract']['actor_uses_privileged_state'] = 0
    if corruption == 'nonfinite': saved['policy']['log_std'][0] = float('nan')
    if corruption == 'weight_shape': saved['policy']['actor.action_net.0.weight'] = torch.zeros(128,792)
    if corruption == 'weight_dtype': saved['policy']['critic.4.bias'] = torch.zeros(1,dtype=torch.float64)
    if corruption == 'weight_missing': saved['policy'].pop('critic.4.bias')
    if corruption == 'weight_extra': saved['policy']['unknown'] = torch.zeros(1)
    if corruption == 'moment_shape': saved['optimizer']['state'][0]['exp_avg'] = torch.zeros(3)
    if corruption == 'moment_dtype': saved['optimizer']['state'][0]['exp_avg'] = torch.zeros(24,dtype=torch.float64)
    if corruption == 'negative_square_moment': saved['optimizer']['state'][0]['exp_avg_sq'][0] = -1.
    if corruption == 'negative_step': saved['optimizer']['state'][0]['step'] = torch.tensor(-1.)
    if corruption == 'vector_step': saved['optimizer']['state'][0]['step'] = torch.tensor([1.,1.])
    if corruption == 'empty_adam': saved['optimizer']['state'] = {}
    if corruption == 'std_above': saved['policy']['log_std'].fill_(0.)
    if corruption == 'std_below': saved['policy']['log_std'].fill_(-20.)
    if corruption == 'cpu_rng': saved['torch_rng'] = torch.ones(1,dtype=torch.uint8)
    if corruption == 'maximize': saved['optimizer']['param_groups'][0]['maximize'] = True
    broken = tmp_path/'broken.pt'; torch.save(saved, broken)
    before = broken.read_bytes()
    args.output = tmp_path/'rejected'; args.output.mkdir()
    args.resume = broken
    if corruption == 'downsize': args.vision_resume_downsize = True
    def fail(*a, **kw): raise AssertionError('reset/camera must not run')
    env.reset = fail
    camera = Camera(2); camera.sample = fail
    with pytest.raises(ValueError): run(dw, task, env, args, camera=camera)
    assert not list(args.output.iterdir())
    assert broken.read_bytes() == before


def test_existing_output_is_rejected_without_changing_old_bytes(tmp_path):
    dw, task, env, args = setup(tmp_path)
    marker = args.output/'monitor.json'; marker.write_bytes(b'original output')
    with pytest.raises(FileExistsError): run(dw, task, env, args)
    assert marker.read_bytes() == b'original output'
    assert env.tick == 0
    assert sorted(p.name for p in args.output.iterdir()) == ['monitor.json']


def test_completed_budget_exact_resume_copies_adam_lr_rng_counters_and_derivation(tmp_path):
    dw, task, env, args = setup(tmp_path)
    run(dw, task, env, args)
    source = args.output/'vision_student_ppo.pt'
    saved = torch.load(source, weights_only=False)
    saved['action_derivation'] = {'kind': 'warm_start_not_exact_resume', 'source_sha256': 'historical'}
    saved['resume_migration'] = {'version': 'earlier_downsize', 'source_iteration': 5}
    torch.save(saved, source)
    original = source.read_bytes()
    args.resume = source; args.output = tmp_path/'resumed'; args.output.mkdir()
    run(dw, task, env, args)
    result = torch.load(args.output/'vision_student_ppo.pt', weights_only=False)
    for key in ('policy', 'optimizer', 'learning_rate', 'torch_rng', 'cuda_rng', 'iteration',
                'timesteps', 'episodes', 'successes', 'action_derivation', 'resume_migration'):
        assert_same(result[key], saved[key])
    assert source.read_bytes() == original
    start = json.loads((args.output/'training_start.json').read_text())
    assert start['resume_environment_state'] == 'cold_reset_after_resume'
    assert start['resume_restored_state'] == ['policy', 'Adam', 'learning_rate', 'Torch_RNG', 'counters']


def test_legacy_default_retains_unextended_v2_contract(tmp_path):
    dw, task, env, args = setup(tmp_path)
    del args.vision_action_distribution
    run(dw, task, env, args)
    saved = torch.load(args.output/'vision_student_ppo.pt', weights_only=False)
    assert saved['version'] == training.VERSION
    assert 'action_distribution' not in saved['contract']
    assert 'ppo_diagnostics' not in json.loads((args.output/'monitor.json').read_text())
