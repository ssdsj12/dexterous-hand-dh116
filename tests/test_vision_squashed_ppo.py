"""Saved-latent storage, weighted PPO, and Adam/microbatch equivalence."""
from copy import deepcopy
import importlib
import importlib.util
import math

import pytest
import torch
from tensordict import TensorDict


def factory():
    name = 'go2_pvcnn.control.dh116.tabletop_vision_squashed_policy'
    assert importlib.util.find_spec(name), 'squashed PPO factory is missing'
    return importlib.import_module(name).make_squashed_vision_algorithm


def obs(n=5):
    policy = torch.randn(n,3844)*.1
    frames = policy.reshape(n,4,961)
    frames[...,768:896] = 1.
    return TensorDict(dict(policy=policy, critic=torch.randn(n,371)), [n])


def collect(algorithm, observations, steps=3):
    for i in range(steps):
        actions = algorithm.act(observations)
        algorithm.process_env_step(observations, 1-actions.square().sum(-1),
                                   torch.full((len(observations),), i==1, dtype=torch.bool), {})
        assert algorithm.transition.latent_actions is None
    algorithm.compute_returns(observations)


def test_storage_copies_latents_bounded_commands_done_and_reuse():
    make = factory()
    torch.set_num_threads(2)
    observations = obs()
    algorithm = make(observations,3,inference_batch_size=2)
    with torch.no_grad():
        algorithm.policy.actor.action_net[-2].bias.fill_(20.)
    actions = algorithm.act(observations)
    z = algorithm.transition.latent_actions.clone()
    logp = algorithm.transition.actions_log_prob.clone()
    algorithm.process_env_step(observations,torch.ones(5),torch.ones(5,dtype=torch.bool),{})
    torch.testing.assert_close(algorithm.storage.latent_actions[0],z,atol=0,rtol=0)
    torch.testing.assert_close(algorithm.storage.actions[0],actions,atol=0,rtol=0)
    torch.testing.assert_close(actions,z.tanh(),atol=0,rtol=0)
    torch.testing.assert_close(algorithm.storage.actions_log_prob[0,:,0],logp,atol=0,rtol=0)
    assert algorithm.storage.dones[0].all()
    for name in ('actions','latent_actions','actions_log_prob','mu','sigma','values'):
        tensor = getattr(algorithm.storage,name)
        assert tensor.dtype == torch.float32 and tensor.device.type == 'cpu'
    algorithm.storage.clear()
    with torch.no_grad():
        algorithm.policy.actor.action_net[-2].bias.zero_()
    algorithm.act(observations)
    new_z = algorithm.transition.latent_actions.clone()
    algorithm.process_env_step(observations,torch.ones(5),torch.zeros(5,dtype=torch.bool),{})
    torch.testing.assert_close(algorithm.storage.latent_actions[0],new_z,atol=0,rtol=0)
    assert not algorithm.storage.dones[0].any()


def test_frozen_policy_ratio_kl_clip_and_diagnostics_are_exact():
    observations = obs()
    algorithm = factory()(observations,3,microbatch_size=2)
    algorithm.num_learning_epochs = 1
    algorithm.num_mini_batches = 1
    collect(algorithm,observations)
    actions = algorithm.storage.actions.flatten(0,1).clone()
    z = algorithm.storage.latent_actions.clone()
    old_logp = algorithm.storage.actions_log_prob.clone()
    losses = algorithm.update()
    d = algorithm.last_update_diagnostics
    assert set(losses) == {'value_function','surrogate','entropy'}
    assert d['frozen_log_prob_abs_error_max'] < 2e-5
    assert abs(d['approx_kl']) < 1e-9 and abs(d['analytic_kl_old_new']) < 1e-9
    assert d['clip_fraction'] == 0.
    assert math.isclose(d['action_raw_abs_max'],float(z.abs().max()))
    torch.testing.assert_close(torch.tensor(d['bounded_action_mean']),actions.mean(0))
    torch.testing.assert_close(torch.tensor(d['bounded_action_std']),actions.std(0,unbiased=False))
    assert d['old_log_prob_min'] == float(old_logp.min())
    assert d['old_log_prob_max'] == float(old_logp.max())
    assert d['bounded_policy_entropy'] == losses['entropy']
    assert algorithm.storage.step == 0


@pytest.mark.parametrize('microbatch',[1,3,100])
def test_full_vs_microbatch_same_rollout_noise_tail_adam_gradients_lr_and_diagnostics(microbatch):
    torch.set_num_threads(2)
    torch.manual_seed(41)
    observations = obs()
    full = factory()(observations,3,microbatch_size=100)
    actual = factory()(observations,3,microbatch_size=microbatch)
    actual.policy.load_state_dict(full.policy.state_dict())
    for algorithm in (full,actual):
        algorithm.num_learning_epochs = 2
        algorithm.num_mini_batches = 4  # 15 samples => 4/4/4/3, keeps tail.
    collect(full,observations)
    actual.storage = deepcopy(full.storage)
    gradients = [[],[]]
    for i,algorithm in enumerate((full,actual)):
        algorithm.optimizer.register_step_pre_hook(lambda opt,args,kwargs,i=i:
            gradients[i].append(torch.cat([p.grad.flatten().clone() for p in (full,actual)[i].policy.parameters()])))
    torch.manual_seed(92)
    expected = full.update()
    torch.manual_seed(92)
    result = actual.update()
    for key in result:
        assert math.isclose(result[key],expected[key],rel_tol=2e-4,abs_tol=5e-6)
    assert len(gradients[0]) == len(gradients[1]) == 8
    for a,b in zip(gradients[0],gradients[1]):
        torch.testing.assert_close(a,b,atol=6e-6,rtol=3e-3)
    for name,value in full.policy.state_dict().items():
        torch.testing.assert_close(actual.policy.state_dict()[name],value,atol=8e-6,rtol=5e-4)
    for a,b in zip(full.optimizer.state.values(),actual.optimizer.state.values()):
        for key in a:
            torch.testing.assert_close(a[key],b[key],atol=6e-6,rtol=5e-3)
    assert actual.learning_rate == full.learning_rate
    for key,value in full.last_update_diagnostics.items():
        torch.testing.assert_close(torch.tensor(actual.last_update_diagnostics[key]),torch.tensor(value),
                                   atol=3e-5,rtol=2e-3)
    assert all(.001-1e-7 <= s <= .3+1e-7 for s in actual.policy.log_std.exp())


def test_nonfinite_gradient_never_steps_adam():
    observations = obs(2)
    algorithm = factory()(observations,1)
    collect(algorithm,observations,1)
    before = deepcopy(algorithm.policy.state_dict())
    parameter = next(algorithm.policy.parameters())
    handle = parameter.register_hook(lambda grad: torch.full_like(grad,float('nan')))
    with pytest.raises((ValueError,RuntimeError),match='finite'):
        algorithm.update()
    handle.remove()
    assert not algorithm.optimizer.state
    for name,value in before.items():
        torch.testing.assert_close(algorithm.policy.state_dict()[name],value,atol=0,rtol=0)


def test_changed_policy_ppo_matches_independent_loss_gradient_entropy_and_kl_oracle():
    """The oracle uses Normal/TanhTransform directly, not production math helpers."""
    from torch.distributions import Normal, TransformedDistribution, TanhTransform, kl_divergence
    torch.manual_seed(705)
    observations = obs()
    algorithm = factory()(observations,3,microbatch_size=3)
    algorithm.num_learning_epochs = algorithm.num_mini_batches = 1
    algorithm.normalize_advantage_per_mini_batch = True
    collect(algorithm,observations)
    with torch.no_grad():
        algorithm.policy.actor.action_net[-2].bias.add_(.03)
        algorithm.policy.log_std.add_(.08)
    reference = deepcopy(algorithm.policy)
    storage = algorithm.storage
    torch.manual_seed(331)
    indices = torch.randperm(15)
    epsilon = torch.randn(15,24)
    flattened = storage.observations.flatten(0,1)[indices]
    z = storage.latent_actions.flatten(0,1)[indices]
    old_logp = storage.actions_log_prob.flatten(0,1)[indices].flatten()
    old_mu = storage.mu.flatten(0,1)[indices]
    old_std = storage.sigma.flatten(0,1)[indices]
    mu = reference.actor(flattened['policy'])
    std = reference.log_std.exp().expand_as(mu)
    normal = Normal(mu,std)
    logp = TransformedDistribution(normal,[TanhTransform()]).log_prob(z.tanh()).sum(-1)
    ratio = (logp-old_logp).exp()
    advantage = storage.advantages.flatten(0,1)[indices].flatten()
    advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
    surrogate = torch.maximum(-advantage*ratio,-advantage*ratio.clamp(.8,1.2)).mean()
    current_z = mu+std*epsilon
    entropy = (normal.entropy()+torch.log1p(-current_z.tanh().square())).sum(-1).mean()
    values = reference.evaluate(flattened)
    old_values = storage.values.flatten(0,1)[indices]
    returns = storage.returns.flatten(0,1)[indices]
    clipped_values = old_values+(values-old_values).clamp(-.2,.2)
    value_loss = torch.maximum((values-returns).square(),(clipped_values-returns).square()).mean()
    expected_loss = surrogate+.5*value_loss-.005*entropy
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(),.5)
    expected_gradients = torch.cat([p.grad.flatten() for p in reference.parameters()])
    kl = kl_divergence(Normal(old_mu,old_std),normal).sum(-1).mean()
    actual_gradients = []
    algorithm.optimizer.register_step_pre_hook(lambda *_:
        actual_gradients.append(torch.cat([p.grad.flatten().clone() for p in algorithm.policy.parameters()])))
    torch.manual_seed(331)
    losses = algorithm.update()
    for name,expected in [('surrogate',surrogate),('value_function',value_loss),('entropy',entropy)]:
        assert math.isclose(losses[name],float(expected.detach()),rel_tol=2e-5,abs_tol=4e-6)
    torch.testing.assert_close(actual_gradients[0],expected_gradients,atol=2e-6,rtol=1e-3)
    d = algorithm.last_update_diagnostics
    assert math.isclose(d['analytic_kl_old_new'],float(kl.detach()),rel_tol=2e-5,abs_tol=2e-6)
    assert math.isclose(d['preclip_gradient_norm'],float(expected_norm),rel_tol=2e-5)
    assert math.isclose(d['approx_kl'],float((torch.expm1(logp-old_logp)-(logp-old_logp)).mean().detach()),rel_tol=3e-5)


@pytest.mark.parametrize('field,value', [('actions_log_prob',-1e5), ('returns',1e20), ('advantages',float('nan'))])
def test_nonfinite_ratio_loss_or_rollout_never_steps_adam(field,value):
    observations = obs(2)
    algorithm = factory()(observations,1)
    algorithm.num_mini_batches = 1
    collect(algorithm,observations,1)
    getattr(algorithm.storage,field).fill_(value)
    with pytest.raises((ValueError,RuntimeError),match='finite'):
        algorithm.update()
    assert not algorithm.optimizer.state


def test_single_sample_rollout_has_finite_zero_normalized_advantage():
    observations = obs(1)
    algorithm = factory()(observations,1,microbatch_size=1)
    algorithm.num_learning_epochs = 1
    collect(algorithm,observations,1)
    assert torch.equal(algorithm.storage.advantages,torch.zeros(1,1,1))
    assert all(math.isfinite(value) for value in algorithm.update().values())


def test_tail_diagnostics_and_entropy_equal_independent_sample_weighted_oracle():
    from torch.distributions import Normal
    torch.manual_seed(711)
    observations = obs()
    algorithm = factory()(observations,3,microbatch_size=2)
    algorithm.num_learning_epochs = 1
    algorithm.num_mini_batches = 4
    algorithm.schedule = 'fixed'
    for group in algorithm.optimizer.param_groups:
        group['lr'] = 0.
    collect(algorithm,observations)
    # Make saturation, old/new logp and KL visibly nonuniform across rows.
    with torch.no_grad():
        algorithm.policy.actor.action_net[-2].bias.add_(torch.linspace(-.05,.05,24))
        algorithm.policy.log_std.add_(.1)
    storage = algorithm.storage
    flattened = storage.observations.flatten(0,1)
    mu = algorithm.policy.actor(flattened['policy']).detach()
    std = algorithm.policy.log_std.exp().expand_as(mu).detach()
    z = storage.latent_actions.flatten(0,1)
    # Independent stable oracle is unnecessary here: all latents are moderate.
    normal = Normal(mu,std)
    logp = (normal.log_prob(z)-torch.log1p(-z.tanh().square())).sum(-1)
    old_logp = storage.actions_log_prob.flatten()
    logratio = logp-old_logp
    torch.manual_seed(512)
    groups = torch.tensor_split(torch.randperm(15),4)
    entropy_sum = 0.
    for indices in groups:
        epsilon = torch.randn(len(indices),24)
        current_z = mu[indices]+std[indices]*epsilon
        entropy_sum += (normal.entropy()[indices]+torch.log1p(-current_z.tanh().square())).sum().item()
    torch.manual_seed(512)
    result = algorithm.update()
    d = algorithm.last_update_diagnostics
    assert math.isclose(result['entropy'],entropy_sum/15,abs_tol=5e-6)
    assert math.isclose(d['new_log_prob_mean'],float(logp.mean()),abs_tol=5e-6)
    assert math.isclose(d['approx_kl'],float((logratio.expm1()-logratio).mean()),abs_tol=5e-6)
    assert math.isclose(d['clip_fraction'],float(((logratio.exp()-1).abs()>.2).float().mean()),abs_tol=5e-6)
    assert math.isclose(d['new_log_prob_min'],float(logp.min()),abs_tol=5e-6)
    assert math.isclose(d['new_log_prob_max'],float(logp.max()),abs_tol=5e-6)


def test_gae_done_and_advantage_match_rsl_and_update_never_resamples_actions():
    from rsl_rl.storage import RolloutStorage
    observations = obs(3)
    algorithm = factory()(observations,3)
    algorithm.num_learning_epochs = 1
    collect(algorithm,observations)
    reference = RolloutStorage('rl',3,3,observations,[24],device='cpu')
    for name in ('values','rewards','dones'):
        getattr(reference,name).copy_(getattr(algorithm.storage,name))
    with torch.no_grad():
        last_values = algorithm.policy.evaluate(observations)
    reference.compute_returns(last_values,.995,.95,normalize_advantage=True)
    torch.testing.assert_close(algorithm.storage.returns,reference.returns,atol=0,rtol=0)
    torch.testing.assert_close(algorithm.storage.advantages,reference.advantages,atol=0,rtol=0)
    def no_resample(*args,**kwargs):
        raise AssertionError('PPO must evaluate saved latents without sampling actions')
    algorithm.policy.sample_actions = no_resample
    algorithm.update()


def test_saturation_and_per_environment_any_use_bounded_commands():
    observations = obs(3)
    algorithm = factory()(observations,2,microbatch_size=2)
    algorithm.num_learning_epochs = algorithm.num_mini_batches = 1
    with torch.no_grad():
        algorithm.policy.actor.action_net[-2].weight.zero_()
        algorithm.policy.actor.action_net[-2].bias.zero_()
        algorithm.policy.actor.action_net[-2].bias[0] = 4.
    collect(algorithm,observations,2)
    algorithm.update()
    d = algorithm.last_update_diagnostics
    assert math.isclose(d['action_saturation_rate'],1/24,abs_tol=1e-8)
    assert d['action_saturation_env_any_rate'] == 1.


def test_noise_projection_runs_after_each_new_policy_adam_step_without_changing_legacy_contract():
    from go2_pvcnn.control.dh116.tabletop_policy_noise import CONTRACT
    original = deepcopy(CONTRACT)
    observations = obs(2)
    algorithm = factory()(observations,2)
    algorithm.num_learning_epochs = 1
    algorithm.num_mini_batches = 2
    collect(algorithm,observations,2)
    # A pre-step hook exercises post-Adam projection on every logical step.
    def force_outside_bounds(*args):
        with torch.no_grad():
            algorithm.policy.log_std[0] = math.log(.8)
            algorithm.policy.log_std[1] = math.log(.0001)
    algorithm.optimizer.register_step_pre_hook(force_outside_bounds)
    steps = []
    def inspect_projection(*args):
        steps.append(algorithm.policy.log_std.exp().clone())
        state = algorithm.optimizer.state[algorithm.policy.log_std]
        for name in ('exp_avg','exp_avg_sq'):
            assert (state[name][:2] == 0.).all()
    algorithm.optimizer.register_step_post_hook(inspect_projection)
    algorithm.update()
    assert len(steps) == 2
    for std in steps:
        assert .001-1e-7 <= std.min() and std.max() <= .3+1e-7
    assert CONTRACT == original
