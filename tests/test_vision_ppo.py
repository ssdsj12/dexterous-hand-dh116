"""CPU rollout and microbatch PPO equivalence against installed RSL PPO."""
from copy import deepcopy
import importlib
import importlib.util
import math

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from go2_pvcnn.control.dh116.tabletop_policy_noise import install_noise_projection


MODULE = 'go2_pvcnn.control.dh116.tabletop_vision_ppo'


def memory_ppo():
    assert importlib.util.find_spec(MODULE), 'bounded-memory vision PPO is missing'
    return importlib.import_module(MODULE)


def visual_obs(n):
    from go2_pvcnn.control.dh116.tabletop_vision_student import POLICY_DIM
    policy = torch.zeros(n, POLICY_DIM)
    # Four history frames contain real finite coordinate samples and valid masks.
    frames = policy.reshape(n, 4, 961)
    frames[..., :768] = torch.randn(n, 4, 768) * .05
    frames[..., 768:896] = 1.
    frames[..., 896:] = torch.randn(n, 4, 65) * .1
    return TensorDict(dict(policy=policy, critic=torch.randn(n, 371)), [n])


def collect(algorithm, obs, steps):
    with torch.inference_mode():
        for step in range(steps):
            actions = algorithm.act(obs)
            algorithm.process_env_step(obs, 1-actions.square().sum(-1),
                                       torch.full((len(obs),), step == 1, dtype=torch.bool), {})
        algorithm.compute_returns(obs)


def test_cpu_float32_rollout_and_bounded_actor_batches_retain_raw_likelihoods():
    memory_ppo()
    from go2_pvcnn.control.dh116.tabletop_vision_student import make_vision_algorithm
    torch.set_num_threads(2)
    obs = visual_obs(11)
    algorithm = make_vision_algorithm(obs, steps=4, microbatch_size=3, inference_batch_size=3)
    observed = []
    handle = algorithm.policy.actor.register_forward_pre_hook(
        lambda module, args: observed.append(len(args[0])))
    with torch.inference_mode():
        raw = algorithm.act(obs)
        # Verify full-batch recorded probabilities with the same frozen policy.
        algorithm.policy.act(obs)
        expected = algorithm.policy.get_actions_log_prob(raw)
        torch.testing.assert_close(algorithm.transition.actions_log_prob, expected)
        assert algorithm.transition.actions.shape == (11, 24)
    observed.clear()
    algorithm.process_env_step(obs, torch.ones(11), torch.zeros(11, dtype=torch.bool), {})
    for _ in range(3):
        with torch.inference_mode():
            algorithm.act(obs)
        algorithm.process_env_step(obs, torch.ones(11), torch.zeros(11, dtype=torch.bool), {})
    algorithm.compute_returns(obs)
    storage = algorithm.storage
    for key in ('policy', 'critic'):
        assert storage.observations[key].device.type == 'cpu'
        assert storage.observations[key].dtype == torch.float32
        torch.testing.assert_close(storage.observations[key][0], obs[key], atol=0., rtol=0.)
    for name in ('actions', 'values', 'returns', 'advantages', 'actions_log_prob', 'mu', 'sigma'):
        assert getattr(storage, name).device.type == 'cpu'
        assert getattr(storage, name).dtype == torch.float32
    algorithm.update()
    handle.remove()
    assert observed and max(observed) <= 3
    assert storage.step == 0


def test_microbatch_gradients_match_full_batch_rsl_update_and_global_gae():
    memory_ppo()
    from go2_pvcnn.control.dh116.tabletop_vision_student import make_vision_algorithm
    torch.set_num_threads(2)
    torch.manual_seed(517)
    obs = visual_obs(4)
    actual = make_vision_algorithm(obs, steps=4, microbatch_size=3, inference_batch_size=3)
    actual.num_learning_epochs = 2
    actual.num_mini_batches = 2
    reference = PPO(deepcopy(actual.policy), num_learning_epochs=2, num_mini_batches=2,
                    clip_param=.2, gamma=.995, lam=.95, value_loss_coef=.5,
                    entropy_coef=.005, learning_rate=3e-4, max_grad_norm=.5,
                    schedule='adaptive', desired_kl=.02, device='cpu')
    reference.policy_noise_contract = deepcopy(actual.policy_noise_contract)
    install_noise_projection(reference)
    collect(actual, obs, 4)
    reference.storage = deepcopy(actual.storage)
    # Reference GAE recomputation starts from the exact frozen state values.
    with torch.inference_mode():
        reference.compute_returns(obs)
    torch.testing.assert_close(actual.storage.returns, reference.storage.returns)
    torch.testing.assert_close(actual.storage.advantages, reference.storage.advantages)
    torch.manual_seed(991)
    expected_losses = reference.update()
    torch.manual_seed(991)
    actual_losses = actual.update()
    for key in expected_losses:
        assert math.isclose(actual_losses[key], expected_losses[key], rel_tol=3e-5, abs_tol=3e-6)
    for name, value in reference.policy.state_dict().items():
        torch.testing.assert_close(actual.policy.state_dict()[name], value, atol=4e-6, rtol=4e-5)
    assert actual.learning_rate == reference.learning_rate
