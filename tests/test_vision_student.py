"""CPU regressions for a sensor-only PointNet actor and asymmetric PPO."""
from io import BytesIO
import importlib
import importlib.util

import pytest
import torch
from tensordict import TensorDict

MODULE = 'go2_pvcnn.control.dh116.tabletop_vision_student'


def api():
    assert importlib.util.find_spec(MODULE), 'vision student interface is missing'
    return importlib.import_module(MODULE)


def packet(module, n=4):
    wrists = torch.zeros(n, 2, 7)
    wrists[..., 3] = 1.
    return dict(
        points_in_hands=torch.zeros(n, 2, module.POINTS, 3),
        point_valid=torch.zeros(n, module.POINTS, dtype=torch.bool),
        q=torch.zeros(n, 12), qd=torch.zeros(n, 12), wrist_poses=wrists,
        previous_action=torch.zeros(n, 24), dt=1/30,
        image_valid=torch.ones(n, dtype=torch.bool), object_valid=torch.zeros(n, dtype=torch.bool),
    )


def observations(module, n=4):
    history = module.SensorHistory(n)
    policy = history.update(**packet(module, n))
    return TensorDict(dict(policy=policy, critic=torch.randn(n, 371)), batch_size=[n])


def test_sensor_history_clears_selected_rows_and_has_no_privileged_packet_input():
    module = api()
    history = module.SensorHistory(2)
    values = packet(module, 2)
    values['q'].fill_(.3)
    for _ in range(module.HISTORY):
        result = history.update(**values)
    assert result.shape == (2, 3844)
    result = history.update(**values, reset_mask=torch.tensor([True, False]))
    frames = result.reshape(2, module.HISTORY, module.FRAME_DIM)
    assert not frames[0, :-1].any()
    assert frames[1, :-1].any()
    for forbidden in ('task', 'world', 'object_pose_gt'):
        with pytest.raises(TypeError):
            history.update(**values, **{forbidden: torch.zeros(1)})


def test_masked_padding_and_empty_clouds_do_not_change_actor_embedding():
    module = api()
    actor = module.VisionStudentActor()
    inputs = torch.zeros(2, 3844)
    baseline = actor(inputs)
    changed = inputs.reshape(2, module.HISTORY, module.FRAME_DIM).clone()
    changed[:, :, :2 * module.POINTS * 3] = float('nan')
    torch.testing.assert_close(actor(changed.flatten(1)), baseline, atol=0., rtol=0.)
    assert torch.isfinite(baseline).all()


def test_actor_ignores_critic_and_gradients_flow_through_valid_camera_points():
    module = api()
    obs = observations(module)
    frame = obs['policy'].reshape(4, module.HISTORY, module.FRAME_DIM)
    frame[:, -1, :2 * module.POINTS * 3] = torch.randn(4, 2 * module.POINTS * 3) * .03
    frame[:, -1, 2 * module.POINTS * 3:2 * module.POINTS * 3 + module.POINTS] = 1.
    algorithm = module.make_vision_algorithm(obs, steps=2)
    changed = obs.clone()
    changed['critic'] = torch.randn_like(changed['critic']) * 1000.
    first = algorithm.policy.act_inference(obs)
    second = algorithm.policy.act_inference(changed)
    torch.testing.assert_close(first, second, atol=0., rtol=0.)
    assert first.shape == (4, 24) and first.abs().max() <= 1.
    loss = (first - .25).square().mean()
    loss.backward()
    assert sum(p.grad.abs().sum() for p in algorithm.policy.actor.point_net.parameters()
               if p.grad is not None) > 0.
    assert all(p.grad is None for p in algorithm.policy.critic.parameters())
    torch.testing.assert_close(algorithm.policy.log_std.exp(), torch.full((24,), .1))


def test_two_step_ppo_updates_save_and_load_exactly_with_bounded_noise():
    module = api()
    torch.manual_seed(123)
    obs = observations(module)
    algorithm = module.make_vision_algorithm(obs, steps=2)
    before = {name: value.clone() for name, value in algorithm.policy.actor.state_dict().items()}
    for _ in range(2):
        with torch.inference_mode():
            for step in range(2):
                action = algorithm.act(obs)
                assert action.shape == (4, 24)
                algorithm.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
            algorithm.compute_returns(obs)
        losses = algorithm.update()
        assert all(torch.isfinite(torch.as_tensor(value)).all() for value in losses.values())
    assert any(not torch.equal(before[k], value) for k, value in algorithm.policy.actor.state_dict().items())
    checkpoint = BytesIO()
    torch.save(dict(policy=algorithm.policy.state_dict(), optimizer=algorithm.optimizer.state_dict()), checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = module.make_vision_algorithm(obs, steps=2)
    restored.policy.load_state_dict(saved['policy'])
    restored.optimizer.load_state_dict(saved['optimizer'])
    torch.testing.assert_close(restored.policy.act_inference(obs), algorithm.policy.act_inference(obs),
                               atol=0., rtol=0.)
    assert restored.optimizer.state_dict()['state'].keys() == saved['optimizer']['state'].keys()
    std = restored.policy.log_std.exp()
    assert (std >= .001).all() and (std <= .3).all()
