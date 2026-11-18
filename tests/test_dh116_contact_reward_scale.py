"""Requested percentage reward, exercised through scalar and vector tasks."""
import numpy as np
import pytest

from test_dh116_batched_task import make_env, BatchTask


@pytest.mark.parametrize('batched', [False, True])
def test_contact_reward_tracks_ten_newtons_without_chatter_profit(batched):
    env = make_env(0.)
    action = np.r_[np.zeros(18), -1., -1.]
    # Zero -> 10 N earns 5; saturation and contact loss cannot farm reward.
    totals = [.5, 1., 5., 10., 12., 5., 0.]
    expected = [.25, .25, 2., 2.5, 0., -2.5, -2.5]
    actual = []
    for total in totals:
        env.backend.force = total / 4.
        if batched:
            task = BatchTask([env])
            def advance(offset, command, allowed, live):
                env.backend.advance(offset[0], command[0],
                    palm_motion_allowed=bool(allowed[0]),
                    palm_action=task.palm_actions[0], grip_action=np.ones(2))
                return task.stack_measurements([env.backend.measure()])
            task.step(action[None, :], advance)
        else:
            env.step(action)
        actual.append(env.log[-1]['terms']['contact'])
    assert actual == pytest.approx(expected)
    assert sum(actual) == pytest.approx(0.)


def test_reward_only_checkpoint_transfer_is_explicit_and_preserves_state(tmp_path):
    from stable_baselines3 import PPO
    from test_dh116_tabletop_training import SyntheticEnv
    from go2_pvcnn.control.dh116.tabletop_training import save_checkpoint, load_checkpoint
    env = SyntheticEnv()
    model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=1, device='cpu')
    model.learn(8)
    old = dict(reward='object_support_progress_v3', physics_hz=480)
    new = dict(old, reward='object_support_progress_v4_10n_5')
    path = tmp_path/'model.zip'
    save_checkpoint(model, path, old, stage='full')
    with pytest.raises(ValueError, match='contract'):
        load_checkpoint(path, env, new)
    resumed, metadata = load_checkpoint(path, env, new, allow_contact_reward_change=True)
    assert resumed.num_timesteps == model.num_timesteps
    assert resumed._n_updates == model._n_updates
    assert metadata['reward_change'] == dict(previous=old['reward'], current=new['reward'])
    assert resumed.policy.optimizer.state_dict()['state'].keys() == model.policy.optimizer.state_dict()['state'].keys()
    with pytest.raises(ValueError, match='contract'):
        load_checkpoint(path, env, dict(new, physics_hz=240), allow_contact_reward_change=True)
