import torch
from test_dh116_tensor_env import World
from test_dh116_tensor_task import template
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_grasp_acquisition import potential


def make_env():
    task = TensorTask(template(), 2, 'cpu')
    task.m['object_quat'][:] = torch.tensor([1.,0.,0.,0.])
    world = World(task)
    env = TensorEnv(world, task)
    env.grasp_acquisition = True
    env.acquisition_radius_m = .008
    env.acquisition_length_m = .5
    return env, task, world


def test_terminal_shaping_is_settled_before_auto_reset():
    env, task, world = make_env()
    before = potential(task.m)
    _, reward, done, extra = env.step(torch.zeros((2,20)))
    assert done[0]
    torch.testing.assert_close(extra['terms']['grasp_acquisition'][0], -before[0])
    assert torch.isfinite(reward).all()
    assert env.episodes[0] == 1


def test_nonterminal_shaping_uses_current_state_without_rollout_reset():
    env, task, world = make_env()
    world.tick = 100
    before = potential(task.m)
    _, _, done, extra = env.step(torch.zeros((2,20)))
    assert not done.any()
    torch.testing.assert_close(extra['terms']['grasp_acquisition'], .995*potential(task.m)-before)


def test_shaping_adds_only_its_reported_term_to_original_reward():
    shaped, _, shaped_world = make_env()
    baseline, _, base_world = make_env()
    baseline.grasp_acquisition = False
    shaped_world.tick = base_world.tick = 100
    actions = torch.zeros((2,20))
    _, shaped_reward, shaped_done, extra = shaped.step(actions)
    _, base_reward, base_done, _ = baseline.step(actions)
    torch.testing.assert_close(shaped_done, base_done)
    torch.testing.assert_close(shaped_reward-base_reward, extra['terms']['grasp_acquisition'])


def test_first_step_after_reset_uses_new_episode_state_potential():
    env, task, world = make_env()
    env.step(torch.zeros((2,20)))
    world.tick = 100
    before = potential(task.m)
    _, _, done, extra = env.step(torch.zeros((2,20)))
    assert not done.any()
    torch.testing.assert_close(extra['terms']['grasp_acquisition'], .995*potential(task.m)-before)
