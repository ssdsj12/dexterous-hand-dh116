import copy

import pytest
import torch

from test_surface_exploration import surface_task
from go2_pvcnn.control.dh116 import tabletop_surface_exploration as mode
from go2_pvcnn.control.dh116 import tabletop_surface_imitation as bc


def qualified_task():
    task, _ = surface_task()
    task.surface_stable_hold_rewards = True
    task.m['bottom'].fill_(.391)
    task.m['effective_pad_loads_n'][:, :, :2] = 1.
    task.m['effective_pad_vectors_w'][:, :, 0, 0] = 1.
    task.m['effective_pad_vectors_w'][:, :, 1, 0] = -1.
    task.m['effective_pad_valid'][:, :, :2] = True
    return task


@pytest.mark.parametrize('bad', ['ordinary_contact', 'penetration'])
def test_ineligible_grasp_cannot_collect_stable_mode_lift_reward(bad):
    task = qualified_task()
    if bad == 'ordinary_contact':
        task.m['rejected_contact_loads_n'].fill_(1.)
    else:
        task.m['effective_pad_depth_m'].fill_(.00081)
    assert not mode.reward_rates(task)['dex_lift'].any()


def test_continuous_native_hold_progress_is_rewarded_without_new_motion_gate():
    task = qualified_task()
    task.surface_lift_success._duration[:] = torch.tensor([1., 4.])
    rates = mode.reward_rates(task)
    assert rates['surface_hold_progress'][1] > rates['surface_hold_progress'][0] > 0
    before = task.command.clone()
    mode.reward_rates(task)
    assert torch.equal(before, task.command)
    task.surface_lift_success._duration.zero_()
    assert not mode.reward_rates(task)['surface_hold_progress'].any()


def test_only_declared_training_objective_can_reuse_teacher_observations():
    from test_surface_imitation import dataset
    sample = dataset()
    current = copy.deepcopy(sample['contract'])
    current['surface_training_objective'] = copy.deepcopy(mode.STABLE_HOLD_REWARD_CONTRACT)
    bc.validate_dataset(sample, current)
    current['surface_training_objective']['success_hold_s'] = .5
    with pytest.raises(ValueError, match='contract'):
        bc.validate_dataset(sample, current)


def test_real_slip_cost_outweighs_lift_before_the_success_height():
    task = qualified_task()
    mode.enable_stable_hold_rewards(task)
    task.m['bottom'].fill_(.38)  # 20 mm, below the actual 30 mm success.
    task.surface_slip_tracking._drift[:] = torch.tensor([.00001, .0005])
    task.surface_slip_tracking._physical_window_valid.fill_(True)
    before = task.command.clone()
    rates = mode.reward_rates(task)
    assert rates['surface_slip_window'][0] > -.1
    assert (rates['dex_lift'] + rates['surface_slip_window'])[1] < 0
    assert not task.surface_lift_success.duration.any()
    assert torch.equal(task.command, before)


def test_tracking_from_three_mm_does_not_grant_thirty_mm_success():
    from test_surface_success import scene, advance
    task = qualified_task()
    mode.enable_stable_hold_rewards(task)
    data = scene()
    data['bottom'].fill_(.364)
    data['object_pose'][:, 2] = .372
    tracking = advance(task.surface_slip_tracking, data, 2400)
    actual = advance(task.surface_lift_success, data, 2400)
    assert tracking['duration'].min() >= 5 - 1e-9
    assert not actual['duration'].any() and not actual['event'].any()
    task.surface_slip_tracking.reset(torch.tensor([0]))
    assert task.surface_slip_tracking.duration[0] == 0
    assert task.surface_slip_tracking.duration[1] >= 5 - 1e-9


def test_tracking_slip_penalty_is_absent_at_rest_and_in_legacy_mode():
    task = qualified_task()
    mode.enable_stable_hold_rewards(task)
    task.surface_slip_tracking._drift.fill_(.001)
    task.m['bottom'].fill_(.362)
    assert not mode.reward_rates(task)['surface_slip_window'].any()
    task.surface_stable_hold_rewards = False
    assert 'surface_slip_window' not in mode.reward_rates(task)


def test_ineligible_speed_window_does_not_fabricate_slip_from_zeroed_coordinates():
    from test_surface_success import scene, advance
    task = qualified_task()
    mode.enable_stable_hold_rewards(task)
    task.m['bottom'].fill_(.38)
    data = scene()
    data['bottom'].fill_(.38)
    advance(task.surface_slip_tracking, data)
    # The whole object/hand system translates without relative slip. Speed is
    # just above the physical hold limit, so the hold window must be rejected.
    data['object_pose'][:, 0] += .00051
    data['wrist_pose'][:, :, 0] += .00051
    data['object_velocity'][:, 0] = .031
    result = advance(task.surface_slip_tracking, data)
    assert not result['duration'].any()
    assert not mode.reward_rates(task)['surface_slip_window'].any()
