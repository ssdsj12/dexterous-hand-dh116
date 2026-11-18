"""Native-reward assembly stubs; these tests do not run Isaac."""
import copy
from types import SimpleNamespace

import pytest
import torch

from test_surface_stable_hold_rewards import qualified_task
from go2_pvcnn.control.dh116 import tabletop_surface_reward_ablation as base
from go2_pvcnn.control.dh116 import tabletop_lift_reward_experiment as new
from go2_pvcnn.control.dh116 import tabletop_surface_exploration as surface


def task(profile='R0'):
    t = qualified_task()
    from go2_pvcnn.control.dh116.tabletop_effective_pad import DISTAL_LINKS
    t.surface_world = SimpleNamespace(hands=[SimpleNamespace(body_names=list(DISTAL_LINKS),
        data=SimpleNamespace(body_com_pos_w=torch.zeros(2, 5, 3, dtype=torch.float64),
            body_com_vel_w=torch.zeros(2, 5, 6, dtype=torch.float64))) for _ in range(2)])
    t.surface_reward_profile = 'support_deficit'
    t.m['bottom'].fill_(.36)
    t.m['object_velocity'].zero_()
    # Native pad vectors are force on the hand; the object receives -vector.
    t.m['effective_pad_vectors_w'][:, :, :2, 2] = -.4
    t.m['vertical_hand_force_n'].fill_(.8)
    base.enable(t)
    new.install_lift_reward_experiment(t, profile)
    t.m['bottom'].fill_(.3765)
    tracker = t.surface_early_hold
    tracker.window_completed = True
    tracker.window_native_valid.fill_(True)
    tracker._window_good.fill_(True)
    tracker.slip_observable.fill_(True)
    tracker.reward_drift_m.zero_()
    return t


def apply(t):
    pieces = {}
    base.apply_reward(t, pieces, torch.ones(2, dtype=torch.bool), 1/480,
        t.m['vertical_hand_force_n'], torch.ones(2, dtype=torch.float64))
    return pieces


def test_r0_installation_is_an_exact_noop_for_base_reward_and_contract():
    t = task()
    assert not hasattr(t.surface_reward_ablation, 'lift_quality_experiment')
    before = copy.deepcopy(t.surface_reward_ablation.__dict__)
    old_contract = {}; surface.apply_contract(old_contract, t)
    assert new.install_lift_reward_experiment(t, 'R0') is None
    assert t.surface_reward_ablation.__dict__.keys() == before.keys()
    for k, v in before.items():
        if isinstance(v, torch.Tensor): torch.testing.assert_close(v, t.surface_reward_ablation.__dict__[k])
        else: assert v == t.surface_reward_ablation.__dict__[k]
    fresh_contract = {}; surface.apply_contract(fresh_contract, t)
    assert fresh_contract == old_contract and 'surface_lift_reward_experiment' not in fresh_contract
    assert set(apply(t)) == set(base.CUMULATIVE_TERMS['support_deficit'])


def test_hold_assembly_uses_one_completed_native_window_not_one_physics_substep():
    t = task('R1')
    for _ in range(7):
        t.surface_early_hold.window_completed = False
        assert not apply(t)[new.HOLD].any()
    t.surface_early_hold.window_completed = True
    out = apply(t)
    torch.testing.assert_close(out[new.HOLD], torch.full((2,), .5/60, dtype=torch.float64))
    assert set(out) == set(base.CUMULATIVE_TERMS['support_deficit']) | {new.HOLD}


def test_experiment_reset_follows_actual_selected_physical_reset():
    t = task('R2')
    observer = t.surface_reward_ablation.lift_quality_experiment
    apply(t)
    new_bottom = torch.tensor([.42, .36], dtype=torch.float64)
    t.surface_reward_ablation.reset(torch.tensor([0]), new_bottom)
    torch.testing.assert_close(observer.reference, torch.tensor([.42, .36], dtype=torch.float64))
    t.m['bottom'] = torch.tensor([.421, .3765], dtype=torch.float64)
    assert not apply(t)[new.UP].any()


def test_experiment_metadata_does_not_change_observation_action_or_success():
    t = task()
    original_shape = t.observe().shape
    original_action = t.action_size
    original = {}; surface.apply_contract(original, t)
    new.install_lift_reward_experiment(t, 'R1')
    actual = {}; surface.apply_contract(actual, t)
    assert actual.pop('surface_lift_reward_experiment') == new.experiment_contract('R1')
    assert actual == original
    assert t.observe().shape == original_shape and t.action_size == original_action == 24
    assert actual['success_dwell_s'] == 5. and actual['success_slip_m'] == .0005


def test_failed_native_rows_never_collect_positive_hold():
    t = task('R1'); t.failure[:] = 1
    assert not apply(t)[new.HOLD].any()


def test_downward_squeezing_cannot_be_mistaken_for_upward_support():
    t = task('R1')
    t.m['effective_pad_vectors_w'][:, :, :2, 2] = .4
    assert not apply(t)[new.HOLD].any()


def test_non_base_profile_cannot_silently_mix_reward_experiments():
    t = task(); t.surface_reward_profile = 'v3_reference'
    with pytest.raises(ValueError): new.install_lift_reward_experiment(t, 'R1')


def test_failed_terminal_drop_uses_physical_validity_and_inactive_rows_receive_nothing():
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import add_feedback
    t = task('R5')
    def assembled(live):
        pieces = {}
        add_feedback(t, pieces, live, 1/480,
            hand_lift_force_n=t.m['vertical_hand_force_n'],
            object_weight_n=torch.ones(2, dtype=torch.float64))
        return pieces
    assembled(torch.ones(2, dtype=torch.bool))
    t.m['bottom'] -= .001
    t.failure[0] = 1  # Finite last physical sample, terminal before reward assembly.
    pieces = assembled(torch.tensor([True, False]))
    assert pieces[new.DROP][0] < 0.
    assert pieces[new.HOLD][0] == pieces[new.UP][0] == 0.
    assert all(pieces[name][1] == 0. for name in new.FACTORS['R5'])


@pytest.mark.parametrize('failure_substep', [1, 2, 3, 4, 5, 6, 7, 8])
def test_failed_terminal_drop_survives_real_native_tracker_active_mask(failure_substep):
    """TensorTask's Hold tracker receives active=False on a failed final sample."""
    t = task('R5')
    tracker = t.surface_early_hold
    tracker.reset(torch.arange(2))
    tracker.window_completed = False
    def consume(bottom, *, live=True, failure=False, invalid_sample=False):
        t.m['bottom'].fill_(bottom)
        raw_valid = torch.tensor([live and not invalid_sample, False])
        raw_live = torch.tensor([live, False])
        if failure: t.failure[0] = 1
        active = raw_valid & (t.failure == 0)
        m = t.m
        center = m['object_relative'] + m['wrist_pose'][..., :3].mean(1)
        tracker.update(torch.cat((center, m['object_quat']), -1), m['wrist_pose'],
            m['effective_pad_loads_n'], m['effective_pad_vectors_w'],
            m['effective_pad_depth_m'], m['rejected_contact_loads_n'].sum(-1),
            m['bottom'], m['object_velocity'], active, 1/480)
        pieces = {}
        base.apply_reward(t, pieces, active, 1/480, m['vertical_hand_force_n'],
            torch.ones(2, dtype=torch.float64), milestone_valid=active,
            physical_valid=raw_valid, physical_live=raw_live)
        return pieces
    for _ in range(8): consume(.3765)
    for _ in range(failure_substep-1): consume(.3765)
    pieces = consume(.3755, failure=True)
    # Exactly this physical sample's dt, even when failure precedes the flush.
    assert pieces[new.DROP][0] == pytest.approx(-new.WEIGHTS[new.DROP]/480)
    assert pieces[new.HOLD][0] == pieces[new.UP][0] == 0.
    assert all(pieces[name][1] == 0. for name in new.FACTORS['R5'])
    for _ in range(failure_substep, 9):
        pieces = consume(.50, live=False)  # Unrelated reset-tail state must not pay again.
        assert all(not pieces[name].any() for name in new.FACTORS['R5'])


def test_invalid_measured_substep_breaks_drop_bridge_even_if_window_ends_finite():
    t = task('R3')
    apply(t)  # Establish the previous valid completed-window height.
    tracker = t.surface_early_hold
    tracker.window_completed = False
    pieces = {}
    base.apply_reward(t, pieces, torch.zeros(2, dtype=torch.bool), 1/480,
        t.m['vertical_hand_force_n'], torch.ones(2, dtype=torch.float64),
        physical_valid=torch.zeros(2, dtype=torch.bool),
        physical_live=torch.ones(2, dtype=torch.bool))
    t.m['bottom'] -= .002
    tracker.window_completed = True
    assert not apply(t)[new.DROP].any()
    t.m['bottom'] -= .002
    assert (apply(t)[new.DROP] < 0.).all()  # A new contiguous measured descent.


@pytest.mark.parametrize('failure_substep', [1, 2, 7, 8])
def test_tensor_task_consume_keeps_terminal_drop_and_never_charges_reset_tail(failure_substep):
    from go2_pvcnn.control.dh116.tabletop_tensor_task import FAILURE_CODES
    t = task('R5')
    surface.enable_stable_hold_rewards(t)
    def consume(bottom, failed=False):
        m = {k: value.clone() for k, value in t.m.items()}
        m['bottom'][0] = bottom
        if failed: m['joint_failure'][0] = FAILURE_CODES['joint_limit']
        before = t.terms[new.DROP].clone()
        t.consume(m, ~t.done, 1/480, vertical_force_n=torch.ones(2, dtype=torch.float64),
            lift_vertical_force_n=torch.ones(2, dtype=torch.float64),
            hand_lift_force_n=torch.ones(2, 2, dtype=torch.float64),
            object_weight_n=torch.ones(2, dtype=torch.float64))
        return t.terms[new.DROP]-before
    for _ in range(8+failure_substep-1):
        assert not consume(.3765).any()
    cost = consume(.3755, failed=True)
    assert t.done.tolist() == [True, False]
    assert cost[0] == pytest.approx(-new.WEIGHTS[new.DROP]/480)
    for _ in range(8): assert not consume(.34).any()


def test_explicit_physical_validity_does_not_change_r0_terms():
    t = task()
    expected, actual = {}, {}
    valid = torch.ones(2, dtype=torch.bool)
    kwargs = dict(milestone_valid=valid)
    base.apply_reward(t, expected, valid, 1/480,
        t.m['vertical_hand_force_n'], torch.ones(2, dtype=torch.float64), **kwargs)
    t = task()
    base.apply_reward(t, actual, valid, 1/480,
        t.m['vertical_hand_force_n'], torch.ones(2, dtype=torch.float64),
        physical_valid=torch.zeros(2, dtype=torch.bool), **kwargs)
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.parametrize('invalid_action', [False, True])
def test_native_reset_tail_cannot_rebase_new_reward_but_keeps_r0_reference(invalid_action):
    from test_dh116_tensor_env import TerminalWorld
    from test_dh116_tensor_task import template
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
    class DriftingResetWorld(TerminalWorld):
        def advance(self, *args):
            self.tick += 1
            m = self.m
            if self.resets:
                m['bottom'][0] += .0001  # Real inactive reset-tail evolution.
            return m
    t = TensorTask(template(), 2, 'cpu')
    t.continuous_motion = invalid_action
    w = DriftingResetWorld(t)
    env = TensorEnv(w, t)
    t.surface_reward_ablation = base.RewardAblation('support_deficit', t.m['bottom'])
    observer = new.LiftRewardExperiment('R2', t.m['bottom'])
    t.surface_reward_ablation.lift_quality_experiment = observer
    other_reference = observer.reference[1].clone()
    actions = torch.zeros(2, 20)
    if invalid_action:
        actions[0, 0] = float('nan')
    else:
        t.elapsed[0] = 4.01
    _, _, done, _ = env.step(actions)
    assert done.tolist() == [True, False] and w.resets == [[0]]
    assert w.m['bottom'][0] > .30
    assert observer.reference[0] == pytest.approx(.30)
    assert observer.reference[1] == other_reference
    assert t.surface_reward_ablation.reference[0] == w.m['bottom'][0]
    assert t.surface_reward_ablation.reference[0] != observer.reference[0]
