import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import RewardAblation
from go2_pvcnn.control.dh116.tabletop_surface_exploration import reward_rates, enable_stable_hold_rewards
from test_surface_stable_hold_rewards import qualified_task


def step(r, *, bottom=.364, qualified=True, early=True, duration=0., drift=0.,
         relative=0., support=1.2, weight=1., dt=1/480, window=False,
         native=True, slip_observable=True, valid=True, smoothness=0.):
    x = lambda v: torch.full_like(r.reference, v)
    mask = lambda v: x(v).bool()
    return r.update(bottom=x(bottom), qualified=mask(qualified), early_window_good=mask(early),
        early_duration=x(duration), drift=x(drift), relative_speed=x(relative),
        support=x(support), weight=x(weight), valid=mask(valid), dt=dt,
        window_completed=window, window_native_valid=mask(native),
        slip_observable=mask(slip_observable), smoothness=x(smoothness))


def test_height_replaces_legacy_only_and_pays_a_high_water_mark():
    task = qualified_task()
    enable_stable_hold_rewards(task)
    task.m['bottom'].fill_(float(task.surface_reference_bottom_m) + .03)
    historical = reward_rates(task)
    assert historical['dex_lift'].max() > 0
    task.surface_reward_profile = 'v3_lift_only'
    replacement = reward_rates(task)
    assert not replacement['dex_lift'].any()
    assert all(torch.equal(historical[k], replacement[k]) for k in historical if k != 'dex_lift')
    r = RewardAblation('v3_lift_only', torch.tensor([.36], dtype=torch.float64))
    assert step(r, bottom=.39)['surface_lift_progress'].item() == pytest.approx(8 * .0295 / .03)
    for bottom in (.39, .37, .39):
        assert step(r, bottom=bottom)['surface_lift_progress'].item() == 0
    assert step(r, bottom=.3905)['surface_lift_progress'].item() == pytest.approx(8 * .0005 / .03)
    r.reset([0], torch.tensor([.36], dtype=torch.float64))
    assert step(r, bottom=.39, qualified=False)['surface_lift_progress'].item() == 0
    assert step(r, bottom=.39)['surface_lift_progress'].item() > 0


@pytest.mark.parametrize('drift', [.0001, .0004, .0006, .001])
def test_slip_uses_fresh_valid_native_drift_even_after_hold_breaks(drift):
    r = RewardAblation('v3_slip_only', torch.tensor([.36], dtype=torch.float64))
    name = 'surface_slip_window'
    assert step(r, early=False, drift=drift, window=True)[name].item() == pytest.approx(
        -24 * min((drift / .0005)**2, 4) / 60)
    assert step(r, early=False, drift=drift)[name].item() == 0
    assert step(r, early=False, drift=drift, window=True, native=False)[name].item() == 0
    assert step(r, early=False, drift=drift, window=True, slip_observable=False)[name].item() == 0


def test_contact_break_counts_two_completed_windows_and_requires_rearming():
    r = RewardAblation('v3_contact_break_only', torch.tensor([.36], dtype=torch.float64))
    name = 'surface_contact_break'
    step(r, duration=.1, window=True)
    for idx in range(40):
        out = step(r, early=False, window=idx % 8 == 7)
        assert out[name].item() == (-4 if idx == 15 else 0)
    step(r, duration=.1, window=True)
    step(r, early=False, window=True)
    step(r, early=False, native=False, window=True)
    assert step(r, early=False, window=True)[name].item() == 0
    assert step(r, early=False, window=True)[name].item() == -4
    step(r, duration=.1, window=True)
    assert all(step(r, early=False, window=True)[name].item() == 0 for _ in range(4))


def test_early_hold_has_per_episode_budget_and_selective_reset():
    r = RewardAblation('early_hold', torch.tensor([.36, .36], dtype=torch.float64))
    name = 'surface_early_hold_rate'
    assert not step(r, duration=.1)[name].any()
    out = step(r, duration=.1, window=True)
    assert out[name].tolist() == pytest.approx([2/60, 2/60])
    assert out['surface_hold_milestone'].tolist() == [1., 1.]
    out = step(r, duration=.1, window=True)
    assert out['surface_hold_milestone'].tolist() == [0., 0.]
    for _ in range(400):
        step(r, window=True)
    assert r.hold_paid.tolist() == [10., 10.]
    r.reset([0], r.reference.clone())
    assert step(r, window=True)[name].tolist() == pytest.approx([2/60, 0.])


def test_support_uses_signed_runtime_force_and_weight_independently_of_hold():
    r = RewardAblation('v3_support_only', torch.tensor([.36], dtype=torch.float64))
    name = 'surface_support_deficit'
    assert step(r, early=False, support=.8, weight=2.)[name].item() == pytest.approx(-1.5 * .6 / 480)
    assert step(r, support=-1.)[name].item() == pytest.approx(-1.5 / 480)
    for options in ({'support': 1.5}, {'weight': 0.}, {'bottom': .36}, {'qualified': False}):
        assert step(r, **options)[name].item() == 0.


def test_smoothness_is_independent_and_invalid_rows_are_masked():
    r = RewardAblation('v3_smoothness_only', torch.tensor([.36], dtype=torch.float64))
    name = 'surface_executed_smoothness'
    assert step(r, smoothness=.3)[name].item() == pytest.approx(-.3 / 480)
    assert step(r, smoothness=float('nan'))[name].item() == 0.
    assert step(r, smoothness=.3, valid=False)[name].item() == 0.


def test_each_factor_replaces_only_its_declared_legacy_term():
    from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import (
        SINGLE_TERMS, ablation_contract, disabled_legacy_terms)
    task = qualified_task()
    enable_stable_hold_rewards(task)
    task.surface_lift_success._duration.fill_(2.)
    task.surface_slip_tracking._drift.fill_(.0004)
    task.surface_slip_tracking._physical_window_valid.fill_(True)
    baseline = reward_rates(task)
    assert baseline['surface_hold_progress'].min() > 0
    assert baseline['surface_slip_window'].max() < 0
    task.surface_reward_profile = 'v3_reference'
    assert all(torch.equal(value, reward_rates(task)[name]) for name, value in baseline.items())
    for factor, term in SINGLE_TERMS.items():
        profile = f'v3_{factor}_only'
        task.surface_reward_profile = profile
        contract = ablation_contract(profile)
        assert contract['kind'] == 'one_factor'
        assert contract['enabled_terms'] == [term]
        for name, value in reward_rates(task).items():
            expected = torch.zeros_like(value) if name in disabled_legacy_terms(profile) else baseline[name]
            assert torch.equal(value, expected), (profile, name)


def test_height_factor_masks_legacy_milestone_credit_at_reward_assembly():
    from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import enable, apply_reward
    task = qualified_task()
    for profile, expected in (('v3_lift_only', 0.), ('v3_support_only', 3.)):
        task.surface_reward_profile = profile
        enable(task)
        pieces = {'lift_milestone': torch.full_like(task.reward, 3.)}
        apply_reward(task, pieces, ~task.done, 1/480,
                     torch.zeros_like(task.m['vertical_hand_force_n']), torch.ones_like(task.reward))
        assert pieces['lift_milestone'].tolist() == [expected, expected]


def test_anchor_diagnostics_keep_real_slip_in_failed_window_without_altering_success():
    from test_surface_success import scene, advance
    from go2_pvcnn.control.dh116.tabletop_surface_success import SurfaceLiftSuccess
    tracker = SurfaceLiftSuccess(2, 'cpu', .30, reward_diagnostics=True)
    original = SurfaceLiftSuccess(2, 'cpu', .30)
    data = scene()
    for drift in (0., .0006, .0001):
        data['object_pose'][:, 0] += drift
        if drift == .0001:
            data['wrist_pose'][0, 0, 3:] = 0.
            data['object_velocity'][1, 0] = .031
        updated = advance(tracker, data, 8)
        legacy = advance(original, data, 8)
        assert tracker.window_completed
        for key in ('duration', 'event', 'paired', 'window_good', 'drift_m'):
            assert torch.equal(updated[key], legacy[key])
        if drift == .0006:
            assert not tracker.window_good.any()
            assert tracker.slip_observable.all()
            assert tracker.reward_drift_m.tolist() == pytest.approx([.0006, .0006])
    assert not tracker.slip_observable.any()
    assert not tracker.reward_drift_m.any()
    advance(tracker, scene(), 1)
    assert not tracker.window_completed


def test_actual_motion_and_projected_command_have_separate_contributions():
    from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import ExecutedMotion
    zero = torch.zeros(2, 24, dtype=torch.float64)
    valid = torch.ones(2, dtype=torch.bool)
    motion = ExecutedMotion(zero)
    dt = 1 / 480
    delta = motion.rate.expand_as(zero) * dt
    motion.command(delta, dt)
    assert not motion.measure(zero, delta, torch.ones_like(zero), valid, dt).any()
    assert not motion.measure(delta, 2 * delta, torch.ones_like(zero), valid, dt).any()
    motion.command(zero, dt)
    motion.reset([0])
    assert motion.measure(delta, delta, torch.ones_like(zero), valid, dt).tolist() == pytest.approx([.1, 1.1])


def test_support_only_counts_accepted_normals_and_attributable_friction():
    from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import accepted_support_n
    task = qualified_task()
    m = task.m
    m['effective_pad_vectors_w'][:, :, 0, 2] = -.4
    m['effective_pad_vectors_w'][:, :, 3, 2] = -5.  # rejected digit must not contribute
    m['rejected_contact_loads_n'][0, 1] = .1
    m['vertical_hand_force_n'].fill_(.4)
    lift = m['vertical_hand_force_n'] + torch.tensor([.1, -.2], dtype=torch.float64)
    assert accepted_support_n(m, lift).tolist() == pytest.approx([.9, .7])


def test_relative_motion_uses_digit_body_velocity_projected_at_each_pad():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_effective_pad import DISTAL_LINKS
    from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import enable, _relative_tangent_speed

    task = qualified_task()
    native = SimpleNamespace(body_com_pos_w=torch.zeros(2, 5, 3, dtype=torch.float64),
                             body_com_vel_w=torch.zeros(2, 5, 6, dtype=torch.float64))
    native.body_com_vel_w[..., 0] = .1  # normal to the pads
    native.body_com_vel_w[..., 1] = .004
    task.surface_world.hands = [SimpleNamespace(body_names=DISTAL_LINKS, data=native)] * 2
    task.m['object_velocity'].zero_()
    task.m['object_velocity'][:, 1] = .006
    task.surface_reward_profile = 'v3_relative_motion_only'
    enable(task)
    assert _relative_tangent_speed(task).tolist() == pytest.approx([.002, .002])
