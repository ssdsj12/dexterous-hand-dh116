"""CPU reward-factor oracles; no fixture is a native grasp validation."""
import copy
import importlib
import importlib.util
import math

import pytest
import torch

MODULE = 'go2_pvcnn.control.dh116.tabletop_lift_reward_experiment'
HOLD, UP, DROP = ('lift_quality_height_hold', 'lift_quality_upward', 'lift_quality_drop')


def api():
    assert importlib.util.find_spec(MODULE), 'isolated R0-R5 reward factors are missing'
    return importlib.import_module(MODULE)


def experiment(profile, n=2):
    return api().LiftRewardExperiment(profile, torch.full((n,), .36, dtype=torch.float64))


def sample(n=2, **changes):
    value = dict(bottom=torch.full((n,), .3765, dtype=torch.float64),
        opposed_contact=torch.ones(n, 2, dtype=torch.bool),
        support_n=torch.full((n,), 2., dtype=torch.float64),
        weight_n=torch.ones(n, dtype=torch.float64),
        drift_m=torch.zeros(n, dtype=torch.float64), slip_valid=torch.ones(n, dtype=torch.bool),
        object_velocity=torch.zeros(n, 6, dtype=torch.float64),
        measurement_valid=torch.ones(n, dtype=torch.bool), failed=torch.zeros(n, dtype=torch.bool),
        window_good=torch.ones(n, dtype=torch.bool), dt=1/60, window_completed=True)
    value.update(changes)
    return value


def test_profiles_are_single_factors_before_combinations_and_r0_adds_nothing():
    m = api()
    expected = {'R0': (), 'R1': (HOLD,), 'R2': (UP,), 'R3': (DROP,),
                'R4': (HOLD, UP), 'R5': (HOLD, UP, DROP)}
    assert m.PROFILES == tuple(expected)
    for profile, factors in expected.items():
        c = m.experiment_contract(profile)
        assert tuple(c['enabled_terms']) == factors
        assert c['base_reward_profile'] == 'support_deficit'
        assert c['success'] == 'unchanged_30mm_5s_0.5mm'
        assert set(experiment(profile).update(**sample())) == set(factors)
    with pytest.raises(ValueError): experiment('all_factors_auto')


def test_hold_uses_net_height_quality_and_real_seconds_not_support_prelift():
    r = experiment('R1')
    # (.0165-.003)/(.030-.003) == .5; coefficient1 point/s.
    result = r.update(**sample())
    torch.testing.assert_close(result[HOLD], torch.full((2,), .5/60, dtype=torch.float64))
    # Unchanged physical baseline: the initial 60mm pedestal is not added.
    at_reset = sample(bottom=torch.full((2,), .36, dtype=torch.float64))
    assert not r.update(**at_reset)[HOLD].any()
    r.reset(torch.tensor([0]), torch.tensor([.42], dtype=torch.float64))
    below_new_reference = sample(bottom=torch.tensor([.421, .3765], dtype=torch.float64))
    torch.testing.assert_close(r.update(**below_new_reference)[HOLD],
                               torch.tensor([0., .5/60], dtype=torch.float64))


@pytest.mark.parametrize('bad', ['left_contact', 'right_contact', 'support', 'weight',
    'slip_missing', 'slip_excess', 'speed', 'invalid', 'failed', 'window', 'nonfinite'])
def test_positive_factors_fail_closed_on_contact_support_slip_or_invalidity(bad):
    r = experiment('R4')
    s = sample()
    if bad == 'left_contact': s['opposed_contact'][:, 0] = False
    elif bad == 'right_contact': s['opposed_contact'][:, 1] = False
    elif bad == 'support': s['support_n'].fill_(.99)
    elif bad == 'weight': s['weight_n'].zero_()
    elif bad == 'slip_missing':
        s['slip_valid'].fill_(False); s['drift_m'].fill_(float('nan'))
    elif bad == 'slip_excess': s['drift_m'].fill_(.000501)
    elif bad == 'speed': s['object_velocity'][:, 0] = .03
    elif bad == 'invalid': s['measurement_valid'].fill_(False)
    elif bad == 'failed': s['failed'].fill_(True)
    elif bad == 'window': s['window_good'].fill_(False)
    elif bad == 'nonfinite': s['object_velocity'][:, 2] = float('nan')
    result = r.update(**s)
    assert not result[HOLD].any() and not result[UP].any()
    assert all(bool(torch.isfinite(v).all()) for v in result.values())


def test_upward_uses_actual_height_rate_and_does_not_pay_repeated_oscillation():
    r = experiment('R2')
    s = sample(bottom=torch.full((2,), .37, dtype=torch.float64), dt=1.)
    r.update(**s)  # Establish a previously qualified10mm peak.
    s['bottom'] += .00005; s['dt'] = .01
    s['object_velocity'][:, 2] = .005
    # New50um over10ms =>5mm/s, quality=1-.005/.03=5/6.
    expected = torch.full((2,), .5 * (5/6) * .01, dtype=torch.float64)
    torch.testing.assert_close(r.update(**s)[UP], expected, rtol=1e-10, atol=1e-12)
    for _ in range(20):
        s['bottom'] -= .00001
        assert not r.update(**s)[UP].any()
        s['bottom'] += .00001
        assert not r.update(**s)[UP].any()


def test_drop_ignores_small_reversible_vibration_and_charges_real_descent_only():
    r = experiment('R3')
    s = sample(bottom=torch.full((2,), .37, dtype=torch.float64), dt=.1)
    r.update(**s)
    for _ in range(10):
        s['bottom'] -= .0003
        assert not r.update(**s)[DROP].any()
        s['bottom'] += .0003
        assert not r.update(**s)[DROP].any()
    s['bottom'] -= .0003
    assert not r.update(**s)[DROP].any()
    s['bottom'] -= .0003
    # A600um continuous descent crosses the500um deadband by100um.
    expected = -.5 * (.0001 / (.005 * .1)) * .1
    torch.testing.assert_close(r.update(**s)[DROP], torch.full((2,), expected, dtype=torch.float64),
                               rtol=1e-10, atol=1e-12)


def test_incomplete_physics_window_cannot_advance_or_pay_control_window_reward():
    r = experiment('R5')
    before = copy.deepcopy(r.__dict__)
    for _ in range(7):
        result = r.update(**sample(dt=1/480, window_completed=False))
        assert all(not v.any() for v in result.values())
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(r.__dict__[key], value, rtol=0, atol=0)
        else: assert r.__dict__[key] == value


def test_hold_reward_is_time_invariant_and_reset_only_clears_selected_rows():
    total = []
    for dt in (.01, .02):
        r = experiment('R1')
        score = torch.zeros(2, dtype=torch.float64)
        for _ in range(round(1/dt)): score += r.update(**sample(dt=dt))[HOLD]
        total.append(score)
    torch.testing.assert_close(total[0], total[1], rtol=1e-12, atol=1e-12)
    r = experiment('R2')
    r.update(**sample(bottom=torch.full((2,), .37, dtype=torch.float64)))
    r.reset(torch.tensor([0]), torch.tensor([.36], dtype=torch.float64))
    result = r.update(**sample(bottom=torch.full((2,), .369, dtype=torch.float64)))
    assert result[UP][0] > 0 and result[UP][1] == 0


@pytest.mark.parametrize('dt', [0., -1., float('nan'), float('inf'), True])
def test_invalid_dt_rejected(dt):
    with pytest.raises(ValueError): experiment('R1').update(**sample(dt=dt))


def test_physical_reset_capture_is_owned_selected_and_consumed_once():
    r = experiment('R2')
    r.update(**sample())
    other_peak = r._qualified_peak[1].clone()
    measured = torch.tensor([.42, .43], dtype=torch.float64)
    r.capture_physical_reset(torch.tensor([0]), measured)
    measured.fill_(.99)  # A native no-copy measurement may change during the tail.
    r.reset(torch.tensor([0]), torch.tensor([.44, .45], dtype=torch.float64))
    torch.testing.assert_close(r.reference, torch.tensor([.42, .36], dtype=torch.float64))
    assert r._qualified_peak[0] == 0. and r._qualified_peak[1] == other_peak
    r.reset(torch.tensor([0]), torch.tensor([.46, .47], dtype=torch.float64))
    assert r.reference[0] == .46  # The captured baseline was consumed, not reused.
    r.capture_physical_reset(torch.tensor([False, True]), torch.tensor([.48, .49], dtype=torch.float64))
    r.reset(torch.tensor([0]), torch.tensor([.50, .51], dtype=torch.float64))
    r.reset(torch.tensor([1]), torch.tensor([.52, .53], dtype=torch.float64))
    torch.testing.assert_close(r.reference, torch.tensor([.50, .49], dtype=torch.float64))


def test_physics_drop_ledger_freezes_inactive_and_breaks_invalid_live_samples():
    r = experiment('R3')
    def step(bottom, live=(True, True), valid=(True, True)):
        return r.drop_physics_substep(bottom=torch.tensor(bottom, dtype=torch.float64),
            object_velocity=torch.zeros(2, 6, dtype=torch.float64),
            live=torch.tensor(live), measurement_valid=torch.tensor(valid), dt=1/480)
    assert not step((.37, .37)).any()
    # Inactive arbitrary/NaN states do not alter that row's last live state.
    assert not step((float('nan'), .39), live=(False, True), valid=(False, True)).any()
    score = step((.369, .389))
    torch.testing.assert_close(score, torch.full((2,), -.5/480, dtype=torch.float64))
    assert not step((float('nan'), .389), valid=(False, True)).any()
    assert not step((.36, .389)).any()  # Invalid live sample broke the bridge.
    assert step((.359, .389))[0] == pytest.approx(-.5/480)


def test_physics_drop_deadband_reset_and_positive_window_state_are_independent():
    r = experiment('R5')
    positive_state = {k: v.clone() for k, v in r.__dict__.items()
                      if k in ('_qualified_peak', '_prev_h', '_prev_valid', '_descent_anchor')}
    def step(bottom):
        return r.drop_physics_substep(bottom=torch.tensor(bottom, dtype=torch.float64),
            object_velocity=torch.zeros(2, 6, dtype=torch.float64),
            live=torch.ones(2, dtype=torch.bool), measurement_valid=torch.ones(2, dtype=torch.bool),
            dt=1/480)
    step((.37, .37))
    for _ in range(20):
        assert not step((.3697, .3697)).any()
        assert not step((.37, .37)).any()
    for k, value in positive_state.items(): torch.testing.assert_close(r.__dict__[k], value)
    step((.3697, .3697))
    assert (step((.3694, .3694)) < 0.).all()
    r.reset(torch.tensor([0]), torch.tensor([.36, .36], dtype=torch.float64))
    score = step((.369, .369))
    assert score[0] == 0. and score[1] < 0.


def test_physics_drop_rate_uses_actual_substep_seconds():
    totals = []
    # Below the rate cap: one millimetre of descent, less the .5mm deadband.
    for dt in (1/480, 1/240):
        r = experiment('R3', n=1)
        def step(h):
            return r.drop_physics_substep(bottom=torch.tensor([.36+h], dtype=torch.float64),
                object_velocity=torch.zeros(1, 6, dtype=torch.float64),
                live=torch.ones(1, dtype=torch.bool), measurement_valid=torch.ones(1, dtype=torch.bool), dt=dt)
        total = step(.01)
        for i in range(1, round(1/dt)+1): total += step(.01-.001*i*dt)
        totals.append(total)
    torch.testing.assert_close(totals[0], totals[1], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(totals[0], torch.tensor([-.05], dtype=torch.float64), atol=1e-10, rtol=1e-10)
