"""Continuous controller behavior; no Isaac imports or physical simulation."""
import json
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_dh116_policy_control import enabled
from test_force_region_integration import constrained
from test_constant_force_hold import held
from test_manual_commands import commands
from test_policy_lift_hold_integration import configured
from go2_pvcnn.control.dh116.tabletop_policy_force_region import adjust


def upward(t):
    actions = torch.zeros((t.n, 20), dtype=torch.float64)
    actions[:, 14] = actions[:, 17] = .2
    return actions


@pytest.mark.parametrize('phase', [1, 3])
@pytest.mark.parametrize('hand', [0, 1])
@pytest.mark.parametrize('error', [.012556, -.020860, .031])
def test_finite_mimic_keeps_both_wrist_requests_allowed(phase, hand, error):
    t = enabled()
    t.continuous_motion = True
    t.phase[:] = phase
    t.m['mimic_errors'][0, 5 * hand + 1] = error
    measured = {name: t.m[name].clone() for name in
                ('mimic_errors', 'q', 'hand_object_loads', 'adaptive_palm_offset')}
    before = t.command.clone()
    dt = 1 / 480

    allowed = t.control(upward(t), ~t.done, dt)

    assert allowed.all(), 'finite mimic deviation must not veto either wrist'
    assert not t.paused.any() and not t.f_blocked.any()
    assert (t.palm_targets[:, :, 2] > t.m['adaptive_palm_offset'][:, :, 2]).all()
    # Linkage correction still acts in the measured error direction at the
    # existing actuator rate, rather than disabling the correction itself.
    assert (t.command[0, 6 * hand + 2] - before[0, 6 * hand + 2]) * error > 0
    assert ((t.command - before).abs() <= .3 * dt + 1e-12).all()
    assert (t.command >= t.low).all() and (t.command <= t.high).all()
    for name, value in measured.items():
        torch.testing.assert_close(t.m[name], value)


@pytest.mark.parametrize('mode', [False, None])
def test_disabled_or_absent_mode_keeps_legacy_mimic_pause(mode):
    t = enabled()
    if mode is not None:
        t.continuous_motion = mode
    t.m['mimic_errors'][0, 1] = .021
    allowed = t.control(upward(t), ~t.done, 1 / 480)
    assert allowed.tolist() == [False, True]
    assert t.paused.tolist() == [True, False]


def test_continuous_lift_reports_mimic_diagnostic_without_losing_motion_permission():
    from test_dh116_close_downward_penalty import task
    from test_dh116_tensor_contact_window import loads

    t = task(2, grasp_shaping=True)
    t.enable_policy_control()
    t.continuous_motion = True
    t.phase[:] = 3
    t.m = loads(t, 4.)
    t.command.copy_(t.m['q'])
    t.m['mimic_errors'][0, 1] = .021

    allowed = t.control(upward(t), ~t.done, 1 / 480)

    assert t.lift_diagnostic_masks[0, 2], 'retain the actual mimic diagnostic'
    assert allowed.all() and t.lift_diagnostic_masks[:, 1].all()
    assert t.m['mimic_errors'][0, 1] == pytest.approx(.021)


@pytest.mark.parametrize('manual_fast_force', [False, True])
def test_force_region_continues_bounded_normal_correction_despite_finite_mimic(manual_fast_force):
    t = constrained()
    t.continuous_motion = True
    t.manual_fast_force = manual_fast_force
    t.m['mimic_errors'][0, 1] = -.020860
    t.m['mimic_errors'][1, 6] = .031
    raw = t.m['mimic_errors'].clone()
    target = t.command.clone()
    requested = torch.full_like(t.f_requested, 5.)
    live = ~t.done
    dt = .01
    rate = .12 if manual_fast_force else .03

    for tick in range(300):
        before = t.f_offset.clone()
        adjusted = adjust(t, target, requested, live, dt)
        assert (t.f_offset > 0).all(), 'mimic alone must not veto normal-force correction'
        assert ((t.f_offset - before).abs() <= rate * dt + 1e-12).all()
        if tick == 0:
            torch.testing.assert_close(t.f_offset, torch.full_like(t.f_offset, rate * dt))
        assert (t.f_offset.abs() <= .08 + 1e-12).all()
        torch.testing.assert_close(adjusted - target,
            (t.f_offset[:, :, None] * t.direction).flatten(1))
    torch.testing.assert_close(t.f_offset, torch.full_like(t.f_offset, .08))
    torch.testing.assert_close(t.m['mimic_errors'], raw)


@pytest.mark.parametrize('kind', ['invalid_contact', 'above_axis', 'overload', 'inactive'])
def test_continuous_force_correction_keeps_other_gates_and_release(kind):
    t = constrained()
    t.continuous_motion = True
    t.m['mimic_errors'].fill_(.021)
    live = ~t.done
    if kind == 'invalid_contact':
        t.m['contact_valid'][:] = False
    elif kind == 'above_axis':
        t.m['contact_all_below_centerline'][:] = False
    elif kind == 'overload':
        t.m['hand_object_loads'][:] = 13.
    else:
        live[:] = False
    adjust(t, t.command.clone(), torch.full_like(t.f_requested, 5.), live, .01)
    if kind == 'overload':
        assert (t.f_offset < 0).all()
    else:
        assert not t.f_offset.any()


def test_continuous_policy_keeps_pressure_relief_and_joint_slew():
    t = constrained()
    t.continuous_motion = True
    t.m['hand_object_loads'][0] = 13.
    before = t.command.clone()
    allowed = t.control(torch.ones((2, 20), dtype=torch.float64), ~t.done, 1 / 480)
    assert allowed.tolist() == [False, True]
    assert t.paused.tolist() == [True, False]
    assert t.pressure_limited[0].all()
    sign = torch.where(t.direction == 0, 1., t.direction.sign()).flatten(1)
    assert (((t.command[0] - before[0]) * sign[0]) <= 1e-12).all()
    assert ((t.command - before).abs() <= .3 / 480 + 1e-12).all()
    assert (t.command >= t.low).all() and (t.command <= t.high).all()


@pytest.mark.parametrize('force_region', [False, True])
@pytest.mark.parametrize('continuous', [False, True])
def test_lift_hold_mimic_is_diagnostic_only_in_continuous_mode(force_region, continuous):
    if force_region:
        t, h, live = held()
    else:
        t = configured()
        t.phase[:] = 3
        h = t.policy_lift_hold
        live = ~t.done
        h.capture(t, live, t.m['adaptive_palm_offset'])
        t.m['hand_object_loads'] -= .2
    t.continuous_motion = continuous
    t.m['mimic_errors'][0, 1] = -.020860
    raw = t.m['mimic_errors'].clone()
    t.palm_targets[:, :, 2] = .01
    previous = h.closure.clone()

    target, palms, _, paused = h.targets(t, live, .01)

    assert bool(paused[0]) is (not continuous)
    assert not paused[1]
    if continuous:
        assert (palms[:, :, 2] > t.m['adaptive_palm_offset'][:, :, 2]).all()
        assert (h.closure > 0).all(), 'normal-force correction must also continue'
        if force_region:
            assert (h.force_integral > 0).all()
    else:
        torch.testing.assert_close(palms[0], t.m['adaptive_palm_offset'][0])
        assert not h.closure[0].any()
    assert ((h.closure - previous).abs() <= .03 * .01 + 1e-12).all()
    assert ((target - h.command).abs() <= .08 + 1e-12).all()
    torch.testing.assert_close(t.m['mimic_errors'], raw)


@pytest.mark.parametrize('kind', ['contact_loss', 'invalid_region', 'force_tracking', 'tracking', 'slip'])
def test_continuous_hold_keeps_real_pause_causes_even_with_mimic(kind):
    t, h, live = held()
    t.continuous_motion = True
    t.m['mimic_errors'].fill_(.021)
    t.palm_targets[:, :, 2] = .01
    if kind == 'contact_loss':
        t.m['hand_object_loads'].zero_()
    elif kind == 'invalid_region':
        t.m['contact_all_below_centerline'][:] = False
    elif kind == 'force_tracking':
        t.m['hand_object_loads'].fill_(3.)
    elif kind == 'tracking':
        t.command += .04
    else:
        t.slip[:] = .003

    _, palms, _, paused = h.targets(t, live, .01)

    assert paused.all()
    torch.testing.assert_close(palms, t.m['adaptive_palm_offset'])
    torch.testing.assert_close(h.force_target, torch.full_like(h.force_target, 5.))


@pytest.mark.parametrize('phase', [1, 3])
def test_continuous_digit_delegate_preserves_independent_targets_and_bypasses_hold(phase):
    from test_manual_digit_force_balance import setup
    from go2_pvcnn.control.dh116.tabletop_manual_digit_force_balance import enable
    from go2_pvcnn.control.dh116.tabletop_policy_lift_hold import PolicyLiftHold

    t = setup([[3.3, .2], [.1, 3.3]])
    t.continuous_motion = True
    t.manual_stable_grip_ascent = True
    t.direction[:] = t.direction.new_tensor([
        [.156, .662, .733, 0., 0., 0.], [.099, .419, .902, 0., 0., 0.]])
    enable(t)
    t.phase[:] = phase
    t.m['grouped_object_loads'][:, :, 2] = .4
    t.m['hand_object_loads'] = t.m['grouped_object_loads'].sum(-1)
    t.m['contact_valid'][:] = False
    t.m['contact_all_below_centerline'][:] = False
    t.m['mimic_errors'][:, 0] = -.020860
    t.m['mimic_errors'][:, 6] = -.020860
    raw = {k: t.m[k].clone() for k in ('mimic_errors', 'grouped_object_loads', 'hand_object_loads')}
    h = PolicyLiftHold(t.n, t.device)
    t.policy_lift_hold = h
    h.capture(t, ~t.done, t.m['adaptive_palm_offset'])
    h.force_target.fill_(2.)
    h.command += .07
    h.paused[:] = True
    hold_state = {k: v.clone() for k, v in vars(h).items()}
    actions = upward(t)
    actions[:, 18:] = -.4  # Current manual request is 5 N, not the captured 2 N.

    allowed = t.control(actions, ~t.done, .01)

    assert allowed.all() and not t.paused.any()
    assert (t.manual_digit_offsets[:, 0, 0] < 0).all()
    assert (t.manual_digit_offsets[:, 0, 1] > 0).all()
    assert (t.manual_digit_offsets[:, 1, 0] > 0).all()
    assert (t.manual_digit_offsets[:, 1, 1] < 0).all()
    assert (t.manual_digit_offsets.abs() <= .12 * .01 + 1e-12).all()
    torch.testing.assert_close(t.manual_digit_targets.sum(-1), torch.full_like(t.f_target, 4.6))
    ratio = t.manual_digit_targets.amax(-1) / t.manual_digit_targets.amin(-1)
    assert (ratio <= 1.4 + 1e-12).all()
    torch.testing.assert_close(t.f_requested, torch.full_like(t.f_requested, 5.))
    delta = (t.manual_digit_offsets[:, :, :, None] * t.manual_digit_directions).sum(2).flatten(1)
    torch.testing.assert_close(t.joint_governed_target, t.reference + delta)
    torch.testing.assert_close(t.command, t.joint_governed_target)
    for name, value in hold_state.items():
        torch.testing.assert_close(getattr(h, name), value)
    for name, value in raw.items():
        torch.testing.assert_close(t.m[name], value)


@pytest.mark.parametrize('error', [-.021, .021])
@pytest.mark.parametrize('axis_sign', [-1., 1.])
def test_continuous_digit_follower_correction_uses_signed_source_pose_with_original_bounds(error, axis_sign):
    from test_manual_digit_force_balance import setup, step
    from go2_pvcnn.control.dh116.tabletop_manual_digit_force_balance import enable

    t = setup([[2.2, 2.8], [2.8, 2.2]])  # Already at the requested unequal force split.
    t.continuous_motion = True
    t.direction[:] = axis_sign * t.direction.new_tensor([
        [.156, .662, .733, 0., 0., 0.], [.099, .419, .902, 0., 0., 0.]])
    enable(t)
    t.m['mimic_errors'].reshape(t.n, 2, 5)[:, :, :2] = error
    raw = t.m['mimic_errors'].clone()
    live = torch.tensor([True, False])
    for _ in range(100):
        previous = t.manual_digit_offsets.clone()
        target, out = step(t, live)
        assert (t.manual_digit_offsets[0] * error * axis_sign > 0).all()
        assert not t.manual_digit_offsets[1].any()
        assert not t.manual_digit_targets[1].any()
        assert ((t.manual_digit_offsets - previous).abs() <= .12 * .01 + 1e-12).all()
        assert (t.manual_digit_offsets.abs() <= .08 + 1e-12).all()
        delta = (t.manual_digit_offsets[:, :, :, None] * t.manual_digit_directions).sum(2).flatten(1)
        torch.testing.assert_close(out, target + delta)
    torch.testing.assert_close(t.m['mimic_errors'], raw)
    torch.testing.assert_close(t.manual_digit_offsets[0].abs(), torch.full_like(t.manual_digit_offsets[0], .08))


def test_continuous_digit_acquisition_and_latched_low_gain_survive_mimic():
    from test_manual_digit_force_balance import setup, step

    t = setup([[0., 0.], [0., 0.]])
    t.continuous_motion = True
    t.manual_digit_force_gain = .005
    t.m['mimic_errors'].fill_(.009)  # Legacy taper would slow or suppress acquisition.
    step(t)
    torch.testing.assert_close(t.manual_digit_offsets, torch.full_like(t.manual_digit_offsets, .0012))
    assert not t.manual_digit_contact_acquired.any()
    t.m['grouped_object_loads'][:, :, :2] = .2
    t.m['hand_object_loads'] = t.m['grouped_object_loads'].sum(-1)
    previous = t.manual_digit_offsets.clone()
    step(t)
    assert t.manual_digit_contact_acquired.all()
    assert (t.manual_digit_offsets - previous).abs().max() < .0002
    previous = t.manual_digit_offsets.clone()
    t.m['grouped_object_loads'].zero_()
    t.m['hand_object_loads'].zero_()
    step(t)
    assert (t.manual_digit_offsets - previous).abs().max() < .0002


def test_continuous_smooth_digit_control_has_no_postservo_mimic_relay():
    import copy
    from test_manual_digit_force_balance import setup

    a = setup([[1.9, 3.], [.8, .7]])
    a.continuous_motion = True
    a.manual_digit_offsets.fill_(.06)
    b = copy.deepcopy(a)
    a.m['mimic_errors'][0, 0] = -.01199
    b.m['mimic_errors'][0, 0] = -.01201
    for t in (a, b):
        actions = upward(t)
        actions[:, 18:] = -.4
        assert t.control(actions, ~t.done, 1 / 480).all()
        delta = (t.manual_digit_offsets[:, :, :, None] * t.manual_digit_directions).sum(2).flatten(1)
        torch.testing.assert_close(t.joint_governed_target, t.reference + delta)
    assert (a.joint_governed_target - b.joint_governed_target).abs().max() < .001


@pytest.mark.parametrize('continuous', [False, True])
def test_manual_fast_hand_servo_retains_contact_acquisition_and_own_rate(continuous):
    t = constrained()
    t.continuous_motion = continuous
    t.manual_fast_force = True
    t.m['hand_object_loads'].zero_()
    t.m['contact_valid'][:] = False
    t.m['contact_all_below_centerline'][:] = False
    t.m['mimic_errors'].fill_(.031)
    target = t.command.clone()
    out = adjust(t, target, torch.full_like(t.f_target, 5.), ~t.done, .01)
    expected = .0012 if continuous else 0.
    torch.testing.assert_close(t.f_offset, torch.full_like(t.f_offset, expected))
    torch.testing.assert_close(out, target + expected * t.direction.flatten(1))


@pytest.fixture
def manual_loop(monkeypatch, tmp_path):
    """Run the actual GUI loop once with only external Kit/physics IO replaced."""
    from go2_pvcnn.control.dh116 import tabletop_manual_control as manual
    from go2_pvcnn.control.dh116 import tabletop_manual_digit_contacts as contacts
    from go2_pvcnn.control.dh116.tabletop_policy_force_region import enable
    from go2_pvcnn.control.dh116.tabletop_policy_lift_hold import PolicyLiftHold
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask, FAILURE_CODES
    from test_dh116_tensor_task import template
    from test_dh116_joint_envelope import profile

    def run(*, continuous=True, recovered=True, lifting=True, reset_button=False, error=None,
            failure=None, status=False, save=False):
        e = template()
        e.joint_envelope = profile()
        t = TensorTask(e, 1, 'cpu', pressure_relief=True)
        t.enable_policy_control()
        t.policy_lift_hold = PolicyLiftHold(1, 'cpu')
        enable(t)
        t.continuous_motion = continuous
        state = SimpleNamespace(commands=None, resets=0, steps=0, destroyed=False)

        class Panel:
            def __init__(self, c):
                state.commands = c
                c.lifting = lifting
                c.lift_current = .03
                c.lift_goal = .04
                c.lift_speed = .005
                c.auto_lift_goal = None if lifting else .02
                c.force_stable_s = .3
                c.previous_load = torch.full((2,), 5.)
                c.pending_reset = reset_button
                c.pending_save = save
                self.status = SimpleNamespace(text='')
                self.window = SimpleNamespace(visible=True, destroy=self.destroy)

            def destroy(self):
                state.destroyed = True

        class Sim:
            playing = True

            def is_playing(self): return self.playing
            def play(self): self.playing = True
            def pause(self): self.playing = False
            def render(self): pass

        class Env:
            device = 'cpu'
            task = t
            world = SimpleNamespace(dt=.01, decimation=1, w=SimpleNamespace(sim=Sim()))

            def reset(self):
                state.resets += 1
                t.reset([0], t.m)

            def step(self, actions):
                state.steps += 1
                if error is not None:
                    raise error
                # TensorEnv returns done plus explicit recovery evidence after
                # its validated reset; it owns physical recovery and isolation.
                t.reset([0], t.m)
                return (t.observe(), torch.zeros(1), torch.ones(1, dtype=torch.bool),
                        dict(failure=torch.tensor([FAILURE_CODES['joint_limit']]),
                             recovered=torch.tensor([recovered])))

        running = iter([True, False])
        kit = SimpleNamespace(is_running=lambda: next(running, False))
        omni = ModuleType('omni')
        omni.__path__ = []
        omni.kit = ModuleType('omni.kit')
        omni.kit.__path__ = []
        omni.kit.app = ModuleType('omni.kit.app')
        omni.kit.app.get_app = lambda: kit
        for name, module in (('omni', omni), ('omni.kit', omni.kit),
                             ('omni.kit.app', omni.kit.app)):
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(manual, 'ManualPanel', Panel)
        monkeypatch.setattr(manual, 'time', SimpleNamespace(monotonic=lambda: .2 if status else 0., sleep=lambda _: None))
        monkeypatch.setattr(contacts, 'Visuals', lambda: SimpleNamespace(update=lambda *_: None))
        monkeypatch.setattr(contacts, 'collect', lambda _: (
            dict(all_four_digits_below_axis=False, digits={}), None, None))
        args = SimpleNamespace(output=tmp_path, manual_run=True)
        state.env = Env()
        if failure is not None:
            state.env.manual_last_failure = failure
        # Expose state even if the real loop correctly propagates an exception.
        run.state = state
        manual.run_manual_control(state.env, args, {})
        return state

    return run


@pytest.mark.parametrize('lifting', [False, True])
def test_recovered_manual_reset_preserves_run_and_lift_intent(manual_loop, lifting):
    state = manual_loop(lifting=lifting)
    c = state.commands
    assert c.running, 'a validated recovered reset must preserve Run'
    assert c.lifting is lifting
    assert c.lift_current == 0., 'discard the old unexecuted height after reset'
    assert c.lift_goal == pytest.approx(.04) and c.lift_speed == pytest.approx(.005)
    assert c.auto_lift_goal == (None if lifting else .02)
    assert c.force_stable_s == 0. and c.previous_load is None
    assert state.steps == 1 and state.resets == 0 and state.destroyed
    c.actions(.01)
    assert c.lift_current == pytest.approx(.00005 if lifting else 0.)


@pytest.mark.parametrize('continuous,recovered', [(False, True), (True, False)])
def test_manual_done_without_continuous_recovery_keeps_pause(manual_loop, continuous, recovered):
    c = manual_loop(continuous=continuous, recovered=recovered).commands
    assert not c.running and not c.lifting and c.lift_current == 0.


def test_explicit_reset_pause_stays_paused_in_continuous_mode(manual_loop):
    state = manual_loop(reset_button=True)
    c = state.commands
    assert not c.running and not c.lifting and c.lift_current == 0.
    assert c.auto_lift_goal is None
    assert state.resets == 1 and state.steps == 0
    assert c.actions(.01)[0, 14] == 0. and c.actions(.01)[0, 17] == 0.


@pytest.mark.parametrize('message', ['tensor physical safety failure: immediate_reset env_ids=[0]',
                                     'ordinary control failure'])
def test_unrecoverable_runtime_error_surfaces_without_gui_retry(manual_loop, message):
    error = RuntimeError(message)
    with pytest.raises(RuntimeError) as raised:
        manual_loop(error=error)
    assert raised.value is error
    assert manual_loop.state.resets == 0
    assert manual_loop.state.steps == 1 and manual_loop.state.destroyed


def test_legacy_manual_physical_guard_still_pauses_and_resets(manual_loop):
    state = manual_loop(continuous=False, error=RuntimeError('tensor physical safety failure: live_state'))
    assert state.resets == 1
    assert not state.commands.running and not state.commands.lifting


def test_explicit_commands_reset_remains_a_pause():
    c = commands()
    c.start_lift()
    c.lift_current = .03
    c.pause_after_reset()
    assert not c.running and not c.lifting and c.lift_current == 0.


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_recovered_raw_invalid_diagnostics_survive_strict_json(manual_loop, tmp_path, invalid):
    failure = dict(max_mimic_error_rad=[invalid], hand_load_n=[[invalid, 2.]],
                   object_lift_m=[invalid], active_joint_rad=[[invalid] * 12])
    state = manual_loop(failure=failure, status=True, save=True)
    assert state.commands.running and state.commands.lifting

    def reject_constant(value):
        raise AssertionError('invalid JSON numeric token: ' + value)

    saved = json.loads((tmp_path / 'manual_last_failure.json').read_text(), parse_constant=reject_constant)
    status = json.loads((tmp_path / 'manual_status.json').read_text(), parse_constant=reject_constant)
    snapshot = json.loads((tmp_path / 'manual_snapshots.jsonl').read_text(), parse_constant=reject_constant)
    for row in (saved, status['last_failure'], snapshot['last_failure']):
        assert row['max_mimic_error_rad'][0] == {'nonfinite': str(invalid)}
        assert row['hand_load_n'][0][1] == 2.
    raw = state.env.manual_last_failure['max_mimic_error_rad'][0]
    assert math.isnan(raw) if math.isnan(invalid) else raw == invalid
