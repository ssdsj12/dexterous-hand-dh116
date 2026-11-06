"""CPU numerical oracles and physics stubs; these are not native evidence."""
import importlib
from types import SimpleNamespace

import pytest
import torch

from test_dh116_tensor_env import World
from test_dh116_tensor_task import template
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_surface_success import SurfaceLiftSuccess


class LiftWorld(World):
    def __init__(self, task):
        super().__init__(task)
        self.m['bottom'][:] = .41
        self.reset_at = []

    def reset(self, ids):
        self.reset_at.append(self.tick)
        super().reset(ids)
        self.m['bottom'][ids] = .41
        return self.m

    def advance(self, task, allowed, live):
        result = super().advance(task, allowed, live)
        self.m['bottom'] += .011 if self.tick == 2 else .01
        return result


def environment(enabled=True, continuous=False):
    task = TensorTask(template(), 2, 'cpu')
    task.continuous_motion = continuous
    world = LiftWorld(task)
    env = TensorEnv(world, task)
    packets = []
    if enabled:
        assert hasattr(env, 'enable_evaluation_telemetry'), 'opt-in evaluation telemetry missing'
        diagnostics = env.enable_evaluation_telemetry(packets.append)
    else:
        diagnostics = None
    env.reset()
    return env, world, task, packets, diagnostics


def telemetry(n=2):
    module = 'go2_pvcnn.control.dh116.tabletop_evaluation_telemetry'
    assert importlib.util.find_spec(module) is not None, 'evaluation telemetry module missing'
    return importlib.import_module(module).EvaluationTelemetry(n, 'cpu', lambda packet: None)


def raw_measurement(n=2):
    dtype = torch.float64
    wrist = torch.zeros(n, 2, 7, dtype=dtype)
    wrist[..., 3] = 1.
    quat = torch.zeros(n, 4, dtype=dtype)
    quat[:, 0] = 1.
    loads = torch.zeros(n, 2, 5, dtype=dtype)
    loads[..., :2] = .1
    vectors = torch.zeros(n, 2, 5, 3, dtype=dtype)
    vectors[..., 0, 0] = .1
    vectors[..., 1, 0] = -.1
    return dict(bottom=torch.full((n,), .31, dtype=dtype), wrist_pose=wrist,
                object_relative=torch.zeros(n, 3, dtype=dtype), object_quat=quat,
                object_velocity=torch.zeros(n, 6, dtype=dtype),
                effective_pad_loads_n=loads, effective_pad_vectors_w=vectors,
                effective_pad_valid=loads > 0,
                effective_pad_depth_m=torch.zeros_like(loads),
                rejected_contact_loads_n=torch.zeros(n, 2, dtype=dtype))


def raw_task(m):
    n = len(m['bottom'])
    return SimpleNamespace(m=m, success=torch.zeros(n, dtype=torch.bool),
        done=torch.zeros(n, dtype=torch.bool), nonfinite=torch.zeros(n, dtype=torch.bool),
        success_time=torch.zeros(n, dtype=torch.float64),
        surface_lift_success=SurfaceLiftSuccess(n, 'cpu', .3))


def sample(diagnostics, task, world, m, actions=None, anchor=None):
    n = len(m['bottom'])
    return diagnostics.sample(task, world, m, torch.ones(n, dtype=torch.bool), .01,
        torch.zeros(n, 24, dtype=torch.float64) if actions is None else actions,
        measurement_valid=torch.ones(n, dtype=torch.bool), anchor=anchor)


def update_tracker(t, m, dt=1/60):
    center = m['object_relative'] + m['wrist_pose'][..., :3].mean(1)
    return t.surface_lift_success.update(torch.cat((center, m['object_quat']), -1),
        m['wrist_pose'], m['effective_pad_loads_n'], m['effective_pad_vectors_w'],
        m['effective_pad_depth_m'], m['rejected_contact_loads_n'].sum(-1),
        m['bottom'], m['object_velocity'], torch.ones(len(center), dtype=torch.bool), dt)


def test_terminal_sample_precedes_physical_and_task_reset_and_inactive_tail_freezes():
    env, world, task, packets, diagnostics = environment()
    callback_reset_counts = []

    def observe(packet):
        callback_reset_counts.append(len(world.resets))
        packets.append(packet)

    diagnostics.callback = observe
    env.step(torch.zeros(2, 20))
    assert len(packets) == world.decimation
    assert packets[2]['live'].tolist() == [True, True]
    assert packets[2]['done'].tolist() == [True, False]
    assert packets[2]['lift_m'][0].item() == pytest.approx(.031)
    assert callback_reset_counts[2] == 1  # only env.reset, before terminal reset
    assert world.reset_at == [0, 3]
    assert packets[3]['live'].tolist() == [False, True]
    assert not packets[3]['done'].any()
    assert not task.done.any()  # automatic task reset at tick end
    snapshot = diagnostics.snapshot()
    assert snapshot['lift_m'][0].item() == pytest.approx(.031)
    assert snapshot['done'].tolist() == [True, False]
    assert snapshot['lift_m'][1].item() == pytest.approx(.081)
    assert snapshot['dt_s'] == world.dt
    # Neither task reset nor writes by consumers alias diagnostic state.
    packets[2]['lift_m'].fill_(99.)
    task.success_time.fill_(99.)
    assert diagnostics.snapshot()['lift_m'][0].item() == pytest.approx(.031)
    snapshot['done'].zero_()
    assert diagnostics.snapshot()['done'][0]


def test_native_success_event_and_next_episode_reference_are_preserved():
    env, world, task, packets, diagnostics = environment()
    world.tick = 100  # avoid the independent force-fault fixture at tick 3
    task.phase[0] = 3
    task.success_time[0] = .5
    env.step(torch.zeros(2, 20))
    assert packets[0]['native_success'].tolist() == [True, False]
    assert packets[0]['qualified_hold_s'][0] == pytest.approx(.5 + world.dt)
    assert not task.success.any()
    world.tick = 100
    packets.clear()
    env.step(torch.zeros(2, 20))
    # Reference comes from the real substep reset, not end-of-tick task reset.
    assert packets[0]['lift_m'][0] == pytest.approx(.08)


def test_no_anchor_invalid_geometry_and_nonfinite_drift_are_not_valid_zero_slip():
    d = telemetry(); m = raw_measurement(); t = raw_task(m)
    d.reset(torch.arange(2), m)
    assert not sample(d, t, SimpleNamespace(), m)['slip_valid'].any()
    m['bottom'][:] = .34
    update_tracker(t, m)
    before = d.capture_anchor(t)
    assert before is not None
    m['object_relative'][0, 0] += .0007
    m['wrist_pose'][1, 0, 0] = float('nan')
    packet = sample(d, t, SimpleNamespace(), m, anchor=before)
    assert packet['slip_valid'].tolist() == [True, False]
    assert packet['slip_m'][0] == pytest.approx(.0007)
    assert torch.isnan(packet['slip_m'][1])
    assert not packet['measurement_valid'][1]
    # The terminal native tracker may have reanchored. Keep the old anchor.
    m['wrist_pose'][1, 0, 0] = 0.
    update_tracker(t, m)
    assert t.surface_lift_success.duration[0] == 0.
    packet = sample(d, t, SimpleNamespace(), m, anchor=before)
    assert packet['slip_m'][0] == pytest.approx(.0007)


@pytest.mark.parametrize('case,applicable,reason', [
    ('no_anchor', False, 'no_anchor'),
    ('no_contact', False, 'no_contact'),
    ('bad_anchor', True, 'anchor_invalid'),
    ('invalid_pose', True, 'measurement_invalid'),
    ('invalid_contact', True, 'measurement_invalid'),
    ('invalid_pose_without_contact', True, 'measurement_invalid'),
    ('valid', True, 'valid'),
    ('inactive', False, 'inactive'),
])
def test_slip_applicability_distinguishes_legal_na_from_known_anchor_corruption(case, applicable, reason):
    d = telemetry(); m = raw_measurement(); t = raw_task(m)
    m['bottom'][:] = .34
    update_tracker(t, m)
    anchor = d.capture_anchor(t)
    assert anchor[1].all()  # genuine native tracker state, not slip_valid inference
    if case == 'no_anchor':
        t.surface_lift_success.reset(torch.arange(2))
        anchor = d.capture_anchor(t)
    if case == 'bad_anchor':
        anchor[0][0, 0, 0, 0] = float('nan')
    if case in ('no_contact', 'invalid_pose_without_contact'):
        m['effective_pad_loads_n'][0] = 0.
        m['effective_pad_vectors_w'][0] = 0.
        m['effective_pad_valid'][0] = False
    if case in ('invalid_pose', 'invalid_pose_without_contact'):
        m['wrist_pose'][0, 0, 0] = float('nan')
    if case == 'invalid_contact':
        m['effective_pad_vectors_w'][0, 0, 0, 0] = float('nan')
    live = torch.tensor([case != 'inactive', True])
    before = {key: value.clone() for key, value in vars(t.surface_lift_success).items()
              if isinstance(value, torch.Tensor)}
    packet = d.sample(t, SimpleNamespace(), m, live, .01, torch.zeros(2, 24),
        measurement_valid=torch.ones(2, dtype=torch.bool), anchor=anchor)
    assert packet['prior_anchor_latched'].tolist() == ([False, False] if case == 'no_anchor' else [True, True])
    assert packet['slip_applicable'][0].item() is applicable
    assert packet['slip_reason_code'][0] == packet['slip_reason_codes'][reason]
    assert packet['slip_valid'][0].item() is (reason == 'valid')
    if reason != 'valid':
        assert torch.isnan(packet['slip_m'][0])
    if case == 'inactive':
        assert d.snapshot()['slip_reason_code'][0] == packet['slip_reason_codes']['inactive']
    for key, value in before.items():
        assert torch.equal(value, getattr(t.surface_lift_success, key)), key


def test_raw_opposed_contact_is_independent_of_height_gated_native_hold():
    d = telemetry(); m = raw_measurement(); t = raw_task(m)
    d.reset(torch.arange(2), m)
    m['bottom'][1] = .34
    update_tracker(t, m)
    packet = sample(d, t, SimpleNamespace(), m)
    assert packet['bilateral_opposed_contact'].tolist() == [True, True]
    assert packet['contact_valid'].tolist() == [True, True]
    assert packet['qualified_hold_s'].tolist() == pytest.approx([0., 1/60])
    m['effective_pad_vectors_w'][0, 0, 1, 0] = .1
    packet = sample(d, t, SimpleNamespace(), m)
    assert packet['bilateral_opposed_contact'].tolist() == [False, True]


def test_reward_diagnostic_anchor_keeps_high_speed_native_slip_across_reanchor():
    d = telemetry(); m = raw_measurement(); t = raw_task(m)
    t.surface_lift_success = SurfaceLiftSuccess(2, 'cpu', .3,
        minimum_lift_m=.003, reward_diagnostics=True)
    tracker = t.surface_lift_success
    m['bottom'][:] = .303
    m['object_velocity'][:, 0] = .04
    update_tracker(t, m)
    assert not tracker._latched.any()  # speed disqualifies native hold
    assert tracker._slip_latched.all()  # native opposed-contact anchor survives
    anchor = d.capture_anchor(t)
    assert anchor[1].all()
    prior_points = anchor[0].clone()
    m['object_relative'][:, 0] += .0007
    update_tracker(t, m)  # native consume observes the drift and reanchors
    assert tracker.slip_observable.all()
    assert tracker.reward_drift_m.tolist() == pytest.approx([.0007, .0007])
    assert not tracker._latched.any()
    assert not torch.equal(tracker._slip_anchor, prior_points)
    # The pre-consume anchor is detached from native in-place latch/reset writes.
    assert torch.equal(anchor[0], prior_points)
    packet = sample(d, t, SimpleNamespace(), m, anchor=anchor)
    assert packet['prior_anchor_latched'].all()
    assert packet['slip_applicable'].all() and packet['slip_valid'].all()
    assert packet['slip_m'].tolist() == pytest.approx([.0007, .0007])
    assert packet['slip_reason_code'].eq(packet['slip_reason_codes']['valid']).all()
    assert packet['qualified_hold_s'].eq(0.).all()
    tracker.reset(torch.arange(2))
    assert anchor[1].all() and torch.equal(anchor[0], prior_points)


def test_torque_saturation_uses_native_estimate_and_original_limit_not_action_bound():
    d = telemetry(); m = raw_measurement(); t = raw_task(m)
    applied = torch.full((2, 8), .05, dtype=torch.float64)
    computed = torch.full_like(applied, .07)
    applied[0, 2] = .2
    applied[1, 2] = float('nan')
    hands = [SimpleNamespace(data=SimpleNamespace(applied_torque=applied,
        computed_torque=computed)) for _ in range(2)]
    world = SimpleNamespace(hands=hands, w=SimpleNamespace(ids=[[2, 3, 4, 5, 6, 7]] * 2),
        parameters=SimpleNamespace(effort_limit_nm=.2))
    actions = torch.ones(2, 24, dtype=torch.float64)
    actions[0] = 0.
    packet = sample(d, t, world, m, actions=actions)
    assert packet['estimated_torque_nm'].shape == (2, 12)
    assert packet['computed_torque_nm'].shape == (2, 12)
    assert packet['effort_limit_nm'].eq(.2).all()
    assert packet['torque_valid'][0].all()
    assert not packet['torque_valid'][1, 0]
    saturation = packet['estimated_torque_nm'].abs() >= packet['effort_limit_nm']
    assert saturation[0].sum() == 2 and saturation[1].sum() == 0
    assert packet['action'].shape == (2, 24)
    assert packet['action'].eq(actions).all()
    applied.zero_(); actions.zero_()
    assert packet['estimated_torque_nm'][0, 0] == .2
    assert packet['action'][1].eq(1.).all()
    assert 'implicit' in packet['torque_source']


def test_computed_torque_has_independent_validity_and_no_live_row_has_no_snapshot_sample():
    d = telemetry(); m = raw_measurement(); t = raw_task(m)
    applied = torch.full((2, 6), .05, dtype=torch.float64)
    computed = applied.clone(); computed[0, 0] = float('nan')
    world = SimpleNamespace(hands=[SimpleNamespace(data=SimpleNamespace(
        applied_torque=applied, computed_torque=computed)) for _ in range(2)],
        w=SimpleNamespace(ids=[list(range(6))] * 2),
        parameters=SimpleNamespace(effort_limit_nm=.2))
    packet = d.sample(t, world, m, torch.tensor([True, False]), .01,
        torch.zeros(2, 24), measurement_valid=torch.ones(2, dtype=torch.bool))
    assert packet['torque_valid'][0].all()
    assert not packet['computed_torque_valid'][0, 0]
    snapshot = d.snapshot()
    for key in ('measurement_valid', 'lift_valid', 'qualified_hold_valid', 'slip_valid',
                'contact_valid', 'action_valid'):
        assert not snapshot[key][1], key
    assert not snapshot['torque_valid'][1].any()
    assert torch.isnan(snapshot['qualified_hold_s'][1])


def test_missing_torque_stays_none_and_nonfinite_numeric_fields_are_invalid():
    d = telemetry(); m = raw_measurement(); t = raw_task(m)
    d.reset(torch.arange(2), m)
    m['bottom'][0] = float('nan')
    t.surface_lift_success.duration[1] = float('nan')
    actions = torch.zeros(2, 24); actions[1, 0] = float('inf')
    packet = sample(d, t, SimpleNamespace(), m, actions=actions)
    assert packet['estimated_torque_nm'] is None
    assert packet['computed_torque_nm'] is None
    assert packet['effort_limit_nm'] is None
    assert packet['torque_valid'] is None
    assert not packet['lift_valid'][0]
    assert not packet['qualified_hold_valid'][1]
    assert not packet['action_valid'][1]
    assert not packet['measurement_valid'][0]


def test_disabled_and_enabled_keep_calls_parameters_rewards_observations_and_rng_equal():
    baseline, wb, tb, _, _ = environment(False)
    observed, wo, to, packets, diagnostics = environment()
    assert observed.enable_evaluation_telemetry(packets.append) is diagnostics
    baseline_parameters = {key: value.clone() for key, value in vars(tb).items()
                           if isinstance(value, torch.Tensor)}
    rng = torch.get_rng_state().clone()
    for _ in range(2):
        before = torch.get_rng_state().clone()
        a = baseline.step(torch.zeros(2, 20)); b = observed.step(torch.zeros(2, 20))
        assert torch.equal(before, torch.get_rng_state())
        for x, y in zip(a[:3], b[:3]): assert torch.equal(x, y)
        assert a[3].keys() == b[3].keys()
        for key in a[3]['terms']: assert torch.equal(a[3]['terms'][key], b[3]['terms'][key])
        assert wb.resets == wo.resets and wb.tick == wo.tick
        for key, value in vars(tb).items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, getattr(to, key)), key
    assert baseline_parameters['low'].eq(tb.low).all()
    assert torch.equal(torch.get_rng_state(), rng)


def test_nonfinite_terminal_is_sampled_before_recovery_and_stays_invalid():
    env, world, task, packets, diagnostics = environment(continuous=True)
    original_advance = world.advance

    def advance(*args):
        m = original_advance(*args)
        if world.tick == 101: m['bottom'][0] = float('nan')
        return m

    world.advance = advance
    world.tick = 100  # isolate numeric recovery from the tick-3 force fault
    env.step(torch.zeros(2, 20))
    assert packets[0]['done'][0]
    assert not packets[0]['measurement_valid'][0]
    assert not packets[0]['lift_valid'][0]
    assert torch.isnan(packets[0]['lift_m'][0])
    assert diagnostics.snapshot()['done'][0]
