"""Completed native control intervals, using CPU Torch only (no Isaac)."""
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch


MODULE = 'go2_pvcnn.control.dh116.tabletop_surface_feedback'
DTYPE = torch.float64
FIELDS = (
    ('support_n', 2, 10.), ('support_margin', 1, 1.),
    ('support_imbalance', 1, 1.), ('paired', 2, 1.),
    ('ordinary_fraction', 1, 1.), ('pad_depth', 1, .0008),
    ('anchor_drift', 1, .0005), ('slip_speed', 1, .01),
    ('hold_duration', 1, 5.), ('window_good', 1, 1.),
    ('pressure_limited', 2, 1.), ('mimic_blocked', 2, 1.),
    ('wrist_z_request', 2, .01), ('wrist_z_increment', 2, .001),
)


def module():
    assert importlib.util.find_spec(MODULE) is not None, 'surface feedback observer is missing'
    return importlib.import_module(MODULE)


def scene(n=2, dtype=DTYPE):
    def zeros(*tail):
        return torch.zeros(n, *tail, dtype=dtype)
    wrist = zeros(2, 7)
    wrist[..., 3] = 1.
    wrist[..., 2] = .4
    loads = zeros(2, 5)
    loads[..., :2] = 2.
    vectors = zeros(2, 5, 3)
    vectors[..., 0, 0] = 2.
    vectors[..., 1, 0] = -2.
    vectors[..., :2, 2] = -1.
    measurement = dict(
        wrist_pose=wrist, object_relative=zeros(3), object_quat=zeros(4),
        effective_pad_loads_n=loads, effective_pad_vectors_w=vectors,
        effective_pad_valid=loads > 0., effective_pad_depth_m=zeros(2, 5),
        rejected_contact_loads_n=zeros(2), mimic_errors=zeros(10),
        effective_pad_bad_env=torch.zeros(n, dtype=torch.bool),
    )
    measurement['object_quat'][:, 0] = 1.
    task = SimpleNamespace(
        m={k: v.clone() for k, v in measurement.items()},
        done=torch.zeros(n, dtype=torch.bool), actions=zeros(24),
        pressure_limited=torch.zeros(n, 2, dtype=torch.bool),
        f_blocked=torch.zeros(n, 2, dtype=torch.bool),
        surface_lift_success=SimpleNamespace(
            duration=zeros(), measured_drift_m=zeros(),
            _window_good=torch.zeros(n, dtype=torch.bool)),
    )
    world = SimpleNamespace(vertical_components=zeros(2, 2), object_weight_n=zeros() + 10.)
    world.vertical_components[..., 0] = 2.
    world.vertical_components[..., 1] = 3.
    return task, world, measurement


def observer(n=2, dtype=DTYPE):
    return module().SurfaceFeedback(n, 'cpu', dtype=dtype)


def run(o, t, w, m, dt=.01, live=None):
    if live is None:
        live = torch.ones(o.n, dtype=torch.bool)
    o.sample(t, w, m, live, dt)


def test_fixed_schema_and_reset_state():
    o = observer()
    assert module().FEATURES == FIELDS
    assert o.features().shape == (2, 20)
    assert o.features().dtype == DTYPE
    assert o.valid.shape == (2,) and o.valid.dtype == torch.bool
    assert not o.valid.any() and not o.features().any()


def test_exact_columns_signed_support_physical_scales_and_controller_readback():
    o = observer()
    t, w, m = scene()
    t.actions[:, [14, 17]] = torch.tensor([.25, -.5], dtype=DTYPE)
    t.pressure_limited[:, 0] = True
    t.f_blocked[:, 1] = True
    m['mimic_errors'][:, 5] = .012
    m['rejected_contact_loads_n'][:, 1] = 2.
    # Unattributable right-hand friction must disappear; raw total is still 5 N.
    w.vertical_components[:, 1] = torch.tensor([-1., 6.], dtype=DTYPE)
    m['effective_pad_depth_m'][..., 0] = .0004
    t.surface_lift_success.duration[:] = 2.5
    t.surface_lift_success.measured_drift_m[:] = .00025
    t.surface_lift_success._window_good[:] = True
    o.begin(t, w)
    m['wrist_pose'][:, 0, 2] += .0002
    m['wrist_pose'][:, 1, 2] -= .0003
    run(o, t, w, m)
    o.finish(t, w)
    expected = torch.tensor([
        .5, -.1, -.6, 1., 1., 1., .2, .5, .5, 3., .5, 1.,
        1., 0., 0., 1., .25, -.5, .2, -.3,
    ], dtype=DTYPE)
    # Object_relative is held fixed: unequal wrist moves change the measured
    # object center by their mean. Max object-in-wrist movement is 0.25 mm.
    expected[9] = .00025 / .01 / .01
    torch.testing.assert_close(o.features(), expected.expand(2, -1))
    assert o.valid.all()


def test_dt_weighted_full_interval_not_last_sample_or_drift_derivative():
    o = observer()
    t, w, m = scene()
    o.begin(t, w)
    run(o, t, w, m, dt=.01)
    # Move the actual object, with wrist and fixed-anchor drift held constant.
    m['object_relative'][:, 0] += .0006
    w.vertical_components[..., 1] = 5.
    t.actions[:, [14, 17]] = 1.
    t.pressure_limited[:] = True
    t.surface_lift_success.duration[:] = 4.
    t.surface_lift_success._window_good[:] = True
    m['effective_pad_depth_m'][..., 0] = .0016
    run(o, t, w, m, dt=.03)
    # Max over samples, even when the last depth and drift return to zero.
    t.surface_lift_success.measured_drift_m[:] = .001
    run(o, t, w, m, dt=.01)
    t.surface_lift_success.measured_drift_m.zero_()
    m['effective_pad_depth_m'].zero_()
    run(o, t, w, m, dt=.01)
    o.finish(t, w)
    f = o.features()
    torch.testing.assert_close(f[:, :2], torch.full((2, 2), 2./3., dtype=DTYPE))
    torch.testing.assert_close(f[:, 9], torch.full((2,), 1., dtype=DTYPE))
    assert (f[:, 7] == 2.).all() and (f[:, 8] == 2.).all()
    assert (f[:, 10] == .8).all() and (f[:, 11] == 1.).all()
    torch.testing.assert_close(f[:, 12:14], torch.full((2, 2), 5./6., dtype=DTYPE))
    torch.testing.assert_close(f[:, 16:18], torch.full((2, 2), 5./6., dtype=DTYPE))


def test_terminal_sample_included_then_frozen_before_world_or_tracker_reset():
    o = observer()
    t, w, m = scene()
    o.begin(t, w)
    t.done[0] = True  # consume has already detected termination on this sample.
    t.surface_lift_success.duration[0] = 5.
    m['wrist_pose'][0, :, 2] += .001
    run(o, t, w, m, dt=.01)
    t.surface_lift_success.duration.zero_()
    m['wrist_pose'][0].fill_(float('nan'))  # post-terminal physical reset
    w.vertical_components[0].fill_(float('nan'))
    w.vertical_components[1, :, 1] = 5.
    for _ in range(3):
        run(o, t, w, m, dt=.01, live=torch.tensor([False, True]))
    o.finish(t, w)
    assert o.valid.all()
    torch.testing.assert_close(o.features()[:, :2], torch.tensor([[.5, .5], [.65, .65]], dtype=DTYPE))
    assert o.features()[0, 10] == 1.
    torch.testing.assert_close(o.features()[0, 18:], torch.ones(2, dtype=DTYPE))


def test_completed_features_are_published_only_at_finish_and_selected_reset_isolated():
    o = observer()
    t, w, m = scene()
    o.begin(t, w)
    run(o, t, w, m)
    assert not o.features().any()
    o.finish(t, w)
    before = o.features().clone()
    o.begin(t, w)
    w.vertical_components[..., 1] = 9.
    run(o, t, w, m)
    torch.testing.assert_close(o.features(), before)
    o.reset(torch.tensor([True, False]))
    o.finish(t, w)
    assert not o.features()[0].any() and not o.valid[0]
    assert o.valid[1] and o.features()[1, 0] == 1.1
    t.m = {k: v.clone() for k, v in m.items()}
    o.begin(t, w)
    run(o, t, w, m)
    o.finish(t, w)
    assert o.valid.all()


def test_sample_valid_is_immediate_sticky_and_distinct_from_published_valid():
    o = observer()
    t, w, m = scene()
    o.begin(t, w)
    assert o.sample_valid.all() and not o.valid.any()
    run(o, t, w, m)
    assert o.sample_valid.all() and not o.valid.any()
    m['effective_pad_depth_m'][0, 0, 0] = -.001
    run(o, t, w, m)
    assert o.sample_valid.tolist() == [False, True]
    m['effective_pad_depth_m'].zero_()
    run(o, t, w, m)
    assert o.sample_valid.tolist() == [False, True]
    o.finish(t, w)
    assert torch.equal(o.sample_valid, o.valid)
    o.reset([1])
    assert not o.sample_valid.any() and not o.valid.any()


def test_mimic_blocked_requires_actual_block_and_severe_measured_error_on_that_hand():
    o = observer()
    t, w, m = scene()
    t.f_blocked[:] = torch.tensor([[True, True], [False, True]])
    m['mimic_errors'][0, 0] = .011999
    m['mimic_errors'][0, 5] = -.012
    m['mimic_errors'][1, 0] = .02
    # A severe error alone cannot manufacture a guard or block the other hand.
    o.begin(t, w)
    run(o, t, w, m)
    o.finish(t, w)
    torch.testing.assert_close(o.features()[:, 14:16], torch.tensor([[0., 1.], [0., 0.]], dtype=DTYPE))
    assert torch.equal(t.f_blocked, torch.tensor([[True, True], [False, True]]))


@pytest.mark.parametrize('bad', ['nan_load', 'negative_load', 'negative_depth', 'negative_rejected',
    'zero_weight', 'nan_vertical', 'zero_object_quat', 'zero_wrist_quat', 'mask_mismatch',
    'bad_env', 'nan_action', 'nan_hold', 'negative_drift'])
def test_invalid_live_row_zeroes_whole_interval_and_remains_invalid(bad):
    o = observer()
    t, w, m = scene()
    o.begin(t, w)
    run(o, t, w, m)
    if bad == 'nan_load': m['effective_pad_loads_n'][0, 0, 0] = float('nan')
    if bad == 'negative_load': m['effective_pad_loads_n'][0, 0, 0] = -1.
    if bad == 'negative_depth': m['effective_pad_depth_m'][0, 0, 0] = -.001
    if bad == 'negative_rejected': m['rejected_contact_loads_n'][0, 0] = -1.
    if bad == 'zero_weight': w.object_weight_n[0] = 0.
    if bad == 'nan_vertical': w.vertical_components[0, 0, 1] = float('nan')
    if bad == 'zero_object_quat': m['object_quat'][0].zero_()
    if bad == 'zero_wrist_quat': m['wrist_pose'][0, 0, 3:].zero_()
    if bad == 'mask_mismatch': m['effective_pad_valid'][0, 0, 0] = False
    if bad == 'bad_env': m['effective_pad_bad_env'][0] = True
    if bad == 'nan_action': t.actions[0, 14] = float('nan')
    if bad == 'nan_hold': t.surface_lift_success.duration[0] = float('nan')
    if bad == 'negative_drift': t.surface_lift_success.measured_drift_m[0] = -.001
    run(o, t, w, m)
    # A later good sample must never conceal earlier invalid feedback.
    t2, w2, m2 = scene()
    run(o, t2, w2, m2)
    o.finish(t, w)
    assert o.valid.tolist() == [False, True]
    assert not o.features()[0].any() and torch.isfinite(o.features()).all()


def test_no_contact_is_valid_physics_but_not_paired_or_supporting():
    o = observer()
    t, w, m = scene()
    for key in ('effective_pad_loads_n', 'effective_pad_vectors_w', 'effective_pad_valid'):
        m[key].zero_()
    w.vertical_components.zero_()
    o.begin(t, w)
    run(o, t, w, m)
    o.finish(t, w)
    assert o.valid.all()
    assert (o.features()[:, 2] == -1.).all()
    assert not o.features()[:, 4:6].any()


def test_optional_existing_tracker_and_block_flags_have_no_new_control_side_effects():
    o = observer()
    t, w, m = scene()
    del t.f_blocked
    t.surface_slip_tracking = SimpleNamespace(measured_drift_m=torch.full((2,), .0015, dtype=DTYPE))
    t.surface_lift_success.measured_drift_m[:] = .0001
    t.m['effective_pad_loads_n'].fill_(999.)  # consume's snapshot is NOT raw m.
    before = {k: v.clone() for k, v in m.items()}
    action = t.actions.clone()
    o.begin(t, w)
    run(o, t, w, m)
    o.finish(t, w)
    assert (o.features()[:, 8] == 3.).all()
    assert not o.features()[:, 14:16].any()
    assert not hasattr(t, 'f_blocked') and not hasattr(t, 'surface_stable_hold_rewards')
    torch.testing.assert_close(t.actions, action)
    for key in m:
        torch.testing.assert_close(m[key], before[key])


def test_tracker_absence_does_not_enable_rewards_or_fabricate_hold():
    o = observer()
    t, w, m = scene()
    del t.surface_lift_success
    o.begin(t, w)
    run(o, t, w, m)
    o.finish(t, w)
    assert o.valid.all() and not o.features()[:, 8:12].any()
    assert not hasattr(t, 'surface_lift_success') and not hasattr(t, 'surface_slip_tracking')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_dtype_and_interval_baseline_restart(dtype):
    o = observer(dtype=dtype)
    t, w, m = scene(dtype=dtype)
    for _ in range(2):
        o.begin(t, w)
        m['wrist_pose'][..., 2] += .001
        run(o, t, w, m)
        o.finish(t, w)
        assert o.features().dtype == dtype
        torch.testing.assert_close(o.features()[:, 18:], torch.ones(2, 2, dtype=dtype), atol=2e-5, rtol=0.)
        t.m = {k: v.clone() for k, v in m.items()}


@pytest.mark.parametrize('key', ['effective_pad_loads_n', 'effective_pad_vectors_w',
    'effective_pad_depth_m', 'rejected_contact_loads_n', 'object_relative', 'wrist_pose'])
def test_malformed_shapes_and_dtypes_raise_instead_of_broadcasting(key):
    o = observer()
    t, w, m = scene()
    o.begin(t, w)
    m[key] = m[key][:1]
    with pytest.raises(ValueError): run(o, t, w, m)
    t, w, m = scene()
    m[key] = m[key].float()
    with pytest.raises(ValueError): run(o, t, w, m)


@pytest.mark.parametrize('which', ['live_shape', 'live_type', 'pad_mask_type', 'pressure_type',
    'blocked_shape', 'weight_shape', 'actions_shape', 'bad_env_type'])
def test_strict_boolean_masks_and_world_task_shapes(which):
    o = observer()
    t, w, m = scene()
    o.begin(t, w)
    live = torch.ones(2, dtype=torch.bool)
    if which == 'live_shape': live = live[:, None]
    if which == 'live_type': live = live.float()
    if which == 'pad_mask_type': m['effective_pad_valid'] = m['effective_pad_valid'].double()
    if which == 'pressure_type': t.pressure_limited = t.pressure_limited.double()
    if which == 'blocked_shape': t.f_blocked = t.f_blocked[:, :1]
    if which == 'weight_shape': w.object_weight_n = w.object_weight_n[:, None]
    if which == 'actions_shape': t.actions = t.actions[:, :18]
    if which == 'bad_env_type': m['effective_pad_bad_env'] = m['effective_pad_bad_env'].double()
    with pytest.raises(ValueError): run(o, t, w, m, live=live)


@pytest.mark.parametrize('dt', [0., -.01, float('nan'), float('inf'), True])
def test_invalid_dt_rejected(dt):
    o = observer()
    t, w, m = scene()
    o.begin(t, w)
    with pytest.raises(ValueError): run(o, t, w, m, dt=dt)


def test_missing_interval_and_no_live_samples_are_not_valid_transitions():
    o = observer()
    t, w, m = scene()
    with pytest.raises(RuntimeError): run(o, t, w, m)
    with pytest.raises(RuntimeError): o.finish(t, w)
    o.begin(t, w)
    run(o, t, w, m, live=torch.zeros(2, dtype=torch.bool))
    o.finish(t, w)
    assert not o.valid.any() and not o.features().any()


@pytest.mark.parametrize('n', [0, -1, True, 1.5])
def test_constructor_rejects_invalid_count(n):
    with pytest.raises(ValueError): module().SurfaceFeedback(n, 'cpu')
