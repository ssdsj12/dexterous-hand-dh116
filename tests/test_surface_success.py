"""CPU contracts for measured lift success, without a simulator or force goals."""
import importlib
import importlib.util
import math

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_surface_force_sharing import surface_targets

PERIOD = 1 / 60
RAW_DT = 1 / 480
DTYPE = torch.float64


def helper(n=2, reference=.30, **kwargs):
    name = 'go2_pvcnn.control.dh116.tabletop_surface_success'
    assert importlib.util.find_spec(name) is not None, 'surface success helper missing'
    return importlib.import_module(name).SurfaceLiftSuccess(n, 'cpu', reference, **kwargs)


def scene(n=2, dtype=DTYPE):
    obj = torch.zeros(n, 7, dtype=dtype)
    obj[:, 2] = .339
    obj[:, 3] = 1
    wrist = torch.zeros(n, 2, 7, dtype=dtype)
    wrist[..., 3] = 1
    loads = torch.tensor([[2., .2, 1.8, 0., 0.], [1.1, .11, .99, 0., 0.]],
                         dtype=dtype).expand(n, 2, 5).clone()
    normals = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.],
                            [0., 0., 0.], [0., 0., 0.]], dtype=dtype)
    return dict(object_pose=obj, wrist_pose=wrist, loads=loads,
                vectors=loads[..., None] * normals,
                depth=torch.zeros_like(loads), ordinary=torch.zeros(n, dtype=dtype),
                bottom=torch.full((n,), .331, dtype=dtype),
                object_velocity=torch.zeros(n, 6, dtype=dtype),
                valid=torch.ones(n, dtype=torch.bool))


def advance(h, data, steps=8, dt=RAW_DT):
    for _ in range(steps):
        result = h.update(**data, dt=dt)
    return result


def test_unequal_digit_and_hand_forces_succeed_only_after_actual_five_seconds():
    h, data = helper(), scene()
    assert not surface_targets(data['loads'], data['vectors'])['balanced'].any()
    before = {key: value.clone() for key, value in data.items()}
    result = advance(h, data, 2399)
    assert not result['updated'] and not result['event'].any()
    torch.testing.assert_close(h.duration, torch.full((2,), 299 * PERIOD, dtype=DTYPE))
    result = advance(h, data, 1)
    assert set(result) == {'duration', 'event', 'updated', 'paired', 'window_good', 'drift_m'}
    assert result['updated'] and result['paired'].all() and result['window_good'].all()
    assert result['event'].tolist() == [True, True]
    torch.testing.assert_close(result['duration'], torch.full((2,), 5., dtype=DTYPE), atol=1e-12, rtol=0)
    assert not advance(h, data)['event'].any()  # A new-success pulse, not a repeated reward.
    for key in data:
        torch.testing.assert_close(data[key], before[key])


def test_alternating_single_face_contacts_use_mean8_and_owned_samples():
    h, data = helper(), scene()
    for i in range(8):
        data['loads'].zero_()
        data['vectors'].zero_()
        digit = i % 2
        data['loads'][..., digit] = 4. if digit == 0 else .4
        data['vectors'][..., digit, 0] = 4. if digit == 0 else -.4
        assert not surface_targets(data['loads'], data['vectors'])['paired'].any()
        result = advance(h, data, 1)
        assert result['updated'] == (i == 7)
    assert result['paired'].all() and result['window_good'].all()
    data['loads'].zero_()
    data['vectors'].zero_()
    result = advance(h, data)
    assert not result['paired'].any() and not result['window_good'].any()  # No extra EMA.


def test_real_dt_weights_loads_and_vectors_instead_of_sample_count():
    h, data = helper(), scene()
    data['loads'].zero_(); data['vectors'].zero_()
    data['loads'][..., 0] = .6; data['vectors'][..., 0, 0] = .6
    assert not advance(h, data, 1)['updated']
    data['loads'].zero_(); data['vectors'].zero_()
    data['loads'][..., 1] = .08; data['vectors'][..., 1, 0] = -.08
    result = advance(h, data, 1, dt=7 * RAW_DT)
    assert result['updated'] and result['paired'].all() and result['window_good'].all()


def test_raised_reference_is_not_a_lift_and_vector_reference_is_copied():
    ref = torch.tensor([.4, .8], dtype=DTYPE)
    h, data = helper(reference=ref), scene()
    data['bottom'].copy_(ref)
    ref.zero_()
    assert not advance(h, data)['window_good'].any()
    data['bottom'] += .03
    assert advance(h, data)['window_good'].all()


@pytest.mark.parametrize('fault', ['height', 'depth', 'negative_depth', 'speed', 'valid'])
def test_every_raw_physical_frame_counts_and_breaks_only_its_environment(fault):
    h, data = helper(), scene()
    advance(h, data)
    advance(h, data, 3)
    damaged = {key: value.clone() for key, value in data.items()}
    if fault == 'height': damaged['bottom'][0] = .32999
    if fault == 'depth': damaged['depth'][0, 1, 4] = .00080001
    if fault == 'negative_depth': damaged['depth'][0, 1, 4] = -.00001
    if fault == 'speed': damaged['object_velocity'][0, 0] = .03
    if fault == 'valid': damaged['valid'][0] = False
    result = advance(h, damaged, 1)
    assert not result['updated']
    torch.testing.assert_close(h.duration, torch.full((2,), PERIOD, dtype=DTYPE))
    result = advance(h, data, 4)
    assert result['window_good'].tolist() == [False, True]
    torch.testing.assert_close(h.duration, torch.tensor([0., 2 * PERIOD], dtype=DTYPE))
    assert advance(h, data)['window_good'].all()


def test_ordinary_uses_period_mean_and_total_contact_ratio():
    h, data = helper(), scene()
    data['ordinary'][:] = 1.
    advance(h, data, 1)
    data['ordinary'].zero_()
    assert advance(h, data, 7)['window_good'].all()
    data['ordinary'] = data['loads'].sum((1, 2)) / 19
    assert advance(h, data)['window_good'].all()  # Inclusive 5% of all contacts.
    data['ordinary'] *= 1.01
    assert not advance(h, data)['window_good'].any()


def test_ordinary_is_weighted_by_real_dt():
    h, data = helper(), scene()
    data['ordinary'] = data['loads'].sum((1, 2)) * .1
    advance(h, data, 1)
    data['ordinary'].zero_()
    assert advance(h, data, 1, dt=7 * RAW_DT)['window_good'].all()


def test_opposing_pair_is_required_in_both_hands():
    h, data = helper(), scene()
    data['vectors'][0, 0, :, 0].abs_()
    data['loads'][1, 1].zero_(); data['vectors'][1, 1].zero_()
    result = advance(h, data)
    assert result['paired'].tolist() == [[False, True], [True, False]]
    assert not result['window_good'].any()


@pytest.mark.parametrize('motion', ['wrist_translation', 'object_rotation', 'wrist_rotation', 'axial_rotation'])
def test_returning_raw_slip_cannot_be_hidden_by_final_pose(motion):
    h, data = helper(), scene()
    advance(h, data)
    damaged = {key: value.clone() for key, value in data.items()}
    if motion == 'wrist_translation': damaged['wrist_pose'][0, 1, 0] += .0006
    if motion == 'object_rotation':
        damaged['object_pose'][0, 3:] = torch.tensor([math.cos(.002), 0., 0., math.sin(.002)])
    if motion == 'wrist_rotation':
        damaged['wrist_pose'][0, 1, 3:] = torch.tensor([math.cos(.002), 0., 0., math.sin(.002)])
    if motion == 'axial_rotation':
        damaged['object_pose'][0, 3:] = torch.tensor([math.cos(.04), 0., math.sin(.04), 0.])
    advance(h, damaged, 1)
    # Mutating the original input cannot erase a collected excursion.
    for key in damaged: damaged[key].copy_(data[key])
    result = advance(h, data, 7)
    assert result['drift_m'][0] > .0005 and result['window_good'].tolist() == [False, True]
    assert h.duration[0] == 0
    assert advance(h, data)['window_good'].all()


def test_anchor_does_not_follow_cumulative_slip_but_common_transport_is_allowed():
    h, data = helper(), scene()
    advance(h, data)
    for step in range(3):
        data['object_pose'][0, 0] += .0002
        data['object_pose'][1, :3] += .01
        data['wrist_pose'][1, :, :3] += .01
        result = advance(h, data)
        assert result['window_good'][0].item() == (step < 2)
        assert result['window_good'][1]


@pytest.mark.parametrize('mid_interval', [False, True])
def test_partial_reset_clears_only_selected_rows_and_discards_their_pending_interval(mid_interval):
    h, data = helper(), scene()
    advance(h, data, 16)
    if mid_interval: advance(h, data, 4)
    h.reset(torch.tensor([0]))
    assert h.duration[0] == 0 and h.duration[1] == pytest.approx(2 * PERIOD)
    # Pre-reset geometry may be arbitrarily far from the new episode's pose.
    data['wrist_pose'][0, :, 0] += 1.
    result = advance(h, data, 4 if mid_interval else 8)
    assert result['window_good'].tolist() == [not mid_interval, True]
    assert h.duration[0] == pytest.approx(0. if mid_interval else PERIOD)
    assert h.duration[1] == pytest.approx(3 * PERIOD)
    assert advance(h, data)['window_good'].all()
    h.reset(torch.tensor([True, False]))
    assert h.duration[0] == 0 and h.duration[1] == pytest.approx(4 * PERIOD)


@pytest.mark.parametrize('field', [key for key in scene() if key != 'valid'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_raw_measurements_are_isolated_and_recovery_is_finite(field, bad):
    h, data = helper(), scene()
    advance(h, data)
    damaged = {key: value.clone() for key, value in data.items()}
    damaged[field][0].reshape(-1)[0] = bad
    advance(h, damaged, 1)
    result = advance(h, data, 7)
    assert result['window_good'].tolist() == [False, True]
    assert torch.isfinite(result['duration']).all() and torch.isfinite(result['drift_m']).all()
    assert advance(h, data)['window_good'].all()


def test_zero_quaternion_and_negative_loads_do_not_poison_other_rows():
    h, data = helper(4), scene(n=4)
    data['object_pose'][0, 3:] = 0.
    data['wrist_pose'][1, 0, 3:] = 0.
    data['loads'][2, 0, 0] = -1.
    assert advance(h, data)['window_good'].tolist() == [False, False, False, True]


@pytest.mark.parametrize('dt', [0., -RAW_DT, float('nan'), float('inf')])
def test_invalid_dt_raises_before_changing_duration_or_pending_samples(dt):
    h, data = helper(), scene()
    advance(h, data, 4)
    with pytest.raises(ValueError): h.update(**data, dt=dt)
    assert advance(h, data, 4)['window_good'].all()


def test_custom_control_period_and_float32():
    h, data = helper(1, dtype=torch.float32, control_period_s=1 / 120), scene(1, torch.float32)
    result = advance(h, data, 4)
    assert result['updated'] and h.duration.dtype == torch.float32
    torch.testing.assert_close(h.duration, torch.tensor([1 / 120], dtype=torch.float32))


def test_substep_crossing_a_period_boundary_retains_only_its_remainder():
    h, data = helper(), scene()
    assert not advance(h, data, 1, dt=.75 * PERIOD)['updated']
    result = advance(h, data, 1, dt=.5 * PERIOD)
    assert result['updated']
    torch.testing.assert_close(h.duration, torch.full((2,), PERIOD, dtype=DTYPE))
    result = advance(h, data, 1, dt=.75 * PERIOD)
    assert result['updated']
    torch.testing.assert_close(h.duration, torch.full((2,), 2 * PERIOD, dtype=DTYPE))


@pytest.mark.parametrize('dt', [.02, .5, 1.])
def test_multi_period_dt_splits_constant_sample_and_keeps_remainder(dt):
    h, data = helper(), scene()
    result = advance(h, data, 1, dt=dt)
    full_periods = math.floor(dt / PERIOD + 1e-9)
    assert result['updated'] and result['window_good'].all() and not result['event'].any()
    torch.testing.assert_close(h.duration, torch.full((2,), full_periods * PERIOD, dtype=DTYPE))
    remainder = dt - full_periods * PERIOD
    advance(h, data, 1, dt=PERIOD - remainder)
    torch.testing.assert_close(h.duration, torch.full((2,), (full_periods + 1) * PERIOD, dtype=DTYPE))


def test_event_survives_later_flushes_in_the_same_multi_period_update():
    h, data = helper(), scene()
    assert not advance(h, data, 4, dt=1.)['event'].any()
    result = advance(h, data, 1, dt=1.5)
    assert result['event'].all() and result['window_good'].all()
    torch.testing.assert_close(h.duration, torch.full((2,), 5.5, dtype=DTYPE), atol=1e-12, rtol=0)
    assert not advance(h, data, 1, dt=.5)['event'].any()


def test_partial_reset_invalidates_only_first_flush_of_a_long_update():
    h, data = helper(), scene()
    advance(h, data, 12)
    h.reset([0])
    data['wrist_pose'][0, :, 0] += 1.
    result = advance(h, data, 1, dt=2.5 * PERIOD)
    assert result['window_good'].all()
    torch.testing.assert_close(h.duration, torch.tensor([2 * PERIOD, 4 * PERIOD], dtype=DTYPE))
