"""CPU contracts for causal contact history; no simulator or native objects."""
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch


MODULE = 'go2_pvcnn.control.dh116.tabletop_surface_contact_history'
PERIOD = 1 / 60
RAW_DT = 1 / 480
DTYPE = torch.float64


def history(n=2, **kwargs):
    assert importlib.util.find_spec(MODULE) is not None, 'native surface contact history module missing'
    return importlib.import_module(MODULE).NativeSurfaceHistory(n, 'cpu', **kwargs)


def sample(n=2, scale=1., depth=.0004, dtype=DTYPE):
    loads = torch.arange(1, 11, dtype=dtype).reshape(1, 2, 5).expand(n, 2, 5).clone() * scale
    direction = torch.tensor([1., -2., .5], dtype=dtype)
    return dict(loads=loads, vectors=loads[..., None] * direction,
                depth=torch.full_like(loads, depth))


def features(loads, vectors, depth):
    return torch.cat((loads.log1p().flatten(1), (vectors / 10).flatten(1),
                      (depth / .001).flatten(1)), dim=1)


def advance(h, data, count=8, dt=RAW_DT):
    for _ in range(count):
        h.update(**data, dt=dt)
    return h.features()


def test_fixed_contract_and_zero_initialized_features():
    h = history()
    module = importlib.import_module(MODULE)
    assert module.OBSERVATION_VERSION == 'dh116-surface-sharing-history-v3'
    assert module.CONTRACT == dict(version='native_surface_contact_history_v1', feature_size=50,
                                   control_period_s=PERIOD, ema_alpha=.2, load_log1p=True,
                                   vector_scale_n=10, depth_scale_m=.001)
    result = h.features()
    assert result.shape == (2, 50) and result.dtype == DTYPE and result.device.type == 'cpu'
    assert not result.any()


def test_eight_480hz_samples_produce_mean_then_ema_and_per_pad_max_depth():
    h = history()
    first = sample()
    mean_loads, mean_vectors = torch.zeros_like(first['loads']), torch.zeros_like(first['vectors'])
    max_depth = torch.zeros_like(first['depth'])
    for tick in range(8):
        data = sample(scale=tick + 1, depth=.0001)
        data['depth'][:, tick % 2, tick % 5] = .0002 * (tick + 1)
        mean_loads += data['loads'] / 8
        mean_vectors += data['vectors'] / 8
        max_depth = torch.maximum(max_depth, data['depth'])
        h.update(**data, dt=RAW_DT)
        if tick < 7:
            assert not h.features().any()
    torch.testing.assert_close(h.features(), features(.2 * mean_loads, .2 * mean_vectors, max_depth))
    second = sample(scale=.5, depth=.00005)
    advance(h, second)
    torch.testing.assert_close(h.features(), features(.8 * .2 * mean_loads + .2 * second['loads'],
                                                     .8 * .2 * mean_vectors + .2 * second['vectors'],
                                                     second['depth']))


def test_unfinished_period_cannot_leak_load_vector_or_depth_features():
    h, old, future = history(), sample(), sample(scale=100., depth=.01)
    previous = advance(h, old).clone()
    advance(h, future, count=7)
    assert torch.equal(h.features(), previous)
    h.update(**future, dt=RAW_DT)
    torch.testing.assert_close(h.features(), features(.16 * old['loads'] + .2 * future['loads'],
                                                     .16 * old['vectors'] + .2 * future['vectors'],
                                                     future['depth']))


def test_means_are_weighted_by_actual_dt_instead_of_sample_count():
    h, a, b = history(), sample(scale=2., depth=.0008), sample(scale=6., depth=.0002)
    h.update(**a, dt=PERIOD / 4)
    h.update(**b, dt=3 * PERIOD / 4)
    torch.testing.assert_close(h.features(), features(.2 * (.25 * a['loads'] + .75 * b['loads']),
                                                     .2 * (.25 * a['vectors'] + .75 * b['vectors']),
                                                     a['depth']))


def test_crossing_boundary_splits_sample_and_preserves_only_its_remainder():
    h = history()
    a, b, c = sample(scale=1., depth=.0006), sample(scale=3., depth=.0002), sample(scale=7., depth=.0004)
    h.update(**a, dt=.75 * PERIOD)
    h.update(**b, dt=.5 * PERIOD)
    first_loads = .2 * (.75 * a['loads'] + .25 * b['loads'])
    first_vectors = .2 * (.75 * a['vectors'] + .25 * b['vectors'])
    torch.testing.assert_close(h.features(), features(first_loads, first_vectors, a['depth']))
    h.update(**c, dt=.75 * PERIOD)
    torch.testing.assert_close(h.features(), features(.8 * first_loads + .2 * (.25 * b['loads'] + .75 * c['loads']),
                                                     .8 * first_vectors + .2 * (.25 * b['vectors'] + .75 * c['vectors']),
                                                     c['depth']))


def test_one_update_can_complete_multiple_periods_and_keep_a_partial_period():
    h, a, b = history(), sample(scale=2., depth=.0006), sample(scale=5., depth=.0003)
    h.update(**a, dt=2.5 * PERIOD)
    torch.testing.assert_close(h.features(), features(.36 * a['loads'], .36 * a['vectors'], a['depth']))
    h.update(**b, dt=.5 * PERIOD)
    torch.testing.assert_close(h.features(), features(.8 * .36 * a['loads'] + .2 * (.5 * a['loads'] + .5 * b['loads']),
                                                     .8 * .36 * a['vectors'] + .2 * (.5 * a['vectors'] + .5 * b['vectors']),
                                                     a['depth']))


@pytest.mark.parametrize('selection', ['indices', 'mask', 'list'])
def test_selected_reset_clears_its_partial_clock_and_buffers_without_affecting_sibling(selection):
    h, old = history(), sample()
    pre_reset = sample(scale=40., depth=.009)
    post_reset = sample(scale=2., depth=.0002)
    later = sample(scale=4., depth=.0003)
    previous = advance(h, old).clone()
    advance(h, pre_reset, count=4)
    ids = {'indices': torch.tensor([0]), 'mask': torch.tensor([True, False]), 'list': [0]}[selection]
    h.reset(ids)
    assert not h.features()[0].any()
    assert torch.equal(h.features()[1], previous[1])
    advance(h, post_reset, count=4)
    sibling_loads = .16 * old['loads'] + .2 * (.5 * pre_reset['loads'] + .5 * post_reset['loads'])
    sibling_vectors = .16 * old['vectors'] + .2 * (.5 * pre_reset['vectors'] + .5 * post_reset['vectors'])
    sibling = features(sibling_loads, sibling_vectors, pre_reset['depth'])
    assert not h.features()[0].any()  # A full post-reset period is required.
    torch.testing.assert_close(h.features()[1], sibling[1])
    advance(h, later, count=4)
    fresh = features(.2 * (.5 * post_reset['loads'] + .5 * later['loads']),
                     .2 * (.5 * post_reset['vectors'] + .5 * later['vectors']), later['depth'])
    torch.testing.assert_close(h.features()[0], fresh[0])
    torch.testing.assert_close(h.features()[1], sibling[1])
    advance(h, later, count=4)
    torch.testing.assert_close(h.features()[0], fresh[0])
    torch.testing.assert_close(h.features()[1], features(.8 * sibling_loads + .2 * later['loads'],
                                                       .8 * sibling_vectors + .2 * later['vectors'],
                                                       later['depth'])[1])


def test_reset_at_boundary_and_empty_reset_preserve_other_rows_and_recovery():
    h, data = history(), sample()
    previous = advance(h, data).clone()
    h.reset(torch.empty(0, dtype=torch.long))
    assert torch.equal(h.features(), previous)
    h.reset(torch.tensor([0, 1]))
    assert not h.features().any()
    torch.testing.assert_close(advance(h, data), previous)


@pytest.mark.parametrize('field,bad', [(field, bad) for field in ('loads', 'vectors', 'depth')
                                      for bad in (float('nan'), float('inf'), -float('inf'))]
                         + [('loads', -1.), ('depth', -.0001)])
def test_invalid_row_clears_history_immediately_and_discards_that_whole_period(field, bad):
    h, data = history(), sample()
    advance(h, data)
    advance(h, data, count=3)
    damaged = {key: value.clone() for key, value in data.items()}
    damaged[field][0].reshape(-1)[0] = bad
    sibling = h.features()[1].clone()
    h.update(**damaged, dt=RAW_DT)
    assert not h.features()[0].any()
    assert torch.equal(h.features()[1], sibling)
    advance(h, data, count=4)
    assert not h.features()[0].any()  # Later valid samples cannot redeem the bad period.
    expected = features(.36 * data['loads'], .36 * data['vectors'], data['depth'])
    torch.testing.assert_close(h.features()[1], expected[1])
    advance(h, data)
    torch.testing.assert_close(h.features()[0], features(.2 * data['loads'], .2 * data['vectors'], data['depth'])[0])
    torch.testing.assert_close(h.features()[1], features(.488 * data['loads'], .488 * data['vectors'], data['depth'])[1])
    assert torch.isfinite(h.features()).all()


def test_invalid_constant_sample_spanning_many_periods_cannot_recover_early():
    h, data = history(), sample()
    advance(h, data)
    damaged = {key: value.clone() for key, value in data.items()}
    damaged['vectors'][0, 0, 0, 0] = float('nan')
    h.update(**damaged, dt=2.5 * PERIOD)
    assert not h.features()[0].any()
    torch.testing.assert_close(h.features()[1], features((1 - .8 ** 3) * data['loads'],
                                                       (1 - .8 ** 3) * data['vectors'], data['depth'])[1])
    h.update(**data, dt=.5 * PERIOD)
    assert not h.features()[0].any()
    h.update(**data, dt=PERIOD)
    torch.testing.assert_close(h.features()[0], features(.2 * data['loads'], .2 * data['vectors'], data['depth'])[0])


@pytest.mark.parametrize('field', ['loads', 'vectors', 'depth'])
@pytest.mark.parametrize('fault', ['shape', 'dtype', 'device', 'not_tensor'])
def test_schema_validation_raises_before_changing_any_history_or_pending_samples(field, fault):
    h, data = history(), sample()
    advance(h, data, count=3)
    damaged = {key: value.clone() for key, value in data.items()}
    if fault == 'shape': damaged[field] = damaged[field][:1]
    if fault == 'dtype': damaged[field] = damaged[field].float()
    if fault == 'device': damaged[field] = torch.empty_like(damaged[field], device='meta')
    if fault == 'not_tensor': damaged[field] = damaged[field].tolist()
    before = h.features().clone()
    with pytest.raises(ValueError, match='shape|dtype|device|tensor'):
        h.update(**damaged, dt=PERIOD)
    assert torch.equal(h.features(), before)
    advance(h, data, count=5)
    torch.testing.assert_close(h.features(), features(.2 * data['loads'], .2 * data['vectors'], data['depth']))


@pytest.mark.parametrize('dt', [0., -RAW_DT, float('nan'), float('inf'), -float('inf')])
def test_invalid_dt_raises_before_changing_history_or_partial_buffers(dt):
    h, data = history(), sample()
    advance(h, data, count=3)
    with pytest.raises(ValueError, match='dt'):
        h.update(**data, dt=dt)
    assert not h.features().any()
    advance(h, data, count=5)
    torch.testing.assert_close(h.features(), features(.2 * data['loads'], .2 * data['vectors'], data['depth']))


def test_inputs_and_returned_features_are_owned_and_do_not_build_autograd_graphs():
    h, data = history(), sample()
    for value in data.values():
        value.requires_grad_()
    before = {key: value.detach().clone() for key, value in data.items()}
    h.update(**data, dt=PERIOD / 2)
    for key in data:
        assert torch.equal(data[key], before[key])
    with torch.no_grad():
        data['loads'].mul_(3)
        data['vectors'].mul_(3)
        data['depth'].zero_()
    changed = {key: value.detach().clone() for key, value in data.items()}
    h.update(**data, dt=PERIOD / 2)
    expected = features(.4 * before['loads'], .4 * before['vectors'], before['depth'])
    actual = h.features()
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    actual.zero_()
    torch.testing.assert_close(h.features(), expected)
    h.reset([0])
    for key in data:
        assert torch.equal(data[key], changed[key])


def test_custom_period_and_float32_preserve_feature_dtype_and_fixed_contract():
    h, data = history(dtype=torch.float32, period=1 / 120), sample(dtype=torch.float32)
    advance(h, data, count=3)
    assert not h.features().any()
    actual = advance(h, data, count=1)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, features(.2 * data['loads'], .2 * data['vectors'], data['depth']))
    assert importlib.import_module(MODULE).CONTRACT['control_period_s'] == PERIOD


def test_explicit_enable_uses_task_dtype_and_rebuilds_an_owned_observation():
    h = history()  # Missing module is a test failure before implementation upload.
    module = importlib.import_module(MODULE)
    task = SimpleNamespace(n=2, device='cpu', command=torch.zeros(2, 12, dtype=torch.float32),
                           last_observation=torch.ones(2, 351))
    observed = []

    def observe():
        value = torch.cat((torch.ones(2, 351), task.surface_contact_history.features()), dim=1)
        observed.append(value)
        return value

    task.observe = observe
    attached = module.enable_contact_history(task)
    assert isinstance(attached, module.NativeSurfaceHistory)
    assert attached is task.surface_contact_history and attached is not h
    assert attached.features().dtype == task.command.dtype
    assert len(observed) == 1 and task.last_observation.shape == (2, 401)
    assert torch.equal(task.last_observation, observed[0])
    assert task.last_observation.data_ptr() != observed[0].data_ptr()
    assert not task.last_observation[:, 351:].any()
    assert not task.command.any()


@pytest.mark.parametrize('kwargs', [dict(n=0), dict(n=-1), dict(n=True), dict(n=1.5),
                                   dict(period=0), dict(period=-PERIOD), dict(period=float('nan')),
                                   dict(period=float('inf')), dict(dtype=torch.int64)])
def test_invalid_construction_is_rejected(kwargs):
    with pytest.raises(ValueError):
        history(**kwargs)
