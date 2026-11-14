"""RGB-only analytic contracts for the experimental streaming costs."""
import importlib
import math

import pytest
import torch


torch.set_num_threads(4)


def engine():
    name = 'go2_pvcnn.control.dh116.tabletop_stereo_matching_costs'
    assert importlib.util.find_spec(name) is not None, 'streaming cost engine is required'
    return importlib.import_module(name).match_cost


def image_pair(disparity=3., height=23, width=75):
    y = torch.arange(height, dtype=torch.float64)[None, :, None]
    x = torch.arange(width, dtype=torch.float64)[None, None, :]

    def signal(u):
        return (.48 + .13 * torch.sin(.61 * u + .33 * y)
                + .11 * torch.cos(.37 * u - .51 * y)
                + .08 * torch.sin(1.17 * u + .73 * y)
                + .05 * torch.cos(.19 * u * 1.07 + .83 * y))

    return signal(x).float(), signal(x + disparity).float()


def run(a, b, metric, **kwargs):
    options = dict(method=metric, min_disparity=1, max_disparity=8,
                   block_size=5, subpixel=False, uniqueness_threshold=.01,
                   texture_threshold=.001)
    options.update(kwargs)
    return engine()(a, b, -1, **options)


def core(value):
    return value[:, 6:-6, 17:-10]


@pytest.mark.parametrize('method', ['census', 'zncc', 'gradient'])
def test_known_shift_and_opposite_sign_positive_disparity(method):
    a, b = image_pair()
    d, confidence, good, trace = run(a, b, method)
    assert core(good).float().mean() > .95
    assert torch.equal(core(d)[core(good)], torch.full_like(core(d)[core(good)], 3.))
    reverse = engine()(b, a, 1, method=method, min_disparity=1, max_disparity=8,
                       block_size=5, subpixel=False, uniqueness_threshold=.01,
                       texture_threshold=.001)
    assert core(reverse[2]).float().mean() > .95
    assert torch.equal(core(reverse[0])[core(reverse[2])],
                       torch.full_like(core(reverse[0])[core(reverse[2])], 3.))
    assert torch.isfinite(confidence).all()
    assert torch.isnan(d[~good]).all()
    assert torch.equal(confidence[~good], torch.zeros_like(confidence[~good]))
    assert trace['integer_good'].dtype == torch.bool


@pytest.mark.parametrize(('method', 'transform'), [
    ('zncc', lambda b: .63 * b + .17),
    ('census', lambda b: b.pow(3) + .1),
    ('gradient', lambda b: b + .23),
])
def test_metric_illumination_invariance(method, transform):
    a, b = image_pair()
    baseline = run(a, b, method)
    altered = run(a, transform(b), method)
    accepted = core(baseline[2]) & core(altered[2])
    assert accepted.float().mean() > .95
    assert torch.equal(core(baseline[0])[accepted], core(altered[0])[accepted])
    assert torch.allclose(core(baseline[3]['best_cost'])[accepted],
                          core(altered[3]['best_cost'])[accepted], atol=2e-5)


@pytest.mark.parametrize('method', ['census', 'zncc', 'gradient'])
def test_known_fractional_shift_uses_metric_evidence_or_explicit_integer_fallback(method):
    a, b = image_pair(3.25)
    d, _, good, trace = run(a, b, method, subpixel=True)
    refinement = core(trace['refinement_valid'])
    assert core(good).float().mean() > .85
    if method == 'census':
        assert refinement.float().mean() > .05
    else:
        assert refinement.float().mean() > .7
    assert (core(d)[refinement] - 3.25).abs().quantile(.95) <= .25
    fallback = good & ~trace['refinement_valid']
    assert torch.equal(d[fallback], trace['integer_disparity'][fallback])
    assert trace['subpixel_policy']['fallback'] == 'integer'
    assert trace['metric'] == method


@pytest.mark.parametrize('method', ['census', 'zncc', 'gradient'])
@pytest.mark.parametrize('case', ['flat', 'low_variance', 'periodic'])
def test_flat_low_variance_and_repeated_texture_rejected(method, case):
    if case == 'periodic':
        x = torch.arange(75, dtype=torch.float64)[None, None, :]
        a = (.5 + .3 * torch.cos(2 * math.pi * x / 3.7)).expand(1, 23, 75).float()
        b = (.5 + .3 * torch.cos(2 * math.pi * (x + 3.25) / 3.7)).expand_as(a).float()
    else:
        a, b = image_pair()
        a = torch.full_like(a, .5) if case == 'flat' else .5 + 1e-6 * a
        b = torch.full_like(b, .5) if case == 'flat' else .5 + 1e-6 * b
    d, confidence, good, _ = run(a, b, method, subpixel=True, uniqueness_threshold=0.)
    assert not core(good).any()
    assert torch.isnan(core(d)).all()
    assert not core(confidence).any()


@pytest.mark.parametrize('method', ['census', 'zncc', 'gradient'])
def test_raw_mask_support_and_fill_independence(method):
    a, b = image_pair(3.25)
    ma, mb = torch.ones_like(a, dtype=torch.bool), torch.ones_like(b, dtype=torch.bool)
    ma[:, 10, 35] = False
    mb[:, 13, 44] = False
    original = run(a, b, method, a_valid=ma, b_valid=mb, subpixel=True)
    fill_a, fill_b = a.clone(), b.clone()
    fill_a[~ma], fill_b[~mb] = float('nan'), float('inf')
    filled = run(fill_a, fill_b, method, a_valid=ma, b_valid=mb, subpixel=True)
    for index in (0, 1):
        assert torch.allclose(original[index], filled[index], equal_nan=True)
    assert torch.equal(original[2], filled[2])
    assert not original[2][:, 8:13, 33:38].any()
    if method == 'census':
        assert not original[2][:, 6:15, 31:40].any()
    if method == 'gradient':
        # Central x/y stencils add a cross-shaped support border.
        assert not original[2][:, 7:8, 33:38].any()
        assert not original[2][:, 8:13, 32:33].any()
        assert not original[2][:, 13:14, 33:38].any()
        assert not original[2][:, 8:13, 38:39].any()


@pytest.mark.parametrize('method', ['census', 'zncc', 'gradient'])
def test_nonfinite_is_missing_not_zero_filled_signal(method):
    a, b = image_pair()
    a[:, 10, 35] = float('nan')
    b[:, 14, 40] = float('inf')
    d, c, good, trace = run(a, b, method, subpixel=True)
    assert torch.isnan(d[~good]).all()
    assert not good[:, 8:13, 33:38].any()
    assert torch.isfinite(c).all()
    assert not torch.isnan(trace['best_cost']).any()


def test_local_prior_missing_outside_and_boundary_report_fail_closed():
    a, b = image_pair()
    prior = torch.full_like(a, 3.2)
    prior[:, :, 20:23] = float('nan')
    prior[:, :, 25:28] = 50.
    d, _, good, trace = run(a, b, 'zncc', prior=prior, search_radius=2)
    assert trace['local_failed'][:, :, 20:23].all()
    assert trace['local_failed'][:, :, 25:28].all()
    assert not good[:, :, 20:23].any()
    assert not good[:, :, 25:28].any()
    assert core(good)[:, :, 14:].float().mean() > .8
    assert torch.isnan(d[~good]).all()
    edge = run(a, b, 'zncc', prior=torch.full_like(a, 5.), search_radius=2)
    assert core(edge[3]['search_boundary']).all()
    assert core(edge[3]['local_failed']).all()


@pytest.mark.parametrize('method', ['census', 'zncc', 'gradient'])
def test_runtime_metadata_and_trace_do_not_keep_disparity_volume(method):
    a, b = image_pair()
    result = run(a, b, method, subpixel=True)
    metadata = result[3]['resource_metadata']
    assert metadata['cost_volume_allocated'] is False
    assert metadata['disparity_planes_in_memory'] == 1
    assert metadata['cached_cost_state_planes'] is None
    assert metadata['cost_state_bound_independent_of_disparity'] is True

    def visit(value):
        if isinstance(value, torch.Tensor):
            assert value.shape == a.shape
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)

    visit(result[3])


@pytest.mark.parametrize('kwargs', [
    {'method': 'sad'}, {'min_disparity': -1}, {'max_disparity': 0},
    {'min_disparity': 1.5}, {'block_size': 4}, {'subpixel': 1},
    {'uniqueness_threshold': float('nan')}, {'uniqueness_threshold': 1.},
    {'texture_threshold': -1.}, {'search_radius': -1},
])
def test_invalid_parameters_raise(kwargs):
    a, b = image_pair()
    with pytest.raises(ValueError):
        run(a, b, 'zncc', **kwargs)


def test_invalid_masks_shapes_dtypes_sign_and_prior_raise():
    a, b = image_pair()
    for kwargs in ({'a_valid': torch.ones_like(a)}, {'b_valid': torch.ones_like(b[..., :-1], dtype=torch.bool)},
                   {'prior': torch.ones_like(a[..., :-1])}):
        with pytest.raises(ValueError):
            run(a, b, 'zncc', **kwargs)
    with pytest.raises(ValueError):
        run(a, b.double(), 'zncc')
    with pytest.raises(ValueError):
        run(a[0], b[0], 'zncc')
    with pytest.raises(ValueError):
        engine()(a, b, 0, method='zncc', min_disparity=1, max_disparity=8,
                 block_size=5, subpixel=False, uniqueness_threshold=.01, texture_threshold=.001)


def test_gradient_is_declared_central_difference_l1_metric():
    a, b = image_pair()
    module = importlib.import_module(engine().__module__)
    valid = torch.ones_like(a, dtype=torch.bool)
    metric = module._Metric(a, b, valid, valid, 'gradient', 3, -1, .001)
    y, x, disparity = 12, 30, 4
    expected = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            row, column = y + dy, x + dx
            other = column - disparity
            ax = .5 * (a[0, row, column + 1] - a[0, row, column - 1])
            ay = .5 * (a[0, row + 1, column] - a[0, row - 1, column])
            bx = .5 * (b[0, row, other + 1] - b[0, row, other - 1])
            by = .5 * (b[0, row + 1, other] - b[0, row - 1, other])
            expected.append(.5 * ((ax - bx).abs() + (ay - by).abs()))
    torch.testing.assert_close(metric.integer_cost(disparity)[0, y, x], torch.stack(expected).mean())


def test_zncc_rejects_variance_at_declared_floor_even_with_zero_texture_threshold():
    a, b = image_pair(3.25)
    a, b = .5 + a * 1e-4, .5 + b * 1e-4
    result = run(a, b, 'zncc', texture_threshold=0., subpixel=True)
    assert not result[2].any()
    module = importlib.import_module(engine().__module__)
    valid = torch.ones_like(a, dtype=torch.bool)
    metric = module._Metric(a, b, valid, valid, 'zncc', 5, -1, 0.)
    assert not torch.isfinite(metric.integer_cost(3)).any()
    assert not torch.isfinite(metric.fractional_cost(torch.full_like(a, 3.), .25)).any()


@pytest.mark.parametrize('method', ['census', 'zncc', 'gradient'])
def test_fractional_metric_uses_uniform_center_displacement_and_own_metric(method):
    a, b = image_pair(3.25)
    module = importlib.import_module(engine().__module__)
    valid = torch.ones_like(a, dtype=torch.bool)
    metric = module._Metric(a, b, valid, valid, method, 3, -1, .001)
    winner = torch.full_like(a, 3.)
    y, x = 12, 35
    cost = metric.fractional_cost(winner, .25)[0, y, x]
    winner[:, y - 2:y + 3, x - 2:x + 3] = 7.
    winner[0, y, x] = 3.
    changed_neighbors = metric.fractional_cost(winner, .25)[0, y, x]
    torch.testing.assert_close(cost, changed_neighbors, rtol=0., atol=0.)

    def sample(row, column):
        lo, hi = math.floor(column), math.ceil(column)
        fraction = column - lo
        return b[0, row, lo].double() * (1 - fraction) + b[0, row, hi].double() * fraction

    left_values, right_values, individual = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            row, column = y + dy, x + dx
            other = column - 3.25
            left, right = a[0, row, column].double(), sample(row, other)
            left_values.append(left)
            right_values.append(right)
            if method == 'gradient':
                ax = .5 * (a[0, row, column + 1] - a[0, row, column - 1])
                ay = .5 * (a[0, row + 1, column] - a[0, row - 1, column])
                bx = .5 * (sample(row, other + 1) - sample(row, other - 1))
                by = .5 * (sample(row + 1, other) - sample(row - 1, other))
                individual.append(.5 * ((ax - bx).abs() + (ay - by).abs()))
            if method == 'census':
                mismatch = 0
                for sy in range(-2, 3):
                    for sx in range(-2, 3):
                        if sy == sx == 0:
                            continue
                        mismatch += int((a[0, row + sy, column + sx] < left)
                                        != (sample(row + sy, other + sx) < right))
                individual.append(torch.tensor(mismatch / 24., dtype=torch.float64))
    if method == 'zncc':
        left = torch.stack(left_values); left = left - left.mean()
        right = torch.stack(right_values); right = right - right.mean()
        expected = 1 - ((left * right).sum() / (left.square().sum() * right.square().sum()).sqrt()).clamp(-1, 1)
    else:
        expected = torch.stack(individual).mean()
    torch.testing.assert_close(cost, expected.to(cost.dtype), atol=2e-6, rtol=2e-5)
