"""J2-B pyramid numerical/semantic fixtures, never native accuracy evidence."""
import importlib
import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration


@pytest.fixture(scope='module', autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def implementation():
    name = 'go2_pvcnn.control.dh116.tabletop_stereo_pyramid'
    assert importlib.util.find_spec(name) is not None, 'J2-B pyramid implementation is missing'
    return importlib.import_module(name)


def calibration(width, height):
    k = [[90., 0., width / 2], [0., 85., height / 2], [0., 0., 1.]]
    left = torch.eye(4, dtype=torch.float64)
    right = left.clone(); right[0, 3] = .08
    return StereoCalibration(k, k, left, right, width, height, .08)


def run(a, b=None, *, matcher=None, **changes):
    config = dict(calibration=calibration(a.shape[-1], a.shape[-2]),
        min_disparity=2, max_disparity=24, block_size=5, subpixel=False,
        uniqueness_threshold=.01, texture_threshold=.001)
    config.update(changes)
    return implementation().match_pyramid(a, a if b is None else b, -1,
        matcher=matcher, **config)


def result(a, disparity, *, good=None, refinement=None, boundary=None):
    d = torch.full_like(a, float(disparity))
    g = torch.ones_like(a, dtype=torch.bool) if good is None else good
    r = torch.zeros_like(g) if refinement is None else refinement
    boundary = torch.zeros_like(g) if boundary is None else boundary
    c = torch.where(g, torch.full_like(a, .9), 0.)
    return d, c, g, dict(integer_disparity=d.floor(), integer_confidence=c,
        integer_good=g, refinement_valid=r, search_boundary=boundary,
        local_failed=~g, metric='semantic_spy')


class ScaleSpy:
    def __init__(self, width):
        self.width = width
        self.calls = []

    def __call__(self, a, b, sign, **kw):
        self.calls.append((a.clone(), b.clone(), sign, kw))
        return result(a, 8 * a.shape[-1] / self.width)


def test_actual_odd_size_levels_scale_intrinsics_and_disparity_without_final_resize():
    a = torch.rand(2, 67, 101)
    spy = ScaleSpy(101)
    d, c, g, trace = run(a, matcher=spy)
    assert [tuple(call[0].shape[-2:]) for call in spy.calls] == [
        (16, 25), (33, 50), (33, 50), (67, 101), (67, 101)]
    assert all(call[2] == -1 and call[3]['method'] == 'zncc' for call in spy.calls)
    assert d.shape == c.shape == g.shape == a.shape
    torch.testing.assert_close(d, torch.full_like(a, 8.))
    levels = trace['pyramid']['levels']
    assert list(levels) == ['quarter', 'half', 'full']
    for name, width, height in [('quarter', 25, 16), ('half', 50, 33), ('full', 101, 67)]:
        level = levels[name]
        sx, sy = width / 101, height / 67
        assert level['scale_wh'] == [sx, sy]
        expected = torch.diag(torch.tensor([sx, sy, 1.], dtype=torch.float64)) @ calibration(101, 67).K_left
        torch.testing.assert_close(torch.tensor(level['K_left'], dtype=torch.float64), expected)
        assert level['K_left'] == level['K_right']
        assert level['disparity_range_px'] == [math.floor(2 * sx), min(width - 1, math.ceil(24 * sx))]
    for call, scale in [(spy.calls[1], 50 / 25), (spy.calls[3], 101 / 50)]:
        assert call[3]['search_radius'] == 2
        expected = 8 * call[0].shape[-1] / 101
        torch.testing.assert_close(call[3]['prior'], torch.full_like(call[0], expected))
    assert trace['pyramid']['output_resolution_wh'] == [101, 67]
    assert trace['pyramid']['effective_no_search_speedup'] is True
    assert levels['half']['global_audit_executed'] is True
    assert levels['full']['global_audit_executed'] is True


def test_wrong_but_accepted_local_peak_is_recovered_by_global_audit():
    calls = []
    def spy(a, b, sign, **kw):
        calls.append(kw)
        # A spurious local minimum can be accepted, interior and high confidence.
        return result(a, 1 if a.shape[-1] == 25 or kw.get('prior') is not None else 10)
    d, _, good, trace = run(torch.rand(1, 64, 100), matcher=spy)
    assert good.all() and bool((d == 10).all())
    for name in ('half', 'full'):
        level = trace['pyramid']['levels'][name]
        assert level['global_fallback_mask'].all()
        assert level['local_disagreement'].all()
    assert len(calls) == 5


def test_global_uniqueness_rejection_overrides_confident_local_match():
    def spy(a, b, sign, **kw):
        if a.shape[-1] == 100 and kw.get('prior') is None:
            return result(a, 8, good=torch.zeros_like(a, dtype=torch.bool))
        return result(a, 8 * a.shape[-1] / 100)
    d, c, g, _ = run(torch.rand(1, 64, 100), matcher=spy)
    assert not g.any() and torch.isnan(d).all() and bool((c == 0).all())


def test_invalid_prior_and_failed_local_match_retry_full_range():
    calls = []
    def spy(a, b, sign, **kw):
        calls.append((a.shape, kw))
        if a.shape[-1] == 25:
            good = torch.ones_like(a, dtype=torch.bool); good[..., 5:8] = False
            return result(a, 2, good=good)
        if kw.get('prior') is not None:
            return result(a, 4, good=torch.zeros_like(a, dtype=torch.bool))
        return result(a, 8 * a.shape[-1] / 100)
    d, _, g, trace = run(torch.rand(1, 64, 100), matcher=spy)
    half = trace['pyramid']['levels']['half']
    assert (~half['prior_valid']).any() and half['prior_valid'].any()
    assert torch.isnan(calls[1][1]['prior'][~half['prior_valid']]).all()
    assert half['global_fallback_mask'].all()
    assert g.all() and bool((d == 8).all())


def test_raw_invalid_pixels_and_nan_are_conservatively_downsampled():
    a = torch.rand(1, 65, 103); b = a.clone()
    am = torch.ones_like(a, dtype=torch.bool); bm = am.clone()
    am[..., 7, 11] = False; b[..., 29, 42] = float('nan')
    spy = ScaleSpy(103)
    run(a, b, matcher=spy, a_valid=am, b_valid=bm)
    for ai, bi, _, kw in spy.calls:
        size = ai.shape[-2:]
        expected_a = F.adaptive_max_pool2d((~am).float()[:, None], size)[:, 0] == 0
        expected_b = F.adaptive_max_pool2d((~torch.isfinite(b)).float()[:, None], size)[:, 0] == 0
        torch.testing.assert_close(kw['a_valid'], expected_a)
        torch.testing.assert_close(kw['b_valid'], expected_b)
        assert torch.isfinite(ai).all() and torch.isfinite(bi).all()


def test_nan_disparity_never_becomes_a_valid_upsampled_prior():
    calls = []
    def spy(a, b, sign, **kw):
        calls.append(kw)
        d, c, g, t = result(a, 8 * a.shape[-1] / 100)
        if a.shape[-1] == 25:
            d[..., 3, 7] = float('nan')  # Even inconsistent backend g=True fails closed.
        return d, c, g, t
    run(torch.rand(1, 64, 100), matcher=spy)
    prior = calls[1]['prior']
    expected_bad = F.interpolate(torch.zeros(1, 1, 16, 25).index_put_(
        (torch.tensor([0]), torch.tensor([0]), torch.tensor([3]), torch.tensor([7])), torch.tensor([1.])),
        (32, 50), mode='bilinear', align_corners=False)[:, 0] > 0
    assert torch.isnan(prior[expected_bad]).all()
    assert torch.isfinite(prior[~expected_bad]).all()


def test_small_coarsest_level_is_explicitly_skipped_and_full_resolution_runs():
    spy = ScaleSpy(41)
    d, _, g, trace = run(torch.rand(1, 15, 41), matcher=spy, block_size=5)
    levels = trace['pyramid']['levels']
    assert levels['quarter']['status'] == 'skipped'
    assert levels['quarter']['skip_reason'] == 'image_smaller_than_block_size'
    assert levels['half']['status'] == levels['full']['status'] == 'processed'
    assert trace['pyramid']['processed_level_count'] == 2
    assert d.shape == g.shape == (1, 15, 41)


def test_no_supported_level_fails_closed_without_calling_matcher():
    spy = ScaleSpy(7)
    d, c, g, trace = run(torch.rand(1, 3, 7), matcher=spy, block_size=5, max_disparity=5)
    assert spy.calls == []
    assert not g.any() and torch.isnan(d).all() and bool((c == 0).all())
    assert trace['pyramid']['processed_level_count'] == 0
    assert trace['refinement_valid'].shape == d.shape


def test_final_refinement_metadata_comes_from_final_global_result():
    def spy(a, b, sign, **kw):
        r = torch.full_like(a, kw.get('prior') is None, dtype=torch.bool)
        return result(a, 8 * a.shape[-1] / 100, refinement=r)
    _, _, _, trace = run(torch.rand(1, 64, 100), matcher=spy, subpixel=True)
    assert trace['refinement_valid'].all()
    assert trace['integer_disparity'].shape == (1, 64, 100)
    assert trace['pyramid']['final_refinement_valid'] is trace['refinement_valid']
    assert trace['pyramid']['cost_volume_allocated'] is False


def test_heterogeneous_invalid_environments_preserve_chunking_trace_contract():
    a = torch.rand(2, 64, 100)
    masks = torch.ones_like(a, dtype=torch.bool); masks[1] = False
    def spy(ai, bi, sign, **kw):
        return result(ai, 8 * ai.shape[-1] / 100, good=kw['a_valid'] & kw['b_valid'])
    full = run(a, matcher=spy, a_valid=masks, b_valid=masks)
    chunks = [run(a[i:i + 1], matcher=spy, a_valid=masks[i:i + 1],
        b_valid=masks[i:i + 1]) for i in range(2)]
    def compare(batched, pieces):
        if isinstance(batched, torch.Tensor):
            torch.testing.assert_close(batched, torch.cat(pieces), rtol=0, atol=0, equal_nan=True)
        elif isinstance(batched, dict):
            assert all(set(piece) == set(batched) for piece in pieces)
            for key, value in batched.items():
                compare(value, [piece[key] for piece in pieces])
        else:
            assert all(piece == batched for piece in pieces), 'batch-dependent scalar diagnostics'
    for index, value in enumerate(full):
        compare(value, [chunk[index] for chunk in chunks])


def test_calibration_shape_and_raw_validity_are_validated():
    a = torch.rand(1, 64, 100)
    with pytest.raises(ValueError, match='calibration'):
        run(a, matcher=ScaleSpy(100), calibration=calibration(99, 64))
    with pytest.raises(ValueError, match='valid'):
        run(a, matcher=ScaleSpy(100), a_valid=torch.ones_like(a))


def real_matcher():
    name = 'go2_pvcnn.control.dh116.tabletop_stereo_matching_costs'
    assert importlib.util.find_spec(name) is not None, 'J2-B streaming cost dependency is missing'
    return importlib.import_module(name).match_cost


def shifted_gray(disparity=16):
    a = torch.rand(1, 64, 128, generator=torch.Generator().manual_seed(11623))
    b = torch.zeros_like(a); b[..., :-disparity] = a[..., disparity:]
    return a, b


def test_real_zncc_pyramid_recovers_integer_translation_at_original_size():
    real_matcher()
    a, b = shifted_gray()
    d, _, good, trace = run(a, b, subpixel=True)
    interior = (..., slice(8, -8), slice(32, -8))
    assert float(good[interior].float().mean()) > .8
    assert float((d[interior][good[interior]] - 16).abs().median()) < .1
    assert d.shape == a.shape and trace['pyramid']['processed_level_count'] == 3


def test_real_fine_search_recovers_deliberately_wrong_coarse_prior():
    match = real_matcher()
    a, b = shifted_gray()
    def corrupt_coarsest(a, b, sign, **kw):
        if a.shape[-1] == 32:
            return result(a, 2)  # True quarter disparity=4; the half prior=4 excludes true=8.
        return match(a, b, sign, **kw)
    d, _, good, trace = run(a, b, matcher=corrupt_coarsest)
    interior = (..., slice(8, -8), slice(32, -8))
    assert float(good[interior].float().mean()) > .8
    assert bool((d[interior][good[interior]] == 16).all())
    assert trace['pyramid']['levels']['half']['global_fallback_mask'][..., 5:-5, 12:-5].any()


def test_real_raw_mask_hole_cannot_leak_through_a_matching_patch():
    real_matcher()
    a, b = shifted_gray()
    am = torch.ones_like(a, dtype=torch.bool); am[..., 27, 61] = False
    _, _, good, _ = run(a, b, a_valid=am)
    assert not good[..., 25:30, 59:64].any()
    assert good[..., 8:20, 40:80].any()
