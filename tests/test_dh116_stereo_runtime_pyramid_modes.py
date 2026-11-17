"""CPU numerical/orchestration tests; local-search confidence is not global proof."""
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_stereo_matching_costs as costs
from go2_pvcnn.control.dh116 import tabletop_stereo_pyramid as pyramid
from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth


@pytest.fixture(scope='module', autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(min(old, 4))
    yield
    torch.set_num_threads(old)


def calibration(w=128, h=40):
    k = [[80., 0., w / 2], [0., 80., h / 2], [0., 0., 1.]]
    left = torch.eye(4, dtype=torch.float64)
    right = left.clone(); right[0, 3] = .08
    return StereoCalibration(k, k, left, right, w, h, .08)


def images(batch=1, h=40, w=128, disparity=16):
    a = torch.rand(batch, h, w, generator=torch.Generator().manual_seed(11691))
    b = torch.zeros_like(a); b[..., :-disparity] = a[..., disparity:]
    return a, b


def call(a, b, mode='audit', **changes):
    kw = dict(calibration=calibration(a.shape[-1], a.shape[-2]), min_disparity=2,
              max_disparity=24, block_size=5, subpixel=False,
              uniqueness_threshold=.01, texture_threshold=.001)
    kw.update(changes)
    if 'pyramid_mode' in inspect.signature(pyramid.match_pyramid).parameters:
        kw['pyramid_mode'] = mode
    return pyramid.match_pyramid(a, b, -1, **kw)


def match(a, b, **changes):
    kw = dict(method='zncc', min_disparity=2, max_disparity=24,
              block_size=5, subpixel=False, uniqueness_threshold=.01,
              texture_threshold=.001)
    kw.update(changes)
    accepted = inspect.signature(costs.match_cost).parameters
    return costs.match_cost(a, b, -1, **{k: v for k, v in kw.items() if k in accepted})


def frozen(name):
    path = Path(__file__).resolve().parents[1] / 'artifacts/stereo-runtime-repair-20261111/pyramid' / (name + '.py')
    if not path.exists():
        pytest.skip('private frozen parent source is optional outside this repair workspace')
    spec = importlib.util.spec_from_file_location('go2_pvcnn.control.dh116.' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_default_matches_frozen_parent_actual_outputs_exactly():
    a, b = images(h=25, w=80, disparity=8)
    current = call(a, b, subpixel=True)
    old = frozen('frozen_pyramid').match_pyramid(a, b, -1,
        calibration=calibration(80, 25), min_disparity=2, max_disparity=24,
        block_size=5, subpixel=True, uniqueness_threshold=.01,
        texture_threshold=.001, matcher=frozen('frozen_matching_costs').match_cost)
    for new, prior in zip(current[:3], old[:3]):
        torch.testing.assert_close(new, prior, rtol=0, atol=0, equal_nan=True)


def test_local_cost_engine_actually_prunes_integer_loop(monkeypatch):
    a, b = images(h=24, w=64, disparity=8)
    calls = []
    original = costs._Metric.integer_cost
    def record(self, disparity):
        calls.append(disparity)
        return original(self, disparity)
    monkeypatch.setattr(costs._Metric, 'integer_cost', record)
    _, _, good, trace = match(a, b, prior=torch.full_like(a, 8.), prune_search=True)
    assert len(calls) == 10  # five integer candidates, two scans, not 2*D
    assert set(calls) == {6, 7, 8, 9, 10}
    assert good[..., 5:-5, 16:-5].any()
    assert trace['search_statistics']['integer_plane_calls'].item() == len(calls)
    assert trace['search_statistics']['integer_cost_evaluations'].item() == len(calls) * a.numel()


def test_candidate_eligibility_masks_never_enable_other_pixels_or_environments():
    a, b = images(batch=2, h=24, w=64, disparity=8)
    mask = torch.zeros_like(a, dtype=torch.bool)
    mask[0, 8:16, 20:40] = True
    _, _, good, trace = match(a, b, prior=torch.full_like(a, 8.),
                             search_mask=mask, prune_search=True)
    assert not good[~mask].any()
    stat = trace['search_statistics']
    assert stat['integer_plane_calls'].tolist() == [10, 0]
    assert not stat['integer_candidates'][~mask].any()
    assert stat['integer_candidates'][mask].unique().tolist() == [10]


def test_subpixel_search_reports_actual_counts_and_skips_empty_environment():
    a, b = images(batch=2, h=16, w=48, disparity=8)
    mask = torch.ones_like(a, dtype=torch.bool); mask[1] = False
    _, _, _, trace = match(a, b, max_disparity=16, subpixel=True,
                           search_mask=mask, prune_search=True)
    stat = trace['search_statistics']
    assert stat['fractional_plane_calls'][0] > 0
    assert stat['fractional_plane_calls'][1] == 0
    assert stat['fractional_cost_evaluations'][1] == 0
    assert not stat['fractional_candidates'][1].any()
    assert torch.equal(stat['total_cost_evaluations'], stat['integer_cost_evaluations'] + stat['fractional_cost_evaluations'])


def result(a, d, *, confidence=.9, good=None, boundary=False, tie=False):
    g = torch.ones_like(a, dtype=torch.bool) if good is None else good
    disparity = torch.full_like(a, float(d))
    c = torch.where(g, torch.full_like(a, confidence), 0.)
    flag = torch.full_like(g, boundary)
    return disparity, c, g, dict(integer_disparity=disparity,
        integer_confidence=c, integer_good=g, refinement_valid=g,
        search_boundary=flag, local_failed=~g,
        reasons={'fractional_tie': torch.full_like(g, tie)})


def test_accelerated_trusted_prior_uses_local_only_and_explicitly_disclaims_global_certification():
    seen = []
    def spy(a, b, sign, **kw):
        seen.append((a.shape[-1], kw))
        return result(a, 16 * a.shape[-1] / 128)
    _, _, good, trace = call(*images(), mode='accelerated', matcher=spy)
    assert len(seen) == 3
    assert good.all()
    assert all(kw['prune_search'] is True for _, kw in seen[1:])
    assert trace['pyramid']['mode'] == 'accelerated'
    assert trace['pyramid']['global_certification'] is False
    full = trace['pyramid']['levels']['full']
    assert not full['global_audit_executed']
    assert not full['global_fallback_mask'].any()


@pytest.mark.parametrize('failure', ['invalid', 'weak', 'boundary', 'tie'])
def test_accelerated_untrusted_local_result_falls_back_globally(failure):
    seen = []
    def spy(a, b, sign, **kw):
        seen.append((a.shape[-1], kw))
        if kw.get('prior') is not None:
            return result(a, 4, confidence=.001 if failure == 'weak' else .9,
                good=torch.zeros_like(a, dtype=torch.bool) if failure == 'invalid' else None,
                boundary=failure == 'boundary', tie=failure == 'tie')
        return result(a, 16 * a.shape[-1] / 128)
    d, _, good, trace = call(*images(), mode='accelerated', matcher=spy)
    assert good.all() and (d == 16).all()
    for level in ('half', 'full'):
        assert trace['pyramid']['levels'][level]['global_fallback_mask'].all()
    assert len(seen) == 5


def test_accelerated_missing_coarse_prior_recovers_with_global_search():
    def spy(a, b, sign, **kw):
        return result(a, 16 * a.shape[-1] / 128,
                      good=torch.zeros_like(a, dtype=torch.bool) if a.shape[-1] == 32 else None)
    d, _, good, trace = call(*images(), mode='accelerated', matcher=spy)
    assert good.all() and (d == 16).all()
    assert trace['pyramid']['levels']['half']['global_fallback_mask'].all()


def test_accelerated_real_match_has_smaller_actual_search_count_than_audit():
    a, b = images()
    audit = call(a, b)
    fast = call(a, b, mode='accelerated')
    roi = (..., slice(8, -8), slice(32, -8))
    assert fast[2][roi].float().mean() > .8
    assert ((fast[0][roi][fast[2][roi]] - 16).abs() < 1e-5).all()
    assert fast[3]['pyramid']['search_statistics']['total_cost_evaluations'].item() < audit[3]['pyramid']['search_statistics']['total_cost_evaluations'].item()
    assert fast[3]['resource_metadata']['cost_volume_allocated'] is False
    assert fast[3]['pyramid']['levels']['full']['fallback_rate'].shape == (1,)


@pytest.mark.parametrize('mode', ['audit', 'accelerated'])
def test_odd_intrinsics_and_heterogeneous_chunk_metadata_stay_consistent(mode):
    a, b = images(batch=2, h=25, w=83, disparity=8)
    masks = torch.ones_like(a, dtype=torch.bool); masks[1] = False
    full = call(a, b, mode=mode, a_valid=masks, b_valid=masks)
    pieces = [call(a[i:i+1], b[i:i+1], mode=mode, a_valid=masks[i:i+1], b_valid=masks[i:i+1]) for i in range(2)]
    def compare(value, chunks):
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, torch.cat(chunks), atol=0, rtol=0, equal_nan=True)
        elif isinstance(value, dict):
            for key in value:
                compare(value[key], [chunk[key] for chunk in chunks])
        else:
            assert all(chunk == value for chunk in chunks)
    for value, chunks in zip(full, zip(*pieces)):
        compare(value, list(chunks))
    for level in full[3]['pyramid']['levels'].values():
        sx, sy = level['scale_wh']
        expected = torch.diag(torch.tensor([sx, sy, 1.], dtype=torch.float64)) @ calibration(83, 25).K_left
        torch.testing.assert_close(torch.tensor(level['K_left'], dtype=torch.float64), expected)


def test_pyramid_mode_serializes_only_for_coarse_to_fine_and_round_trips():
    assert len(StereoDepth().to_config()) == 12
    default = StereoDepth(method='coarse_to_fine')
    assert default.to_config().get('pyramid_mode') == 'audit'
    if 'pyramid_mode' in inspect.signature(StereoDepth).parameters:
        fast = StereoDepth(method='coarse_to_fine', pyramid_mode='accelerated')
        assert StereoDepth.from_config(fast.to_config()).to_config() == fast.to_config()
        with pytest.raises(ValueError, match='pyramid'):
            StereoDepth(pyramid_mode='accelerated')
        with pytest.raises(ValueError, match='pyramid'):
            StereoDepth(method='coarse_to_fine', pyramid_mode='guess')


def test_wrapper_dispatches_the_serialized_accelerated_mode():
    options = {'pyramid_mode': 'accelerated'} if 'pyramid_mode' in inspect.signature(StereoDepth).parameters else {}
    depth = StereoDepth(method='coarse_to_fine', max_disparity=24, min_depth_m=.1,
                        max_depth_m=2., uniqueness_threshold=.01, **options)
    a, b = images()
    rgb_a, rgb_b = a[:, None].expand(-1, 3, -1, -1), b[:, None].expand(-1, 3, -1, -1)
    result = depth.compute(rgb_a, rgb_b, calibration())
    assert result['diagnostics']['pyramid'].get('mode') == 'accelerated'
