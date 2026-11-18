"""J2-A CPU contract tests; no native, training, or accuracy qualification.

The old matcher is a verbatim, SHA-pinned test fixture, not a copy of the
working-tree matcher. External SGBM output is injected only where a backend
boundary is needed to test reason semantics without requiring OpenCV.
"""
import hashlib

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth
from stereo_j2a_legacy_oracle import (
    BASELINE_COMMIT, BASELINE_SOURCE, BASELINE_SOURCE_SHA256, load_legacy_stereo_depth,
)


LEGACY_KEYS = ('depth_m', 'disparity_px', 'valid', 'occlusion', 'confidence')
REASON_KEYS = (
    'low_texture_left', 'low_texture_right', 'search_boundary',
    'left_right_inconsistent', 'uniqueness', 'occlusion_composite',
    'true_occlusion', 'depth_range', 'subpixel_rejected', 'patch_border',
)


@pytest.fixture(scope='module', autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def stereo_fixture(*, disparity=4, batch=1, width=96, height=24):
    """An exact integer translation with independent seeded RGB per image."""
    left = torch.rand(batch, 3, height, width, generator=torch.Generator().manual_seed(11654))
    right = torch.zeros_like(left)
    right[..., :-disparity] = left[..., disparity:]
    k = [[40., 0., width/2], [0., 40., height/2], [0., 0., 1.]]
    a = torch.eye(4, dtype=torch.float64)
    b = a.clone(); b[0, 3] = .08
    return left, right, StereoCalibration(k, k, a, b, width, height, .08)


def assert_legacy_tensors_equal(actual, expected):
    for key in LEGACY_KEYS:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True,
                                   msg=lambda message: f'{key}: {message}')


def assert_diagnostics_schema(result, method):
    shape = result['depth_m'].shape
    diagnostics = result['diagnostics']
    assert diagnostics['version'] == 'dh116-stereo-matcher-diagnostics-v1'
    assert diagnostics['method'] == method
    assert diagnostics['confidence_is_probability'] is False
    assert diagnostics['true_occlusion_available'] is False
    if method == 'sgbm':
        assert diagnostics['confidence_kind'] == 'compatibility_validity_indicator'
        assert diagnostics['quality_confidence_available'] is False
        assert diagnostics['refinement_scope'] == 'OpenCV_internal_not_exposed'
        assert diagnostics['wrapper_refinement_applied'] is False
    else:
        assert diagnostics['confidence_kind'] == 'relative_cost_margin'
        assert diagnostics['quality_confidence_available'] is True
    reasons = diagnostics['reasons']
    assert set(REASON_KEYS) <= reasons.keys()
    assert reasons['true_occlusion'] is None
    for key in REASON_KEYS:
        # OpenCV does not expose either internal uniqueness or fit rejection.
        # Reporting False would claim those tests ran and accepted the pixel.
        if key == 'true_occlusion' or (method == 'sgbm' and key in ('uniqueness', 'subpixel_rejected')):
            assert reasons[key] is None, f'{key} cannot be inferred from this backend'
        else:
            assert isinstance(reasons[key], torch.Tensor), key
            assert reasons[key].shape == shape, key
            assert reasons[key].dtype == torch.bool, key
    torch.testing.assert_close(reasons['occlusion_composite'], result['occlusion'], rtol=0, atol=0)
    return reasons


def test_legacy_oracle_is_fixed_commit_source_without_current_matcher_import():
    assert BASELINE_COMMIT == '692e09d2473502574d0870dfaffff38b6c0a544c'
    assert BASELINE_SOURCE_SHA256 == '36b35189d7576dde7790f22b468b1be4393e7acfbea8060dc0380073aed28008'
    assert hashlib.sha256(BASELINE_SOURCE.encode()).hexdigest() == BASELINE_SOURCE_SHA256
    assert 'reliable_sad' not in BASELINE_SOURCE
    assert load_legacy_stereo_depth() is not StereoDepth


@pytest.mark.parametrize('scenario', ['default_sad5', 'occluded_and_remapped', 'textureless', 'subpixel', 'batch3'])
def test_default_and_existing_sad_outputs_match_fixed_legacy_bits(scenario):
    left, right, calibration = stereo_fixture(batch=3 if scenario == 'batch3' else 1)
    kwargs = {'subpixel': True} if scenario == 'subpixel' else {}
    masks = {}
    if scenario == 'occluded_and_remapped':
        right[..., 35:41] = .25
        left_mask = torch.ones_like(left[:, 0], dtype=torch.bool)
        right_mask = left_mask.clone()
        left_mask[..., 55:59] = False
        right_mask[..., 61:64] = False
        masks = dict(left_valid=left_mask, right_valid=right_mask)
    elif scenario == 'textureless':
        left.fill_(.5); right.fill_(.5)
    expected = load_legacy_stereo_depth()(**kwargs).compute(left, right, calibration, **masks)
    actual = StereoDepth(**kwargs).compute(left, right, calibration, **masks)
    assert_legacy_tensors_equal(actual, expected)
    if scenario == 'textureless':
        assert not actual['valid'].any()
    else:
        assert int(actual['valid'].sum()) > 300


@pytest.mark.parametrize('method', ['blockmatch', 'sgbm'])
def test_existing_config_serialization_is_unchanged(method):
    options = dict(method=method, max_disparity=16, block_size=5, subpixel=True,
                   image_processing_batch_size=2)
    expected = load_legacy_stereo_depth()(**options).to_config()
    actual = StereoDepth(**options).to_config()
    assert actual == expected
    assert 'reliability' not in actual
    assert StereoDepth.from_config({'depth_estimator': actual}).to_config() == expected


@pytest.mark.parametrize('method', ['blockmatch', 'reliable_sad', 'sgbm'])
def test_matcher_result_shapes_units_aliases_and_diagnostics(method, monkeypatch):
    if method == 'sgbm':
        monkeypatch.setattr(StereoDepth, '_sgbm', lambda self, a, b: (torch.full_like(a, 4.), torch.full_like(b, 4.)))
    left, right, calibration = stereo_fixture(batch=2)
    # Fix the reference-visible ROI before examining predictions. The shifted
    # left border has no right correspondence; the frozen legacy matcher can
    # accept random matches there. Contract additions must preserve those old
    # full-image bits, not mislabel them as known-d=4 observations.
    reference_visible = torch.zeros_like(left[:, 0], dtype=torch.bool)
    radius, maximum_disparity = 2, 8
    reference_visible[:, radius:-radius, maximum_disparity+radius:-(maximum_disparity+radius)] = True
    result = StereoDepth(method=method, max_disparity=8).compute(left, right, calibration)
    assert result['valid_mask'] is result['valid']
    assert result['occlusion_mask'] is result['occlusion']
    for key in LEGACY_KEYS:
        assert result[key].shape == (2, 24, 96)
        if key in ('valid', 'occlusion'):
            assert result[key].dtype == torch.bool
        else:
            assert result[key].is_floating_point()
    valid = result['valid_mask']
    assert int(valid.sum()) > 600
    assert torch.isnan(result['depth_m'][~valid]).all()
    assert torch.isnan(result['disparity_px'][~valid]).all()
    assert (result['confidence'][~valid] == 0).all()
    assert ((result['confidence'] >= 0) & (result['confidence'] <= 1)).all()
    torch.testing.assert_close(result['depth_m'][valid], 3.2/result['disparity_px'][valid], rtol=0, atol=0)
    assert valid[reference_visible].float().mean() > .9
    observed = reference_visible & valid
    torch.testing.assert_close(result['disparity_px'][observed], torch.full_like(result['disparity_px'][observed], 4.), rtol=0, atol=0)
    torch.testing.assert_close(result['depth_m'][observed], torch.full_like(result['depth_m'][observed], .8), rtol=1e-6, atol=0)
    assert_diagnostics_schema(result, method)


def test_sgbm_keeps_legacy_confidence_but_marks_quality_and_uniqueness_unavailable(monkeypatch):
    old_class = load_legacy_stereo_depth()
    def backend(self, a, b):
        dl = torch.full_like(a, 4.); dr = torch.full_like(b, 4.)
        dl[..., 1] = 0.; dl[..., -1] = 20.
        return dl, dr
    monkeypatch.setattr(StereoDepth, '_sgbm', backend)
    monkeypatch.setattr(old_class, '_sgbm', backend)
    left, right, calibration = stereo_fixture()
    options = dict(method='sgbm', max_disparity=8)
    actual = StereoDepth(**options).compute(left, right, calibration)
    expected = old_class(**options).compute(left, right, calibration)
    assert_legacy_tensors_equal(actual, expected)
    assert int(actual['valid'].sum()) > 300
    assert torch.equal(actual['confidence'][actual['valid']], torch.ones_like(actual['confidence'][actual['valid']]))
    reasons = assert_diagnostics_schema(actual, 'sgbm')
    assert reasons['uniqueness'] is None
    assert reasons['true_occlusion'] is None


@pytest.mark.parametrize('invalid_disparity', [float('nan'), float('inf'), 0., -.5])
def test_invalid_disparity_never_clamped_into_positive_depth(invalid_disparity, monkeypatch):
    # An external/raw matcher can report support=True for an invalid disparity.
    # A wide valid metric range exposes the old zero/negative clamp bug rather
    # than letting an ordinary far-depth cutoff accidentally hide it.
    def raw_match(self, a, b, sign, *args, **kwargs):
        return torch.full_like(a, invalid_disparity), torch.ones_like(a), torch.ones_like(a, dtype=torch.bool)
    monkeypatch.setattr(StereoDepth, '_match', raw_match)
    left, right, calibration = stereo_fixture()
    result = StereoDepth(max_disparity=8, min_depth_m=.01, max_depth_m=1e10).compute(left, right, calibration)
    assert not result['valid'].any()
    assert torch.isnan(result['depth_m']).all()
    assert torch.isnan(result['disparity_px']).all()
    assert (result['confidence'] == 0).all()


@pytest.mark.parametrize('side', ['left', 'right'])
def test_texture_reasons_distinguish_each_view_from_lr_and_true_occlusion(side, monkeypatch):
    monkeypatch.setattr(StereoDepth, '_sgbm', lambda self, a, b: (torch.full_like(a, 4.), torch.full_like(b, 4.)))
    left, right, calibration = stereo_fixture()
    (left if side == 'left' else right).fill_(.5)
    result = StereoDepth(method='sgbm', max_disparity=8).compute(left, right, calibration)
    reasons = assert_diagnostics_schema(result, 'sgbm')
    center = (0, 12, 48)
    assert reasons[f'low_texture_{side}'][center]
    assert not reasons[f'low_texture_{"right" if side == "left" else "left"}'][center]
    assert not reasons['left_right_inconsistent'][center]
    assert reasons['true_occlusion'] is None
    assert not result['valid'][center]


def test_left_right_and_depth_range_rejections_have_distinct_predicates(monkeypatch):
    left, right, calibration = stereo_fixture()
    center = (0, 12, 48)
    monkeypatch.setattr(StereoDepth, '_sgbm', lambda self, a, b: (torch.full_like(a, 4.), torch.full_like(b, 5.)))
    mismatch = StereoDepth(method='sgbm', max_disparity=8, lr_tolerance_px=.1).compute(left, right, calibration)
    a = assert_diagnostics_schema(mismatch, 'sgbm')
    assert a['left_right_inconsistent'][center]
    assert a['occlusion_composite'][center]
    assert not a['depth_range'][center]
    monkeypatch.setattr(StereoDepth, '_sgbm', lambda self, a, b: (torch.full_like(a, 4.), torch.full_like(b, 4.)))
    too_far = StereoDepth(method='sgbm', max_disparity=8, max_depth_m=.5).compute(left, right, calibration)
    b = assert_diagnostics_schema(too_far, 'sgbm')
    assert b['depth_range'][center]
    assert not b['left_right_inconsistent'][center]
    assert not b['occlusion_composite'][center]
    assert not too_far['valid'][center]


def test_uniqueness_rejection_is_available_for_sad_and_not_merged_with_texture():
    left, right, calibration = stereo_fixture()
    left.fill_(.5); right.fill_(.5)
    result = StereoDepth(max_disparity=8, texture_threshold=0, uniqueness_threshold=.05).compute(left, right, calibration)
    reasons = assert_diagnostics_schema(result, 'blockmatch')
    center = (0, 12, 48)
    assert reasons['uniqueness'][center]
    assert not reasons['low_texture_left'][center]
    assert not reasons['low_texture_right'][center]
    assert not reasons['left_right_inconsistent'][center]
    assert not result['valid'][center]


def test_search_endpoint_and_subpixel_rejection_are_distinct_from_patch_border():
    left, right, calibration = stereo_fixture(disparity=1)
    options = dict(max_disparity=8, min_depth_m=.1, max_depth_m=4.)
    integer = StereoDepth(**options, subpixel=False).compute(left, right, calibration)
    fitted = StereoDepth(**options, subpixel=True).compute(left, right, calibration)
    a = assert_diagnostics_schema(integer, 'blockmatch')
    b = assert_diagnostics_schema(fitted, 'blockmatch')
    center = (0, 12, 48)
    assert a['search_boundary'][center]
    assert not a['subpixel_rejected'][center]
    assert not a['patch_border'][center]
    assert integer['valid'][center], 'diagnostic addition must not add a legacy integer gate'
    assert b['search_boundary'][center]
    assert b['subpixel_rejected'][center]
    assert not b['patch_border'][center]
    assert not fitted['valid'][center]
    assert a['patch_border'][:, :2].all() and a['patch_border'][:, -2:].all()
    assert a['patch_border'][:, :, :2].all() and a['patch_border'][:, :, -2:].all()


def test_out_of_view_right_sample_is_not_reported_as_low_right_texture(monkeypatch):
    monkeypatch.setattr(StereoDepth, '_sgbm', lambda self, a, b: (torch.full_like(a, 4.), torch.full_like(b, 4.)))
    left, right, calibration = stereo_fixture()
    result = StereoDepth(method='sgbm', max_disparity=8).compute(left, right, calibration)
    reasons = assert_diagnostics_schema(result, 'sgbm')
    outside = (0, 12, 3)  # An otherwise supported left pixel maps to x=-1.
    assert not reasons['low_texture_right'][outside]
    assert reasons['occlusion_composite'][outside]
    assert not result['valid'][outside]


def test_diagnostics_survive_image_chunking_with_na_reasons(monkeypatch):
    monkeypatch.setattr(StereoDepth, '_sgbm', lambda self, a, b: (torch.full_like(a, 4.), torch.full_like(b, 4.)))
    left, right, calibration = stereo_fixture(batch=3)
    options = dict(method='sgbm', max_disparity=8)
    chunked = StereoDepth(**options, image_processing_batch_size=1).compute(left, right, calibration)
    full = StereoDepth(**options, image_processing_batch_size=3).compute(left, right, calibration)
    assert_legacy_tensors_equal(chunked, full)
    a = assert_diagnostics_schema(chunked, 'sgbm')
    b = assert_diagnostics_schema(full, 'sgbm')
    for key in REASON_KEYS:
        if a[key] is None:
            assert b[key] is None
        else:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


@pytest.mark.parametrize('keyword', ['reference', 'reference_depth_m', 'wire_nodes', 'privileged_state'])
def test_runtime_matcher_rejects_privileged_reference_arguments(keyword):
    left, right, calibration = stereo_fixture()
    with pytest.raises(TypeError, match=keyword):
        StereoDepth(max_disparity=8).compute(left, right, calibration, **{keyword: torch.ones(1)})
