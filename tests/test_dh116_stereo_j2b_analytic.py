"""J2-B analytic software contracts, independent of native camera accuracy.

The qualification guards and private runner use ``measure_fixed_reference``
with the unchanged J2-A continuous fixture. An unqualified candidate is not
made to pass by reducing the GT denominator or relaxing numerical thresholds.
"""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth


METHODS = ('census', 'zncc', 'gradient', 'coarse_to_fine')
DISPARITIES = (6.1, 6.25, 6.5, 6.75)
REFERENCE_DENOMINATOR = 2800
MIN_COVERAGE = .70
MAX_P95_DISPARITY_ERROR_PX = .2


def fixture_module():
    """Load by file: the repository's tests directory is not a Python package."""
    name = '_j2b_continuous_fixture'
    if name not in sys.modules:
        path = Path(__file__).with_name('stereo_j2a_analytic_fixtures.py')
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def candidate_matcher(method, **changes):
    config = dict(method=method, min_disparity=2, max_disparity=12,
        block_size=9, min_depth_m=.2, max_depth_m=2., lr_tolerance_px=.5,
        texture_threshold=.001, uniqueness_threshold=.01, subpixel=True)
    config.update(changes)
    return StereoDepth(**config)


def measure_fixed_reference(data, result):
    """Keep missed pixels in coverage/correct-fraction denominators.

    Error statistics describe accepted reference pixels only. Empty acceptance
    has undefined errors (None) and fails qualification rather than looking
    like a perfect match. Depth errors are axial Z errors, never 3-D errors.
    """
    support = data.reference_valid
    denominator = int(support.sum())
    assert denominator > 0
    accepted = support & result['valid_mask']
    count = int(accepted.sum())
    disparity_error = (result['disparity_px'][accepted]
                       - data.disparity_reference_px[accepted]).abs()
    depth_error = (result['depth_m'][accepted]
                   - data.depth_reference_m[accepted]).abs()
    correct = int((disparity_error <= MAX_P95_DISPARITY_ERROR_PX).sum())
    p95 = float(disparity_error.quantile(.95)) if count else None
    coverage = count / denominator
    return dict(reference_pixels=denominator, accepted_reference_pixels=count,
        coverage=coverage, disparity_correct_fraction_fixed_support=correct / denominator,
        disparity_mae_px=float(disparity_error.mean()) if count else None,
        disparity_rmse_px=float(disparity_error.square().mean().sqrt()) if count else None,
        disparity_p95_abs_error_px=p95,
        depth_z_mae_m=float(depth_error.mean()) if count else None,
        depth_z_rmse_m=float(depth_error.square().mean().sqrt()) if count else None,
        depth_z_p95_abs_error_m=float(depth_error.quantile(.95)) if count else None,
        qualified=bool(coverage >= MIN_COVERAGE and p95 is not None
                       and p95 <= MAX_P95_DISPARITY_ERROR_PX))


@pytest.fixture(scope='module', autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('method', METHODS)
def test_candidate_explicit_optin_constructs_real_stereo_depth(method):
    matcher = candidate_matcher(method)
    assert matcher.to_config()['method'] == method
    assert StereoDepth.from_config(matcher.to_config()).to_config() == matcher.to_config()


@pytest.mark.parametrize('disparity', DISPARITIES)
def test_candidates_share_matcher_independent_2800_pixel_continuous_reference(disparity):
    data = fixture_module().analytic_stereo_fixture(disparity)
    assert int(data.reference_valid.sum()) == REFERENCE_DENOMINATOR
    y, x = torch.meshgrid(torch.arange(data.height, dtype=torch.float64),
        torch.arange(data.width, dtype=torch.float64), indexing='ij')
    expected = fixture_module().bandlimited_rgb(x + disparity, y)[None].float()
    torch.testing.assert_close(data.right_rgb, expected, rtol=0., atol=0.)
    expected_depth = float(data.calibration.K_left[0, 0]) * data.calibration.baseline_m / disparity
    torch.testing.assert_close(data.depth_reference_m[data.reference_valid],
        torch.full((REFERENCE_DENOMINATOR,), expected_depth), rtol=0., atol=0.)
    assert data.metadata['native_accuracy_qualified'] is False


def test_all_invalid_result_cannot_pass_by_shrinking_the_reference_denominator():
    data = fixture_module().analytic_stereo_fixture(6.25)
    missing = torch.full_like(data.depth_reference_m, float('nan'))
    summary = measure_fixed_reference(data, dict(valid_mask=torch.zeros_like(data.reference_valid),
        disparity_px=missing, depth_m=missing))
    assert summary['reference_pixels'] == REFERENCE_DENOMINATOR
    assert summary['accepted_reference_pixels'] == 0
    assert summary['coverage'] == summary['disparity_correct_fraction_fixed_support'] == 0.
    assert summary['disparity_p95_abs_error_px'] is None
    assert summary['depth_z_p95_abs_error_m'] is None
    assert summary['qualified'] is False


@pytest.mark.parametrize('method', METHODS)
def test_candidate_direct_continuous_result_uses_calibration_and_nan_invalid_contract(method):
    data = fixture_module().analytic_stereo_fixture(6.25)
    result = candidate_matcher(method).compute(data.left_rgb, data.right_rgb, data.calibration,
        left_valid=data.left_valid, right_valid=data.right_valid)
    valid = result['valid_mask']
    assert (valid & data.reference_valid).any(), 'observable fixed support must not pass as all-invalid'
    assert result['depth_m'].shape == result['disparity_px'].shape == data.reference_valid.shape
    assert torch.isnan(result['depth_m'][~valid]).all()
    assert torch.isnan(result['disparity_px'][~valid]).all()
    assert not result['confidence'][~valid].any()
    expected_depth = (float(data.calibration.K_left[0, 0]) * data.calibration.baseline_m
                      / result['disparity_px'][valid])
    torch.testing.assert_close(result['depth_m'][valid], expected_depth)
    summary = measure_fixed_reference(data, result)
    assert summary['reference_pixels'] == REFERENCE_DENOMINATOR


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('texture', ['constant', 'repeated'])
def test_candidate_rejects_unobservable_and_exact_repeated_continuous_texture(method, texture):
    data = fixture_module().analytic_stereo_fixture(6.25, texture=texture)
    result = candidate_matcher(method).compute(data.left_rgb, data.right_rgb, data.calibration,
        left_valid=data.left_valid, right_valid=data.right_valid)
    assert not result['valid_mask'][data.reference_valid].any()
    assert torch.isnan(result['depth_m'][data.reference_valid]).all()


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('missing_view', ['left', 'right'])
def test_candidate_missing_full_view_cannot_manufacture_a_match(method, missing_view):
    data = fixture_module().analytic_stereo_fixture(6.25)
    masks = dict(left_valid=data.left_valid, right_valid=data.right_valid)
    masks[missing_view + '_valid'] = torch.zeros_like(data.reference_valid)
    result = candidate_matcher(method).compute(data.left_rgb, data.right_rgb,
        data.calibration, **masks)
    assert not result['valid_mask'].any()
    assert torch.isnan(result['depth_m']).all()
    assert torch.isnan(result['disparity_px']).all()
    assert not result['confidence'].any()


@pytest.mark.parametrize('method', METHODS)
def test_candidate_cannot_match_a_patch_missing_for_every_disparity(method):
    data = fixture_module().analytic_stereo_fixture(6.25, right_missing_columns=(38, 53))
    result = candidate_matcher(method).compute(data.left_rgb, data.right_rgb, data.calibration,
        left_valid=data.left_valid, right_valid=data.right_valid)
    # All configured candidate centres fall inside the missing right stripe.
    # This mask depends only on the declared search and input observation mask.
    certainly_missing = torch.zeros_like(data.reference_valid)
    certainly_missing[..., 38 + 12:53 + 2] = True
    assert certainly_missing.any()
    assert not result['valid_mask'][certainly_missing].any()
    assert not result['valid_mask'][data.boundary_left].any()


def assert_observable_numerical_qualification(data, result, method):
    summary = measure_fixed_reference(data, result)
    context = f"{method}, d={data.metadata['disparity_px']}: {summary}"
    assert summary['reference_pixels'] == REFERENCE_DENOMINATOR
    assert summary['coverage'] >= MIN_COVERAGE, context
    assert summary['disparity_p95_abs_error_px'] is not None, context
    assert summary['disparity_p95_abs_error_px'] <= MAX_P95_DISPARITY_ERROR_PX, context
    assert summary['disparity_correct_fraction_fixed_support'] >= .65, context
    assert summary['depth_z_p95_abs_error_m'] <= .04, context


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('disparity', DISPARITIES)
def test_candidate_fractional_continuous_numerical_qualification(method, disparity):
    data = fixture_module().analytic_stereo_fixture(disparity)
    result = candidate_matcher(method).compute(data.left_rgb, data.right_rgb, data.calibration,
        left_valid=data.left_valid, right_valid=data.right_valid)
    assert_observable_numerical_qualification(data, result, method)


@pytest.mark.parametrize('method', ['census', 'zncc'])
@pytest.mark.parametrize('disparity', DISPARITIES)
def test_census_zncc_affine_brightness_numerical_qualification(method, disparity):
    data = fixture_module().analytic_stereo_fixture(disparity)
    right = .7 * data.right_rgb + .12
    assert bool(((right >= 0) & (right <= 1)).all())
    result = candidate_matcher(method).compute(data.left_rgb, right, data.calibration,
        left_valid=data.left_valid, right_valid=data.right_valid)
    assert_observable_numerical_qualification(data, result, method)
