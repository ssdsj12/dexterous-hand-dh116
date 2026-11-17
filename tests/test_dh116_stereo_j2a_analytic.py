"""Continuous RGB numerical checks; these are not native accuracy evidence."""
import importlib
import importlib.util
import math

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth


def fixture_module():
    name = 'stereo_j2a_analytic_fixtures'
    assert importlib.util.find_spec(name) is not None, 'continuous analytic stereo fixture is missing'
    return importlib.import_module(name)


@pytest.mark.parametrize('disparity', [6.1, 6.25, 6.5, 6.75])
def test_analytic_fixture_samples_continuous_right_image_and_metric_depth(disparity):
    m = fixture_module()
    data = m.analytic_stereo_fixture(disparity)
    y, x = torch.meshgrid(torch.arange(data.height, dtype=torch.float64),
                          torch.arange(data.width, dtype=torch.float64), indexing='ij')
    expected = m.bandlimited_rgb(x + disparity, y)
    torch.testing.assert_close(data.right_rgb, expected[None].float(), rtol=0., atol=0.)
    torch.testing.assert_close(data.left_rgb, m.bandlimited_rgb(x, y)[None].float(), rtol=0., atol=0.)
    assert data.left_rgb.shape == data.right_rgb.shape == (1, 3, data.height, data.width)
    assert bool(((data.left_rgb >= 0) & (data.left_rgb <= 1)).all())
    assert data.left_rgb.dtype == data.right_rgb.dtype == torch.float32
    assert data.reference_valid.dtype == torch.bool
    assert int(data.reference_valid.sum()) > 1000
    assert not (data.reference_valid & data.boundary_left).any()
    expected_depth = float(data.calibration.K_left[0, 0]) * data.calibration.baseline_m / disparity
    torch.testing.assert_close(data.depth_reference_m[data.reference_valid],
        torch.full_like(data.depth_reference_m[data.reference_valid], expected_depth))
    torch.testing.assert_close(data.disparity_reference_px[data.reference_valid],
        torch.full_like(data.disparity_reference_px[data.reference_valid], disparity))
    assert data.metadata['rgb_generation'] == 'direct_continuous_bandlimited_sampling'
    assert data.metadata['native_accuracy_qualified'] is False


def test_analytic_fixture_is_not_linear_interpolation_of_discrete_left_image():
    data = fixture_module().analytic_stereo_fixture(6.5)
    linear = .5 * (data.left_rgb[..., 6:-1] + data.left_rgb[..., 7:])
    assert float((data.right_rgb[..., :-7] - linear).abs().max()) > .001


def test_analytic_fixture_integer_translation_and_scene_labels_are_exact():
    m = fixture_module()
    data = m.analytic_stereo_fixture(6.)
    torch.testing.assert_close(data.right_rgb[..., :-6], data.left_rgb[..., 6:], rtol=0., atol=0.)
    assert not data.occlusion_left[data.reference_valid].any()
    assert bool(data.occlusion_left[..., :6].all())
    assert not data.constant_left.any()
    assert not data.repeated_left.any()
    for kind, name in [('constant', 'constant_left'), ('repeated', 'repeated_left')]:
        degenerate = m.analytic_stereo_fixture(6.25, texture=kind)
        assert bool(getattr(degenerate, name)[degenerate.reference_valid].all())
        assert degenerate.metadata['texture_kind'] == kind


def test_analytic_fixture_labels_missing_right_support_without_fabricating_gt():
    m = fixture_module()
    data = m.analytic_stereo_fixture(6.25, right_missing_columns=(38, 53))
    assert not data.right_valid[..., 38:53].any()
    assert data.occlusion_left.any()
    assert not (data.reference_valid & data.occlusion_left).any()
    assert torch.isnan(data.depth_reference_m[~data.reference_valid]).all()
    assert torch.isnan(data.disparity_reference_px[~data.reference_valid]).all()


def test_analytic_fixture_distinguishes_observed_occluder_from_missing_observation():
    data = fixture_module().analytic_stereo_fixture(6.25, right_occluder_columns=(34, 59))
    assert data.right_valid.all()
    assert data.occlusion_left.any()
    assert not (data.reference_valid & data.occlusion_left).any()
    assert bool((data.right_rgb[..., 34:59] == .95).all())
    assert data.metadata['observed_right_occluder_stripe'] == (34, 59)


def reliable_matcher(**changes):
    config = dict(method='reliable_sad', min_disparity=2, max_disparity=12,
        block_size=9, min_depth_m=.2, max_depth_m=2., lr_tolerance_px=.5,
        texture_threshold=.001, uniqueness_threshold=.01, subpixel=True)
    config.update(changes)
    return StereoDepth(**config)


@pytest.mark.parametrize('disparity', [6.1, 6.25, 6.5, 6.75])
def test_reliable_sad_fractional_disparity_has_fixed_support_coverage_and_point_accuracy(disparity):
    data = fixture_module().analytic_stereo_fixture(disparity)
    result = reliable_matcher().compute(data.left_rgb, data.right_rgb, data.calibration,
        left_valid=data.left_valid, right_valid=data.right_valid)
    support = data.reference_valid
    accepted = result['valid_mask'] & support
    coverage = float(accepted.sum()) / int(support.sum())
    assert coverage >= .70, f'fixed reference coverage {coverage:.4f} at d={disparity}'
    error = (result['disparity_px'][accepted] - disparity).abs()
    assert float(error.quantile(.95)) <= .2, f'P95 disparity error {float(error.quantile(.95)):.4f} px'
    correct = accepted & ((result['disparity_px'] - disparity).abs() <= .2)
    assert float(correct.sum()) / int(support.sum()) >= .65
    expected_z = float(data.calibration.K_left[0, 0]) * data.calibration.baseline_m / disparity
    assert float((result['depth_m'][accepted] - expected_z).abs().quantile(.95)) <= .04
    assert torch.isnan(result['depth_m'][~result['valid_mask']]).all()
    assert torch.isnan(result['disparity_px'][~result['valid_mask']]).all()
    assert not result['confidence'][~result['valid_mask']].any()


@pytest.mark.parametrize('texture', ['constant', 'repeated'])
def test_reliable_sad_rejects_unobservable_and_ambiguous_continuous_texture(texture):
    data = fixture_module().analytic_stereo_fixture(6.25, texture=texture)
    result = reliable_matcher().compute(data.left_rgb, data.right_rgb, data.calibration,
        left_valid=data.left_valid, right_valid=data.right_valid)
    assert not result['valid_mask'][data.reference_valid].any()
    assert torch.isnan(result['depth_m'][data.reference_valid]).all()


@pytest.mark.parametrize('period', [2.7, 5.3])
@pytest.mark.parametrize('phase', [0., .37])
def test_reliable_sad_rejects_fractional_remote_periodic_aliases(period, phase):
    data = fixture_module().analytic_stereo_fixture(6.25)
    _, x = torch.meshgrid(torch.arange(data.height, dtype=torch.float64),
                          torch.arange(data.width, dtype=torch.float64), indexing='ij')

    def radiance(coordinate):
        gray = .5 + .25 * torch.cos((2 * math.pi / period) * coordinate + phase)
        return torch.stack((.1 + .8 * gray, .05 + .9 * gray, gray))[None].float()

    # Both images directly sample the same continuous, sub-Nyquist radiance.
    # Every listed disparity gives an identical continuous correspondence, so
    # selecting one high-margin sampled integer does not resolve the ambiguity.
    aliases = [6.25 + multiple * period for multiple in range(-8, 9)
               if 2 <= 6.25 + multiple * period <= 12]
    assert len(aliases) >= 2 and 1 / period < .5
    left_rgb, right_rgb = radiance(x), radiance(x + 6.25)
    for alias in aliases:
        torch.testing.assert_close(radiance(x + alias), right_rgb, rtol=0., atol=1e-6)
    result = reliable_matcher().compute(left_rgb, right_rgb, data.calibration,
        left_valid=data.left_valid, right_valid=data.right_valid)
    assert int(data.reference_valid.sum()) == 2800
    accepted = result['valid_mask'] & data.reference_valid
    coverage = float(accepted.sum()) / int(data.reference_valid.sum())
    assert coverage <= .01, (
        f'periodic aliases {aliases}, phase={phase}: ambiguous valid coverage {coverage:.4f}')
    assert torch.isnan(result['depth_m'][~result['valid_mask']]).all()
    assert not result['confidence'][~result['valid_mask']].any()


def test_reliable_sad_missing_observation_masks_fail_closed_for_whole_matching_patch():
    data = fixture_module().analytic_stereo_fixture(6.25)
    missing = torch.zeros_like(data.right_valid)
    result = reliable_matcher().compute(data.left_rgb, data.right_rgb, data.calibration,
        left_valid=data.left_valid, right_valid=missing)
    assert not result['valid_mask'].any()
    assert torch.isnan(result['depth_m']).all()
    assert torch.isnan(result['disparity_px']).all()
    assert not result['confidence'].any()


def test_reliable_sad_missing_right_patch_and_image_boundaries_fail_closed():
    data = fixture_module().analytic_stereo_fixture(6.25, right_missing_columns=(38, 53))
    result = reliable_matcher().compute(data.left_rgb, data.right_rgb, data.calibration,
        left_valid=data.left_valid, right_valid=data.right_valid)
    # Every disparity in the configured search reaches the missing stripe here;
    # accepting a different integer winner cannot manufacture observed support.
    certainly_missing = torch.zeros_like(data.reference_valid)
    certainly_missing[..., 12 + 38:2 + 53] = True
    assert certainly_missing.any()
    assert not result['valid_mask'][certainly_missing].any()
    assert not result['valid_mask'][data.boundary_left].any()
    coverage = float((result['valid_mask'] & data.reference_valid).sum()) / int(data.reference_valid.sum())
    assert coverage >= .70, f'missing-stripe fixed reference coverage {coverage:.4f}'


def test_reliable_sad_rejects_observed_textureless_occluding_patch():
    data = fixture_module().analytic_stereo_fixture(6.25, right_occluder_columns=(34, 59))
    result = reliable_matcher().compute(data.left_rgb, data.right_rgb, data.calibration,
        left_valid=data.left_valid, right_valid=data.right_valid)
    # For these left pixels every integer candidate and its full block remains
    # inside the independently observed occluder, so no plane match is possible.
    occluded = torch.zeros_like(data.reference_valid)
    occluded[..., 34 + 12 + 4:59 + 2 - 4] = True
    assert occluded.any()
    assert not result['valid_mask'][occluded].any()
    assert bool(result['occlusion_mask'][occluded].all())
    coverage = float((result['valid_mask'] & data.reference_valid).sum()) / int(data.reference_valid.sum())
    assert coverage >= .70, f'occluder fixed reference coverage {coverage:.4f}'
