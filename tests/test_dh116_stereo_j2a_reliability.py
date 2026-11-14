"""CPU analytic regressions; saved J1 costs are not native qualification."""
import importlib

import pytest
import torch


def module():
    return importlib.import_module(
        'go2_pvcnn.control.dh116.tabletop_stereo_matcher_candidates')


def refine(costs, integer=None, **kwargs):
    costs = torch.as_tensor(costs, dtype=torch.float64)
    if costs.ndim == 1:
        costs = costs[None]
    if integer is None:
        integer = torch.full(costs.shape[:-1], 43., dtype=costs.dtype)
    return module().refine_winner_neighborhood(integer, *costs.unbind(-1), **kwargs)


def parabola(offset, *, curvature=.004, base=.02):
    x = torch.arange(-2, 3, dtype=torch.float64)
    offset = torch.as_tensor(offset, dtype=torch.float64)
    return base + .5 * curvature * (x - offset[..., None]).square()


def test_j1_pixel_a_positive_curvature_does_not_prove_subpixel_reliable():
    # J1 enriched record: env=0, frame_id=21, x=347, y=268. All five
    # costs are present in top5_integer_costs; GT is used only below.
    costs = [.0023238924331963062, .0007746308110654354,
             .0003873154055327177, .0015492616221308708,
             .0030985232442617416]
    out = refine(costs)
    assert out['curvature'].item() > 0
    assert out['legacy_offset'].item() == pytest.approx(-.25)
    assert out['fit_residual'].item() == pytest.approx(.75)
    assert out['reasons']['poor_quadratic_fit'].item()
    assert not out['refinement_valid'].item()
    assert out['valid_mask'].item()
    assert out['disparity_px'].item() == 43.
    gt_disparity = 43.149105072021484
    assert abs(out['disparity_px'].item() - gt_disparity) < abs(42.75 - gt_disparity)


@pytest.mark.parametrize('offset', [-.4, -.25, -.1, .0, .1491, .25, .4])
def test_analytic_fractional_parabola_retains_real_subpixel_precision(offset):
    out = refine(parabola(offset))
    assert out['valid_mask'].all()
    assert out['refinement_valid'].all()
    assert out['disparity_px'].item() == pytest.approx(43. + offset, abs=1e-10)
    assert out['fit_residual'].item() < 1e-10
    assert not any(value.any() for value in out['reasons'].values())


def test_quantized_step_curve_falls_back_without_special_casing_offset():
    out = refine(torch.tensor([6., 2., 1., 4., 8.]) / 2582.)
    assert out['reasons']['poor_quadratic_fit'].all()
    assert not out['refinement_valid'].any()
    assert out['disparity_px'].item() == 43.
    # The same -.25 vertex is accepted when the surrounding model is quadratic.
    assert refine(parabola(-.25))['refinement_valid'].all()


@pytest.mark.parametrize('offset', [-.4, -.25, -.1, .0, .1491, .25, .4])
def test_analytic_absolute_cost_retains_fractional_sad_precision(offset):
    x = torch.arange(-2, 3, dtype=torch.float64)
    costs = .003 + .007 * (x - offset).abs()
    out = refine(costs)
    assert out['refinement_valid'].all()
    assert out['disparity_px'].item() == pytest.approx(43. + offset, abs=1e-10)
    assert out['model_selected'].item() == 2
    assert out['absolute_fit_residual'].item() < 1e-10
    assert not any(value.any() for value in out['reasons'].values())


@pytest.mark.parametrize('offset', [-.5, .5])
@pytest.mark.parametrize('model', ['quadratic', 'absolute'])
def test_half_pixel_adjacent_tie_is_one_observable_basin(offset, model):
    x = torch.arange(-2, 3, dtype=torch.float64)
    costs = parabola(offset) if model == 'quadratic' else .003 + .007 * (x - offset).abs()
    out = refine(costs)
    assert out['refinement_valid'].all()
    assert out['valid_mask'].all()
    assert out['disparity_px'].item() == pytest.approx(43. + offset, abs=1e-10)
    assert not any(value.any() for value in out['reasons'].values())


def test_j1_pixel_a_cannot_pass_absolute_cost_model_with_negative_floor():
    out = refine(torch.tensor([6., 2., 1., 4., 8.]) * .0003873154055327177)
    assert out['absolute_floor'].item() < 0
    assert out['absolute_fit_residual'].item() == pytest.approx(.25)
    assert not out['absolute_fit_valid'].any()
    assert out['model_selected'].item() == 0


def test_asymmetric_outer_slopes_cannot_certify_absolute_model():
    costs = torch.tensor([2.25, 1.25, .25, .75, 2.75], dtype=torch.float64)
    out = refine(costs)
    assert out['absolute_slope_residual'].item() > .05
    assert not out['absolute_fit_valid'].any()
    assert not out['refinement_valid'].any()


@pytest.mark.parametrize(('costs', 'reason'), [
    ([1., 1., 1., 1., 1.], 'low_curvature'),
    ([3., 2., 1., 2., 2.5], 'nonconvex'),
    ([4., 1., 0., 1., 0.], 'multimodal'),
    ([4., .1, .2, 1., 4.], 'winner_not_minimum'),
])
def test_degenerate_curves_reject_with_specific_tensor_reason(costs, reason):
    out = refine(costs)
    assert out['reasons'][reason].all()
    assert not out['refinement_valid'].any()
    assert not out['valid_mask'].any()
    assert torch.isnan(out['disparity_px']).all()


def test_small_positive_curvature_is_rejected_even_for_exact_parabola():
    out = refine(parabola(.2, curvature=1e-9, base=0.))
    assert out['reasons']['low_curvature'].all()
    assert not out['refinement_valid'].any()


def test_legacy_offset_uses_historical_float32_order_and_curvature_floor():
    costs = torch.cat((parabola(torch.tensor([-.3, .25])),
                       parabola(torch.tensor([.2]), curvature=5e-9, base=0.))).float()
    cm1, c0, cp1 = costs[:, 1], costs[:, 2], costs[:, 3]
    curvature = cm1 + cp1 - 2 * c0
    refinable = torch.isfinite(curvature) & (curvature > 1e-8)
    numerator = torch.where(refinable, cm1 - cp1, 0.)
    denominator = torch.where(refinable, curvature, 1.)
    expected = (.5 * numerator / denominator).clamp(-.5, .5)
    out = module().refine_winner_neighborhood(torch.full((3,), 43.), *costs.unbind(-1))
    assert torch.equal(out['legacy_offset'], expected)
    assert torch.equal(out['legacy_refinement_valid'], refinable)


def test_far_competing_peak_is_rejected_without_retaining_cost_volume():
    costs = parabola(.2)
    out = refine(costs, second_nonlocal_cost=torch.tensor([float(costs[2]) + 1e-5]))
    assert out['reasons']['multimodal'].all()
    assert not out['refinement_valid'].any()
    assert refine(costs, second_nonlocal_cost=torch.tensor([float('inf')]))[
        'refinement_valid'].all()


def test_missing_five_point_support_is_search_boundary_and_keeps_integer():
    out = refine(parabola(.2), available=torch.tensor([False]))
    assert out['reasons']['search_boundary'].all()
    assert out['valid_mask'].all()
    assert out['disparity_px'].item() == 43.
    assert not out['refinement_valid'].any()


@pytest.mark.parametrize('bad', [float('inf'), float('nan')])
def test_nonfinite_neighbor_is_invalid_without_inventing_finite_disparity(bad):
    costs = parabola(.2)
    costs[0] = bad
    out = refine(costs)
    assert out['reasons']['nonfinite_cost'].all()
    assert not out['valid_mask'].any()
    assert torch.isnan(out['disparity_px']).all()
    assert not out['refinement_valid'].any()


def test_invalid_center_cannot_be_made_valid_by_integer_fallback():
    costs = parabola(.2)
    costs[2] = float('inf')
    out = refine(costs)
    assert not out['valid_mask'].any()
    assert torch.isnan(out['disparity_px']).all()


def test_image_ambiguity_is_sensor_evidence_and_explicit_rejection_reason():
    out = refine(parabola(.2), image_ambiguity=torch.tensor([True]))
    assert out['reasons']['image_boundary_ambiguity'].all()
    assert not out['refinement_valid'].any()
    assert out['disparity_px'].item() == 43.


def test_invalid_fallback_keeps_nan_and_false_mask():
    out = refine([6., 2., 1., 4., 8.], policy={'fallback': 'invalid'})
    assert not out['valid_mask'].any()
    assert torch.isnan(out['disparity_px']).all()


@pytest.mark.parametrize('integer', [float('nan'), float('inf'), 0., -1., 43.2])
def test_invalid_integer_is_never_clamped_or_rounded_into_valid_disparity(integer):
    out = refine(parabola(.2), integer=torch.tensor([integer]))
    assert out['reasons']['invalid_integer'].all()
    assert not out['valid_mask'].any()
    assert torch.isnan(out['disparity_px']).all()


def test_streaming_planes_preserve_batch_shape_dtype_device_and_inputs():
    integer = torch.full((2, 3, 4), 43., dtype=torch.float32)
    costs = parabola(torch.full((2, 3, 4), .2)).float()
    original = costs.clone()
    out = module().refine_winner_neighborhood(integer, *costs.unbind(-1))
    assert torch.equal(costs, original)
    for key in ['disparity_px', 'valid_mask', 'valid', 'refinement_valid',
                'legacy_offset', 'refined_offset', 'curvature', 'fit_residual']:
        assert out[key].shape == integer.shape
        assert out[key].device == integer.device
    assert out['disparity_px'].dtype == costs.dtype
    assert out['disparity_px'].allclose(integer + .2)
    assert torch.equal(out['valid'], out['valid_mask'])
    assert all(value.dtype == torch.bool for value in out['reasons'].values())


def test_selected_samples_wrapper_matches_five_plane_streaming_api():
    disparities = torch.arange(39, 48, dtype=torch.float64)
    offsets = torch.tensor([-.25, .1491, .35], dtype=torch.float64)
    costs = .02 + .002 * (disparities[None] - (43. + offsets[:, None])).square()
    integer = torch.full((3,), 43., dtype=torch.float64)
    out = module().reliable_subpixel_refinement(integer, costs, disparities=disparities)
    planes = module().refine_winner_neighborhood(integer, *costs[:, 2:7].unbind(-1))
    assert torch.equal(out['refinement_valid'], planes['refinement_valid'])
    assert out['disparity_px'].allclose(43. + offsets, atol=1e-10, rtol=0.)


def test_selected_support_holes_and_search_limits_reject_subpixel():
    disparities = torch.arange(41, 46, dtype=torch.float64)
    costs = parabola(torch.tensor([.2, .2, .2]))
    integer = torch.tensor([43., 41., 43.], dtype=torch.float64)
    support = torch.ones_like(costs, dtype=torch.bool)
    support[2, 4] = False
    out = module().reliable_subpixel_refinement(
        integer, costs, disparities=disparities, support=support)
    assert out['refinement_valid'].tolist() == [True, False, False]
    assert out['reasons']['search_boundary'].tolist() == [False, True, True]


def test_selected_costs_reject_dense_image_cost_volume():
    with pytest.raises(ValueError, match='selected.*P.*D'):
        module().reliable_subpixel_refinement(
            torch.full((1, 3, 4), 43.), torch.ones(1, 9, 3, 4),
            disparities=torch.arange(39, 48))


def test_selected_winner_missing_from_search_or_support_is_invalid():
    disparities = torch.arange(41, 46, dtype=torch.float64)
    costs = parabola(torch.tensor([.2, .2, .2]))
    integer = torch.tensor([40., 46., 43.], dtype=torch.float64)
    support = torch.ones_like(costs, dtype=torch.bool)
    support[2, 2] = False
    out = module().reliable_subpixel_refinement(
        integer, costs, disparities=disparities, support=support)
    assert not out['valid_mask'].any()
    assert torch.isnan(out['disparity_px']).all()


def test_resolve_policy_returns_independent_validated_serializable_copy():
    value = module()
    custom = {'fit_tolerance': .2}
    resolved = value.resolve_policy(custom)
    assert resolved['fit_tolerance'] == .2
    assert resolved['fallback'] == 'integer'
    assert custom == {'fit_tolerance': .2}
    resolved['fallback'] = 'invalid'
    assert value.resolve_policy()['fallback'] == 'integer'
    assert value.DEFAULT_RELIABILITY_POLICY['fallback'] == 'integer'


@pytest.mark.parametrize('policy', [
    {'fallback': 'clamp'}, {'min_curvature': -1.}, {'fit_tolerance': float('nan')},
    {'mode_margin': 1.}, {'absolute_fit_tolerance': -.1}, {'unknown_threshold': .2},
])
def test_invalid_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        refine(parabola(.2), policy=policy)


def photometric_fixture(disparity=3.25):
    # Direct analytic image evaluation, with no resampling of the left image.
    y = torch.arange(8, dtype=torch.float32)[None, :, None]
    x = torch.arange(32, dtype=torch.float32)[None, None, :]
    left = .1 + .009 * x + .004 * y
    right = .1 + .009 * (x + disparity) + .004 * y
    valid = torch.ones_like(left, dtype=torch.bool)
    integer = torch.full_like(left, round(disparity))
    return left, right, integer, valid


@pytest.mark.parametrize('offset', [-.5, -.375, -.25, -.125, 0., .125, .25, .375, .5])
@pytest.mark.parametrize('sign', [-1, 1])
def test_photometric_search_verifies_fractional_image_cost_at_real_samples(offset, sign):
    left, right, integer, support = photometric_fixture(3. + offset)
    if sign == 1:
        left, right = right, left
    # Keep the claimed integer center fixed to test both half-pixel choices.
    integer.fill_(3.)
    out = module().photometric_refine_winner(left, right, integer, sign, 3, support, support)
    assert out['valid_mask'][0, 4, 16]
    assert out['disparity_px'][0, 4, 16].item() == pytest.approx(3. + offset)
    assert out['best_cost'][0, 4, 16].item() < 1e-7
    assert out['curvature'][0, 4, 16].item() > 0
    assert out['model_selected'][0, 4, 16].item() == 3


def test_photometric_patch_uses_center_disparity_for_all_neighbours():
    left, right, integer, support = photometric_fixture()
    integer[:, 3:6, 14:17] = 7.
    integer[0, 4, 15] = 3.
    out = module().photometric_refine_winner(left, right, integer, -1, 3, support, support)
    # Applying each neighbouring pixel's own 7px warp before pooling gives the
    # wrong answer. The complete patch at this center must use 3.25px.
    assert out['valid_mask'][0, 4, 15]
    assert out['disparity_px'][0, 4, 15].item() == 3.25
    assert out['best_cost'][0, 4, 15].item() < 1e-7


def test_photometric_search_rejects_flat_curve_and_incomplete_image_support():
    left, right, integer, support = photometric_fixture()
    flat = module().photometric_refine_winner(left * 0 + .5, right * 0 + .5,
                                              integer, -1, 3, support, support)
    assert not flat['valid_mask'].any()
    assert flat['reasons']['low_curvature'][0, 4, 16]
    assert torch.isnan(flat['disparity_px']).all()
    support[0, 4, 15] = False
    missing = module().photometric_refine_winner(left, right, integer, -1, 3, support,
                                                 torch.ones_like(support))
    assert not missing['valid_mask'][0, 4, 16]
    assert missing['reasons']['search_boundary'][0, 4, 16]
    assert not missing['valid_mask'][0, 0, 16]


def test_photometric_best_outside_half_pixel_basin_remains_invalid():
    left, right, integer, support = photometric_fixture(3.75)
    integer.fill_(3.)
    out = module().photometric_refine_winner(left, right, integer, -1, 3, support, support)
    assert not out['valid_mask'][0, 4, 16]
    assert out['reasons']['search_boundary'][0, 4, 16]
    assert torch.isnan(out['disparity_px'][0, 4, 16])


def test_photometric_accepts_single_channel_gray_batch_layout():
    left, right, integer, support = photometric_fixture()
    out = module().photometric_refine_winner(left[:, None], right[:, None], integer,
                                             -1, 3, support, support)
    assert out['disparity_px'].shape == integer.shape
    assert out['disparity_px'].dtype == left.dtype
    assert out['valid_mask'][0, 4, 16]


def test_photometric_missing_remote_support_cannot_fake_uniqueness():
    left, right, integer, support = photometric_fixture(3.375)
    right_support = torch.zeros_like(support)
    right_support[:, :, 11:15] = True
    out = module().photometric_refine_winner(left, right, integer, -1, 3, support, right_support)
    # Best and its +/-0.125px neighbors fit columns 11..14, while every
    # separated competitor needs missing column 15. Uniqueness is unknown.
    assert out['best_cost'][0, 4, 16] < 1e-7
    assert torch.isinf(out['second_nonlocal_cost'][0, 4, 16])
    assert not out['valid_mask'][0, 4, 16]
    assert out['reasons']['search_boundary'][0, 4, 16]


@pytest.mark.parametrize('period', [2.7, 4., 5.3])
@pytest.mark.parametrize('phase', [0., .31, 2.43])
def test_noninteger_periodic_rgb_recurrence_is_observable_from_valid_triplets(period, phase):
    x = torch.arange(80, dtype=torch.float64)
    gray = (.5 + .2 * torch.cos(2 * torch.pi * x / period + phase))[None, None].float()
    valid = torch.ones_like(gray, dtype=torch.bool)
    out = module().repeated_texture_evidence(gray, valid, 10.)
    assert out['repeated_texture'][0, 0, 20:60].all()
    assert torch.equal(out['mask'], out['repeated_texture'])
    assert out['support'][0, 0, 40].item() == 31
    assert out['period_px'][0, 0, 40].item() == pytest.approx(period, abs=1e-4)
    assert out['normalized_residual'][0, 0, 40].item() < 1e-5


def test_periodic_recurrence_excludes_all_masked_values_from_moments():
    x = torch.arange(80, dtype=torch.float64)
    original = (.5 + .2 * torch.cos(2 * torch.pi * x / 2.7 + .31))[None, None].float()
    valid = torch.ones_like(original, dtype=torch.bool)
    valid[..., [27, 38, 49]] = False
    changed = original.clone()
    changed[..., 27] = float('nan')
    changed[..., 38] = 1e30
    changed[..., 49] = -1e30
    before = module().repeated_texture_evidence(original, valid, 10.)
    after = module().repeated_texture_evidence(changed, valid, 10.)
    assert before['repeated_texture'][0, 0, 40]
    for key in ['mask', 'support', 'period_px', 'normalized_residual', 'coefficient_a', 'coefficient_b']:
        assert torch.allclose(before[key], after[key], atol=0., rtol=0., equal_nan=True), key


def test_recurrence_does_not_call_constant_ramp_or_multitone_periodic():
    x = torch.arange(96, dtype=torch.float64)
    signals = torch.stack((torch.full_like(x, .5), .1 + x / 200.,
                           .5 + .13 * torch.cos(2 * torch.pi * x / 2.7)
                           + .07 * torch.sin(2 * torch.pi * x / 5.3 + .7)))[:, None].float()
    out = module().repeated_texture_evidence(signals, torch.ones_like(signals, dtype=torch.bool), 10.)
    assert not out['repeated_texture'].any()
    assert out['normalized_residual'][2, 0, 40].item() > 1e-5


def test_recurrence_requires_supported_repetitions_inside_disparity_search():
    x = torch.arange(80, dtype=torch.float64)
    gray = (.5 + .2 * torch.cos(2 * torch.pi * x / 5.3))[None, None].float()
    valid = torch.ones_like(gray, dtype=torch.bool)
    assert not module().repeated_texture_evidence(gray, valid, 5.)['repeated_texture'].any()
    valid[..., :35] = False
    valid[..., 45:] = False
    out = module().repeated_texture_evidence(gray, valid, 10.)
    assert not out['repeated_texture'].any()
    assert out['support'].max() < 17


def test_recurrence_accepts_gray_channel_layout_and_retains_batch_shape():
    x = torch.arange(80, dtype=torch.float64)
    gray = (.5 + .2 * torch.cos(2 * torch.pi * x / 4.))[None, None].float()
    valid = torch.ones_like(gray, dtype=torch.bool)
    out = module().repeated_texture_evidence(gray[:, None], valid, 10.)
    assert out['mask'].shape == gray.shape
    assert out['mask'].device == gray.device
    assert out['mask'].dtype == torch.bool
