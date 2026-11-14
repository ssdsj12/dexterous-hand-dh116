"""J2-A regression checks for LR sampling, masked RGB, and source provenance.

These are CPU software invariants, never native or accuracy qualification.
"""
import hashlib
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth


@pytest.fixture(scope='module', autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def calibration(width=48, height=24, focal=40.):
    k = [[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]]
    a = torch.eye(4, dtype=torch.float64)
    b = a.clone(); b[0, 3] = .08
    return StereoCalibration(k, k, a, b, width, height, .08)


@pytest.mark.parametrize('fraction', [0., 5e-7])
@pytest.mark.parametrize('high_valid', [False, True])
def test_unused_nonfinite_high_neighbor_cannot_poison_finite_low_sample(fraction, high_valid):
    value = torch.tensor([[[5., float('nan'), 7.]]])
    valid = torch.tensor([[[True, high_valid, True]]])
    x = torch.full_like(value, fraction)
    sampled, accepted = StereoDepth.sample_horizontal(value, x, valid)
    assert accepted.all(), 'only the finite low neighbor is used within the existing 1e-6 tolerance'
    assert torch.isfinite(sampled).all()
    torch.testing.assert_close(sampled, torch.full_like(sampled, 5.), atol=0, rtol=0)


@pytest.mark.parametrize('fraction', [2e-6, .5])
@pytest.mark.parametrize('bad_high', [float('nan'), float('inf')])
def test_interpolated_sample_requires_both_used_values_finite_even_if_masks_are_true(fraction, bad_high):
    value = torch.tensor([[[5., bad_high, 7.]]])
    valid = torch.ones_like(value, dtype=torch.bool)
    sampled, accepted = StereoDepth.sample_horizontal(value, torch.full_like(value, fraction), valid)
    assert not accepted.any(), 'a validity bit cannot turn a used NaN/Inf value into a valid LR sample'


def test_nonfinite_used_low_neighbor_is_rejected_even_at_integer_coordinate():
    value = torch.tensor([[[float('nan'), 5., 7.]]])
    valid = torch.ones_like(value, dtype=torch.bool)
    _, accepted = StereoDepth.sample_horizontal(value, torch.zeros_like(value), valid)
    assert not accepted.any()


def test_unused_nan_neighbor_cannot_hide_real_lr_disagreement(monkeypatch):
    # Replace only the raw matcher boundary. The public adapter must perform
    # its real LR/depth logic: left d=4 versus used right d=5 must reject.
    def controlled(self, a, b, sign, *args, **kwargs):
        d = torch.full_like(a, 4. if sign == -1 else 5.)
        good = torch.ones_like(a, dtype=torch.bool)
        confidence = torch.ones_like(a)
        if sign == 1:
            d[:, :, 17] = float('nan')
            good[:, :, 17] = False
            confidence[:, :, 17] = 0.
        return d, confidence, good, dict(search_boundary=torch.zeros_like(good), refinement_valid=good)
    monkeypatch.setattr(StereoDepth, '_reliable_match', controlled)
    rgb = torch.rand(1, 3, 24, 48, generator=torch.Generator().manual_seed(18))
    result = StereoDepth(method='reliable_sad', max_disparity=8, subpixel=True,
                         lr_tolerance_px=.1).compute(rgb, rgb, calibration())
    center = (0, 12, 20)  # xr=16: used right d=5; unused right x=17 is NaN.
    assert result['diagnostics']['reasons']['left_right_inconsistent'][center]
    assert result['occlusion'][center]
    assert not result['valid'][center]
    assert torch.isnan(result['depth_m'][center])


def test_real_seed41_candidate_never_accepts_nonfinite_lr_sample(monkeypatch):
    left = torch.rand(1, 3, 17, 48, generator=torch.Generator().manual_seed(41))
    right = torch.zeros_like(left); right[..., :-4] = left[..., 4:]
    original = StereoDepth.sample_horizontal
    observed = []
    def observe(value, x, valid):
        sampled, accepted = original(value, x, valid)
        observed.append((sampled.clone(), accepted.clone()))
        return sampled, accepted
    monkeypatch.setattr(StereoDepth, 'sample_horizontal', staticmethod(observe))
    result = StereoDepth(method='reliable_sad', max_disparity=8, block_size=3,
                         subpixel=True).compute(left, right, calibration(height=17, focal=30.))
    assert result['valid'].any(), 'all-invalid would not demonstrate the sampling invariant'
    sampled_disparity, sample_accepted = observed[0]
    assert torch.isfinite(sampled_disparity[sample_accepted]).all()
    assert torch.isfinite(sampled_disparity[result['valid']]).all()


@pytest.fixture(scope='module')
def ignored_rgb_outputs():
    # Evaluate a continuous image independently at each view's pixel centers;
    # no left-image resampling or reference data enters the runtime matcher.
    y, x = torch.meshgrid(torch.arange(24), torch.arange(48), indexing='ij')
    x, y = x.float(), y.float()
    def image(u):
        return .4 + .04*torch.sin(.17*u) + .025*torch.sin(.33*y) + .02*torch.sin(.21*u+.19*y)
    left = image(x)[None, None].expand(1, 3, -1, -1).clone()
    right = image(x+3.75)[None, None].expand(1, 3, -1, -1).clone()
    left_valid = torch.ones_like(left[:, 0], dtype=torch.bool)
    right_valid = left_valid.clone()
    left_valid[0, 10, 23] = False
    matcher = StereoDepth(method='reliable_sad', max_disparity=10, block_size=5,
                          subpixel=True, texture_threshold=.0001)
    before = matcher.compute(left, right, calibration(), left_valid=left_valid, right_valid=right_valid)
    changed = left.clone(); changed[0, :, 10, 23] = 1.
    after = matcher.compute(changed, right, calibration(), left_valid=left_valid, right_valid=right_valid)
    assert before['valid'].sum() > 300 and after['valid'].sum() > 300
    assert not (~left_valid)[0, 5:10, 19:24].any(), 'observed point (7,21) has a fully valid patch'
    return before, after


def test_masked_rgb_values_cannot_change_output_depth_disparity_or_validity(ignored_rgb_outputs):
    before, after = ignored_rgb_outputs
    for key in ('valid', 'depth_m', 'disparity_px', 'confidence', 'occlusion'):
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0, equal_nan=True,
                                   msg=lambda message: f'{key}: {message}')


def test_masked_rgb_values_cannot_change_actual_boundary_proxy(ignored_rgb_outputs):
    before, after = ignored_rgb_outputs
    assert before['diagnostics']['rgb_boundary_proxy'] == after['diagnostics']['rgb_boundary_proxy']
    # The dictionary above records thresholds. Check the actual sensor-derived
    # map as well: comparing only constant metadata would miss the regression.
    a = before['diagnostics']['refinement']['primary']['reasons']['image_boundary_ambiguity']
    b = after['diagnostics']['refinement']['primary']['reasons']['image_boundary_ambiguity']
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_error_diagnostic_identity_binds_actual_candidate_source():
    from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import implementation_identity
    from go2_pvcnn.control.dh116 import tabletop_stereo_matcher_candidates as candidate
    sources = implementation_identity()['sources']
    assert 'tabletop_stereo_matcher_candidates' in sources
    binding = sources['tabletop_stereo_matcher_candidates']
    actual_path = Path(candidate.__file__).resolve()
    assert Path(binding['path']).resolve() == actual_path
    assert binding['sha256'] == hashlib.sha256(actual_path.read_bytes()).hexdigest()
