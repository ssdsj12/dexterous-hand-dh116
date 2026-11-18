import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth


@pytest.fixture(scope='module', autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(min(old, 4))
    yield
    torch.set_num_threads(old)


def wire_scene():
    h, w, disparity = 31, 52, 7
    rng = torch.Generator().manual_seed(116)
    left = torch.rand((1, 1, h, w), generator=rng).expand(-1, 3, -1, -1).clone() * .4
    right = torch.rand((1, 1, h, w), generator=rng).expand(-1, 3, -1, -1).clone() * .4
    scale = torch.rand((22,), generator=rng) * .4 + .5
    warm = torch.tensor([.95, .60, .30])[:, None] * scale
    left[0, :, 15, 18:40] = warm
    right[0, :, 15, 18-disparity:40-disparity] = warm * .89
    # An interruption exists only in the right image. The left pixel must
    # not get a synthetic depth from its valid neighbours.
    right[0, :, 15, 26-disparity] = .12
    K = [[80., 0., w / 2], [0., 80., h / 2], [0., 0., 1.]]
    T = torch.eye(4, dtype=torch.float64)
    right_pose = T.clone()
    right_pose[0, 3] = .08
    calibration = StereoCalibration(K, K, T, right_pose, w, h, .08)
    return left, right, calibration, disparity


@pytest.mark.parametrize('method', ['wire_census', 'wire_zncc'])
def test_local_wire_matching_ignores_independent_background_and_leaves_occlusion_missing(method):
    left, right, calibration, disparity = wire_scene()
    result = StereoDepth(method=method, min_disparity=5, max_disparity=9, block_size=5,
                         subpixel=False, uniqueness_threshold=.05).compute(left, right, calibration)
    accepted = result['valid'][0]
    assert accepted[15, 23] and accepted[15, 33]
    assert result['disparity_px'][0, 15, 23] == disparity
    assert result['disparity_px'][0, 15, 33] == disparity
    assert not accepted[15, 26]
    assert not accepted[:15].any() and not accepted[16:].any()
    assert torch.isnan(result['depth_m'][0][~accepted]).all()


def test_first_reject_attribution_preserves_overlap_and_accepted_mask():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_diagnostics import wire_intersections
    wire = torch.ones(1, 1, 4, dtype=torch.bool)
    stereo = torch.tensor([[[False, False, True, True]]])
    sampled = torch.tensor([[[False, False, False, True]]])
    reasons = dict(low_texture_left=torch.tensor([[[True, False, True, False]]]),
                   left_right_inconsistent=torch.tensor([[[True, True, False, False]]]),
                   depth_range=torch.tensor([[[True, False, False, False]]]))
    maps, counts = wire_intersections(wire, stereo, sampled, reasons)
    assert maps['first_reject_code'].tolist() == [[[1, 2, 4, 0]]]
    assert counts[0]['overlapping_rejections']['left_right_inconsistent'] == 2
    assert counts[0]['first_rejections']['low_texture_left'] == 1
    assert counts[0]['first_rejections']['not_sampled_as_3d_point'] == 1
    assert counts[0]['valid_point_count'] == 1


def test_flat_wire_silhouette_requires_unique_observed_shape_in_both_views():
    left, right, calibration, disparity = wire_scene()
    flat = torch.tensor([.75, .48, .24])[:, None]
    left[0, :, 15, 18:40] = flat
    right[0, :, 15, 18-disparity:40-disparity] = flat * .89
    right[0, :, 15, 26-disparity] = .12
    result = StereoDepth(method='wire_silhouette', min_disparity=4, max_disparity=10,
                         block_size=5, subpixel=False, lr_tolerance_px=.25).compute(left, right, calibration)
    assert result['disparity_px'][0, 15, 18] == disparity  # unique visible end
    assert not result['valid'][0, 15, 26]  # occluded in right view
    assert not result['valid'][0, 15, 33]  # repeating horizontal silhouette


def test_observed_ribbon_keeps_missing_centerline_and_actual_measurement_pixels():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import TrackedWirePointExtractor
    left, _, calibration, _ = wire_scene()
    left[0, :, 14:17, 18:40] = torch.tensor([.75, .48, .24])[:, None, None]
    depth = torch.full_like(left[:, 0], float('nan'))
    depth[0, 14, 20:23] = .8  # off the skeleton but adjacent to the visible path
    depth[0, 17, 21] = .6  # outside the visible RGB mask
    result = TrackedWirePointExtractor(point_sampling='observed_ribbon').extract(
        left, depth, torch.isfinite(depth), calibration.K_left, calibration.T_world_left)
    surface = result['point_provenance'][0] == 2
    center = result['point_provenance'][0] == 1
    assert center.any() and not result['point_valid'][0, center].any()
    assert torch.isnan(result['points_world'][0, center]).all()
    assert result['point_valid'][0, surface].all()
    assert set(map(tuple, result['point_pixels'][0, surface].tolist())) == {(14, 20), (14, 21), (14, 22)}
    assert torch.isnan(result['arc_length_m'][0, surface]).all()
    assert (result['track_id'][0, surface] == -1).all()
    assert result['points_world'].shape == (1, 128, 3)


def test_ribbon_expansion_cannot_cross_an_unobserved_colour_gap():
    import numpy as np
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import _ribbon_pixels
    path = np.array([[2, 2], [2, 3]])
    foreground = np.zeros((8, 8), dtype=bool)
    foreground[2, 2:4] = True
    foreground[3:5, 2] = True
    foreground[2, 5] = True  # nearby but disconnected by an unobserved pixel
    measured = np.zeros_like(foreground)
    measured[4, 2] = measured[2, 5] = True
    assert _ribbon_pixels([path], foreground, measured, radius=1) == []
    assert _ribbon_pixels([path], foreground, measured, radius=3) == [(0, 0, 4, 2)]


def test_integer_fallback_rejected_by_right_view_is_not_a_subpixel_rejection(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_stereo_depth
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_diagnostics import wire_intersections
    left, right, calibration, disparity = wire_scene()
    index = 0, 15, 23

    def matcher(a, b, a_wire, b_wire, sign, **kwargs):
        candidate = torch.full_like(a, float('nan'))
        confidence = torch.zeros_like(a)
        good = torch.zeros_like(a_wire)
        used = torch.zeros_like(a_wire)
        if sign == -1:
            candidate[index] = disparity
            confidence[index] = .9
            good[index] = True
        return candidate, confidence, good, dict(search_boundary=used, refinement_valid=used,
                                                 optional_refinement_used=used)

    monkeypatch.setattr(tabletop_stereo_depth, 'match_wire', matcher)
    output = StereoDepth(method='wire_zncc', min_disparity=5, max_disparity=9,
                         subpixel=True).compute(left, right, calibration)
    diagnostics = output['diagnostics']
    assert not output['valid'][index]
    assert diagnostics['reasons']['right_sample_unavailable'][index]
    assert not diagnostics['reasons']['subpixel_rejected'][index]
    assert not diagnostics['optional_refinement_used'][index]
    roi = torch.zeros_like(output['valid'])
    roi[index] = True
    _, rows = wire_intersections(roi, output['valid'], torch.zeros_like(roi), diagnostics['reasons'])
    assert rows[0]['first_rejections']['right_sample_unavailable'] == 1
