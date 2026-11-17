"""Optional existing OpenCV numerical CPU checks; not camera/native evidence."""
import pytest
import torch

pytest.importorskip('cv2', reason='optional existing server OpenCV; no dependency installation')

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration, rectify_rgb
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth


def fixture():
    k = torch.tensor([[80., 0., 64.], [0., 80., 20.], [0., 0., 1.]], dtype=torch.float64)
    left_pose = torch.eye(4, dtype=torch.float64)
    right_pose = left_pose.clone(); right_pose[0, 3] = .08
    calibration = StereoCalibration(k, k, left_pose, right_pose, 128, 40, .08)
    left = torch.rand(1, 3, 40, 128, generator=torch.Generator().manual_seed(721))
    right = torch.zeros_like(left); right[..., :-6] = left[..., 6:]
    return calibration, left, right


def test_existing_opencv_sgbm_preserves_metric_depth_and_float_disparity():
    calibration, left, right = fixture()
    result = StereoDepth(method='sgbm', min_disparity=1, max_disparity=16,
        min_depth_m=.1, max_depth_m=2., block_size=3, subpixel=True).compute(left, right, calibration)
    assert int(result['valid'].sum()) > 1000
    assert result['disparity_px'].is_floating_point()
    assert float((result['depth_m'][result['valid']]-80*.08/6).abs().quantile(.95)) < .03
    assert not torch.isfinite(result['depth_m'][~result['valid']]).any()


def test_existing_opencv_rectification_returns_matching_new_geometry_and_masks():
    calibration, left, right = fixture()
    data = calibration.to_dict(); data['rectified'] = False
    original = StereoCalibration.from_dict(data)
    a, b, rectified, ma, mb = rectify_rgb(left, right, original, return_masks=True)
    assert not original.rectified and rectified.rectified
    assert a.shape == b.shape == left.shape and ma.dtype == mb.dtype == torch.bool
    assert rectified.baseline_m == calibration.baseline_m
    torch.testing.assert_close(rectified.K_left, rectified.K_right, atol=1e-8, rtol=0.)
    result = StereoDepth(max_disparity=12, min_depth_m=.1, max_depth_m=2.).compute(
        a, b, rectified, left_valid=ma, right_valid=mb)
    assert int(result['valid'].sum()) > 1000
    assert not (result['valid'] & ~ma).any()
    assert float((result['depth_m'][result['valid']]-80*.08/6).abs().quantile(.95)) < .03
