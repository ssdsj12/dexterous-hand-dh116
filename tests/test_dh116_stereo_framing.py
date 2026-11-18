"""Front rig projection arithmetic, never native visibility/accuracy evidence."""
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration

ROOT = Path(__file__).resolve().parents[1]
SIZES = [(224, 96), (320, 240), (640, 360), (640, 480)]


def framing():
    name = 'go2_pvcnn.control.dh116.tabletop_stereo_framing'
    assert importlib.util.find_spec(name), 'independent front-v3 framing helper missing'
    return importlib.import_module(name)


def fixture_points():
    # SDK readback supplied by the native framing review; .43 is the camera
    # aim Z, while the actual initial object centre is at Z=.368m.
    palms = [[.502083, .188617, .487541], [.522067, -.197186, .478162],
             [.483137, .177137, .495522], [.515256, -.191129, .480714]]
    wire = [[.43, -.25, .368], [.43, .25, .368]]
    return torch.tensor(palms + wire, dtype=torch.float64)


@pytest.mark.parametrize('size', SIZES)
def test_front_rig_square_K_parallel_axes_and_real_baseline(size):
    c = framing().front_calibration(*size)
    fx = size[0] / (2 * math.tan(math.radians(75) / 2))
    torch.testing.assert_close(c.K_left, torch.tensor([[fx, 0, size[0]/2], [0, fx, size[1]/2], [0, 0, 1]], dtype=torch.float64))
    torch.testing.assert_close(c.T_world_left[:3, 3], torch.tensor([1.02, -.04, .88], dtype=torch.float64))
    torch.testing.assert_close(c.T_world_right[:3, 3], torch.tensor([1.02, .04, .88], dtype=torch.float64))
    right = torch.tensor([0., 1., 0.], dtype=torch.float64)
    forward = torch.tensor([.43-1.02, 0., .43-.88], dtype=torch.float64)
    forward /= forward.norm()
    torch.testing.assert_close(c.T_world_left[:3, 0], right)
    torch.testing.assert_close(c.T_world_left[:3, 2], forward)
    torch.testing.assert_close(c.T_world_left[:3, :3], c.T_world_right[:3, :3])
    relative = torch.linalg.inv(c.T_world_left) @ c.T_world_right
    torch.testing.assert_close(relative[:3, 3], torch.tensor([.08, 0., 0.], dtype=torch.float64))


@pytest.mark.parametrize('size', SIZES)
def test_both_actual_style_palms_wire_endpoints_and_workspace_boxes_fit_both_eyes(size):
    m = framing(); c = m.front_calibration(*size)
    result = m.assess_framing(c, fixture_points(), workspace_margin_m=.05)
    assert result['all_in_view'] is True
    assert result['expanded_point_count'] == len(fixture_points()) * 8
    assert result['minimum_border_margin_px'] > 1
    assert result['minimum_optical_depth_m'] > .5
    assert result['source_scope'] == 'CPU_projection_arithmetic'
    assert result['native_visibility_qualified'] is False
    for eye in ('left', 'right'):
        assert result[eye]['in_view'].all()
        assert result[eye]['uv_px'].shape == (48, 2)


def test_projection_has_known_stereo_sign_and_rejects_behind_or_outside_without_padding():
    m = framing(); c = m.front_calibration(224, 96)
    centre = torch.tensor([[.43, 0., .43]], dtype=torch.float64)
    result = m.project_world(c, centre)
    disparity = result['left']['uv_px'][:, 0] - result['right']['uv_px'][:, 0]
    torch.testing.assert_close(disparity, c.K_left[0, 0] * c.baseline_m / result['left']['optical_depth_m'])
    torch.testing.assert_close(result['left']['uv_px'][:, 1], torch.tensor([48.], dtype=torch.float64))
    camera_centre = (c.T_world_left[:3, 3] + c.T_world_right[:3, 3])/2
    bad = torch.stack((camera_centre-c.T_world_left[:3, 2], torch.tensor([.43, 5., .43], dtype=torch.float64)))
    for eye in m.project_world(c, bad).values():
        assert not eye['in_view'].any()
    assert not m.assess_framing(c, bad, workspace_margin_m=0)['all_in_view']
    with pytest.raises(ValueError, match='finite'):
        m.project_world(c, torch.full((1, 3), float('nan')))


@pytest.mark.parametrize('size', SIZES)
def test_front_profiles_bind_new_calibration_and_keep_legacy_profiles_readonly(size):
    m = framing(); path = ROOT/f'configs/dh116_training/stereo_camera_front_{size[0]}x{size[1]}_v3.json'
    assert path.is_file(), 'new front-v3 profile missing'
    value = json.loads(path.read_text(encoding='utf-8')); c = StereoCalibration.from_dict(value)
    assert c.sha256 == m.front_calibration(*size).sha256 == value['calibration_sha256']
    old_path = ROOT/'configs/dh116_training'/('stereo_camera_v1.json' if size == (224, 96) else f'stereo_camera_{size[0]}x{size[1]}_v2.json')
    assert c.sha256 != StereoCalibration.from_json(old_path).sha256
    assert value['enabled_by_default'] is False and value['default_accuracy_profile'] is False
    assert value['source_metadata']['source_scope'] == 'authored_front_rig_from_reported_native_framing_coordinates'
    assert value['source_metadata']['native_capture_performed_for_this_rig'] is False
    torch.testing.assert_close(torch.tensor(value['framing_reference_world_m'], dtype=torch.float64), fixture_points(), rtol=0, atol=0)
    assert value['source_metadata']['reported_wire_center_xyz_m'] == [.4299999475, 0., .3680000007]
    assert value['source_metadata']['source_artifact']['sha256'] == hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
    assert value['framing_projection']['all_in_view'] is True
    depth = value['depth_estimator']
    assert depth['subpixel'] is True and depth['image_processing_batch_size'] == 1
    assert depth['max_disparity'] >= math.ceil(float(c.K_left[0, 0])*.08/depth['min_depth_m'])
    assert value['model_compatibility'] == 'new_calibration_SHA_required;no_silent_reuse_of_old_rig_models'


@pytest.mark.parametrize('size', SIZES)
def test_existing_camera_factory_uses_new_front_K_T_and_depth_budget_without_native_import(size, monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_stereo_camera as camera_module
    calls = []
    class CPUReader:
        def __init__(self, sim, origins, calibration, device, **resources):
            calls.append((calibration.sha256, origins.clone(), resources))
    monkeypatch.setattr(camera_module, '_NativeRGBPairReader', CPUReader)
    path = ROOT/f'configs/dh116_training/stereo_camera_front_{size[0]}x{size[1]}_v3.json'
    camera = camera_module.StereoCamera.from_native(None, torch.zeros(2, 3), path, device='cpu')
    assert calls[0][0] == framing().front_calibration(*size).sha256
    assert camera.depth_estimator.max_disparity >= math.ceil(float(camera.calibration.K_left[0, 0])*.08/.2)
    assert camera.depth_estimator.image_processing_batch_size == 1


def test_comparison_only_forwards_public_verified_wire_depth_measurements(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as promotion
    from go2_pvcnn.control.dh116.tabletop_stereo_resolution import compare_resolution_results
    keys = ('wire_depth_mae_m', 'wire_depth_rmse_m', 'wire_depth_p95_m', 'wire_depth_validity_coverage', 'wire_depth_count')
    verified = dict(measurements_verified=True, qualified=False, source_scope='native', width=640, height=480,
        metrics=dict(zip(keys, [.012, .014, .025, .4, 300])), blocked_reasons=['accuracy_failed'])
    monkeypatch.setattr(promotion, 'measure_stereo_evidence', lambda proof: verified)
    result = compare_resolution_results([dict(width=640, height=480, source_scope='native', schema_only=False,
        stereo_qualification='public_proof', wire_depth_mae_m=0.)])
    assert result['resolutions'][0]['wire_depth_metrics'] == {key: verified['metrics'][key] for key in keys}
    assert result['selected_resolution'] is None
    row = compare_resolution_results([dict(width=640, height=480, source_scope='cpu_fixture', schema_only=True,
        wire_depth_mae_m=0.)])['resolutions'][0]
    assert row['wire_depth_metrics'] == dict.fromkeys(keys)
