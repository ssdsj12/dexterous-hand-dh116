import importlib
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def probe():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_depth_camera_probe')


def test_front_camera_forward_axis_and_world_up(probe):
    eye = np.array([.1, 0., 1.05])
    target = np.array([.48, 0., .43])
    transform = probe.look_at_optical(eye, target, [0., 0., 1.])
    assert np.allclose(transform[:3, 3], eye)
    assert np.allclose(transform[:3, 2], (target-eye)/np.linalg.norm(target-eye))
    assert np.allclose(transform[:3, 0], [0., -1., 0.])
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.)
    assert transform[2, 1] < 0.


def test_usd_to_optical_flips_y_z_only(probe):
    actual = probe.usd_pose_to_optical([1., 2., 3.], [1., 0., 0., 0.])
    assert np.allclose(actual[:3, :3], np.diag([1., -1., -1.]))
    assert np.allclose(actual[:3, 3], [1., 2., 3.])


def test_brown_mask_uses_visible_color_and_rejects_dark_grey(probe):
    image = np.array([[[160, 90, 30], [255, 255, 255], [10, 5, 1],
                       [90, 90, 90], [20, 60, 140], [255, 0, 0]]], dtype=np.uint8)
    assert probe.brown_rod_mask(image).tolist() == [[True, False, False, False, False, False]]


def test_brown_mask_float_and_uint8_agree(probe):
    image = np.array([[[160, 90, 30], [255, 255, 255]]], dtype=np.uint8)
    assert np.array_equal(probe.brown_rod_mask(image), probe.brown_rod_mask(image/255.))


def test_finite_cylinder_distance_includes_caps(probe):
    points = np.array([[.008, 0., 0.], [0., .25, 0.], [0., 0., 0.],
                       [.018, .26, 0.]])
    distances = probe.cylinder_surface_distances(points, [0, 0, 0, 1, 0, 0, 0], .008, .5)
    assert np.allclose(distances, [0., 0., .008, np.sqrt(2.)*.01])


def test_cylinder_distance_translated_and_rotated(probe):
    # +90 degrees around Z maps cylinder local Y to world -X.
    points = np.array([[.75, 2., 3.], [1., 2.008, 3.]])
    distances = probe.cylinder_surface_distances(points, [1, 2, 3, 2**-.5, 0, 0, 2**-.5], .008, .5)
    assert np.allclose(distances, 0., atol=1e-12)


def test_capture_refuses_large_environment_count_before_importing_isaac(probe):
    with pytest.raises(ValueError, match='exactly two'):
        probe.capture_depth_coordinates(None, SimpleNamespace(n=4096), None, SimpleNamespace())


def test_camera_extension_is_enabled_before_import(probe, monkeypatch):
    calls = []
    camera_type = object()
    manager = SimpleNamespace(set_extension_enabled_immediate=lambda name, enabled:
                              calls.append(('enable', name, enabled)))
    def import_module(name):
        if name == 'omni.kit.app':
            return SimpleNamespace(get_app=lambda: SimpleNamespace(get_extension_manager=lambda: manager))
        assert name == 'isaacsim.sensors.camera'
        assert calls == [('enable', 'isaacsim.sensors.camera', True)]
        calls.append(('import', name))
        return SimpleNamespace(Camera=camera_type)
    monkeypatch.setattr(importlib, 'import_module', import_module)
    assert probe._load_camera_class() is camera_type
    assert calls[-1] == ('import', 'isaacsim.sensors.camera')


def test_frame_artifacts_keep_pixel_mapping_and_roundtrip(probe, tmp_path):
    depth = np.array([[1., np.inf], [1., 0.]])
    rgb = np.array([[[160, 90, 30], [160, 90, 30]], [[255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    poses = np.array([[0, 0, 0, 1, 0, 0, 0], [2, 3, 4, 1, 0, 0, 0]], dtype=float)
    result = probe.save_frame_artifacts(tmp_path, 'pinch', rgb, depth, np.eye(3), np.eye(4),
        poses, [0, 0, 1, 1, 0, 0, 0], .008, .5, min_depth_m=.1, max_depth_m=2.)
    assert result['valid_pixels'] == 2
    assert result['object_mask_points'] == 1
    assert result['hand_world_roundtrip_max_m'] < 1e-12
    saved = np.load(tmp_path/'pinch_depth_coordinates.npz')
    assert saved['points_in_hands'].shape == (2, 1, 3)
    assert np.array_equal(saved['pixels_uv'], [[0, 0]])
    assert saved['all_points_world'].shape == (2, 3)
    assert not any('gt' in key or 'rod_pose' in key or 'contact' in key for key in saved.files)
    assert result['sample_point']['pixel_uv'] == [0, 0]
    assert result['sample_point']['left_hand_xyz_m'] == [0., 0., 1.]
    assert result['sample_point']['right_hand_xyz_m'] == [-2., -3., -3.]
    assert (tmp_path/'pinch_rgb_depth.png').is_file()


def test_artifact_rejects_no_visible_rod(probe, tmp_path):
    with pytest.raises(RuntimeError, match='brown'):
        probe.save_frame_artifacts(tmp_path, 'missing', np.zeros((2,2,3), dtype=np.uint8),
            np.ones((2,2)), np.eye(3), np.eye(4),
            np.array([[0,0,0,1,0,0,0]]*2), [0,0,0,1,0,0,0], .008, .5,
            min_depth_m=.1, max_depth_m=2.)
