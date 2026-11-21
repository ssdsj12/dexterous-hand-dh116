from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116 import tabletop_teacher_stereo_alignment as alignment


ROOT = Path(__file__).resolve().parents[1]


def test_teacher_and_student_load_the_same_versioned_hand_order_profile():
    legacy = StereoCalibration.from_json(ROOT / alignment.STUDENT_CAMERA)
    named = alignment.student_camera_calibration(ROOT, "hand_order")
    assert named.sha256 != legacy.sha256
    assert (named.width, named.height) == (640, 480)
    assert named.baseline_m == legacy.baseline_m == .08
    assert alignment.student_camera_calibration(ROOT, "front").sha256 == legacy.sha256
    with pytest.raises(ValueError, match="Student camera profile"):
        alignment.student_camera_calibration(ROOT, "../arbitrary_calibration.json")


def test_both_fixed_images_keep_the_physical_hands_in_left_right_order():
    calibration = alignment.student_camera_calibration(ROOT, "hand_order")
    palms = np.array([[.530815, .129565, .432040],
                      [.472626, -.124716, .454442]])
    node_y = np.linspace(-.20, .20, 41)
    nodes = np.c_[np.full(41, .43), node_y, .368 - .028*(1-(node_y/.20)**2)]
    points = np.vstack((palms, nodes))
    labels = [alignment._projection(points, K.numpy(), T.numpy(), calibration.width, calibration.height)
              for K,T in ((calibration.K_left,calibration.T_world_left),
                          (calibration.K_right,calibration.T_world_right))]
    left, right = [np.array(p["rod_pixels_uv"]) for p in labels]
    assert all(all(p["rod_in_frame"]) for p in labels)
    assert left[0,0] < left[1,0] and right[0,0] < right[1,0]
    np.testing.assert_allclose(left[:,1], right[:,1], atol=1e-10, rtol=0)
    assert (left[:,0] - right[:,0] > 0.).all()
    assert Rotation.from_matrix(calibration.T_world_left[:3,:3].numpy()).as_matrix()[1,0] < 0.


def test_named_handedness_native_plan_freezes_its_exact_file(monkeypatch, tmp_path):
    from scripts import diagnose_dh116_teacher_stereo_alignment as script
    monkeypatch.setattr(script.cleanup, "freeze_plan", lambda *args: {"frozen_files": {}})
    monkeypatch.setattr(script.legacy, "sha", lambda path: str(path))
    monkeypatch.setattr(script, "verify_tiled_sources", lambda root: {})
    plan = script.freeze_plan(ROOT, tmp_path / "unused", tmp_path / "bindings", camera_profile="hand_order")
    path = ROOT / plan["teacher_stereo"]["camera_config"]
    assert str(path) in plan["frozen_files"]
    assert plan["teacher_stereo"]["camera_profile"] == "hand_order"
    assert plan["teacher_stereo"]["calibration_sha256"] == alignment.student_camera_calibration(
        ROOT, "hand_order").sha256
