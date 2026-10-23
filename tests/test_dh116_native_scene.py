import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.native_scene import (
    NativeSceneSpec,
    _pose_matrix,
    _PANDA_GRIPPER_FINGER_LINKS,
    _root_transform_for_pad_alignment,
    candidate3_pose,
    default_native_scene_spec,
    validate_scene_spec,
    wire_orientation_wxyz,
)


ROOT = Path(__file__).parents[1]
COMBINED_USD = ROOT / "assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd"


def test_candidate3_loader_returns_distinct_left_and_right_states():
    pose = candidate3_pose(ROOT / "runs/forward_feasibility3/pose.json")
    assert pose["left"].shape == (13,)
    assert pose["right"].shape == (13,)
    assert not np.array_equal(pose["left"], pose["right"])


def test_scene_spec_rejects_combined_asset_and_duplicate_hands():
    spec = NativeSceneSpec(
        "/World/DH116DualPandaScene",
        {"left": "/World/LeftPanda", "right": "/World/RightPanda"},
        {"left": "/World/LeftDH116", "right": "/World/RightDH116"},
        {"left": COMBINED_USD, "right": COMBINED_USD},
        "panda_link8",
    )
    with pytest.raises(ValueError, match="combined|distinct"):
        validate_scene_spec(spec)


def test_native_scene_uses_two_distinct_dh116_usd_paths():
    spec = default_native_scene_spec(ROOT)
    validate_scene_spec(spec)
    assert spec.hand_usd["left"].name == "d134680ab619_DH116_left.usd"
    assert spec.hand_usd["right"].name == "3046ad2ed855_DH116_right.usd"
    assert spec.hand_usd["left"] != spec.hand_usd["right"]
    assert spec.flange_link == "panda_hand"


def test_native_panda_hand_mount_stays_visible_when_stock_fingers_are_replaced():
    assert _PANDA_GRIPPER_FINGER_LINKS == {"panda_leftfinger", "panda_rightfinger"}
    source = (ROOT / "src/go2_pvcnn/control/dh116/native_scene.py").read_text()
    assert "names = _PANDA_GRIPPER_FINGER_LINKS" in source
    assert '"panda_hand", "panda_leftfinger", "panda_rightfinger"' not in source


def test_wire_orientation_is_a_numeric_wxyz_tuple():
    orientation = wire_orientation_wxyz()
    assert isinstance(orientation, tuple)
    assert len(orientation) == 4
    assert all(isinstance(value, float) for value in orientation)


def test_pose_matrix_preserves_non_identity_quaternion_rotation():
    matrix = np.asarray(_pose_matrix(
        [0.38, 0.23, 0.36, 0.70710678, 0.70710678, 0.0, 0.0]
    ), dtype=float)
    # Gf matrices use row-vector translation storage; ExtractTranslation is
    # the binding-independent assertion for the authored world position.
    assert np.allclose(_pose_matrix(
        [0.38, 0.23, 0.36, 0.70710678, 0.70710678, 0.0, 0.0]
    ).ExtractTranslation(), [0.38, 0.23, 0.36])
    assert not np.allclose(matrix[:3, :3], np.eye(3))


def test_pad_alignment_transform_places_root_near_panda_flange():
    base_pose = np.asarray([0.38, 0.23, 0.36, 1.0, 0.0, 0.0, 0.0])
    current_pads = np.asarray([[0.44, 0.20, 0.31], [0.42, 0.20, 0.31]])
    desired_pads = current_pads + np.asarray([0.0, -0.002, 0.0])
    root_pose = _root_transform_for_pad_alignment(base_pose, current_pads, desired_pads)
    assert np.linalg.norm(root_pose[:3] - base_pose[:3]) < 0.01
    rotation = _quat_rotation(root_pose[3:])
    relative = (current_pads - base_pose[:3]) @ _quat_rotation(base_pose[3:])
    resolved = relative @ rotation.T + root_pose[:3]
    assert np.allclose(resolved, desired_pads, atol=1e-9)


def _quat_rotation(q):
    from go2_pvcnn.control.dh116.native_scene import _quat_to_matrix_wxyz

    return _quat_to_matrix_wxyz(q)

