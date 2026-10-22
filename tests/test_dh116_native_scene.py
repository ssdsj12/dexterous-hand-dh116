import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.native_scene import (
    NativeSceneSpec,
    candidate3_pose,
    validate_scene_spec,
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
    left = ROOT / "assets/fixed_dual_panda_dh116_v1/dependencies/d134680ab619_DH116_left.usd"
    right = ROOT / "assets/fixed_dual_panda_dh116_v1/dependencies/3046ad2ed855_DH116_right.usd"
    spec = NativeSceneSpec(
        "/World/DH116DualPandaScene",
        {"left": "/World/LeftPanda", "right": "/World/RightPanda"},
        {"left": "/World/LeftDH116", "right": "/World/RightDH116"},
        {"left": left, "right": right},
        "panda_link8",
    )
    validate_scene_spec(spec)
    assert spec.hand_usd["left"].name == "d134680ab619_DH116_left.usd"
    assert spec.hand_usd["right"].name == "3046ad2ed855_DH116_right.usd"
    assert spec.hand_usd["left"] != spec.hand_usd["right"]

