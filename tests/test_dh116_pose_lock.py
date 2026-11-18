import hashlib
import json

import numpy as np
import pytest

from go2_pvcnn.control.dh116.pose_lock import load_pose_lock
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE


def write_pose(tmp_path, **updates):
    data = {
        "schema": "dh116-static-pose-v1",
        "asset_sha256": "a" * 64,
        "layers": {"source": "b" * 64},
        "states": {
            "left": [0.0] * 7 + [0.2] * 6,
            "right": [0.0] * 7 + [0.2] * 6,
        },
        "grasp_qualified": False,
    }
    data.update(updates)
    path = tmp_path / "candidate3.json"
    path.write_text(json.dumps(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate3_pose_lock_exposes_one_active_reference_and_bounds(tmp_path):
    path, digest = write_pose(tmp_path)
    lock = load_pose_lock(path, digest, "a" * 64, {"source": "b" * 64}, tolerance_rad=0.25)
    assert lock.active_reference.shape == (12,)
    assert lock.active_reference.tolist() == [0.2] * 12
    lower, upper = lock.active_bounds(np.zeros(12), np.ones(12))
    np.testing.assert_allclose(lower, 0.0)
    np.testing.assert_allclose(upper, 0.45)
    assert lock.active_names == tuple(f"{side}_{name}" for side in ("left", "right") for name in ACTIVE)


@pytest.mark.parametrize("bad", ["sha", "asset", "layers", "qualified"])
def test_candidate3_pose_lock_rejects_identity_or_scope_changes(tmp_path, bad):
    updates = {"grasp_qualified": True} if bad == "qualified" else {}
    path, digest = write_pose(tmp_path, **updates)
    if bad == "sha":
        digest = "0" * 64
    if bad == "asset":
        with pytest.raises(ValueError, match="identity"):
            load_pose_lock(path, digest, "c" * 64, {"source": "b" * 64})
        return
    if bad == "layers":
        with pytest.raises(ValueError, match="identity"):
            load_pose_lock(path, digest, "a" * 64, {"other": "b" * 64})
        return
    with pytest.raises(ValueError):
        load_pose_lock(path, digest, "a" * 64, {"source": "b" * 64})


def test_candidate3_pose_lock_rejects_nonfinite_or_invalid_tolerance(tmp_path):
    path, digest = write_pose(tmp_path)
    data = json.loads(path.read_text())
    data["states"]["left"][7] = float("nan")
    path.write_text(json.dumps(data, allow_nan=True))
    with pytest.raises(ValueError):
        load_pose_lock(path, hashlib.sha256(path.read_bytes()).hexdigest(), "a" * 64,
                       {"source": "b" * 64})
    path, digest = write_pose(tmp_path)
    with pytest.raises(ValueError):
        load_pose_lock(path, digest, "a" * 64, {"source": "b" * 64}, tolerance_rad=0.)
