import numpy as np
import pytest

from scripts import verify_dh116_pinch_left_staged_contact as staged


@pytest.mark.parametrize("invalid_z_offset_m", [-.016512256994, -.0041, .0501])
def test_pinch_rejects_runtime_wrist_saturation_in_either_phase(invalid_z_offset_m):
    seed = np.array([.50, .20, .4848647085577069])
    projection = {
        "projected_root_poses": np.array([[[*seed, 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0.]]]),
        "palm_offsets_m": np.zeros((1, 2, 3)),
    }
    safe = {"pose_wxyz": [*seed, 1., 0., 0., 0.]}
    outside = {"pose_wxyz": [seed[0], seed[1], seed[2] + invalid_z_offset_m, 1., 0., 0., 0.]}
    for waypoints in ([safe, outside], [outside, safe]):
        with pytest.raises(ValueError, match=r"left Pinch waypoint .* outside original wrist"):
            staged.validate_left_wrist_workspace(projection, waypoints)


def test_pinch_uses_the_measured_reset_origin_and_allows_the_original_boundary():
    projection = {
        "projected_root_poses": np.array([[[.52, .20, .481, 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0.]]]),
        "palm_offsets_m": np.array([[[.02, 0., -.003], [0., 0., 0.]]]),
    }
    staged.validate_left_wrist_workspace(projection, [
        {"pose_wxyz": [.505, .20, .480, 1., 0., 0., 0.]},
        {"pose_wxyz": [.515, .21, .490, 1., 0., 0., 0.]},
    ])
