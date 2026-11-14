import numpy as np
import pytest

from scripts.prepare_dh116_paired_pinch_reset import independent_sweep_gap, make_joint_seed


def test_sweep_gap_requires_separated_source_bounds():
    assert independent_sweep_gap([.15,.26], [-.26,-.14]) == pytest.approx(.29)
    with pytest.raises(ValueError, match="overlap"):
        independent_sweep_gap([-.15,.26], [-.26,-.14])
    with pytest.raises(ValueError, match="bounds"):
        independent_sweep_gap([.26,.15], [-.26,-.14])


def test_seed_preserves_two_independent_world_poses_and_original_mimic():
    from go2_pvcnn.control.dh116.linear_coupling import expand
    left = dict(side="left", base_pose_wxyz=[.4,.2,.42,1.,0.,0.,0.], active_q_rad=[.4]*6)
    right = dict(side="right", base_pose_wxyz=[.6,-.2,.43,1.,0.,0.,0.], active_q_rad=[.5]*6)
    seed = make_joint_seed(left, right)
    assert seed["reset_root_pose_w"] == [left["base_pose_wxyz"],right["base_pose_wxyz"]]
    assert seed["authored_root_seed_w"] == seed["reset_root_pose_w"]
    assert np.array_equal(seed["joint_positions"], [expand([.4]*6),expand([.5]*6)])
    assert seed["initial_pose_qualified"] is False
    with pytest.raises(ValueError, match="independent"):
        make_joint_seed(left, left)
