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


def test_left_clearance_margin_changes_only_candidate_height_and_keeps_provenance():
    from scripts.prepare_dh116_paired_pinch_reset import left_clearance_candidate

    reference = dict(candidate_id="commissioned-left", side="left",
                     base_pose_wxyz=[.4, .2, .42, 1., 0., 0., 0.], active_q_rad=[.4]*6)
    trial = left_clearance_candidate(reference, .00005)
    assert trial["base_pose_wxyz"][2] == pytest.approx(.42016)
    assert trial["active_q_rad"] == reference["active_q_rad"]
    assert trial["parent_candidate_id"] == "commissioned-left"
    assert reference["base_pose_wxyz"][2] == .42
    with pytest.raises(ValueError, match="margin"):
        left_clearance_candidate(reference, -.00001)


def test_rejected_margin_is_recorded_without_a_reset_seed(monkeypatch, tmp_path):
    import json
    import sys
    import scripts.prepare_dh116_paired_pinch_reset as pinch

    path = tmp_path / "reject.json"
    monkeypatch.setattr(sys, "argv", ["pinch", "--right-result", str(tmp_path / "right.json"),
                                     "--output", str(path), "--left-extra-clearance-m", "0.00005"])
    result = dict(version="dh116-paired-pinch-reset-preflight-cpu-margin-v1", reset_seed=None,
                  passes_static_policy_clearance=False, independent_hand_y_gap_m=.2)
    monkeypatch.setattr(pinch, "prepare", lambda *args, **kwargs: result)
    assert pinch.main() == 3
    assert json.loads(path.read_text()) == result


def test_left_contact_recenter_is_bounded_and_changes_only_measured_axes():
    from scripts.prepare_dh116_paired_pinch_reset import left_clearance_candidate

    reference = dict(candidate_id="commissioned-left", side="left",
                     base_pose_wxyz=[.4, .2, .42, 1., 0., 0., 0.], active_q_rad=[.4]*6)
    trial = left_clearance_candidate(reference, .00005, root_x_m=.000055, thumb_seed_rad=.001)
    assert trial["base_pose_wxyz"] == pytest.approx([.400055, .2, .42016, 1., 0., 0., 0.])
    assert trial["active_q_rad"] == pytest.approx([.4, .401, .4, .4, .4, .4])
    assert trial["local_change"] == {"root_z_m": .00016, "root_x_m": .000055, "thumb_seed_rad": .001}
    with pytest.raises(ValueError, match="recenter"):
        left_clearance_candidate(reference, 0., root_x_m=.000055)
    with pytest.raises(ValueError, match="recenter"):
        left_clearance_candidate(reference, .00005, root_x_m=.010)
