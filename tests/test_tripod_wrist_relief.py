import numpy as np
import pytest


LINKS = ("left.finger14_link", "left.finger23_link", "left.finger33_link")


def plane(normal, *, gap, radius=.005):
    h = .001
    origin = np.full(3, -2 * h)
    axes = np.indices((5, 5, 5)) * h + origin[:, None, None, None]
    return dict(values=radius + gap + np.einsum("i,ijkl->jkl", normal, axes),
                origin=origin, spacing=h)


def screen(*, index_gap=.00002, thumb_gap=-.00002):
    from scripts.audit_dh116_tripod_wrist_relief import propose_wrist_relief

    nodes = np.array([[0., .02, 0.], [0., 0., -.04], [0., 0., .04]])
    quaternion = np.array([0., 0., 0., 1.])
    poses = {name: (node, quaternion) for name, node in zip(LINKS, nodes, strict=True)}
    grids = dict(zip(LINKS, (
        plane([1., 0., 0.], gap=thumb_gap), plane([-1., 0., 0.], gap=index_gap),
        plane([-1., 0., 0.], gap=-.00002)), strict=True))
    return propose_wrist_relief(palm=(np.zeros(3), quaternion), poses=poses, grids=grids,
        nodes=nodes, witness_nodes=dict(zip(LINKS, range(3), strict=True)),
        radius=.005, max_depth=.00025, old_index_gap=-.000008)


def test_projected_pose_recovers_index_gap_while_retaining_thumb_and_middle():
    result = screen()
    assert result["status"] == "sampled_pose_candidate"
    assert result["gap_at_end_m"][1] <= -.000008
    assert all(gap < 0. for gap in (result["gap_at_end_m"][0], result["gap_at_end_m"][2]))
    assert result["opposite_gap_at_end_m"][1] > result["gap_at_start_m"][1]
    assert np.linalg.norm(result["world_translation_m"]) <= .01 / 60
    assert np.linalg.norm(result["world_rotation_vector_rad"]) <= .5 / 60


def test_pose_screen_rejects_an_already_separated_protected_thumb():
    result = screen(thumb_gap=.000001)
    assert result["status"] == "protected_contact_or_depth_already_invalid"
    assert "world_translation_m" not in result


@pytest.mark.parametrize("index_gap", [.002, -.000005])
def test_pose_screen_does_not_issue_a_seed_without_a_bounded_positive_to_negative_crossing(index_gap):
    result = screen(index_gap=index_gap)
    # A target beyond the control window, or an index already in geometric contact, is ineligible.
    assert result["status"] in ("bounded_direction_did_not_restore_geometry", "no_index_gap_to_recover")
    assert "world_translation_m" not in result


def test_pose_screen_can_recover_within_a_control_window_when_one_sdk_cannot():
    result = screen(index_gap=.000080)
    assert result["status"] == "sampled_pose_candidate"
    assert result["sample_fraction"] > 1. / 8
    assert result["gap_after_one_sdk_rate_bound_m"] > 0.
    assert result["gap_at_end_m"][1] <= -.000008



def test_hull_screen_passes_the_intact_archived_rod_to_original_helper(monkeypatch):
    from scripts import screen_dh116_tripod_measured_precontact as measured
    from scripts.audit_dh116_tripod_wrist_relief import _screen_hulls

    nodes = np.column_stack((np.zeros(41), np.linspace(-.25, .25, 41), np.full(41, .4)))

    def sample(side, raw, current, tube, boxes, grids, goal, table_top, **kwargs):
        assert side == "left"
        np.testing.assert_array_equal(current, nodes)
        np.testing.assert_array_equal(goal["active_q_rad"], np.zeros(6))
        np.testing.assert_allclose(goal["pose_wxyz"][:3], [.00001, 0., 0.])
        assert tube[1].shape[0] == 20
        return [dict(shape_count=22, table_clearance_m=.02, support_clearance_m=.01,
                     noncontact_rod_clearance_m=.01, pad_gaps_m=[-.00003, -.00001, -.00002])]

    monkeypatch.setattr(measured, "measure_side", sample)
    row = dict(rod={"node_positions": nodes[None]},
               sdk_hand_geometry={"poses": {"left.base_link": (np.zeros(3), [0., 0., 0., 1.])}},
               action_readback={"post": {"hands": {"left": {"q": np.zeros((1, 6)),
                                                           "active_indices": list(range(6))}}}})
    seed = dict(world_translation_m=[.00001, 0., 0.], world_rotation_vector_rad=[0., 0., .0002])
    result = _screen_hulls(row, seed=seed, radius=.005, max_depth=.00025, grids={}, boxes=[], table_top=.3)
    assert result["sampled_hull_screen_passed"]



@pytest.mark.parametrize("damage", ["stale_post_clock", "wrong_rod_clock", "wrong_native_env"])
def test_wrist_relief_joint_readback_is_bound_to_the_physical_row(damage):
    from scripts.audit_dh116_tripod_wrist_relief import _check_readback_clock
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import VERSION

    rod = dict(run_uuid="run", env_id=0, episode_uid="episode", reset_id=1, rod_uid="rod",
               physics_step=1404, rod_tick=(1404 - 2) * 12, timestamp_s=1404 / 480,
               solver_time_s=(1404 - 2) / 480)
    source = dict(rod, policy_step=145, substep=3, clock_valid=True)
    readback = dict(version=VERSION, env_ids=[0], post=dict(identity=[source]))
    row = dict(policy_step=145, substep=3, action_readback=readback)
    _check_readback_clock(row, "post", rod)
    if damage == "stale_post_clock":
        source["physics_step"] -= 1
    elif damage == "wrong_rod_clock":
        source["solver_time_s"] -= 1 / 480
    else:
        readback["env_ids"] = [1]
    with pytest.raises(ValueError, match="same-clock"):
        _check_readback_clock(row, "post", rod)
