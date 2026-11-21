import numpy as np
import pytest


def test_archived_rod_tube_rejects_intermediate_intrusion_with_clear_end_samples():
    from scripts.screen_dh116_tripod_supported_reset import archived_tube, separation

    rod = np.zeros((3, 41, 3))
    rod[:, :, 1] = np.linspace(-.25, .25, 41)
    rod[:, :, 0] = np.array([.01, 0., .01])[:, None]
    shape = np.array([[x, y, z] for x in (-.001, .001) for y in (-.001, .001) for z in (-.001, .001)])
    assert separation(shape, *archived_tube(rod[[0, 2]])) > 0.
    assert separation(shape, *archived_tube(rod)) < 0.
    with pytest.raises(ValueError, match="41"):
        archived_tube(rod[:, :-1])


def test_tripod_scene_scope_uses_separate_seed_and_preserves_original_supports():
    from scripts.screen_dh116_tripod_supported_reset import scene_request, SCOPE
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state

    request, config = scene_request()
    table = {"center_m": [0., 0., .275], "size_m": [1., 1., .05]}
    assert config["scope"] == request["scope"] == SCOPE
    with pytest.raises(ValueError):
        prepare_initial_state(config, table)
    seed, active, nodes, supports = prepare_initial_state(config, table, expected_scope=SCOPE)
    assert active.shape == (2, 6) and nodes.shape == (41, 3)
    assert set(supports) == {"support_0", "support_1"}
    assert seed["reset_root_pose_w"][0][2] > seed["reset_root_pose_w"][1][2]-.03


def test_tripod_approach_is_a_four_window_translation_from_real_reset():
    from scripts.verify_dh116_tripod_supported_reset import approach_commands

    schedule = approach_commands()
    assert schedule.shape == (4, 24)
    np.testing.assert_array_equal(np.flatnonzero(schedule[0]), [14, 15, 16, 17])
    np.testing.assert_allclose(schedule[:, 14], -.75)
    assert np.any(schedule[:, 15:18]) and not np.any(schedule[:, 18:])


@pytest.mark.parametrize("remaining, good", [(8, False), (32, True), (40, False)])
def test_shared_motion_schedule_checks_exact_sdk_budget(remaining, good):
    from scripts.verify_dh116_power_shared_rod import supported_action_schedule

    if good:
        assert supported_action_schedule(np.zeros((4, 24)), remaining).shape == (4, 24)
    else:
        with pytest.raises(ValueError, match="control"):
            supported_action_schedule(np.zeros((4, 24)), remaining)


def test_waypoint_controls_use_original_action_units_and_intrinsic_wrist():
    from scipy.spatial.transform import Rotation
    from scripts.verify_dh116_tripod_supported_reset import waypoint_controls, screen
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE

    _, scene = screen.scene_request()
    seeds = [scene["hands"][s]["pose_wxyz"] for s in ("left", "right")]
    seed = np.asarray(seeds)
    nominal = seed.copy()
    nominal[:, 0] -= .0002
    states = {s:{"joint_names":list(ACTIVE), "active_indices":list(range(6)),
                 "backend_targets":np.asarray([scene["hands"][s]["q_rad"]])}
              for s in ("left", "right")}
    row = dict(wrist_projection=dict(projected_root_poses=nominal[None], projected_angles=np.zeros((1,2,3))),
               action_readback=dict(post=dict(hands=states)))
    desired = {}
    for i,side in enumerate(("left","right")):
        pose=seed[i].copy()
        target=Rotation.from_quat(pose[[4,5,6,3]]) * Rotation.from_euler("XYZ",[.01,-.02,0.])
        pose[3:]=target.as_quat()[[3,0,1,2]]
        desired[side]=dict(pose_wxyz=pose,active_q_rad=np.asarray(scene["hands"][side]["q_rad"])+.001)
    planned=waypoint_controls(scene,row,desired)
    assert planned.shape == (20,24)
    np.testing.assert_allclose(planned[:,:12].sum(0)*.3/60,.001)
    np.testing.assert_allclose(planned[:,12:18].sum(0)*.01/60,[.0002,0,0,.0002,0,0])
    np.testing.assert_allclose(planned[:,18:24].sum(0)*.5/60,[.01,-.02,0,.01,-.02,0],atol=1e-12)
