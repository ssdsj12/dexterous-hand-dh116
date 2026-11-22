"""A legal first pad contact must be followed by original PhysX reaction delivery."""
import importlib

import numpy as np
import torch


def api():
    name = "go2_pvcnn.control.dh116.tabletop_wire_first_distal"
    assert importlib.util.find_spec(name), "measured first distal contact state machine is missing"
    return importlib.import_module(name)


def row(sdk, *, pad=None, node=35, received=False):
    def route(load):
        return {side: dict(body_ids=[2], is_global=True, torques=np.zeros((1,1,3)),
                           forces=np.full((1,1,3), load if side == "left" else 0.))
                for side in ("left","right")}
    stable = {"hands": {"right": {"backend_targets": np.zeros((1,17))}}}
    pairs = [] if pad is None else [dict(mesh_name=pad,node_index=node,normal_force_on_rod_n=.02)]
    supports = [dict(mesh_name=name,node_index=i,normal_force_on_rod_n=.01)
                for name,i in (("support_0",1),("support_1",39))]
    wrenches = {"table":np.zeros(6),"support_0":np.array([0.,0.,-.01,0.,0.,0.]),
                "support_1":np.array([0.,0.,-.01,0.,0.,0.])}
    return dict(physics_step=sdk,reaction_source_physics_step=sdk-1,
                rod=dict(physics_step=sdk,rod_tick=(sdk-2)*12, hand_wrenches_com_world=route(.02 if pad else 0.),
                         rod_mesh_contact_pairs=dict(pairs=pairs+supports),external_support_wrenches_world=wrenches),
                action_readback=dict(pre=stable,post=stable),issued_composer_wrench=route(.02 if received else 0.))


def test_next_window_retains_targets_and_captures_real_reaction_at_sdk_plus_one():
    state = api().FirstDistalContact(row(17))  # Tick 18 is the final step of a window.
    pulse = torch.zeros((1,24))
    pulse[0,14] = .99
    issued, reason = state.select(pulse, {"physics_step":17})
    assert torch.equal(issued, pulse)
    assert not reason["paused"]
    assert state.observe(row(18, pad="left.finger14_link")) is None
    assert state.first_contact_sdk == 18
    assert state.first_transfer_sdk is None
    issued, reason = state.select(pulse, {"physics_step":18})
    assert not issued.any() and reason["paused"]
    assert state.observe(row(19, received=True)) is None
    assert state.first_transfer_sdk == 19


def test_wrong_link_and_wrong_material_half_cannot_be_admitted():
    for pad,node in (("right.finger14_link",35),("left.finger14_link",3),("left.finger13_link",35)):
        state = api().FirstDistalContact(row(10))
        assert state.observe(row(11,pad=pad,node=node)) == "unexpected_distal_station"
        assert state.first_contact_sdk is None


def test_right_relief_uses_world_direction_and_only_right_intrinsic_z():
    import pytest

    u = np.array([.3826834323650897,0.,-.9238795325112867])
    action = api().right_relief_action(u, travel_m=.003, rotation_rad=.02, windows=18)
    assert not action[np.r_[:15,18:23]].any()
    np.testing.assert_allclose(action[15:18],u,rtol=0.,atol=1e-14)
    assert action[23] == pytest.approx(.02*60/(18*.5))
    with pytest.raises(ValueError,match="rate"):
        api().right_relief_action(u, travel_m=.003, rotation_rad=.02, windows=17)


def test_contact_pair_without_native_wrench_is_rejected():
    state = api().FirstDistalContact(row(10))
    empty = row(11, pad="left.finger14_link")
    empty["rod"]["hand_wrenches_com_world"]["left"]["forces"][:] = 0.
    assert state.observe(empty) == "distal_pair_without_reaction"
    assert state.first_contact_sdk is None


def test_inactive_wrist_accepts_only_seed_bound_mount_conversion():
    from copy import deepcopy

    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import compose_rotation
    from go2_pvcnn.control.dh116.tabletop_palm_mount import local_translation

    reset_pose = np.array([.53014244464784, .12239645099196261, .44256242988167543,
                           -.4263382994182291, -.4701642785671765, .6254986943669114, .45379796049989923])
    anchor = reset_pose.astype(np.float32)[None]
    projected = reset_pose.copy()
    projected[3:] = compose_rotation(torch.from_numpy(reset_pose[3:]), torch.zeros(3)).numpy()
    zero = np.zeros((1, 17), dtype=np.float32)
    post = zero.copy()
    post[:, :3] = local_translation(torch.from_numpy(reset_pose[None]),
                                    torch.from_numpy(anchor).double()).float().numpy()
    assert np.any(post)
    hand = dict(joint_names=["wrist_x", "wrist_y", "wrist_z", *[f"other{i}" for i in range(14)]],
                mount_anchor_pose_w=anchor)
    readback = dict(pre={"hands": {"left": dict(hand, backend_targets=zero)}},
                    post={"hands": {"left": dict(hand, backend_targets=post)}},
                    request={"action24": np.zeros((1, 24))},
                    submission={"projected_root_poses": np.tile(projected, (1, 2, 1)),
                                "projected_angles": np.zeros((1, 2, 3))})
    held = api().HeldPalmTarget("left", reset_pose)
    assert held.observe(readback)
    stable = deepcopy(readback)
    stable["pre"]["hands"]["left"]["backend_targets"] = post.copy()
    assert held.observe(stable)
    for key in ("pose", "action", "backend"):
        discrepancy = deepcopy(stable)
        if key == "pose":
            discrepancy["submission"]["projected_root_poses"][0, 0, 0] += 1e-10
        elif key == "action":
            discrepancy["request"]["action24"][0, 12] = 1e-10
        else:
            discrepancy["post"]["hands"]["left"]["backend_targets"][0, 0] += 1e-10
        assert not held.observe(discrepancy)


def test_tangential_load_without_a_valid_normal_pair_blocks_the_approach():
    state = api().FirstDistalContact(row(10))
    rubbing = row(11, pad="left.finger14_link")
    rubbing["rod"]["rod_mesh_contact_pairs"]["pairs"][0]["normal_force_on_rod_n"] = 0.
    assert state.observe(rubbing) == "left_wrench_without_positive_normal_pair"
    assert state.first_contact_sdk is None


def test_settled_left_wrist_path_only_issues_the_screened_translation():
    import pytest

    mod = api()
    delta = np.array([.0056, -.0014, .0114])
    action = mod.settled_left_action(delta, windows=96)
    assert np.count_nonzero(action) == 3
    np.testing.assert_allclose(action[12:15] * .01 * 96 / 60, delta)
    state = mod.FirstDistalContact(row(1426), approach=action)
    command = torch.as_tensor(action)[None]
    assert torch.equal(state.select(command, {"physics_step": 1426})[0], command)
    wrong = command.clone()
    wrong[0, 13] *= -1.
    with pytest.raises(ValueError, match="screened"):
        state.select(wrong, {"physics_step": 1426})
    assert state.observe(row(1427, pad="left.finger23_link")) == "unexpected_distal_station"
    assert state.observe(row(1427, pad="left.finger14_link")) is None
    assert not state.select(command, {"physics_step": 1427})[0].any()
    assert state.observe(row(1428, received=True)) is None
    assert state.first_transfer_sdk == 1428
    with pytest.raises(ValueError, match="rate"):
        mod.settled_left_action(delta, windows=12)


def test_live_quiet_boundary_rejects_a_different_wire_and_joint_pose():
    from copy import deepcopy

    import pytest
    from scripts.verify_dh116_wire_live_palm_approach import validate_settled_handoff

    sdk = 1426
    actual = dict(rod=dict(physics_step=sdk, rod_tick=(sdk-2)*12,
                          node_positions=np.zeros((1, 41, 3))),
        hand_link_poses={side+".base_link": (np.zeros(3), np.array([0., 0., 0., 1.]))
                         for side in ("left", "right")},
        native_joint_state={side: dict(names=["x", "y"], position=np.zeros((1, 2)),
                                      target=np.zeros((1, 2))) for side in ("left", "right")})
    assert validate_settled_handoff(actual, deepcopy(actual), sdk)["node_max_error_m"] == 0.
    modified = deepcopy(actual)
    modified["rod"]["node_positions"][0, 35, 2] += 2e-6
    with pytest.raises(ValueError, match="centerline"):
        validate_settled_handoff(actual, modified, sdk)
    modified = deepcopy(actual)
    modified["native_joint_state"]["left"]["position"][0, 1] += .002
    with pytest.raises(ValueError, match="joint"):
        validate_settled_handoff(actual, modified, sdk)
