from copy import deepcopy

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
from go2_pvcnn.control.dh116.tabletop_grasp_finger_hold import GraspFingerHold
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT


def test_participating_joint_and_wrist_channels_preserve_values_and_gradients():
    initial = torch.linspace(.01, .9, 48, dtype=torch.float64).reshape(2, 24).requires_grad_()
    for grasp, expected in (("pinch", (3, 4, 5, 9, 10, 11)), ("tripod", (4, 5, 10, 11))):
        hold = GraspFingerHold(grasp)
        assert hold.held_action_indices == expected
        result = hold.project_action(initial)
        unchanged = [i for i in range(24) if i not in expected]
        assert result.dtype == initial.dtype and result.device == initial.device
        torch.testing.assert_close(result[:, unchanged], initial[:, unchanged], rtol=0, atol=0)
        assert bool((result[:, expected] == 0).all()) and bool((initial[:, expected] > 0).all())
        gradient, = torch.autograd.grad(result.sum(), initial)
        assert bool((gradient[:, expected] == 0).all())
        assert bool((gradient[:, unchanged] == 1).all())
        hold.verify_action(result)
        with pytest.raises(ValueError, match="held action"):
            hold.verify_action(initial)

    power = GraspFingerHold("power")
    assert power.project_action(initial) is initial
    power.verify_action(initial)


def _row(sdk, *, reset=1, offset=0.0):
    def snapshot(tick):
        ident = {
            "run_uuid": "scope", "env_id": 7, "episode_uid": f"episode-{reset}", "reset_id": reset,
            "rod_uid": "rod", "physics_step": tick, "rod_tick": 12 * (tick - 2),
            "timestamp_s": tick / 480, "solver_time_s": (tick - 2) / 480, "clock_valid": True,
            "policy_step": (sdk - 433) // 8 + 1, "substep": (sdk - 433) % 8,
        }
        hands = {}
        for i, side in enumerate(("left", "right")):
            names = tuple(reversed(JOINTS)) if i == 0 else JOINTS
            values = np.arange(.11, .22, .009, dtype=np.float32)[None, :len(names)].copy()
            values += np.float32(offset)
            hands[side] = dict(
                joint_names=list(names), active_indices=[names.index(joint) for joint in ACTIVE],
                prim_paths=[f"/World/envs/env_7/{side}Hand"], row_indices=[0], mounted=False,
                instance_object_id=10 + i, view_object_id=20 + i,
                q=values.copy(), backend_targets=values, backend_source="physx_backend_get_dof_position_targets",
            )
        return {"identity": [ident], "hands": hands}

    pre, post = snapshot(sdk - 1), snapshot(sdk)
    action = np.zeros((1, 24), dtype=np.float64)
    action[:, (0, 2, 6, 8)] = .3
    # The measured nonparticipating digits may comply with contact.
    for hand in post["hands"].values():
        hand["q"] += np.float32(.0005)
    expected = {
        side: {"joint_indices": hand["active_indices"], "values": hand["backend_targets"][:, hand["active_indices"]]}
        for side, hand in pre["hands"].items()
    }
    return dict(
        version="dh116-deformable-bimanual-step-v1", env_ids=[7], pre=pre, post=post,
        request={"action24": action, "action_contract": deepcopy(ACTION_CONTRACT)},
        submission={"expected_backend_targets": expected}, observed_sdk_step_delta=1, observed_rod_tick_delta=12,
    )


def test_observe_retains_first_binary32_targets_and_requires_a_new_phase_for_reset():
    hold = GraspFingerHold("tripod")
    first = _row(440)
    wrapper = dict(physics_step=440, guard_failure=None, action_readback=first)
    hold.observe(encode_owned(wrapper))
    hold.observe(_row(441))
    with pytest.raises(ValueError, match="clock|Reset"):
        hold.observe(_row(442, reset=2, offset=.015))
    hold.clear()
    hold.observe(_row(442, reset=2, offset=.015))


def test_observe_rejects_single_ulp_drift_even_when_the_target_is_stable_for_this_row():
    hold = GraspFingerHold("pinch")
    hold.observe(_row(433))
    bad = _row(434)
    for when in ("pre", "post"):
        hand = bad[when]["hands"]["left"]
        i = hand["joint_names"].index("finger31")
        value = hand["backend_targets"][0, i]
        hand["backend_targets"][0, i] = np.nextafter(value, np.float32(np.inf))
    with pytest.raises(ValueError, match="held target"):
        hold.observe(bad)


def test_observe_rejects_missing_side_and_cache_or_float64_targets():
    hold = GraspFingerHold("tripod")
    record = _row(569)
    record["pre"]["hands"].pop("right")
    with pytest.raises(ValueError, match="one left and one right"):
        hold.observe(record)
    record = _row(569)
    record["pre"]["hands"]["right"]["backend_source"] = "isaaclab_user_command_cache_joint_pos_target"
    with pytest.raises(ValueError, match="owned backend"):
        hold.observe(record)
    record = _row(569)
    hand = record["pre"]["hands"]["right"]
    hand["backend_targets"] = hand["backend_targets"].astype(np.float64)
    with pytest.raises(ValueError, match="binary32"):
        hold.observe(record)


def test_observe_rejects_wrong_joint_binding_and_guard_adjusted_submission():
    hold = GraspFingerHold("tripod")
    reordered = _row(569)
    indices = reordered["pre"]["hands"]["left"]["active_indices"]
    indices[4], indices[5] = indices[5], indices[4]
    with pytest.raises(ValueError, match="ACTIVE"):
        hold.observe(reordered)

    moved = _row(569)
    submitted = moved["submission"]["expected_backend_targets"]["right"]["values"]
    submitted[0, 5] = np.nextafter(submitted[0, 5], np.float32(np.inf))
    with pytest.raises(ValueError, match="held target"):
        hold.observe(moved)


def test_numpy_projection_preserves_full_shape_and_checks_held_channel_before_delivery():
    hold = GraspFingerHold("pinch")
    raw = np.linspace(.001, .999, 24, dtype=np.float32)
    action = hold.project_action(raw)
    assert action.dtype == raw.dtype and action.shape == raw.shape
    unchanged = [i for i in range(24) if i not in hold.held_action_indices]
    np.testing.assert_array_equal(action[unchanged], raw[unchanged])
    hold.verify_action(action)
    action[3] = np.nextafter(np.float32(0.0), np.float32(1.0))
    with pytest.raises(ValueError, match="held action"):
        hold.verify_action(action)
    raw[4] = np.nan
    with pytest.raises(ValueError, match="bounded"):
        hold.project_action(raw)
