import numpy as np
import pytest


def test_largest_clear_advance_preserves_the_original_channel_ratios():
    from scripts.audit_dh116_tripod_early_boundary import select_precontact_advance

    authored = np.full(24, .1)
    result = select_precontact_advance(authored, lambda move: move[0] <= .035)
    assert .349 < result["fraction"] <= .35
    np.testing.assert_allclose(result["action24"], authored * result["fraction"])


def test_precontact_advance_cannot_leave_an_inadmissible_start():
    from scripts.audit_dh116_tripod_early_boundary import select_precontact_advance

    checked = []
    def blocked(move):
        checked.append(move.copy())
        return False

    assert select_precontact_advance(np.full(24, .1), blocked)["status"] == "zero_action_proof_failed"
    assert len(checked) == 1 and not checked[0].any()


def test_scan_refuses_a_midwindow_row_even_if_its_sdk_is_consecutive():
    from scripts.audit_dh116_tripod_early_boundary import require_boundary

    assert require_boundary({"physics_step": 592, "policy_step": 43, "substep": 7},
                            sdk=592, start_sdk=240) == 44
    with pytest.raises(ValueError, match="complete authored control"):
        require_boundary({"physics_step": 592, "policy_step": 43, "substep": 3}, sdk=592, start_sdk=240)



def test_precontact_action_uses_both_sides_of_the_original_target_integrator():
    from scripts.audit_dh116_tripod_early_boundary import _controller_endpoint

    seed = np.array([[.4, .1, .44, 1., 0., 0., 0.],
                     [.4, -.1, .44, 1., 0., 0., 0.]])
    measured = seed.copy()
    actual = np.tile([.35, .48, .71, .76, .70, .95], 2)
    state = (seed, np.zeros((2, 3)), measured, {}, actual, actual)
    row = dict(sdk_hand_geometry={"table_clearance_m": np.full((1, 2), .04)})
    command = np.zeros(24)
    command[[2, 9, 12, 16, 18, 23]] = [.02, -.03, .12, -.14, .09, -.11]
    plan = _controller_endpoint(command, row, seed, state)

    assert plan is not None
    np.testing.assert_allclose(plan["finger_targets_rad"], actual + command[:12] * (.3 / 60), atol=1e-13)
    np.testing.assert_allclose(np.array(plan["target_poses_wxyz"])[:, :3] - seed[:, :3],
                               command[12:18].reshape(2, 3) * (.01 / 60), atol=1e-13)
    np.testing.assert_allclose(plan["intrinsic_angles_rad"], command[18:24].reshape(2, 3) * (.5 / 60),
                               atol=1e-13)
