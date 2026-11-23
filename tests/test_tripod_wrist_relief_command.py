import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def setup_wrist():
    root_xyz = np.array([[.01, .02, .4], [.02, -.03, .4]])
    root_world = Rotation.from_euler("XYZ", [[.3, -.2, .5], [-.2, -.1, .3]])
    angles = np.array([[.31, -.13, .2], [.11, -.21, .1]])
    rotations = root_world * Rotation.from_euler("XYZ", angles)
    palm_offsets = np.array([[.001, -.002, .001], [.003, -.004, .002]])
    seed = np.column_stack((root_xyz, root_world.as_quat()[:, [3, 0, 1, 2]]))
    targets = np.column_stack((root_xyz + palm_offsets, rotations.as_quat()[:, [3, 0, 1, 2]]))
    measured = targets.copy()
    measured[0, :3] += [.000001, -.000002, 0.]
    return seed, targets, angles, measured


def test_compiler_integrates_the_requested_left_target_via_original_projector():
    from scripts.audit_dh116_tripod_wrist_relief_command import compile_left_wrist

    seed, targets, angles, measured = setup_wrist()
    target_rotation = Rotation.from_quat(np.roll(targets[0, 3:], -1))
    shift = np.array([.00001, -.00002, .000008])
    rotation = Rotation.from_rotvec([-.0009, .0007, -.001]) * target_rotation
    goal = np.r_[targets[0, :3] + shift, np.roll(rotation.as_quat(), 1)]
    plan = compile_left_wrist(seed=seed, targets=targets, angles=angles, measured=measured,
                              clearances=[.04, .04], goal=goal, steps=4)
    assert np.flatnonzero(plan["action24"]).tolist() == [12, 13, 14, 18, 19, 20]
    np.testing.assert_allclose(plan["target_poses"][-1, 0], goal, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(plan["target_poses"][:, 1], np.tile(targets[1], (5, 1)), atol=1e-12, rtol=0.)
    assert not np.allclose(plan["action24"][18:21] * (.5 * 4 / 480), [-.0009, .0007, -.001])


def test_compiler_rejects_a_goal_that_requires_clipping_in_remaining_substeps():
    from scripts.audit_dh116_tripod_wrist_relief_command import compile_left_wrist

    seed, targets, angles, measured = setup_wrist()
    goal = targets[0].copy()
    goal[0] += .00005
    with pytest.raises(ValueError, match="original wrist rate"):
        compile_left_wrist(seed=seed, targets=targets, angles=angles, measured=measured,
                           clearances=[.04, .04], goal=goal, steps=1)


def test_compiler_rejects_under_table_projection_even_if_the_desired_delta_is_small():
    from scripts.audit_dh116_tripod_wrist_relief_command import compile_left_wrist

    seed, targets, angles, measured = setup_wrist()
    goal = targets[0].copy()
    goal[2] -= .00002
    with pytest.raises(ValueError, match="projection"):
        compile_left_wrist(seed=seed, targets=targets, angles=angles, measured=measured,
                           clearances=[.0021, .04], goal=goal, steps=4)


def test_control_budget_distinguishes_completed_midwindow_exchange():
    from scripts.audit_dh116_tripod_wrist_relief_command import remaining_control_steps

    assert remaining_control_steps(3) == 4
    assert remaining_control_steps(7) == 8
    with pytest.raises(ValueError, match="substep"):
        remaining_control_steps(8)



def test_grid_interval_keeps_an_explicit_between_sample_gap_reserve():
    from scripts.audit_dh116_tripod_wrist_relief_command import certify_pad_grid

    axis = np.arange(5) * .001 - .002
    values = np.broadcast_to((.005 + axis)[:, None, None], (5, 5, 5))
    grid = dict(values=values, origin=np.full(3, axis[0]), spacing=.001)
    sample = np.array([[[-.000001, 0., 0.]], [[-.000001, 0., 0.]]])
    proof = certify_pad_grid(grid, sample, np.array([.000002]), 0, .005, .00025)
    assert proof["witness_upper_gap_m"] == pytest.approx(.000001)
    assert proof["lower_gap_m"] == pytest.approx(-.000003)


def test_grid_interval_declines_possible_entry_of_an_unobserved_node():
    from scripts.audit_dh116_tripod_wrist_relief_command import certify_pad_grid

    grid = dict(values=np.full((5, 5, 5), .004), origin=np.full(3, -.002), spacing=.001)
    sample = np.array([[[0., 0., 0.], [.002001, 0., 0.]]] * 2)
    proof = certify_pad_grid(grid, sample, np.array([.000002, .000002]), 0, .005, .00025)
    assert not proof["lattice_covered"] and not proof["passed_depth"]
