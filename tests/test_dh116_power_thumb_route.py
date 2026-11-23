import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_command_kernels import finger_pre_guard, wrist_requested_targets
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts.certify_dh116_power_thumb_route import compile_stage, palm_grid_margins, state_at, validate_palm_lattice


def test_negative_grid_edge_remains_a_possible_contact_when_sample_is_outside():
    grid = dict(values=np.array([[[.02] * 2] * 2, [[-.002] * 2] * 2]),
                origin=np.zeros(3), spacing=.01)
    centers = np.array([[.005, .005, .005], [.01001, .005, .005]])
    margins = palm_grid_margins(grid, centers, radius=.005, lipschitz=2.2)
    assert margins["palm_domain"] > .004
    assert 0. < margins["palm_clearance"] < .000011
    # The previously out-of-grid node cannot be ignored for motion beyond that margin.
    centers[1, 0] -= .000020
    assert palm_grid_margins(grid, centers, radius=.005, lipschitz=2.2)["palm_clearance"] < 0.


def test_all_nodes_outside_grid_cannot_prove_owned_palm_coverage():
    grid = dict(values=np.full((2, 2, 2), .02), origin=np.zeros(3), spacing=.01)
    margins = palm_grid_margins(grid, np.array([[.011, .005, .005]]), radius=.005, lipschitz=1.)
    assert margins["palm_clearance"] > 0.
    assert margins["palm_domain"] < 0.


def test_compiled_actions_reproduce_the_interpolated_24d_controller_targets():
    low, high = active_limits()
    start = np.zeros((2, 12))
    start[:, 6:] = low + .25 * (high - low)
    end = start.copy()
    end[0, :3] += [.002, .003, .004]
    end[1, 3:6] += [.12, -.08, .054]
    end[:, 6:8] += [.13, .09]
    retreat = np.array([[0., .0005, .001], [0., 0., 0.]])
    states, actions = compile_stage(start, end, retreat_world_m=retreat, flex_exponents=(1., 2.))
    np.testing.assert_allclose(states[[0, -1]], [start, end], atol=1e-14)
    assert actions.shape == (len(states) - 1, 24)
    f = 1.5 / len(actions)
    midpoint = (states[1] + states[2]) / 2.
    np.testing.assert_allclose(state_at(states, f), midpoint, atol=1e-14)
    rates = dict(wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"],
                 rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"],
                 joint_rate_rad_s=ACTION_CONTRACT["joint_rate_rad_s"])
    for index, old, target in zip(range(len(actions)), states[:-1], states[1:], strict=True):
        a = torch.tensor(actions[index:index + 1])
        wrist_m, angles = wrist_requested_targets(torch.tensor(old[None, :, :3]),
                                                  torch.tensor(old[None, :, 3:6]), a, rates, 1. / 60)
        joints = torch.tensor(old[None, :, 6:].reshape(1, 12))
        q = finger_pre_guard(joints, a, joints, torch.zeros((1, 10)),
                             torch.tensor(np.tile(low, 2)), torch.tensor(np.tile(high, 2)),
                             1. / 60, rates)
        np.testing.assert_allclose(wrist_m.numpy()[0], target[:, :3], atol=1e-12)
        np.testing.assert_allclose(angles.numpy()[0], target[:, 3:6], atol=1e-12)
        np.testing.assert_allclose(q.numpy().reshape(2, 6), target[:, 6:], atol=1e-12)

def test_palm_certificate_rejects_an_uncovered_lattice_face():
    values = np.full((3, 3, 3), .012)
    values[1, 1, 1] = -.001
    assert validate_palm_lattice({"values": values}, radius=.005) == .012
    values[1, 2, 1] = .004999
    with pytest.raises(ValueError, match="face"):
        validate_palm_lattice({"values": values}, radius=.005)


def test_distant_node_velocity_does_not_take_clearance_from_near_node():
    grid = dict(values=np.full((2, 2, 2), .005), origin=np.zeros(3), spacing=.01)
    centers = np.array([[.005, .005, .005], [.10, .005, .005]])
    margins = palm_grid_margins(grid, centers, radius=.005, lipschitz=1.,
                                weights=np.array([1., .01]))
    assert margins["palm_clearance"] > .000249
    assert margins["palm_domain"] > .004999


def test_concurrent_wrist_rotation_includes_joint_motion_in_the_lever_arm():
    from scripts.certify_dh116_power_thumb_route import shape_speed_bounds

    shapes = {"jointed": np.array([[.06, 0., 0.]]), "fixed": np.array([[.04, 0., 0.]])}
    joint_speed = {"jointed": .03, "fixed": 0.}
    bounds = shape_speed_bounds(shapes, np.zeros(3), linear_speed=.01, angular_speed=1.5,
                                joint_motion=joint_speed)
    assert bounds["jointed"] == pytest.approx(.01 + .03 + 1.5 * (.06 + .03))
    assert bounds["fixed"] == pytest.approx(.01 + 1.5 * .04)
    # All pieces contribute to velocity when the finger rotates outward as the wrist rotates.
    theta = np.linspace(0., 1.5, 1001)
    radius = np.linspace(.06, .09, len(theta))
    points = radius[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() * 1000 < bounds["jointed"]


def test_right_extension_holds_left_targets_and_opens_thumb_early():
    from scripts.certify_dh116_power_right_loaded_target import right_extension

    low, high = active_limits()
    source = np.zeros((2, 12))
    source[:, 6:] = (low + high) / 2.
    right = source[1].copy()
    right[0] += .005
    right[7] -= .03
    states, actions = right_extension(source, right, opening_exponent=1.15)
    np.testing.assert_allclose(states[:, 0], np.broadcast_to(source[0], states[:, 0].shape), atol=0.)
    np.testing.assert_allclose(states[-1, 1], right, atol=0.)
    assert not np.any(actions[:, [0, 1, 2, 3, 4, 5, 12, 13, 14, 18, 19, 20]])
    assert state_at(states, .5)[1, 7] < .5 * (source[1, 7] + right[7])
    with pytest.raises(ValueError, match="bounded right thumb"):
        right_extension(source, right, opening_exponent=2.1)


def test_right_static_target_stays_bound_to_the_prior_plateau_and_material_nodes():
    from scripts.certify_dh116_power_right_loaded_target import candidate_for_boundary

    parent = dict(archive={"receipt": "same"}, clock={"physics_step": 6504})
    stations = {"right.base_link": [9], "right.finger14_link": [4], "right.finger43_link": [3]}
    target = np.arange(12, dtype=float)
    packet = dict(**parent, stations=stations, fixed_rod_only=True, native_runs=0, native_execution_ready=False,
                  hands={"right": {"accepted": [dict(result="accepted_static", geometry={"passed": True},
                                                    state=target.tolist())]}})
    np.testing.assert_array_equal(candidate_for_boundary(packet, parent, stations), target)
    with pytest.raises(ValueError, match="different archived"):
        candidate_for_boundary(packet, parent, {**stations, "right.finger43_link": [5]})
    with pytest.raises(ValueError, match="different archived"):
        candidate_for_boundary({**packet, "native_execution_ready": True}, parent, stations)


def test_certificate_chunks_cover_every_original_command_and_share_exact_endpoints():
    from scripts.certify_dh116_power_right_loaded_target import command_sections

    states = np.arange(83 * 12, dtype=float).reshape(83, 12)
    sections = list(command_sections(states, count=4))
    assert [(a, b) for a, b, _ in sections] == [(0, 20), (20, 41), (41, 61), (61, 82)]
    measured = np.concatenate([np.diff(section, axis=0) for _, _, section in sections])
    np.testing.assert_array_equal(measured, np.diff(states, axis=0))
    for (_, end, first), (start, _, second) in zip(sections, sections[1:]):
        assert end == start
        np.testing.assert_array_equal(first[-1], second[0])
