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
