import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_material_frame_is_orthonormal_despite_axial_palm_component():
    from tools import audit_dh116_power_coupled_wrist as audit

    nodes = np.zeros((41, 3))
    nodes[:, 1] = np.arange(41) * .01
    axes = audit.material_frame(nodes, 29, [1., .5, 0.])
    assert np.allclose(axes.T @ axes, np.eye(3))
    assert np.linalg.det(axes) == pytest.approx(1.)
    assert axes[:, 0] == pytest.approx([1., 0., 0.])
    assert axes[:, 1] == pytest.approx([0., 1., 0.])
    with pytest.raises(ValueError, match="normal"):
        audit.material_frame(nodes, 29, [0., 1., 0.])


def test_pivoted_roll_and_translation_are_independent_and_invertible():
    from tools import audit_dh116_power_coupled_wrist as audit

    pivot = np.array([.4, .2, .35])
    start = np.r_[pivot + [.1, 0., 0.], [.01, .02, .03], np.zeros(6)]
    local_delta = np.array([.001, .002, 0., 0., .12, 0.])
    seed = Rotation.from_euler("Z", .4)
    moved = audit.transform_wrist(start, seed, pivot, np.eye(3), local_delta)
    roll = Rotation.from_euler("Y", .12)
    assert moved[:3] == pytest.approx(pivot + roll.apply(start[:3] - pivot) + local_delta[:3])
    original_rotation = seed * Rotation.from_euler("XYZ", start[3:6])
    actual_rotation = seed * Rotation.from_euler("XYZ", moved[3:6])
    assert actual_rotation.as_matrix() == pytest.approx((roll * original_rotation).as_matrix())
    assert np.array_equal(moved[6:], start[6:])
    inverse_delta = np.r_[-roll.inv().apply(local_delta[:3]), -local_delta[3:]]
    restored = audit.transform_wrist(moved, seed, pivot, np.eye(3), inverse_delta)
    assert restored == pytest.approx(start)


def test_rate_checked_schedule_holds_the_right_hand_and_reverses_the_exact_waypoints():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    from tools import audit_dh116_power_coupled_wrist as audit

    low, _ = active_limits()
    start = np.stack((np.r_[np.zeros(6), low], np.r_[np.zeros(6), low]))
    end = start.copy()
    end[0, 0] += .002
    end[0, 3] += .08
    end[0, 9] += .10
    states, forward, inverse = audit.command_pair(start, end)
    assert np.array_equal(states[0], start)
    assert np.array_equal(states[-1], end)
    assert np.array_equal(states[:, 1], np.broadcast_to(start[1], states[:, 1].shape))
    assert inverse == pytest.approx(-forward[::-1])
    assert np.max(np.abs(forward)) <= 1.
    end[1, 0] += .01
    with pytest.raises(ValueError, match="right"):
        audit.command_pair(start, end)


def test_sampled_route_rejects_retained_contacts_that_stop_opposing(monkeypatch):
    from tools import audit_dh116_power_coupled_wrist as audit

    palm, thumb, partner, added = "base_link", "finger14_link", "finger23_link", "finger33_link"
    slots = {palm: 29, thumb: 37, partner: 36, added: 35}
    hand = dict(preload_targets=dict(list(slots.items())[:3]), seed=Rotation.identity(), model={},
                pad_offsets={link: np.eye(4) for link in (thumb, partner, added)},
                grids={link: {"values": np.full((3, 3, 3), .01), "link": link} for link in slots})
    monkeypatch.setattr(audit, "fk_poses", lambda *a: {link: np.eye(4) for link in slots})

    def sample(grid, local, *, dtype):
        node_id = slots[grid["link"]]
        phi = np.full(41, .02)
        phi[node_id] = .00498
        normals = np.zeros((41, 3))
        normals[node_id] = [1., 0., 0.]
        return phi, normals, np.ones(41, dtype=bool)

    monkeypatch.setattr(audit, "sample_grid", sample)
    result = audit.sample_route_stations(hand, np.zeros((2, 12)), np.zeros((41, 3)),
                                         {"wire_radius_m": .005}, {added: 35})
    assert result == {"passed": False, "tick": 0, "reason": "retained_opposition", "cosine": 1.}


def test_overlapping_hull_seed_uses_original_certificate_normal_when_refinement_has_none():
    from itertools import product

    from go2_pvcnn.control.dh116.tabletop_polyline_closest_support import refine
    from tools import audit_dh116_power_coupled_wrist as audit

    hull = np.array(list(product([-.006, .006], repeat=3)))
    nodes = np.zeros((41, 3))
    nodes[:, 1] = np.linspace(-.22, .22, 41)
    packet = refine(hull, nodes, max_segments=12)
    overlapping, = [row for row in packet["segments"] if row["segment"] == 20]
    assert overlapping["proposal"]["new_normal"] is None
    planes = audit.capsule_planes({"cooked:finger33_link": hull}, nodes, .005, ("finger33_link",))
    assert planes["cooked:finger33_link", 20] == pytest.approx(packet["original_certificate"]["segments"][20]["normal"])
