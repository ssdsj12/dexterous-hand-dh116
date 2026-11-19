import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_power_material_roll_rotates_transverse_offset_about_same_station():
    from scripts.audit_dh116_power_material_reach import material_wrist_seed

    root = np.array([0.02, .1, .03, 1., 0., 0., 0.])
    station = np.array([0., .1, .03])
    tangent = np.array([0., 1., 0.])
    inward = np.array([1., 0., 0.])
    pose = material_wrist_seed(root, station, tangent, inward, angle_deg=-30.,
                              lateral_m=.004, backoff_m=.004)
    turn = Rotation.from_quat(np.roll(pose[3:], -1))
    delta = (root[:3] - station + [.0, .0, -.004] - [.004, .0, .0])
    np.testing.assert_allclose(pose[:3] - station, turn.apply(delta), atol=1e-14)
    assert np.linalg.det(turn.as_matrix()) == pytest.approx(1.)


def test_power_palm_seating_search_brackets_before_penetration():
    from scripts.audit_dh116_power_material_reach import palm_approach_samples

    def measure(inward):
        return {"palm_gap_m": .0197 - inward}

    rows = palm_approach_samples(measure)
    assert any(0. <= row["palm_gap_m"] <= .0005 for row in rows)
    assert any(row["palm_gap_m"] < 0. for row in rows)
    assert max(row["inward_m"] for row in rows) <= .024


def test_power_wrist_spin_keeps_the_world_palm_witness_fixed():
    from scipy.spatial.transform import Rotation
    from scripts.audit_dh116_power_material_reach import rotate_root_about_witness

    root = np.array([0., 0., 0., 1., 0., 0., 0.])
    palm_world = np.array([1., 0., 0.])
    around_rod = rotate_root_about_witness(root, palm_world, [0., 1., 0.], 20.)
    around_up = rotate_root_about_witness(root, palm_world, [0., 0., 1.], 20.)
    for result in (around_rod, around_up):
        rotation = Rotation.from_quat(np.roll(result[3:], -1))
        np.testing.assert_allclose(result[:3] + rotation.apply(palm_world), palm_world, atol=1e-12)
    assert around_rod[2] > .3
    assert around_up[2] == 0.


def test_power_station_change_transports_wrist_and_inward_in_the_rod_frame():
    from scripts.audit_dh116_power_material_reach import transport_power_station
    from scripts.prepare_dh116_wire_pose_candidate import _material_tangents

    nodes = np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.], [0., 3., 0.]])
    root = np.array([1., 1., .3, 1., 0., 0., 0.])
    target, inward = transport_power_station(root, [-1., 0., 0.], nodes,
                                             _material_tangents(nodes), source=1, target=2)
    np.testing.assert_allclose(target, [1., 2., .3, 1., 0., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(inward, [-1., 0., 0.], atol=1e-12)


def test_power_digit_bound_accounts_for_a_gap_between_sample_knots():
    from scripts.audit_dh116_power_material_reach import summarize_middle_reach

    rows = [
        {"gap_lower_m": .001, "gap_upper_m": .0011, "clear": True},
        {"gap_lower_m": .001, "gap_upper_m": .0011, "clear": False},
    ]
    possible = summarize_middle_reach(rows, point_travel_m=.0011)
    assert not possible["fixed_wrist_reach_excluded"]
    assert possible["best_clear_sample_upper_m"] == .0011
    excluded = summarize_middle_reach(rows, point_travel_m=.0004)
    assert excluded["fixed_wrist_reach_excluded"]
    assert excluded["continuous_minimum_lower_m"] > .0005
