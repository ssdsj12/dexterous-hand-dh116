from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_trilinear_outer_box_keeps_entire_cells_adjacent_to_low_corner():
    from tools import audit_dh116_power_relocated_palm as audit

    grid = dict(origin=np.array([0., .02, .04]), spacing=.001, values=np.full((5, 5, 5), .02))
    grid["values"][2, 1, 3] = .003
    box = audit.sdf_outer_box(grid, .005)
    assert box.min(axis=0) == pytest.approx([.001, .020, .042])
    assert box.max(axis=0) == pytest.approx([.003, .022, .044])
    grid["values"][:] = .02
    with pytest.raises(ValueError, match="sublevel"):
        audit.sdf_outer_box(grid, .005)


def test_palm_index_chord_uses_conservative_between_sample_reserve():
    from tools import audit_dh116_power_relocated_palm as audit

    palm = np.array(list(product([0., .01], repeat=3)))
    index = palm + [.04, 0., 0.]
    chord = audit.chord_bounds(palm, [index, index + [.01, 0., 0.]], .001)
    assert chord[0] == pytest.approx(.029 - audit.NUMERIC_GUARD)
    assert chord[1] == pytest.approx(np.sqrt(.06**2 + 2*.01**2) + .001 + audit.NUMERIC_GUARD)


def test_tilted_family_transports_the_material_pivot_and_cannot_cancel_tilt_with_roll():
    from tools import audit_dh116_power_relocated_palm as audit

    old = np.array([0., .2, .4])
    new = np.array([0., .25, .37])
    seed = Rotation.from_euler("Z", .1)
    start = np.r_[old, np.zeros(9)]
    xyz = np.array([.001, 0., 0.])
    state, turn = audit.family_state(start, seed, old, new, np.eye(3), xyz, .7, audit.MIN_TILT_RAD, -.4)
    assert state[:3] == pytest.approx(new+xyz)
    assert turn >= audit.MIN_TILT_RAD
    reached = seed * Rotation.from_euler("XYZ", state[3:6])
    assert (reached * seed.inv()).magnitude() == pytest.approx(turn)
    assert np.array_equal(state[6:], start[6:])


def test_orientation_sector_exclusion_includes_entire_shift_box():
    from tools import audit_dh116_power_relocated_palm as audit

    start = np.zeros(12)
    old = np.zeros(3)
    axes = np.eye(3)
    low, high = np.full(3, -.01), np.full(3, .01)
    impossible = audit.sector_workspace(start, old, np.array([.09, 0., 0.]), axes, low, high, 0.)
    assert impossible["excluded"]
    assert impossible["minimum_axis_excess_lower_m"] == pytest.approx(.09-.01-.012-audit.NUMERIC_GUARD)
    feasible = audit.sector_workspace(start, old, np.array([.015, 0., 0.]), axes, low, high, 0.)
    assert not feasible["excluded"]


def test_new_station_tuple_requires_right_half_support_clearance_and_distinct_index():
    from tools import audit_dh116_power_relocated_palm as audit

    allowed = np.zeros(41, dtype=bool)
    allowed[3:20] = True
    stations = dict(base_link=11, finger14_link=7, finger43_link=8, finger23_link=4)
    audit.validate_stations(stations, allowed, previous_palm=9)
    with pytest.raises(ValueError, match="relocated"):
        audit.validate_stations(dict(stations, base_link=9), allowed, previous_palm=9)
    with pytest.raises(ValueError, match="distinct"):
        audit.validate_stations(dict(stations, finger23_link=7), allowed, previous_palm=9)
    with pytest.raises(ValueError, match="support-clear"):
        audit.validate_stations(dict(stations, finger23_link=2), allowed, previous_palm=9)
