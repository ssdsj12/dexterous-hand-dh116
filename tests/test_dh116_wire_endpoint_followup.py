import itertools

import numpy as np
import pytest


def test_convex_witness_can_close_a_vertex_only_gap():
    from scripts.audit_dh116_wire_contact_witness import endpoint_gap

    # Tall face: the wire passes its interior, far from the available vertices.
    hull = np.array(list(itertools.product((-.001, .001), (-.008, .008), (-.008, .008))))
    nodes = np.array([[.0064, -.002, 0.], [.0064, .002, 0.]])
    result = endpoint_gap(hull, nodes, .005)
    assert result["vertex_upper_m"] > .005
    assert .000397 < result["lower_m"] < result["upper_m"] < .000403
    assert result["status"] == "clear"
    assert result["witness"]["direction"] == pytest.approx([-1., 0., 0.], abs=2e-6)
    assert result["actual_contact_measurement"] is False


def test_convex_overlap_does_not_supply_an_opposition_normal():
    from scripts.audit_dh116_wire_contact_witness import endpoint_gap

    hull = np.array(list(itertools.product((-.001, .001), (-.008, .008), (-.008, .008))))
    nodes = np.array([[.004, -.002, 0.], [.004, .002, 0.]])
    result = endpoint_gap(hull, nodes, .005)
    assert result["upper_m"] < 0.
    assert result["status"] == "intersection_proven" and result["witness"]["direction"] is None


def test_palm_anchor_preserves_offset_and_uses_real_convex_support():
    from scripts.prepare_dh116_power_palm_seed import seat_palm

    palm = np.array(list(itertools.product((-.020, .020), (-.015, .015), (-.006, .006))))
    station = np.array([.4, .2, .3])
    shift, direction = seat_palm(palm, [0., .1, .06], station, [0., 1., 0.], .00525)
    assert direction == pytest.approx([0., 0., 1.])
    assert (palm + shift)[:, 2].max() == pytest.approx(station[2] - .00525)
    with pytest.raises(ValueError, match="transverse"):
        seat_palm(palm, [0., .1, 0.], station, [0., 1., 0.], .00525)

def test_competing_segments_with_opposite_residuals_remain_unknown():
    from scripts.audit_dh116_wire_contact_witness import endpoint_gap

    hull = np.array(list(itertools.product((-.001, .001), repeat=3)))
    nodes = np.array([[.007, -.004, 0.], [.007, .04, 0.], [-.007, .04, 0.], [-.007, -.004, 0.]])
    result = endpoint_gap(hull, nodes, .005)
    assert result["status"] == "clear"
    assert result["witness"]["direction"] is None


def test_power_support_overlap_rejects_prepose_even_with_clear_table():
    from scripts.prepare_dh116_power_palm_seed import support_collisions

    hull = np.array(list(itertools.product((-.001, .001), repeat=3))) + [.43, .2, .32]
    supports = [{"center_m": [.43, .2, .32], "size_m": [.01, .01, .01]},
                {"center_m": [.43, -.2, .32], "size_m": [.01, .01, .01]}]
    assert support_collisions({"source:base_link": hull}, supports) == [
        {"source": "source:base_link", "support_index": 0}]


def test_tripod_required_source_hull_must_also_be_checked():
    from scripts.solve_dh116_wire_pregrasp import _required_source_clearance

    corners = np.array(list(itertools.product((-.001, .001), repeat=3)))
    nodes = np.array([[.004, -.01, 0.], [.004, .01, 0.]])
    shapes = {"source:finger33_link": corners, "cooked:finger33_link": corners + [-.004, 0., 0.]}
    source = _required_source_clearance(shapes, ("finger33_link",), nodes, .005)
    assert source["source:finger33_link"] < 0.
