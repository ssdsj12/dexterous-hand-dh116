import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_rod_separation_certificate import RodCapsuleSeparation


def _box(center, half=.001):
    offsets = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    return np.asarray(center) + half * offsets


def test_finite_rod_endpoint_leaves_axial_clearance():
    nodes = [[0., -.02, 0.], [0., .02, 0.]]
    rod = RodCapsuleSeparation(nodes, .005, node_motion_bound_m=np.zeros(2))
    assert rod.gap_lower_m(_box([0., .04, 0.])) == pytest.approx(.014 - 2e-6, abs=1e-10)


def test_curved_rod_does_not_fill_the_interior_of_its_convex_hull():
    rod = RodCapsuleSeparation([[-.03, -.03, 0.], [-.03, .03, 0.],
                                [.03, .03, 0.], [.03, -.03, 0.]],
                               .005, node_motion_bound_m=np.zeros(4))
    assert rod.gap_lower_m(_box([0., 0., 0.])) == pytest.approx(.024 - 2e-6, abs=1e-10)


def test_drift_bounds_are_applied_to_each_node_and_its_adjacent_segments():
    nodes = [[0., -.02, 0.], [0., .02, 0.]]
    fixed = RodCapsuleSeparation(nodes, .005, node_motion_bound_m=np.zeros(2))
    uncertain = RodCapsuleSeparation(nodes, .005, node_motion_bound_m=[0., .003])
    box = _box([0., .027, 0.])
    assert fixed.gap_lower_m(box) > 0.
    assert uncertain.gap_lower_m(box) < 0.
    with pytest.raises(ValueError, match="motion"):
        RodCapsuleSeparation(nodes, .005, node_motion_bound_m=[.001, -.001])


def test_clear_endpoints_do_not_certify_a_path_crossing_the_rod():
    rod = RodCapsuleSeparation([[0., -.02, 0.], [0., .02, 0.]], .005,
                               node_motion_bound_m=np.zeros(2))
    path = lambda t: {"proximal": _box([.015 - .03 * t, 0., 0.])}
    travel = {"proximal": .03}
    assert rod.gap_lower_m(path(0.)["proximal"]) > 0.
    assert rod.gap_lower_m(path(1.)["proximal"]) > 0.
    check = rod.certify_path(path, travel, margin_m=0.)
    assert check["proved"] is False
    assert check["violations"][0]["fraction"] == .5


def test_clear_arm_path_proves_nonnegative_continuous_separation():
    rod = RodCapsuleSeparation([[0., -.02, 0.], [0., .02, 0.]], .005,
                               node_motion_bound_m=[.001, .001])
    check = rod.certify_path(lambda t: {"proximal": _box([.015, .01 * t, 0.])},
                             {"proximal": .01}, margin_m=.002)
    assert check["proved"] is True
    assert check["minimum_clearance_margin_m"] >= 0.
