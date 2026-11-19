import itertools

import numpy as np
import pytest


def test_tripod_search_scores_authored_and_locked_hulls_separately():
    from scripts.solve_dh116_wire_pregrasp import _constraint_violations

    hull = np.array(list(itertools.product((-.001, .001), repeat=3)))
    wire = np.array([[.004, -.010, 0.], [.004, .010, 0.]])
    row = {"shapes": {"source:pad": hull, "cooked:pad": hull + [-.006, 0., 0.]}, "dots": {}}
    violations = _constraint_violations(row, ("pad",), wire, .005)
    assert violations["source:pad:nonpenetration"] > 0.
    assert violations["cooked:pad:nonpenetration"] == 0.
    assert violations["cooked:pad:approach"] > 0.


def test_power_palm_alignment_is_a_rotation_and_aligns_span_with_rod():
    from scripts.prepare_dh116_power_palm_seed import align_palm

    palm = np.zeros((4, 3))
    fingers = np.array([[0., .4, 0.], [-.1, .5, 0.], [0., .5, 0.],
                        [.1, .5, 0.], [.2, .5, 0.]])
    frame = np.eye(3)[:, [2, 0, 1]]
    rotation = align_palm(palm, fingers, frame)
    np.testing.assert_allclose(rotation.apply([1., 0., 0.]), frame[:, 0], atol=1e-12)
    assert np.dot(rotation.apply(fingers.mean(0) - palm.mean(0)), frame[:, 1]) > 0.
    assert np.linalg.det(rotation.as_matrix()) > .999


def test_power_samples_both_proper_axis_signs():
    from scripts.prepare_dh116_power_palm_seed import palm_axis_frames

    original = np.eye(3)
    forward, reverse = palm_axis_frames(original)
    for sign, frame in (forward, reverse):
        np.testing.assert_array_equal(frame[:, 0], sign * original[:, 0])
        np.testing.assert_array_equal(frame[:, 1], original[:, 1])
        assert np.linalg.det(frame) == 1.
    with pytest.raises(ValueError, match="right-handed"):
        palm_axis_frames(np.diag([-1., 1., 1.]))


def test_power_support_clearance_measures_separation_and_keeps_partner():
    from scripts.prepare_dh116_power_palm_seed import support_clearance

    hull = np.array(list(itertools.product((-.001, .001), repeat=3)))
    supports = [dict(center_m=[0., 0., 0.], size_m=[.002] * 3)]
    assert support_clearance({"source:base_link": hull}, supports)[0] == [
        dict(source="source:base_link", support_index=0)]
    hits, separation = support_clearance({"source:base_link": hull + [.005, 0., 0.]}, supports)
    assert hits == []
    assert separation == pytest.approx(.003)


def test_tripod_witness_prefers_proven_endpoint_over_proxy_winner():
    from scripts.solve_dh116_wire_pregrasp import _tripod_witness_rank

    def row(status, dot, good=False, upper=.0001):
        gaps = {"source:thumb": dict(status=status, upper_m=upper)}
        return dict(endpoint_geometry_passed=good,
                    endpoint_witness=dict(gaps=gaps, witness_direction_dots={"cooked:middle": dot}))

    hit = row("intersection_proven", -.8, upper=-.001)
    shallow_hit = row("intersection_proven", -.1, upper=-.0001)
    unknown = row("unresolved", None)
    wrong_direction = row("clear", -.1)
    valid = row("clear", -.4, good=True)
    assert sorted([hit, unknown, wrong_direction, shallow_hit, valid], key=_tripod_witness_rank) == [
        valid, wrong_direction, unknown, shallow_hit, hit]
