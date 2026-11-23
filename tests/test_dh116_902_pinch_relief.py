"""CPU regressions for physical Pinch relief loci and registration."""
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import screen_dh116_902_pinch_relief as relief


def test_trilinear_iso_edges_use_meter_origin_and_exact_radius():
    values = np.broadcast_to(np.arange(9)[:, None, None] * .001, (9, 4, 4))
    grid = dict(values=values, origin=np.array([.04, -.03, .02]), spacing=.001)
    points, normals = relief.iso_edges(grid, level=.0055)
    assert len(points) == 9
    np.testing.assert_allclose(points[:, 0], .0455, rtol=0., atol=1e-12)
    np.testing.assert_allclose(normals, np.tile([1., 0., 0.], (len(points), 1)))
    assert points[:, 1].min() == pytest.approx(-.03)

    grid["values"] = values.swapaxes(0, 1)
    y_points, y_normals = relief.iso_edges(grid, level=.0055)
    np.testing.assert_allclose(y_points[:, 1], -.0245, atol=1e-12)
    np.testing.assert_allclose(y_normals, np.tile([0., 1., 0.], (len(y_points), 1)))


def test_support_margin_distinguishes_protruding_source_from_cooked_boundary():
    source = np.array([[0., 0., 0.], [.0006, .001, 0.], [.0006, 0., .001]])
    cooked = source.copy()
    cooked[:, 0] = 0.
    centers = np.array([[.005, 0., 0.]])
    normals = np.array([[1., 0., 0.]])
    np.testing.assert_allclose(relief.directed_gap(source, centers, normals), [-.0006])
    np.testing.assert_allclose(relief.directed_gap(cooked, centers, normals), [0.], atol=1e-12)


def test_tilt_axis_keeps_the_selected_opposition_transverse_to_rod():
    normal = np.array([.8, .6, 0.])
    tangent = np.array([0., 1., 0.])
    delta = relief.transverse_aim(normal, tangent, twist=.4)
    assert delta.apply(normal) @ tangent == pytest.approx(0., abs=1e-12)


def test_geometry_guard_ranks_proximal_clearance_separately_from_sdf():
    labels = ["rod:source_colliders:finger13_link", "support:0", "sdf_penetration:finger14_link"]
    cost, blocker = relief.geometry_guard([-2., .4, -.3, .2, -.1], labels)
    assert cost == pytest.approx(.3)
    assert blocker == labels[0]


def test_right_registered_capture_and_radius_match_the_sealed_archive():
    receipt = os.environ.get("DH116_PINCH_ALIGNMENT_RECEIPT")
    prior = os.environ.get("DH116_PINCH_RELIEF_RIGHT")
    if not receipt or not prior:
        pytest.skip("sealed 902 Pinch facet archive is external")
    report = relief.screen(Path(receipt), Path(prior), side="right", pair_budget=1)
    assert report["sdk_range"] == [424, 432]
    assert report["registration"]["radius_m"] == .005
    source = report["registration"]["pads"]["finger14_link"]
    assert source["capture_sha256"] == "62615e9350789c8f4ccd319c2c78cbd584830a56e0a2b9ed39cc61d57b8c9c29"
    assert source["maximum_basis_delta"] < 1e-6
    assert report["atlas"]["finger14_link"]["iso_edge_count"] > 100
    assert report["pairs_colocated"] > 0
    assert report["proposals"] > 0
    assert report["unsafe_comparison"]["endpoint"]["rod_envelopes"]["source_colliders:finger14_link"]["lower_m"] < -.00015
    assert report["command_path_qualified"] is False
