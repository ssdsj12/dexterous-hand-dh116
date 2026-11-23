"""CPU regressions for 902 Pinch facet and continuous material screening."""
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import screen_dh116_902_pinch_facets as facets


def test_sdf_offset_locus_locates_the_original_radius_between_voxels():
    grid = dict(values=np.broadcast_to(np.arange(11)[:, None, None] * .001, (11, 3, 3)),
                origin=np.zeros(3), spacing=.001)
    locus = facets.sdf_locus(grid, [.0057, .001, .001], [1., 0., 0.])
    assert locus["offset_m"] == pytest.approx(-.0007)
    assert locus["center_body_m"] == pytest.approx([.005, .001, .001])
    assert locus["normal_body"] == pytest.approx([1., 0., 0.])


def test_facet_pair_aligns_the_center_offsets_with_neighboring_rod_material():
    pose = facets.align_pair([0., 0., 0.], [.001, 0., 0.],
                             [0., .1, 0.], [0., 1., 0.], [0., 0., 0.])
    rotation, origin, _ = pose
    points = rotation.apply([[0., 0., 0.], [.001, 0., 0.]]) + origin
    np.testing.assert_allclose(points, [[0., .0995, 0.], [0., .1005, 0.]], atol=1e-12)


def test_facet_family_roll_repositions_the_wrist_without_moving_either_locus():
    pad_centers = np.array([[0., 0., 0.], [.001, 0., 0.]])
    wrist = np.array([0., .03, 0.])
    target = [0., .1, 0.]
    rod_tangent = [0., 1., 0.]
    flat, flat_origin, _ = facets.align_pair(*pad_centers, target, rod_tangent, wrist)
    twist, twist_origin, _ = facets.align_pair(*pad_centers, target, rod_tangent, wrist, spin=.5)
    assert not np.allclose(flat_origin, twist_origin)
    np.testing.assert_allclose(flat.apply(pad_centers - wrist) + flat_origin,
                               twist.apply(pad_centers - wrist) + twist_origin, atol=1e-12)


def test_real_right_probe_maps_source_and_cooked_thumb_against_the_same_segment():
    receipt = os.environ.get("DH116_PINCH_ALIGNMENT_RECEIPT")
    prior = os.environ.get("DH116_PINCH_FACET_RIGHT")
    if not receipt or not prior:
        pytest.skip("sealed 902 facet replay is external")
    result = facets.screen(Path(receipt), Path(prior), side="right", pair_budget=1)
    assert result["sdk_range"] == [424, 432]
    thumb = result["facets"]["finger14_link"]
    assert thumb["station"] == 4
    assert thumb["sdf_gap_m"] == pytest.approx(.00039695098068)
    assert set(thumb["active_envelopes"]) == {"source_colliders", "cooked_pads"}
    for shape in thumb["active_envelopes"].values():
        assert shape["segment_start_node"] in range(2, 6)
        assert shape["facet_body"]["normal"]
        assert shape["gap_bounds_m"][0] <= shape["gap_bounds_m"][1]
    assert thumb["sdf_locus"] is not None
    source = thumb["active_envelopes"]["source_colliders"]
    cooked = thumb["active_envelopes"]["cooked_pads"]
    assert source["at_sdf_locus_gap_bounds_m"][1] < -.00015 < cooked["at_sdf_locus_gap_bounds_m"][0]
    assert "terminal_rod_envelope" in source
    assert any(row["seed"] == "guarded_pad_ray" for row in result["atlas"]["finger14_link"])
    assert result["command_path_qualified"] is False
