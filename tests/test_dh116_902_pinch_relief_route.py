"""CPU regression for paired 902 Pinch endpoint and command replay bounds."""
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import certify_dh116_902_pinch_relief_route as route


def test_cross_hand_bound_uses_every_hull_and_preserves_native_side_order():
    left = {"pad": np.array([[0., .19, 0.]]), "base": np.array([[0., .16, 0.]])}
    right = {"pad": np.array([[0., -.17, 0.]]), "base": np.array([[0., -.2, 0.]])}
    assert route.cross_hand_separation(left, right) == pytest.approx(.33)
    assert route.cross_hand_separation(right, left) < 0.


def test_retreat_choice_rejects_a_clear_endpoint_with_an_unsafe_intermediate_sample():
    def probe(delta):
        x, _, z = delta
        unsafe_midpoint = x > .001 and .002 < np.linalg.norm(delta) < .006
        return None if unsafe_midpoint else abs(x) * 10 + z

    chosen = route.retreat_direction(probe, distance_m=.008)
    assert chosen["offset_m"][0] <= .001
    assert np.linalg.norm(chosen["offset_m"]) == pytest.approx(.008)


def test_paired_relief_endpoints_use_simultaneous_archived_material_frames():
    required = ("DH116_PINCH_ALIGNMENT_RECEIPT", "DH116_PINCH_RELIEF_ROUTE_LEFT", "DH116_PINCH_RELIEF_ROUTE_RIGHT")
    paths = [os.environ.get(k) for k in required]
    if not all(paths):
        pytest.skip("sealed paired endpoint reports are external")
    proof = route.screen(*map(Path, paths), run_routes=False)
    assert proof["joint_endpoint_passed"]
    assert [row["sdk"] for row in proof["paired_frames"]] == list(range(424, 433))
    assert all(row["positive_y_separation_m"] > .25 for row in proof["paired_frames"])
    assert proof["qualified_cartesian_pairs"] > 1
    assert not proof["command_path_qualified"]
