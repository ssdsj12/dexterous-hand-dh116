from pathlib import Path

import numpy as np
import pytest

from scripts import screen_dh116_power_measured_reseat as screen
from test_dh116_power_fixed_wrist_digits import PALM_RECEIPT

ROOT = Path(__file__).resolve().parents[1]


def test_live_palm_anchor_uses_its_force_bearing_material_node():
    row, _, _ = screen.load_measured_palm_boundary(PALM_RECEIPT, project=ROOT)
    _, nodes, _ = screen.measured_palm_boundary(row)
    anchors = {side: screen.palm_anchor(row["rod"], nodes, side) for side in ("left", "right")}
    for side, station in (("left", 31), ("right", 9)):
        anchor = anchors[side]
        assert anchor["node_id"] == station
        assert np.linalg.norm(anchor["pivot_world_m"] - nodes[station]) == pytest.approx(.005, abs=.00002)
        assert np.linalg.norm(anchor["normal_world"]) == pytest.approx(1.)
        assert np.linalg.norm(anchor["transverse_world"]) == pytest.approx(1.)


def test_near_node_does_not_hide_other_penetration_or_wrong_side():
    gaps = np.array([[.00001, .001, -.001], [.001, .00002, .002], [.001, .001, .00004]])
    normal_dot = np.full(gaps.shape, -.6)
    allowed = np.array([False, True, True])
    assert screen.pick_pad_witness(gaps, normal_dot, allowed, thumb=False) == (1, 1)
    gaps[1, 1] = np.nan
    assert screen.pick_pad_witness(gaps, normal_dot, allowed, thumb=False) == (2, 2)


def test_support_fast_separation_and_near_contact_keep_the_original_gate():
    from itertools import product
    from scipy.spatial.transform import Rotation
    from scripts.audit_dh116_pinch_initial_pose import box_relation

    cube = np.array(list(product((-0.003, 0.003), repeat=3)))
    shape = Rotation.from_euler("XYZ", [20, 10, 35], degrees=True).apply(cube)
    support = {"center_m": [0., 0., 0.], "size_m": [.01, .01, .01]}
    for x in (0.0, .009, .012, .030):
        vertices = shape + [x, 0., 0.]
        exact = box_relation(vertices, support["center_m"], support["size_m"])["max_separating_axis_gap_m"]
        bound = screen._support_clearance_lower({"cooked:finger53_link": vertices}, [support])
        assert bound <= exact + 1e-12
        assert (bound >= screen.CLEARANCE_M) == (exact >= screen.CLEARANCE_M)
