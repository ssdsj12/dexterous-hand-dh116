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


def test_transported_palm_preserves_local_witness_and_uses_proposed_node():
    from scipy.spatial.transform import Rotation
    from scripts.prepare_dh116_wire_pose_candidate import _material_tangents, _station_frame

    row, _, _ = screen.load_measured_palm_boundary(PALM_RECEIPT, project=ROOT)
    _, nodes, _ = screen.measured_palm_boundary(row)
    side = "left"
    initial = screen.palm_anchor(row["rod"], nodes, side)
    position, xyzw = row["sdk_hand_geometry"]["poses"][f"{side}.base_link"]
    pose = np.r_[position, np.roll(xyzw, 1)]
    candidate, target = screen.transport_anchor(pose, initial, nodes, target=29, side=side)
    tangents = _material_tangents(nodes)
    original_basis = _station_frame(tangents, initial["node_id"])
    new_basis = _station_frame(tangents, target["node_id"])
    np.testing.assert_allclose(new_basis.T @ (target["pivot_world_m"] - nodes[29]),
                               original_basis.T @ (initial["pivot_world_m"] - nodes[31]))
    old_rotation = Rotation.from_quat(np.roll(pose[3:], -1))
    new_rotation = Rotation.from_quat(np.roll(candidate[3:], -1))
    np.testing.assert_allclose(new_rotation.inv().apply(target["normal_world"]),
                               old_rotation.inv().apply(initial["normal_world"]))
    assert target["node_id"] == 29
    assert initial["node_id"] == 31
    with pytest.raises(ValueError, match="assigned wire half"):
        screen.transport_anchor(pose, initial, nodes, target=19, side=side)


def test_support_adjacent_nodes_excluded_even_if_only_neighbour_is_loaded():
    row, scene, _ = screen.load_measured_palm_boundary(PALM_RECEIPT, project=ROOT)
    _, nodes, _ = screen.measured_palm_boundary(row)
    mask = screen.support_adjacent_nodes(row["rod"], nodes, scene)
    assert mask[0] and mask[1] and mask[39] and mask[40]
    assert not mask[28] and not mask[12]
