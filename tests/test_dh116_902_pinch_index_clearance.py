"""CPU regressions for measured left-index staged Pinch transitions."""
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import certify_dh116_902_pinch_index_clearance as path


def test_index_opening_carries_the_same_offset_through_aim_only():
    start = {"left": {"pose_wxyz": [0., 1., 2., 1., 0., 0., 0.], "active_q_rad": np.full(6, .5)},
             "right": {"pose_wxyz": [0., -1., 2., 1., 0., 0., 0.], "active_q_rad": np.full(6, .6)}}
    finish = {"left": {"pose_wxyz": [.1, 1., 2., 1., 0., 0., 0.], "active_q_rad": np.full(6, .4)},
              "right": start["right"]}
    stages = path.index_stages(start, finish, side="left", offset_rad=-.05)
    assert [name for name, _ in stages] == ["withdraw_index", "aim_with_index_reserve", "seat_at_fixed_wrist"]
    assert stages[0][1]["left"]["active_q_rad"][2] == pytest.approx(.45)
    assert stages[1][1]["left"]["active_q_rad"][2] == pytest.approx(.35)
    np.testing.assert_array_equal(stages[0][1]["left"]["pose_wxyz"], start["left"]["pose_wxyz"])
    np.testing.assert_array_equal(stages[2][1]["left"]["pose_wxyz"], stages[1][1]["left"]["pose_wxyz"])
    np.testing.assert_array_equal(stages[2][1]["left"]["active_q_rad"], finish["left"]["active_q_rad"])
    assert all(stage["right"] is start["right"] for _, stage in stages)


def test_paired_index_reserves_do_not_move_the_other_digits_or_terminal_wrist():
    start = {side: dict(pose_wxyz=np.ones(7), active_q_rad=np.full(6, .5)) for side in ("left", "right")}
    finish = {side: dict(pose_wxyz=np.zeros(7), active_q_rad=np.full(6, .4)) for side in ("left", "right")}
    stages = path.paired_index_stages(start, finish, offsets={"left": -.03, "right": -.06})
    for side, target in (("left", .37), ("right", .34)):
        np.testing.assert_allclose(stages[1][1][side]["active_q_rad"], [.4, .4, target, .4, .4, .4])
        np.testing.assert_array_equal(stages[-1][1][side]["pose_wxyz"], stages[-2][1][side]["pose_wxyz"])
        np.testing.assert_allclose(stages[-1][1][side]["active_q_rad"], np.full(6, .4))


def test_blocker_probe_replays_actual_r5_float32_command_and_finds_a_reserve():
    names = ("DH116_PINCH_ALIGNMENT_RECEIPT", "DH116_PINCH_RELIEF_ROUTE_LEFT",
             "DH116_PINCH_RELIEF_ROUTE_RIGHT", "DH116_PINCH_R6_BLOCKER")
    sources = [os.environ.get(n) for n in names]
    if not all(sources):
        pytest.skip("sealed R5 route and endpoints are external")
    proof = path.screen(*map(Path, sources), run_routes=False)
    point = proof["blocker_probe"]
    assert point["issued_substep"] == 210
    assert point["pad_hull"] == "source_colliders:finger23_link"
    assert point["original_refinement"]["subset_upper_m"] < -.00015
    assert any(t["index_offset_rad"] != 0 and t["full_geometry_passed"] for t in point["alternatives"])
    assert proof["joint_endpoint_passed"]
