import numpy as np
import pytest


def test_relief_uses_only_transverse_right_wrist_and_preserves_index_gap_to_first_order():
    from scripts.audit_dh116_tripod_transverse_approach import relief_action

    away = np.array([1., .1, .05])
    tangent = np.array([0., 1., 0.])
    index_gradient = np.array([.9, .3, .5])
    action = relief_action(away, tangent, index_gradient, amplitude=.75)
    np.testing.assert_array_equal(action[np.r_[:15, 18:24]], 0.)
    direction = action[15:18]
    assert direction @ away > 0.
    assert direction @ tangent == pytest.approx(0., abs=1e-12)
    assert direction @ index_gradient == pytest.approx(0., abs=1e-12)
    assert np.linalg.norm(direction) == pytest.approx(.75)


def test_relief_rejects_a_thumb_direction_conflicting_with_index_preservation():
    from scripts.audit_dh116_tripod_transverse_approach import relief_action

    assert relief_action([0., 1., 0.], [0., 1., 0.], [1., 0., 0.], amplitude=.75) is None


def test_two_windows_share_the_compiled_target_endpoint_not_a_future_measurement():
    from scripts.audit_dh116_tripod_transverse_approach import compile_two_windows

    root = np.array([[.4, .1, .44, 1., 0., 0., 0.],
                     [.4, -.1, .44, 1., 0., 0., 0.]])
    q = np.tile([.35, .48, .71, .76, .70, .95], 2)
    angles = np.zeros((2, 3))
    state = (root, angles, root.copy(), {}, q.copy(), q.copy())
    inputs = np.zeros((2, 24))
    inputs[0, 15] = .5
    inputs[1, [7, 17, 22]] = [.05, .2, -.09]
    row = {"sdk_hand_geometry": {"table_clearance_m": np.full((1, 2), .042)}}
    pair = compile_two_windows(inputs, row=row, seed=root, state=state)
    assert pair is not None
    first, second = pair
    assert first["target_poses_wxyz"][1][0] == pytest.approx(.4 + .5*.01/60.)
    assert second["target_poses_wxyz"][1][0] == pytest.approx(first["target_poses_wxyz"][1][0])
    assert second["target_poses_wxyz"][1][2] == pytest.approx(.44 + .2*.01/60.)
    assert second["finger_targets_rad"][7] == pytest.approx(q[7] + .05*.3/60.)



@pytest.mark.parametrize("minimum_rod_margin, expected", [(1e-6, False), (1e-4, True)])
def test_thumb_endpoint_gain_requires_full_route_headroom_over_matched_control(minimum_rod_margin, expected):
    from scripts.audit_dh116_tripod_transverse_approach import _counterfactual

    def endpoints(thumb, index):
        result = {}
        for mode in ("archived_offset_held", "absolute_backend"):
            result[f"{mode}_window_1_start"] = {"index_same_node_sdf_gap_m": .001}
            result[f"{mode}_window_2_end"] = dict(thumb_hull_lower_gap_m=thumb, thumb_hull_upper_gap_m=thumb,
                                               index_same_node_sdf_gap_m=index)
        return result

    reference = dict(passed=True, minimum_rod_margin_m=1e-5, endpoints=endpoints(.0001, .0009))
    candidate = dict(passed=True, minimum_rod_margin_m=minimum_rod_margin, endpoints=endpoints(.0002, .0009))
    admitted, comparisons = _counterfactual(candidate, reference, reference)
    assert all(row["thumb_gain_lower_m"] > 0. and row["forward_index_reduction_m"] > 0.
               for row in comparisons.values())
    assert admitted is expected



def test_relief_can_trade_hold_clearance_for_endpoint_thumb_headroom():
    from scripts.audit_dh116_tripod_transverse_approach import _counterfactual

    base = {}
    moved = {}
    for mode in ("archived_offset_held", "absolute_backend"):
        k = f"{mode}_window_1_start"
        base[k] = moved[k] = {"index_same_node_sdf_gap_m": .001}
        k = f"{mode}_window_2_end"
        base[k] = dict(thumb_hull_lower_gap_m=.0001, thumb_hull_upper_gap_m=.0001,
                       index_same_node_sdf_gap_m=.0009)
        moved[k] = dict(base[k], thumb_hull_lower_gap_m=.00015, thumb_hull_upper_gap_m=.00015)
    hold_forward = dict(passed=True, minimum_rod_margin_m=33e-6, endpoints=base)
    authored = dict(passed=False, minimum_rod_margin_m=-28e-6, endpoints=base)
    candidate = dict(passed=True, minimum_rod_margin_m=28e-6, endpoints=moved)
    admitted, _ = _counterfactual(candidate, hold_forward, authored)
    assert admitted



def test_two_window_rod_reserve_uses_larger_of_actual_history_and_immediate_travel():
    from scripts.audit_dh116_tripod_transverse_approach import prior_rod_reserve

    nodes = np.zeros((17, 41, 3))
    nodes[0, 8, 0] = .0002
    nodes[-2, 3, 0] = .000005
    reserve = prior_rod_reserve(nodes)
    assert reserve["measured_horizon_m"] == pytest.approx(.0002)
    assert reserve["last_step_scaled_m"] == pytest.approx(.00008)
    assert reserve["allowance_m"] == pytest.approx(.0002)
    nodes[-2, 3, 0] = .00002
    assert prior_rod_reserve(nodes)["allowance_m"] == pytest.approx(.00032)
