"""Runtime-floor qualification remains an independent static gate."""
from copy import deepcopy

import pytest

from scripts.validate_dh116_right_pinch_runtime_clearance import qualifies, runtime_candidate


def inputs():
    candidate = dict(candidate_id="right-targeted-092", side="right",
        base_pose_wxyz=[.4, .2, .4, 1., 0., 0., 0.],
        active_q_rad=[.4] * 6, strict_endpoint_passed=True, native_ready=False)
    path = dict(candidate_id=candidate["candidate_id"], geometry_preparation_passed=True,
        reset_geometry_passed=True,
        continuous_table_clearance_lower_bound_m={"source":{"finger13_link":.00212},
                                                    "cooked":{"finger14_link":.0027}})
    return candidate, path


def test_bounded_z_and_thumb_change_preserve_other_channels():
    original, _ = inputs()
    frozen = deepcopy(original)
    candidate = runtime_candidate(original)
    assert original == frozen
    assert candidate["base_pose_wxyz"][2] == pytest.approx(.400065)
    assert candidate["active_q_rad"][1] == pytest.approx(.4001)
    assert candidate["active_q_rad"][2:] == original["active_q_rad"][2:]
    assert candidate["native_ready"] is False


@pytest.mark.parametrize("gap,expected", [(.00209999,False),(.0021,True),(.002112,True)])
def test_static_1p9_pass_does_not_waive_actual_2p1_floor(gap, expected):
    candidate, path = inputs()
    path["continuous_table_clearance_lower_bound_m"]["source"]["finger13_link"] = gap
    assert qualifies(candidate, path) is expected
    candidate["strict_endpoint_passed"] = False
    assert not qualifies(candidate, path)


def test_path_for_another_candidate_is_rejected():
    candidate, path = inputs()
    path["candidate_id"] = "foreign"
    assert not qualifies(candidate, path)
