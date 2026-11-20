import numpy as np
import pytest

from scripts import run_dh116_pinch_bilateral_closure as closure


def test_closure_only_uses_selected_four_digits_and_preserves_joint_target_totals():
    low, high = closure.prelift.active_limits()
    q = np.r_[(low + high) / 2, np.zeros(6)]
    states = {side: (q, q.copy()) for side in ("left", "right")}
    goal = {"left": q[1:3] + [.016, .003], "right": q[1:3] + [.006, .010]}
    prefix = np.zeros((19, 24))
    out, targets = closure.compile_closure(prefix, states, goal)
    np.testing.assert_array_equal(out[:19], prefix)
    np.testing.assert_array_equal(out[19:, closure.UNCHANGED], 0.)
    assert np.max(np.abs(out[19:])) <= 1.
    np.testing.assert_allclose(out[19:].sum(0) * closure.prelift.WINDOW_SCALE,
        closure.prelift.global_channel_order(targets["left"] - q, targets["right"] - q))
    np.testing.assert_array_equal(out[-1], 0.)


def test_only_four_named_pads_are_eligible_for_contact():
    rows = [{"mesh_name": name} for name in closure.CONTACT_MESHES]
    row = {"guard_failure": None, "rod": {"rod_mesh_contact_pairs": {"pairs": rows}}}
    closure.guard_selected_contact(row)
    assert row["guard_failure"] is None
    rows.append({"mesh_name": "right.finger33_link"})
    closure.guard_selected_contact(row)
    assert row["guard_failure"] == "pinch_unselected_contact_pair"
    row["guard_failure"] = "prior_native_guard"
    closure.guard_selected_contact(row)
    assert row["guard_failure"] == "prior_native_guard"


def test_closure_rejects_a_wrong_side_goal():
    q = np.zeros(12)
    with pytest.raises(ValueError):
        closure.compile_closure(np.zeros((19, 24)), {"left": (q,q), "right": (q,q)},
                                {"left": q[1:3], "right": [10.,10.]})
