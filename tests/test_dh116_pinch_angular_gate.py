import numpy as np
import pytest

from scripts import run_dh116_pinch_angular_gate as gate


def test_only_right_rotation_changes_at_last_window():
    prior = np.zeros((9, 24))
    prior[7, :6] = .1
    prior[7, 18:21] = .02
    prior[7, 6:12] = .2
    prior[7, 15:18] = .1
    prior[7, 21:24] = .04
    angular = np.zeros(24)
    angular[21:24] = [-.1, -.3, 0.]
    actions = gate.angular_schedule(prior, angular)
    assert actions.shape == (8, 24)
    np.testing.assert_array_equal(actions[:7], prior[:7])
    np.testing.assert_array_equal(actions[7, :6], prior[7, :6])
    np.testing.assert_array_equal(actions[7, 12:15], prior[7, 12:15])
    np.testing.assert_array_equal(actions[7, 18:21], prior[7, 18:21])
    np.testing.assert_array_equal(actions[7, 6:12], 0.)
    np.testing.assert_array_equal(actions[7, 15:18], 0.)
    np.testing.assert_array_equal(actions[7, 21:], angular[21:])
    angular[7] = .01
    with pytest.raises(ValueError, match="rotation only"):
        gate.angular_schedule(prior, angular)


def test_gate_uses_current_snapshot_and_reports_first_reserve_escape(monkeypatch):
    monkeypatch.setattr(gate.late, "prefix_measurements", lambda row, **kw: row["measurement"])
    monkeypatch.setattr(gate.late, "compare_prefix", lambda measured, reference: dict(passed=measured == reference))
    monkeypatch.setattr(gate.prelift, "_same_clock", lambda row: ["run", row["physics_step"]])
    monkeypatch.setattr(gate.prelift, "joint_and_wrist", lambda row, side: (np.zeros(12), np.zeros(12)))
    monkeypatch.setattr(gate.wrist, "phase_envelope", lambda *a, **kw: {"angle_first": (np.zeros(4), np.ones(4))})
    touched = []
    def prove(row, actual, issued, target, envelope, *, nodes, **kwargs):
        touched.append((row["physics_step"], nodes.copy(), target.copy()))
        return {"angle_first": {"proved": True}}, None
    g = gate.AngularGate({"prefix": "same", "nominal": [1.] * 12, "rotation": [.1, .2, .3]}, prove)
    row = {"physics_step": 58, "guard_failure": None, "measurement": "same",
           "rod": {"node_positions": np.zeros((1, 41, 3))}}
    g.observe(row)
    assert row["guard_failure"] is None
    assert row["angular_gate"]["passed"]
    assert touched[0][0] == 58
    np.testing.assert_array_equal(touched[0][1], 0.)
    np.testing.assert_allclose(touched[0][2][9:], np.array([.1, .2, .3]) * gate.prelift.WINDOW_SCALE[21:])
    row["rod"]["node_positions"][0, 0, 0] = 2 * gate.prelift.NODE_RESERVE_M
    row.update(physics_step=59)
    g.observe(row)
    assert row["guard_failure"] == "pinch_angular_rod_reserve_exceeded"
    assert row["angular_rod_motion_m"] == pytest.approx(2 * gate.prelift.NODE_RESERVE_M)
    assert len(touched) == 1


def test_failed_prefix_never_evaluates_geometry(monkeypatch):
    monkeypatch.setattr(gate.late, "prefix_measurements", lambda *_a, **_kw: {})
    monkeypatch.setattr(gate.late, "compare_prefix", lambda *_: {"passed": False})
    g = gate.AngularGate({"prefix": {}}, lambda *a, **kw: pytest.fail("geometry ran after rejection"))
    row = dict(physics_step=58, guard_failure=None)
    g.observe(row)
    assert row["guard_failure"] == "pinch_angular_prefix_mismatch"
    assert row["angular_gate"]["passed"] is False
