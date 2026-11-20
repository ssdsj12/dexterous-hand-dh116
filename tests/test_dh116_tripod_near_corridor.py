import numpy as np
import pytest

from scripts import run_dh116_tripod_near_corridor as near


def test_near_evidence_requires_all_completed_steps_in_order():
    rows = [{"sdk_step": step, "min_table": {"table_gap_m": .0024},
             "nearest_non_target": {"rod_gap_lower_m": .00006},
             "native_dof_tracking_delta": [0.] * 12} for step in range(3, 499)]
    assert near.validate_stress_rows(rows) == pytest.approx((.0024, .00006))
    with pytest.raises(ValueError, match="per completed SDK step"):
        near.validate_stress_rows(rows[:120] + rows[121:])
    rows[-1]["min_table"]["table_gap_m"] = .002
    with pytest.raises(ValueError, match="clearances"):
        near.validate_stress_rows(rows)


def test_live_guard_checks_non_target_clearance_and_all_target_pads():
    nodes = np.array([[0., y, .305] for y in np.linspace(-.25, .25, 41)])
    clear = np.array([[.020, 0., .312], [.020, .01, .312],
                      [.020, .007, .315], [.023, 0., .312]])
    hit = clear - np.array([.019, 0., .007])
    geometry = dict(source={name: clear for name in near.PADS},
                    cooked={name: clear for name in near.PADS})
    geometry["source"]["finger13_link"] = clear
    geometry["cooked"]["finger13_link"] = hit
    identity_poses = {name: np.eye(4) for name in geometry["source"]}
    pack = near.measure_clearance(geometry, identity_poses, nodes)
    assert pack["rod_lower_m"]["cooked:finger13_link"] < 0
    assert pack["table_m"]["cooked:finger13_link"] >= .0021
    assert set(pack["rod_lower_m"]) == {
        f"{kind}:{name}" for kind, shapes in geometry.items() for name in shapes
    }
    assert not pack["passed"]


def test_guard_persists_failed_tick_before_raising(monkeypatch):
    from scripts import verify_dh116_tripod_right_tracking as tracking

    recorded = []
    exchange = {
        "guard_failure": None,
        "tripod_near_geometry": {"identity": {"physics_step": 3}, "passed": False},
        "action_readback": {"post": {"hands": {side: {
            "joint_names": list(range(17)), "mounted": True,
            "backend_targets": np.zeros((1, 17)), "native_anchor_pose_w": np.zeros((1, 7)),
        } for side in ("left", "right")}}},
    }
    monkeypatch.setattr(tracking, "extract", lambda row: dict.fromkeys(tracking.IDENTITY, "case"))
    monkeypatch.setattr(tracking, "validate_tracking_sample", lambda *args, **kwargs: None)
    recorder = tracking._TrackingRecorder(np.zeros((62, 24)), recorded.append, None, near.verify_boundary)
    payload = {"kind": "physical", "exchange": exchange}
    with pytest.raises(tracking.TrackingStopped, match="did not certify clearance"):
        recorder(payload)
    assert recorded == [payload]
    assert exchange["guard_failure"] is None
    with pytest.raises(RuntimeError, match="stopped"):
        recorder(payload)
    assert recorded == [payload]
