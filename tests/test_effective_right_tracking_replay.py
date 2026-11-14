"""Native evidence replay is CPU analysis, never a new contact qualification."""
from copy import deepcopy
import numpy as np
import pytest
from scripts import audit_dh116_effective_right_tracking as a

def test_named_joint_delta_preserves_original_order_and_rejects_missing():
    names=list(reversed(a.ACTIVE));values=np.arange(6,dtype=float)[None]
    record={"joint_names":names,"q":values,"qd":values*0,"backend_targets":{}}
    np.testing.assert_array_equal(a.active_values(record),np.arange(6,dtype=float)[::-1])
    with pytest.raises(ValueError,match="named"):a.active_values({**record,"joint_names":names[:-1]})
    with pytest.raises(ValueError,match="finite"):a.active_values({**record,"q":values*np.nan})

def test_contact_displacement_units_are_millimetres():
    v=a.displacement_mm(np.array([[.4301,0,.305]]),np.array([[.43,0,.305]]))
    np.testing.assert_allclose(v,[[.1,0,0]],atol=1e-10)
    with pytest.raises(ValueError,match="shape"):a.displacement_mm(np.zeros((2,3)),np.zeros((1,3)))

def test_actual_frozen_receipt_and_ledger_replay():
    packet=a.load_source()
    assert len(packet["exchanges"])==232
    assert packet["receipt"]["outer_exit_code"]==0
    summary=a.contact_summary(packet["exchanges"])
    assert summary["raw_pairs"]=={"right.finger14_link":1002,"right.finger23_link":1127}
    assert summary["accepted_by_original_region"]==0
    assert summary["both_right_region_windows"]==0
    assert summary["identity"]["env_id"]==0
    assert summary["physical_steps"]==list(range(3,235))
    assert summary["native_success_qualified"] is False

def test_terminal_clock_mismatch_is_not_silently_repaired():
    packet=a.load_source()
    rows=packet["exchanges"]
    broken=deepcopy(rows[-1]);broken["closure_geometry"]["raw"]["identity"]["physics_step"]-=1
    with pytest.raises(ValueError,match="identity"):
        a.contact_summary(rows[:-1]+[broken])

def test_actual_pose_and_rod_decomposition_matches_solver():
    packet=a.load_source();r=a.compare_endpoint(packet["exchanges"][-1])
    for n,p in r["pads"].items():
        assert p["command_nominal"]["ellipse_value"]<1
        assert p["measured_links_actual_rod"]["ellipse_value"]>1
        assert p["solver_point_prediction_error_mm"]<.001
        assert p["solver_region"]["accepted_by_original_patch"] is False
    assert r["new_physics_steps"]==r["ppo_updates"]==0
    assert r["teacher_admission"] is False
    assert r["rod_displacement_max_abs_mm"][0]>.1
    assert max(abs(v) for v in r["right_active_tracking_error_rad"])>.001

def test_wrong_receipt_hash_rejected_before_replay():
    with pytest.raises(ValueError,match="receipt"):
        a.load_source(expected_sha="0"*64)
