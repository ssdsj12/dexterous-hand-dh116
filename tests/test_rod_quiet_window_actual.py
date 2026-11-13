"""Actual Pinch quiet evidence stays historical and cannot admit Tripod or PPO."""
import hashlib,json
from pathlib import Path
import pytest

def test_actual_readonly_quiet_statistics_retain_all_windows_and_no_success_claim():
    p=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/rod-quiet-window-20261114-run01/result.json")
    if not p.exists():pytest.skip("private immutable native-ledger analysis unavailable")
    assert hashlib.sha256(p.read_bytes()).hexdigest()=="d283ee366cdb32ebce9967be73934e1fd3ea00baa3201ad604287aa8f3a572d2"
    r=json.loads(p.read_text())
    assert r["original_physical_samples"]==232
    assert r["new_sdk_steps"]==r["ppo_updates"]==0
    assert r["no_threshold_or_stage_change"] is True
    assert r["native_reset_qualified"] is r["teacher_admitted"] is r["tripod_qualified"] is r["power_qualified"] is False
    assert [s["samples"] for s in r["snapshots"]]==[8,16,24,32,48]
    for s in r["snapshots"]:
        assert s["table_loaded_solver_substeps"]==s["samples"]*12
        assert s["hand_loaded_solver_substeps"]==0
        assert s["max_node_speed_m_s"]==pytest.approx(3.832923802692624e-5)
        assert s["node_center_z_range_m"]==[.3049999177455902,.3049999177455902]
        assert s["stable_grasp"] is None
        assert s["hand_joint_speed"]["right"]["finger12"]["max_abs"]==pytest.approx(1.3649122365677613e-6)
