"""Frozen unchanged physics budget; new live observation code fingerprint."""
import json
from pathlib import Path
import pytest
from scripts import verify_dh116_live_r0_reset as entry
from scripts import run_dh116_live_r0_work_packages as wrapper
from scripts import prepare_dh116_live_r0_native as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ROOT=Path(__file__).resolve().parents[1]
RECIPE="configs/dh116_execution/live_r0_reset_native_v1.json"

def preflight():
    with bindings((entry.previous,"FILES",entry.FILES)):
        return entry.preflight(ROOT)

def cleanup(spec):
    with bindings((wrapper.base,"ENTRY",entry.ENTRY),(wrapper.base,"AUDIT_SOURCES",wrapper.AUDIT_SOURCES)):
        return wrapper.cleanup_case(spec)

def test_source_is_frozen_and_original_physics_profile_unchanged():
    c=preflight()
    spec=json.loads((ROOT/RECIPE).read_bytes())
    assert set(entry.FILES)<=set(c["sources"]["local_sources"])
    assert spec["validator"]["equals"]["code_config_sha256"]==c["code_config_sha256"]
    assert c["profile_sha256"]==entry.previous.PROFILE_SHA
    assert c["live_observer_version"]==operation.VERSION
    assert spec["entrypoint"]==entry.ENTRY
    assert cleanup(spec)=="explicit"
    assert spec["validator"]["equals"]["version"]==entry.VERSION
    assert operation.SCOPE=="actual_coupled_reset_then_live_original_R0_observer_234_steps_only"

@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),
    ("gpu_index",2),("timeout_seconds",181)])
def test_exact_budget_no_automatic_expansion(key,value):
    spec=json.loads((ROOT/RECIPE).read_bytes())
    spec["budget"][key]=value
    with pytest.raises(ValueError):cleanup(spec)

def test_worker_rejects_missing_per_attempt_marker_before_app(tmp_path,monkeypatch):
    import sys
    monkeypatch.delenv(entry.WORKER_ENV,raising=False)
    monkeypatch.setattr(sys,"argv",[entry.ENTRY,"--cleanup","explicit","--headless",
        "--execute-native","--output",str(tmp_path/"native"),"--left-scheduler-receipt",str(entry.PARENT_RECEIPT)])
    with pytest.raises((RuntimeError,ValueError,SystemExit)):entry.main()
    assert not (tmp_path/"native").exists()
