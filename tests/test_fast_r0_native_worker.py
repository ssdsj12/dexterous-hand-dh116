"""Independent bounded entry uses the byte-exact codec without new physics."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import verify_dh116_fast_r0_reset as entry
from scripts import run_dh116_fast_r0_work_packages as wrapper
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_deformable_r0_replay as codec
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import encode_owned as fast
ROOT=Path(__file__).resolve().parents[1]
RECIPE="configs/dh116_execution/fast_r0_reset_native_v1.json"

def preflight():
    with bindings((entry.previous.previous,"FILES",entry.FILES)):
        return entry.preflight(ROOT)

def cleanup(spec):
    with bindings((wrapper.base,"ENTRY",entry.ENTRY),(wrapper.base,"AUDIT_SOURCES",wrapper.AUDIT_SOURCES)):
        return wrapper.cleanup_case(spec)

def test_frozen_recipe_has_distinct_identity_and_same_budget():
    spec=json.loads((ROOT/RECIPE).read_bytes()); c=preflight()
    assert c["code_config_sha256"]==spec["validator"]["equals"]["code_config_sha256"]
    assert entry.WORKER_ENV!=entry.previous.WORKER_ENV
    assert cleanup(spec)=="explicit"
    assert set(entry.FILES)<=set(c["sources"]["local_sources"])
    assert c["profile_sha256"]==entry.previous.PROFILE_SHA
    assert "codec_scope.json" in spec["validator"]["artifacts"]
    assert c["codec_version"]=="dh116-owned-finite-array-codec-v1"

@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),
    ("gpu_index",2),("timeout_seconds",181)])
def test_budget_expansion_rejected(key,value):
    spec=json.loads((ROOT/RECIPE).read_bytes());spec["budget"][key]=value
    with pytest.raises(ValueError):cleanup(spec)

@pytest.mark.parametrize("fail",[False,True])
def test_only_codec_changes_and_restores_even_if_actual_owner_fails(tmp_path,monkeypatch,fail):
    original=codec.encode_owned
    args=SimpleNamespace(output=tmp_path)
    def owner(*passed,**kw):
        assert passed[0] is args
        assert codec.encode_owned is fast
        if fail:raise RuntimeError("owner failure")
        return 17
    monkeypatch.setattr(entry,"old_execute",owner)
    if fail:
        with pytest.raises(RuntimeError,match="owner failure"):entry.execute(args,ROOT,{})
        assert not (tmp_path/"codec_scope.json").exists()
    else:
        assert entry.execute(args,ROOT,{})==17
        r=json.loads((tmp_path/"codec_scope.json").read_text())
        assert r["binding_restored"] and r["new_physics_steps"]==0
    assert codec.encode_owned is original

def test_worker_requires_actual_parent_marker_before_app(tmp_path,monkeypatch):
    import sys
    monkeypatch.delenv(entry.WORKER_ENV,raising=False)
    monkeypatch.setattr(sys,"argv",[entry.ENTRY,"--cleanup","explicit","--headless",
        "--execute-native","--output",str(tmp_path/"native"),"--left-scheduler-receipt",str(entry.PARENT_RECEIPT)])
    with pytest.raises((RuntimeError,ValueError,SystemExit)):entry.main()
    assert not (tmp_path/"native").exists()

def test_codec_evidence_rejects_unproven_restoration(tmp_path):
    p=tmp_path/"codec_scope.json"
    p.write_text(json.dumps(dict(version=entry.CODEC_VERSION,binding_restored=False,new_physics_steps=0,ppo_updates=0)))
    with pytest.raises(ValueError):wrapper.check_codec_evidence(tmp_path)
