"""Independent bounded entry and actual Python owner in CPU doubles."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_dh116_dual_pinch_builder import prepared,geometry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
ROOT=Path(__file__).resolve().parents[1]
def api():
    name="scripts.verify_dh116_origin_r0"
    assert importlib.util.find_spec(name) is not None,"distinct origin native entry missing"
    return importlib.import_module(name)

def test_recipe_identity_scope_and_asset_protection():
    e=api()
    from scripts import run_dh116_origin_work_packages as w
    from scripts.run_dh116_work_packages import release_path
    spec=json.loads((ROOT/"configs/dh116_execution/origin_r0_native_v1.json").read_text())
    old=json.loads((ROOT/"configs/dh116_execution/complete_pair_r0_native_v1.json").read_text())
    with bindings((e.previous.previous.previous,"FILES",e.FILES)):p=e.preflight(ROOT)
    assert p["code_config_sha256"]==spec["validator"]["equals"]["code_config_sha256"]
    assert spec["budget"]==old["budget"] and spec["inputs"]==old["inputs"]
    assert e.WORKER_ENV!=e.previous.WORKER_ENV
    assert e.SCOPE!=e.previous.SCOPE
    for name in spec["source_files"]:assert release_path(name)==name
    with bindings((w.base,"ENTRY",e.ENTRY),(w.base,"AUDIT_SOURCES",w.AUDIT_SOURCES)):
        assert w.cleanup_case(spec)=="explicit"

@pytest.mark.parametrize("fail",[False,True])
def test_original_bindings_restored_including_exception(tmp_path,monkeypatch,fail):
    e=api();reader=e.runtime.read_pair_capture;observer=e.operation.LiveR0Observer
    def owner(*args,**kw):
        assert e.runtime.read_pair_capture is e.read_complete_pair_capture
        assert e.operation.LiveR0Observer is e.OriginLiveR0Observer
        if fail:raise RuntimeError("owner failed")
        return 19
    monkeypatch.setattr(e,"old_execute",owner)
    args=SimpleNamespace(output=tmp_path)
    if fail:
        with pytest.raises(RuntimeError,match="owner failed"):e.execute(args,ROOT,{})
        assert not (tmp_path/"origin_scope.json").exists()
    else:
        assert e.execute(args,ROOT,{})==19
        assert json.loads((tmp_path/"origin_scope.json").read_text())==e.scope_marker()
    assert e.runtime.read_pair_capture is reader and e.operation.LiveR0Observer is observer

@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),("gpu_index",2),("timeout_seconds",181)])
def test_no_budget_expansion(key,value):
    e=api()
    from scripts import run_dh116_origin_work_packages as w
    s=json.loads((ROOT/"configs/dh116_execution/origin_r0_native_v1.json").read_text())
    s["budget"][key]=value
    with bindings((w.base,"ENTRY",e.ENTRY),(w.base,"AUDIT_SOURCES",w.AUDIT_SOURCES)):
        with pytest.raises(ValueError):w.cleanup_case(s)

def test_real_python_owner_records_all_samples_and_independent_audit(monkeypatch,tmp_path,prepared,geometry):
    e=api()
    from test_live_r0_native_session import test_real_python_owner_initializes_before_first_action_then_settles as original
    from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope
    from scripts.validate_dh116_origin_r0 import audit
    with codec_scope(),bindings((e.runtime,"read_pair_capture",e.read_complete_pair_capture),
            (e.operation,"LiveR0Observer",e.OriginLiveR0Observer)):
        original(monkeypatch,tmp_path,prepared,geometry)
    paths=list(tmp_path.rglob("live_r0.jsonl"));assert len(paths)==1
    result=audit(paths[0].parent)
    assert result["physical_rows"]==232 and result["origin_count"]==4
    assert result["per_frame_provenance_recomputed"] and not result["teacher_admission"]
