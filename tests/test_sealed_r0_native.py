"""The real Python owner is exercised with CPU scene doubles, not Isaac."""
import importlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_dh116_dual_pinch_builder import prepared,geometry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
ROOT=Path(__file__).resolve().parents[1]
def api():
    name="scripts.verify_dh116_sealed_r0"
    assert importlib.util.find_spec(name) is not None,"isolated sealed owner missing"
    return importlib.import_module(name)

@pytest.mark.parametrize("fail",[False,True])
def test_owner_defers_only_full_replay_and_restores_on_failure(tmp_path,monkeypatch,fail):
    e=api();old=e.parameter.inspect_interval
    def owner(*args,**kwargs):
        assert e.parameter.inspect_interval is e.seal_pending
        if fail:raise RuntimeError("owner failed")
        return 7
    monkeypatch.setattr(e,"old_execute",owner)
    if fail:
        with pytest.raises(RuntimeError):e.execute(SimpleNamespace(output=tmp_path))
        assert not (tmp_path/"sealed_audit_scope.json").exists()
    else:
        assert e.execute(SimpleNamespace(output=tmp_path))==7
        assert json.loads((tmp_path/"sealed_audit_scope.json").read_text())==e.scope_marker()
    assert e.parameter.inspect_interval is old

def test_frozen_new_recipe_protection_and_scope():
    e=api()
    from scripts import run_dh116_sealed_work_packages as w
    from scripts.run_dh116_work_packages import release_path
    s=json.loads((ROOT/"configs/dh116_execution/sealed_r0_native_v1.json").read_text())
    old=json.loads((ROOT/"configs/dh116_execution/origin_r0_native_v1.json").read_text())
    with bindings((e.previous.previous.previous.previous,"FILES",e.FILES)):p=e.preflight(ROOT)
    assert p["code_config_sha256"]==s["validator"]["equals"]["code_config_sha256"]
    assert s["budget"]==old["budget"] and s["inputs"]==old["inputs"]
    assert e.SCOPE!=e.previous.SCOPE and e.WORKER_ENV!=e.previous.WORKER_ENV
    for name in s["source_files"]:assert release_path(name)==name
    with bindings((w.base,"ENTRY",e.ENTRY),(w.base,"AUDIT_SOURCES",w.AUDIT_SOURCES)):
        assert w.cleanup_case(s)=="explicit"

@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),("gpu_index",2),("timeout_seconds",181)])
def test_budget_not_expanded(key,value):
    e=api()
    from scripts import run_dh116_sealed_work_packages as w
    s=json.loads((ROOT/"configs/dh116_execution/sealed_r0_native_v1.json").read_text());s["budget"][key]=value
    with bindings((w.base,"ENTRY",e.ENTRY),(w.base,"AUDIT_SOURCES",w.AUDIT_SOURCES)):
        with pytest.raises(ValueError):w.cleanup_case(s)

def test_real_owner_retains_all_atomic_rows_and_pending_seal(monkeypatch,tmp_path,prepared,geometry):
    e=api()
    import test_effective_right_pinch_owner as old_fixture
    from sealed_owner_cpu_fixture import run_actual_episode
    from scripts import prepare_dh116_effective_right_native as closure
    from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope
    from scripts.validate_dh116_origin_r0 import audit
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_rod_runtime as runtime
    from scripts import prepare_dh116_live_r0_native as live
    from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import read_complete_pair_capture
    from go2_pvcnn.control.dh116.tabletop_r0_measurement_origin import OriginLiveR0Observer
    monkeypatch.setattr(old_fixture,"run_actual_episode",run_actual_episode)
    monkeypatch.setattr(closure,"run_existing_app",live.run_existing_app)
    seals=[]
    def record_seal(path,**kw):
        result=e.seal_pending(path,**kw);seals.append((Path(path),result));return result
    with codec_scope(),e.clock_projection_scope(),bindings(
            (e.parameter,"inspect_interval",record_seal),
            (runtime,"read_pair_capture",read_complete_pair_capture),
            (live,"LiveR0Observer",OriginLiveR0Observer)):
        out,dirs,params=old_fixture.run_owner(monkeypatch,tmp_path,prepared,geometry)
    assert dirs==params==list(range(2,235))
    assert len(seals)==1
    path,seal=seals[0]
    from scripts.dh116_deferred_physical_audit import validate_seal
    assert validate_seal(path,seal)["full_audit_performed"] is False
    entries=[json.loads(line) for line in path.read_text().splitlines()]
    assert sum(row.get("operation")=="physical" for row in entries)==232
    assert sum(row.get("operation")=="end" for row in entries)==29
    assert (out/"app_close_returned.json").exists()
    result=audit(path.parent)
    assert result["physical_rows"]==232 and result["origin_count"]==4
    assert result["per_frame_provenance_recomputed"] and not result["teacher_admission"]
