"""Single bounded measured-approach entry; CPU tests do not confer native credit."""
import importlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
ROOT=Path(__file__).resolve().parents[1]
def api():
    name="scripts.verify_dh116_approach_r0"
    assert importlib.util.find_spec(name) is not None,"independent measured approach native entry missing"
    return importlib.import_module(name)

@pytest.mark.parametrize("fail",[False,True])
def test_session_scoped_and_restored(tmp_path,monkeypatch,fail):
    e=api();old=e.operation.LiveSession
    def owner(*args,**kw):
        assert e.operation.LiveSession is e.MeasuredApproachSession
        if fail:raise RuntimeError("owner failed")
        return 11
    monkeypatch.setattr(e,"old_execute",owner)
    if fail:
        with pytest.raises(RuntimeError):e.execute(SimpleNamespace(output=tmp_path))
        assert not (tmp_path/"approach_scope.json").exists()
    else:
        assert e.execute(SimpleNamespace(output=tmp_path))==11
        assert json.loads((tmp_path/"approach_scope.json").read_text())==e.scope_marker()
    assert e.operation.LiveSession is old

def test_recipe_code_budget_and_dependency_scope():
    e=api()
    from scripts import run_dh116_approach_work_packages as w
    from scripts.run_dh116_work_packages import release_path
    spec=json.loads((ROOT/"configs/dh116_execution/approach_r0_native_v1.json").read_text())
    old=json.loads((ROOT/"configs/dh116_execution/sealed_r0_native_v1.json").read_text())
    p=e.preflight(ROOT)
    assert p["code_config_sha256"]==spec["validator"]["equals"]["code_config_sha256"]
    assert spec["budget"]==old["budget"] and spec["inputs"]==old["inputs"]
    assert e.WORKER_ENV!=e.previous.WORKER_ENV and e.SCOPE!=e.previous.SCOPE
    assert "approach_scope.json" in spec["validator"]["artifacts"]
    for name in spec["source_files"]:assert release_path(name)==name
    with bindings((w.base,"ENTRY",e.ENTRY),(w.base,"AUDIT_SOURCES",w.AUDIT_SOURCES)):
        assert w.cleanup_case(spec)=="explicit"

@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),("gpu_index",2),("timeout_seconds",181)])
def test_budget_cannot_expand(key,value):
    e=api()
    from scripts import run_dh116_approach_work_packages as w
    spec=json.loads((ROOT/"configs/dh116_execution/approach_r0_native_v1.json").read_text())
    spec["budget"][key]=value
    with bindings((w.base,"ENTRY",e.ENTRY),(w.base,"AUDIT_SOURCES",w.AUDIT_SOURCES)):
        with pytest.raises(ValueError):w.cleanup_case(spec)

def test_full_cpu_owner_measures_actual_reset_and_all_rows(monkeypatch,tmp_path,prepared,geometry):
    e=api()
    import test_effective_right_pinch_owner as fixture
    from sealed_owner_cpu_fixture import run_actual_episode
    from scripts import prepare_dh116_effective_right_native as closure
    from scripts import prepare_dh116_live_r0_native as live
    from scripts import verify_dh116_origin_r0 as origin
    from scripts import verify_dh116_sealed_r0 as seal
    from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope
    from scripts.validate_dh116_approach_r0 import audit
    monkeypatch.setattr(fixture,"run_actual_episode",run_actual_episode)
    monkeypatch.setattr(closure,"run_existing_app",live.run_existing_app)
    with codec_scope(),seal.clock_projection_scope(),bindings(
        (seal.parameter,"inspect_interval",seal.seal_pending),
        (origin.runtime,"read_pair_capture",origin.read_complete_pair_capture),
        (live,"LiveR0Observer",origin.OriginLiveR0Observer),
        (live,"LiveSession",e.MeasuredApproachSession)):
        out,dirs,params=fixture.run_owner(monkeypatch,tmp_path,prepared,geometry)
    assert dirs==params==list(range(2,235))
    result=audit(out,ROOT)
    assert result["physical_rows"]==232 and result["control_windows"]==29
    assert result["all_live_rows_recomputed"] and not result["teacher_admission"]
    assert result["new_physics_steps"]==0
    # Non-approach or approach tampering must both fail full equality checks.
    ledger=out/"live_r0.jsonl";original=ledger.read_text()
    rows=[json.loads(line) for line in original.splitlines()]
    target=next(r for r in rows if r["operation"]=="physical")
    target["payload"]["r0_row"]["r0"]["training_reward"]=1.
    ledger.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    with pytest.raises(ValueError):audit(out,ROOT)
    ledger.write_text(original)

from test_dh116_dual_pinch_builder import prepared,geometry

@pytest.mark.parametrize("key,value",[("valid_approach_samples",0),("valid_approach_samples",231),("reset_baseline_available",False)])
def test_native_scope_refuses_incomplete_approach_coverage(tmp_path,monkeypatch,key,value):
    e=api()
    from scripts import run_dh116_approach_work_packages as w
    (tmp_path/"approach_scope.json").write_text(json.dumps(e.scope_marker()))
    monkeypatch.setattr(w.previous,"audit_outcome",lambda *args:dict(lifecycle_qualified=True))
    result=dict(physical_rows=232,control_windows=29,valid_approach_samples=232,reset_baseline_available=True)
    result[key]=value
    monkeypatch.setattr(w,"audit",lambda *args:result)
    with pytest.raises(ValueError,match="approach coverage"):w.audit_outcome(tmp_path,None,"explicit")
