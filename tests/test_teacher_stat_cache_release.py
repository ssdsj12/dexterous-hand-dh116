"""CPU source-qualified Kit cache release and legacy boundary regression."""
import ast
import copy
import gc
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest

ROOT=Path(__file__).resolve().parents[1]
SDK=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/omni/kernel/py/omni/ext/_impl/stat_cache.py")


def api():
    name="go2_pvcnn.control.dh116.tabletop_stat_cache_release"
    assert importlib.util.find_spec(name) is not None, "source-qualified cache-release adapter missing"
    return importlib.import_module(name)


def entry():
    name="scripts.diagnose_dh116_teacher_stat_cache_release"
    assert importlib.util.find_spec(name) is not None, "independent cache-release entry missing"
    return importlib.import_module(name)


@pytest.fixture
def sdk(monkeypatch,tmp_path):
    assert SDK.is_file()
    original=importlib._bootstrap_external._path_stat
    m=ModuleType("omni.ext._impl.stat_cache");m.__file__=str(SDK)
    from functools import lru_cache
    settings_path=SDK.with_name("ext_settings.py")
    settings=ModuleType("omni.ext._impl.ext_settings")
    settings.__file__=str(settings_path)
    settings.lru_cache=lru_cache
    settings.carb=SimpleNamespace(settings=SimpleNamespace(
        get_settings=lambda:SimpleNamespace(get=lambda path:None)))
    settings._settings_iface=None
    settings_tree=ast.parse(settings_path.read_text())
    settings_tree.body=[n for n in settings_tree.body if isinstance(n,ast.FunctionDef)
        and n.name in ("_get_setting","_is_reentrant_test_mode","_is_path_stat_cache_enabled")]
    exec(compile(settings_tree,str(settings_path),"exec"),vars(settings))
    m.__dict__["_is_path_stat_cache_enabled"]=settings._is_path_stat_cache_enabled
    tree=ast.parse(SDK.read_text())
    tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom)]
    exec(compile(tree,str(SDK),"exec"),m.__dict__)
    app=SimpleNamespace(number=1,reads=0,updates=0,reset_reads=0)
    def update_number():
        app.reads+=1
        if sys._getframe(1).f_code is m.refresh_stat_cache.__code__:app.reset_reads+=1
        return app.number
    app.get_update_number=update_number
    def forbidden_update():
        app.updates+=1;raise AssertionError("CPU test App update forbidden")
    app.update=forbidden_update
    omni=ModuleType("omni");omni.__path__=[]
    kit=ModuleType("omni.kit");kit.__path__=[]
    app_module=ModuleType("omni.kit.app");app_module.get_app=lambda:app
    omni.kit=kit;kit.app=app_module
    for name,value in {"omni":omni,"omni.kit":kit,"omni.kit.app":app_module,
                       "omni.ext._impl.stat_cache":m,"omni.ext._impl.ext_settings":settings}.items():
        monkeypatch.setitem(sys.modules,name,value)
    package=tmp_path/"dh116_cache_test_package";package.mkdir()
    (package/"__init__.py").write_text("# CPU package fixture\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    m.reset_stat_cache()
    def clock():
        return {"sdk_step":2,"sdk_time_s":2/480,"app_update_number":app.number}
    try:yield m,app,clock,original
    finally:
        importlib._bootstrap_external._path_stat=original
        m._stat_cache.clear()


def retained_scene():
    class Scene:
        def __init__(self):
            importlib.util.find_spec("dh116_cache_test_package")
    def create_runtime():
        scene=Scene()
        return weakref.ref(scene)
    ref=create_runtime();gc.collect()
    assert ref() is not None, "actual SDK cache did not reproduce frame retention"
    return ref


def test_exact_sdk_cache_release_unblocks_retained_scene(sdk):
    m,app,clock,_=sdk;reference=retained_scene()
    a=api();contract=a.stat_cache_source_contract();reads=app.reads
    receipt=a.release_owned_stat_cache(m,contract=contract,clock_reader=clock)
    assert receipt["reset_api_calls"]==1 and receipt["reset_api_returned"] is True
    assert receipt["clock_before"]==receipt["clock_after"]==clock()
    assert receipt["before"]["cached_exceptions"]>0 and receipt["after"]["cache_entries_total"]==0
    assert receipt["native_release_verified"] is False and app.updates==0 and app.reads==reads+1
    gc.collect();assert reference() is None


def test_snapshot_is_scalar_and_does_not_keep_exceptions_or_frames(sdk):
    m,app,clock,_=sdk;reference=retained_scene();a=api()
    snapshot=a.snapshot_stat_cache(m,contract=a.stat_cache_source_contract())
    assert snapshot["cached_exceptions"]>0 and snapshot["exception_records"]
    assert json.loads(json.dumps(snapshot))==snapshot
    m.reset_stat_cache();gc.collect()
    assert reference() is None


def test_changed_source_contract_refuses_before_reset(sdk):
    m,app,clock,_=sdk;reference=retained_scene();a=api()
    bad=copy.deepcopy(a.stat_cache_source_contract());bad["sources"]["cache"]["sha256"]="0"*64
    before=len(m._stat_cache);reads=app.reads
    with pytest.raises(ValueError,match="source|contract"):
        a.release_owned_stat_cache(m,contract=bad,clock_reader=clock)
    assert len(m._stat_cache)==before and app.reads==reads


def test_wrong_loaded_module_file_refuses_before_reset(sdk):
    m,app,clock,_=sdk;retained_scene();a=api()
    m.__file__="/not/the/pinned/source.py"
    with pytest.raises(ValueError,match="source|module"):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=clock)


def test_inactive_cache_hook_is_not_silently_enabled(sdk):
    m,app,clock,original=sdk;reference=retained_scene();a=api()
    importlib._bootstrap_external._path_stat=original
    reads=app.reads
    with pytest.raises(ValueError,match="active"):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=clock)
    assert importlib._bootstrap_external._path_stat is original and app.reads==reads


@pytest.mark.parametrize("step",[3,False])
def test_wrong_sdk_clock_refuses_before_reset(sdk,step):
    m,app,clock,_=sdk;a=api();before=len(m._stat_cache);reads=app.reads
    def wrong():
        row=clock();row["sdk_step"]=step;return row
    with pytest.raises(ValueError,match="clock"):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=wrong)
    assert len(m._stat_cache)==before and app.reads==reads


def test_clock_change_after_api_is_failure(sdk):
    m,app,clock,_=sdk;retained_scene();a=api();calls=[]
    def changing():
        row=clock();row["app_update_number"]+=len(calls);calls.append(1);return row
    with pytest.raises(ValueError,match="clock"):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=changing)
    assert m._stat_cache=={} and len(calls)==2


def test_observer_io_failure_does_not_skip_source_qualified_release(sdk):
    m,app,clock,_=sdk;reference=retained_scene();a=api();reads=app.reads
    def fail(*args):raise OSError("CPU evidence IO failure")
    with pytest.raises(RuntimeError,match="CPU evidence IO failure"):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=clock,observer=fail)
    assert app.reads==reads+1 and m._stat_cache=={}
    gc.collect();assert reference() is None


def test_original_owned_gc_runs_once_after_reset_even_on_failure(sdk,monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
    assert "before_collect" in inspect.signature(run_owned_call).parameters, "pre-existing-GC seam missing"
    m,app,clock,_=sdk;a=api();events=[];refs=[]
    original_gc=gc.collect
    def collect():
        events.append("gc");return original_gc()
    monkeypatch.setattr(gc,"collect",collect)
    class Scene:
        def __init__(self):importlib.util.find_spec("dh116_cache_test_package")
    def operation(owned):
        scene=Scene();refs.append(weakref.ref(scene));owned.keep("runtime",scene)
        raise RuntimeError("expected acquisition failure")
    def before():
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=clock)
        events.append("cache reset")
    class Trace:
        def checkpoint(self,phase):
            events.append("checkpoint")
            assert all(r() is None for r in refs)
    with pytest.raises(RuntimeError,match="expected acquisition failure"):
        run_owned_call(operation,synchronize=lambda:events.append("sync"),ownership=Trace(),before_collect=before)
    assert events==["sync","cache reset","gc","checkpoint"]


def test_callback_failure_still_runs_original_gc_and_checkpoint(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
    assert "before_collect" in inspect.signature(run_owned_call).parameters
    events=[];original=gc.collect
    monkeypatch.setattr(gc,"collect",lambda:(events.append("gc") or original()))
    def fail():
        events.append("before");raise RuntimeError("CPU reset fault")
    with pytest.raises(RuntimeError,match="CPU reset fault"):
        run_owned_call(lambda owned:{"ok":True},synchronize=lambda:events.append("sync"),
                       ownership=SimpleNamespace(checkpoint=lambda p:events.append("checkpoint")),before_collect=fail)
    assert events==["sync","before","gc","checkpoint"]


def test_release_code_does_not_clear_frames_collect_or_update_app():
    a=api();tree=ast.parse(inspect.getsource(a))
    for n in ast.walk(tree):
        if isinstance(n,ast.Attribute):
            assert n.attr not in ("f_locals","clear_frames","collect","update"), n.attr


def test_independent_worker_version_and_freeze_cannot_use_legacy_approval(tmp_path):
    w=entry()
    plan=w.freeze_plan(ROOT,tmp_path/"native",w.legacy.BINDINGS)
    assert plan["version"]==w.VERSION and w.VERSION!=w.legacy.VERSION
    assert plan["stat_cache_release"]["max_reset_calls"]==1
    w.verify_plan(plan,w.legacy.digest(plan))
    bad=copy.deepcopy(plan);bad["version"]=w.legacy.VERSION
    with pytest.raises(ValueError):w.verify_plan(bad,w.legacy.digest(bad))


def test_worker_refuses_native_without_exact_fingerprint(tmp_path,monkeypatch):
    w=entry();plan=w.freeze_plan(ROOT,tmp_path/"native",w.legacy.BINDINGS)
    path=tmp_path/"plan.json";path.write_text(json.dumps(plan));calls=[]
    monkeypatch.setattr(w.legacy,"execute_native",lambda *a,**k:calls.append(1))
    with pytest.raises(ValueError):
        w.main(["--project",str(ROOT),"--output",plan["output"],"--plan",str(path),"--execute-native"])
    assert calls==[]


def test_worker_callback_writes_real_source_api_receipts_and_is_once_only(sdk,tmp_path):
    w=entry();a=api();m,app,clock,_=sdk;reference=retained_scene()
    plan={"output":str(tmp_path),"stat_cache_release":{"max_reset_calls":1,"source_contract":a.stat_cache_source_contract()}}
    callback=w.make_before_collect(plan)
    class ClockSim:
        current_time_step_index=2
        current_time=2/480
    sim=ClockSim()
    callback(sim)
    before=json.loads((tmp_path/"stat_cache_before.json").read_text())
    after=json.loads((tmp_path/"stat_cache_after.json").read_text())
    assert before["cached_exceptions"]>0 and after["cache_entries_total"]==0
    with pytest.raises(RuntimeError,match="once"):
        callback(sim)
    gc.collect();assert reference() is None


def test_legacy_hooks_are_optional_and_new_entry_supplies_one():
    w=entry()
    assert inspect.signature(w.legacy.execute_native).parameters["before_owner_collect"].default is None
    assert inspect.signature(w.legacy._execute_app_owned).parameters["before_owner_collect"].default is None
    tree=ast.parse(inspect.getsource(w))
    assert any(isinstance(n,ast.keyword) and n.arg=="before_owner_collect" for n in ast.walk(tree))



def test_disabled_cache_refuses_without_claiming_reset(sdk):
    m,app,clock,_=sdk;reference=retained_scene();a=api();reads=app.reads
    settings=sys.modules["omni.ext._impl.ext_settings"]
    settings._settings_iface=SimpleNamespace(get=lambda path:False if path.endswith("pathStatCacheEnabled") else None)
    settings._is_path_stat_cache_enabled.cache_clear()
    with pytest.raises(ValueError,match="enabled"):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=clock)
    assert m._stat_cache and app.reads==reads


def test_negative_app_update_number_refuses_before_api(sdk):
    m,app,clock,_=sdk;a=api();reads=app.reads
    def wrong():
        row=clock();row["app_update_number"]=-1;return row
    with pytest.raises(ValueError,match="clock"):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=wrong)
    assert app.reads==reads


def test_hook_interrupt_runs_original_gc_before_propagating(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
    assert "before_collect" in inspect.signature(run_owned_call).parameters
    events=[];original=gc.collect
    monkeypatch.setattr(gc,"collect",lambda:(events.append("gc") or original()))
    def interrupt():
        events.append("before");raise KeyboardInterrupt("CPU interruption")
    with pytest.raises(KeyboardInterrupt):
        run_owned_call(lambda owned:{"ok":True},synchronize=lambda:events.append("sync"),
            ownership=SimpleNamespace(checkpoint=lambda p:events.append("checkpoint")),before_collect=interrupt)
    assert events==["sync","before","gc","checkpoint"]


def test_observer_interrupt_still_calls_reset_and_propagates(sdk):
    m,app,clock,_=sdk;reference=retained_scene();a=api();reads=app.reads
    def interrupt(*args):raise KeyboardInterrupt("CPU observer interruption")
    with pytest.raises(KeyboardInterrupt):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=clock,observer=interrupt)
    assert app.reads==reads+1 and not m._stat_cache
    gc.collect();assert reference() is None



def test_worker_rejects_nonweak_clock_owner_without_strong_fallback(sdk,tmp_path):
    w=entry();a=api();m,app,clock,_=sdk;reference=retained_scene();reads=app.reads
    plan={"output":str(tmp_path),"stat_cache_release":{"max_reset_calls":1,"source_contract":a.stat_cache_source_contract()}}
    callback=w.make_before_collect(plan)
    with pytest.raises(RuntimeError,match="weak reference"):
        callback(SimpleNamespace(current_time_step_index=2,current_time=2/480))
    assert app.reads==reads and m._stat_cache
    row=json.loads((tmp_path/"stat_cache_error.json").read_text())
    assert row["native_release_verified"] is False and row["reset_retried"] is False

def _repopulate_cache():
    assert importlib.util.find_spec("dh116_cache_test_package") is not None


def test_successful_after_observer_cannot_repopulate_cache_unnoticed(sdk):
    m,app,clock,_=sdk;a=api();reads=app.reads
    def observe(phase,row):
        if phase=="after":_repopulate_cache()
    with pytest.raises(ValueError,match="cache|repopulat"):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=clock,observer=observe)
    assert m._stat_cache and app.reads==reads+1


def test_successful_receipt_write_cannot_repopulate_cache_unnoticed(sdk,tmp_path,monkeypatch):
    w=entry();a=api();m,app,clock,_=sdk;reads=app.reset_reads
    plan={"output":str(tmp_path),"stat_cache_release":{"max_reset_calls":1,"source_contract":a.stat_cache_source_contract()}}
    original=w.legacy.write_new
    def write(path,row):
        original(path,row)
        if Path(path).name=="stat_cache_release.json":_repopulate_cache()
    monkeypatch.setattr(w.legacy,"write_new",write)
    class ClockSim:
        current_time_step_index=2
        current_time=2/480
    with pytest.raises(RuntimeError,match="cache|repopulat"):
        w.make_before_collect(plan)(ClockSim())
    assert m._stat_cache and app.reset_reads==reads+1
    assert json.loads((tmp_path/"stat_cache_error.json").read_text())["reset_retried"] is False


def test_forged_function_location_does_not_qualify_loaded_body(sdk):
    from types import FunctionType
    m,app,clock,_=sdk;a=api();reads=app.reads
    old=m.reset_stat_cache
    bad=compile("def reset_stat_cache():\n    raise AssertionError('forged body')\n",str(SDK),"exec")
    code=next(c for c in bad.co_consts if inspect.iscode(c))
    m.reset_stat_cache=FunctionType(code.replace(co_firstlineno=old.__code__.co_firstlineno),vars(m))
    with pytest.raises(ValueError,match="source|body"):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=clock)
    assert app.reads==reads


def test_untrusted_enabled_predicate_cannot_claim_source_validation(sdk):
    m,app,clock,_=sdk;a=api();reads=app.reads
    m._is_path_stat_cache_enabled=lambda:True
    with pytest.raises(ValueError,match="predicate|settings|source"):
        a.release_owned_stat_cache(m,contract=a.stat_cache_source_contract(),clock_reader=clock)
    assert app.reads==reads


def test_last_clock_reader_cannot_repopulate_cache_after_snapshot(sdk):
    m,app,clock,_=sdk;a=api();calls=[]
    def reader():
        calls.append(1)
        if len(calls)==2:_repopulate_cache()
        return clock()
    with pytest.raises(ValueError,match="cache|repopulat"):
        a.verify_stat_cache_empty(m,contract=a.stat_cache_source_contract(),
            clock_reader=reader,expected_clock=clock())
    assert len(calls)==2 and m._stat_cache
