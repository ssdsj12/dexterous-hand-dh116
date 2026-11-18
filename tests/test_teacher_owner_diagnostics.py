"""CPU ownership diagnostics only; simulated external holders are not native proof."""
import ast
import gc
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest

from go2_pvcnn.control.dh116 import tabletop_simulation_owner_cleanup as owners
from go2_pvcnn.control.dh116 import tabletop_teacher_force_runtime as force
from scripts import diagnose_dh116_teacher_initialization as worker

class Owner: pass

def identity_api():
    api=getattr(owners,"runtime_owner_ids",None)
    assert callable(api), "scalar runtime owner-ID adapter is missing"
    return api

def test_owner_identity_correlates_factory_force_state_namespace_and_known_objects():
    api=identity_api()
    sim,left,right=Owner(),Owner(),Owner()
    state=dict(sim=sim,left_hands=left,right_hands=right)
    values=dict(sim=sim,left_hands=left,right_hands=right,force_state=state)
    namespace={"create_runtime":lambda:None}
    result=api(values,namespace)
    assert result["factory_locals_id"]==id(values)
    assert result["force_state_id"]==id(state)
    assert result["namespace_id"]==id(namespace)
    assert result["known_object_ids"]=={name:id(values[name]) for name in ("sim","left_hands","right_hands")}
    assert result["root_cause_proven"] is False
    assert result["native_release_verified"] is False
    assert json.loads(json.dumps(result))==result
    refs=[weakref.ref(v) for v in (sim,left,right)]
    values.clear();state.clear();del sim,left,right
    gc.collect()
    assert all(ref() is None for ref in refs), "ID result must not retain objects"

def test_actual_force_adapter_opt_in_observer_receives_only_scalar_ids(tmp_path,monkeypatch):
    assert "owner_observer" in inspect.signature(force.create_force_runtime).parameters
    from go2_pvcnn.control.dh116 import tabletop_teacher_supported_runtime as supported
    monkeypatch.setattr(supported,"supported_seed_plan",lambda root:({"seeds_by_style":{"pinch":{}}},{}))
    tree=ast.parse("def create_runtime(args,app,preview_seeds,support_plan,force_hook):\n    return app()\n")
    monkeypatch.setattr(force,"compile_force_runtime",lambda *a:tree)
    source=tmp_path/"scripts/train_dh116_gpu_teacher_ppo.py"
    source.parent.mkdir();source.write_text("# CPU factory test double\n")
    values=dict(sim=Owner(),left_hands=Owner(),right_hands=Owner())
    values["force_state"]=dict(values)
    refs=[weakref.ref(values[k]) for k in ("sim","left_hands","right_hands")]
    expected=id(values);state_id=id(values["force_state"]);rows=[]
    rt=force.create_force_runtime(tmp_path,SimpleNamespace(mode="pinch"),lambda:values,
        "unused-by-cpu-compiler",SimpleNamespace(validate_arguments=lambda args:None),
        owner_observer=rows.append)
    assert len(rows)==1
    assert rows[0]["factory_locals_id"]==expected and rows[0]["force_state_id"]==state_id
    assert rows[0]["known_object_ids"]["sim"]==id(rt.sim)
    rt.__dict__.clear();values["force_state"].clear();values.clear()
    gc.collect()
    assert all(ref() is None for ref in refs)

def test_simulated_external_constructor_holder_reports_bounded_callers_without_retention():
    assert "max_frame_depth" in inspect.signature(owners.probe_simulation_owner_edges).parameters
    holder={}
    class Model:
        def __init__(self):holder["external_frame"]=sys._getframe()
    def factory():
        sim=Model()
        left_hands=Owner()
        return weakref.ref(sim),weakref.ref(left_hands)
    sim_ref,left_ref=factory()
    gc.collect()
    assert sim_ref() is not None and left_ref() is not None
    result=owners.probe_simulation_owner_edges(sim_ref,max_referrers=8,max_frame_depth=2)
    edge=next(e for e in result["selected_owner_edges"] if e["direct"]["kind"]=="frame")
    assert [v["function"] for v in edge["direct"]["caller_chain"]][:1]==["factory"]
    assert len(edge["direct"]["caller_chain"])<=2
    assert any("external_frame" in p.get("matching_keys",[]) for p in edge["upstream"])
    assert len(edge["upstream"])<=8 and result["max_hops"]==2
    assert json.loads(json.dumps(result))==result
    assert result["root_cause_proven"] is False
    holder.clear();gc.collect()
    assert sim_ref() is None and left_ref() is None

@pytest.mark.parametrize("depth",[0,9,True])
def test_caller_depth_is_explicitly_bounded(depth):
    assert "max_frame_depth" in inspect.signature(owners.probe_simulation_owner_edges).parameters
    target=Owner()
    with pytest.raises(ValueError):
        owners.probe_simulation_owner_edges(weakref.ref(target),max_frame_depth=depth)

def test_worker_probe_writes_scalar_artifact_without_extending_lifetime(tmp_path):
    api=getattr(worker,"write_owner_probe",None)
    assert callable(api), "worker owner-edge artifact wiring is missing"
    target=Owner();ref=weakref.ref(target)
    api(tmp_path,ref,"before_clear")
    row=json.loads((tmp_path/"before_clear.owner_edges.json").read_text())
    assert row["alive"] is True and row["native_release_verified"] is False
    del target;gc.collect()
    assert ref() is None

def test_worker_wires_opt_in_ids_and_both_existing_lifecycle_boundaries():
    tree=ast.parse(inspect.getsource(worker._execute_app_owned))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    factory=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="create_force_runtime")
    assert "owner_observer" in [k.arg for k in factory.keywords]
    phases=[n.args[2].value for n in calls if isinstance(n.func,ast.Name) and n.func.id=="write_owner_probe"]
    assert phases==["before_clear","after_stop"]
    text=inspect.getsource(worker)
    assert "factory_owner_ids.json" in text

def test_diagnostic_helpers_do_not_read_or_clear_frame_locals_or_collect():
    tree=ast.parse(inspect.getsource(owners))
    names={"runtime_owner_ids","probe_simulation_owner_edges","_frame_callers"}
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(funcs)==3, "bounded helper implementation is missing"
    for fn in funcs:
        for n in ast.walk(fn):
            if isinstance(n,ast.Attribute):
                assert n.attr not in ("f_locals","clear","collect"), (fn.name,n.attr)

def test_owner_probe_failure_does_not_skip_existing_cleanup_boundary(tmp_path,monkeypatch):
    api=getattr(worker,"write_owner_probe",None)
    assert callable(api)
    def fail(*args):raise OSError("CPU evidence write denied")
    monkeypatch.setattr(worker,"write_new",fail)
    target=Owner();ref=weakref.ref(target)
    failure=api(tmp_path,ref,"before_clear")
    assert "CPU evidence write denied" in failure
    del target;gc.collect()
    assert ref() is None

def test_upstream_description_scan_has_a_hard_count_bound(monkeypatch):
    holder={}
    class Model:
        def __init__(self):holder["frame"]=sys._getframe()
    sim=Model();ref=weakref.ref(sim);frame=holder["frame"]
    original=owners.gc.get_referrers
    def referrers(target):
        if target is sim:return [frame]
        if target is frame:return [{"frame":frame} for _ in range(100)]
        return original(target)
    monkeypatch.setattr(owners.gc,"get_referrers",referrers)
    result=owners.probe_simulation_owner_edges(ref,max_referrers=3)
    edge=result["selected_owner_edges"][0]
    assert edge["upstream_enumerated_count"]==100
    assert edge["upstream_scanned_count"]==3 and len(edge["upstream"])==3
    assert edge["upstream_truncated"] is True
    holder.clear()

def test_owner_id_helper_rejects_custom_mappings_without_dereferencing_them():
    api=identity_api()
    class Mapping(dict):
        def get(self,*a):pytest.fail("custom mapping dereferenced")
    with pytest.raises(TypeError):api(Mapping(),{})
    with pytest.raises(TypeError):api({},Mapping())


def test_factory_id_write_failure_still_hands_runtime_to_original_owned_cleanup(tmp_path,monkeypatch):
    safe_write=getattr(worker,"write_factory_owner_ids",None)
    assert callable(safe_write), "factory ID write needs a non-interrupting worker observer"
    from go2_pvcnn.control.dh116 import tabletop_teacher_supported_runtime as supported
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
    monkeypatch.setattr(supported,"supported_seed_plan",lambda root:({"seeds_by_style":{"pinch":{}}},{}))
    tree=ast.parse("def create_runtime(args,app,preview_seeds,support_plan,force_hook):\n    return app()\n")
    monkeypatch.setattr(force,"compile_force_runtime",lambda *a:tree)
    source=tmp_path/"scripts/train_dh116_gpu_teacher_ppo.py"
    source.parent.mkdir();source.write_text("# CPU factory test double\n")
    errors=[];events=[];refs=[];exception_refs=[]
    class WriteFailure(OSError):
        __slots__=("__weakref__",)
    def fail(path,row):
        assert path.name=="factory_owner_ids.json"
        assert json.loads(json.dumps(row))==row
        error=WriteFailure("CPU factory identity IO failure")
        exception_refs.append(weakref.ref(error))
        raise error
    monkeypatch.setattr(worker,"write_new",fail)
    def make_values():
        values=dict(sim=Owner(),left_hands=Owner(),right_hands=Owner())
        refs.extend(weakref.ref(values[k]) for k in ("sim","left_hands","right_hands"))
        values["force_state"]=dict(values)
        return values
    def operation(owned):
        rt=None
        try:
            rt=force.create_force_runtime(tmp_path,SimpleNamespace(mode="pinch"),make_values,
                "unused-by-cpu-compiler",SimpleNamespace(validate_arguments=lambda args:None),
                owner_observer=lambda row:safe_write(tmp_path,row,errors))
            events.append("runtime returned")
            assert all(ref() is None for ref in exception_refs)
            owned.keep("runtime",rt)
            events.append("owned handoff")
            assert rt.sim is not None
            return {"runtime_handed_off":True}
        finally:
            if rt is not None:rt.__dict__.clear()
            rt=None
            events.append("operation finally")
    class Trace:
        def checkpoint(self,phase):
            events.append(phase)
            assert all(ref() is None for ref in refs)
            assert all(ref() is None for ref in exception_refs)
    result=run_owned_call(operation,synchronize=lambda:events.append("original synchronize"),ownership=Trace())
    assert result=={"runtime_handed_off":True}
    assert events==["runtime returned","owned handoff","operation finally","original synchronize",
                    "owned hand and solver references dropped"]
    assert len(errors)==1 and type(errors[0]) is str
    assert "CPU factory identity IO failure" in errors[0]
    # The production worker's existing final error gate still fails this run.
    with pytest.raises(RuntimeError,match="CPU factory identity IO failure"):
        if errors:raise RuntimeError("\n".join(errors)) from None

@pytest.mark.parametrize("interrupt",[KeyboardInterrupt,SystemExit])
def test_factory_id_safe_writer_preserves_process_interrupts(tmp_path,monkeypatch,interrupt):
    safe_write=getattr(worker,"write_factory_owner_ids",None)
    assert callable(safe_write), "factory ID safe writer is missing"
    def fail(*args):raise interrupt("CPU interruption")
    monkeypatch.setattr(worker,"write_new",fail)
    errors=[]
    with pytest.raises(interrupt):
        safe_write(tmp_path,{"factory_locals_id":123},errors)
    assert errors==[]

def test_worker_factory_observer_uses_safe_writer_and_original_error_list():
    source=inspect.getsource(worker._execute_app_owned)
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    factory=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="create_force_runtime")
    observer=next(k.value for k in factory.keywords if k.arg=="owner_observer")
    assert isinstance(observer,ast.Lambda)
    assert ast.unparse(observer.body)=="write_factory_owner_ids(output, row, errors)"
