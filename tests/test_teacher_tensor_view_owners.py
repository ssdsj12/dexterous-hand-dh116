"""CPU source-shaped views only; no native release or crash qualification."""
import ast
import gc
import inspect
import json
import weakref

import pytest

from go2_pvcnn.control.dh116 import tabletop_simulation_owner_cleanup as module
from scripts import diagnose_dh116_teacher_initialization as worker

class Resource: pass

def api():
    cls=getattr(module,"TensorViewOwners",None)
    assert callable(cls), "weak-only tensor view ownership observer is missing"
    return cls()

def graph():
    backend=Resource();view=Resource();view._backend=backend
    left=Resource();right=Resource()
    left._physics_sim_view=right._physics_sim_view=view
    left._data=Resource();right._data=Resource()
    left._data._physics_sim_view=right._data._physics_sim_view=view
    runtime=Resource();runtime.left_hands=left;runtime.right_hands=right
    class Manager: _physics_sim_view=view
    return runtime,Manager,view,backend

def test_shared_asset_and_data_view_remains_visible_when_manager_is_none():
    probe=api();runtime,manager,view,backend=graph()
    initial=probe.capture(runtime,manager)
    assert initial["resources"]["manager_sim_view"]["object_id"]==id(view)
    for name in ("left_asset_sim_view","left_data_sim_view","right_asset_sim_view","right_data_sim_view"):
        assert initial["resources"][name]["object_id"]==id(view)
        assert initial["resources"][name+"_backend"]["object_id"]==id(backend)
    manager._physics_sim_view=None
    # Source-shaped STOP invalidation drops the root wrapper, not cached sim views.
    runtime.left_hands._root_physx_view=None
    runtime.right_hands._root_physx_view=None
    after=probe.observe(manager)
    assert after["manager_current"]["state"]=="none"
    assert after["resources"]["manager_sim_view"]["python_liveness"]=="alive"
    assert after["current_fields"]["left_asset._physics_sim_view"]["object_id"]==id(view)
    assert after["current_fields"]["left_data._physics_sim_view"]["object_id"]==id(view)
    assert after["current_fields"]["left_asset_sim_view._backend"]["object_id"]==id(backend)
    assert after["native_release_verified"] is False
    assert after["root_cause_proven"] is False
    assert json.loads(json.dumps(after))==after

def test_view_observer_and_returned_json_do_not_keep_any_resource_alive():
    probe=api();runtime,manager,view,backend=graph()
    refs=[weakref.ref(v) for v in (runtime.left_hands,runtime.right_hands,
        runtime.left_hands._data,runtime.right_hands._data,view,backend)]
    result=probe.capture(runtime,manager)
    manager._physics_sim_view=None
    runtime.__dict__.clear()
    del runtime,view,backend
    gc.collect()
    assert all(ref() is None for ref in refs)
    after=probe.observe(manager)
    assert all(row["python_liveness"]=="gone" for row in after["resources"].values())
    assert json.loads(json.dumps(result))==result

def test_unknown_dictionary_or_weakref_never_falls_back_to_strong_ownership():
    probe=api()
    released=[]
    class NoDict:
        __slots__=()
    class NoWeakref:
        __slots__=("__dict__",)
        def __del__(self):released.append(self.__dict__["label"])
    view=NoWeakref();view.label="view";view._backend=NoDict()
    hand=NoWeakref();hand.label="hand";hand._physics_sim_view=view;hand._data=NoDict()
    runtime=Resource();runtime.left_hands=hand;runtime.right_hands=NoDict()
    class Manager: _physics_sim_view=view
    result=probe.capture(runtime,Manager)
    assert result["resources"]["left_asset_sim_view"]["python_liveness"]=="unknown"
    assert result["resources"]["left_asset_sim_view"]["weakref_status"]=="unsupported"
    assert result["resources"]["left_data_sim_view"]["state"]=="unknown"
    assert result["resources"]["right_asset_sim_view"]["state"]=="unknown"
    assert result["resources"]["left_asset_sim_view_backend"]["python_liveness"]=="unknown"
    assert result["resources"]["left_asset_sim_view_backend"]["object_id"]==id(view._backend)
    Manager._physics_sim_view=None
    runtime.__dict__.clear();del runtime,hand,view
    gc.collect()
    assert probe.observe(Manager)["resources"]["left_asset_sim_view"]["python_liveness"]=="unknown"
    assert sorted(released)==["hand","view"], "unsupported weakrefs must not become strong aliases"

def test_instance_properties_are_not_evaluated_to_fill_missing_fields():
    probe=api()
    class Hand:
        @property
        def _physics_sim_view(self):pytest.fail("native-like getter evaluated")
        @property
        def _data(self):pytest.fail("data getter evaluated")
    class Manager:pass
    runtime=Resource();runtime.left_hands=runtime.right_hands=Hand()
    row=probe.capture(runtime,Manager)
    assert row["manager_current"]["state"]=="unknown"
    assert row["resources"]["left_asset_sim_view"]["state"]=="unknown"

def test_worker_tensor_diagnostic_io_failure_is_only_text_and_keeps_interrupts(tmp_path,monkeypatch):
    safe=getattr(worker,"write_tensor_view_owners",None)
    assert callable(safe), "worker tensor-view evidence wiring is missing"
    probe=api();runtime,manager,view,backend=graph();errors=[]
    def fail(*args):raise OSError("CPU tensor owner evidence IO failure")
    monkeypatch.setattr(worker,"write_new",fail)
    safe(tmp_path,probe,manager,"registered",errors,runtime=runtime)
    assert len(errors)==1 and type(errors[0]) is str
    assert "CPU tensor owner evidence IO failure" in errors[0]
    for kind in (KeyboardInterrupt,SystemExit):
        def interrupt(*args):raise kind("CPU interrupt")
        monkeypatch.setattr(worker,"write_new",interrupt)
        with pytest.raises(kind):safe(tmp_path,probe,manager,"after_stop",errors)

def test_worker_captures_after_owned_handoff_and_observes_original_boundaries():
    text=inspect.getsource(worker._execute_app_owned)
    assert "TensorViewOwners" in text
    assert text.index("owners.keep('runtime',rt)")<text.index("write_tensor_view_owners(output,view_owners")
    tree=ast.parse(text)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=="write_tensor_view_owners"]
    phases=[n.args[3].value for n in calls]
    assert sorted(phases)==["after_stop","before_clear","registered"]

def test_view_observer_never_invokes_native_properties_or_mutates_owner_fields():
    cls=getattr(module,"TensorViewOwners",None)
    assert callable(cls)
    text=inspect.getsource(cls)
    for forbidden in ("is_valid","invalidate(",".clear(","gc.collect","f_locals","getattr("):
        assert forbidden not in text
    tree=ast.parse(text)
    for n in ast.walk(tree):
        if isinstance(n,(ast.Assign,ast.AnnAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            for target in targets:
                if isinstance(target,ast.Attribute):
                    assert isinstance(target.value,ast.Name) and target.value.id=="self"


def test_tensor_evidence_write_failure_does_not_retain_views_or_error_traceback(tmp_path,monkeypatch):
    safe=worker.write_tensor_view_owners
    probe=api();runtime,manager,view,backend=graph();errors=[];exception_refs=[]
    refs=[weakref.ref(value) for value in (runtime.left_hands,runtime.right_hands,
        runtime.left_hands._data,runtime.right_hands._data,view,backend)]
    class Failure(OSError):
        __slots__=("__weakref__",)
    def fail(*args):
        error=Failure("CPU failed tensor artifact")
        exception_refs.append(weakref.ref(error))
        raise error
    monkeypatch.setattr(worker,"write_new",fail)
    safe(tmp_path,probe,manager,"registered",errors,runtime=runtime)
    assert all(ref() is None for ref in exception_refs)
    manager._physics_sim_view=None
    runtime.__dict__.clear();del runtime,view,backend
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert errors==["registered tensor view owners failed: Failure: CPU failed tensor artifact"]
