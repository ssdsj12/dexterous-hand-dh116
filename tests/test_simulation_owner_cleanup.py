"""CPU event doubles executing the installed registration/clear AST unchanged."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import weakref

import pytest

from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import (
    simulation_source_contract,clear_known_simulation_instance,probe_simulation_owner_edges,
)

def context(*,retain_frame=False,unsubscribe_failure=False):
    contract=simulation_source_contract()
    source=contract["sources"]["core"]["path"]
    tree=ast.parse(Path(source).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="SimulationContext")
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    assignments=[n for n in methods["__init__"].body if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Attribute)
                 and n.targets[0].attr in ("_message_bus","_on_post_physics_ready_callback")]
    assert len(assignments)==2
    events=[]
    class Subscription:
        def __init__(self,fn):self.fn=fn;self.enabled=True
        def unsubscribe(self):
            events.append("unsubscribe")
            if unsubscribe_failure:raise RuntimeError("CPU unsubscribe failure")
            self.fn=None;self.enabled=False
    class Bus:
        def __init__(self):self.tokens=[]
        def create_subscription_to_pop_by_type(self,event_type,fn):
            sub=Subscription(fn);self.tokens.append(sub);return sub
    bus=Bus()
    init=ast.parse("""def __init__(self):
    SimulationContext._instance=self
    SimulationContext._sim_context_initialized=True
    self._physics_callback_functions={}
    self._physics_functions={}
    self._stage_callback_functions={}
    self._timeline_callback_functions={}
    self._render_callback_functions={}
    self._stage_open_callback=self._physics_timer_callback=self._event_timer_callback=None
""").body[0]
    init.body.extend(assignments)
    body=[ast.parse("_instance=None").body[0],ast.parse("_sim_context_initialized=False").body[0],
          init,*[methods[n] for n in ("clear_instance","clear_all_callbacks","_on_post_physics_ready")]]
    fixture=ast.Module(body=[ast.ClassDef(name="SimulationContext",bases=[],keywords=[],body=body,decorator_list=[])],type_ignores=[])
    ns=dict(omni=SimpleNamespace(kit=SimpleNamespace(app=SimpleNamespace(get_app=lambda:SimpleNamespace(get_message_bus_event_stream=lambda:bus)))),
        IsaacEvents=SimpleNamespace(PHYSICS_READY=SimpleNamespace(value=1)),
        SimulationManager=SimpleNamespace(set_backend=lambda name:events.append("original_backend_"+name)),
        gc=SimpleNamespace(collect=lambda:events.append("original_clear_all_callbacks_gc")))
    exec(compile(ast.fix_missing_locations(fixture),source,"exec"),ns)
    core=ns["SimulationContext"]
    sink={}
    ns2=dict(Base=core,sys=sys,sink=sink)
    text="class LabContext(Base):\n def __init__(self):\n  super().__init__()\n"
    if retain_frame:text+="  sink['creation_frame']=sys._getframe()\n"
    exec(compile(text,contract["sources"]["lab"]["path"],"exec"),ns2)
    return ns2["LabContext"](),bus,events,contract,sink,core

def test_installed_original_clear_leaves_ready_subscription_owner():
    sim,bus,events,contract,sink,core=context()
    ref=weakref.ref(sim)
    type(sim).clear_instance()
    del sim
    assert ref() is not None
    assert core._instance is None and bus.tokens[0].enabled
    assert events==["original_clear_all_callbacks_gc","original_backend_numpy"]
    bus.tokens[0].unsubscribe()
    assert ref() is None

def test_narrow_cleanup_releases_callback_before_original_clear_without_extra_gc():
    sim,bus,events,contract,sink,core=context()
    ref=weakref.ref(sim)
    result=clear_known_simulation_instance(sim,contract=contract)
    assert result["unsubscribe_returned"] and result["subscription_field_cleared"]
    assert result["original_clear_instance_returned"]
    assert result["native_release_verified"] is False
    assert sim._on_post_physics_ready_callback is None
    assert events==["unsubscribe","original_clear_all_callbacks_gc","original_backend_numpy"]
    del sim
    assert ref() is None

def test_constructor_frame_remains_unknown_after_known_callback_is_removed():
    sim,bus,events,contract,sink,core=context(retain_frame=True)
    ref=weakref.ref(sim)
    clear_known_simulation_instance(sim,contract=contract)
    del sim
    result=probe_simulation_owner_edges(ref)
    assert result["alive"] is True
    frame=[r for r in result["selected_owner_edges"] if r["direct"]["kind"]=="frame"]
    assert frame and any("creation_frame" in r.get("matching_keys",[]) for r in frame[0]["upstream"])
    assert result["root_cause_proven"] is False and result["native_thread_attribution"] is None
    sink.clear()
    assert ref() is None

def test_probe_does_not_keep_referents_or_materialize_frame_locals():
    sim,bus,events,contract,sink,core=context()
    ref=weakref.ref(sim)
    result=probe_simulation_owner_edges(ref)
    assert result["alive"]
    clear_known_simulation_instance(sim,contract=contract)
    del sim
    assert ref() is None
    assert result["selected_owner_edges"]
    dead=probe_simulation_owner_edges(ref)
    assert dead["alive"] is False

def test_unsubscribe_failure_preserves_evidence_but_still_calls_original_clear():
    sim,bus,events,contract,sink,core=context(unsubscribe_failure=True)
    ref=weakref.ref(sim)
    try:clear_known_simulation_instance(sim,contract=contract)
    except RuntimeError as exc:saved=exc
    else:pytest.fail("failure must not become successful cleanup")
    assert core._instance is None
    assert sim._on_post_physics_ready_callback is bus.tokens[0]
    assert events==["unsubscribe","original_clear_all_callbacks_gc","original_backend_numpy"]
    # Release only the explicit CPU event double; holding the returned traceback
    # must not introduce another owner after the real boundary helper failed.
    bus.tokens[0].fn=None
    del sim
    assert ref() is None
    assert "CPU unsubscribe failure" in str(saved)

def test_unsupported_context_fails_before_any_mutation():
    events=[]
    sim=SimpleNamespace(_on_post_physics_ready_callback=SimpleNamespace(unsubscribe=lambda:events.append(1)))
    with pytest.raises(RuntimeError,match="source"):clear_known_simulation_instance(sim,contract=simulation_source_contract())
    assert events==[]

def test_source_and_contract_tampering_fail_closed(tmp_path):
    contract=simulation_source_contract()
    changed=deepcopy(contract)
    changed["sources"]["core"]["sha256"]="0"*64
    sim,bus,events,_,sink,core=context()
    with pytest.raises(RuntimeError):clear_known_simulation_instance(sim,contract=changed)
    assert events==[]
    file=tmp_path/"core.py";file.write_text("class SimulationContext: pass\n")
    with pytest.raises(ValueError):simulation_source_contract(core_source=file)
    bus.tokens[0].unsubscribe();type(sim).clear_instance()

def test_already_absent_is_explicit_and_not_native_release_proof():
    sim,bus,events,contract,sink,core=context()
    bus.tokens[0].unsubscribe();sim._on_post_physics_ready_callback=None
    result=clear_known_simulation_instance(sim,contract=contract)
    assert result["subscription_status"]=="already_absent"
    assert result["unsubscribe_returned"] is False and result["native_release_verified"] is False

@pytest.mark.parametrize("bound",[0,65,True])
def test_probe_requires_explicit_bounded_read_only_scan(bound):
    class Target:pass
    target=Target()
    with pytest.raises(ValueError):probe_simulation_owner_edges(weakref.ref(target),max_referrers=bound)


@pytest.mark.parametrize("retain_frame",[False,True])
def test_frozen_v3_error_lifecycle_uses_helper_before_close_and_keeps_unknown_frame(tmp_path,retain_frame):
    from scripts.dh116_pinch_startup_v3 import run_lifecycle
    state={}
    def make():
        sim,bus,events,contract,sink,core=context(retain_frame=retain_frame)
        sim.current_time_step_index=240;sim.current_time=.5000000260770321
        state.update(ref=weakref.ref(sim),bus=bus,events=events,contract=contract,sink=sink)
        return sim
    def acquire(sim,owners,trace,cleanup):
        raise ValueError("CPU same-position validator precision failure")
    def clear(sim):return clear_known_simulation_instance(sim,contract=state["contract"])
    def close():
        state["close_probe"]=probe_simulation_owner_edges(state["ref"])
    app=SimpleNamespace(update=lambda:None,close=close)
    with pytest.raises(RuntimeError,match="validator precision failure"):
        run_lifecycle(app,make_sim=make,acquire=acquire,synchronize=lambda:None,
            clear_sim=clear,stop=lambda:state["events"].append("original_stop"),
            manager_is_none=lambda:True,freeze=lambda:False,output=tmp_path,cleanup="explicit")
    assert state["close_probe"]["alive"] is retain_frame
    assert state["close_probe"]["root_cause_proven"] is False
    assert state["events"]==["unsubscribe","original_clear_all_callbacks_gc","original_backend_numpy","original_stop"]
    assert not state["bus"].tokens[0].enabled
    state["sink"].clear()
    assert state["ref"]() is None
