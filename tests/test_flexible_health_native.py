"""SDK-shaped CPU owner samples only; no native launch or implicit health zeros."""
import importlib
from copy import deepcopy
from types import SimpleNamespace
import sys
import pytest
from test_r0_numerical_health_source import health_case,row_for,capture
from test_flexible_parameter_readback import case

MODULE="go2_pvcnn.control.dh116.tabletop_health_getter_evidence"
def api():return importlib.import_module(MODULE)

@pytest.mark.parametrize("code",[0,9])
def test_actual_getter_boundaries_preserved_without_extra_steps(health_case,monkeypatch,code):
    c=health_case
    monkeypatch.setattr(type(c.rope),"last_error",property(lambda self:code))
    row=row_for(c)
    got=api().capture_boundary(c.runtime,backend=c.backend,sim=c.sim,exchange=row)
    assert got["solver_status"]["last_error"]==code
    assert got["boundary_before"]==got["boundary_after"]
    assert got["boundary_before"]["runtime_clock"]==c.expected
    assert got["boundary_before"]["sdk_step"]==3
    assert got["boundary_before"]["sdk_time_s"]==c.sim.current_time
    assert got["boundary_before"]["source_instance_id"]==id(c.rope)
    assert got["capture_error"] is None and c.events==[]
    checked=api().validate_capture(got,row)
    assert checked["last_error"]==code and checked["source_complete"]
    assert got["native_qualification"] is False

@pytest.mark.parametrize("damage",["getter","sdk_time","clock","backend"])
def test_capture_failure_keeps_completed_owned_row_and_no_references(health_case,monkeypatch,damage):
    c=health_case
    def read(self):
        if damage=="getter":raise RuntimeError("CPU injected getter error")
        if damage=="sdk_time":c.sim.current_time+=1
        if damage=="clock":c.base.physics_step+=1
        if damage=="backend":c.backend.__file__="/bad/source"
        return 0
    monkeypatch.setattr(type(c.rope),"last_error",property(read))
    row=row_for(c)
    got=api().capture_boundary(c.runtime,backend=c.backend,sim=c.sim,exchange=row)
    assert got["capture_error"] and got["boundary_before"]["sdk_step"]==3
    assert row["physics_step"]==3
    assert api().validate_capture(got,row)["source_complete"] is False
    assert all(not hasattr(v,"rope") for v in got.values())

@pytest.mark.parametrize("damage",["sdk_time_s","sdk_step","source_instance_id","runtime_clock"])
def test_boundary_tamper_rejected(health_case,damage):
    c=health_case;row=row_for(c);got=api().capture_boundary(c.runtime,backend=c.backend,sim=c.sim,exchange=row)
    if damage=="runtime_clock":got["boundary_after"][damage]["rod_tick"]+=12
    else:got["boundary_after"][damage]+=1
    with pytest.raises(ValueError):api().validate_capture(got,row)

@pytest.mark.parametrize("old_guard",[None,"post_step_source_clearance:left"])
def test_mixin_preserves_parent_terminal_and_does_not_import_native(health_case,monkeypatch,old_guard):
    c=health_case;row=row_for(c);row["guard_failure"]=old_guard
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    row["closure_guard_failure"]=old_guard
    monkeypatch.setitem(sys.modules,"ropesim._ropesim",c.backend)
    monkeypatch.setitem(sys.modules,"ropesim",SimpleNamespace(_ropesim=c.backend))
    monkeypatch.setattr(type(c.rope),"last_error",property(lambda self:7))
    class Parent:
        def _step(self,action,substep):return row
    class Child(api().HealthEvidenceMixin,Parent):pass
    instance=Child();instance._rod=c.runtime;instance._sim=c.sim
    got=instance._step(None,0)
    assert got is row and got["health_prior_guard_failure"]==old_guard
    assert got["guard_failure"]==(old_guard or "numerical_health_failure")
    assert got["health_getter"]["solver_status"]["last_error"]==7

def test_scope_idempotent_restoration_and_original_schedule():
    from scripts import prepare_dh116_health_projected_pinch as operation
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    from scripts import prepare_dh116_projected_pinch_native as old
    parent=director.DirectorExchange
    with operation.scenario_scope():
        first=director.DirectorExchange
        with operation.scenario_scope():
            assert director.DirectorExchange is first
            assert sum(c is api().HealthEvidenceMixin for c in first.__mro__)==1
            assert operation.action_schedule().equal(old.action_schedule())
    assert director.DirectorExchange is parent


def test_unexpected_observer_exception_keeps_the_completed_physical_row(health_case,monkeypatch):
    c=health_case;row=row_for(c)
    class Parent:
        def _step(self,action,substep):return row
    class Child(api().HealthEvidenceMixin,Parent):pass
    child=Child();child._rod=c.runtime;child._sim=c.sim
    def fail(*a,**k):raise RuntimeError("CPU observer adapter fault")
    monkeypatch.setattr(api(),"capture_boundary",fail)
    got=child._step(None,0)
    assert got is row and got["guard_failure"]=="health_source_invalid"
    assert got["health_getter"]["solver_status"] is None
    assert "observer adapter fault" in got["health_getter"]["capture_error"]
