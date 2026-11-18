"""Original full Python owner, CPU SDK geometry/getter transport only."""
import json,sys
from pathlib import Path
import pytest
from test_dh116_dual_pinch_builder import prepared,geometry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

@pytest.mark.parametrize("failure,old_guard",[(False,False),(True,False),(True,True)])
def test_full_owner_or_getter_failure_keeps_durable_prefix(monkeypatch,tmp_path,prepared,geometry,failure,old_guard):
    import test_left_pinch_approach_owner as old
    from scripts import prepare_dh116_health_projected_pinch as operation
    from scripts import validate_dh116_health_projected_pinch as audit
    from go2_pvcnn.control.dh116 import tabletop_r0_health_separation as health
    real=health.capture_solver_health;calls=[]
    def capture(runtime,*,backend,sim,expected_clock):
        calls.append(sim.current_time_step_index)
        if failure and sim.current_time_step_index==11:
            def broken(self):raise RuntimeError("CPU injected health getter failure")
            with monkeypatch.context() as patch:
                patch.setattr(type(runtime.rope),"last_error",property(broken),raising=False)
                return real(runtime,backend=backend,sim=sim,expected_clock=expected_clock)
        return real(runtime,backend=backend,sim=sim,expected_clock=expected_clock)
    monkeypatch.setattr(health,"capture_solver_health",capture)
    install=old.director_tests.install_cpu_directors
    def install_module(*a,**k):
        result=install(*a,**k)
        # Exact module registry shape of a Python extension import, CPU double only.
        old_install=old.prior_test.install_sdk_execution
        def after(*a,**k):
            old_install(*a,**k)
            monkeypatch.setitem(sys.modules,"ropesim._ropesim",sys.modules["ropesim"]._ropesim)
        monkeypatch.setattr(old.prior_test,"install_sdk_execution",after)
        return result
    monkeypatch.setattr(old.director_tests,"install_cpu_directors",install_module)
    out=tmp_path/"closure"
    with bindings((old,"closure",operation)):
        if failure and not old_guard:
            with pytest.raises(RuntimeError,match="health_source_invalid"):
                old.run_owner(monkeypatch,tmp_path,prepared,geometry)
        else:old.run_owner(monkeypatch,tmp_path,prepared,geometry,old_guard=old_guard)
    checked=operation.inspect_interval(out/"physical.jsonl",complete=not failure)
    count=9 if failure else 232
    assert checked["measured_physics_rows"]==count
    assert calls==list(range(3,12 if failure else 235))
    assert (out/"app_close_returned.json").exists()
    assert len((out/"sdk_clock.jsonl").read_text().splitlines())==count
    entries=[json.loads(x) for x in (out/"physical.jsonl").read_text().splitlines()]
    physical=[decode_owned(x["payload"])["exchange"] for x in entries if x.get("operation")=="physical"]
    assert len(physical)==count and physical[-1]["physics_step"]==(11 if failure else 234)
    if failure:
        last=physical[-1]
        assert last["health_getter"]["capture_error"]
        assert last["health_getter"]["solver_status"]["last_error"] is None
        assert last["guard_failure"]==("post_step_source_clearance:right" if old_guard else "health_source_invalid")
        assert checked["health_getter_audit"]["source_complete"] is False
        assert sum(x.get("operation")=="end" for x in entries)==2
    else:
        assert checked["health_getter_audit"]["source_complete"]
        assert all(x["health_getter"]["solver_status"]["last_error"]==0 for x in physical)
        replay=audit.replay_health_r0(out)
        assert replay["status"]=="completed_offline" and replay["control_windows"]==29
        assert replay["live_r0_qualified"] is False and replay["training_reward"] is None
