"""Actual owner on CPU SDK doubles, plus parameter readback; never native evidence."""
import json,sys
from pathlib import Path
import pytest,torch
from scripts import prepare_dh116_dual_pinch_parameter_interval as operation
from scripts import verify_dh116_dual_pinch_parameter_interval as entry
from scripts import validate_dh116_dual_pinch_parameter_interval as audit
import test_dh116_dual_pinch_director_interval as director_tests
import test_dh116_dual_pinch_closure_interval as prior_test
from test_dh116_dual_pinch_builder import prepared,geometry
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

ROOT=Path(__file__).resolve().parents[1]

def run_cpu(monkeypatch,tmp_path,prepared,geometry,*,failure=None,old_guard=False):
    calls=director_tests.install_cpu_directors(monkeypatch)
    original_install=prior_test.install_sdk_execution
    parameter_calls=[]
    def install(sim,rod,bindings,events):
        original_install(sim,rod,bindings,events)
        rod.rope.num_envs=1
        rod.rope.fixed_root_orientation=False
        getter=rod.rope.inv_masses
        def read():
            step=sim.current_time_step_index
            parameter_calls.append(step)
            if step==failure:raise ValueError("CPU injected parameter getter failure")
            return getter()[None]  # Actual installed CUDA getter shape is [1,41].
        rod.rope.inv_masses=read
    monkeypatch.setattr(prior_test,"install_sdk_execution",install)
    monkeypatch.setattr(prior_test,"closure",operation)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda device:None)
    monkeypatch.setattr(torch.cuda,"current_stream",lambda device:0)
    monkeypatch.setattr(torch.cuda,"default_stream",lambda device:0)
    if failure is not None and not old_guard:
        with pytest.raises(RuntimeError,match="material_director"):
            prior_test.run_actual_episode(monkeypatch,tmp_path,prepared,geometry,False)
    else:
        prior_test.run_actual_episode(monkeypatch,tmp_path,prepared,geometry,old_guard)
    return tmp_path/"closure",calls,parameter_calls

def test_full_owner_readbacks_and_disk_audit(monkeypatch,tmp_path,prepared,geometry):
    out,dirs,parameters=run_cpu(monkeypatch,tmp_path,prepared,geometry)
    assert dirs==parameters==list(range(2,19))
    r=operation.inspect_interval(out/"physical.jsonl")
    assert r["parameter_audit"]["valid_packets"]==17
    assert r["parameter_audit"]["complete_endpoint_sequence"] is True
    assert r["parameter_audit"]["teacher_admission"] is False
    assert json.loads((out/"after_stop.refs.json").read_text())["resources"]["material_director_reader"]["alive"] is False
    rows=[json.loads(s) for s in (out/"physical.jsonl").read_text().splitlines()]
    sample=next(decode_owned(x["payload"]) for x in rows if x.get("operation")=="physical")["exchange"]
    packet=sample["material_director"]["parameter_readback"]
    assert packet["identity"]["physics_step"]==3
    assert packet["fields"]["radius_m"]["native_measured"] is False
    assert packet["fields"]["mass_kg"]["native_measured"] is True

@pytest.mark.parametrize("old_guard",[False,True])
def test_readback_failure_keeps_raw_terminal_physics(monkeypatch,tmp_path,prepared,geometry,old_guard):
    out,dirs,parameters=run_cpu(monkeypatch,tmp_path,prepared,geometry,failure=11,old_guard=old_guard)
    r=operation.inspect_interval(out/"physical.jsonl",complete=False)
    assert r["parameter_audit"]["valid_packets"]==9
    assert r["measured_physics_rows"]==9 and r["last_sdk_step"]==11
    assert parameters==list(range(2,12)) and dirs==list(range(2,11))
    expected="post_step_source_clearance:right" if old_guard else "material_director_source_invalid"
    assert r["terminal_reason"]==expected
    assert (out/"app_close_returned.json").exists()

def test_reset_failure_cannot_execute_action(monkeypatch,tmp_path,prepared,geometry):
    out,dirs,parameters=run_cpu(monkeypatch,tmp_path,prepared,geometry,failure=2)
    assert parameters==[2] and dirs==[]
    r=operation.inspect_interval(out/"physical.jsonl",complete=False)
    assert r["parameter_audit"]["valid_packets"]==0
    assert r["measured_physics_rows"]==0 and r["last_sdk_step"]==2
    assert (out/"app_close_returned.json").exists()

def test_execute_reaches_original_owner_once(monkeypatch):
    calls=[]
    monkeypatch.setattr(operation.director.closure,"run_original_existing_app",
        lambda *a,**k:calls.append(k["prepared"]) or "owner")
    monkeypatch.setattr(entry.prior.prior.original,"execute",
        lambda *unused:entry.prior.prior.original.interval.run_existing_app(prepared={},project=ROOT))
    assert entry.execute(None,ROOT,{})=="owner" and calls==[{}]
