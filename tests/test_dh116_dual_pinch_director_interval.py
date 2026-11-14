"""Real builder/closure execution with explicit CPU endpoint-view substitutes."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

from scripts import prepare_dh116_dual_pinch_director_interval as operation
from scripts import verify_dh116_dual_pinch_director_interval as entry
import test_dh116_dual_pinch_closure_interval as prior_test
from test_dh116_dual_pinch_builder import prepared,geometry

ROOT=Path(__file__).resolve().parents[1]


def install_cpu_directors(monkeypatch,*,failure=None):
    from go2_pvcnn.control.dh116 import tabletop_material_director_readback as adapter
    old_install=prior_test.install_sdk_execution
    old_setup=prior_test.setup
    def setup_with_shared_view_release(*args,**kwargs):
        values=list(old_setup(*args,**kwargs))
        sim,drop=values[0],values[-1]
        def release():
            drop()
            # The CPU Sim object survives for clock assertions. Emulate its
            # original SDK clear/STOP release of the shared Tensor simulation
            # view; do not patch observations or weaken the real validator.
            sim.physics_sim_view=None
        values[-1]=release
        return tuple(values)
    monkeypatch.setattr(prior_test,"setup",setup_with_shared_view_release)
    calls=[]
    def install(sim,rod,bindings,events):
        old_install(sim,rod,bindings,events)
        sim.device="cuda:0"  # Declared device only; no CUDA operation in this CPU fixture.
        backend=sys.modules["ropesim"]._ropesim
        backend.RopeBatch=type(rod.rope)
        rod.rope.verts_per_rope=41
        rod.rope.last_error=0  # Installed getter is integer; old fixture used a falsey empty string.
        def quaternions():
            step=sim.current_time_step_index
            calls.append(step)
            if failure==step:raise ValueError("CPU injected director getter failure")
            q=np.zeros((1,41,4),dtype=np.float32);q[...,3]=1
            q[0,0,:]=[np.sin(step*.001),0,0,np.cos(step*.001)]
            q[0,-1]=[2,3,4,5]
            return q
        rod.rope.quaternions=quaternions
    def cpu_copy(rope,device):
        # The adapter's actual CUDA ordering/body has separate protocol tests.
        # This owner integration substitutes only the GPU memory transport.
        assert device=="cuda:0"
        return (torch.as_tensor(rope.positions()).cpu().numpy().copy(),
                np.array(rope.quaternions(),copy=True))
    monkeypatch.setattr(prior_test,"install_sdk_execution",install)
    monkeypatch.setattr(adapter,"_copy_endpoint",cpu_copy)
    monkeypatch.setattr(prior_test,"closure",operation)
    return calls


def run_cpu_interval(monkeypatch,tmp_path,prepared,geometry,*,failure=None,old_guard=False):
    calls=install_cpu_directors(monkeypatch,failure=failure)
    if failure is not None and not old_guard:
        with pytest.raises(RuntimeError,match="material_director"):
            prior_test.run_actual_episode(monkeypatch,tmp_path,prepared,geometry,False)
    else:
        prior_test.run_actual_episode(monkeypatch,tmp_path,prepared,geometry,old_guard)
    return tmp_path/"closure",calls


def test_full_actual_owner_has_one_reset_and_sixteen_director_endpoints(monkeypatch,tmp_path,prepared,geometry):
    out,calls=run_cpu_interval(monkeypatch,tmp_path,prepared,geometry)
    assert calls==list(range(2,19))
    audit=operation.inspect_interval(out/"physical.jsonl")
    assert audit["measured_physics_rows"]==16 and audit["last_sdk_step"]==18
    assert (out/"director_reset.json").exists()
    assert json.loads((out/"result.json").read_text())["failure"] is None
    assert (out/"app_close_returned.json").exists()
    resources=json.loads((out/"after_stop.refs.json").read_text())["resources"]
    assert resources["left_asset_sim_view"]["alive"] is False
    assert resources["right_asset_sim_view"]["alive"] is False


@pytest.mark.parametrize("old_guard",[False,True])
def test_director_failure_keeps_completed_ninth_row_and_original_guard(monkeypatch,tmp_path,prepared,geometry,old_guard):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    out,calls=run_cpu_interval(monkeypatch,tmp_path,prepared,geometry,failure=11,old_guard=old_guard)
    audit=operation.inspect_interval(out/"physical.jsonl",complete=False)
    assert audit["measured_physics_rows"]==9 and audit["last_sdk_step"]==11
    reason="post_step_source_clearance:right" if old_guard else "material_director_source_invalid"
    assert audit["terminal_reason"]==reason
    entries=[json.loads(x) for x in (out/"physical.jsonl").read_text().splitlines()]
    rows=[decode_owned(x["payload"]) for x in entries[1:] if x["operation"]=="physical"]
    last=rows[-1]["exchange"]
    assert last["material_director"] is None
    assert last["director_capture_error"]=="ValueError: CPU injected director getter failure"
    assert last["closure_guard_failure"]==("post_step_source_clearance:right" if old_guard else None)
    assert last["guard_failure"]==reason
    assert calls==list(range(2,12))
    assert len((out/"sdk_clock.jsonl").read_text().splitlines())==9
    assert sum(x.get("operation")=="end" for x in entries)==2
    assert (out/"app_close_returned.json").exists()


def test_reset_director_failure_preserves_two_setup_steps_and_reset(monkeypatch,tmp_path,prepared,geometry):
    out,calls=run_cpu_interval(monkeypatch,tmp_path,prepared,geometry,failure=2)
    packet=json.loads((out/"director_reset.json").read_text())
    assert packet["payload"] is None and "CPU injected" in packet["capture_error"]
    assert packet["identity"]["physics_step"]==2 and packet["identity"]["rod_tick"]==0
    audit=operation.inspect_interval(out/"physical.jsonl",complete=False)
    assert audit["reset_recorded"] is True and audit["measured_physics_rows"]==0
    assert audit["last_sdk_step"]==2 and calls==[2]
    assert (out/"sdk_clock.jsonl").read_text()==""
    assert (out/"app_close_returned.json").exists()


def test_execute_rebinding_reaches_original_owner_once(monkeypatch):
    calls=[]
    monkeypatch.setattr(operation.closure,"run_original_existing_app",
        lambda *a,**k:calls.append(k["prepared"]) or "owner")
    monkeypatch.setattr(entry.prior.original,"execute",
        lambda *unused:entry.prior.original.interval.run_existing_app(prepared={},project=ROOT))
    assert entry.execute(None,ROOT,{})=="owner"
    assert calls==[{}]


@pytest.mark.parametrize("packet,error",[(None,None),({}, "broken"),(None,"")])
def test_inconsistent_director_error_packet_is_rejected(packet,error):
    row=dict(original_guard_failure=None,guard_failure=None,closure_guard_failure=None,
        geometry_capture_error=None,table_capture_error=None,
        closure_geometry=dict(passed=True,terminal_reason=None),hand_table_contacts={},
        director_capture_error=error,material_director=packet)
    with pytest.raises(ValueError,match="director"):
        operation.director_observation_failure(row)
