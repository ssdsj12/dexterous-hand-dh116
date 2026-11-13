"""CPU endpoint reader packets and disk tampering; no native execution."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from test_dh116_material_director_readback import case
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned, decode_owned

MODULE="scripts.validate_dh116_dual_pinch_director_interval"
VERSION="dh116-dual-pinch-director-endpoints-v1"
PROFILE="67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498"


def api():
    return importlib.import_module(MODULE)


def write(path, value):
    path.write_text(json.dumps(encode_owned(value),allow_nan=False)+"\n")


def journal(path, rows):
    path.write_text("".join(json.dumps(encode_owned(row),allow_nan=False)+"\n" for row in rows))


@pytest.fixture
def evidence(tmp_path, case):
    c=case
    from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import load_pair_proposal
    c.nodes[:]=np.asarray(load_pair_proposal()["rod_seed"]["proposed_nodes_m"],dtype=np.float32)
    c.rod.p[:]=c.nodes
    c.owner=c.m.MaterialDirectorReadback(c.nodes,device="cuda:0")
    packet=c.owner.capture_reset(c.row(),material_reference_uid="cpu-material-1",**c.args)
    reset=dict(version=c.m.VERSION,payload=packet,capture_error=None,identity=dict(c.clock))
    rows=[dict(version="CPU_PROTOCOL_FIXTURE_NOT_NATIVE"),
          dict(operation="reset",payload=dict(reset=dict(rod_reset=c.row(),after_clock=dict(c.clock),
               after_actual_clocks=dict(sdk_physics_step=2,sdk_timestamp_s=c.sim.current_time,
                   sdk_dt_s=1/480,rod=dict(c.clock)))))]
    for i in range(16):
        c.advance()
        c.rod.q[0,i]=[np.sin(.01*i),0,0,np.cos(.01*i)]
        packet=c.owner.capture(c.row(),**c.args)
        rows.append(dict(operation="physical",payload=dict(exchange=dict(rod=c.row(),
            post_step_clearance=dict(sdk_time_s=c.sim.current_time),
            closure_guard_failure=None,guard_failure=None,material_director=packet,
            director_capture_error=None))))
    out=tmp_path/"cpu-endpoints";out.mkdir()
    write(out/"director_reset.json",reset)
    journal(out/"physical.jsonl",rows)
    # Explicitly synthetic reader liveness, used only by audit composition units.
    from scripts.validate_dh116_dual_pinch_interval import PHASES
    for phase in PHASES[1:]:
        write(out/(phase+".refs.json"),dict(resources={"material_director_reader":dict(alive=False)}))
    return SimpleNamespace(out=out,rows=rows,reset=reset)


def test_complete_reset_plus_16_real_reader_cpu_packets(evidence):
    r=api().inspect_directors(evidence.out/"physical.jsonl")
    assert r["reset_packet_valid"] is True
    assert r["valid_endpoint_packets"]==16 and r["valid_packets"]==17
    assert r["last_sdk_step"]==18 and r["last_rod_tick"]==192
    assert r["source_instance_id"]==evidence.reset["payload"]["source_instance_id"]
    assert r["complete_endpoint_sequence"] is True
    assert r["r0_tangential_slip_m"] is None
    assert r["native_qualification"] is r["teacher_admission"] is False
    # Padding deliberately [2,3,4,5], not a real segment unit quaternion.
    assert evidence.reset["payload"]["padding_quaternion_xyzw"].tolist()==[[2,3,4,5]]


@pytest.mark.parametrize("fault",[
    "source","instance","material","identity","tick","sdk_time","solver_time","nodes","rest",
    "dtype","shape","unit","split","padding","index","steps","qualification","slip","device",
    "missing_error","error_and_packet","terminal","contact_valid","identity_bool"])
def test_packet_damage_is_rejected(evidence,fault):
    row=evidence.rows[3]["payload"]["exchange"]
    p=row["material_director"]
    if fault=="source":p["sources"]["quaternion_order"]="wxyz"
    elif fault=="instance":p["source_instance_id"]+=1
    elif fault=="material":p["material_reference_uid"]="other"
    elif fault=="identity":p["identity"]["episode_uid"]="other"
    elif fault=="tick":p["solver_substeps_since_reset"]+=1
    elif fault=="sdk_time":p["sdk_time_s"]+=.001
    elif fault=="solver_time":p["solver_time_readback_s"]+=.001
    elif fault=="nodes":p["node_positions_m"][0,0,0]+=.001
    elif fault=="rest":p["constructor_rest_lengths_m"][0]+=.001
    elif fault=="dtype":p["segment_quaternions_xyzw"]=p["segment_quaternions_xyzw"].astype("float64")
    elif fault=="shape":p["quaternions_xyzw_with_padding"]=p["quaternions_xyzw_with_padding"][:,:40]
    elif fault=="unit":
        p["quaternions_xyzw_with_padding"][0,0]=0
        p["segment_quaternions_xyzw"][0,0]=0
    elif fault=="split":p["segment_quaternions_xyzw"][0,0,0]+=.001
    elif fault=="padding":p["padding_quaternion_xyzw"][0,0]+=.001
    elif fault=="index":p["real_segment_indices"][-1]=40
    elif fault=="steps":p["new_sdk_steps"]=1
    elif fault=="qualification":p["native_qualification"]=True
    elif fault=="slip":p["r0_tangential_slip_m"]=0.
    elif fault=="device":p["declared_device"]="cpu"
    elif fault=="missing_error":del row["director_capture_error"]
    elif fault=="error_and_packet":row["director_capture_error"]="CPU error"
    elif fault=="terminal":row["guard_failure"]="material_director_source_invalid"
    elif fault=="contact_valid":row["rod"]["contact_valid"]=False
    elif fault=="identity_bool":p["identity"]["env_id"]=False
    journal(evidence.out/"physical.jsonl",evidence.rows)
    with pytest.raises((ValueError,KeyError,TypeError)):
        api().inspect_directors(evidence.out/"physical.jsonl")


@pytest.mark.parametrize("fault",["identity","nodes","rest","clock","after_clock","reset_contact","reset_time","reset_uid"])
def test_reset_is_bound_to_actual_completed_reset(evidence,fault):
    reset=evidence.rows[1]["payload"]["reset"]
    p=evidence.reset["payload"]
    if fault=="identity":evidence.reset["identity"]["run_uuid"]="other"
    elif fault=="nodes":reset["rod_reset"]["node_positions"][0,1,0]+=.002
    elif fault=="rest":reset["rod_reset"]["rest_lengths_m"][0]+=.001
    elif fault=="clock":reset["after_actual_clocks"]["sdk_physics_step"]=3
    elif fault=="after_clock":reset["after_clock"]["rod_uid"]="other"
    elif fault=="reset_contact":reset["rod_reset"]["contact_valid"]=True
    elif fault=="reset_time":p["sdk_time_s"]+=.001
    elif fault=="reset_uid":p["material_reference_uid"]=""
    write(evidence.out/"director_reset.json",evidence.reset)
    journal(evidence.out/"physical.jsonl",evidence.rows)
    with pytest.raises((ValueError,KeyError,TypeError)):
        api().inspect_directors(evidence.out/"physical.jsonl")


def test_terminal_capture_error_keeps_actual_prefix(evidence):
    rows=evidence.rows[:6]
    last=rows[-1]["payload"]["exchange"]
    last.update(material_director=None,director_capture_error="RuntimeError: CPU getter failed",
                guard_failure="material_director_source_invalid")
    journal(evidence.out/"physical.jsonl",rows)
    r=api().inspect_directors(evidence.out/"physical.jsonl",complete=False)
    assert r["measured_physics_rows"]==4 and r["valid_endpoint_packets"]==3
    assert r["capture_error_rows"]==[6] and r["last_sdk_step"]==6
    assert r["complete_endpoint_sequence"] is False
    with pytest.raises(ValueError):api().inspect_directors(evidence.out/"physical.jsonl")


def test_prior_guard_wins_but_capture_error_cannot_allow_later_step(evidence):
    rows=evidence.rows[:4]
    last=rows[-1]["payload"]["exchange"]
    last.update(material_director=None,director_capture_error="CPU error",
        closure_guard_failure="post_step_source_clearance:left",
        guard_failure="post_step_source_clearance:left")
    journal(evidence.out/"physical.jsonl",rows)
    assert api().inspect_directors(evidence.out/"physical.jsonl",complete=False)["capture_error_rows"]==[4]
    rows.append(evidence.rows[4]);journal(evidence.out/"physical.jsonl",rows)
    with pytest.raises(ValueError):api().inspect_directors(evidence.out/"physical.jsonl",complete=False)


def test_reset_capture_failure_is_unknown_and_zero_measured(evidence):
    evidence.reset.update(payload=None,capture_error="CPU reset getter failed")
    write(evidence.out/"director_reset.json",evidence.reset)
    journal(evidence.out/"physical.jsonl",evidence.rows[:2])
    r=api().inspect_directors(evidence.out/"physical.jsonl",complete=False)
    assert r["reset_packet_valid"] is False and r["valid_packets"]==0
    assert r["reset_capture_error"]=="CPU reset getter failed"
    with pytest.raises(ValueError):api().inspect_directors(evidence.out/"physical.jsonl")
    journal(evidence.out/"physical.jsonl",evidence.rows[:3])
    with pytest.raises(ValueError):api().inspect_directors(evidence.out/"physical.jsonl",complete=False)


@pytest.mark.parametrize("fault",["short","long","skip","nonfinite","truncated"])
def test_exact_count_sequence_and_json_required(evidence,fault):
    if fault=="short":evidence.rows.pop()
    elif fault=="long":evidence.rows.append(deepcopy(evidence.rows[-1]))
    elif fault=="skip":
        exchange=evidence.rows[4]["payload"]["exchange"]
        exchange["rod"]["physics_step"]+=1
    elif fault=="nonfinite":
        evidence.rows[4]["payload"]["exchange"]["material_director"]["segment_quaternions_xyzw"][0,0,0]=np.nan
    journal(evidence.out/"physical.jsonl",evidence.rows)
    if fault=="truncated":
        p=evidence.out/"physical.jsonl";p.write_text(p.read_text().rstrip())
    with pytest.raises((ValueError,KeyError,TypeError)):
        api().inspect_directors(evidence.out/"physical.jsonl")


@pytest.mark.parametrize("base_passed",[True,False])
def test_audit_runs_frozen_closure_gate_and_restores_bindings(evidence,monkeypatch,base_passed):
    from scripts import prepare_dh116_dual_pinch_closure_interval as old
    from scripts import validate_dh116_dual_pinch_closure_interval as base
    prior_version,prior_inspect=old.VERSION,old.inspect_interval
    inspection=api().inspect_directors(evidence.out/"physical.jsonl")
    inspector=lambda path,complete=True:dict(material_director_audit=api().inspect_directors(path,complete=complete))
    operation=SimpleNamespace(VERSION=VERSION,inspect_interval=inspector)
    monkeypatch.setitem(sys.modules,"scripts.prepare_dh116_dual_pinch_director_interval",operation)
    monkeypatch.setattr(importlib.import_module("scripts"),"prepare_dh116_dual_pinch_director_interval",operation,raising=False)
    calls=[]
    def old_gate(output,outer_exit,cleanup,*,profile_sha256):
        calls.append((output,outer_exit,cleanup,profile_sha256))
        assert old.VERSION==VERSION and old.inspect_interval is inspector
        return dict(lifecycle_qualified=base_passed,reason=None if base_passed else "CPU outer failure",
                    journal=dict(material_director_audit=inspection),
                    same_rod_contact_qualified=False,teacher_admission=False,hand_table_contacts=None)
    monkeypatch.setattr(base,"audit_outcome",old_gate)
    r=api().audit_outcome(evidence.out,evidence.out/"outer.json","explicit",profile_sha256=PROFILE)
    assert len(calls)==1 and calls[0][3]==PROFILE
    assert r["lifecycle_qualified"] is r["endpoint_director_readback_qualified"] is base_passed
    assert r["same_rod_contact_qualified"] is r["teacher_admission"] is False
    assert r["r0_tangential_slip_m"] is None
    assert old.VERSION==prior_version and old.inspect_interval is prior_inspect


def test_no_director_journal_can_pass_even_if_base_gate_is_stubbed_success(evidence,monkeypatch):
    from scripts import validate_dh116_dual_pinch_closure_interval as base
    operation=SimpleNamespace(VERSION=VERSION,inspect_interval=lambda *a,**k:{})
    monkeypatch.setitem(sys.modules,"scripts.prepare_dh116_dual_pinch_director_interval",operation)
    monkeypatch.setattr(importlib.import_module("scripts"),"prepare_dh116_dual_pinch_director_interval",operation,raising=False)
    monkeypatch.setattr(base,"audit_outcome",lambda *a,**k:dict(lifecycle_qualified=True,journal={}))
    (evidence.out/"director_reset.json").unlink()
    r=api().audit_outcome(evidence.out,evidence.out/"outer.json","explicit",profile_sha256=PROFILE)
    assert r["lifecycle_qualified"] is r["endpoint_director_readback_qualified"] is False
    assert r["material_director_audit_error"]


def test_old_actual_closure_has_no_director_scope():
    path=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-dual-pinch-closure-20261113-b9BDuO/native-runs/flex.f7-dual-pinch-guarded-closure-native/attempt-1ouifpxg/native/physical.jsonl")
    with pytest.raises(FileNotFoundError):api().inspect_directors(path)
    attempt=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-dual-pinch-guarded-closure-native/74f8378ee4648f30c8ecb2ba8f0aa5f2caf9ef396a8e49b9c532cbf44ff7fad6/attempt-1ouifpxg")
    result=api().audit_outcome(path.parent,attempt/"outer_exit.json","explicit",profile_sha256=PROFILE)
    assert result["lifecycle_qualified"] is result["endpoint_director_readback_qualified"] is False
    assert "execution scope differs" in result["reason"]
    assert result["material_director_audit_error"].startswith("FileNotFoundError")


def test_failed_reset_still_requires_real_finite_sdk_time(evidence):
    evidence.reset.update(payload=None,capture_error="CPU reset failure")
    evidence.rows[1]["payload"]["reset"]["after_actual_clocks"]["sdk_timestamp_s"]=False
    write(evidence.out/"director_reset.json",evidence.reset)
    journal(evidence.out/"physical.jsonl",evidence.rows[:2])
    with pytest.raises(ValueError):
        api().inspect_directors(evidence.out/"physical.jsonl",complete=False)


@pytest.mark.parametrize("fault",["missing","alive","unknown"])
def test_new_reader_requires_explicit_after_stop_release(evidence,monkeypatch,fault):
    from scripts import validate_dh116_dual_pinch_closure_interval as base
    inspection=api().inspect_directors(evidence.out/"physical.jsonl")
    operation=SimpleNamespace(VERSION=VERSION,inspect_interval=lambda *a,**k:{})
    monkeypatch.setitem(sys.modules,"scripts.prepare_dh116_dual_pinch_director_interval",operation)
    monkeypatch.setattr(importlib.import_module("scripts"),"prepare_dh116_dual_pinch_director_interval",operation,raising=False)
    monkeypatch.setattr(base,"audit_outcome",lambda *a,**k:dict(lifecycle_qualified=True,
        journal=dict(material_director_audit=inspection)))
    path=evidence.out/"after_stop.refs.json"
    value=json.loads(path.read_text())
    if fault=="missing":del value["resources"]["material_director_reader"]
    else:value["resources"]["material_director_reader"]["alive"]=True if fault=="alive" else None
    write(path,value)
    result=api().audit_outcome(evidence.out,evidence.out/"outer.json","explicit",profile_sha256=PROFILE)
    assert result["lifecycle_qualified"] is result["endpoint_director_readback_qualified"] is False


def test_consistent_reset_translation_cannot_replace_frozen_pair_seed(evidence,monkeypatch):
    from scripts import validate_dh116_dual_pinch_closure_interval as base
    # A constant X translation leaves the original Y segment spacing unchanged.
    delta=np.array([.01,0,0],np.float32)
    evidence.reset["payload"]["node_positions_m"]+=delta
    evidence.rows[1]["payload"]["reset"]["rod_reset"]["node_positions"]+=delta
    for row in evidence.rows[2:]:
        e=row["payload"]["exchange"]
        e["rod"]["node_positions"]+=delta
        e["material_director"]["node_positions_m"]+=delta
    write(evidence.out/"director_reset.json",evidence.reset)
    journal(evidence.out/"physical.jsonl",evidence.rows)
    inspection=api().inspect_directors(evidence.out/"physical.jsonl")
    operation=SimpleNamespace(VERSION=VERSION,inspect_interval=lambda *a,**k:{})
    monkeypatch.setitem(sys.modules,"scripts.prepare_dh116_dual_pinch_director_interval",operation)
    monkeypatch.setattr(importlib.import_module("scripts"),"prepare_dh116_dual_pinch_director_interval",operation,raising=False)
    monkeypatch.setattr(base,"audit_outcome",lambda *a,**k:dict(lifecycle_qualified=True,
        journal=dict(material_director_audit=inspection)))
    result=api().audit_outcome(evidence.out,evidence.out/"outer.json","explicit",profile_sha256=PROFILE)
    assert result["lifecycle_qualified"] is result["endpoint_director_readback_qualified"] is False
    assert "frozen paired constructor" in result["material_director_audit_error"]


# Build the real Python owner anew with explicit SDK/CUDA substitutes. Do not
# relabel the archived producer which retained a shared CPU simulation view.
from test_dh116_dual_pinch_builder import prepared,geometry

@pytest.fixture(scope="module")
def current_cpu_owner(tmp_path_factory,prepared,geometry):
    from test_dh116_dual_pinch_director_interval import run_cpu_interval
    root=tmp_path_factory.mktemp("director-current-cpu-owner")
    with pytest.MonkeyPatch.context() as patch:
        output,calls=run_cpu_interval(patch,root,prepared,geometry)
    assert calls==list(range(2,19))
    return output


@pytest.fixture
def complete_cpu_owner(tmp_path,current_cpu_owner):
    import shutil
    from test_dh116_dual_pinch_postexit import fixture as outer_fixture
    out=outer_fixture(tmp_path)
    for path in current_cpu_owner.iterdir():
        if path.is_file():shutil.copyfile(path,out/path.name)
    # CPU operation deliberately had no native profile authorization; this
    # synthetic audit fixture supplies the expected identity to exercise gates.
    result=json.loads((out/"result.json").read_text())
    assert result["acquisition"]["profile_sha256"] is None
    result["acquisition"]["profile_sha256"]=PROFILE
    write(out/"result.json",result)
    return out


def test_full_real_cpu_owner_runs_every_frozen_postexit_gate(complete_cpu_owner,tmp_path):
    result=api().audit_outcome(complete_cpu_owner,tmp_path/"outer_exit.json","explicit",profile_sha256=PROFILE)
    assert result["lifecycle_qualified"] is result["endpoint_director_readback_qualified"] is True,result
    assert result["material_director_audit"]["valid_packets"]==17
    assert result["physics_steps"]==18 and result["measurement_steps"]==16
    assert set(result["initial_geometry"])=={"left","right"}
    assert result["owned_non_sim_released"] is True
    assert all(v is False for v in result["director_reader_liveness"].values())
    assert result["hand_table_contacts"] is result["r0_tangential_slip_m"] is None
    assert result["same_rod_contact_qualified"] is result["teacher_admission"] is False


@pytest.mark.parametrize("fault",["sdk_time","clear_receipt","reader_trace","shared_view"])
def test_complete_directors_cannot_bypass_original_clock_or_release(complete_cpu_owner,tmp_path,fault):
    out=complete_cpu_owner
    if fault=="sdk_time":
        rows=[json.loads(line) for line in (out/"sdk_clock.jsonl").read_text().splitlines()]
        rows[0]["sdk_time_s"]+=.001
        journal(out/"sdk_clock.jsonl",rows)
    elif fault=="clear_receipt":
        value=json.loads((out/"simulation_clear_receipt.json").read_text())
        value["unsubscribe_returned"]=False
        write(out/"simulation_clear_receipt.json",value)
    else:
        value=json.loads((out/"after_stop.refs.json").read_text())
        if fault=="reader_trace":del value["resources"]["material_director_reader"]
        else:value["resources"]["left_asset_sim_view"]["alive"]=True
        write(out/"after_stop.refs.json",value)
    result=api().audit_outcome(out,tmp_path/"outer_exit.json","explicit",profile_sha256=PROFILE)
    assert result["lifecycle_qualified"] is result["endpoint_director_readback_qualified"] is False,result
