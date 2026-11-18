"""Independent CPU artifact gates; synthetic records are explicitly test-only."""
import importlib
import json
from pathlib import Path
import pytest
from scripts import run_dh116_work_packages as scheduler
from scripts.probe_dh116_physx_lifetime import OWNED_NAMES, VERSION
ROOT=Path(__file__).resolve().parents[1]

def validator():
    assert (ROOT/"scripts/validate_dh116_physx_lifetime.py").exists(), "independent validator missing"
    return importlib.import_module("scripts.validate_dh116_physx_lifetime")

def write(path, value):
    path.write_text(json.dumps(value))

def evidence(tmp):
    out=tmp/"native";out.mkdir()
    log=tmp/"native.log"
    log.write_text('DH116_GDB_INFERIOR_STATUS {"exit_code": 0, "signal": null}\n')
    write(tmp/"outer_exit.json",dict(outer_exit_code=0,duration_seconds=20,
        log_sha256=scheduler.file_hash(log),native_lock_observed=True,**scheduler.debugger_status(log)))
    rows=[dict(phase="before_reset",sdk_step=0,sdk_time_s=0),
          dict(phase="after_reset",sdk_step=2,sdk_time_s=2/480)]
    (out/"initialization_boundaries.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    acquisition=dict(root_sim_aliases_identical=True,provider_bound_to_runtime=True,
        rod_tick=0,solver_time_origin_s=0.,solver_time_s=0.,physics_step=2,
        measured_solver_step_performed=False,provider_snapshot_performed=False,
        native_environments=1,render_products_created=0,teacher_admission=False,grasp_qualified=False)
    write(out/"acquisition.json",acquisition)
    clocks=[dict(phase=p,clock_available=True,sdk_step=2,sdk_time_s=2/480)
            for p in ("before_stop","after_stop","before_app_close","after_app_close")]
    result=dict(version=VERSION,case="registry",acquisition=dict(acquisition,case="registry",state={}),
        clocks=clocks,physics_steps=2,measurement_steps=0,rod_steps=0,ppo_updates=0,
        play_simulations=False,lifetime_released_before_close=True,failure=None,
        process_exit_verified=False,grasp_qualified=False,teacher_admission=False)
    write(out/"result.json",result)
    write(out/"before_app_close.json",dict(case="registry",play_simulations=False,
        process_exit_verified=False,lifetime_released=True,clocks=clocks[:3]))
    write(out/"app_close_returned.json",dict(case="registry",app_close_returned=True,process_exit_verified=False))
    (out/"ownership.jsonl").write_text(json.dumps(dict(phase="SDK STOP",manager_view_released=True))+"\n")
    for phase in ("resources_acquired","owned_hand_and_solver_references_dropped","before_stop",
                  "after_stop","before_app_close","after_app_close"):
        write(out/(phase+".refs.json"),{"resources":{n:{"alive":phase=="resources_acquired"} for n in OWNED_NAMES}})
        (out/(phase+".maps")).write_text("1000-2000 r-xp 0000 00:01 123 /fixture\n")
    content={"maps":"1000-2000 r-xp 0000 00:01 123 /fixture\n","bt.txt":"#0 Py_FinalizeEx\n",
        "threads.txt":"Thread 1\n#0 Py_FinalizeEx\n","registers.txt":"rip 0x1234\n",
        "instructions.txt":"=> 0x1234: ret\n","libraries.txt":"0x1000 0x2000 Yes /fixture.so\n",
        "mappings.txt":"Start Addr End Addr\n0x1000 0x2000 0x1000 0x0 /fixture.so\n"}
    hashes={}
    for suffix,text in content.items():
        name="gdb_finalize."+suffix;(out/name).write_text(text);hashes[name]=scheduler.file_hash(out/name)
    write(out/"gdb_finalize.json",dict(phase="finalize",inferior_pid=123,all_commands_succeeded=True,
        errors=[],artifact_sha256=hashes,process_exit_verified=False))
    return out

def test_complete_independently_reopened_outcome_remains_diagnostic(tmp_path):
    v=validator();out=evidence(tmp_path)
    result=v.validate_artifacts(out,tmp_path/"outer_exit.json","registry")
    assert result["lifecycle_qualified"] is True
    assert result["grasp_qualified"] is result["teacher_admission"] is False

@pytest.mark.parametrize("damage",["marker_only","empty_threads","changed_registers","clock","live",
    "unknown","outer","signal","duration","manager","solver","close","log_hash","case"])
def test_independent_gate_rejects_incomplete_or_changed_evidence(tmp_path,damage):
    v=validator();out=evidence(tmp_path)
    def change(path,key,value):
        data=json.loads(path.read_text());data[key]=value;write(path,data)
    if damage=="marker_only": (out/"gdb_finalize.json").unlink()
    if damage=="empty_threads":
        p=out/"gdb_finalize.threads.txt";p.write_text(" ")
        r=json.loads((out/"gdb_finalize.json").read_text());r["artifact_sha256"][p.name]=scheduler.file_hash(p);write(out/"gdb_finalize.json",r)
    if damage=="changed_registers": (out/"gdb_finalize.registers.txt").write_text("changed")
    if damage=="clock":
        r=json.loads((out/"result.json").read_text());r["clocks"][-1]["sdk_step"]=3;write(out/"result.json",r)
    if damage in ("live","unknown"):
        r=json.loads((out/"before_app_close.refs.json").read_text());r["resources"]["hand"]["alive"]=True if damage=="live" else None;write(out/"before_app_close.refs.json",r)
    if damage=="outer":change(tmp_path/"outer_exit.json","outer_exit_code",255)
    if damage=="signal":(tmp_path/"native.log").write_text('DH116_GDB_INFERIOR_STATUS {"exit_code": null, "signal": 11}\n')
    if damage=="duration":change(tmp_path/"outer_exit.json","duration_seconds",181)
    if damage=="manager":(out/"ownership.jsonl").write_text('{"phase":"SDK STOP","manager_view_released":false}\n')
    if damage=="solver":change(out/"acquisition.json","rod_tick",12)
    if damage=="close":(out/"app_close_returned.json").unlink()
    if damage=="log_hash":change(tmp_path/"outer_exit.json","log_sha256","f"*64)
    if damage=="case":change(out/"result.json","case","explicit")
    with pytest.raises((ValueError,FileNotFoundError)):
        v.validate_artifacts(out,tmp_path/"outer_exit.json","registry")

def test_failed_case_inventory_retains_both_partial_capture_and_reason(tmp_path):
    v=validator();out=evidence(tmp_path)
    (out/"gdb_finalize.json").unlink()
    result=v.audit_outcome(out,tmp_path/"outer_exit.json","registry")
    assert result["lifecycle_qualified"] is False
    assert result["reason"]
    assert "gdb_finalize.maps" in result["available_artifact_sha256"]
