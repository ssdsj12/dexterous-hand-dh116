"""CPU-only audit fixtures; synthetic outer/GDB records are not native evidence."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import pytest
from test_dh116_physx_lifetime_validation import evidence as old_fixture,write
from scripts import run_dh116_work_packages as scheduler
from scripts.prepare_dh116_dual_pinch_interval import OWNED_NAMES
from scripts.dh116_pinch_startup_v3 import BOUNDARY_VERSION
from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import VERSION as CLEAR_VERSION

ROOT=Path(__file__).resolve().parents[1]
CPU=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-entry-cpu-xth6emme/cpu-fixture-0")
MODULE="scripts.validate_dh116_dual_pinch_interval"
PHASES=("dual_reset_complete","owned_hand_and_solver_references_dropped","before_stop","after_stop","before_app_close","after_app_close")
def api():return importlib.import_module(MODULE)
def change(path,key,value):
    row=json.loads(path.read_text());row[key]=value;write(path,row)

def fixture(tmp_path):
    out=old_fixture(tmp_path)
    for name in ("physical.jsonl","result.json","before_app_close.json","app_close_returned.json"):
        shutil.copyfile(CPU/name,out/name)
    result=json.loads((out/"result.json").read_text())
    standard=dict(sdk_initial_step=0,sdk_after_setup_step=2,sdk_first_measurement_step=3,
        sdk_last_measurement_step=10,sdk_final_step=10,setup_steps=2,measured_steps=8,total_steps=10)
    budget=scheduler.native_physics_budget(initialization_physics_steps=2,warmup_physics_steps=0,
        measurement_physics_steps=8,other_physics_advancing_steps=0)
    for row in (result,result["acquisition"]):
        row.update(code_config_sha256="a"*64,physics_clock=standard,physics_budget=budget)
    write(out/"result.json",result)
    (out/"sdk_clock.jsonl").write_text("".join(json.dumps(dict(phase_step=i,sdk_step=i+2))+"\n" for i in range(1,9)))
    (out/"ownership.jsonl").write_text(json.dumps(dict(phase="SDK STOP",manager_view_released=True))+"\n")
    for phase in PHASES:
        resources={n:dict(alive=(True if n=="simulation" or phase=="dual_reset_complete" else False)) for n in OWNED_NAMES}
        if phase=="dual_reset_complete":resources.pop("table_contact_source")
        write(out/(phase+".refs.json"),dict(resources=resources))
        (out/(phase+".maps")).write_text("1000-2000 r-xp 0000 00:01 123 /CPU-fixture\n")
    write(out/"simulation_clear_receipt.json",dict(version=CLEAR_VERSION,subscription_status="explicitly_unsubscribed",
        unsubscribe_returned=True,subscription_field_cleared=True,original_clear_instance_returned=True,
        extra_gc_calls=0,native_release_verified=False,simulation_liveness_after_return=None,
        constructor_frame_owner_resolved=False,physics_steps_executed=0))
    write(out/"before_app_launcher.json",dict(version=BOUNDARY_VERSION,phase="before_isaaclab_app_import",
        loaded_usd_modules=[],native_execution_performed=False))
    change(out/"gdb_finalize.json","stopped_thread_count",2)
    return out

def test_complete_fixture_preserves_live_simulation_and_only_execution_qualification(tmp_path):
    out=fixture(tmp_path)
    result=api().audit_outcome(out,tmp_path/"outer_exit.json","explicit")
    assert result["lifecycle_qualified"] is True,result
    assert result["simulation_liveness"]["before_app_close"] is True
    assert result["owned_non_sim_released"] is True and result["physics_steps"]==10
    assert result["same_rod_contact_qualified"] is result["teacher_admission"] is False
    assert result["hand_table_contacts"] is None
    assert set(result["initial_geometry"])=={"left","right"}

@pytest.mark.parametrize("damage",["outer","duration","gdb_marker","empty_threads","stop","live","unknown",
    "dishonest_sim_flag","ready","clock","code","missing_clock","boundary","seed","geometry_threshold","bad_ref_type"])
def test_rejects_wrong_execution_release_or_geometry_without_promoting_na(tmp_path,damage):
    out=fixture(tmp_path)
    if damage=="outer":change(tmp_path/"outer_exit.json","outer_exit_code",255)
    elif damage=="duration":change(tmp_path/"outer_exit.json","duration_seconds",181)
    elif damage=="gdb_marker":(out/"gdb_finalize.json").unlink()
    elif damage=="empty_threads":
        path=out/"gdb_finalize.threads.txt";path.write_text(" ")
        row=json.loads((out/"gdb_finalize.json").read_text());row["artifact_sha256"][path.name]=scheduler.file_hash(path);write(out/"gdb_finalize.json",row)
    elif damage=="stop":(out/"ownership.jsonl").write_text('{"phase":"SDK STOP","manager_view_released":false}\n')
    elif damage in ("live","unknown"):
        row=json.loads((out/"before_app_close.refs.json").read_text());row["resources"]["left_view"]["alive"]=True if damage=="live" else None;write(out/"before_app_close.refs.json",row)
    elif damage=="bad_ref_type":
        row=json.loads((out/"before_app_close.refs.json").read_text());row["resources"]["left_view"]=None;write(out/"before_app_close.refs.json",row)
    elif damage=="dishonest_sim_flag":change(out/"result.json","lifetime_released_before_close",True)
    elif damage=="ready":change(out/"simulation_clear_receipt.json","unsubscribe_returned",False)
    elif damage=="clock":(out/"sdk_clock.jsonl").write_text('{"phase_step":1,"sdk_step":99}\n')
    elif damage=="code":change(out/"result.json","code_config_sha256","b"*64)
    elif damage=="missing_clock":(out/"sdk_clock.jsonl").unlink()
    elif damage=="boundary":change(out/"before_app_launcher.json","loaded_usd_modules",["pxr"])
    elif damage in ("seed","geometry_threshold"):
        from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned
        lines=[json.loads(x) for x in (out/"physical.jsonl").read_text().splitlines()]
        for entry in lines[1:]:
            row=decode_owned(entry["payload"])
            if damage=="seed" and entry["operation"]=="reset":row["reset"]["seed"]["sha256"]="0"*64
            if damage=="geometry_threshold" and entry["operation"]=="initial_geometry":row["evidence"]["thresholds"]["minimum_table_gap_m"]=0.
            entry["payload"]=encode_owned(row)
        (out/"physical.jsonl").write_text("".join(json.dumps(x)+"\n" for x in lines))
        # Update the producer digest too: semantic checks must catch this,
        # not just an unchanged summary hash.
        from scripts.prepare_dh116_dual_pinch_interval import inspect_interval
        data=json.loads((out/"result.json").read_text())
        data["acquisition"]["journal"]=inspect_interval(out/"physical.jsonl")
        write(out/"result.json",data)
    result=api().audit_outcome(out,tmp_path/"outer_exit.json","explicit")
    assert result["lifecycle_qualified"] is False,result
    assert result["same_rod_contact_qualified"] is result["teacher_admission"] is False
    assert result["reason"] and result["hand_table_contacts"] is None
