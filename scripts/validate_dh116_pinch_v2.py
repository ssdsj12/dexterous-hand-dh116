"""Independent Pinch v2 geometry/pair/quiet/clock/release/outer audit."""
import hashlib
import json
import math
from pathlib import Path
import re

from scripts.run_dh116_work_packages import debugger_status
from scripts.dh116_pinch_acquisition_v2 import OWNED_NAMES,VERSION
from scripts.dh116_pinch_evidence_v2 import require_quiet,validate_action_targets
from scripts.validate_dh116_physx_lifetime import capture_evidence

PHASES=("resources_acquired","owned_hand_and_solver_references_dropped",
        "before_stop","after_stop","before_app_close","after_app_close")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def validate_artifacts(output, outer_exit, cleanup):
    output,outer_exit=Path(output),Path(outer_exit)
    require(cleanup in ("registry","explicit"),"unknown cleanup case")
    outer=read(outer_exit); log=outer_exit.parent/"native.log"
    require(outer.get("outer_exit_code")==0 and outer.get("native_lock_observed") is True,
            "outer exit or native lock failed")
    duration=outer.get("duration_seconds")
    require(type(duration) in (int,float) and math.isfinite(duration) and 0<=duration<=180,
            "180 second qualification budget exceeded or unknown")
    require(outer.get("log_sha256")==sha(log),"outer log SHA changed")
    status=debugger_status(log)
    require(status.get("debugger_status_verified") is True and
            status.get("debugger_inferior_exit_code")==0 and status.get("debugger_inferior_signal") is None,
            "independent inferior exit is not zero")
    require(all(outer.get(k)==v for k,v in status.items()),"outer/debugger status mismatch")
    require("DH116_GDB_FATAL_SIGNAL" not in log.read_text(errors="replace")
            and not list(output.glob("gdb_fatal*")),"fatal capture present")
    boundaries=[json.loads(line) for line in (output/"initialization_boundaries.jsonl").read_text().splitlines()]
    require(len(boundaries)==2,"exactly two observed reset boundaries required")
    for row,phase,step in zip(boundaries,("before_reset","after_reset"),(0,2)):
        require(row.get("phase")==phase and row.get("sdk_step")==step
                and math.isclose(row["sdk_time_s"],step/480,rel_tol=0,abs_tol=2e-7),
                "reset boundary clock mismatch")
    result=read(output/"result.json")
    expected=dict(version=VERSION,case=cleanup,physics_steps=240,measurement_steps=238,rod_steps=238,
        ppo_updates=0,play_simulations=False,lifetime_released_before_close=True,failure=None,
        process_exit_verified=False,grasp_qualified=False,teacher_admission=False)
    require(all(k in result and result[k]==v for k,v in expected.items()),"result scope/exit identity differs")
    clocks=result["clocks"]
    require([r.get("phase") for r in clocks]==["before_stop","after_stop","before_app_close","after_app_close"],
            "lifecycle clock phases differ")
    require(clocks[0].get("clock_available") is True,"pre-STOP clock unavailable")
    for row in clocks:
        require(row.get("clock_available") in (True,False),"unknown clock availability")
        if row["clock_available"]:
            require(row.get("sdk_step")==240 and math.isclose(row["sdk_time_s"],240/480,rel_tol=0,abs_tol=2e-7),
                    "clock advanced or reset after the final measurement")
        else:
            require("sdk_step" not in row and "sdk_time_s" not in row,"unavailable clock contains fabricated values")
    acquisition=read(output/"acquisition.json")
    plan=read(output/"plan.json")
    measurement=read(output/"measurement.json")
    require(result["acquisition"]==measurement,"measurement disk/result differ")
    require(result.get("physics_clock")==measurement["physics_clock"]
        and result.get("physics_budget")==measurement["physics_budget"],"top-level physical clocks differ")
    require(all(plan.get(k)==v for k,v in acquisition.items()),"acquisition disk/plan differ")
    require(plan.get("code_config_sha256")==result.get("code_config_sha256")
            ==measurement.get("code_config_sha256"),"candidate fingerprints differ")
    required=dict(root_sim_aliases_identical=True,provider_bound_to_runtime=True,
        rod_tick=0,physics_step=2,measured_solver_step_performed=False,provider_snapshot_performed=False,
        native_environments=1,render_products_created=0,teacher_admission=False,grasp_qualified=False)
    require(all(k in acquisition and acquisition[k]==v for k,v in required.items()),"resource/solver acquisition differs")
    require(math.isfinite(acquisition["solver_time_s"]) and
            acquisition["solver_time_s"]==acquisition["solver_time_origin_s"],"rod solver clock advanced")
    before=read(output/"before_app_close.json"); closed=read(output/"app_close_returned.json")
    require(before.get("case")==cleanup and before.get("play_simulations") is False
            and before.get("lifetime_released") is True and before.get("clocks")==clocks[:3],
            "pre-close frozen evidence differs")
    require(closed==dict(case=cleanup,app_close_returned=True,process_exit_verified=False),"actual close return missing")
    ownership=[json.loads(line) for line in (output/"ownership.jsonl").read_text().splitlines()]
    stop=[row for row in ownership if row.get("phase")=="SDK STOP"]
    require(len(stop)==1 and stop[0].get("manager_view_released") is True,"manager root view survived STOP")
    liveness={}
    for phase in PHASES:
        state=read(output/(phase+".refs.json"))["resources"]
        require(set(OWNED_NAMES)<=set(state),"resource liveness missing")
        require(re.search(r"(?m)^[0-9a-f]+-[0-9a-f]+\s", (output/(phase+".maps")).read_text()),
                "phase full process map missing")
        liveness[phase]={name:state[name].get("alive") for name in OWNED_NAMES}
    require(all(v is True for v in liveness["resources_acquired"].values()),"initial resource weakrefs unavailable")
    require(all(v is False for v in liveness["before_app_close"].values()),"native owner live/unknown before close")
    require(all(v is False for v in liveness["after_app_close"].values()),"native owner live/unknown after close")
    for name,version in (("build_frame_detached.json","dh116-pinch-builder-detachment-v2"),
                         ("measurement_frame_detached.json","dh116-pinch-measurement-detachment-v2")):
        require(read(output/name)==dict(version=version,clearing_completed=True,
            operation_raised=False,native_release_verified=False),"retained native frame cleanup missing")
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import validate_pinch_ledger
    from scripts.run_dh116_work_packages import validate_native_physics_clock,native_physics_budget
    def lines(name):
        raw=(output/name).read_text()
        require(raw.endswith("\n"),"incomplete final ledger line")
        return [json.loads(line) for line in raw.splitlines()]
    rows,pairs,contacts=lines("physics.jsonl"),lines("pairs.jsonl"),lines("table_contacts.jsonl")
    validate_pinch_ledger(rows)
    project=Path(__file__).resolve().parents[1]
    cfg=read(project/"configs/dh116_tasks/flexible_pinch_candidate_v1.json")
    cfg["initial_pose"]=read(project/"configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json")
    validate_action_targets(rows,plan,cfg)
    require(len(pairs)==len(contacts)==len(rows)==238,"exact measured ledgers required")
    for row,pair,contact in zip(rows,pairs,contacts):
        require_quiet(pair,contact,row)
    measured_clock=validate_native_physics_clock(measurement,[(r["phase_step"],r["sdk_step"]) for r in rows],
        setup_steps=2,total_steps=240)
    budget=native_physics_budget(initialization_physics_steps=2,warmup_physics_steps=0,
        measurement_physics_steps=238,other_physics_advancing_steps=0)
    require(measurement["physics_budget"]==plan["physics_budget"]==budget,"physics budget differs")
    require(measurement["failure"] is None and measurement["grasp_qualified"] is False
        and measurement["full_state371_bound"] is False,"measurement scope differs")
    require(len(plan["contact_reporting"])==17 and len({r["body_path"] for r in plan["contact_reporting"]})==17
        and all(r["contact_report_enabled"] is True and r["threshold"]==0
            and r["authored_sleep_threshold_before"]==r["authored_sleep_threshold_after"]
            for r in plan["contact_reporting"]),
        "original full hand contact reporting was not configured")
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    from go2_pvcnn.control.dh116.tabletop_pinch_initial_geometry_v2 import validate_pinch_initial_geometry
    geometry=load_geometry(Path(__file__).resolve().parents[1],side="left")
    layout=geometry["reference"]["reference_layout"]
    geometry_result=validate_pinch_initial_geometry(read(output/"initial_geometry.json"),geometry,
        table_center_m=layout["table_center"],table_size_m=layout["table_size"],
        rod_nodes_w=layout["wire_points"],rod_radius_m=.005,
        minimum_table_gap_m=0.,minimum_rod_gap_m=0.)
    require(geometry_result["valid"] is True and geometry_result["initial_geometry_passed"] is True,
        "independent actual initial geometry failed")
    capture=capture_evidence(output,"finalize")
    require(type(read(output/"gdb_finalize.json").get("stopped_thread_count")) is int
        and read(output/"gdb_finalize.json")["stopped_thread_count"]>0,"all-stop capture missing")
    return dict(lifecycle_qualified=True,case=cleanup,physics_steps=240,measurement_steps=238,rod_steps=238,
                ppo_updates=0,grasp_qualified=False,teacher_admission=False,liveness=liveness,
                normal_finalize=capture,outer_exit_sha256=sha(outer_exit),physics_clock=measured_clock,
                pair_rows=238,quiet_rows=48,hand_table_contacts=0,code_config_sha256=result["code_config_sha256"])

def audit_outcome(output, outer_exit, cleanup):
    """Persist useful independent partial evidence even if this paired case fails."""
    output=Path(output)
    try:
        result=validate_artifacts(output,outer_exit,cleanup)
    except (OSError,ValueError,KeyError,TypeError,OverflowError) as error:
        result=dict(lifecycle_qualified=False,case=cleanup,reason=f"{type(error).__name__}: {error}",
                    grasp_qualified=False,teacher_admission=False)
    result["available_artifact_sha256"]={path.name:sha(path) for path in sorted(output.glob("*"))
        if path.is_file() and not path.is_symlink()}
    return result
