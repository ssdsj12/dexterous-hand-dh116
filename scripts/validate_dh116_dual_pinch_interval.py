"""Independent exact 2+8 execution audit; no contact or Teacher qualification."""
import hashlib
import json
import math
from pathlib import Path
import re

from scripts.prepare_dh116_dual_pinch_interval import VERSION,OWNED_NAMES,inspect_interval
from scripts.run_dh116_work_packages import (
    debugger_status,native_artifact,native_physics_budget,validate_native_physics_clock)
from scripts.validate_dh116_physx_lifetime import capture_evidence
from scripts.dh116_pinch_startup_v3 import BOUNDARY_VERSION

AUDIT_NAME="dual-pinch-interval-post-exit-audit.json"
PHASES=("dual_reset_complete","owned_hand_and_solver_references_dropped",
        "before_stop","after_stop","before_app_close","after_app_close")
NON_SIM=tuple(n for n in OWNED_NAMES if n!="simulation")
DT=1/480


def require(value,reason):
    if not value:raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _constant(value):
    raise ValueError("nonstandard JSON constant: "+value)


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=_constant)


def rows(path):
    raw=Path(path).read_text()
    require(raw.endswith("\n"),"incomplete final ledger line")
    return [json.loads(line,parse_constant=_constant) for line in raw.splitlines()]


def exact(row,expected,reason):
    require(all(k in row and type(row[k]) is type(v) and row[k]==v for k,v in expected.items()),reason)


def _outer(output,outer_exit):
    outer=read(outer_exit);log=outer_exit.parent/"native.log"
    exact(outer,dict(outer_exit_code=0,native_lock_observed=True),"outer exit/native lock failed")
    duration=outer.get("duration_seconds")
    require(type(duration) in (int,float) and math.isfinite(duration) and 0<=duration<=180,
            "180 second qualification budget exceeded or unknown")
    require(outer.get("log_sha256")==sha(log),"outer log SHA changed")
    status=debugger_status(log)
    exact(status,dict(debugger_status_verified=True,debugger_inferior_exit_code=0,
                     debugger_inferior_signal=None),"independent inferior exit is not zero")
    require(all(outer.get(k)==v for k,v in status.items()),"outer/debugger status mismatch")
    text=log.read_text(errors="replace")
    require(not any(x in text for x in ("DH116_GDB_FATAL_SIGNAL","DH116_GDB_COMMAND_ERROR"))
            and not list(output.glob("gdb_fatal*")),"fatal/error capture present")
    for suffix in ("json","maps","bt.txt","threads.txt","registers.txt","instructions.txt","libraries.txt","mappings.txt"):
        native_artifact(output,"gdb_finalize."+suffix)
    capture=capture_evidence(output,"finalize")
    stopped=read(output/"gdb_finalize.json").get("stopped_thread_count")
    require(type(stopped) is int and stopped>0,"normal finalize all-stop capture missing")
    return dict(normal_finalize=capture,outer_exit_sha256=sha(outer_exit),duration_seconds=duration)


def _lifetime(output,result):
    clocks=result["clocks"]
    require([r.get("phase") for r in clocks]==list(PHASES[2:]),"lifecycle clock phases differ")
    require(clocks[0].get("clock_available") is True,"pre-STOP actual clock missing")
    for row in clocks:
        require(type(row.get("clock_available")) is bool,"unknown lifecycle clock availability")
        if row["clock_available"]:
            require(type(row.get("sdk_step")) is int and row["sdk_step"]==10
                and math.isclose(row["sdk_time_s"],10*DT,rel_tol=0,abs_tol=2e-7),
                "clock advanced/reset after measured interval")
        else:require("sdk_step" not in row and "sdk_time_s" not in row,"fabricated unavailable clock")
    stop=[r for r in rows(native_artifact(output,"ownership.jsonl")) if r.get("phase")=="SDK STOP"]
    require(len(stop)==1 and stop[0].get("manager_view_released") is True,"manager root view survived STOP")
    liveness={};simulation={}
    for phase in PHASES:
        state=read(native_artifact(output,phase+".refs.json"))["resources"]
        expected=set(OWNED_NAMES)-({"table_contact_source"} if phase==PHASES[0] else set())
        require(expected<=set(state),"actual owned resource liveness missing")
        require(re.search(r"(?m)^[0-9a-f]+-[0-9a-f]+\s",native_artifact(output,phase+".maps").read_text()),
                "phase process maps missing")
        liveness[phase]={name:item.get("alive") for name,item in state.items()}
        simulation[phase]=state["simulation"].get("alive")
        require(simulation[phase] is None or type(simulation[phase]) is bool,"invalid simulation liveness")
        if phase==PHASES[0]:
            require(all(state[n].get("alive") is True for n in expected),"initial actual owners not observed alive")
        if phase in PHASES[3:]:
            require(all(v is False for n,v in liveness[phase].items() if n!="simulation"),
                    "non-Sim owner live or unknown after STOP/close")
    full_released=all(v is False for v in liveness["before_app_close"].values())
    exact(result,dict(lifetime_released_before_close=full_released),"full liveness result contradicts observation")
    before=read(native_artifact(output,"before_app_close.json"))
    exact(before,dict(case="explicit",play_simulations=False,lifetime_released=full_released,
                      process_exit_verified=False),"pre-close frozen/liveness evidence differs")
    require(before["clocks"]==clocks[:3],"pre-close clock evidence differs")
    closed=read(native_artifact(output,"app_close_returned.json"))
    require(closed==dict(case="explicit",app_close_returned=True,process_exit_verified=False),"actual App close return missing")
    from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import VERSION as CLEAR_VERSION
    receipt=read(native_artifact(output,"simulation_clear_receipt.json"))
    expected=dict(version=CLEAR_VERSION,subscription_status="explicitly_unsubscribed",
        unsubscribe_returned=True,subscription_field_cleared=True,original_clear_instance_returned=True,
        extra_gc_calls=0,native_release_verified=False,simulation_liveness_after_return=None,
        constructor_frame_owner_resolved=False,physics_steps_executed=0)
    require(set(receipt)==set(expected),"PHYSICS_READY clear receipt fields differ")
    exact(receipt,expected,"PHYSICS_READY unsubscribe/original clear sequence incomplete")
    boundary=read(native_artifact(output,"before_app_launcher.json"))
    require(boundary==dict(version=BOUNDARY_VERSION,phase="before_isaaclab_app_import",
        loaded_usd_modules=[],native_execution_performed=False),"clean import boundary differs")
    return dict(liveness=liveness,simulation_liveness=simulation,owned_non_sim_released=True,
                simulation_clear_receipt_sha256=sha(output/"simulation_clear_receipt.json"),
                simulation_release_required_for_execution_qualification=False)


def _geometry(output,acquisition,decoded):
    import numpy as np
    from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import (
        load_pair_proposal,PAIR_SHA,SEED_SHA,BACKEND_SHA)
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR
    setup=acquisition["setup"];proposal=load_pair_proposal()
    exact(setup,dict(paired_evidence_sha256=PAIR_SHA,reset_seed_sha256=SEED_SHA,backend_sha256=BACKEND_SHA,
        setup_steps=2,measured_steps=0,solver_steps=0,ppo_updates=0,clearance_floor_m=.0021,
        native_qualification=False,teacher_admission=False,grasp_qualified=False),"builder paired scope differs")
    reset=next(r for op,r in decoded if op=="reset")
    require(reset["reset"]["seed"]==proposal["reset_seed"],"reset seed differs from exact paired proposal")
    require(setup["clock"]==reset["reset"]["after_clock"],"setup/reset actual clocks differ")
    exact(reset["reset"],dict(new_sdk_steps=0,new_solver_steps=0),"reset advanced SDK/solver")
    require(reset["clearance_floor_m"]==.0021 and reset["clearance_passed"] is True,
            "original 2.1mm reset floor differs")
    require(np.allclose(setup["clearance_m"],reset["clearance_m"],atol=0,rtol=0),"setup/reset clearance differs")
    for hand in reset["reset"]["hands"].values():
        require(len(hand["joint_names"])==17 and set(hand["joint_names"])==set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR),
                "original mounted 17DOF identity differs")
    geometry_rows={r["side"]:r["evidence"] for op,r in decoded if op=="initial_geometry"}
    require(set(geometry_rows)=={"left","right"},"both initial source geometries required")
    for side,record in geometry_rows.items():
        require(record["thresholds"]==dict(minimum_table_gap_m=.0021,minimum_rod_gap_m=0.),
                "original independent geometry thresholds differ")
        root="/World/DeformXDualHand/"+side+"Hand"
        require(record["actual_articulation_paths"] in ([root],[root+"/wrist_fixed"])
                and record["actual_articulation_paths"]==reset["reset"]["hands"][side]["articulation_paths"],
                "original actual articulation/reset identity differs")
    # Expensive source/hull recomputation is last; producer booleans cannot pass it.
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    from go2_pvcnn.control.dh116.tabletop_pinch_initial_geometry_v2 import validate_pinch_initial_geometry
    results={}
    for i,side in enumerate(("left","right")):
        geometry=load_geometry(Path(__file__).resolve().parents[1],side=side)
        layout=geometry["reference"]["reference_layout"];record=geometry_rows[side]
        checked=validate_pinch_initial_geometry(record,geometry,table_center_m=layout["table_center"],
            table_size_m=layout["table_size"],rod_nodes_w=np.asarray(proposal["rod_seed"]["proposed_nodes_m"],dtype=np.float32),
            rod_radius_m=proposal["rod_seed"]["radius_m"],minimum_table_gap_m=.0021,minimum_rod_gap_m=0.)
        require(checked["valid"] is True and checked["initial_geometry_passed"] is True,
                "independent "+side+" source17/locked5 geometry failed: "+str(checked["reasons"]))
        source_min=min(r["table"]["top_clearance_m"] for r in record["source_colliders"].values())
        locked_min=min(r["table"]["top_clearance_m"] for r in record["locked_pads"].values())
        require(math.isclose(source_min,float(np.asarray(setup["clearance_m"])[0,i]),abs_tol=1e-7,rel_tol=0)
            and math.isclose(locked_min,setup["locked_pad_table_clearance_m"][i],abs_tol=1e-7,rel_tol=0),
            "independent geometry/initial live clearance differs")
        results[side]=dict(checked,source17_min_table_gap_m=source_min,locked5_min_table_gap_m=locked_min)
    return results


def validate_artifacts(output,outer_exit,cleanup):
    output,outer_exit=Path(output).resolve(),Path(outer_exit).resolve()
    require(cleanup=="explicit","exact explicit cleanup required")
    outer=_outer(output,outer_exit)
    result=read(native_artifact(output,"result.json"))
    exact(result,dict(version=VERSION,case="explicit",physics_steps=10,measurement_steps=8,rod_steps=8,
        ppo_updates=0,play_simulations=False,failure=None,process_exit_verified=False,
        grasp_qualified=False,teacher_admission=False),"result exact execution scope differs")
    acquired=result["acquisition"]
    exact(acquired,dict(version=VERSION,physics_rows=8,ppo_updates=0,native_qualification=False,
        teacher_admission=False,grasp_qualified=False,hand_table_source_available=False,
        no_hand_table_contact_proven=False),"acquisition scope differs")
    code=result.get("code_config_sha256")
    require(isinstance(code,str) and re.fullmatch("[0-9a-f]{64}",code)
            and acquired.get("code_config_sha256")==code,"result/acquisition source identity differs")
    lifetime=_lifetime(output,result)
    journal=inspect_interval(native_artifact(output,"physical.jsonl"))
    require(journal==acquired["journal"],"producer journal summary differs from independent disk replay")
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    decoded=[(r["operation"],decode_owned(r["payload"])) for r in rows(output/"physical.jsonl")[1:]]
    physical=[r for op,r in decoded if op=="physical"]
    clock_rows=[(i,r["exchange"]["physics_step"]) for i,r in enumerate(physical,1)]
    clock=validate_native_physics_clock(result,clock_rows,setup_steps=2,total_steps=10)
    require(acquired["physics_clock"]==clock,"acquisition standard clock differs")
    ledger=rows(native_artifact(output,"sdk_clock.jsonl"))
    require([(r.get("phase_step"),r.get("sdk_step")) for r in ledger]==clock_rows
            and all(type(r.get("phase_step")) is int and type(r.get("sdk_step")) is int for r in ledger),
            "scheduler clock differs from durable physical rows")
    budget=native_physics_budget(initialization_physics_steps=2,warmup_physics_steps=0,
        measurement_physics_steps=8,other_physics_advancing_steps=0)
    require(result["physics_budget"]==acquired["physics_budget"]==budget,"standard all-step budget differs")
    for row in physical:
        require(row["exchange"]["wrist_projection"]["clearance_floor_m"]==.0021,
                "original physical wrist safety floor differs")
    geometries=_geometry(output,acquired,decoded)
    return dict(lifecycle_qualified=True,qualification_scope="exact_2_setup_plus_8_measured_execution_only",
        code_config_sha256=code,physics_steps=10,measurement_steps=8,rod_steps=8,ppo_updates=0,
        physics_clock=clock,physics_budget=budget,journal=journal,initial_geometry=geometries,
        same_rod_contact_qualified=False,grasp_qualified=False,teacher_admission=False,
        hand_table_contacts=None,hand_table_source_available=False,no_hand_table_contact_proven=False,
        **outer,**lifetime)


def audit_outcome(output,outer_exit,cleanup):
    output=Path(output)
    try:result=validate_artifacts(output,outer_exit,cleanup)
    except (OSError,ValueError,KeyError,TypeError,IndexError,OverflowError,RuntimeError,AttributeError) as error:
        result=dict(lifecycle_qualified=False,reason=type(error).__name__+": "+str(error))
    result.update(case=cleanup,same_rod_contact_qualified=False,grasp_qualified=False,
        teacher_admission=False,hand_table_contacts=None,hand_table_source_available=False,
        no_hand_table_contact_proven=False)
    # Preserve observed simulation liveness even if an earlier gate fails.
    result.setdefault("simulation_liveness",{})
    for phase in PHASES:
        try:result["simulation_liveness"][phase]=read(output/(phase+".refs.json"))["resources"]["simulation"].get("alive")
        except (OSError,ValueError,KeyError,TypeError,AttributeError):result["simulation_liveness"][phase]=None
    result["available_artifact_sha256"]={path.name:sha(path) for path in sorted(output.glob("*"))
        if path.is_file() and not path.is_symlink() and path.name!=AUDIT_NAME}
    return result
