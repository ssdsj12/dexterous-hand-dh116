"""Independent 234-step disk replay, preserving the old scoped validators unchanged.

The three bodies retain the existing row, clock, ownership and source checks;
only the separately frozen budget/schedule and native parameter packet differ.
CPU AST-equivalence tests guard these explicit differences.
"""
from pathlib import Path
import json
import math
import re
import numpy as np
from scripts import prepare_dh116_pinch_margin_contact as operation
from scripts import prepare_dh116_dual_pinch_closure_interval as closure
from scripts import validate_dh116_dual_pinch_interval as original
from scripts import validate_dh116_native_parameter_task as parameters
from scripts.dh116_dual_pinch_lifetime_evidence import lifetime_evidence
from scripts.run_dh116_work_packages import native_artifact,native_physics_budget,validate_native_physics_clock
from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import _clock,IDENTITY
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_solver_clock import solver_clock_matches
from scripts.validate_dh116_dual_pinch_director_interval import _capture,_array
SCOPE="left_pinch_margin_2_setup_232_guarded_contact_diagnostic_only"
reader=parameters.reader
_packet=parameters.packet
require=original.require

def inspect_closure(path, *, complete=True):
    import numpy as np
    from go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard import validate_post_step_clearance
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import inspect_evidence
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, encode_owned
    from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import IDENTITY

    audit = inspect_evidence(path)
    if audit["physics_budget_end"] != operation.TOTAL_STEPS or audit["solver_substeps"] != 12:
        raise ValueError("exact frozen closure physics budget required")
    if complete and (audit["last_sdk_step"] != operation.TOTAL_STEPS or audit["measured_physics_rows"] != operation.MEASURED_STEPS
            or audit["complete_control_windows"] != 29 or audit["settled_control_windows"] != 29
            or audit["unsettled_control_window"] or audit["terminal_reason"] is not None):
        raise ValueError("twenty-nine complete safe windows required for closure execution qualification")
    geometry = operation.make_geometry(Path(__file__).resolve().parents[1])
    begins = []
    source_errors = {"geometry": 0, "table": 0}
    table_na_rows = 0
    min_by_window = [None] * 29
    contacts = {}
    table_path = "/World/DeformXDualHand/Table"
    with Path(path).open() as stream:
        next(stream)
        for line in stream:
            entry = json.loads(line)
            row = decode_owned(entry["payload"])
            if entry["operation"] == "begin":
                begins.append(row["action"])
                if int(np.asarray(row["pre_action_state"]["common"]["valid"]).sum()) != 66:
                    raise ValueError("original 66 valid common fields required")
            if entry["operation"] != "physical":
                continue
            exchange = row["exchange"]
            original_exchange = dict(exchange, guard_failure=exchange["original_guard_failure"])
            actual = validate_post_step_clearance(original_exchange)
            reason = closure.observation_failure(exchange)
            if reason != exchange["guard_failure"]:
                raise ValueError("closure terminal differs from original guard and observation results")
            for source in source_errors:
                source_errors[source] += int(exchange[source + "_capture_error"] is not None)
            if exchange["geometry_capture_error"] is None:
                geometry.validate(exchange["closure_geometry"], original_exchange)
            if encode_owned(row["hand_table_contacts"]) != encode_owned(exchange["hand_table_contacts"]):
                raise ValueError("durable table packet differs from completed exchange")
            index = exchange["policy_step"]
            gap = actual["minimum_clearance_m"]
            min_by_window[index] = gap if min_by_window[index] is None else min(gap, min_by_window[index])
            table = row["hand_table_contacts"]
            rod = exchange["rod"]
            if table is not None:
                table_na_rows += 1
            if table is not None and (table["available"] is not False or table["hands"] is not None
                    or table["total_contacts"] is not None or table["native_qualified"] is not False
                    or table["source"]["table_is_static"] is not True
                    or table["source"]["table_path"] != table_path
                    or table["source"]["table_colliders"]["collider_paths"] != [table_path + "/geometry/mesh"]
                    or table["identity"] != {key: rod[key] for key in IDENTITY}):
                raise ValueError("same-clock actual static table with unknown contact required")
            pairs = rod["rod_mesh_contact_pairs"]
            for pair in pairs["pairs"]:
                mesh = pair["mesh_name"]
                contacts[mesh] = contacts.get(mesh, 0) + 1
    schedule = operation.action_schedule().numpy()
    if (len(begins) > len(schedule) or any(not np.array_equal(actual, expected[None])
            for actual, expected in zip(begins, schedule)) or (complete and len(begins) != len(schedule))):
        raise ValueError("frozen zero then finger-only action sequence differs")
    return dict(audit, static_table_contact_na_rows=table_na_rows, observation_source_error_rows=source_errors,
                minimum_source_clearance_m_by_window=min_by_window, rod_contact_pairs_by_mesh=contacts,
                hand_table_source_available=False, no_hand_table_contact_proven=False)

def inspect_directors(path,*,complete=True):
    """Validate disk packets against reset/raw rod boundaries, including failure prefixes."""
    path=Path(path)
    saved=original.read(path.parent/"director_reset.json")
    original.exact(saved,dict(version=reader.VERSION),"director reset schema version differs")
    packet=decode_owned(saved["payload"]);error=saved["capture_error"]
    _capture(packet,error,"reset capture")
    rows=[(entry["operation"],decode_owned(entry["payload"])) for entry in original.rows(path)[1:]]
    resets=[value for op,value in rows if op=="reset"]
    physical=[value for op,value in rows if op=="physical"]
    require(len(resets)==1 and len(physical)<=232,"one reset and at most 232 endpoint rows required")
    require(next((op for op,_ in rows if op in ("reset","physical")),None)=="reset",
            "reset must precede measured endpoints")
    reset=resets[0]["reset"];rod=reset["rod_reset"];identity=_clock(lambda:rod)
    require(identity==_clock(lambda:saved["identity"])==_clock(lambda:reset["after_clock"])
        and identity==saved["identity"]==reset["after_clock"]
        and identity["env_id"]==0 and identity["reset_id"]>=1
        and identity["physics_step"]==2 and identity["rod_tick"]==0
        and rod["contact_valid"] is False,"actual fresh single-rod reset identity differs")
    actual=reset["after_actual_clocks"]
    require(type(actual["sdk_physics_step"]) is int and actual["sdk_physics_step"]==2
        and _clock(lambda:actual["rod"])==actual["rod"]==identity and actual["sdk_dt_s"]==1/480
        and type(actual["sdk_timestamp_s"]) in (int,float) and math.isfinite(actual["sdk_timestamp_s"])
        and math.isclose(actual["sdk_timestamp_s"],identity["timestamp_s"],rel_tol=0,abs_tol=2e-7),
        "actual reset SDK/solver boundary differs")
    nodes=_array(rod["node_positions"],(1,41,3),"reset nodes")
    rest=np.linalg.norm(np.diff(nodes[0],axis=0),axis=1)
    require(np.all(rest>0) and np.array_equal(_array(rod["rest_lengths_m"],(40,),"reset rest"),rest),
            "positive float32 constructor rest differs from actual reset nodes")
    sources=reader.source_basis()
    if error is None:
        _packet(packet,rod,actual["sdk_timestamp_s"],rest,sources,reset=True)
    else:
        require(not physical,"no action permitted after failed reset director capture")
    previous=identity;valid=0;errors=[];stopped=False
    for ordinal,row in enumerate(physical,1):
        require(not stopped,"measured endpoint follows terminal director/closure failure")
        exchange=row["exchange"];raw=exchange["rod"];current=_clock(lambda:raw)
        require(all(current[k]==identity[k] for k in IDENTITY[:5])
            and current["physics_step"]==ordinal+2 and current["rod_tick"]==ordinal*12
            and raw["contact_valid"] is True
            and solver_clock_matches(previous["solver_time_s"],current["solver_time_s"],steps=1),
            "one actual SDK/twelve-substep endpoint with unchanged reset identity required")
        sample=exchange["material_director"];failure=exchange["director_capture_error"]
        _capture(sample,failure,"endpoint capture")
        prior_guard=exchange["closure_guard_failure"]
        require(prior_guard is None or isinstance(prior_guard,str) and bool(prior_guard.strip()),
                "original closure guard must remain explicit")
        expected_guard=prior_guard or ("material_director_source_invalid" if failure is not None else None)
        require(exchange["guard_failure"]==expected_guard,
                "director terminal differs from original guard/capture failure")
        if failure is None:
            _packet(sample,raw,exchange["post_step_clearance"]["sdk_time_s"],rest,sources,reset=False,prior=packet)
            valid+=1
        else:errors.append(current["physics_step"])
        stopped=expected_guard is not None
        previous=current
    full=error is None and valid==232 and not errors and not stopped
    require(not complete or full,"complete reset plus 232 safe director endpoints required")
    return dict(version=reader.VERSION,reset_packet_valid=error is None,reset_capture_error=error,
        measured_physics_rows=len(physical),valid_endpoint_packets=valid,
        valid_packets=valid+int(error is None),capture_error_rows=errors,
        complete_endpoint_sequence=full,last_sdk_step=previous["physics_step"],
        last_rod_tick=previous["rod_tick"],
        source_instance_id=None if packet is None else packet["source_instance_id"],
        material_reference_uid=None if packet is None else packet["material_reference_uid"],
        source_basis=sources,reset_artifact_sha256=original.sha(path.parent/"director_reset.json"),
        substep_director_history_available=False,contact_pair_substep_alignment_verified=False,
        r0_tangential_slip_m=None,continuous_contact_s_m=None,physical_error_bound_m=None,
        native_qualification=False,teacher_admission=False)

def validate_artifacts(output,outer_exit,cleanup,*,profile_sha256):
    from scripts import prepare_dh116_pinch_margin_contact as operation
    output,outer_exit=Path(output).resolve(),Path(outer_exit).resolve()
    original.require(cleanup=="explicit","exact explicit cleanup required")
    original.require((operation.SETUP_STEPS,operation.MEASURED_STEPS,operation.TOTAL_STEPS)==(2,232,234),
                     "exact closure all-step budget required")
    outer=original._outer(output,outer_exit)
    result=original.read(native_artifact(output,"result.json"))
    original.exact(result,dict(version=operation.VERSION,case="explicit",physics_steps=234,
        measurement_steps=232,rod_steps=232,ppo_updates=0,play_simulations=False,failure=None,
        process_exit_verified=False,grasp_qualified=False,teacher_admission=False),
        "result exact guarded closure execution scope differs")
    acquisition=result["acquisition"]
    original.exact(acquisition,dict(version=operation.VERSION,physics_rows=232,ppo_updates=0,
        native_qualification=False,teacher_admission=False,grasp_qualified=False,
        hand_table_source_available=False,no_hand_table_contact_proven=False),
        "acquisition guarded closure scope differs")
    code=result.get("code_config_sha256")
    original.require(isinstance(code,str) and re.fullmatch("[0-9a-f]{64}",code)
        and acquisition.get("code_config_sha256")==code,"result/acquisition source identity differs")
    original.require(isinstance(profile_sha256,str) and re.fullmatch("[0-9a-f]{64}",profile_sha256)
        and acquisition.get("profile_sha256")==profile_sha256,
        "explicit expected frozen profile SHA and acquisition binding required")
    original.require(acquisition.get("action_schedule")==operation.action_schedule().tolist(),
                     "saved action schedule differs from frozen operation")
    lifetime=lifetime_evidence(output,result,expected_steps=234)
    journal=operation.inspect_interval(native_artifact(output,"physical.jsonl"),complete=True)
    original.require(journal==acquisition["journal"],"producer summary differs from independent journal replay")
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    decoded=[(r["operation"],decode_owned(r["payload"])) for r in original.rows(output/"physical.jsonl")[1:]]
    physical=[row for op,row in decoded if op=="physical"]
    clock_rows=[(i,row["exchange"]["physics_step"]) for i,row in enumerate(physical,1)]
    clock=validate_native_physics_clock(result,clock_rows,setup_steps=2,total_steps=234)
    original.require(acquisition["physics_clock"]==clock,"acquisition standard clock differs")
    ledger=original.rows(native_artifact(output,"sdk_clock.jsonl"))
    original.require(ledger==[operation.sdk_clock_sample(row,i) for i,row in enumerate(physical,1)]
        and all(type(row.get(key)) is int for row in ledger
                for key in ("phase_step","sdk_step","policy_step","substep")),
        "scheduler SDK/rod/post clock differs from durable measured rows")
    budget=native_physics_budget(initialization_physics_steps=2,warmup_physics_steps=0,
        measurement_physics_steps=232,other_physics_advancing_steps=0)
    original.require(result["physics_budget"]==acquisition["physics_budget"]==budget,
                     "standard all-step budget differs")
    for row in physical:
        original.require(row["exchange"]["wrist_projection"]["clearance_floor_m"]==.0021,
                         "original physical wrist safety floor differs")
    geometry=original._geometry(output,acquisition,decoded)
    return dict(lifecycle_qualified=True,qualification_scope=SCOPE,code_config_sha256=code,
        profile_sha256=profile_sha256,action_schedule=operation.action_schedule().tolist(),
        physics_steps=234,measurement_steps=232,rod_steps=232,ppo_updates=0,
        physics_clock=clock,physics_budget=budget,journal=journal,initial_geometry=geometry,
        **outer,**lifetime)
