"""Post-exit only: actual getter/source replay, never a live R0/training claim."""
from copy import deepcopy
from pathlib import Path
import json
from go2_pvcnn.control.dh116.tabletop_health_getter_evidence import validate_capture,observation_failure
from go2_pvcnn.control.dh116.tabletop_r0_health_separation import observe_health,HealthSeparatedR0ControlStep
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from scripts import prepare_dh116_health_projected_pinch as operation

def validate_exchange(row):
    capture=validate_capture(row["health_getter"],row)
    reason=observation_failure(row)
    if row["guard_failure"]!=reason:raise ValueError("saved health/parent terminal differs")
    observation=observe_health(row,solver_status=row["health_getter"]["solver_status"])
    return dict(capture=capture,observation=observation,terminal_reason=reason,native_qualification=False)

def inspect_health(path,*,complete=True):
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import inspect_evidence
    journal=inspect_evidence(path)
    if journal["physics_budget_end"]!=234 or journal["solver_substeps"]!=12:
        raise ValueError("exact original 234-step diagnostic budget required")
    summaries=[];owner=None;error_rows=nonzero=0
    with Path(path).open() as stream:
        for line in stream:
            entry=json.loads(line)
            if entry.get("operation")!="physical":continue
            row=decode_owned(entry["payload"])["exchange"]
            checked=validate_exchange(row);capture=checked["capture"]
            if capture["source_complete"]:
                current=row["health_getter"]["boundary_before"]["source_instance_id"]
                if owner is not None and current!=owner:raise ValueError("health getter owner changed within episode")
                owner=current
            error_rows+=int(not capture["source_complete"])
            nonzero+=int(capture["last_error"] not in (None,0))
            observation=checked["observation"]
            summaries.append(dict(identity=observation["identity"],
                numerical_health=observation["numerical_health"]["value"],
                constraint_health=observation["constraint_health"]["value"],
                hold_motion=None,last_error=capture["last_error"],
                observation_sha256=observation["sha256"],terminal_reason=checked["terminal_reason"]))
    if len(summaries)!=journal["measured_physics_rows"]:raise ValueError("all actual health rows required")
    full=(len(summaries)==232 and journal["last_sdk_step"]==234
        and journal["complete_control_windows"]==journal["settled_control_windows"]==29
        and journal["terminal_reason"] is None and not journal["unsettled_control_window"]
        and error_rows==nonzero==0 and all(s["numerical_health"] is True for s in summaries))
    if complete and not full:raise ValueError("232 complete same-owner error-zero getter rows required")
    return dict(journal=journal,rows=len(summaries),source_complete=full,
        capture_error_rows=error_rows,nonzero_error_rows=nonzero,source_instance_id=owner,
        rows_summary=summaries,live_r0_computed=False,native_qualification=False,
        hold_motion_qualified=False,teacher_admission=False,new_physics_steps=0,ppo_updates=0)

def replay_health_r0(output):
    """Existing diagnostic reducer on durable rows; unknown sources stay unknown."""
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import PhysicalR0Diagnostic
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task,R0_TERMS
    from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import BACKEND,BACKEND_SHA
    root=Path(__file__).resolve().parents[1];output=Path(output)
    class Replay(PhysicalR0Diagnostic):
        def _measurements(self,event,clock,auxiliary):
            measurements,diagnostics,na=super()._measurements(event,clock,auxiliary)
            row=event["exchange"]
            packet=row.get("health_getter")
            measurements["health_observation"]=observe_health(row,
                solver_status=None if packet is None else packet["solver_status"])
            self._health_packet_rows+=int(packet is not None)
            return measurements,diagnostics,na
    task=load_flexible_grasp_task(root/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",root)
    budget=json.loads((root/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_text())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"]=2
    budget["physics"]["solver_substeps_per_physics_step"]=12
    # Same explicit CPU diagnostic sequence as the historical offline replay;
    # this does not supply the original null training coefficients.
    weights={name:float(i+1) for i,name in enumerate(R0_TERMS)}
    reset=decode_owned(json.loads((output/"director_reset.json").read_text())["payload"])
    from scripts.replay_dh116_native_parameter_r0 import check_radius_packet
    with (output/"physical.jsonl").open() as stream:
        reset_event=next(decode_owned(entry["payload"]) for entry in map(json.loads,stream)
            if entry.get("operation")=="reset")
    checked_radius=check_radius_packet(reset,reset_event["reset"]["rod_reset"])
    radius=dict(value=checked_radius["value"],
        source="native_host_metadata:RopeBatch.parameter_snapshot.contact_radius_m:"+BACKEND_SHA)
    observer=Replay(task,"pinch",env_id=0,budget_config=budget,diagnostic_coefficients=weights,
        backend_artifact=BACKEND,backend_sha256=BACKEND_SHA,radius_sample=radius)
    observer._health_packet_rows=0
    adapted=deepcopy(task);adapted["backend"]["extension_sha256"]=BACKEND_SHA
    observer._reducer=HealthSeparatedR0ControlStep(adapted,"pinch",env_id=0,
        budget_config=budget,diagnostic_coefficients=weights)
    with (output/"physical.jsonl").open() as stream:
        next(stream)
        for line in stream:
            entry=json.loads(line);payload=decode_owned(entry["payload"])
            if entry["operation"] in ("reset","physical"):observer.record(payload)
            elif entry["operation"]=="begin":observer.begin(**payload)
            elif entry["operation"]=="end":observer.end(payload)
    report=observer.report()
    controls=[c["control_receipt"] for c in report["controls"]]
    return dict(status="completed_offline",control_windows=len(controls),
        health_packet_rows=observer._health_packet_rows,radius_source=checked_radius["source"],
        controls=controls,coefficient_scope="explicit_historical_CPU_diagnostic_sequence_not_training",
        measured_reset_approach_added=False,hold_motion_qualified=False,training_reward=None,
        live_r0_qualified=False,r0_native_reset_complete=False,native_qualification=False,
        teacher_admission=False,new_physics_steps=0,ppo_updates=0)

def audit_outcome(output,parent):
    """A failed parent or missing getter can never be hidden by a generic receipt."""
    result=dict(parent);qualified=False
    try:
        checked=inspect_health(Path(output)/"physical.jsonl",complete=bool(parent.get("lifecycle_qualified")))
        result["health_getter_audit"]=checked
        qualified=bool(parent.get("lifecycle_qualified")) and checked["source_complete"]
        if qualified:
            marker=json.loads((Path(output)/"health_getter_scope.json").read_text())
            if marker!=operation.scope_marker():raise ValueError("exact new observer scope marker required")
    except Exception as caught:
        result["health_audit_error"]=type(caught).__name__+": "+str(caught)
        qualified=False
    if qualified:
        try:result["offline_r0"]=replay_health_r0(output)
        except Exception as caught:
            # Endpoint qualification is explicitly independent of reward-domain completeness.
            result["offline_r0"]=dict(status="blocked",reason=type(caught).__name__+": "+str(caught),
                training_reward=None,live_r0_qualified=False,native_qualification=False,teacher_admission=False)
    result.update(lifecycle_qualified=qualified,same_clock_health_getter_qualified=qualified,
        qualification_scope=operation.SCOPE,teacher_admission=False,grasp_qualified=False,
        contact_success_qualified=False,hold_motion_qualified=False,live_r0_qualified=False,
        mesh_reset_qualified=False,substep_contact_frame_qualified=False)
    return result
