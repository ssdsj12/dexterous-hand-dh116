"""Independent reset/mesh/health telemetry on the unchanged projected 2+232 owner."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from scripts import prepare_dh116_health_projected_pinch as previous
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
from go2_pvcnn.control.dh116 import tabletop_mesh_observation_evidence as evidence
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,_digest
from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import IDENTITY
from scripts import validate_dh116_health_projected_pinch as health_audit

VERSION="dh116-mesh-reset-health-projected-pinch-v1"
SCOPE="mesh_reset_pose_health_and_pair_v2_2_setup_232_diagnostic_only"
ROOT=Path(__file__).resolve().parents[1]
CONFIG=previous.CONFIG
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234
PROFILE_SHA=None
load_pair=previous.load_pair
action_schedule=previous.action_schedule
sdk_clock_sample=previous.sdk_clock_sample
read_locked=previous.read_locked
_PARENT_MARKER=previous.scope_marker
_HEALTH_VALIDATE=health_audit.validate_exchange

def scope_marker():
    return dict(_PARENT_MARKER(),version=VERSION,mesh_evidence_version=evidence.VERSION,
        source_profile_version=source.VERSION,backend_sha256=source.BACKEND_SHA,
        source_tree_sha256=source.TREE_SHA,pair_consumer_version=evidence.pairs.VERSION,
        original_234_step_actions_unchanged=True,new_physics_steps=0,
        live_pair_geometry_computed=False,reset_mesh_qualified=False,
        substep_contact_frame_qualified=False,teacher_admission=False)

@contextmanager
def scenario_scope(project=ROOT):
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    from scripts import replay_dh116_native_parameter_r0 as radius_replay
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_rod_runtime as runtime
    from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import read_complete_pair_capture
    with source.source_scope(),bindings((runtime,"read_pair_capture",read_complete_pair_capture),(previous,"VERSION",VERSION),(previous,"SCOPE",SCOPE),
        (previous,"PROFILE_SHA",PROFILE_SHA),(previous,"scope_marker",scope_marker),
        (radius_replay,"BACKEND",source.BACKEND),(radius_replay,"BACKEND_SHA",source.BACKEND_SHA)):
        with previous.scenario_scope(project):
            original=director.DirectorExchange
            if issubclass(original,evidence.PairV2EvidenceMixin):owned=original
            else:
                class OwnedMeshExchange(evidence.PairV2EvidenceMixin,original):pass
                owned=OwnedMeshExchange
            with bindings((director,"DirectorExchange",owned)):yield

def run_existing_app(*args,**kwargs):
    from go2_pvcnn.control.dh116 import tabletop_deformable_dual_hand_episode as episode
    prepared=reset_type=None
    try:
        project=Path(kwargs.get("project",ROOT));output=Path(kwargs["output"])
        prepared=kwargs.get("prepared")
        if prepared is None:
            from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
            prepared=prepare_scene(project)
        reset_type=evidence.make_reset_type(project,prepared,output)
        with scenario_scope(project),bindings((episode,"DualHandResetTransaction",reset_type)):
            result=previous.run_existing_app(*args,**kwargs)
        evidence._write(output/"mesh_observation_scope.json",scope_marker())
        return result
    finally:args=kwargs=prepared=reset_type=None

def _health_view(row):
    original=deepcopy(row);original["guard_failure"]=row["mesh_observation_prior_guard_failure"]
    return _HEALTH_VALIDATE(original)

def _read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError("owned regular mesh evidence required")
    return decode_owned(json.loads(path.read_text()))

def inspect_observations(output,*,complete=True,parent_lifecycle_qualified=False):
    """Re-read real durable sources; source qualification also requires the caller's parent gate."""
    if type(complete) is not bool or type(parent_lifecycle_qualified) is not bool:
        raise ValueError("explicit complete/parent lifecycle booleans required")
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import inspect_evidence
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_readback import validate_reset_readback
    from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import validate_complete_capture,RAW_KEYS
    output=Path(output);path=output/"physical.jsonl"
    with scenario_scope():
        basis=source.source_basis();journal=inspect_evidence(path)
        if journal["physics_budget_end"]!=234 or journal["solver_substeps"]!=12:
            raise ValueError("exact 2 setup plus 232 measured scope required")
        reset_file=_read(output/evidence.RESET_FILE)
        resets=[];rows=valid=errors=0;sequence=None;owner=None;last_failed=False;pair_count=0
        reset_audit=None
        with path.open() as stream:
            for line in stream:
                entry=json.loads(line);payload=decode_owned(entry.get("payload"))
                if entry.get("operation")=="reset":
                    resets.append(payload["reset"])
                    if len(resets)!=1:raise ValueError("one actual completed reset required")
                    record=reset_file["readback"]
                    if not isinstance(record,dict):raise ValueError("completed direct mesh reset readback missing")
                    saved=record["completed_reset_receipt"]
                    actual={k:v for k,v in resets[0].items() if k!="mesh_state_readback"}
                    if _digest(saved)!=_digest(actual):raise ValueError("durable reset and direct mesh owner receipt differ")
                    captured={k:v for k,v in record.items() if k!="completed_reset_receipt"}
                    if _digest(captured)!=_digest(resets[0]["mesh_state_readback"]):
                        raise ValueError("reset journal mesh evidence differs")
                    if reset_file["version"]!=evidence.VERSION or reset_file["source_profile"]!=basis:
                        raise ValueError("reset profile identity differs")
                    reset_audit=validate_reset_readback(captured,reset_receipt=actual,mesh_plan=reset_file["mesh_plan"])
                    owner=captured["after"]["source_instance_id"]
                    identity={k:actual["rod_reset"][k] for k in IDENTITY}
                    sequence=evidence.pairs.PairObservationSequence(source_profile=basis,reset_identity=identity,
                        source_instance_id=owner,mesh_names=captured["mesh_names"],expected_substeps=12,
                        evidence_kind="native_api_return_unqualified")
                elif entry.get("operation")=="physical":
                    if sequence is None or last_failed:raise ValueError("physical sample before reset or after source terminal")
                    row=payload["exchange"];rows+=1
                    packet=row["mesh_pair_v2"];checked=evidence.validate_pair_boundary(packet,row)
                    reason=row["mesh_observation_prior_guard_failure"] or (
                        None if checked["source_complete"] else evidence.REASON)
                    if row["guard_failure"]!=reason:raise ValueError("durable v2/parent guard differs")
                    if not checked["source_complete"]:
                        errors+=1;last_failed=True
                        continue
                    if packet["boundary_before"]["source_instance_id"]!=owner:
                        raise ValueError("pair owner differs from measured mesh reset")
                    native=sequence.consume(packet["raw"],identity=packet["expected_clock"],source_instance_id=owner)
                    legacy=row["rod"]["rod_mesh_contact_pairs"]
                    validate_complete_capture(legacy)
                    if (_digest(native["legacy_capture"])!=_digest({k:v for k,v in legacy.items() if k!="complete_capture"})
                        or _digest({k:packet["raw"][k] for k in RAW_KEYS})!=_digest(legacy["complete_capture"]["raw"])):
                        raise ValueError("native v1/v2 sample generations differ")
                    valid+=1;pair_count+=packet["raw"]["attempted"]
        if rows!=journal["measured_physics_rows"]:raise ValueError("every durable physical row must be audited")
        with bindings((health_audit,"validate_exchange",_health_view)):
            health=health_audit.inspect_health(path,complete=complete)
        full=(len(resets)==1 and reset_audit is not None and rows==valid==232 and errors==0
            and journal["last_sdk_step"]==234 and journal["complete_control_windows"]==29
            and journal["settled_control_windows"]==29 and journal["terminal_reason"] is None
            and not journal["unsettled_control_window"] and health["source_complete"])
        if complete and not full:raise ValueError("all 232 original rows and direct reset/v2/health observations required")
        if complete and (_read(output/"mesh_observation_scope.json")!=scope_marker()
            or _read(output/"health_getter_scope.json")!=scope_marker()):
            raise ValueError("new source/observer scope markers differ")
        qualified=bool(full and complete and parent_lifecycle_qualified)
        return dict(version=VERSION,journal=journal,complete_observation_sequence=full,
            reset_mesh_audit=reset_audit,health_getter_audit=health,
            pair_v2_audit=dict(rows=rows,valid_packets=valid,capture_error_rows=errors,
                raw_pair_samples=pair_count,source_instance_id=owner,new_sdk_steps=0,new_solver_steps=0),
            parent_lifecycle_qualified=parent_lifecycle_qualified,
            reset_mesh_qualified=qualified,substep_contact_frame_qualified=qualified,
            same_clock_health_getter_qualified=qualified,native_qualification=False,
            contact_success_qualified=False,negative_contact_qualified=False,slip_qualified=False,teacher_admission=False)

def inspect_interval(path,*,complete=True):
    with scenario_scope():
        checked=inspect_observations(Path(path).parent,complete=complete)
        if complete:
            parent=previous.inspect_interval(path,complete=True)
            return dict(parent,mesh_observation_audit=checked)
        return dict(checked["journal"],mesh_observation_audit=checked,full_parent_audit_performed=False)
