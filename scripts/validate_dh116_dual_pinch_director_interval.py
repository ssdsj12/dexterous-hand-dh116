"""Disk-only endpoint director audit, composed with every frozen closure gate.

An endpoint quaternion packet is not substep material/contact history and cannot
qualify tangential slip, bilateral contact, grasp success or Teacher admission.
"""
from pathlib import Path
import math
import numpy as np

from scripts import validate_dh116_dual_pinch_closure_interval as closure
from scripts import validate_dh116_dual_pinch_interval as original
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_material_director_readback as reader
from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import _clock, IDENTITY
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_solver_clock import solver_clock_matches

AUDIT_NAME="dual-pinch-director-post-exit-audit.json"
SCOPE="exact_2_setup_plus_16_material_director_endpoints_only"
ERRORS=closure.ERRORS
require=original.require


def _array(value,shape,label):
    require(isinstance(value,np.ndarray) and value.dtype==np.dtype("float32")
            and value.shape==shape and np.isfinite(value).all(),
            "finite owned float32 "+label+" shape differs")
    return value


def _capture(packet,error,label):
    require((error is None and isinstance(packet,dict))
        or (isinstance(error,str) and bool(error.strip()) and packet is None),
        label+" requires either an owned packet or explicit unknown/error")


def _packet(packet,rod,sdk_time,rest,sources,*,reset,prior=None):
    identity=_clock(lambda:rod)
    original.exact(packet,dict(version=reader.VERSION,
        boundary="completed_reset_endpoint" if reset else "completed_physics_endpoint",
        padding_index=40,native_rest_lengths_readback_m=None,
        rest_source="FreeRodRuntime constructor float32 node differences; no RopeBatch rest getter",
        synchronization="explicit torch.cuda.synchronize(device), default stream, owned blocking CPU copies",
        declared_device="cuda:0",exclusive_owner_read_phase_required=True,
        substep_director_history_available=False,contact_pair_substep_alignment_verified=False,
        continuous_contact_s_m=None,r0_tangential_slip_m=None,physical_error_bound_m=None,
        native_qualification=False,teacher_admission=False,new_sdk_steps=0,new_solver_steps=0),
        "endpoint scope, rest provenance or unknown material/contact fields differ")
    require(_clock(lambda:packet["identity"])==identity and packet["identity"]==identity
        and packet["sources"]==sources,
            "same-clock identity or locked director source basis differs")
    require(type(packet["source_instance_id"]) is int and packet["source_instance_id"]>0
        and isinstance(packet["material_reference_uid"],str) and bool(packet["material_reference_uid"].strip()),
        "actual instance identity and material reference required")
    require(type(packet["solver_substeps_since_reset"]) is int
        and packet["solver_substeps_since_reset"]==identity["rod_tick"],
        "actual endpoint substep count differs")
    require(packet["real_segment_indices"]==list(range(40))
        and all(type(i) is int for i in packet["real_segment_indices"]),
        "exact forty real segment indices required")
    for key,expected in (("sdk_time_s",sdk_time),("solver_time_readback_s",identity["solver_time_s"])):
        require(type(packet[key]) in (int,float) and math.isfinite(packet[key]) and packet[key]==expected,
                "actual "+key+" differs from completed boundary")
    require(type(sdk_time) in (int,float) and math.isfinite(sdk_time)
        and math.isclose(sdk_time,identity["timestamp_s"],rel_tol=0,abs_tol=2e-7),
        "actual SDK endpoint timestamp differs")
    require(np.array_equal(_array(packet["node_positions_m"],(1,41,3),"nodes"),
                           _array(rod["node_positions"],(1,41,3),"rod nodes")),
            "director endpoint positions differ from raw rod")
    require(np.array_equal(_array(packet["constructor_rest_lengths_m"],(40,),"rest"),rest)
        and np.array_equal(_array(rod["rest_lengths_m"],(40,),"raw rod rest"),rest),
        "constructor rest material differs")
    full=_array(packet["quaternions_xyzw_with_padding"],(1,41,4),"directors including padding")
    segments=_array(packet["segment_quaternions_xyzw"],(1,40,4),"segment directors")
    padding=_array(packet["padding_quaternion_xyzw"],(1,4),"padding")
    require(np.array_equal(segments,full[:,:40]) and np.array_equal(padding,full[:,40]),
            "forty real directors/padding split differs")
    require(np.allclose(np.linalg.norm(segments,axis=-1),1.,rtol=0,atol=1e-4),
            "real segment quaternion is not unit length")
    if prior is not None:
        require(all(packet[k]==prior[k] for k in ("source_instance_id","material_reference_uid")),
                "material or actual RopeBatch instance changed")


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
    require(len(resets)==1 and len(physical)<=16,"one reset and at most sixteen endpoint rows required")
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
    full=error is None and valid==16 and not errors and not stopped
    require(not complete or full,"complete reset plus sixteen safe director endpoints required")
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


def audit_outcome(output,outer_exit,cleanup,*,profile_sha256=None):
    """Run all old gates, then require the independently replayed director packets."""
    from scripts import prepare_dh116_dual_pinch_closure_interval as prior
    from scripts import prepare_dh116_dual_pinch_director_interval as operation
    output=Path(output)
    with bindings((prior,"VERSION",operation.VERSION),(prior,"inspect_interval",operation.inspect_interval)):
        result=closure.audit_outcome(output,outer_exit,cleanup,profile_sha256=profile_sha256)
    reader_liveness={}
    for phase in original.PHASES[1:]:
        try:
            reader_liveness[phase]=original.read(output/(phase+".refs.json"))["resources"]["material_director_reader"]["alive"]
        except ERRORS:
            reader_liveness[phase]=None
    result["director_reader_liveness"]=reader_liveness
    try:
        directors=inspect_directors(output/"physical.jsonl",complete=result["lifecycle_qualified"])
        if result["lifecycle_qualified"]:
            from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import load_pair_proposal
            reset_packet=decode_owned(original.read(output/"director_reset.json")["payload"])
            expected=np.asarray(load_pair_proposal()["rod_seed"]["proposed_nodes_m"],dtype=np.float32)[None]
            require(np.array_equal(reset_packet["node_positions_m"],expected),
                    "reset director nodes differ from frozen paired constructor")
            require(all(type(value) is bool for value in reader_liveness.values())
                and all(reader_liveness[phase] is False for phase in original.PHASES[3:]),
                "explicit new reader weakref observation and release after STOP required")
            require(result["journal"]["material_director_audit"]==directors,
                    "producer director summary differs from independent disk replay")
        result["material_director_audit"]=directors
        result["material_director_audit_error"]=None
    except ERRORS as caught:
        result["lifecycle_qualified"]=False
        result["material_director_audit"]=None
        result["material_director_audit_error"]=type(caught).__name__+": "+str(caught)
        if not result.get("reason"):result["reason"]=result["material_director_audit_error"]
    result.update(qualification_scope=SCOPE,
        endpoint_director_readback_qualified=result["lifecycle_qualified"],
        same_rod_contact_qualified=False,grasp_qualified=False,teacher_admission=False,
        hand_table_contacts=None,hand_table_source_available=False,no_hand_table_contact_proven=False,
        r0_tangential_slip_m=None,continuous_contact_s_m=None,physical_error_bound_m=None,
        substep_director_history_available=False,contact_pair_substep_alignment_verified=False)
    result["available_artifact_sha256"]={p.name:original.sha(p) for p in sorted(output.glob("*"))
        if p.is_file() and not p.is_symlink() and p.name!=AUDIT_NAME}
    return result
