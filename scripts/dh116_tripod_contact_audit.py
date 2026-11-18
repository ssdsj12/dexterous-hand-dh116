"""Tripod director validator: original checks plus explicit quiet terminal layer."""
from scripts.dh116_pinch_margin_contact_audit import (
    Path, math, np, original, _capture, _array, require, decode_owned,
    _clock, IDENTITY, solver_clock_matches, reader, _packet)
from scripts import prepare_dh116_tripod_native_interval as operation

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
        require(exchange["quiet_prior_guard_failure"]==expected_guard,
                "quiet layer must retain original material/geometry guard")
        expected_guard=operation.observation_failure(exchange)
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
