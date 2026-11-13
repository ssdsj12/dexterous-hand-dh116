"""Opt-in director readback around the frozen 18-step guarded closure owner."""
from copy import deepcopy
import json
import os
from pathlib import Path

from scripts import prepare_dh116_dual_pinch_closure_interval as closure
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

VERSION="dh116-dual-pinch-director-endpoints-v1"
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,16,18
PROFILE_SHA=None
action_schedule=closure.action_schedule
sdk_clock_sample=closure.sdk_clock_sample
run_closure_existing_app=closure.run_existing_app
run_closure_owned=closure._owned_interval
inspect_closure=closure.inspect_interval
closure_reason=closure.observation_failure


def director_observation_failure(row):
    prior=closure_reason(row)
    if row["closure_guard_failure"]!=prior:
        raise ValueError("saved closure guard differs from original measured guards")
    error,packet=row["director_capture_error"],row["material_director"]
    if error is None:
        if not isinstance(packet,dict):
            raise ValueError("successful director capture requires an owned packet")
    elif not isinstance(error,str) or not error.strip() or packet is not None:
        raise ValueError("failed director capture requires error text and unknown packet")
    return prior or ("material_director_source_invalid" if error is not None else None)


class DirectorExchange(closure.ClosureExchange):
    def _step(self,action,substep):
        row=backend=None
        try:
            row=super()._step(action,substep)
            row["closure_guard_failure"]=row["guard_failure"]
            try:
                from ropesim import _ropesim as backend
                packet=self._director_reader.capture(row["rod"],rope=self._rod.rope,
                    backend=backend,sim=self._sim,read_clock=self._rod.clock)
                if not isinstance(packet,dict):
                    raise ValueError("owned director packet required")
                row["material_director"]=packet
                row["director_capture_error"]=None
            except Exception as exc:
                row["material_director"]=None
                row["director_capture_error"]=type(exc).__name__+": "+str(exc)
            row["guard_failure"]=director_observation_failure(row)
            return row
        finally:
            self=action=row=backend=None


def inspect_interval(path,*,complete=True):
    from scripts.validate_dh116_dual_pinch_director_interval import inspect_directors
    with bindings((closure,"observation_failure",director_observation_failure)):
        audit=inspect_closure(path,complete=complete)
    return dict(audit,material_director_audit=inspect_directors(path,complete=complete))


def make_reader(nodes,device):
    from go2_pvcnn.control.dh116.tabletop_material_director_readback import MaterialDirectorReadback
    return MaterialDirectorReadback(nodes,device=device)


def _write_reset(path,packet,error,identity):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    from go2_pvcnn.control.dh116.tabletop_material_director_readback import VERSION as READER_VERSION
    with Path(path).open("x") as stream:
        json.dump(dict(version=READER_VERSION,payload=encode_owned(packet),
            capture_error=error,identity=deepcopy(identity)),stream,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())


def _owned_interval(sim,owners,trace,**kwargs):
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    original_build=builder.build_reset_owned

    def build_with_directors(sim,owners,**build_args):
        rod=reader=episode=backend=reset_row=original_record=None
        saved=[]
        try:
            original_record=build_args["record"]
            def record(row):
                original_record(row)  # Persist the actual reset before any new read.
                if row["kind"]=="reset":
                    saved.append(deepcopy(row["reset"]["rod_reset"]))
            build_args["record"]=record
            result=original_build(sim,owners,**build_args)
            episode=owners.resources["dual_episode"]
            episode._record=original_record  # Proxy is no longer needed after construction.
            rod=owners.resources["runtime"]
            if len(saved)!=1:
                raise ValueError("exactly one actual completed rod reset row required")
            reset_row=saved[0]
            from ropesim import _ropesim as backend
            reader=owners.keep("material_director_reader",make_reader(rod._base.nodes,str(sim.device)))
            trace.watch("material_director_reader",reader,owner="director endpoint readback",release="owned scope after synchronize")
            error=None;packet=None
            material_uid=build_args["episode_uid"]+":material-reset-"+str(reset_row["reset_id"])
            try:
                packet=reader.capture_reset(reset_row,material_reference_uid=material_uid,
                    rope=rod.rope,backend=backend,sim=sim,read_clock=rod.clock)
                if not isinstance(packet,dict):
                    raise ValueError("owned reset director packet required")
            except Exception as exc:
                error=type(exc).__name__+": "+str(exc)
            _write_reset(Path(kwargs["output"])/"director_reset.json",packet,error,rod.clock())
            if error is not None:
                raise RuntimeError("material_director_reset_source_invalid: "+error) from None
            episode._exchange._director_reader=reader
            return result
        finally:
            sim=owners=build_args=rod=reader=episode=backend=reset_row=original_record=None

    try:
        with bindings((builder,"build_reset_owned",build_with_directors)):
            return run_closure_owned(sim,owners,trace,**kwargs)
    finally:
        sim=owners=trace=kwargs=original_build=None


def run_existing_app(*args,**kwargs):
    try:
        with bindings((closure,"VERSION",VERSION),(closure,"PROFILE_SHA",PROFILE_SHA),
                (closure,"ClosureExchange",DirectorExchange),(closure,"_owned_interval",_owned_interval),
                (closure,"inspect_interval",inspect_interval)):
            return run_closure_existing_app(*args,**kwargs)
    finally:
        args=kwargs=None
