"""Opt-in isolated getter backend on the unchanged guarded 2+8+8 task owner."""
from pathlib import Path
from copy import deepcopy
import json
import os
from scripts import prepare_dh116_dual_pinch_director_interval as director
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as reader

VERSION="dh116-native-parameter-task-v1"
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,16,18
PROFILE_SHA=None
action_schedule=director.action_schedule
sdk_clock_sample=director.sdk_clock_sample
run_director=director.run_existing_app

def make_reader(nodes,device):
    return reader.NativeParameterEndpoints(nodes,device=device)

def write_reset(path,packet,error,identity):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    with Path(path).open("x") as stream:
        json.dump(dict(version=reader.VERSION,payload=encode_owned(packet),
            capture_error=error,identity=deepcopy(identity)),stream,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())

def inspect_interval(path,*,complete=True):
    from scripts import validate_dh116_native_parameter_task as audit
    with bindings((director.closure,"observation_failure",director.director_observation_failure)):
        checked=director.inspect_closure(path,complete=complete)
    return dict(checked,material_director_audit=audit.inspect_parameters(path,complete=complete))

def run_existing_app(*args,**kwargs):
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    try:
        with bindings((builder,"BACKEND",reader.BACKEND),(builder,"BACKEND_SHA",reader.BACKEND_SHA),
                (director,"VERSION",VERSION),(director,"PROFILE_SHA",PROFILE_SHA),
                (director,"make_reader",make_reader),(director,"_write_reset",write_reset),
                (director,"inspect_interval",inspect_interval)):
            return run_director(*args,**kwargs)
    finally:
        args=kwargs=None
