"""Independent bounded Tripod exchange with current-instance quiet admission."""
from copy import deepcopy
import numpy as np
from scripts import prepare_dh116_tripod_owned_approach as plan
from scripts import prepare_dh116_dual_pinch_director_interval as director
from go2_pvcnn.control.dh116.tabletop_tripod_quiet_gate import TripodQuietGate, VERSION as QUIET_VERSION
from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _owned

VERSION="dh116-left-tripod-owned-native-v1"
SCOPE="left_tripod_2_setup_8quiet_200approach_24observe_diagnostic_only"
PROFILE_SHA=None
ROOT=plan.ROOT
CONFIG=plan.CONFIG
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234
load_pair=plan.load_pair
action_schedule=plan.action_schedule

def make_quiet_gate(exchange):
    # Current actual rod reset is captured/validated by the original owner.
    # Constructor nodes are read-only; they never advance or reset physics.
    return TripodQuietGate(reset_nodes_m=np.asarray(exchange._rod._base.nodes))

def observation_failure(row):
    prior=director.director_observation_failure(row)
    if row["quiet_prior_guard_failure"]!=prior:
        raise ValueError("quiet wrapper must preserve original director/geometry guards")
    packet,error=row["tripod_quiet"],row["quiet_capture_error"]
    ordinal=row["physics_step"]-2
    if (type(ordinal) is not int or not 1<=ordinal<=232
            or row["policy_step"]!=(ordinal-1)//8 or row["substep"]!=(ordinal-1)%8):
        raise ValueError("quiet wrapper clock differs")
    if prior is not None:
        if error is not None or packet!={"status":"not_evaluated","reason":"original_guard"}:
            raise ValueError("original failure must remain explicit")
        return prior
    if error is not None:
        if not isinstance(error,str) or not error.strip() or packet is not None:
            raise ValueError("failed quiet source needs error and unknown receipt")
        return "tripod_quiet_source_invalid"
    if not isinstance(packet,dict) or packet.get("version")!=QUIET_VERSION:
        raise ValueError("owned current-instance quiet receipt required")
    expected="pending" if ordinal<8 else "passed"
    if ordinal==8 and packet.get("status")=="failed":expected="failed"
    if (packet.get("status")!=expected or packet.get("samples")!=min(ordinal,8)
            or packet.get("required_samples")!=8
            or packet.get("teacher_admission") is not False
            or packet.get("grasp_qualified") is not False):
        raise ValueError("quiet receipt status/samples differs from completed budget")
    terminal="tripod_quiet_admission_failed" if expected=="failed" else None
    if packet.get("terminal_reason")!=terminal:
        raise ValueError("quiet receipt terminal differs")
    return terminal

class TripodExchange(director.DirectorExchange):
    def _step(self,action,substep):
        row=None
        try:
            if not hasattr(self,"_tripod_quiet_gate"):
                self._tripod_quiet_gate=make_quiet_gate(self)
                self._tripod_schedule=plan.action_schedule().numpy()
                self._tripod_completed=0
                self._tripod_stopped=False
            n=self._tripod_completed
            if (self._tripod_stopped or n>=232 or self._control!=n//8 or substep!=n%8):
                raise RuntimeError("Tripod bounded sequence stopped, skipped or repeated")
            actual=_owned(action)
            if actual.shape!=(1,24) or not np.array_equal(actual,self._tripod_schedule[n//8][None]):
                raise ValueError("frozen Tripod24D action differs before SDK step")
            if n>=8:self._tripod_quiet_gate.require_ready()
            row=super()._step(action,substep)
            self._tripod_completed+=1
            row["quiet_prior_guard_failure"]=row["guard_failure"]
            row["quiet_capture_error"]=None
            if row["guard_failure"] is not None:
                row["tripod_quiet"]={"status":"not_evaluated","reason":"original_guard"}
            else:
                try:
                    row["tripod_quiet"]=(self._tripod_quiet_gate.observe_exchange(row)
                        if n<8 else self._tripod_quiet_gate.require_ready())
                except Exception as exc:
                    row["tripod_quiet"]=None
                    row["quiet_capture_error"]=type(exc).__name__+": "+str(exc)
            row["guard_failure"]=observation_failure(row)
            self._tripod_stopped=row["guard_failure"] is not None
            return row
        finally:
            self=action=row=None

def make_geometry(project):
    from go2_pvcnn.control.dh116.tabletop_tripod_owned_geometry import TripodOwnedGeometry
    return TripodOwnedGeometry(project)

def inspect_interval_bound(path,*,complete=True):
    from scripts import dh116_pinch_margin_contact_audit as base
    from scripts import dh116_tripod_contact_audit as audit
    from scripts.audit_dh116_tripod_quiet_replay import inspect_quiet
    from scripts import prepare_dh116_dual_pinch_closure_interval as closure
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    quiet=inspect_quiet(path,complete=complete)
    with bindings((closure,"observation_failure",observation_failure)):
        checked=base.inspect_closure(path,complete=complete)
    return dict(checked,material_director_audit=audit.inspect_directors(path,complete=complete),
                quiet_admission_audit=quiet)

from contextlib import contextmanager
@contextmanager
def scenario_scope(project=ROOT):
    import sys,hashlib
    from pathlib import Path
    from scripts import prepare_dh116_native_parameter_task as previous
    from scripts import prepare_dh116_dual_pinch_closure_interval as closure
    from scripts import dh116_pinch_margin_contact_audit as base
    from scripts import validate_dh116_dual_pinch_closure_interval as closure_audit
    from scripts import validate_dh116_tripod_owned_contact as own_audit
    from scripts import validate_dh116_dual_pinch_director_interval as director_audit
    from scripts import dh116_tripod_contact_audit as new_audit
    from scripts.prepare_dh116_pinch_margin_contact import required_sdf_valid
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116 import tabletop_measured_closure_geometry as geometry
    p=load_pair(project);path=Path(project)/CONFIG
    with bindings((builder,"PAIR",path),(builder,"PAIR_SHA",hashlib.sha256(path.read_bytes()).hexdigest()),
            (builder,"SEED_SHA",p["reset_seed"]["sha256"]),(builder,"load_pair_proposal",lambda:load_pair(project)),
            (closure,"MEASURED_STEPS",MEASURED_STEPS),(closure,"TOTAL_STEPS",TOTAL_STEPS),
            (closure,"action_schedule",action_schedule),(closure,"make_geometry",make_geometry),
            (closure_audit,"validate_artifacts",own_audit.validate_artifacts),
            (director_audit,"inspect_directors",new_audit.inspect_directors),
            (geometry,"required_sdf_valid",required_sdf_valid),
            (director,"DirectorExchange",TripodExchange),
            (previous,"VERSION",VERSION),(previous,"PROFILE_SHA",PROFILE_SHA),
            (previous,"inspect_interval",inspect_interval_bound),
            (base,"operation",sys.modules[__name__]),(base,"SCOPE",SCOPE)):
        yield

def inspect_interval(path,*,complete=True):
    with scenario_scope():return inspect_interval_bound(path,complete=complete)

def run_existing_app(*args,**kwargs):
    from scripts import prepare_dh116_native_parameter_task as previous
    try:
        with scenario_scope(kwargs.get("project",ROOT)):
            return previous.run_existing_app(*args,**kwargs)
    finally:
        args=kwargs=None

from scripts.prepare_dh116_pinch_margin_contact import read_locked, sdk_clock_sample
