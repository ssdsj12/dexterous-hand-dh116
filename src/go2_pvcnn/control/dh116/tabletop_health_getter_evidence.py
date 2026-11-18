"""Small post-step host-getter evidence; no live R0 or added physics."""
from copy import deepcopy
import math
import sys
from .tabletop_bimanual_table_contacts import IDENTITY,_clock
from . import tabletop_r0_health_separation as health

VERSION="dh116-same-clock-health-getter-evidence-v1"

def _require(ok,message):
    if not ok:raise ValueError(message)

def _boundary(runtime,sim):
    try:
        return dict(sdk_step=sim.current_time_step_index,sdk_time_s=float(sim.current_time),
            solver_time_s=float(runtime.rope.time),runtime_clock=_clock(runtime.clock),
            source_instance_id=id(runtime.rope))
    finally:runtime=sim=None

def validate_capture(packet,exchange):
    """Validate owned bytes only; failed observations remain failed prefix evidence."""
    _require(set(packet)=={"version","expected_clock","boundary_before","boundary_after",
        "solver_status","capture_error","native_qualification","new_sdk_steps","new_solver_steps"},
        "exact getter evidence fields required")
    expected={k:exchange["rod"][k] for k in IDENTITY}
    _require(packet["version"]==VERSION and packet["expected_clock"]==expected
        and packet["native_qualification"] is False
        and type(packet["new_sdk_steps"]) is int and packet["new_sdk_steps"]==0
        and type(packet["new_solver_steps"]) is int and packet["new_solver_steps"]==0,
        "same physical source and zero added steps required")
    error=packet["capture_error"]
    _require(error is None or isinstance(error,str) and bool(error.strip()),"explicit getter failure text required")
    if error is not None:
        return dict(source_complete=False,last_error=None,capture_error=error)
    before,after=packet["boundary_before"],packet["boundary_after"]
    _require(before==after and isinstance(before,dict) and set(before)=={
        "sdk_step","sdk_time_s","solver_time_s","runtime_clock","source_instance_id"},
        "identical actual before/after getter boundaries required")
    _require(type(before["sdk_step"]) is int and before["sdk_step"]==expected["physics_step"]
        and before["runtime_clock"]==expected and type(before["source_instance_id"]) is int
        and before["source_instance_id"]>0 and math.isfinite(before["sdk_time_s"])
        and math.isclose(before["sdk_time_s"],expected["timestamp_s"],rel_tol=0,abs_tol=2e-7)
        and before["solver_time_s"]==expected["solver_time_s"],
        "actual SDK/solver/owner boundary differs")
    status=health._verified(packet["solver_status"])
    from .tabletop_native_parameter_endpoints import VERSION as PARAM_VERSION
    anchor=exchange["material_director"]
    _require(status["version"]==health.VERSION and status["identity"]==expected
        and status["source_scope"]=="native_getter" and status["backend_sha256"]==health.BACKEND_SHA
        and status["getter"]=="RopeBatch.last_error" and status["capture_error"] is None
        and type(status["last_error"]) is int
        and status["source_instance_id"]==before["source_instance_id"]
        and status["native_readback_qualified"] is False
        and status["new_physics_steps"]==status["new_solver_steps"]==0
        and anchor["version"]==PARAM_VERSION and anchor["identity"]==expected
        and anchor["source_instance_id"]==before["source_instance_id"]
        and anchor["sources"]["backend_sha256"]==health.BACKEND_SHA
        and exchange.get("director_capture_error") is None,
        "actual getter and parameter owner/source differ")
    return dict(source_complete=True,last_error=status["last_error"],capture_error=None)

def capture_boundary(runtime,*,backend,sim,exchange):
    """Borrow actual owner, read host property once, then return only scalars/dicts."""
    packet=dict(version=VERSION,expected_clock={k:exchange["rod"][k] for k in IDENTITY},
        boundary_before=None,boundary_after=None,solver_status=None,capture_error=None,
        native_qualification=False,new_sdk_steps=0,new_solver_steps=0)
    try:
        try:
            packet["boundary_before"]=_boundary(runtime,sim)
            packet["solver_status"]=health.capture_solver_health(runtime,backend=backend,
                sim=sim,expected_clock=packet["expected_clock"])
            packet["boundary_after"]=_boundary(runtime,sim)
            status=packet["solver_status"]
            if status["capture_error"] is not None:
                packet["capture_error"]=status["capture_error"]
            else:
                validate_capture(packet,exchange)
        except Exception as caught:
            packet["capture_error"]=type(caught).__name__+": "+str(caught)
        if packet["boundary_after"] is None:
            try:packet["boundary_after"]=_boundary(runtime,sim)
            except Exception as caught:
                packet["capture_error"]=(packet["capture_error"] or "")+"; after: "+type(caught).__name__+": "+str(caught)
        return packet
    finally:runtime=backend=sim=exchange=status=None

def observation_failure(row):
    from scripts.prepare_dh116_dual_pinch_director_interval import director_observation_failure
    prior=director_observation_failure(row)
    _require(row["health_prior_guard_failure"]==prior,"saved parent physical guard differs")
    checked=validate_capture(row["health_getter"],row)
    return prior or ("health_source_invalid" if not checked["source_complete"] else
        ("numerical_health_failure" if checked["last_error"]!=0 else None))

class HealthEvidenceMixin:
    """Scoped only after the original complete physical step; no retained owners."""
    def _step(self,action,substep):
        row=backend=package=None
        try:
            row=super()._step(action,substep)
            row["health_prior_guard_failure"]=row["guard_failure"]
            backend=sys.modules.get("ropesim._ropesim")
            package=sys.modules.get("ropesim")
            if package is None or getattr(package,"_ropesim",None) is not backend:backend=None
            try:
                row["health_getter"]=capture_boundary(self._rod,backend=backend,sim=self._sim,exchange=row)
                checked=validate_capture(row["health_getter"],row)
            except Exception as caught:
                # Even an observer implementation/source fault cannot erase a completed SDK row.
                row["health_getter"]=dict(version=VERSION,
                    expected_clock={k:row["rod"][k] for k in IDENTITY},
                    boundary_before=None,boundary_after=None,solver_status=None,
                    capture_error=type(caught).__name__+": "+str(caught),
                    native_qualification=False,new_sdk_steps=0,new_solver_steps=0)
                checked=dict(source_complete=False,last_error=None)
            # The original owner records this actual row before terminal handling.
            row["guard_failure"]=row["health_prior_guard_failure"] or (
                "health_source_invalid" if not checked["source_complete"] else
                ("numerical_health_failure" if checked["last_error"]!=0 else None))
            return row
        finally:self=action=row=backend=package=None
