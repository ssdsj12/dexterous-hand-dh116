"""Opt-in diagnostic separation: numerical validity, constraints, hold motion.

Successful endpoint reads are not a calibrated flexible stability measurement.
No motion threshold is adopted and no native/training qualification is granted.
The old task, guards, success standard and all training coefficients stay intact.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from .tabletop_r0_hold_necessary_domain import NecessaryHoldR0Episode,NecessaryHoldR0ControlStep
from .tabletop_deformable_r0_replay import _digest
from .tabletop_bimanual_table_contacts import _clock,IDENTITY
from .tabletop_deformable_bimanual_step import _clock as action_clock
from .tabletop_deformable_clearance_guard import validate_post_step_clearance
from .tabletop_native_parameter_endpoints import BACKEND,BACKEND_SHA
from .linear_coupling import JOINTS,ACTIVE
from .tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR

VERSION="dh116-flexible-r0-health-separated-v2"
CONFIG=Path(__file__).resolve().parents[4]/"configs/dh116_tasks/flexible_r0_health_separated_v2.json"
EXPECTED=dict(version=VERSION,enabled_by_default=False,scope="owned_measurement_diagnostic_only",
 training_reward_enabled=False,native_qualification=False,
 instability="nonfinite_recorded_state_or_explicit_native_solver_error",
 constraint_terminal="reuse_original_physical_guards_without_double_instability_penalty",
 hold_motion="independent_qualified_source_or_unknown",hold_motion_thresholds=None,
 legacy_rigid_speed_comparison_m_s=.03,legacy_rigid_speed_adopted=False,
 coefficient_source="explicit_diagnostic_only",success=dict(net_lift_m=.03,hold_s=5.,max_slip_m=.0005))
SAMPLE_KEYS=("env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s")


def _require(value,message):
    if not value:raise ValueError(message)


def contract():
    raw=CONFIG.read_bytes()
    _require(json.loads(raw)==EXPECTED,"exact opt-in health contract required")
    return dict(reward_contract_version=VERSION,reward_config_sha256=hashlib.sha256(raw).hexdigest(),
        reward_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def _seal(packet):
    packet["sha256"]=_digest(packet)
    return packet


def _verified(packet):
    copied=deepcopy(packet);saved=copied.pop("sha256",None)
    _require(saved==_digest(copied),"owned health observation digest differs")
    return copied


def make_solver_status(identity,last_error,*,source_scope,source_instance_id,backend_sha256,capture_error=None):
    """Explicit CPU fixture only; callers cannot label a supplied zero native."""
    _require(source_scope=="cpu_fixture","caller-supplied solver status is CPU fixture only")
    return _solver_status(identity,last_error,source_scope=source_scope,source_instance_id=source_instance_id,
        backend_sha256=backend_sha256,capture_error=capture_error)


def _solver_status(identity,last_error,*,source_scope,source_instance_id,backend_sha256,capture_error=None):
    """Internal serialization; actual capture still grants no qualification."""
    identity=_clock(lambda:identity)
    _require(source_scope in ("native_getter","cpu_fixture")
        and type(source_instance_id) is int and source_instance_id>0
        and backend_sha256==BACKEND_SHA,"explicit getter owner/backend identity required")
    _require(last_error is None or type(last_error) is int,"typed integer last_error or explicit unknown required")
    _require(capture_error is None or isinstance(capture_error,str) and bool(capture_error.strip()),
             "capture failure must be explicit text")
    _require(capture_error is None or last_error is None,"failed getter cannot provide a value")
    return _seal(dict(version=VERSION,identity=identity,last_error=last_error,capture_error=capture_error,
        source_scope=source_scope,source_instance_id=source_instance_id,backend_sha256=backend_sha256,
        getter="RopeBatch.last_error",native_readback_qualified=False,new_physics_steps=0,new_solver_steps=0))


def capture_solver_health(runtime,*,backend,sim,expected_clock):
    """Read the existing host getter at one owned boundary; never step or gather."""
    rope=boundary=None
    try:
        from .tabletop_dual_hand_rod_runtime import DualHandRodRuntime
        from .tabletop_deformable_grasp_runtime import FreeRodRuntime
        expected=_clock(lambda:expected_clock)
        _require(expected==expected_clock and type(runtime) is DualHandRodRuntime
            and type(runtime._base) is FreeRodRuntime,"actual runtime and exact completed clock required")
        rope=runtime.rope
        _require(Path(backend.__file__).resolve()==BACKEND.resolve()
            and hashlib.sha256(BACKEND.read_bytes()).hexdigest()==BACKEND_SHA
            and type(rope) is backend.RopeBatch and rope.num_envs==1 and rope.verts_per_rope==41,
            "locked actual single free rod getter backend required")
        def boundary():
            _require(runtime.rope is rope and type(rope) is backend.RopeBatch
                and Path(backend.__file__).resolve()==BACKEND.resolve()
                and hashlib.sha256(BACKEND.read_bytes()).hexdigest()==BACKEND_SHA,
                "backend or actual rod owner changed during health read")
            actual=_clock(runtime.clock)
            _require(actual==expected and type(sim.current_time_step_index) is int
                and sim.current_time_step_index==expected["physics_step"]
                and math.isclose(float(sim.current_time),expected["timestamp_s"],rel_tol=0,abs_tol=2e-7)
                and float(rope.time)==expected["solver_time_s"],"actual SDK/solver clock changed during health read")
            return actual,float(sim.current_time),float(rope.time)
        before=boundary()
        value=None;error=None
        try:
            value=rope.last_error
            if type(value) is not int:
                raise ValueError("actual native last_error is not an integer")
        except Exception as caught:
            value=None;error=type(caught).__name__+": "+str(caught)
        _require(boundary()==before,"health read advanced or changed clock")
        return _solver_status(expected,value,source_scope="native_getter",source_instance_id=id(rope),
            backend_sha256=BACKEND_SHA,capture_error=error)
    finally:
        runtime=backend=sim=rope=boundary=None


def observe_health(exchange,*,solver_status=None):
    """Pure owned-data diagnostics; missing getter is NA, zero guard is not stability."""
    post=exchange["action_readback"]["post"];clock=action_clock(lambda:post["identity"],(0,))[0]
    rod=exchange["rod"];rodclock=_clock(lambda:rod)
    _require(rodclock=={k:clock[k] for k in IDENTITY} and exchange["physics_step"]==clock["physics_step"],
             "health rod and actual hand boundary differ")
    _require(rod["source"]=="RopeBatch.positions/velocities/contact_forces/mesh_wrenches",
             "actual rod position/velocity source required")
    arrays={}
    for name in ("node_positions","velocities"):
        a=np.asarray(rod[name])
        _require(a.shape==(1,41,3) and a.dtype.kind=="f","actual 41-node state shape required")
        arrays["rod."+name]=a
    for side in ("left","right"):
        hand=post["hands"][side];names=hand["joint_names"]
        _require(len(names)==17 and set(names)==set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)
            and hand["active_indices"]==[names.index(n) for n in ACTIVE]
            and hand["measured_source"]=="physx_backend_get_dof_positions/get_dof_velocities",
            "actual mounted 17DOF named source required")
        for name in ("q","qd","backend_targets"):
            a=np.asarray(hand[name])
            _require(a.shape==(1,17) and a.dtype.kind=="f","finite-shape actual joint buffer required")
            arrays[side+"."+name]=a
    for side in ("left","right"):
        for name in ("forces","torques"):
            a=np.asarray(rod["hand_wrenches_com_world"][side][name])
            _require(a.shape==(1,5,3) and a.dtype.kind=="f","actual five-pad force/torque shape required")
            arrays[side+"."+name]=a
    a=np.asarray(rod["distributed_contact"])
    _require(a.shape==(1,41,3) and a.dtype.kind=="f","actual node contact force shape required")
    arrays["rod.contact_forces"]=a
    finite={name:bool(np.isfinite(a).all()) for name,a in arrays.items()}
    status=None if solver_status is None else _verified(solver_status)
    if status is not None:
        _require(status["version"]==VERSION and status["identity"]==rodclock
            and status["backend_sha256"]==BACKEND_SHA and status["getter"]=="RopeBatch.last_error"
            and status["native_readback_qualified"] is False
            and status["new_physics_steps"]==status["new_solver_steps"]==0,
            "same-clock explicit solver error source required")
        _require(status["last_error"] is None or type(status["last_error"]) is int,
                 "typed solver error required")
    owner_bound=status is not None and status["source_scope"]=="cpu_fixture"
    if status is not None and status["source_scope"]=="native_getter":
        anchor=exchange.get("material_director")
        if anchor is not None:
            from .tabletop_native_parameter_endpoints import VERSION as PARAM_VERSION
            _require(anchor["version"]==PARAM_VERSION and anchor["identity"]==rodclock
                and exchange.get("director_capture_error") is None
                and type(anchor["source_instance_id"]) is int
                and anchor["source_instance_id"]==status["source_instance_id"]
                and anchor["sources"]["backend_sha256"]==BACKEND_SHA,
                "native solver status belongs to another owner/source/boundary")
            owner_bound=True
    code=None if status is None or not owner_bound else status["last_error"]
    value=False if not all(finite.values()) or code is not None and code!=0 else (None if code is None else True)
    guard=None;constraint_value=None;constraint_error=None
    if all(finite.values()):
        # Source/clock/recorded-reason corruption is rejected, not rewarded as unknown.
        raw=dict(exchange,guard_failure=exchange.get("original_guard_failure",exchange["guard_failure"]))
        checked=validate_post_step_clearance(raw)
        guard=checked["terminal_reason"]
        constraint_value=guard is None
    final=exchange["guard_failure"]
    _require(final is None or isinstance(final,str) and bool(final.strip()),"explicit final physical terminal required")
    if guard is not None:
        _require(final==guard,"final physical terminal cannot erase the original constraint")
    capture_errors={name:exchange.get(name+"_capture_error") for name in ("geometry","table","director")}
    for capture_error in capture_errors.values():
        _require(capture_error is None or isinstance(capture_error,str) and bool(capture_error.strip()),"explicit source capture error required")
    if final is None:
        _require(not any(capture_errors.values()),"missing final physical terminal for recorded source failure")
    invalid_source=final is not None and guard is None
    terminal=dict(reason=final,original_constraint_reason=guard,
        classification="recorded_source_or_auxiliary_terminal" if invalid_source else
            ("original_constraint" if guard is not None else None),
        source_errors=capture_errors,measured_constraint_violation=guard is not None,
        reward_usable=not invalid_source)
    velocity=arrays["rod.velocities"]
    speeds=np.linalg.norm(velocity[0].astype(float),axis=1) if finite["rod.velocities"] else None
    origin=VERSION+":"+_digest(dict(contract=contract(),identity={k:clock[k] for k in IDENTITY[:5]},
        scope=None if status is None else status["source_scope"],
        instance=None if status is None else status["source_instance_id"],backend=BACKEND_SHA))
    return _seal(dict(version=VERSION,identity=deepcopy(clock),implementation=contract(),
        node_positions_sha256=_digest(rod["node_positions"]),
        raw_source_sha256=_digest(dict(rod=rod,post=post,clearance=exchange.get("post_step_clearance"))),
        finite_channels=finite,solver_status=deepcopy(solver_status),solver_owner_bound=owner_bound,
        numerical_health=dict(value=value,source=origin if value is not None else None,
            definition="finite_recorded_endpoints_and_explicit_native_error_zero"),
        physical_terminal=terminal,
        constraint_health=dict(value=constraint_value,terminal_reason=guard,source_error=constraint_error,
            source="original_joint_force_and_source_clearance_guards",native_clearance_qualified=False),
        motion=dict(max_node_speed_m_s=None if speeds is None else float(speeds.max()),
            mean_node_speed_m_s=None if speeds is None else float(speeds.mean()),
            active_joint_max_abs_qd_rad_s={s:float(np.abs(arrays[s+".qd"][0,post["hands"][s]["active_indices"]]).max())
                if finite[s+".qd"] else None for s in ("left","right")},
            hold_motion=None,adopted_thresholds=None,rigid_0_03_m_s_comparison_only=True,
            measurement_rate_hz=480,solver_substep_history_available=False),
        original_stable=None,native_readback_qualified=False,native_qualification=False,
        teacher_admission=False,new_physics_steps=0,new_solver_steps=0))


def validate_health(packet,exchange):
    """Recompute the owned observation from its original durable exchange."""
    _verified(packet)
    _require(packet==observe_health(exchange,solver_status=packet["solver_status"]),
             "health packet differs from independently recomputed source")
    return deepcopy(packet)


class HealthSeparatedR0Episode(NecessaryHoldR0Episode):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.contract=contract()

    def step(self,snapshot,measurements):
        try:
            packet=measurements.get("health_observation")
            observed=None if packet is None else _verified(packet)
            if observed is not None:
                _require(observed["version"]==VERSION and observed["implementation"]==contract()
                    and all(observed["identity"][k]==measurements["sample_identity"][k] for k in SAMPLE_KEYS)
                    and observed["node_positions_sha256"]==_digest(snapshot["rod_snapshot"]["node_positions"]),
                    "health observation belongs to a different source/state/boundary")
            health=dict(value=None,source=None) if observed is None else observed["numerical_health"]
            _require(health["value"] is None or type(health["value"]) is bool,"explicit numerical health required")
            scope=snapshot["rod_snapshot"]["source_scope"]
            status=None if observed is None else observed["solver_status"]
            if status is not None and scope!="cpu_fixture":
                _require(_verified(status)["source_scope"]=="native_getter","CPU getter cannot qualify native health")
            motion=deepcopy(measurements.get("hold_motion",dict(value=None,source=None)))
            _require(set(motion)=={"value","source"} and (motion["value"] is None or type(motion["value"]) is bool),
                     "explicit independent hold motion measurement required")
            if motion["value"] is not None:
                _require(scope=="cpu_fixture" and isinstance(motion["source"],str)
                    and motion["source"].startswith("cpu_fixture:"),"no qualified flexible hold motion source installed")
            else:_require(motion["source"] is None,"unknown motion cannot invent a source")
            old_stats=deepcopy(self.stats["instability"])
            old_measurements=deepcopy(measurements)
            old_measurements.pop("health_observation",None);old_measurements.pop("hold_motion",None)
            constraint=None if observed is None else observed["constraint_health"]
            values=(health["value"],None if constraint is None else constraint["value"],motion["value"])
            hold_ok=False if any(v is False for v in values) else (True if all(v is True for v in values) else None)
            terminal=None if observed is None else observed["physical_terminal"]
            if terminal is not None and not terminal["reward_usable"]:
                hold_ok=None  # A source failure is not a measured motion/constraint violation.
            hold_source=None if hold_ok is None else VERSION+":independent_health_constraint_motion_conjunction"
            old_measurements["stable"]=dict(value=hold_ok,source=hold_source)
            row=super().step(snapshot,old_measurements)
            row["legacy_instability"]=deepcopy(row["terms"]["instability"])
            row["diagnostic_reward_before_health_separation"]=row["diagnostic_reward"]
            raw=None if health["value"] is None else float(not health["value"])
            term=row["terms"]["instability"]
            contribution=None if raw is None or term["coefficient"] is None else -term["coefficient"]*raw
            term.update(raw=raw,valid=raw is not None,source=health["source"],contribution=contribution,
                contribution_valid=contribution is not None,applicable=True,
                applicability_source=VERSION,applicability_reason="numerical_health_separate_from_hold_motion")
            self.stats["instability"]=old_stats
            row["term_stats"]["instability"]=old_stats.update(raw,health["source"])
            row["blocked_terms"]=[k for k,t in row["terms"].items() if not t["contribution_valid"]]
            row["diagnostic_reward"]=None if row["blocked_terms"] else math.fsum(t["contribution"] for t in row["terms"].values())
            if terminal is not None and not terminal["reward_usable"]:
                row["blocked_terms"].append("physical_source_validity")
                row["diagnostic_reward"]=None
            row["health_separation"]=dict(numerical_health=deepcopy(health),hold_motion=motion,
                physical_terminal=terminal,
                constraint_health=constraint,hold_gate=dict(value=hold_ok,source=hold_source),
                observation_sha256=None if packet is None else packet["sha256"])
            row.update(version=VERSION,**self.contract)
            return row
        except Exception:
            self.active=False
            raise


class HealthSeparatedR0ControlStep(NecessaryHoldR0ControlStep):
    def __init__(self,task,grasp,*,env_id,budget_config,diagnostic_coefficients):
        super().__init__(task,grasp,env_id=env_id,budget_config=budget_config,diagnostic_coefficients=diagnostic_coefficients)
        self.contract=contract()
        self._implementation[Path(__file__).name]=self.contract["reward_source_sha256"]
        self._recipe_sha256=_digest(dict(parent_recipe_sha256=self._recipe_sha256,**self.contract))
        self._episode=HealthSeparatedR0Episode(deepcopy(task),grasp,env_id=env_id,diagnostic_coefficients=deepcopy(diagnostic_coefficients))

    def push(self,*args,**kwargs):
        ready=super().push(*args,**kwargs)
        terminal=self._rows[-1]["health_separation"]["physical_terminal"]
        reason=None if terminal is None else terminal["reason"]
        if reason is not None:
            self._reasons.append("physical_terminal:"+reason)
            self._ready=True
        return self._ready or ready

    def settle(self):
        invalid_source=any("physical_source_validity" in row["blocked_terms"] for row in (self._rows or []))
        result=super().settle()
        if invalid_source:
            result["diagnostic_reward"]=None
            result["blocked_terms"].append("physical_source_validity")
        result["physical_source_validity"]=not invalid_source
        result["version"]=VERSION
        return result
