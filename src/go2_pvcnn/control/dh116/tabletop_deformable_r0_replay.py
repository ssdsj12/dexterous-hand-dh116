"""Replay real physical-owner packets through the unchanged R0 task.

This is a diagnostic data adapter, never a training source or native reset
executor. Pair load counts/rest-node labels cannot supply opposed contact or
material slip. Explicit auxiliary samples remain caller-supplied provenance.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .linear_coupling import ACTIVE, MIMIC, active_limits
from .tabletop_joint_target_readback import WRIST_LINEAR, WRIST_ANGULAR
from .tabletop_task import joint_safety_failure
from .tabletop_deformable_bimanual_step import _clock as check_clock, SIDES
from .tabletop_deformable_control_step import FlexibleR0ControlStep
from .tabletop_deformable_contact_diagnostics import summarize_contact_pairs
from .tabletop_deformable_grasp_task import load_flexible_grasp_task, RESET_HOOKS
from .tabletop_deformable_solver_clock import solver_clock_matches
from .tabletop_deformable_rod_boundary import RodBoundaryState
from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from .tabletop_teacher_mounted_wrist_source import adapt_mounted_hands

VERSION = "dh116-physical-r0-diagnostic-replay-v1"
IDENTITY = ("run_uuid","env_id","episode_uid","reset_id","rod_uid",
            "physics_step","rod_tick","timestamp_s")
AUXILIARY = ("approach_distance_m","external_support","authored_force","attachment","stable")
UNKNOWN_SLIP = ("material_reference_uid","material_s_m","material_uncertainty_m",
                "tangential_reference_uid","tangential_slip_m","tangential_uncertainty_m")


def encode_owned(value):
    """Strict JSON codec: retain NaN/Inf as explicit tags, never replace by zero."""
    if hasattr(value,"detach"):
        value=value.detach().cpu().numpy()
    if isinstance(value,np.ndarray):
        if value.dtype.kind not in "bifu":
            raise ValueError("numeric owned arrays only")
        return {"__dh116_array__":encode_owned(value.tolist()),"dtype":str(value.dtype)}
    if isinstance(value,np.generic):return encode_owned(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return {"__dh116_nonfinite__":"nan" if np.isnan(value) else "inf" if value>0 else "-inf"}
    if isinstance(value,dict):
        if any(not isinstance(k,str) for k in value):raise ValueError("explicit string dictionary keys required")
        return {k:encode_owned(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [encode_owned(v) for v in value]
    if value is None or type(value) in (str,int,float,bool):return value
    raise ValueError("native handles and arbitrary objects cannot enter replay evidence")


def decode_owned(value):
    if isinstance(value,dict):
        if set(value)=={"__dh116_nonfinite__"}:
            if value["__dh116_nonfinite__"] not in ("nan","inf","-inf"):raise ValueError("invalid nonfinite tag")
            return float(value["__dh116_nonfinite__"])
        if set(value)=={"__dh116_array__","dtype"}:
            dtype=np.dtype(value["dtype"])
            if dtype.kind not in "bifu":raise ValueError("invalid numeric array type")
            return np.array(decode_owned(value["__dh116_array__"]),dtype=dtype,copy=True)
        return {k:decode_owned(v) for k,v in value.items()}
    if isinstance(value,list):return [decode_owned(v) for v in value]
    return value


def _copy(value):
    return decode_owned(encode_owned(value))


def _digest(value):
    return hashlib.sha256(json.dumps(encode_owned(value),sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode()).hexdigest()


def _implementation():
    root=Path(__file__).resolve().parents[4]
    files=("tabletop_deformable_r0_replay.py","tabletop_deformable_control_step.py",
           "tabletop_deformable_grasp_task.py","tabletop_deformable_contact_diagnostics.py",
           "tabletop_deformx_pair_capture.py","tabletop_teacher_mounted_wrist_source.py",
           "tabletop_deformable_dual_hand_episode.py","tabletop_deformable_rod_boundary.py",
           "tabletop_task.py","tabletop_deformable_physical_exchange.py",
           "tabletop_deformable_bimanual_step.py","tabletop_deformable_solver_clock.py",
           "tabletop_stereo_rod_state.py","linear_coupling.py",
           "tabletop_joint_target_readback.py")
    return {name:hashlib.sha256((root/"src/go2_pvcnn/control/dh116"/name).read_bytes()).hexdigest() for name in files}


def _array(value):
    if hasattr(value,"detach"):value=value.detach().cpu().numpy()
    return np.asarray(value)


def _equal(left,right,label):
    if _digest(left)!=_digest(right):raise ValueError(label+" differs")


def _clock(row,env):
    return check_clock(lambda:[row],(env,))[0]


def _measurement(value=None,source=None):
    return dict(value=value,source=source)


class PhysicalR0Diagnostic:
    """Observe reset -> pre/action -> physical rows -> end, or replay this order.

    Existing FlexibleR0ControlStep owns each physical R0 evaluation and settlement.
    This object stores owned data only. It never issues an action or resets/steps
    a live owner. Failure closes it; partial windows are not silently padded.
    """
    def __init__(self,task,grasp,*,env_id,budget_config,diagnostic_coefficients,
                 backend_artifact,backend_sha256,radius_sample):
        root=Path(__file__).resolve().parents[4]
        frozen=load_flexible_grasp_task(root/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",root)
        _equal(task,frozen,"unchanged loaded task")
        artifact=Path(backend_artifact).resolve()
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=backend_sha256:
            raise ValueError("actual backend artifact SHA differs")
        if set(radius_sample)!={"value","source"}:
            raise ValueError("explicit radius value/source, including unknown, required")
        # The old task remains unchanged. Only a labelled diagnostic copy can
        # consume the current, independently file-verified backend packet.
        adapted=deepcopy(task)
        original_sha=adapted["backend"]["extension_sha256"]
        adapted["backend"]["extension_sha256"]=backend_sha256
        self._reducer=FlexibleR0ControlStep(adapted,grasp,env_id=env_id,
            budget_config=budget_config,diagnostic_coefficients=diagnostic_coefficients)
        self._inputs=dict(task=_copy(task),grasp=grasp,env_id=env_id,budget_config=_copy(budget_config),
            diagnostic_coefficients=_copy(diagnostic_coefficients),backend_artifact=str(artifact),
            backend_sha256=backend_sha256,radius_sample=_copy(radius_sample))
        self.backend_binding=dict(original_frozen_backend_sha256=original_sha,
            observed_backend_sha256=backend_sha256,artifact=str(artifact),
            differs_from_frozen_task=original_sha!=backend_sha256,
            scope="explicit_diagnostic_copy_only_not_frozen_task_admission")
        self._env,self._grasp=env_id,grasp
        self._state="new"
        self._control=0
        self._events=[]
        self._controls=[]
        self._rows=None
        self._reset_audit=None
        self._last_clock=self._last_boundary=self._pre_common=self._last_hands=None

    def _boundary(self,event,clock,raw):
        boundary=event["rod_boundary"];packet=boundary["rod_snapshot"]
        if (boundary["run_uuid"]!=clock["run_uuid"] or boundary["rod_uid"]!=clock["rod_uid"]
                or boundary["rod_tick"]!=clock["rod_tick"]
                or packet["env_ids"]!=[self._env] or packet["episode_uids"]!=[clock["episode_uid"]]
                or packet["reset_ids"]!=[clock["reset_id"]] or packet["rod_uids"]!=[clock["rod_uid"]]
                or packet["backend_sha256"]!=self._inputs["backend_sha256"]
                or any(packet[k]!=clock[k] for k in ("physics_step","rod_tick","timestamp_s"))
                or any(raw[k]!=clock[k] for k in IDENTITY)):
            raise ValueError("actual raw rod/provider/backend boundary identity differs")
        for field,expected in (("node_positions",raw["node_positions"]),
                               ("rest_lengths_m",_array(raw["rest_lengths_m"])[None])):
            if not np.array_equal(_array(packet[field]),_array(expected)):
                raise ValueError("provider snapshot differs from this actual rod row")
        if clock["rod_tick"]==0:
            if (boundary["boundary"]!="reset" or raw["contact_valid"] is not False
                    or raw["distributed_contact"] is not None
                    or not np.isnan(_array(packet["distributed_contact"])).all()
                    or _array(packet["contact_valid"]).any()):
                raise ValueError("reset contact must stay explicitly unknown")
        elif (boundary["boundary"]!="post_physics" or raw["contact_valid"] is not True
                or not np.array_equal(_array(packet["distributed_contact"]),_array(raw["distributed_contact"]))):
            raise ValueError("measured contact aggregate/source differs")
        # Reuse the actual boundary/provider computation with a read-only owned
        # clock, including all velocities and derived centerline fields.
        raw_clock={k:v for k,v in clock.items() if k not in ("policy_step","substep")}
        rebuilt=RodBoundaryState(SimpleNamespace(clock=lambda:dict(raw_clock)),
            self._inputs["backend_artifact"],self._inputs["backend_sha256"]).capture(raw)
        _equal(boundary,rebuilt,"complete actual provider reconstruction")
        return _copy(boundary)

    def _reset(self,event):
        if self._state!="new" or event["kind"]!="reset":
            raise RuntimeError("one recorded reset required before replay")
        receipt=event["reset"]
        clock=_clock(dict(receipt["after_clock"],policy_step=0,substep=0),self._env)
        if receipt["status"]!="complete" or clock["rod_tick"]!=0 or receipt["r0_reset_complete"] is not False:
            raise ValueError("completed physical receipt with explicit R0 boundary required")
        boundary=self._boundary(event,clock,receipt["rod_reset"])
        audits={}
        # These callbacks validate historical evidence. They do not execute,
        # reorder or claim the six actual native reset owners.
        def check(name):
            if name=="rod":
                self._boundary(event,clock,receipt["rod_reset"])
                status,source="record_checked","actual raw reset and provider boundary"
            elif name=="joint":
                if set(receipt["hands"])!=set(SIDES):raise ValueError("both reset hand readbacks required")
                for hand in receipt["hands"].values():
                    if any(_array(hand[k]).shape!=(1,17) or not np.isfinite(_array(hand[k])).all() for k in ("q","qd")):
                        raise ValueError("actual complete reset joint readbacks required")
                status,source="record_checked","recorded actual q/qd/target reset receipt"
            else:
                row=receipt["owner_resets"][name]
                if (row["owner"]!=name or row["episode_uid"]!=clock["episode_uid"]
                        or row["status"] not in ("cleared","not_present","unavailable") or not row["source"]):
                    raise ValueError("explicit recorded reset owner source/status required")
                status,source=row["status"],row["source"]
            audits[name]=dict(recorded_status=status,source=source,physical_reset_executed_by_replay=False)
        hooks={name:(lambda n=name:check(n)) for name in RESET_HOOKS}
        self._reducer.reset(hooks,lambda:(boundary,_copy(self._inputs["radius_sample"])))
        self._reset_audit=dict(owners=audits,
            scope="original_R0_initialization_via_record_validation_callbacks_only",
            actual_native_R0_hook_order_verified=False,r0_native_reset_complete=False)
        self._last_clock,self._last_boundary=clock,boundary
        self._state="active"

    def begin(self,pre_action_state,action):
        if self._state!="active" or self._rows is not None:
            raise RuntimeError("active settled diagnostic boundary required")
        sample=_copy(pre_action_state)
        common=sample["common"]
        rebuilt=adapt_mounted_hands(common["source_packets"],expected_identity=common["identity"])
        for key in ("values","valid"):
            if not np.array_equal(_array(common[key]),_array(rebuilt[key]),equal_nan=True):
                raise ValueError("pre-action common packet differs from original source reconstruction")
        clock=_clock(common["identity"],self._env)
        expected=dict(self._last_clock,policy_step=self._control,substep=0)
        _equal(clock,expected,"pre-action clock")
        _equal(sample["rod_boundary"],self._last_boundary,"pre-action same-boundary rod")
        action=_array(action).astype(float,copy=True)
        if action.shape!=(1,24) or not np.isfinite(action).all() or np.any(abs(action)>1):
            raise ValueError("one original normalized action24 required")
        self._reducer.begin(control_index=self._control,recipe_sha256=self._reducer.recipe_sha256,action=action[0])
        self._action=action
        self._pre_common=common
        self._rows=[]
        self._events.append(dict(op="begin",pre_action_state=sample,action=_copy(action)))

    def _measurements(self,event,clock,auxiliary):
        row=event["exchange"]["rod"]
        diagnostics=summarize_contact_pairs(row,grasp=self._grasp)
        source="validated_solver_pair_capture:"+_digest(row)
        measurements=dict(sample_identity={key:clock[key] for key in IDENTITY if key!="run_uuid"})
        measurements.update({key:_measurement() for key in (*SIDES,"table_contact","joint_limit",*AUXILIARY)})
        na={}
        pairs=row.get("rod_mesh_contact_pairs")
        for side in SIDES:
            if diagnostics["hands"] is None:
                na[side]="pair_capture_unavailable_aggregate_is_not_contact_identity"
                continue
            raw_names={p["mesh_name"].split(".",1)[1] for p in pairs["pairs"]
                       if p["mesh_name"].startswith(side+".")}
            loaded=[p for p,v in diagnostics["hands"][side]["per_pad"].items() if v["loaded_substeps"]>0]
            if len(raw_names)>=2:
                na[side]="multiple_pad_pairs_do_not_prove_effective_opposed_contact"
                continue
            # An empty or single-pad complete capture cannot meet the original
            # necessary >=2 pad condition. No positive opposition is inferred.
            value=dict(rod_uid=clock["rod_uid"],pads=loaded,opposed=False,
                       **dict.fromkeys(UNKNOWN_SLIP))
            measurements[side]=_measurement(value,source+":necessary_opposition_condition_false")
        if diagnostics["hands"] is not None:
            measurements["table_contact"]=_measurement(
                any(p["mesh_name"]=="table" for p in pairs["pairs"]),
                source+":any_table_SDF_pair_not_PhysX_hand_table")
        limits=np.stack(active_limits(),axis=-1)
        failures=[]
        for side in SIDES:
            hand=event["exchange"]["action_readback"]["post"]["hands"][side]
            names=hand["joint_names"];q=_array(hand["q"])[0]
            if len(set(names))!=17 or q.shape!=(17,) or not np.isfinite(q).all():
                raise ValueError("actual post-action named q required for original guard")
            residual=[q[names.index(f)]-(r*q[names.index(m)]+o) for f,m,r,o in MIMIC]
            failures.append(joint_safety_failure(q[[names.index(n) for n in ACTIVE]],limits,residual))
        measurements["joint_limit"]=_measurement("joint_limit" in failures,
            "actual_PhysX_q_original_joint_safety_failure_0.02rad_guard:"+_digest(event["exchange"]["action_readback"]["post"]))
        if auxiliary is not None:
            if (set(auxiliary)!={"identity","samples"} or
                    auxiliary["identity"]!={k:clock[k] for k in IDENTITY} or
                    not set(auxiliary["samples"]).issubset(AUXILIARY)):
                raise ValueError("same-clock auxiliary identity and permitted non-slip fields required")
            for name,value in auxiliary["samples"].items():
                if (not isinstance(value,dict) or set(value)!={"value","source"}
                        or (value["value"] is not None and (not isinstance(value["source"],str) or not value["source"]))):
                    raise ValueError("explicit auxiliary measurement source required")
                measurements[name]=_copy(value)
        for name in AUXILIARY:
            if measurements[name]["value"] is None:na[name]="no_bound_measurement_source"
        na["material_anchor_slip"]="discrete_loaded_rest_stations_are_not_a_physical_slip_anchor"
        return measurements,diagnostics,na

    def _physical(self,event,auxiliary):
        if self._state!="active" or self._rows is None:
            raise RuntimeError("pre-action begin required before physical evidence")
        if self._rows and self._rows[-1]["physical_guard_failure"] is not None:
            raise RuntimeError("physical data continued after its actual guard terminal")
        exchange=event["exchange"];action=exchange["action_readback"]
        substep=len(self._rows)
        for phase in ("pre","post"):
            snapshot=action[phase]
            if len(snapshot["identity"])!=1 or set(snapshot["hands"])!=set(SIDES):
                raise ValueError("exact one instance and both original hands required")
            for packet in self._pre_common["source_packets"]:
                hand=snapshot["hands"][packet["side"]]
                if (hand["prim_paths"]!=[packet["articulation_path"]]
                        or hand["joint_names"]!=packet["joint_names"]
                        or hand["active_indices"]!=[hand["joint_names"].index(n) for n in ACTIVE]
                        or hand["row_indices"]!=[0] or hand["mounted"] is not True
                        or any(_array(hand[k]).shape!=(1,17) or not np.isfinite(_array(hand[k])).all()
                               for k in ("q","qd","backend_targets"))):
                    raise ValueError("every pre/post actual hand instance and named joint state must match")
        for side in SIDES:
            for key in ("instance_object_id","view_object_id"):
                if action["pre"]["hands"][side][key]!=action["post"]["hands"][side][key]:
                    raise ValueError("actual pre/post object identity changed")
        if self._last_hands is not None:
            _equal(action["pre"]["hands"],self._last_hands,"pre-state continuity from prior post-state")
        pre=_clock(action["pre"]["identity"][0],self._env)
        post=_clock(action["post"]["identity"][0],self._env)
        expected=dict(self._last_clock,policy_step=self._control,substep=substep)
        _equal(pre,expected,"physical pre-action identity")
        for key in ("run_uuid","env_id","episode_uid","reset_id","rod_uid","policy_step","substep"):
            if post[key]!=pre[key]:raise ValueError("physical sample changed action/reset identity")
        if (post["physics_step"]!=pre["physics_step"]+1 or post["rod_tick"]<=pre["rod_tick"]
                or not solver_clock_matches(pre["solver_time_s"],post["solver_time_s"],steps=1)
                or exchange["policy_step"]!=self._control or exchange["substep"]!=substep
                or exchange["physics_step"]!=post["physics_step"]):
            raise ValueError("physical SDK/solver/action clocks differ")
        if not np.array_equal(_array(action["request"]["action24"]),self._action):
            raise ValueError("action24 differs from recorded pre-action decision")
        if substep==0:
            for packet in self._pre_common["source_packets"]:
                hand=action["pre"]["hands"][packet["side"]]
                if (hand["joint_names"]!=packet["joint_names"] or hand["prim_paths"]!=[packet["articulation_path"]]
                        or not np.array_equal(_array(hand["q"])[0],packet["q"])
                        or not np.array_equal(_array(hand["qd"])[0],packet["qd"])):
                    raise ValueError("first action pre-state differs from measured common source")
        for side in SIDES:
            hand=action["post"]["hands"][side]
            expected_target=action["submission"]["expected_backend_targets"][side]
            indices=[hand["joint_names"].index(n) for n in (*ACTIVE,*WRIST_LINEAR,*WRIST_ANGULAR)]
            if (expected_target["joint_indices"]!=indices
                    or _array(expected_target["values"]).shape!=(1,12)
                    or not np.isfinite(_array(expected_target["values"])).all()):
                raise ValueError("complete named six finger and six mount target values required")
            if not np.allclose(_array(hand["backend_targets"])[:,indices],
                               _array(expected_target["values"]),rtol=0,atol=1e-7):
                raise ValueError("post SDK target differs from submitted target")
        # This original guard method is pure: it reads only its two packet
        # arguments and never touches self, SDK, composers or solver.
        guard=FlexiblePhysicalExchange._guard_failure(None,action["post"]["hands"],
                                                      exchange["rod"]["hand_wrenches_com_world"])
        if exchange["guard_failure"]!=guard:
            raise ValueError("recorded physical guard differs from original measured guard")
        boundary=self._boundary(event,post,exchange["rod"])
        measurements,pair_diagnostics,na=self._measurements(event,post,auxiliary)
        ready=self._reducer.push(boundary,measurements,control_index=self._control,
                                 recipe_sha256=self._reducer.recipe_sha256)
        # Existing reducer already evaluated FlexibleGraspEpisode.step exactly
        # once. Copy that row; never call it again just to get a diagnostic.
        r0=_copy(self._reducer._rows[-1])
        self._rows.append(dict(measurements=measurements,pair_diagnostics=pair_diagnostics,
            unavailable_sources=na,r0=r0,original_reducer_ready=ready,physical_guard_failure=guard,
            auxiliary_scope="caller_supplied_labels_not_native_source_attestation"))
        self._last_clock,self._last_boundary=post,boundary
        self._last_hands=_copy(action["post"]["hands"])

    def record(self,event,*,auxiliary=None):
        if self._state in ("failed","closed"):raise RuntimeError("closed diagnostic cannot consume more evidence")
        event,auxiliary=_copy(event),_copy(auxiliary)
        try:
            if event["kind"]=="reset":
                if auxiliary is not None:raise ValueError("reset auxiliary samples are not physical measurements")
                self._reset(event)
            elif event["kind"]=="physical":self._physical(event,auxiliary)
            else:raise ValueError("only original physical owner reset/physical records accepted")
        except Exception:
            self._state="failed"
            raise
        self._events.append(dict(op="record",event=event,auxiliary=auxiliary))

    def end(self,control_result):
        if self._state!="active" or not self._rows:
            raise RuntimeError("one nonempty pending physical window required")
        result=_copy(control_result)
        if (result["control_step"]!=self._control or result["physical_samples"]!=len(self._rows)
                or result["training_reward"] is not None or result["teacher_admission"] is not False
                or result["termination_reason"]!=self._rows[-1]["physical_guard_failure"]):
            self._state="failed"
            raise ValueError("actual control result differs from collected window")
        complete=self._rows[-1]["original_reducer_ready"]
        if not complete and result["termination_reason"] is None:
            raise RuntimeError("incomplete eight-sample window cannot settle")
        receipt=self._reducer.settle() if complete else None
        blocker=None if complete else "physical_guard_closed_partial_window_without_original_R0_terminal_source"
        output=dict(control_receipt=receipt,physical_rows=_copy(self._rows),control_result=result,
            diagnostic_backend_binding=deepcopy(self.backend_binding),reset_replay_audit=_copy(self._reset_audit),
            r0_native_reset_complete=False,teacher_admission=False,native_qualification=False,
            training_reward=None,settlement_blocker=blocker,
            approach_first_sample_is_unavailable=True,material_slip_without_anchor_is_unavailable=True)
        self._controls.append(output)
        self._events.append(dict(op="end",control_result=result))
        self._control+=1
        self._rows=None
        self._state="closed" if (result["termination_reason"] or blocker
                                or (receipt and (receipt["terminated"] or receipt["truncated"]))) else "active"
        return _copy(output)

    def report(self):
        return dict(version=VERSION,scope="physical_data_R0_diagnostic_only",
            implementation_sha256=_implementation(),controls=_copy(self._controls),
            diagnostic_backend_binding=deepcopy(self.backend_binding),reset_replay_audit=_copy(self._reset_audit),
            recorded_control_windows=len(self._controls),
            settled_control_windows=sum(c["control_receipt"] is not None for c in self._controls),
            native_qualification=False,teacher_admission=False,
            training_reward=None,new_physics_steps=0,optimizer_updates=0,
            frozen_task_mutated=False,full_371d_bound=False)

    def bundle(self):
        return dict(version=VERSION,implementation_sha256=_implementation(),
                    inputs=_copy(self._inputs),events=_copy(self._events))


def replay_diagnostic_bundle(bundle):
    if bundle["version"]!=VERSION or bundle["implementation_sha256"]!=_implementation():
        raise ValueError("diagnostic replay source/version differs")
    observer=PhysicalR0Diagnostic(**_copy(bundle["inputs"]))
    for row in bundle["events"]:
        if row["op"]=="record":observer.record(row["event"],auxiliary=row["auxiliary"])
        elif row["op"]=="begin":observer.begin(row["pre_action_state"],row["action"])
        elif row["op"]=="end":observer.end(row["control_result"])
        else:raise ValueError("unknown replay operation")
    if observer._rows is not None:raise ValueError("replay ended with an unsettled physical window")
    return observer.report()
