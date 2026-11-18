"""Independent force-gated N=1 diagnostics around a frozen candidate Teacher.

No PPO optimization, expert admission or native success claim is implemented.
Raw policy action is kept separately from the projected, actually issued action.
CPU phase receipts do not qualify the physical source or final fixed-anchor slip.
"""
import ast
from dataclasses import asdict
import math
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .tabletop_native_phase_contract_v2 import PhaseOwnerV2, PhaseSampleV2
from .tabletop_teacher_supported_runtime import compile_supported_runtime

DT=1/480
JOINT_GUARD_MARGIN_RAD=.02


def compile_force_runtime(source, expected_sha):
    """SHA-bound opt-in adapter; never edits the source or enables training."""
    tree=compile_supported_runtime(source,expected_sha)
    factory=tree.body[0]
    factory.args.args.append(ast.arg(arg="force_hook"))
    factory.body.insert(0,ast.parse("force_hook.validate_arguments(args)").body[0])
    counts=dict(capture=0,before=0,after=0,reset=0,table=0,spawn=0,assets=0)
    for node in ast.walk(factory):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and node.func.attr=="Set" and len(node.args)==1
                and isinstance(node.args[0],ast.Constant) and node.args[0].value is False
                and isinstance(node.func.value,ast.Call)
                and isinstance(node.func.value.func,ast.Attribute)
                and node.func.value.func.attr=="GetCollisionEnabledAttr"):
            # The legacy fast candidate disabled PhysX table collision. This NEW
            # diagnostic task explicitly retains it; old source remains frozen.
            node.args[0]=ast.Constant(True);counts["table"]+=1
    # Only N=1 is allowed. Reuse the scene-owned original hand assets and
    # author the selected S0 before the first sim.reset(), not after two steps.
    for node in ast.walk(factory):
        if (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Name) and node.value.func.id=="spawn_scene"):
            node.value.func.id="spawn_force_scene"
            node.value.keywords.append(ast.keyword(arg="seed",
                value=ast.Subscript(value=ast.Name(id="preview_seeds",ctx=ast.Load()),
                    slice=ast.Constant(0),ctx=ast.Load())))
            # Expr -> Assign in its parent is handled by the transformer below.
            counts["spawn"]+=1
    class SelectAssets(ast.NodeTransformer):
        def visit_Expr(self,node):
            if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                    and node.value.func.id=="spawn_force_scene"):
                return ast.Assign(targets=[ast.Name(id="selected_scene",ctx=ast.Store())],value=node.value)
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                    and node.value.func.id=="_make_art"):
                side=node.value.args[0].value
                node.value=ast.parse("selected_scene['hands']["+str(0 if side=="left" else 1)+"]",
                    mode="eval").body
                counts["assets"]+=1
            return self.generic_visit(node)
    factory=SelectAssets().visit(factory)
    tree.body[0]=factory
    out=[]
    for node in factory.body:
        out.append(node)
        if (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Attribute)
                and isinstance(node.value.func.value,ast.Name)
                and node.value.func.value.id=="rope"
                and node.value.func.attr=="enable_contact_forces"):
            out.extend(ast.parse("rope.enable_contact_pair_capture(force_hook.pair_capacity)").body)
            counts["capture"]+=1
        if isinstance(node,ast.FunctionDef) and node.name=="reset_all":
            node.body.extend(ast.parse("force_hook.reset_episode(dict(force_state, **locals()))").body);counts["reset"]+=1
        if isinstance(node,ast.FunctionDef) and node.name=="step_env":
            # Keep raw input immutable across all eight substeps. Reusing the
            # previous masked act would prevent motion after a mid-control release.
            a=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=="act" for t in n.targets))
            node.body[a+1:a+1]=ast.parse(
                """source_action=act.clone()
force_hook.begin_control(actions,dict(force_state, **locals()))""").body
            loop=next(n for n in node.body if isinstance(n,ast.For))
            loop.body.insert(0,ast.parse(
                "act=force_hook.before_substep(source_action,dict(force_state, **locals()))").body[0])
            counts["before"]+=1
            j=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.If)
                and isinstance(n.test,ast.Attribute) and n.test.attr=="last_error")
            loop.body[j:j]=ast.parse("force_hook.after_substep(act,dict(force_state, **locals()))").body
            handler=ast.ExceptHandler(type=ast.Name(id="BaseException",ctx=ast.Load()),
                name="physical_error",body=ast.parse("""
force_hook.fail_substep(physical_error,dict(force_state, **locals()))
raise
""").body)
            loop.body=[ast.Try(body=loop.body,handlers=[handler],orelse=[],finalbody=[])]
            counts["after"]+=1
    # Snapshot stable native identities only. Current nonlocal command tensors
    # are overlaid from each callback's locals, since the source rebinds them.
    pos=next(i for i,n in enumerate(out) if isinstance(n,ast.Return))
    out[pos:pos]=ast.parse("""force_state=dict(N=N,args=args,sim=sim,rope=rope,
        rope_pos_view=rope_pos_view,left_hands=left_hands,right_hands=right_hands,
        l_base_id=l_base_id,r_base_id=r_base_id,l_wrist_ids=l_wrist_ids,r_wrist_ids=r_wrist_ids)
""").body
    # Observe the first setup result before cfg_rod/seed_q/post-reset writes.
    pos=next(i for i,n in enumerate(out) if isinstance(n,ast.Assign)
        and any(isinstance(v,ast.Name) and v.id=="cfg_rod" for v in n.targets))
    out[pos:pos]=ast.parse("""
initialization_readback=audit_seed_after_setup(locals(),preview_seeds[0])
force_hook.accept_initialization(initialization_readback)
""").body
    factory.body=out
    if counts!=dict(capture=1,before=1,after=1,reset=1,table=1,spawn=1,assets=2):
        raise ValueError("force diagnostic source layout differs: "+str(counts))
    return ast.fix_missing_locations(tree)


def _array(value):
    if hasattr(value,"detach"):
        value=value.detach().cpu().numpy()
    return np.array(value,dtype=float,copy=True)


def _json_value(value):
    return _array(value).tolist()


def _encode_nonfinite(value):
    if isinstance(value,dict):
        return {k:_encode_nonfinite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_encode_nonfinite(v) for v in value]
    if isinstance(value,np.ndarray):
        return _encode_nonfinite(value.tolist())
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value>0 else "-Infinity")
    if isinstance(value,np.generic):
        return value.item()
    return value


class ForceDiagnosticLedger:
    """Exclusive JSONL writer: every complete source row is flushed and fsynced."""
    def __init__(self,path):
        self._stream=Path(path).open("x",encoding="utf-8")
        self.rows_written=0

    def __call__(self,row):
        encoded=_encode_nonfinite(row)
        encoded["nonfinite_encoding"]="tagged_non_measurements"
        line=json.dumps(encoded,allow_nan=False,separators=(",",":"))
        self._stream.write(line+"\n")
        self._stream.flush();os.fsync(self._stream.fileno())
        self.rows_written+=1

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self._stream.close()


class ForceContactHook:
    """Bounded caller-owned source observer, with fail-closed wrist gating.

    The emit callback must durably save each row in the native worker. This
    observer deliberately cannot certify final success: contact slip, Power
    palm/proximal qualification and natural shutdown are still separate gates.
    """
    pair_capacity=32768

    def __init__(self, *, grasp, run_uid, emit, prelift_stable_s=.25):
        if grasp not in ("pinch","tripod","power") or not isinstance(run_uid,str) or not run_uid.strip():
            raise ValueError("explicit grasp/run identity required")
        if not callable(emit):
            raise ValueError("physical ledger emit callback required")
        self.grasp,self.run_uid,self.emit=grasp,run_uid,emit
        self.owner=PhaseOwnerV2(prelift_stable_s=prelift_stable_s)
        self._reset_count=0
        self._faulted=False
        self._ready=False
        self._hold=None
        self._previous_geometry=None
        self._episode_uid=None

    def accept_initialization(self,receipt):
        self.emit(dict(receipt,run_uid=self.run_uid))
        if not receipt["passed"]:
            raise RuntimeError("selected S0 failed actual initialization readback")

    def validate_arguments(self,args):
        if (args.num_envs!=1 or args.updates!=0 or args.eval_only is not True
                or args.mode!=self.grasp):
            raise ValueError("force diagnostic requires selected style, N=1, zero updates, eval_only")

    def _check_identity(self,state):
        identities=tuple(id(state[k]) for k in ("sim","rope","left_hands","right_hands"))
        if hasattr(self,"_binding") and identities!=self._binding:
            raise ValueError("native environment identity changed")
        return identities

    def reset_episode(self,state):
        identities=self._check_identity(state)
        if state["N"]!=1:
            raise ValueError("pair capture is restricted to one rod environment")
        self._binding=identities
        self._reset_count+=1
        self._episode_uid=f"{self.run_uid}/env0/reset{self._reset_count}"
        self._tick=int(state["sim"].current_time_step_index)
        self._sdk_time=float(state["sim"].current_time)
        self._rod_time=float(state["rope"].time)
        self._rod_tick=0
        self._policy_step=0
        self._substep=0
        self.owner.reset(episode_uid=self._episode_uid,physics_tick=self._tick)
        self._faulted=False;self._ready=False;self._hold=None
        self._previous_geometry=None
        self.emit(dict(event="episode_reset",episode_uid=self._episode_uid,
            env_id=0,physics_step=self._tick,timestamp=self._sdk_time,
            solver_time_s=self._rod_time,native_success=False))

    def begin_control(self,source_action,state):
        if self._episode_uid is None or self._faulted:
            raise RuntimeError("reset required or diagnostic faulted")
        if state["N"]!=1 or tuple(source_action.shape)!=(1,24):
            raise ValueError("one 24D source action required")
        a=_array(source_action)
        if not np.isfinite(a).all():
            raise ValueError("finite policy action required")
        self._source_action=a
        self._issued=None
        self._control_action=np.clip(a,-1,1)
        self._policy_step+=1
        self._substep=0

    def before_substep(self,source_action,state):
        if self._faulted:
            raise RuntimeError("force diagnostic faulted; no subsequent action allowed")
        if self._episode_uid is None:
            raise RuntimeError("reset required before physical actions")
        self._check_identity(state)
        if (int(state["sim"].current_time_step_index)!=self._tick
                or not math.isclose(float(state["sim"].current_time),self._sdk_time,abs_tol=1e-12,rel_tol=0)
                or not math.isclose(float(state["rope"].time),self._rod_time,abs_tol=1e-12,rel_tol=0)):
            raise RuntimeError("physics clock changed outside this observer")
        if self._substep>=8:
            raise RuntimeError("policy control exceeded eight physical substeps")
        if not np.array_equal(_array(source_action),self._control_action):
            raise ValueError("source action changed within policy control")
        issued=source_action.clone()
        if not self._ready:
            if self._hold is None:
                self._hold={}
                for prefix,hand_key,base_key,wrist_key in (
                    ("l","left_hands","l_base_id","l_wrist_ids"),
                    ("r","right_hands","r_base_id","r_wrist_ids")):
                    data=state[hand_key].data
                    self._hold[prefix+"_wrist_pos"]=data.body_link_pos_w[:,state[base_key]].clone()
                    self._hold[prefix+"_wrist_ang"]=data.joint_pos[:,state[wrist_key][3:6]].clone()
            for key,value in self._hold.items():
                state[key].copy_(value)
            issued[:,12:24]=0
        else:
            self._hold=None
        self._issued=_array(issued)
        self._substep+=1
        return issued

    def _hands(self,state):
        result={}
        for side,short in (("left","l"),("right","r")):
            data=state[side+"_hands"].data
            result[side]=dict(
                joint_names=list(state[side+"_hands"].joint_names),
                joint_position_rad=_json_value(data.joint_pos),
                joint_velocity_rad_s=_json_value(data.joint_vel),
                joint_limits_rad=_json_value(data.joint_pos_limits),
                body_link_position_world_m=_json_value(data.body_link_pos_w),
                body_link_quaternion_wxyz=_json_value(data.body_link_quat_w),
                controller_active_target_rad=_json_value(state[short+"_q"]),
                controller_wrist_target_local=_json_value(state[short+"_loc"]) if short+"_loc" in state else None,
                controller_wrist_position_world_m=_json_value(state[short+"_wrist_pos"]),
                sdk_joint_position_target_rad=_json_value(data.joint_pos_target)
                    if hasattr(data,"joint_pos_target") else None,
                actuator_torque_estimate_nm=_json_value(data.applied_torque)
                    if hasattr(data,"applied_torque") else None,
                torque_source="simulated_implicit_actuator_estimate_not_hardware")
        return result

    def fail_substep(self,error,state):
        """Preserve a failed attempted step; do not fabricate contact data."""
        if self._faulted:
            return
        self._faulted=True;self._ready=False;self._hold=None
        row=dict(event="physical_failure",episode_uid=self._episode_uid,env_id=0,
            policy_step=self._policy_step,policy_substep=self._substep,
            physics_step=int(state["sim"].current_time_step_index),
            timestamp=float(state["sim"].current_time),source_action=self._source_action.tolist(),
            issued_action=None,
            projected_action=self._issued.tolist() if self._issued is not None else None,
            execution_status="not_confirmed_complete",
            contact_valid=False,terminal=True,failure="physical_step_error",
            error=str(error),native_success=False)
        try:
            row["hands"]=self._hands(state)
            row["rod_nodes_world_m"]=_json_value(state["rope_pos_view"])
        except Exception as capture_error:
            row["failure_snapshot_error"]=str(capture_error)
        self.emit(row)

    def after_substep(self,issued,state):
        from .tabletop_teacher_force_closure import reduce_force_closure
        self._check_identity(state)
        tick=int(state["sim"].current_time_step_index)
        sdk_time=float(state["sim"].current_time)
        rod_time=float(state["rope"].time)
        row=dict(event="physical_sample",episode_uid=self._episode_uid,env_id=0,
            policy_step=self._policy_step,policy_substep=self._substep,
            physics_step=tick,rod_tick=self._rod_tick+1,timestamp=sdk_time,
            solver_time_s=rod_time,source_action=self._source_action.tolist(),
            clipped_source_action=self._control_action.tolist(),
            action_clipping_occurred=bool(np.any(self._source_action!=self._control_action)),
            issued_action=_json_value(issued),hands=self._hands(state),
            rod_nodes_world_m=_json_value(state["rope_pos_view"]),
            native_success=False,contact_slip_m=None,
            final_success_blocked_reason="fixed_anchor_contact_slip_source_not_qualified",
            source_scope="deformx_simulated_contact_reaction_not_hardware",
            terminal=False)
        def fail(reason,error):
            self._faulted=True;self._ready=False;self._hold=None
            row.update(terminal=True,failure=reason,error=str(error))
            self.emit(row)
        if getattr(state["rope"],"last_error",None):
            error=RuntimeError("solver error: "+str(state["rope"].last_error))
            fail("solver_error",error);raise error
        if (tick!=self._tick+1 or not math.isclose(sdk_time-self._sdk_time,DT,abs_tol=2e-6)
                or not math.isclose(rod_time-self._rod_time,DT,abs_tol=2e-6)):
            error=RuntimeError("SDK/rod physics clock discontinuity")
            fail("clock_discontinuity",error);raise error
        try:
            packet=state["rope"].contact_pair_samples()
            row["contact_packet"]={k:(v.tolist() if isinstance(v,np.ndarray) else v)
                for k,v in packet.items()}
            closure=reduce_force_closure(packet,grasp=self.grasp,dt_s=DT)
        except (ValueError,RuntimeError,KeyError,TypeError) as error:
            fail("invalid_contact_packet",error);raise
        # Keep the original packet, captured above before validation, alongside
        # its reduction; never replace measurements with fitted forces.
        row["closure"]=closure
        nodes=_array(state["rope_pos_view"])
        palms=np.stack([_array(state[s+"_hands"].data.body_link_pos_w)[:,state[b]]
            for s,b in (("left","l_base_id"),("right","r_base_id"))])
        if not np.isfinite(nodes).all() or not np.isfinite(palms).all():
            error=RuntimeError("nonfinite physical geometry")
            fail("invalid_physical_geometry",error);raise error
        stable=False
        if self._previous_geometry is not None:
            old_nodes,old_palms=self._previous_geometry
            node_speed=float(np.linalg.norm(nodes-old_nodes,axis=-1).max()/DT)
            palm_speed=float(np.linalg.norm(palms-old_palms,axis=-1).max()/DT)
            stable=node_speed<=.002 and palm_speed<=.001
            row["measured_max_node_speed_m_s"]=node_speed
            row["measured_max_palm_speed_m_s"]=palm_speed
        self._previous_geometry=(nodes,palms)
        receipt=self.owner.observe(PhaseSampleV2(self._episode_uid,tick,
            bool(closure["bilateral"]),bool(stable),False))
        row.update(phase=receipt.phase,phase_math=asdict(receipt),
            phase_math_native_ready=receipt.native_ready,
            prelift_stable_s=receipt.prelift_stable_s)
        was_ready=self._ready
        self._ready=receipt.phase in ("Lift","LiftHold")
        if was_ready and not self._ready:
            self._hold=None  # Cancel prior command lead at the next measured boundary.
        for side in ("left","right"):
            hand=row["hands"][side]
            q=np.asarray(hand["joint_position_rad"])
            qd=np.asarray(hand["joint_velocity_rad_s"])
            bounds=np.asarray(hand["joint_limits_rad"])
            if (not np.isfinite(q).all() or not np.isfinite(qd).all()
                    or np.any(q<bounds[...,0]-JOINT_GUARD_MARGIN_RAD)
                    or np.any(q>bounds[...,1]+JOINT_GUARD_MARGIN_RAD)):
                error=RuntimeError("actual joint position limit violated")
                fail("joint_position_limit",error);raise error
        self._tick=tick;self._sdk_time=sdk_time;self._rod_time=rod_time;self._rod_tick+=1
        self.emit(row)


def create_force_runtime(root,args,app,source_sha,force_hook,*,owner_observer=None):
    """Construct only the selected N=1 native diagnostic, no automatic run."""
    import importlib.util
    from .tabletop_deformx_end_supported_scene import bind_native_end_supports,extend_mesh_pose_buffers
    from .tabletop_teacher_supported_runtime import supported_seed_plan
    from .tabletop_teacher_seed_initialization import spawn_force_scene,audit_seed_after_setup
    force_hook.validate_arguments(args)
    root=Path(root);path=root/"scripts/train_dh116_gpu_teacher_ppo.py"
    code=compile_force_runtime(path.read_text(),source_sha)
    plan,config=supported_seed_plan(root)
    spec=importlib.util.spec_from_file_location("_dh116_force_diagnostic_source",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace=dict(module.__dict__,bind_native_end_supports=bind_native_end_supports,
        extend_mesh_pose_buffers=extend_mesh_pose_buffers,spawn_force_scene=spawn_force_scene,
        audit_seed_after_setup=audit_seed_after_setup)
    exec(compile(code,str(path)+"[force-diagnostic-only]","exec"),namespace)
    values=namespace["create_runtime"](args,app,[plan["seeds_by_style"][args.mode]],plan,force_hook)
    if owner_observer is not None:
        from .tabletop_simulation_owner_cleanup import runtime_owner_ids
        owner_observer(runtime_owner_ids(values,namespace))
    values.update(supported_scene_plan=plan,supported_scene_config=config,
        diagnostic_contract="force_contact_seeded_v2_no_ppo_no_success_admission",
        frozen_teacher_source_sha256=source_sha)
    return SimpleNamespace(**values)
