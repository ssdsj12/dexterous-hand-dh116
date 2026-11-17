"""Seed the independent N=1 diagnostic before its first PhysX reset.

Only the newly assembled in-memory stage is authored. Original USD files,
joint axes/frames/limits, actuator parameters and prior checkpoints stay read only.
PhysxSchema JointStateAPI angular state/drive targets are in degrees.
"""
from copy import deepcopy
import math
import numpy as np

LEGACY_SPAWN_Z_OFFSET_M=.04


def seed_arrays(seed):
    from .linear_coupling import ACTIVE,JOINTS,expand
    try:
        nodes=np.asarray(seed["nodes"],dtype=float)
        full=np.asarray(seed["joints"],dtype=float)
        roots=np.asarray(seed["roots"],dtype=float)
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError("finite named seed arrays required") from error
    if (nodes.shape!=(41,3) or full.shape!=(2,11) or roots.shape!=(2,7)
            or not all(np.isfinite(v).all() for v in (nodes,full,roots))
            or not np.allclose(np.linalg.norm(roots[:,3:],axis=1),1.,atol=1e-6,rtol=0)):
        raise ValueError("finite 41-node, two-hand seed with unit quaternions required")
    active=full[:,[JOINTS.index(n) for n in ACTIVE]]
    for q,joints in zip(active,full):
        if not np.allclose(expand(q),joints,atol=1e-8,rtol=0):
            raise ValueError("seed violates original passive coupling")
    return nodes.copy(),full.copy(),roots.copy(),active.copy()


def prepare_seed_layout(layout,seed):
    """Counter the retained builder's explicit +40mm spawn offset, exactly once."""
    nodes,full,roots,active=seed_arrays(seed)
    result=deepcopy(layout)
    result["hand_roots"]={side:roots[i].copy() for i,side in enumerate(("left","right"))}
    result["active_joints"]={side:active[i].copy() for i,side in enumerate(("left","right"))}
    for pose in result["hand_roots"].values():
        pose[2]-=LEGACY_SPAWN_Z_OFFSET_M
    result["wire_center"]=nodes.mean(axis=0)
    return result


def author_seed_before_reset(stage,sources,seed,*,prefix="/World/envs/env_0",origin=(0.,0.,0.)):
    """Initialize source FK transforms and all joint states before native parsing.

    This is a reset initializer, never a runtime contact, attachment, force or
    success measurement. Call before SimulationContext.reset(), not after it.
    """
    nodes,full,roots,active=seed_arrays(seed)
    from .hand_point_kinematics import HandPointFK
    from .linear_coupling import JOINTS
    from .tabletop_palm_mount import JOINT_NAMES as WRIST,ANGULAR_JOINT_NAMES
    from .tabletop_dual_hand_scene_preparation import _load_offline_usd_schema
    _load_offline_usd_schema()
    from pxr import Gf,Usd,UsdGeom,UsdPhysics,PhysxSchema
    if set(sources)!={"left","right"} or prefix!="/World/envs/env_0":
        raise ValueError("one explicit two-hand env0 source mapping required")
    origin=np.asarray(origin,dtype=float)
    if origin.shape!=(3,) or not np.isfinite(origin).all():
        raise ValueError("finite environment origin required")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage),1.,abs_tol=1e-12):
        raise ValueError("seed initialization requires metre stage units")
    work=[]
    # Validate BOTH hand trees before writing either side to the stage.
    for hi,side in enumerate(("left","right")):
        pose=roots[hi].copy();pose[:3]+=origin
        fk=HandPointFK(sources[side],side,pose)
        poses=fk.poses(active[hi])
        root=stage.GetPrimAtPath(prefix+"/"+side+"Hand")
        if not root or root.IsInstanceProxy():
            raise ValueError("editable newly assembled hand root required")
        bodies={}
        for path,transform in poses.items():
            name=path.rsplit("/",1)[-1]
            prim=stage.GetPrimAtPath(str(root.GetPath())+"/"+name)
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError("source FK body missing from assembled stage: "+name)
            bodies[name]=(prim,transform)
        joints={}
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.Joint) and prim.GetName() in JOINTS:
                if prim.GetName() in joints:
                    raise ValueError("duplicate source joint identity")
                joints[prim.GetName()]=prim
        if set(joints)!=set(JOINTS) or len(bodies)!=17:
            raise ValueError("expected original eleven joints and seventeen source bodies")
        work.append((side,hi,root,bodies,joints,fk.root_pose))
    reports={}
    for side,hi,root,bodies,joints,root_matrix in work:
        UsdGeom.Xformable(root).MakeMatrixXform().Set(Gf.Matrix4d(*root_matrix.T.ravel()))
        inverse=np.linalg.inv(root_matrix)
        for prim,world in bodies.values():
            local=inverse@world
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(*local.T.ravel()))
        for name,qrad in zip(JOINTS,full[hi]):
            prim=joints[name]
            state=PhysxSchema.JointStateAPI.Apply(prim,"angular")
            state.CreatePositionAttr(float(np.degrees(qrad)));state.CreateVelocityAttr(0.)
            # Never add a drive to a passive joint. Existing targets are aligned
            # to the initial state without changing stiffness/damping/maxForce.
            if prim.HasAPI(UsdPhysics.DriveAPI,"angular"):
                drive=UsdPhysics.DriveAPI(prim,"angular")
                drive.CreateTargetPositionAttr(float(np.degrees(qrad)))
                drive.CreateTargetVelocityAttr(0.)
        for prim in Usd.PrimRange(root):
            if prim.GetName() in WRIST+ANGULAR_JOINT_NAMES and prim.IsA(UsdPhysics.Joint):
                axis="linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular"
                state=PhysxSchema.JointStateAPI.Apply(prim,axis)
                state.CreatePositionAttr(0.);state.CreateVelocityAttr(0.)
        reports[side]=dict(body_count=len(bodies),joint_count=len(joints),
            root_pose_wxyz=(roots[hi]+np.r_[origin,np.zeros(4)]).tolist(),
            full_joint_state_rad=full[hi].tolist(),angular_usd_units="degrees",
            native_readback_verified=False)
    return dict(hands=reports,physics_steps=0,native_ready=False,
        source="original_per_hand_FK_and_PhysxSchema_JointStateAPI",
        initialization_scope="new_in_memory_stage_only")


def spawn_force_scene(sim,args,layout,sources,prefix,origin,*,external_wire,seed):
    """Reuse one scene's actual assets; do not create duplicate articulation views."""
    from .tabletop_vec_isaac import spawn_scene
    if external_wire is not True:
        raise ValueError("real external DeformX rod required")
    selected=prepare_seed_layout(layout,seed)
    scene=spawn_scene(sim,args,selected,sources,prefix,origin,external_wire=True)
    scene["seed_initialization"]=author_seed_before_reset(
        sim.stage,sources,seed,prefix=prefix,origin=origin)
    return scene


def audit_seed_after_setup(state,seed):
    """Read actual SDK state BEFORE any post-reset write can conceal drift."""
    from .linear_coupling import JOINTS
    from .tabletop_palm_mount import JOINT_NAMES as WRIST,ANGULAR_JOINT_NAMES
    _,full,roots,_=seed_arrays(seed)
    def array(value):
        if hasattr(value,"detach"):value=value.detach().cpu().numpy()
        return np.asarray(value,dtype=float)
    result={}
    for hi,side in enumerate(("left","right")):
        hand=state[side+"_hands"]
        ids=[hand.joint_names.index(n) for n in JOINTS]
        q=array(hand.data.joint_pos)[0,ids]
        qd=array(hand.data.joint_vel)[0,ids]
        pose=array(hand.data.body_link_pose_w)[0,hand.body_names.index("base_link")]
        target=roots[hi].copy();target[:3]+=np.asarray(state["env_origins_np"])[0]
        finite=bool(np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(pose).all())
        qnorm=float(np.linalg.norm(pose[3:]))
        rotation=None
        if finite and qnorm>0:
            cosine=abs(float(np.dot(pose[3:]/qnorm,target[3:]/np.linalg.norm(target[3:]))))
            rotation=2*math.acos(min(1.,cosine))
        joint_error=float(np.max(np.abs(q-full[hi]))) if finite else None
        position_error=float(np.linalg.norm(pose[:3]-target[:3])) if finite else None
        velocity=float(np.max(np.abs(qd))) if finite else None
        wrist_names=WRIST+ANGULAR_JOINT_NAMES
        wrist_available=all(name in hand.joint_names for name in wrist_names)
        wrist_pos=wrist_vel=None
        translation_error=angular_error=linear_speed=angular_speed=None
        wrist_passed=False
        if wrist_available:
            wrist_ids=[hand.joint_names.index(name) for name in wrist_names]
            wrist_pos=array(hand.data.joint_pos)[0,wrist_ids]
            wrist_vel=array(hand.data.joint_vel)[0,wrist_ids]
            if np.isfinite(wrist_pos).all() and np.isfinite(wrist_vel).all():
                translation_error=float(np.max(np.abs(wrist_pos[:3])))
                angular_error=float(np.max(np.abs(wrist_pos[3:])))
                linear_speed=float(np.max(np.abs(wrist_vel[:3])))
                angular_speed=float(np.max(np.abs(wrist_vel[3:])))
                wrist_passed=(translation_error<=1e-4 and angular_error<=1e-4
                    and linear_speed<=.001 and angular_speed<=.01)
        passed=bool(finite and wrist_passed and rotation is not None and joint_error<=1e-4
            and position_error<=1e-4 and rotation<=1e-3 and velocity<=.01)
        result[side]=dict(passed=passed,joint_names=list(JOINTS),actual_q_rad=q.tolist(),
            actual_qd_rad_s=qd.tolist(),actual_base_pose_wxyz=pose.tolist(),
            desired_q_rad=full[hi].tolist(),desired_base_pose_wxyz=target.tolist(),
            joint_max_error_rad=joint_error,base_position_error_m=position_error,
            base_rotation_error_rad=rotation,max_abs_joint_velocity_rad_s=velocity,
            wrist_joint_names=list(wrist_names),wrist_readback_available=wrist_available,
            wrist_q_m_rad=wrist_pos.tolist() if wrist_pos is not None else None,
            wrist_qd_m_s_rad_s=wrist_vel.tolist() if wrist_vel is not None else None,
            wrist_max_translation_error_m=translation_error,
            wrist_max_angular_error_rad=angular_error,
            wrist_max_linear_speed_m_s=linear_speed,wrist_max_angular_speed_rad_s=angular_speed)
    tick=int(state["sim"].current_time_step_index)
    timestamp=float(state["sim"].current_time)
    return dict(event="initialization_readback",physics_step=tick,timestamp=timestamp,
        hands=result,passed=bool(tick==2 and math.isclose(timestamp,2/480,abs_tol=2e-6)
            and all(r["passed"] for r in result.values())),
        post_reset_seed_write_performed=False,native_success=False)
