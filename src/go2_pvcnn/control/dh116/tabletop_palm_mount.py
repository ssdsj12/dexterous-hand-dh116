"""GPU-driven three-axis wrist carriage; normal motion never teleports the hand.

The support is a simulated robot wrist, separate from DH116 finger actuation.
Its prismatic joints are driven through PhysX, so contact sees actual velocity.
"""
import torch

JOINT_NAMES=('wrist_x','wrist_y','wrist_z')
CONTRACT=dict(version='driven_wrist_carriage_v2',motion='three_prismatic_drives',
    stiffness_n_m=100000.,damping_n_s_m=1000.,effort_limit_n=100.,velocity_limit_m_s=.05,
    limit_m=.1,tracking_guard_m=.003,linear_speed_guard_m_s=.06,
    palm_observation='measured_offset_from_common_lift_target',reset='teleport_anchor_and_zero_carriage',qualification='diagnostic_candidate')


def author_mount(stage,hand_path):
    from pxr import Gf,UsdGeom,UsdPhysics,Sdf
    root=UsdPhysics.FixedJoint(stage.GetPrimAtPath(hand_path+'/root_joint'))
    base=stage.GetPrimAtPath(hand_path+'/base_link')
    if not root or not base:raise ValueError('DH116 base and authored root joint required')
    root.CreateJointEnabledAttr(False);root.GetPrim().RemoveAPI(UsdPhysics.ArticulationRootAPI)
    links=[]
    for name in ('wrist_anchor','wrist_slide_x','wrist_slide_y'):
        prim=UsdGeom.Xform.Define(stage,hand_path+'/'+name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass=UsdPhysics.MassAPI.Apply(prim);mass.CreateMassAttr(.1);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001))
        prim.AddAppliedSchema('PhysxRigidBodyAPI')
        prim.CreateAttribute('physxRigidBody:disableGravity',Sdf.ValueTypeNames.Bool).Set(True)
        links.append(prim.GetPath())
    fixed=UsdPhysics.FixedJoint.Define(stage,hand_path+'/wrist_fixed')
    fixed.CreateBody1Rel().SetTargets([links[0]])
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim());fixed.GetPrim().AddAppliedSchema('PhysxArticulationAPI')
    for attr in root.GetPrim().GetAttributes():
        if attr.GetName().startswith('physxArticulation:') and attr.HasAuthoredValueOpinion():
            fixed.GetPrim().CreateAttribute(attr.GetName(),attr.GetTypeName()).Set(attr.Get())
    root.GetPrim().SetActive(False)
    links.append(base.GetPath())
    for i,(name,axis) in enumerate(zip(JOINT_NAMES,('X','Y','Z'))):
        j=UsdPhysics.PrismaticJoint.Define(stage,hand_path+'/'+name)
        j.CreateBody0Rel().SetTargets([links[i]]);j.CreateBody1Rel().SetTargets([links[i+1]])
        j.CreateAxisAttr(axis);j.CreateLowerLimitAttr(-.1);j.CreateUpperLimitAttr(.1)
        drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'linear')
        drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(CONTRACT['stiffness_n_m'])
        drive.CreateDampingAttr(CONTRACT['damping_n_s_m']);drive.CreateMaxForceAttr(CONTRACT['effort_limit_n'])
        j.CreateLocalPos0Attr(Gf.Vec3f(0));j.CreateLocalPos1Attr(Gf.Vec3f(0))
        j.CreateLocalRot0Attr(Gf.Quatf(1));j.CreateLocalRot1Attr(Gf.Quatf(1))


def local_translation(target,anchor):
    delta=target[:,:3]-anchor[:,:3]
    xyz=-anchor[:,4:];cross=2*torch.linalg.cross(xyz,delta)
    return delta+anchor[:,3:4]*cross+torch.linalg.cross(xyz,cross)


class MountView:
    def __init__(self,hand):
        self.hand=hand;self.ids=[hand.joint_names.index(n) for n in JOINT_NAMES]
        self.base_id=hand.body_names.index('base_link')

    def move(self,pose):
        local=local_translation(pose,self.hand.data.root_pose_w.to(pose.dtype))
        self.hand.set_joint_position_target(local.float(),joint_ids=self.ids)

    def reset(self,pose,ids):
        # Caller writes root pose and zero default joint state in the same reset.
        target=torch.zeros((len(ids),3),device=pose.device)
        self.hand.set_joint_position_target(target,joint_ids=self.ids,env_ids=ids)


def measured_offset(actual_pose,target_pose,palms):
    return palms+actual_pose[...,:3]-target_pose[...,:3]


def transport_safe(actual_pose,target_pose,velocity):
    error=(actual_pose[...,:3]-target_pose[...,:3]).norm(dim=-1)
    speed=velocity.norm(dim=-1)
    return (torch.isfinite(actual_pose).flatten(1).all(1)&torch.isfinite(speed).all(1)
            &(error<=CONTRACT['tracking_guard_m']).all(1)
            &(speed<=CONTRACT['linear_speed_guard_m_s']).all(1))


def check_transport(hands,mounts,target):
    actual=torch.stack([h.data.body_link_pose_w[:,m.base_id] for h,m in zip(hands,mounts)],1)
    velocity=torch.stack([h.data.body_lin_vel_w[:,m.base_id] for h,m in zip(hands,mounts)],1)
    safe=transport_safe(actual,target,velocity)
    if not bool(safe.all()):
        raise RuntimeError('wrist carriage transport guard: '+str(dict(
            indices=(~safe).nonzero().flatten().cpu().tolist(),
            tracking_error_m=(actual[...,:3]-target[...,:3]).norm(dim=-1)[~safe].cpu().tolist(),
            linear_speed_m_s=velocity.norm(dim=-1)[~safe].cpu().tolist())))
