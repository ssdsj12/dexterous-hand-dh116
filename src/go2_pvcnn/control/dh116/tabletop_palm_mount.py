"""PhysX-driven external wrist support, with opt-in intrinsic XYZ rotation.

The support is a simulated robot wrist, separate from DH116 finger actuation.
Its prismatic joints are driven through PhysX, so contact sees actual velocity.
"""
import math
import torch

JOINT_NAMES=('wrist_x','wrist_y','wrist_z')
ANGULAR_JOINT_NAMES=('wrist_roll','wrist_pitch','wrist_yaw')
CONTRACT=dict(version='driven_wrist_carriage_v2',motion='three_prismatic_drives',
    stiffness_n_m=100000.,damping_n_s_m=1000.,effort_limit_n=100.,velocity_limit_m_s=.05,
    limit_m=.1,tracking_guard_m=.003,linear_speed_guard_m_s=.06,
    palm_observation='measured_offset_from_common_lift_target',reset='teleport_anchor_and_zero_carriage',qualification='diagnostic_candidate')
ROTATION_CONTRACT=dict(version='external_wrist_rotation_v1',motion='intrinsic_XYZ_revolute_drives',
    angle_limit_rad=math.pi/2,stiffness_nm_rad=40.,damping_nms_rad=4.,
    effort_limit_nm=2.,velocity_limit_rad_s=1.,calibration_status='simulation_estimate',
    mechanism='external_support_not_DH116_finger_dof',usd_angle_unit='degree',readback_angle_unit='rad')


def author_mount(stage,hand_path,rotation=False):
    from pxr import Gf,UsdGeom,UsdPhysics,Sdf
    root=UsdPhysics.FixedJoint(stage.GetPrimAtPath(hand_path+'/root_joint'))
    base=stage.GetPrimAtPath(hand_path+'/base_link')
    if not root or not base:raise ValueError('DH116 base and authored root joint required')
    root.CreateJointEnabledAttr(False);root.GetPrim().RemoveAPI(UsdPhysics.ArticulationRootAPI)
    links=[]
    link_names=('wrist_anchor','wrist_slide_x','wrist_slide_y')
    if rotation:link_names+=('wrist_slide_z','wrist_rotate_x','wrist_rotate_y')
    for name in link_names:
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
    names=JOINT_NAMES+(ANGULAR_JOINT_NAMES if rotation else ())
    for i,(name,axis) in enumerate(zip(names,('X','Y','Z')*2)):
        angular=i>=3
        joint_type=UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint
        j=joint_type.Define(stage,hand_path+'/'+name)
        j.CreateBody0Rel().SetTargets([links[i]]);j.CreateBody1Rel().SetTargets([links[i+1]])
        limit=math.degrees(ROTATION_CONTRACT['angle_limit_rad']) if angular else CONTRACT['limit_m']
        j.CreateAxisAttr(axis);j.CreateLowerLimitAttr(-limit);j.CreateUpperLimitAttr(limit)
        drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular' if angular else 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(ROTATION_CONTRACT['stiffness_nm_rad'] if angular else CONTRACT['stiffness_n_m'])
        drive.CreateDampingAttr(ROTATION_CONTRACT['damping_nms_rad'] if angular else CONTRACT['damping_n_s_m'])
        drive.CreateMaxForceAttr(ROTATION_CONTRACT['effort_limit_nm'] if angular else CONTRACT['effort_limit_n'])
        j.CreateLocalPos0Attr(Gf.Vec3f(0));j.CreateLocalPos1Attr(Gf.Vec3f(0))
        j.CreateLocalRot0Attr(Gf.Quatf(1));j.CreateLocalRot1Attr(Gf.Quatf(1))


def local_translation(target,anchor):
    delta=target[:,:3]-anchor[:,:3]
    xyz=-anchor[:,4:];cross=2*torch.linalg.cross(xyz,delta)
    return delta+anchor[:,3:4]*cross+torch.linalg.cross(xyz,cross)


class MountView:
    def __init__(self,hand):
        self.hand=hand;self.ids=[hand.joint_names.index(n) for n in JOINT_NAMES]
        angular=[n for n in ANGULAR_JOINT_NAMES if n in hand.joint_names]
        if angular and (len(angular)!=3 or any(hand.joint_names.count(n)!=1 for n in angular)):
            raise ValueError('complete angular wrist joint set required')
        self.angular_ids=[hand.joint_names.index(n) for n in angular]
        self.base_id=hand.body_names.index('base_link')

    def move(self,pose,angles=None):
        """Drive translation and explicit intrinsic XYZ angles [N,3] in radians.

        Quaternion composition belongs to the caller. Omitted angles on a
        six-axis mount target zero rotation relative to the reset anchor.
        """
        local=local_translation(pose,self.hand.data.root_pose_w.to(pose.dtype))
        if self.angular_ids:
            angles=torch.zeros_like(local) if angles is None else torch.as_tensor(angles,device=local.device,dtype=local.dtype)
            if (angles.shape!=local.shape or not bool(torch.isfinite(angles).all())
                    or bool((angles.abs()>ROTATION_CONTRACT['angle_limit_rad']+1e-6).any())):
                raise ValueError('finite angles [N,3] within wrist angle limits required')
            local=torch.cat((local,angles),dim=-1)
        elif angles is not None:
            raise ValueError('rotation commands require a complete angular wrist mount')
        self.hand.set_joint_position_target(local.float(),joint_ids=self.ids+self.angular_ids)

    def reset(self,pose,ids):
        # Caller writes root pose and zero default joint state in the same reset.
        joint_ids=self.ids+self.angular_ids
        target=torch.zeros((len(ids),len(joint_ids)),device=pose.device)
        self.hand.set_joint_position_target(target,joint_ids=joint_ids,env_ids=ids)


def measured_offset(actual_pose,target_pose,palms):
    return palms+actual_pose[...,:3]-target_pose[...,:3]


def transport_safe(actual_pose,target_pose,velocity):
    error=(actual_pose[...,:3]-target_pose[...,:3]).norm(dim=-1)
    speed=velocity.norm(dim=-1)
    return (torch.isfinite(actual_pose).flatten(1).all(1)&torch.isfinite(speed).all(1)
            &(error<=CONTRACT['tracking_guard_m']).all(1)
            &(speed<=CONTRACT['linear_speed_guard_m_s']).all(1))


def check_transport(hands,mounts,target,*,return_validity=False):
    actual=torch.stack([h.data.body_link_pose_w[:,m.base_id] for h,m in zip(hands,mounts)],1)
    velocity=torch.stack([h.data.body_lin_vel_w[:,m.base_id] for h,m in zip(hands,mounts)],1)
    safe=transport_safe(actual,target,velocity)
    if return_validity:
        # Tracking lag and finite speed excursions are diagnostics; actuator
        # limits constrain commands. Only invalid transport readback is a
        # state-recovery event in continuous mode.
        return (torch.isfinite(actual).flatten(1).all(1)&torch.isfinite(target).flatten(1).all(1)
                &torch.isfinite(velocity).flatten(1).all(1))
    if not bool(safe.all()):
        raise RuntimeError('wrist carriage transport guard: '+str(dict(
            indices=(~safe).nonzero().flatten().cpu().tolist(),
            tracking_error_m=(actual[...,:3]-target[...,:3]).norm(dim=-1)[~safe].cpu().tolist(),
            linear_speed_m_s=velocity.norm(dim=-1)[~safe].cpu().tolist())))
