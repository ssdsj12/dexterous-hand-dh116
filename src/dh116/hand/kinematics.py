"""Measured joint/wrist coordinates, actuator rates and command/projection arithmetic."""
import math as contr_math__math
import numpy as runti_mimic__np
import math as comma_kerne__math
import math as defor_wrist_proje__math
import numpy as joint_targe_readb__np
import math as palm_mount__math
import math as tenso_wrist_proje__math
veloc_limit__ACTIVE_VELOCITY_LIMIT_RAD_S = 0.3
import torch as contr_math__torch
from dh116.env.contracts import MIMIC as runti_mimic__MIMIC
import torch as comma_kerne__torch
import torch as defor_wrist_proje__torch
from dh116.env.contracts import JOINTS
import torch as palm_mount__torch
import torch as tenso_wrist_proje__torch
LEGACY_PASSIVE_VELOCITY_LIMIT_RAD_S = 3.0

def install_hand_mimic(stage, hand_path):
    from pxr import UsdPhysics
    from pxr import PhysxSchema
    for (follower, master, ratio, offset) in runti_mimic__MIMIC:
        joint = stage.GetPrimAtPath(f'{hand_path}/joints/{follower}')
        reference = stage.GetPrimAtPath(f'{hand_path}/joints/{master}')
        for prim in (joint, reference):
            if not prim.IsValid() or UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() != 'X':
                raise ValueError(f'invalid DH116 mimic joint: {prim}')
        api = PhysxSchema.PhysxMimicJointAPI.Apply(joint, 'rotX')
        api.CreateReferenceJointRel().SetTargets([reference.GetPath()])
        api.CreateReferenceJointAxisAttr('rotX')
        api.CreateGearingAttr(-ratio)
        api.CreateOffsetAttr(-float(runti_mimic__np.degrees(offset)))
    return len(runti_mimic__MIMIC)
BACKEND_SOURCE = 'physx_backend_get_dof_position_targets'
JOINT_NAMES = ('wrist_x', 'wrist_y', 'wrist_z')
PASSIVE_VELOCITY_HEADROOM_RAD_S = 2.97

def configure_mimic_compliance(stage, hand_path, natural_frequency, damping_ratio, *, allow_rigid=False):
    """Author compliance, or opt in explicitly to zero-frequency rigid coupling.

    Isaac 5.1 / PhysX 107.3 exposes these as codeless USD attributes, not
    PhysxMimicJointAPI Create* methods. Frequency is in inverse seconds.
    """
    from pxr import Sdf
    from pxr import UsdPhysics
    if not runti_mimic__np.isfinite([natural_frequency, damping_ratio]).all() or natural_frequency < 0 or (natural_frequency == 0 and (not allow_rigid)) or (damping_ratio < 1.0):
        raise ValueError('finite nonnegative frequency (zero requires allow_rigid) and damping ratio >=1 required')
    joints = [stage.GetPrimAtPath(f'{hand_path}/joints/{f}') for (f, _, _, _) in runti_mimic__MIMIC]
    if any((not j.IsValid() or UsdPhysics.RevoluteJoint(j).GetAxisAttr().Get() != 'X' for j in joints)):
        raise ValueError('five valid X-axis follower joints required')
    for joint in joints:
        joint.CreateAttribute('physxMimicJoint:rotX:naturalFrequency', Sdf.ValueTypeNames.Float).Set(float(natural_frequency))
        joint.CreateAttribute('physxMimicJoint:rotX:dampingRatio', Sdf.ValueTypeNames.Float).Set(float(damping_ratio))
    return len(joints)
CACHE_SOURCE = 'isaaclab_user_command_cache_joint_pos_target'
ANGULAR_JOINT_NAMES = ('wrist_roll', 'wrist_pitch', 'wrist_yaw')
CLOSE_COMMAND_RATE_RAD_S = 0.2
WRIST_LINEAR = ('wrist_x', 'wrist_y', 'wrist_z')
palm_mount__CONTRACT = dict(version='driven_wrist_carriage_v2', motion='three_prismatic_drives', stiffness_n_m=100000.0, damping_n_s_m=1000.0, effort_limit_n=100.0, velocity_limit_m_s=0.05, limit_m=0.1, tracking_guard_m=0.003, linear_speed_guard_m_s=0.06, palm_observation='measured_offset_from_common_lift_target', reset='teleport_anchor_and_zero_carriage', qualification='diagnostic_candidate')
VELOCITY_HEADROOM_CONTRACT = dict(version='passive_velocity_headroom_candidate_v1', passive_velocity_limit_rad_s=PASSIVE_VELOCITY_HEADROOM_RAD_S, nominal_guard_limit_rad_s=3.0, qualification='controlled_simulation_candidate', guaranteed_measured_velocity_bound=False)
WRIST_ANGULAR = ('wrist_roll', 'wrist_pitch', 'wrist_yaw')
ROTATION_CONTRACT = dict(version='external_wrist_rotation_v1', motion='intrinsic_XYZ_revolute_drives', angle_limit_rad=palm_mount__math.pi / 2, stiffness_nm_rad=40.0, damping_nms_rad=4.0, effort_limit_nm=2.0, velocity_limit_rad_s=1.0, calibration_status='simulation_estimate', mechanism='external_support_not_DH116_finger_dof', usd_angle_unit='degree', readback_angle_unit='rad')

def passive_velocity_limit(control_mode, requested=LEGACY_PASSIVE_VELOCITY_LIMIT_RAD_S):
    """Keep existing modes and defaults; accept only the reviewed candidate."""
    if isinstance(requested, bool) or not isinstance(requested, (int, float)) or requested not in (LEGACY_PASSIVE_VELOCITY_LIMIT_RAD_S, PASSIVE_VELOCITY_HEADROOM_RAD_S):
        raise ValueError('passive velocity limit must be 3.0 or the 2.97 rad/s candidate')
    if control_mode in ('joint_target', 'adaptive_grasp'):
        return float(requested)
    return veloc_limit__ACTIVE_VELOCITY_LIMIT_RAD_S
contr_math__ACTIVE_VELOCITY_LIMIT_RAD_S = veloc_limit__ACTIVE_VELOCITY_LIMIT_RAD_S
comma_kerne__ACTIVE_VELOCITY_LIMIT_RAD_S = veloc_limit__ACTIVE_VELOCITY_LIMIT_RAD_S

def _owned(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return joint_targe_readb__np.array(value, copy=True)

def author_mount(stage, hand_path, rotation=False):
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import Sdf
    root = UsdPhysics.FixedJoint(stage.GetPrimAtPath(hand_path + '/root_joint'))
    base = stage.GetPrimAtPath(hand_path + '/base_link')
    if not root or not base:
        raise ValueError('DH116 base and authored root joint required')
    root.CreateJointEnabledAttr(False)
    root.GetPrim().RemoveAPI(UsdPhysics.ArticulationRootAPI)
    links = []
    link_names = ('wrist_anchor', 'wrist_slide_x', 'wrist_slide_y')
    if rotation:
        link_names += ('wrist_slide_z', 'wrist_rotate_x', 'wrist_rotate_y')
    for name in link_names:
        prim = UsdGeom.Xform.Define(stage, hand_path + '/' + name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(0.1)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.001))
        prim.AddAppliedSchema('PhysxRigidBodyAPI')
        prim.CreateAttribute('physxRigidBody:disableGravity', Sdf.ValueTypeNames.Bool).Set(True)
        links.append(prim.GetPath())
    fixed = UsdPhysics.FixedJoint.Define(stage, hand_path + '/wrist_fixed')
    fixed.CreateBody1Rel().SetTargets([links[0]])
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
    fixed.GetPrim().AddAppliedSchema('PhysxArticulationAPI')
    for attr in root.GetPrim().GetAttributes():
        if attr.GetName().startswith('physxArticulation:') and attr.HasAuthoredValueOpinion():
            fixed.GetPrim().CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
    root.GetPrim().SetActive(False)
    links.append(base.GetPath())
    names = JOINT_NAMES + (ANGULAR_JOINT_NAMES if rotation else ())
    for (i, (name, axis)) in enumerate(zip(names, ('X', 'Y', 'Z') * 2)):
        angular = i >= 3
        joint_type = UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint
        j = joint_type.Define(stage, hand_path + '/' + name)
        j.CreateBody0Rel().SetTargets([links[i]])
        j.CreateBody1Rel().SetTargets([links[i + 1]])
        limit = palm_mount__math.degrees(ROTATION_CONTRACT['angle_limit_rad']) if angular else palm_mount__CONTRACT['limit_m']
        j.CreateAxisAttr(axis)
        j.CreateLowerLimitAttr(-limit)
        j.CreateUpperLimitAttr(limit)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'angular' if angular else 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(ROTATION_CONTRACT['stiffness_nm_rad'] if angular else palm_mount__CONTRACT['stiffness_n_m'])
        drive.CreateDampingAttr(ROTATION_CONTRACT['damping_nms_rad'] if angular else palm_mount__CONTRACT['damping_n_s_m'])
        drive.CreateMaxForceAttr(ROTATION_CONTRACT['effort_limit_nm'] if angular else palm_mount__CONTRACT['effort_limit_n'])
        j.CreateLocalPos0Attr(Gf.Vec3f(0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0))
        j.CreateLocalRot0Attr(Gf.Quatf(1))
        j.CreateLocalRot1Attr(Gf.Quatf(1))
contr_math__CONTRACT = dict(version='free_wrist_grasp_v1', action_size=26, action_order='joint_velocity12_world_translation_velocity6_intrinsic_XYZ_velocity6_normal_force2', joint_rate_rad_s=contr_math__ACTIVE_VELOCITY_LIMIT_RAD_S, joint_target_lead_rad=0.1, palm_lower_m=[-0.03, -0.03, -0.004], palm_upper_m=[0.03, 0.03, 0.05], wrist_rate_m_s=0.01, rotation_rate_rad_s=0.5, rotation_limit_rad=contr_math__math.pi / 2, joint_bounds='actual_affine_mimic_compatible_limits_not_reset_pose_envelope', posture_reference='reset_only;zero_action_holds_current_targets', normal_force_range_n=[2.0, 12.0], friction_excluded=True, stage_flow=None, pose_latch=False, automatic_ascent=False, orientation='seed_world_quaternion_times_intrinsic_X_then_Y_then_Z', hardware_calibration_complete=False)

def _read(getter, source, names, names_valid, *, count=None, integer=False):
    result = dict(available=False, source=source, values=None, finite_env=None, dtype=None, shape=None, error=None)
    try:
        value = _owned(getter())
        result.update(dtype=str(value.dtype), shape=list(value.shape))
        if not names_valid or value.ndim != 2 or value.shape[0] < 1 or (value.shape[1] != len(names)) or (count is not None and len(value) != count) or (value.dtype.kind not in ('iu' if integer else 'f')):
            raise ValueError('named complete [environment,DOF] target/type array required')
        result.update(available=True, values=value, finite_env=joint_targe_readb__np.isfinite(value).all(1))
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    return result

def local_translation(target, anchor):
    delta = target[:, :3] - anchor[:, :3]
    xyz = -anchor[:, 4:]
    cross = 2 * palm_mount__torch.linalg.cross(xyz, delta)
    return delta + anchor[:, 3:4] * cross + palm_mount__torch.linalg.cross(xyz, cross)
contr_math__CONTRACT.update(finger_active_joints=12, finger_passive_joints=10, external_wrist_drives=12, normal_force_parameters=2, hand_body_dof_count=22, policy_command_size=26)

def read_joint_target_evidence(hand):
    """Return complete per-hand backend/cache/type evidence as owned NumPy arrays.

    ``available`` means a named array was obtained, independently of each
    environment's ``finite_env`` flag. Missing/failed backend reads never use
    a cache fallback. Units come from native DofType, not a joint-name guess;
    expected source-contract units remain separately labelled.
    """
    names = list(getattr(hand, 'joint_names', []))
    names_valid = bool(names) and all((isinstance(name, str) and name for name in names)) and (len(set(names)) == len(names))
    backend = _read(lambda : hand.root_physx_view.get_dof_position_targets(), BACKEND_SOURCE, names, names_valid)
    count = len(backend['values']) if backend['available'] else None
    cache = _read(lambda : hand.data.joint_pos_target, CACHE_SOURCE, names, names_valid, count=count)
    if count is None and cache['available']:
        count = len(cache['values'])
    types = _read(lambda : hand.root_physx_view.get_dof_types(), 'physx_backend_get_dof_types', names, names_valid, count=count, integer=True)
    units = dict.fromkeys(names)
    if types['available']:
        for (index, name) in enumerate(names):
            kinds = types['values'][:, index]
            if joint_targe_readb__np.all(kinds == 0):
                units[name] = 'rad'
            elif joint_targe_readb__np.all(kinds == 1):
                units[name] = 'm'
    expected = {name: 'm' if name in WRIST_LINEAR else 'rad' if name in WRIST_ANGULAR or name in JOINTS else None for name in names}
    return dict(joint_names=names, joint_names_valid=names_valid, backend=backend, command_cache=cache, dof_types=types, units_by_name=units, unit_source='native_DofType_0_rotation_1_translation;unknown_or_mixed_is_null', expected_units_by_name=expected, expected_unit_source='original_DH116_and_driven_carriage_contract;not_native_type_readback', wrist_joint_indices={name: names.index(name) if names_valid and name in names else None for name in WRIST_LINEAR + WRIST_ANGULAR}, wrist_target_frame='anchor_local_translation;seed_intrinsic_XYZ_rotation;not_measured_world_palm', native_target_readback_is_not_actual_joint_position=True)

class MountView:

    def __init__(self, hand):
        self.hand = hand
        self.ids = [hand.joint_names.index(n) for n in JOINT_NAMES]
        angular = [n for n in ANGULAR_JOINT_NAMES if n in hand.joint_names]
        if angular and (len(angular) != 3 or any((hand.joint_names.count(n) != 1 for n in angular))):
            raise ValueError('complete angular wrist joint set required')
        self.angular_ids = [hand.joint_names.index(n) for n in angular]
        self.base_id = hand.body_names.index('base_link')

    def move(self, pose, angles=None):
        """Drive translation and explicit intrinsic XYZ angles [N,3] in radians.

        Quaternion composition belongs to the caller. Omitted angles on a
        six-axis mount target zero rotation relative to the reset anchor.
        """
        local = local_translation(pose, self.hand.data.root_pose_w.to(pose.dtype))
        if self.angular_ids:
            angles = palm_mount__torch.zeros_like(local) if angles is None else palm_mount__torch.as_tensor(angles, device=local.device, dtype=local.dtype)
            if angles.shape != local.shape or not bool(palm_mount__torch.isfinite(angles).all()) or bool((angles.abs() > ROTATION_CONTRACT['angle_limit_rad'] + 1e-06).any()):
                raise ValueError('finite angles [N,3] within wrist angle limits required')
            local = palm_mount__torch.cat((local, angles), dim=-1)
        elif angles is not None:
            raise ValueError('rotation commands require a complete angular wrist mount')
        self.hand.set_joint_position_target(local.float(), joint_ids=self.ids + self.angular_ids)

    def reset(self, pose, ids):
        joint_ids = self.ids + self.angular_ids
        target = palm_mount__torch.zeros((len(ids), len(joint_ids)), device=pose.device)
        self.hand.set_joint_position_target(target, joint_ids=joint_ids, env_ids=ids)
MEASUREMENT_KEYS = ('wrist_pose', 'wrist_velocity', 'wrist_joint_angles')

def measured_offset(actual_pose, target_pose, palms):
    return palms + actual_pose[..., :3] - target_pose[..., :3]

def multiply(a, b):
    (aw, ax, ay, az) = a.unbind(-1)
    (bw, bx, by, bz) = b.unbind(-1)
    return contr_math__torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)

def transport_safe(actual_pose, target_pose, velocity):
    error = (actual_pose[..., :3] - target_pose[..., :3]).norm(dim=-1)
    speed = velocity.norm(dim=-1)
    return palm_mount__torch.isfinite(actual_pose).flatten(1).all(1) & palm_mount__torch.isfinite(speed).all(1) & (error <= palm_mount__CONTRACT['tracking_guard_m']).all(1) & (speed <= palm_mount__CONTRACT['linear_speed_guard_m_s']).all(1)

def compose_rotation(seed, angles):
    if seed.shape[:-1] != angles.shape[:-1] or seed.shape[-1] != 4 or angles.shape[-1] != 3:
        raise ValueError('matching wxyz quaternion and XYZ angle batches required')
    if not contr_math__torch.isfinite(seed).all() or not contr_math__torch.isfinite(angles).all():
        raise ValueError('finite wrist rotation required')
    scale = seed.abs().amax(-1, keepdim=True)
    if (scale <= 0).any():
        raise ValueError('nonzero wrist quaternion required')
    q = seed / scale
    q = q / q.norm(dim=-1, keepdim=True)
    for axis in range(3):
        step = contr_math__torch.zeros_like(q)
        step[..., 0] = contr_math__torch.cos(angles[..., axis] / 2)
        step[..., axis + 1] = contr_math__torch.sin(angles[..., axis] / 2)
        q = multiply(q, step)
    return q / q.norm(dim=-1, keepdim=True)

def check_transport(hands, mounts, target, *, return_validity=False):
    actual = palm_mount__torch.stack([h.data.body_link_pose_w[:, m.base_id] for (h, m) in zip(hands, mounts)], 1)
    velocity = palm_mount__torch.stack([h.data.body_lin_vel_w[:, m.base_id] for (h, m) in zip(hands, mounts)], 1)
    safe = transport_safe(actual, target, velocity)
    if return_validity:
        return palm_mount__torch.isfinite(actual).flatten(1).all(1) & palm_mount__torch.isfinite(target).flatten(1).all(1) & palm_mount__torch.isfinite(velocity).flatten(1).all(1)
    if not bool(safe.all()):
        raise RuntimeError('wrist carriage transport guard: ' + str(dict(indices=(~safe).nonzero().flatten().cpu().tolist(), tracking_error_m=(actual[..., :3] - target[..., :3]).norm(dim=-1)[~safe].cpu().tolist(), linear_speed_m_s=velocity.norm(dim=-1)[~safe].cpu().tolist())))

def constrain_palm_z_movement(movement, clearances, clearance_floor_m):
    """Keep palm descent above the configured floor without changing X/Y."""
    if not contr_math__torch.isfinite(contr_math__torch.as_tensor(clearance_floor_m)) or clearance_floor_m < 0:
        raise ValueError('clearance floor must be finite and nonnegative')
    z = contr_math__torch.maximum(movement[:, :, 2], -(clearances - clearance_floor_m).clamp_min(0))
    return contr_math__torch.cat((movement[:, :, :2], z[:, :, None]), dim=-1)
comma_kerne__constrain_palm_z_movement = constrain_palm_z_movement
defor_wrist_proje__WRIST = contr_math__CONTRACT
defor_wrist_proje__compose_rotation = compose_rotation
tenso_wrist_proje__WRIST = contr_math__CONTRACT
tenso_wrist_proje__multiply = multiply
from dh116.env.contracts import linea_coupl__ACTIVE as ACTIVE
from dh116.env.contracts import MIMIC as comma_kerne__MIMIC

def finger_pre_guard(command, action, q, mimic_errors, low, high, dt, rates, *, policy_owned=False, pinch_force_hold=False, teacher_force_controlled_digits=None):
    CONTRACT = contr_math__CONTRACT
    target = command + action[:, :12] * rates['joint_rate_rad_s'] * dt
    for (j, (_, master, ratio, _)) in enumerate(comma_kerne__MIMIC):
        if policy_owned or (pinch_force_hold and j < 2):
            continue
        for h in range(2):
            k = 6 * h + ACTIVE.index(master)
            error = mimic_errors[:, 5 * h + j]
            correction = (5.0 * error / ratio * dt).clamp(-comma_kerne__ACTIVE_VELOCITY_LIMIT_RAD_S * dt, comma_kerne__ACTIVE_VELOCITY_LIMIT_RAD_S * dt)
            correct = error.abs() >= 0.012
            if teacher_force_controlled_digits is not None:
                correct &= ~teacher_force_controlled_digits[:, h, j]
            target[:, k] += comma_kerne__torch.where(correct, correction, 0.0)
    lead = CONTRACT['joint_target_lead_rad']
    target = target.maximum(q - lead).minimum(q + lead).maximum(low).minimum(high)
    return command + (target - command).clamp(-comma_kerne__ACTIVE_VELOCITY_LIMIT_RAD_S * dt, comma_kerne__ACTIVE_VELOCITY_LIMIT_RAD_S * dt)

def surface_final_guard(command, target, low, high, dt):
    target = command + (target - command).clamp(-comma_kerne__ACTIVE_VELOCITY_LIMIT_RAD_S * dt, comma_kerne__ACTIVE_VELOCITY_LIMIT_RAD_S * dt)
    return (target, target.maximum(low).minimum(high))

def limited_step(action, rate, dt):
    norm = action.norm(dim=-1, keepdim=True).clamp_min(1.0)
    return action / norm * (rate * dt)

def wrist_requested_targets(palms, angles, actions, rates, dt):
    CONTRACT = contr_math__CONTRACT
    n = actions.shape[0]
    lo = actions.new_tensor(CONTRACT['palm_lower_m'])
    hi = actions.new_tensor(CONTRACT['palm_upper_m'])
    palms = (palms + limited_step(actions[:, 12:18].reshape(n, 2, 3), rates['wrist_rate_m_s'], dt)).maximum(lo).minimum(hi)
    angles = (angles + limited_step(actions[:, 18:24].reshape(n, 2, 3), rates['rotation_rate_rad_s'], dt)).clamp(-comma_kerne__math.pi / 2, comma_kerne__math.pi / 2)
    return (palms, angles)

def palm_transport(palms, target, clearances, clearance_floor, rate, dt, lower, upper, allowed, live, *, vector_norm):
    movement = (target - palms).clamp(-rate * dt, rate * dt)
    if vector_norm:
        norm = movement.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        movement = movement * (rate * dt / norm).clamp_max(1.0)
    movement = comma_kerne__constrain_palm_z_movement(movement, clearances, clearance_floor)
    pre_projection = palms + movement
    reasons = (movement.abs().amax(-1) >= rate * dt - 1e-12).long()
    palms = comma_kerne__torch.where((allowed & live)[:, None, None], comma_kerne__torch.maximum(comma_kerne__torch.minimum(palms + movement, upper), lower), palms)
    return (palms, pre_projection, reasons)

def per_hand_safe_z(palms, roots, root_seed, offset, clearances, clearance_floor):
    measured_z = roots[:, :, 2] - root_seed[:, :, 2] - offset[:, None]
    safe_z = measured_z - (clearances - clearance_floor)
    return (comma_kerne__torch.maximum(palms[:, :, 2], safe_z), safe_z)
defor_wrist_proje__palm_transport = palm_transport
defor_wrist_proje__per_hand_safe_z = per_hand_safe_z
tenso_wrist_proje__palm_transport = palm_transport
tenso_wrist_proje__per_hand_safe_z = per_hand_safe_z
from dh116.env.contracts import surfa_contr__ACTION_CONTRACT as defor_wrist_proje__ACTION_CONTRACT
defor_wrist_proje__DT = 1 / 480
defor_wrist_proje__CLEARANCE_M = 0.0021
defor_wrist_proje__SOURCE = 'tabletop_command_kernels.palm_transport+per_hand_safe_z:policy_owned_default_clearance'

def project_policy_wrist_targets(*, root_seed_w, measured_base_w, common_lift_offset_m, measured_clearance_m, palm_offsets_m, requested_palm_offsets_m, angles_rad, requested_angles_rad, allowed, live):
    """Mirror the original policy-owned, independently projected wrist branch."""
    if not isinstance(root_seed_w, defor_wrist_proje__torch.Tensor) or root_seed_w.ndim != 3 or root_seed_w.shape[1:] != (2, 7) or (len(root_seed_w) < 1) or (not root_seed_w.is_floating_point()):
        raise ValueError('measured and authored wrist poses must be floating [N,2,7] WXYZ')
    n = len(root_seed_w)
    for (name, value, shape) in (('reset seed', root_seed_w, (n, 2, 7)), ('measured base', measured_base_w, (n, 2, 7)), ('common lift offset', common_lift_offset_m, (n,)), ('measured clearance', measured_clearance_m, (n, 2)), ('palm offsets', palm_offsets_m, (n, 2, 3)), ('requested palm offsets', requested_palm_offsets_m, (n, 2, 3)), ('angles', angles_rad, (n, 2, 3)), ('requested angles', requested_angles_rad, (n, 2, 3))):
        if not isinstance(value, defor_wrist_proje__torch.Tensor) or value.shape != shape or value.device != root_seed_w.device or (value.dtype != root_seed_w.dtype) or (not bool(defor_wrist_proje__torch.isfinite(value).all())):
            raise ValueError(f'finite same-device {name} with shape {shape} required')
    for mask in (allowed, live):
        if not isinstance(mask, defor_wrist_proje__torch.Tensor) or mask.shape != (n,) or mask.dtype != defor_wrist_proje__torch.bool or (mask.device != root_seed_w.device):
            raise ValueError('explicit same-device boolean allowed/live masks required')
    for pose in (root_seed_w, measured_base_w):
        norm = defor_wrist_proje__torch.linalg.vector_norm(pose[..., 3:], dim=-1)
        if not defor_wrist_proje__torch.allclose(norm, defor_wrist_proje__torch.ones_like(norm), atol=1e-05, rtol=0):
            raise ValueError('unit WXYZ quaternions required')
    if any((bool((angles.abs() > defor_wrist_proje__math.pi / 2).any()) for angles in (angles_rad, requested_angles_rad))):
        raise ValueError('original intrinsic XYZ limits required')
    lower = root_seed_w.new_tensor(defor_wrist_proje__WRIST['palm_lower_m'])
    upper = root_seed_w.new_tensor(defor_wrist_proje__WRIST['palm_upper_m'])
    (palms, pre_projection, reasons) = defor_wrist_proje__palm_transport(palm_offsets_m, requested_palm_offsets_m, measured_clearance_m, defor_wrist_proje__CLEARANCE_M, defor_wrist_proje__ACTION_CONTRACT['translation_rate_m_s'], defor_wrist_proje__DT, lower, upper, allowed, live, vector_norm=True)
    (projected_z, safe_z) = defor_wrist_proje__per_hand_safe_z(palms, measured_base_w, root_seed_w, common_lift_offset_m, measured_clearance_m, defor_wrist_proje__CLEARANCE_M)
    clipped = live[:, None] & ((projected_z - palms[:, :, 2]).abs() > 1e-12)
    palms[:, :, 2] = defor_wrist_proje__torch.where(live[:, None], projected_z, palms[:, :, 2])
    angles = defor_wrist_proje__torch.where((allowed & live)[:, None, None], requested_angles_rad, angles_rad)
    roots = root_seed_w.clone()
    roots[..., :3] += palms
    roots[..., 2] += common_lift_offset_m[:, None]
    roots[..., 3:] = defor_wrist_proje__compose_rotation(root_seed_w[..., 3:], angles)
    return dict(projected_root_poses=roots, projected_angles=angles, palm_offsets_m=palms, pre_projection_palm_offsets_m=pre_projection, safe_z_offset_m=safe_z, safe_z_clipped=clipped, projection_reasons=reasons | 2 * clipped.long(), projection_source=defor_wrist_proje__SOURCE, clearance_floor_m=defor_wrist_proje__CLEARANCE_M, native_clearance_qualified=False, teacher_admission=False)
tenso_wrist_proje__CLEARANCE_M = defor_wrist_proje__CLEARANCE_M
tenso_wrist_proje__DT = defor_wrist_proje__DT
tenso_wrist_proje__SOURCE = defor_wrist_proje__SOURCE
from dh116.env.contracts import surfa_contr__ACTION_CONTRACT as tenso_wrist_proje__ACTION_CONTRACT

def project_tensor_policy_wrist_targets(*, root_seed_w, measured_base_w, common_lift_offset_m, measured_clearance_m, palm_offsets_m, requested_palm_offsets_m, angles_rad, requested_angles_rad, allowed, live):
    """Project the full same-device batch without per-field host decisions."""
    if not isinstance(root_seed_w, tenso_wrist_proje__torch.Tensor) or root_seed_w.ndim != 3 or root_seed_w.shape[1:] != (2, 7) or (len(root_seed_w) < 1) or (not root_seed_w.is_floating_point()):
        raise ValueError('measured and authored wrist poses must be floating [N,2,7] WXYZ')
    n = len(root_seed_w)
    values = ((root_seed_w, (n, 2, 7)), (measured_base_w, (n, 2, 7)), (common_lift_offset_m, (n,)), (measured_clearance_m, (n, 2)), (palm_offsets_m, (n, 2, 3)), (requested_palm_offsets_m, (n, 2, 3)), (angles_rad, (n, 2, 3)), (requested_angles_rad, (n, 2, 3)))
    for (value, shape) in values:
        if not isinstance(value, tenso_wrist_proje__torch.Tensor) or value.shape != shape or value.device != root_seed_w.device or (value.dtype != root_seed_w.dtype):
            raise ValueError('complete same-device same-dtype wrist projection inputs required')
    for mask in (allowed, live):
        if not isinstance(mask, tenso_wrist_proje__torch.Tensor) or mask.shape != (n,) or mask.dtype != tenso_wrist_proje__torch.bool or (mask.device != root_seed_w.device):
            raise ValueError('explicit same-device boolean allowed/live masks required')
    finite = tenso_wrist_proje__torch.isfinite(tenso_wrist_proje__torch.cat([value.reshape(-1) for (value, _) in values])).all()
    norms = tenso_wrist_proje__torch.linalg.vector_norm(tenso_wrist_proje__torch.stack((root_seed_w[..., 3:], measured_base_w[..., 3:])), dim=-1)
    unit = tenso_wrist_proje__torch.isclose(norms, tenso_wrist_proje__torch.ones_like(norms), atol=1e-05, rtol=0).all()
    angles_valid = ~(tenso_wrist_proje__torch.cat((angles_rad.reshape(-1), requested_angles_rad.reshape(-1))).abs() > tenso_wrist_proje__math.pi / 2).any()
    seed_scale = root_seed_w[..., 3:].abs().amax(-1, keepdim=True)
    nonzero = (seed_scale > 0).all()
    if not bool(finite & unit & angles_valid & nonzero):
        raise ValueError('finite wrist inputs, unit WXYZ quaternions and original intrinsic XYZ limits required')
    lower = root_seed_w.new_tensor(tenso_wrist_proje__WRIST['palm_lower_m'])
    upper = root_seed_w.new_tensor(tenso_wrist_proje__WRIST['palm_upper_m'])
    (palms, pre_projection, reasons) = tenso_wrist_proje__palm_transport(palm_offsets_m, requested_palm_offsets_m, measured_clearance_m, tenso_wrist_proje__CLEARANCE_M, tenso_wrist_proje__ACTION_CONTRACT['translation_rate_m_s'], tenso_wrist_proje__DT, lower, upper, allowed, live, vector_norm=True)
    (projected_z, safe_z) = tenso_wrist_proje__per_hand_safe_z(palms, measured_base_w, root_seed_w, common_lift_offset_m, measured_clearance_m, tenso_wrist_proje__CLEARANCE_M)
    clipped = live[:, None] & ((projected_z - palms[:, :, 2]).abs() > 1e-12)
    palms[:, :, 2] = tenso_wrist_proje__torch.where(live[:, None], projected_z, palms[:, :, 2])
    angles = tenso_wrist_proje__torch.where((allowed & live)[:, None, None], requested_angles_rad, angles_rad)
    roots = root_seed_w.clone()
    roots[..., :3] += palms
    roots[..., 2] += common_lift_offset_m[:, None]
    q = root_seed_w[..., 3:] / seed_scale
    q = q / q.norm(dim=-1, keepdim=True)
    for axis in range(3):
        step = tenso_wrist_proje__torch.zeros_like(q)
        step[..., 0] = tenso_wrist_proje__torch.cos(angles[..., axis] / 2)
        step[..., axis + 1] = tenso_wrist_proje__torch.sin(angles[..., axis] / 2)
        q = tenso_wrist_proje__multiply(q, step)
    roots[..., 3:] = q / q.norm(dim=-1, keepdim=True)
    return dict(projected_root_poses=roots, projected_angles=angles, palm_offsets_m=palms, pre_projection_palm_offsets_m=pre_projection, safe_z_offset_m=safe_z, safe_z_clipped=clipped, projection_reasons=reasons | 2 * clipped.long(), projection_source=tenso_wrist_proje__SOURCE, clearance_floor_m=tenso_wrist_proje__CLEARANCE_M, native_clearance_qualified=False, teacher_admission=False)
