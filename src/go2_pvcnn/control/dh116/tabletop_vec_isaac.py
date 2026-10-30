"""Independent CPU scenes sharing exactly one Isaac physics clock.

Import only after AppLauncher. Per-environment reset never advances physics.
"""
import copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from pxr import Usd, UsdGeom, UsdPhysics
from .linear_coupling import ACTIVE, JOINTS, MIMIC, expand
from .runtime_mimic import install_hand_mimic, configure_mimic_compliance
from .tabletop_load_parameters import LEGACY_LOAD_PARAMETERS
from .tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S, passive_velocity_limit
from .tabletop_control import prepared_approach_reference, track_root_xy, moving_goal_error
from .tabletop_task import joint_safety_failure
from .tabletop_direct import adjust_palms
from .tabletop_adaptive_backend import adaptive_step, pad_relative_positions, select_grasp_command


def rotation(q):
    return Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()


def wire_spawn_configuration(args):
    """Keep the legacy solid proxy unless an explicit rigid asset is supplied."""
    object_usd = getattr(args, 'object_usd', None)
    if object_usd is not None:
        try:
            mass = float(getattr(args, 'object_mass_kg', None))
        except (TypeError, ValueError) as error:
            raise ValueError('an external object requires an explicit finite positive mass') from error
        if not str(object_usd) or not np.isfinite(mass) or mass <= 0:
            raise ValueError('an external object requires a path and finite positive mass')
        return sim_utils.UsdFileCfg(usd_path=str(object_usd), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass))
    radius=float(getattr(args,'solid_cylinder_radius_m',.008))
    mass=float(getattr(args,'object_mass_kg',.12))
    if not np.isfinite(radius) or radius<=0 or not np.isfinite(mass) or mass<=0:
        raise ValueError('solid cylinder radius and mass must be finite and positive')
    return sim_utils.CylinderCfg(radius=radius, height=.5, axis='Y',
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.32, .16, .06)
            if getattr(args,'object_material','default') in ('rough_rod','extreme_grip') else (1., .35, .02),
            **({'roughness':1.} if getattr(args,'object_material','default') == 'extreme_grip' else {})))


def spawn_scene(sim, args, original_layout, sources, prefix, origin, *, external_wire=False):
    layout = copy.deepcopy(original_layout)
    origin = np.asarray(origin, dtype=float)
    for pose in layout['hand_roots'].values():
        pose[:3] += origin
    for key in ('wire_center', 'table_center'):
        layout[key] = np.asarray(layout[key]) + origin
    UsdGeom.Xform.Define(sim.stage, prefix)
    table = sim_utils.CuboidCfg(size=layout['table_size'],
                               collision_props=sim_utils.CollisionPropertiesCfg())
    table.func(prefix+'/Table', table, translation=layout['table_center'])
    from .tabletop_material import (hand_material_contract, bind_object_colliders, verify_collider_bindings,
                                    material_config_fields, bind_hand_materials)
    hand_profile = hand_material_contract(getattr(args,'hand_material','default'))
    hand_colliders, hand_bindings = {}, {}
    hand_material = pad_material = None
    if hand_profile:
        from pxr import UsdShade
        settings = hand_profile['hand_contact_material']
        cfg = sim_utils.RigidBodyMaterialCfg(**material_config_fields(settings))
        cfg.func(prefix+'/HandContactMaterial',cfg)
        hand_material = UsdShade.Material(sim.stage.GetPrimAtPath(prefix+'/HandContactMaterial'))
        if 'pad_compliance' in settings:
            cfg=sim_utils.RigidBodyMaterialCfg(**material_config_fields(settings,compliant=True))
            cfg.func(prefix+'/PadContactMaterial',cfg)
            pad_material=UsdShade.Material(sim.stage.GetPrimAtPath(prefix+'/PadContactMaterial'))
    hands, sensors = [], []
    for side in ('left', 'right'):
        pose = layout['hand_roots'][side].copy()
        pose[2] += .04
        cfg = ArticulationCfg(prim_path=f'{prefix}/{side}Hand',
            spawn=sim_utils.UsdFileCfg(usd_path=sources[side], activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                    enabled_self_collisions=False, solver_position_iteration_count=args.position_iterations,
                    solver_velocity_iteration_count=getattr(args,'velocity_iterations',4))),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:]),
                joint_pos=dict(zip(JOINTS, map(float, expand(
                    layout['active_joints'][side] if args.control_mode == 'adaptive_grasp'
                    else prepared_approach_reference(layout['active_joints'][side])))))),
            actuators={'active': ImplicitActuatorCfg(joint_names_expr=list(ACTIVE),
                stiffness=10., damping=.2,
                effort_limit_sim=(getattr(args,'load_parameters',None) or LEGACY_LOAD_PARAMETERS).effort_limit_nm if args.control_mode == 'adaptive_grasp' else .15,
                velocity_limit_sim=ACTIVE_VELOCITY_LIMIT_RAD_S),
                'passive': ImplicitActuatorCfg(joint_names_expr=[n for n in JOINTS if n not in ACTIVE],
                    stiffness=0., damping=0., effort_limit_sim=0.,
                    # A low solver brake on passive followers competes with
                    # mimic/contact correction. Keep active drives at .3.
                    velocity_limit_sim=passive_velocity_limit(args.control_mode,
                        getattr(args, 'passive_velocity_limit_rad_s', 3.)))})
        if getattr(args,'palm_transport','fixed_pose')=='driven_carriage':
            from .tabletop_palm_mount import JOINT_NAMES,CONTRACT as WRIST,ANGULAR_JOINT_NAMES,ROTATION_CONTRACT
            cfg.actuators['wrist']=ImplicitActuatorCfg(joint_names_expr=list(JOINT_NAMES),
                stiffness=WRIST['stiffness_n_m'],damping=WRIST['damping_n_s_m'],
                effort_limit_sim=WRIST['effort_limit_n'],velocity_limit_sim=WRIST['velocity_limit_m_s'])
            if getattr(args,'free_wrist_grasp',False):
                cfg.actuators['wrist_rotation']=ImplicitActuatorCfg(joint_names_expr=list(ANGULAR_JOINT_NAMES),
                    stiffness=ROTATION_CONTRACT['stiffness_nm_rad'],damping=ROTATION_CONTRACT['damping_nms_rad'],
                    effort_limit_sim=ROTATION_CONTRACT['effort_limit_nm'],
                    velocity_limit_sim=ROTATION_CONTRACT['velocity_limit_rad_s'])
        hands.append(Articulation(cfg))
        if getattr(args,'palm_transport','fixed_pose')=='driven_carriage':
            from .tabletop_palm_mount import author_mount
            author_mount(sim.stage,f'{prefix}/{side}Hand',rotation=bool(getattr(args,'free_wrist_grasp',False)))
        if hand_material is not None:
            paths,binding_report=bind_hand_materials(sim.stage,f'{prefix}/{side}Hand',hand_material,
                                                    hand_profile['hand_contact_material'],pad_material)
            hand_colliders[side] = paths
            hand_bindings[side] = binding_report
        install_hand_mimic(sim.stage, f'{prefix}/{side}Hand')
        configure_mimic_compliance(sim.stage, f'{prefix}/{side}Hand', float(args.mimic_frequency), 1.,
            allow_rigid=getattr(args,'load_parameters',None) is not None)
        for link in (() if external_wire else ('finger14_link', 'finger23_link')):
            sensors.append(ContactSensor(ContactSensorCfg(prim_path=f'{prefix}/{side}Hand/{link}',
                track_contact_points=bool(getattr(args,'tensor_pipeline',False)),
                filter_prim_paths_expr=[prefix+'/Wire'], update_period=0.)))
    if external_wire:
        return dict(hands=hands,sensors=[],wire=None,layout=layout,prefix=prefix,origin=origin,
                    object_material_colliders=[],hand_material_colliders=hand_colliders,
                    hand_material_bindings=hand_bindings)
    wire = RigidObject(RigidObjectCfg(prim_path=prefix+'/Wire',
        spawn=wire_spawn_configuration(args),
        init_state=RigidObjectCfg.InitialStateCfg(pos=layout['wire_center'])))

    from .tabletop_material import material_contract, bind_object_colliders
    profile=material_contract(getattr(args,'object_material','default'))
    material_paths=[]
    if profile:
        from pxr import UsdShade
        settings=profile['object_contact_material']
        cfg=sim_utils.RigidBodyMaterialCfg(**{k:settings[k] for k in (
            'static_friction','dynamic_friction','restitution',
            'friction_combine_mode','restitution_combine_mode')})
        material_path=prefix+'/ObjectContactMaterial'
        cfg.func(material_path,cfg)
        material_paths=bind_object_colliders(sim.stage,prefix+'/Wire',
            UsdShade.Material(sim.stage.GetPrimAtPath(material_path)))
        verify_collider_bindings(sim.stage,material_paths,settings)

    return dict(hands=hands, sensors=sensors, wire=wire, layout=layout, prefix=prefix, origin=origin,
                object_material_colliders=material_paths,hand_material_colliders=hand_colliders,
                hand_material_bindings=hand_bindings)


def bind_scene(sim, args, scene, seed_state):
    hands, sensors, wire, layout, prefix = (scene[k] for k in ('hands','sensors','wire','layout','prefix'))
    dt, decimation = 1/args.physics_hz, args.physics_hz//60
    ids = [[h.joint_names.index(n) for n in ACTIVE] for h in hands]
    # Cache convex-hull vertices in each link frame. Runtime transforms come
    # from PhysX link poses, never stale USD world transforms or COM poses.
    shapes = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for hi, side in enumerate(('left', 'right')):
        base = f'{prefix}/{side}Hand'
        for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(base), Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if '/collisions/' not in path or not prim.IsA(UsdGeom.Mesh):
                continue
            link = path[len(base)+1:].split('/')[0]
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(
                sim.stage.GetPrimAtPath(base+'/'+link)).GetInverse()
            points = (np.c_[points, np.ones(len(points))] @ np.asarray(relative))[:, :3]
            points = np.unique(points, axis=0)
            points = points[ConvexHull(points).vertices]
            shapes.append((hi, hands[hi].body_names.index(link), points))
    if len(shapes) != 34:
        raise ValueError(f'expected 34 collision meshes, found {len(shapes)}')
    if args.control_mode == 'adaptive_grasp':
        from .native_scene import _PAD_LOCAL_POINTS
        pad_body_ids = [[hand.body_names.index(name) for name in ('finger14_link', 'finger23_link')]
                        for hand in hands]
        pad_local_points = np.asarray([_PAD_LOCAL_POINTS[side][digit]
            for side in ('left', 'right') for digit in ('thumb', 'index')])


    class IsaacBackend:
        def __init__(self):
            self.dt, self.decimation = dt, decimation
            self.adaptive = args.control_mode == 'adaptive_grasp'
            self.grasp = np.concatenate([layout['active_joints'][s] for s in ('left','right')])
            if self.adaptive:
                self.grasp = select_grasp_command(self.grasp, seed_state)
            self.seed = copy.deepcopy(seed_state)
            self.seed['wire_pose'][:3] += scene['origin']
            self.initial_command = self.seed['initial_command'].copy()
            self.forces = np.zeros(4)
            self.palm_z = np.zeros(2)
            self.clearance = .005
            self.hand_clearance = np.zeros(2)
            self.adaptive_palm_offset = np.zeros((2, 3))
            self.grip_stiffness = np.full(2, 10.)
        def _write_grip_stiffness(self, index):
            hand = hands[index]
            stiffness = torch.full((1, len(ids[index])), float(self.grip_stiffness[index]),
                                   device=args.device, dtype=torch.float32)
            hand.write_joint_stiffness_to_sim(stiffness, joint_ids=ids[index])
            # IsaacLab's simulator setter explicitly does not update the
            # actuator model. Keep its approximate PD torque diagnostics honest.
            hand.actuators['active'].stiffness[:] = stiffness
        def reset(self, offset):
            self.root_xy = np.zeros(2) if self.seed is None else self.seed['root_xy'].copy()
            self.palm_z = np.zeros(2)
            self.adaptive_palm_offset[:] = 0.
            self.grip_stiffness[:] = 10.
            self.hand_clearance[:] = 0.  # No descent before fresh collision measurements.
            for hi, (hand, side) in enumerate(zip(hands, ('left', 'right'))):
                hand.reset()
                if self.adaptive:
                    self._write_grip_stiffness(hi)
                q = hand.data.default_joint_pos.clone()
                qd = torch.zeros_like(q)
                named_ids = [hand.joint_names.index(n) for n in JOINTS]
                q[:, named_ids] = torch.tensor(
                    expand(prepared_approach_reference(layout['active_joints'][side])) if self.seed is None
                    else self.seed['joint_positions'][hi], device=args.device, dtype=torch.float32)
                if self.seed is not None:
                    qd[:, named_ids] = torch.tensor(self.seed['joint_velocities'][hi], device=args.device, dtype=torch.float32)
                hand.write_joint_state_to_sim(q, qd)
                hand.set_joint_position_target(q[:, ids[hi]], joint_ids=ids[hi])
                pose = layout['hand_roots'][side].copy()
                pose[:2] += self.root_xy
                pose[2] += offset
                hand.write_root_pose_to_sim(torch.tensor(pose[None], device=args.device, dtype=torch.float32))
                hand.write_root_velocity_to_sim(torch.zeros((1, 6), device=args.device))
                hand.write_data_to_sim()
            wire.reset()
            wire.write_root_pose_to_sim(wire.data.default_root_state[:, :7].clone() if self.seed is None
                else torch.tensor(self.seed['wire_pose'][None],device=args.device,dtype=torch.float32))
            wire.write_root_velocity_to_sim(torch.zeros((1, 6), device=args.device) if self.seed is None
                else torch.tensor(self.seed['wire_velocity'][None],device=args.device,dtype=torch.float32))
            for sensor in sensors:
                sensor.reset()
            sim.physics_sim_view.update_articulations_kinematic()
            for hand in hands:
                hand.update(dt)
            wire.update(dt)
            self.forces[:] = 0.
        def apply(self, offset, command, *, phase, palm_motion_allowed,
                  palm_action=None, grip_action=None):
            if self.adaptive:
                new_palm, new_stiffness = adaptive_step(
                    palm_offset=self.adaptive_palm_offset, stiffness=self.grip_stiffness,
                    palm_action=palm_action, grip_action=grip_action,
                    hand_clearance=self.hand_clearance, dt=dt,
                    palm_motion_allowed=palm_motion_allowed,parameters=getattr(args,'load_parameters',None))
                changed = new_stiffness != self.grip_stiffness
                self.adaptive_palm_offset, self.grip_stiffness = new_palm, new_stiffness
                for hi in np.flatnonzero(changed):
                    self._write_grip_stiffness(int(hi))
            else:
                if palm_action is not None or grip_action is not None:
                    raise ValueError('palm/grip actions require adaptive_grasp mode')
                # Preserve legacy controller tracking and close adjustment.
                self.root_xy = track_root_xy(self.root_xy,
                    wire.data.root_pos_w[0].detach().cpu().numpy(), layout['wire_center'], phase, dt, allowed=palm_motion_allowed)
                if self.seed is not None:
                    self.root_xy = np.clip(self.root_xy, self.seed['root_xy']-.003, self.seed['root_xy']+.003)
                    self.palm_z = adjust_palms(self.palm_z, self.forces, self.clearance,
                                              phase, dt, allowed=palm_motion_allowed)
            for hi, (hand, side) in enumerate(zip(hands, ('left', 'right'))):
                pose = layout['hand_roots'][side].copy()
                pose[:2] += self.root_xy
                pose[2] += offset + self.palm_z[hi]
                if self.adaptive:
                    pose[:3] += self.adaptive_palm_offset[hi]
                hand.write_root_pose_to_sim(torch.tensor(pose[None], device=args.device, dtype=torch.float32))
                hand.set_joint_position_target(torch.tensor(command[hi*6:hi*6+6][None],
                    device=args.device, dtype=torch.float32), joint_ids=ids[hi])
                hand.write_data_to_sim()
        def refresh(self):
            for hand in hands:
                hand.update(dt)
            wire.update(dt)
            for i, sensor in enumerate(sensors):
                sensor.update(dt, force_recompute=True)
                self.forces[i] = float(torch.linalg.vector_norm(sensor.data.force_matrix_w).item())

        def measure(self):
            poses = [h.data.body_link_pose_w[0].detach().cpu().numpy() for h in hands]
            heights = [float((p @ rotation(poses[i][j, 3:]).T + poses[i][j, :3])[:, 2].min())
                       for i, j, p in shapes]
            lowest = int(np.argmin(heights))
            self.clearance = heights[lowest]-.30
            self.hand_clearance = np.array([min(height for height, shape in zip(heights, shapes)
                                               if shape[0] == i)-.30 for i in range(2)])
            hi, bi, _ = shapes[lowest]
            diagnostics = []
            joint_failure = None
            for hand_index, hand in enumerate(hands):
                q = hand.data.joint_pos[0].detach().cpu().numpy()
                named = dict(zip(hand.joint_names, map(float, q)))
                errors = {f: named[f]-r*named[m]-o for f, m, r, o in MIMIC}
                diagnostics.append(dict(joints_rad=named, mimic_error_rad=errors,
                    joint_vel_rad_s=hand.data.joint_vel[0].detach().cpu().tolist(),
                    applied_torque_nm=hand.data.applied_torque[0].detach().cpu().tolist(),
                    torque_measurement='implicit_pd_estimate_not_measured_drive_torque',
                    force_on_pads_from_object_normal_w_n=[s.data.force_matrix_w.reshape(-1,3).sum(dim=0).detach().cpu().tolist()
                        for s in sensors[hand_index*2:hand_index*2+2]],
                    root_pose=hand.data.root_pose_w[0].detach().cpu().tolist()))
                failure = joint_safety_failure(q,
                    hand.data.joint_pos_limits[0].detach().cpu().numpy(), list(errors.values()))
                joint_failure = joint_failure or failure
            wp = wire.data.root_pos_w[0].detach().cpu().numpy()
            wq = wire.data.root_quat_w[0].detach().cpu().numpy()
            axis_z = abs(rotation(wq)[2, 1])
            bottom = float(wp[2] - .25*axis_z - .008*np.sqrt(max(0., 1-axis_z**2)))
            roots = np.stack([h.data.root_pos_w[0].detach().cpu().numpy() for h in hands])
            goal = np.stack([layout['hand_roots'][s][:3] for s in ('left', 'right')])
            measured = dict(q=np.concatenate([h.data.joint_pos[0,k].detach().cpu().numpy()
                                          for h,k in zip(hands,ids)]),
                qd=np.concatenate([h.data.joint_vel[0,k].detach().cpu().numpy() for h,k in zip(hands,ids)]),
                forces=self.forces.copy(), clearance=heights[lowest]-.30,
                object_relative=wp-roots.mean(axis=0), object_quat=wq.copy(),
                object_velocity=wire.data.root_vel_w[0].detach().cpu().numpy().copy(),
                bottom=bottom, approach_error=moving_goal_error(roots, goal, wp, layout['wire_center']),
                joint_failure=joint_failure, diagnostics=diagnostics,
                mimic_errors=np.array([e for h in diagnostics for e in h['mimic_error_rad'].values()]),
                palm_adjustment=np.r_[self.root_xy-self.seed['root_xy'],self.palm_z]
                    if self.seed is not None else np.zeros(4),
                hand_clearance=self.hand_clearance.copy(),
                lowest_link=f'{("left", "right")[hi]}/{hands[hi].body_names[bi]}')
            if self.adaptive:
                pad_poses = np.asarray([poses[i][body] for i in range(2) for body in pad_body_ids[i]])
                measured.update(adaptive_palm_offset=self.adaptive_palm_offset.copy(),
                    grip_stiffness=self.grip_stiffness.copy(),
                    pad_relative=pad_relative_positions(pad_poses, pad_local_points, wp))
            return measured

    return IsaacBackend()


def advance_world(sim, backends, requests, *, render=False):
    for backend, request in zip(backends, requests):
        if request is not None:
            offset, command, kwargs = request
            backend.apply(offset, command, **kwargs)
    sim.step(render=render)
    for backend in backends:
        backend.refresh()


def isolate_scenes(stage, scenes):
    """Exclude cross-environment contacts even if an object escapes its table."""
    groups = []
    for i, scene in enumerate(scenes):
        group = UsdPhysics.CollisionGroup.Define(stage, f'/World/collisionGroups/env_{i}')
        group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(scene['prefix'])
        groups.append(group)
    for i, group in enumerate(groups):
        group.CreateFilteredGroupsRel().SetTargets([g.GetPath() for j, g in enumerate(groups) if j != i])
