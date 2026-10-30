"""Device-resident rigid physics adapter; host scene construction remains unchanged."""
import torch
import time
from .linear_coupling import MIMIC
from .tabletop_object_contact import link_groups
from .tabletop_contact_geometry import aggregate_contact_geometry

ROOT_POSE_WRITE_CONTRACT = dict(version='changed_fixed_root_pose_v1',
    comparison='last_submitted_float32', reset='always_write_selected')
PALM_TARGET_CONTRACT = dict(version='synchronized_measured_clearance_v2',
    synchronization='world_z_exploration_offset_preserve_authored_roots',
    clearance='highest_per_hand_measured_safe_target',
    descent_backlog='cancel_unsafe_clearance_or_release_hold_target_even_when_motion_paused',
    clearance_undershoot='restore_signed_deficit_even_when_motion_paused',
    reposition_goal_feedback='raise_moving_goal_after_safety_projection_preserve_pending_release_goal',
    scope='live_coordinated_or_z_down')


def contact_effort_permission(task, live):
    from .tabletop_coordinated_grasp import CONTRACT as CONTROL
    eligible=live[:,None].expand(-1,2).clone()
    # Position pause/lead projection does not disable pressure generation.
    # Only the existing severe coupling guard withdraws effort permission.
    eligible &= task.m['mimic_errors'].reshape(task.n,2,5).abs().amax(2)<CONTROL['severe_coupling_rad']
    eligible &= ~task.pressure_limited & ~task.force_guard_conflict
    eligible &= task.m['q'][:,[1,7]]>=task.low[[1,7]]+CONTROL['thumb_limit_margin_rad']
    if hasattr(task,'palm_reposition'):
        eligible &= ~((task.palm_reposition.state==1)|(task.palm_reposition.state==2))
    return eligible


def write_contact_effort(hand, joint_ids, effort, total_limit, *, env_ids=None):
    """Independent actuation + implicit drive share one motor effort budget."""
    if not torch.isfinite(effort).all() or total_limit<=0:
        raise ValueError('finite effort and positive total motor budget required')
    effort=effort.float().clamp(-total_limit,total_limit)
    drive_budget=(total_limit-effort.abs()).clamp_min(0.)
    # Implicit actuator's estimated applied_torque is not a measured torque.
    # Only this PhysX drive limit plus bounded explicit effort enforces the sum.
    hand.write_joint_effort_limit_to_sim(drive_budget,joint_ids=joint_ids,env_ids=env_ids)
    hand.set_joint_effort_target(effort,joint_ids=joint_ids,env_ids=env_ids)
    return drive_budget


def rotated_heights(poses, points):
    poses = poses.double()
    w, x, y, z = poses[..., 3:].unbind(-1)
    row = torch.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)
    return row @ points.double().T + poses[..., 2, None]


def rotated_points(poses, points):
    poses = poses.double()
    xyz = poses[..., 4:]
    points = points.expand_as(xyz)
    cross = 2*torch.linalg.cross(xyz, points)
    return points + poses[..., 3:4]*cross + torch.linalg.cross(xyz, cross) + poses[..., :3]


def signed_hand_vertical_force(vertical_components):
    """Reduce [environment, hand, (normal, friction)] to signed hand Fz."""
    if vertical_components.ndim != 3 or vertical_components.shape[-1] != 2:
        raise ValueError('vertical components must have shape (N, 2, 2)')
    return vertical_components.sum(dim=2)


def signed_hand_normal_vertical_force(vertical_components):
    """Reduce [environment, hand, (normal, friction)] to normal-only Fz."""
    if (vertical_components.ndim != 3 or tuple(vertical_components.shape[-2:]) != (2,2)
            or not torch.isfinite(vertical_components).all()):
        raise ValueError('vertical components must have shape (N, 2, 2) and be finite')
    return vertical_components[...,0]


def signed_hand_lift_vertical_force(vertical_components):
    """Reduce per-hand normal and friction components for lift checks."""
    if (vertical_components.ndim != 3 or tuple(vertical_components.shape[-2:]) != (2,2)
            or not torch.isfinite(vertical_components).all()):
        raise ValueError('vertical components must be N x 2 x 2 and finite')
    return vertical_components[..., 0] + vertical_components[..., 1]


def contact_geometry_from_sensors(sensors, object_center_w, object_axis_w=None):
    """Reduce ordered PhysX contact reports to per-hand geometry."""
    if not sensors or len(sensors) % 2:
        raise ValueError('contact sensors must contain an even nonzero hand/link list')
    positions=[]; forces=[]
    for sensor in sensors:
        position=sensor.data.contact_pos_w
        force=sensor.data.force_matrix_w
        if position is None or force is None:
            raise ValueError('contact point and normal force reports are required')
        if position.ndim != 4 or position.shape[1:3] != (1,1) or position.shape[-1] != 3:
            raise ValueError('contact positions must be N x 1 x 1 x 3')
        if force.shape != position.shape:
            raise ValueError('contact force report shape must match contact positions')
        positions.append(position[:,0,0,:])
        forces.append(force[:,0,0,:])
    n=positions[0].shape[0]
    positions=torch.stack(positions,dim=1).reshape(n,2,-1,3)
    forces=torch.stack(forces,dim=1).reshape(n,2,-1,3)
    return aggregate_contact_geometry(positions,forces,object_center_w,object_axis_w=object_axis_w)


def constrain_palm_z_movement(movement, clearances, clearance_floor_m):
    """Keep palm descent above the configured floor without changing X/Y."""
    if not torch.isfinite(torch.as_tensor(clearance_floor_m)) or clearance_floor_m < 0:
        raise ValueError('clearance floor must be finite and nonnegative')
    result=movement.clone()
    result[:,:,2]=torch.maximum(result[:,:,2],
        -(clearances-clearance_floor_m).clamp_min(0))
    return result


class TensorWorld:
    def __init__(self, world):
        if not world.full_hand_contact:
            raise ValueError('tensor training requires rigid full-hand contact')
        self.w = world
        self.continuous_motion=bool(getattr(world.args,'continuous_motion',False))
        self.n, self.dt, self.device = world.n, world.dt, world.device
        self.decimation = world.args.physics_hz//60
        self._physics_steps = 0
        self.parameters = world.parameters
        self.hands, self.wire = world.hands, world.wire
        self.origins = self.t(world.origins)
        self.limits = self.t(world.limits)
        self.palm_limit = self.t([.003, .003, .004])
        self.root_seed = self.t([world.layout['hand_roots'][s] for s in ('left','right')])
        self.root_seed = self.root_seed[None].repeat(self.n,1,1)
        self.root_seed[:,:,:3] += self.origins[:,None]
        self.root_seed[:,:,:2] += self.t(world.seed['root_xy'])
        self.initial_offset = float(world.seed['root_offset'])
        self.seed_q = []
        self.seed_qd = []
        for hi, hand in enumerate(self.hands):
            q = hand.data.default_joint_pos.clone()
            q[:,world.named_ids[hi]] = self.t(world.seed['joint_positions'][hi], torch.float32)
            qd = torch.zeros_like(q)
            qd[:,world.named_ids[hi]] = self.t(world.seed['joint_velocities'][hi], torch.float32)
            self.seed_q.append(q); self.seed_qd.append(qd)
        self.seed_wire = self.t(world.seed['wire_pose'])[None].repeat(self.n,1)
        self.seed_wire[:,:3] += self.origins
        self.seed_velocity = self.t(world.seed['wire_velocity'])[None].repeat(self.n,1)
        self.goal_roots = self.t([world.layout['hand_roots'][s][:3] for s in ('left','right')])[None]+self.origins[:,None]
        self.spawn = self.t(world.layout['wire_center'])[None]+self.origins
        self.palms = self.t(world.palms)
        self.gains = self.t(world.gains)
        self.root_targets = self.t(world.root_targets)
        self.commands = self.t(world.commands)
        self.shapes = [(h,b,self.t(p)) for h,b,p in world.shapes]
        self.shape_hands = [[i for i,(h,_,_) in enumerate(self.shapes) if h==hand] for hand in range(2)]
        groups = link_groups(world.sensor_links)
        self.group_matrix = self.t([[float(g==i) for i in range(6)] for g in groups])
        self.pad_indices = [h*len(world.sensor_links)+world.sensor_links.index(link)
                            for h in range(2) for link in ('finger14_link','finger23_link')]
        self.pad_points = self.t(world.pad_points)
        self.mimic_ids = [[(hand.joint_names.index(f),hand.joint_names.index(m),r,o)
                          for f,m,r,o in MIMIC] for hand in self.hands]
        self.link_loads = torch.zeros((self.n,2,len(groups)),device=self.device,dtype=torch.float64)
        self.forces = torch.zeros((self.n,4),device=self.device,dtype=torch.float64)
        self.vertical_components=torch.zeros((self.n,2,2),device=self.device,dtype=torch.float64)
        self.adapter_fault_bits=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.transport_warning_mask=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.contact_center_w=torch.full((self.n,2,3),float('nan'),device=self.device,dtype=torch.float64)
        self.contact_normal_w=torch.full_like(self.contact_center_w,float('nan'))
        self.contact_normal_load_n=torch.zeros((self.n,2),device=self.device,dtype=torch.float64)
        self.contact_valid=torch.zeros((self.n,2),device=self.device,dtype=torch.bool)
        self.contact_all_below_centerline=torch.zeros((self.n,2),device=self.device,dtype=torch.bool)
        self.contact_count=torch.zeros((self.n,2),device=self.device,dtype=torch.long)
        gravity=world.sim.cfg.gravity
        if gravity[0]!=0 or gravity[1]!=0 or gravity[2]>=0:
            raise ValueError('vertical force telemetry requires downward world-Z gravity')
        self.object_weight_n=self.wire.root_physx_view.get_masses().to(self.device).double().reshape(self.n)*(-gravity[2])
        for sensor in world.sensors:
            if list(sensor.contact_physx_view.sensor_paths)!=list(sensor.body_physx_view.prim_paths):
                raise ValueError('vertical force sensor ordering mismatch')
            if (sensor.contact_physx_view.filter_count!=1 or not sensor.cfg.track_friction_forces
                    or not sensor.cfg.track_contact_points):
                raise ValueError('object-only normal, friction, and contact-point sensor required')
        self.refresh(sensors=False)
        self._root_pose_written = self.roots.float().clone()

    def t(self, value, dtype=torch.float64):
        return torch.as_tensor(value,device=self.device,dtype=dtype)

    def write_root_pose(self, hi, pose, env_ids=None):
        """Avoid redundant fixed-base state writes; explicit resets always write.

        Compare submitted float32 poses, not PhysX's normalized readback or the
        higher precision controller target. Rewriting an unchanged base changed
        contact dynamics in the saved 4096-environment fault-action replay.
        """
        pose = pose.float()
        if env_ids is not None:
            self.hands[hi].write_root_pose_to_sim(pose,env_ids=env_ids)
            self._root_pose_written[env_ids,hi] = pose
            return
        ids = (self._root_pose_written[:,hi] != pose).any(-1).nonzero().flatten()
        if ids.numel():
            self.hands[hi].write_root_pose_to_sim(pose[ids],env_ids=ids)
        self._root_pose_written[:,hi].copy_(pose)

    def refresh(self, sensors=True):
        w = self.w
        continuous=getattr(self,'continuous_motion',False)
        if continuous and sensors:self.adapter_fault_bits.zero_()
        for hand in self.hands: hand.update(self.dt)
        self.wire.update(self.dt)
        self.q = torch.stack([h.data.joint_pos for h in self.hands],1).double()
        self.qd = torch.stack([h.data.joint_vel for h in self.hands],1).double()
        self.poses = torch.stack([h.data.body_link_pose_w for h in self.hands],1).double()
        self.roots = torch.stack([h.data.root_pose_w for h in self.hands],1).double()
        if getattr(self.w,'mounts',None) is not None:
            self.roots=torch.stack([h.data.body_link_pose_w[:,mount.base_id] for h,mount in zip(self.hands,self.w.mounts)],1).double()
        self.wp = self.wire.data.root_pos_w.double()
        self.wq = self.wire.data.root_quat_w.double()
        self.wv = self.wire.data.root_vel_w.double()
        qw,qx,qy,qz=self.wq.unbind(1)
        axis_y=torch.stack((2*(qx*qy-qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz+qw*qx)),1)
        if continuous:
            # A finite all-zero quaternion is still invalid. Validate reset
            # readback as well as the ordinary sensor refresh.
            invalid_object=(~torch.isfinite(self.wp).all(1)|~torch.isfinite(self.wq).all(1)
                |~torch.isfinite(axis_y).all(1)|(self.wq.norm(dim=1)<=0)|(axis_y.norm(dim=1)<=0))
            self.adapter_fault_bits.bitwise_or_(invalid_object.long())
        if sensors:
            for sensor in w.sensors: sensor.update(self.dt,force_recompute=True)
            vectors = torch.stack([s.data.force_matrix_w[:,0,0,:] for s in w.sensors],1).double()
            self.link_loads = vectors.reshape(self.n,2,-1,3).norm(dim=-1)
            self.forces = vectors[:,self.pad_indices].norm(dim=-1)
            geometry_axis=axis_y
            if continuous:
                self.adapter_fault_bits.bitwise_or_((~torch.isfinite(vectors).flatten(1).all(1)).long()*2)
                # Only derived geometry uses a placeholder axis for excluded
                # bad rows. Preserve the original quaternion/center readback.
                geometry_axis=torch.where(invalid_object[:,None],axis_y.new_tensor([0.,1.,0.]),axis_y)
            geometry=contact_geometry_from_sensors(w.sensors,self.wp,geometry_axis)
            if continuous:
                for key in ('valid','all_below_centerline'):geometry[key][invalid_object]=False
                for key in ('contact_center_w','contact_normal_w'):geometry[key][invalid_object]=float('nan')
                geometry['contact_count'][invalid_object]=0
            self.contact_center_w=geometry['contact_center_w']
            self.contact_normal_w=geometry['contact_normal_w']
            self.contact_normal_load_n=geometry['normal_load_n']
            self.contact_valid=geometry['valid']
            self.contact_all_below_centerline=geometry['all_below_centerline']
            self.contact_count=geometry['contact_count']
            from .tabletop_vertical_force import object_vertical_components
            friction=torch.stack([s.data.friction_forces_w[:,0,0,:] for s in w.sensors],1).double()
            if continuous:
                self.adapter_fault_bits.bitwise_or_((~torch.isfinite(friction).flatten(1).all(1)).long()*4)
            self.vertical_components=object_vertical_components(
                vectors.reshape(self.n,2,-1,3),friction.reshape(self.n,2,-1,3))
        heights = torch.stack([rotated_heights(self.poses[:,h,b],p).amin(-1)
                               for h,b,p in self.shapes],1)
        self.clearances = torch.stack([heights[:,idx].amin(1) for idx in self.shape_hands],1)-.30
        az = (2*(self.wq[:,2]*self.wq[:,3]+self.wq[:,0]*self.wq[:,1])).abs()
        self.bottom = self.wp[:,2]-.25*az-.008*(1-az.square()).clamp_min(0).sqrt()
        self.errors = torch.stack([torch.stack([self.q[:,h,f]-r*self.q[:,h,m]-o
                                    for f,m,r,o in ids],1) for h,ids in enumerate(self.mimic_ids)],1)
        pads = torch.cat([self.poses[:,h,w.pad_ids[h]] for h in range(2)],1)
        self.pad_relative = rotated_points(pads,self.pad_points)-self.wp[:,None]
        if continuous and getattr(w,'mounts',None) is not None:
            # Also runs for reset's sensors=False readback, before another
            # sim.step. Clearing the previous fault must not certify a bad reset.
            from .tabletop_palm_mount import check_transport, transport_safe
            valid=check_transport(self.hands,w.mounts,self.root_targets,return_validity=True)
            self.adapter_fault_bits.bitwise_or_((~valid).long()*8)
            velocity=torch.stack([h.data.body_lin_vel_w[:,mount.base_id]
                                  for h,mount in zip(self.hands,w.mounts)],1)
            self.transport_warning_mask.copy_(~transport_safe(self.roots,self.root_targets,velocity))
        return self.measure()

    def measure(self):
        from .tabletop_tensor_task import FAILURE_CODES
        continuous=getattr(self,'continuous_motion',False)
        failure = torch.zeros(self.n,device=self.device,dtype=torch.long)
        for hi in range(2):
            q, bounds, error = self.q[:,hi], self.limits[:,hi], self.errors[:,hi]
            for name, condition in (
                ('nonfinite_joint_state',~torch.isfinite(q).all(1)|~torch.isfinite(error).all(1)),
                ('joint_limit',((q<bounds[:,:,0]-.02)|(q>bounds[:,:,1]+.02)).any(1)),
                ('mimic_error',(error.abs().amax(1)>.02)&~torch.full_like(failure,continuous,dtype=torch.bool))):
                failure = torch.where((failure==0)&condition,FAILURE_CODES[name],failure)
        goals = self.goal_roots.clone()
        goals[:,:,:2] += (self.wp-self.spawn)[:,None,:2]
        palm_measure=self.palms
        if getattr(self.w,'mounts',None) is not None:
            from .tabletop_palm_mount import measured_offset
            palm_measure=measured_offset(self.roots,self.root_targets,self.palms)
        result=dict(q=torch.cat([self.q[:,h,self.w.ids[h]] for h in range(2)],1),
            qd=torch.cat([self.qd[:,h,self.w.ids[h]] for h in range(2)],1),
            forces=self.forces,clearance=self.clearances.amin(1),
            object_relative=self.wp-self.roots[:,:,:3].mean(1),object_quat=self.wq,
            object_velocity=self.wv,bottom=self.bottom,
            approach_error=(self.roots[:,:,:3]-goals).norm(dim=-1).amax(1),
            mimic_errors=self.errors.flatten(1),hand_clearance=self.clearances,
            adaptive_palm_offset=palm_measure,grip_stiffness=self.gains,pad_relative=self.pad_relative,
            commanded_palm_offset_m=self.palms,
            contact_center_w=self.contact_center_w,
            contact_center_relative_m=self.contact_center_w-self.wp[:,None,:],
            contact_normal_w=self.contact_normal_w,
            contact_normal_load_n=self.contact_normal_load_n,
            contact_valid=self.contact_valid,
            contact_all_below_centerline=self.contact_all_below_centerline,
            contact_count=self.contact_count,
            vertical_hand_force_n=(self.vertical_components[...,0] if continuous else
                                   signed_hand_normal_vertical_force(self.vertical_components)),
            object_link_loads=self.link_loads,grouped_object_loads=self.link_loads@self.group_matrix,
            hand_object_loads=self.link_loads.sum(-1),joint_failure=failure)
        if continuous:result['adapter_fault_bits']=self.adapter_fault_bits
        if continuous and hasattr(self,'transport_warning_mask'):
            result['transport_warning']=self.transport_warning_mask
        return result

    def advance(self, task, allowed, live):
        policy_control = getattr(task,"policy_control",False)
        if hasattr(task,'policy_lift_hold') and not getattr(task,'manual_digit_force_balance',False):
            hold=task.policy_lift_hold
            paused=live&hold.active&(task.phase==3)&(hold.paused|task.paused)
            # Cancel the queued upward drive target at measured carriage pose;
            # freezing only new increments leaves the old target still pulling.
            self.palms=torch.where(paused[:,None,None],task.m['adaptive_palm_offset'],self.palms)
        if policy_control:
            from .tabletop_policy_control import CONTRACT as POLICY
            target = task.palm_targets
            if hasattr(task,'manual_pinch_home'):
                from .tabletop_manual_pinch_geometry import measured_target
                target=measured_target(task,self,target)
                task.manual_pinch_compensation[live]=(target-task.palm_targets)[live]
            palm_lower = self.t(POLICY["palm_lower_m"])
            palm_upper = self.t(POLICY["palm_upper_m"])
            rate = POLICY["wrist_rate_m_s"]
        else:
            target = task.palm_actions*self.palm_limit
            palm_lower, palm_upper = -self.palm_limit, self.palm_limit
            rate = .002
        movement = (target-self.palms).clamp(-rate*self.dt,rate*self.dt)
        if policy_control:
            # The physical transport guard measures Cartesian speed, not each
            # axis independently. Diagonal commands must share one speed budget.
            norm=movement.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            movement=movement*((rate*self.dt)/norm).clamp_max(1.)
        clearance_floor=getattr(task,'diagnostic_palm_descent_clearance_m',
                                getattr(task,'palm_descent_clearance_m',.0021))
        movement=constrain_palm_z_movement(movement,self.clearances,clearance_floor)
        self.palms = torch.where((allowed&live)[:,None,None],
            torch.maximum(torch.minimum(self.palms+movement,palm_upper),palm_lower),self.palms)
        synchronized=getattr(task,'coordinated_control',torch.zeros_like(live))
        if getattr(task,'palm_exploration_axis',None)=='z_down':
            synchronized=torch.ones_like(live)
        if policy_control:
            synchronized=torch.zeros_like(live)
        safety_controlled=torch.ones_like(live) if policy_control else synchronized
        if hasattr(task,'phase'):
            synchronized=synchronized&(task.phase<3)
        if hasattr(task,'close_grasp_hold'):
            synchronized &= ~task.close_grasp_hold.active.any(1)
        if policy_control or hasattr(task,'coordinated_control') or getattr(task,'palm_exploration_axis',None)=='z_down':
            # Roots are measured bases (including driven-carriage lag). Express
            # their safe heights in the exploration-offset frame of the *new*
            # common lift target, preserving the authored left/right root gap.
            measured_z=self.roots[:,:,2]-self.root_seed[:,:,2]-task.offset[:,None]
            # Finger motion can lower a hard link even with palm descent held.
            # Restore the signed deficit so the target clears that link rather
            # than holding its already unsafe measured height.
            safe_z=measured_z-(self.clearances-clearance_floor)
            per_hand_z=torch.maximum(self.palms[:,:,2],safe_z)
            shared_z=per_hand_z.amax(1,keepdim=True)
            if hasattr(task,'palm_reposition'):
                reposition=task.palm_reposition
                waiting=(reposition.state==1)&((task.phase==1)&reposition.mask)[:,None]
                if hasattr(task,'close_grasp_hold'):waiting &= ~task.close_grasp_hold.active
                per_hand_z=torch.where(waiting,torch.maximum(per_hand_z,reposition.hold[:,:,2]),per_hand_z)
                # A release hold forbids further descent even with ample table
                # clearance. It is already in the exploration-offset frame;
                # do not add/subtract the common lift a second time.
                hold_z=torch.where(waiting,reposition.hold[:,:,2],-torch.inf).amax(1,keepdim=True)
                shared_z=torch.maximum(shared_z,hold_z)
                # Feed the final safety bound back into the motion goal, even
                # when the world is already at that bound. Ordinary slew is not
                # an arrival goal; a slower release must retain pending descent.
                safe_goal_z=torch.where(synchronized[:,None],safe_z.amax(1,keepdim=True),safe_z)
                moving=((safety_controlled&live&(task.phase==1)&reposition.mask
                         )[:,None]&(reposition.state==2)&~waiting
                        &~(synchronized&waiting.any(1))[:,None]
                        &(safe_goal_z>reposition.goal[:,:,2]))
                reposition.goal[:,:,2]=torch.where(moving,
                    safe_goal_z,reposition.goal[:,:,2])
                reposition.goal_clipped |= moving
            # This projection also cancels a previously submitted unsafe lower
            # target. Merely stopping new descent leaves a driven wrist pushing
            # toward its queued target. Raising that target to the measured safe
            # height takes priority over the ordinary exploration slew limit.
            # Close shares a safe exploration height. Lift retains each hand's
            # captured offset while applying its own measured clearance bound.
            projected_z=torch.where(synchronized[:,None],shared_z,per_hand_z)
            self.palms[:,:,2]=torch.where((safety_controlled&live)[:,None],projected_z,self.palms[:,:,2])
        self.gains = torch.where(live[:,None],(self.gains+self.parameters.stiffness_rate*self.dt)
                                .clamp_max(self.parameters.stiffness_max),self.gains)
        if policy_control:
            self.gains = torch.where(live[:,None], task.policy_stiffness, self.gains)
        pose = self.root_seed.clone()
        pose[:,:,:3] += self.palms
        pose[:,:,2] += task.offset[:,None]
        self.root_targets = torch.where(live[:,None,None],pose,self.root_targets)
        self.commands = torch.where(live[:,None,None],task.command.reshape(self.n,2,6),self.commands)
        if hasattr(task,'contact_effort'):
            self.contact_effort_enabled=True
            loads=task.m['hand_object_loads']
            eligible=contact_effort_permission(task,live)
            retained=torch.zeros_like(eligible)
            if hasattr(task,'lift_grasp_hold'):
                retained |= task.lift_grasp_hold.active[:,None]&(task.phase==3)[:,None]
            if hasattr(task,'close_grasp_hold'):
                retained |= task.close_grasp_hold.active&(task.phase<3)[:,None]
            self.effort_feedforward=task.contact_effort.step(
                task.f_target,loads,task.direction,eligible,self.dt,retained=retained)
        for hi, hand in enumerate(self.hands):
            if getattr(self.w,'mounts',None) is None:
                self.write_root_pose(hi,self.root_targets[:,hi])
            else:
                self.w.mounts[hi].move(self.root_targets[:,hi])
            hand.set_joint_position_target(self.commands[:,hi].float(),joint_ids=self.w.ids[hi])
            gains = self.gains[:,hi,None].expand(-1,6).float()
            hand.write_joint_stiffness_to_sim(gains,joint_ids=self.w.ids[hi])
            hand.actuators['active'].stiffness[:] = gains
            if hasattr(task,'contact_effort'):
                write_contact_effort(hand,self.w.ids[hi],self.effort_feedforward[:,hi],
                                     self.parameters.effort_limit_nm)
            hand.write_data_to_sim()
        # Physics/contact feedback still runs at physics_hz; present at 60 Hz.
        # Rendering every substep made a four-second close take minutes in GUI.
        self._physics_steps = getattr(self, '_physics_steps', 0) + 1
        interval = getattr(self,'render_interval_steps',getattr(self, 'decimation', max(1, round(1 / (60 * self.dt)))))
        render=not self.w.args.headless and self._physics_steps % interval == 0
        if not self.w.args.headless and hasattr(self,'manual_render_wall_interval_s'):
            now=time.monotonic()
            render=now-getattr(self,'last_manual_render_wall_s',0.)>=self.manual_render_wall_interval_s
            if render:self.last_manual_render_wall_s=now
        self.w.sim.step(render=render)
        measurement=self.refresh()
        if getattr(self.w,'mounts',None) is not None and not getattr(self,'continuous_motion',False):
            from .tabletop_palm_mount import check_transport
            check_transport(self.hands,self.w.mounts,self.root_targets)
        return measurement

    def reset(self, ids):
        if ids.numel()==0: return self.measure()
        if hasattr(self,'adapter_fault_bits'):self.adapter_fault_bits[ids]=0
        if hasattr(self,'transport_warning_mask'):self.transport_warning_mask[ids]=False
        for hi, hand in enumerate(self.hands):
            hand.reset(ids)
            q, qd = self.seed_q[hi][ids], self.seed_qd[hi][ids]
            hand.write_joint_state_to_sim(q,qd,env_ids=ids)
            pose = self.root_seed[ids,hi].clone(); pose[:,2] += self.initial_offset
            if getattr(self.w,'mounts',None) is not None:self.w.mounts[hi].reset(pose,ids)
            self.write_root_pose(hi,pose,env_ids=ids)
            hand.write_root_velocity_to_sim(torch.zeros((len(ids),6),device=self.device),env_ids=ids)
            hand.set_joint_position_target(q[:,self.w.ids[hi]],joint_ids=self.w.ids[hi],env_ids=ids)
            gains = torch.full((len(ids),6),10.,device=self.device)
            hand.write_joint_stiffness_to_sim(gains,joint_ids=self.w.ids[hi],env_ids=ids)
            hand.actuators['active'].stiffness[ids] = 10.
            if getattr(self,'contact_effort_enabled',False):
                write_contact_effort(hand,self.w.ids[hi],torch.zeros((len(ids),6),device=self.device),
                                     self.parameters.effort_limit_nm,env_ids=ids)
                if hasattr(self,'effort_feedforward'):self.effort_feedforward[ids,hi]=0.
            hand.write_data_to_sim()
            self.root_targets[ids,hi] = pose
            self.commands[ids,hi] = q[:,self.w.ids[hi]].double()
        self.palms[ids] = 0.; self.gains[ids] = 10.
        self.wire.reset(ids)
        self.wire.write_root_pose_to_sim(self.seed_wire[ids].float(),env_ids=ids)
        self.wire.write_root_velocity_to_sim(self.seed_velocity[ids].float(),env_ids=ids)
        for sensor in self.w.sensors: sensor.reset(ids)
        self.link_loads[ids] = 0.; self.forces[ids] = 0.
        self.vertical_components[ids]=0.
        self.contact_center_w[ids]=float('nan'); self.contact_normal_w[ids]=float('nan')
        self.contact_normal_load_n[ids]=0.; self.contact_valid[ids]=False
        self.contact_all_below_centerline[ids]=False; self.contact_count[ids]=0
        self.w.sim.physics_sim_view.update_articulations_kinematic()
        return self.refresh(sensors=False)
