"""Device-resident rigid physics adapter; host scene construction remains unchanged."""
import torch
from .linear_coupling import MIMIC
from .tabletop_object_contact import link_groups

ROOT_POSE_WRITE_CONTRACT = dict(version='changed_fixed_root_pose_v1',
    comparison='last_submitted_float32', reset='always_write_selected')


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


class TensorWorld:
    def __init__(self, world):
        if world.flexible or not world.full_hand_contact:
            raise ValueError('tensor training requires rigid full-hand contact')
        self.w = world
        self.n, self.dt, self.device = world.n, world.dt, world.device
        self.decimation = world.args.physics_hz//60
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
        gravity=world.sim.cfg.gravity
        if gravity[0]!=0 or gravity[1]!=0 or gravity[2]>=0:
            raise ValueError('vertical force telemetry requires downward world-Z gravity')
        self.object_weight_n=self.wire.root_physx_view.get_masses().to(self.device).double().reshape(self.n)*(-gravity[2])
        for sensor in world.sensors:
            if list(sensor.contact_physx_view.sensor_paths)!=list(sensor.body_physx_view.prim_paths):
                raise ValueError('vertical force sensor ordering mismatch')
            if sensor.contact_physx_view.filter_count!=1 or not sensor.cfg.track_friction_forces:
                raise ValueError('object-only normal and friction sensor required')
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
        if sensors:
            for sensor in w.sensors: sensor.update(self.dt,force_recompute=True)
            vectors = torch.stack([s.data.force_matrix_w[:,0,0,:] for s in w.sensors],1).double()
            self.link_loads = vectors.reshape(self.n,2,-1,3).norm(dim=-1)
            self.forces = vectors[:,self.pad_indices].norm(dim=-1)
            from .tabletop_vertical_force import object_vertical_components
            friction=torch.stack([s.data.friction_forces_w[:,0,0,:] for s in w.sensors],1).double()
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
        return self.measure()

    def measure(self):
        from .tabletop_tensor_task import FAILURE_CODES
        failure = torch.zeros(self.n,device=self.device,dtype=torch.long)
        for hi in range(2):
            q, bounds, error = self.q[:,hi], self.limits[:,hi], self.errors[:,hi]
            for name, condition in (
                ('nonfinite_joint_state',~torch.isfinite(q).all(1)|~torch.isfinite(error).all(1)),
                ('joint_limit',((q<bounds[:,:,0]-.02)|(q>bounds[:,:,1]+.02)).any(1)),
                ('mimic_error',error.abs().amax(1)>.02)):
                failure = torch.where((failure==0)&condition,FAILURE_CODES[name],failure)
        goals = self.goal_roots.clone()
        goals[:,:,:2] += (self.wp-self.spawn)[:,None,:2]
        palm_measure=self.palms
        if getattr(self.w,'mounts',None) is not None:
            from .tabletop_palm_mount import measured_offset
            palm_measure=measured_offset(self.roots,self.root_targets,self.palms)
        return dict(q=torch.cat([self.q[:,h,self.w.ids[h]] for h in range(2)],1),
            qd=torch.cat([self.qd[:,h,self.w.ids[h]] for h in range(2)],1),
            forces=self.forces,clearance=self.clearances.amin(1),
            object_relative=self.wp-self.roots[:,:,:3].mean(1),object_quat=self.wq,
            object_velocity=self.wv,bottom=self.bottom,
            approach_error=(self.roots[:,:,:3]-goals).norm(dim=-1).amax(1),
            mimic_errors=self.errors.flatten(1),hand_clearance=self.clearances,
            adaptive_palm_offset=palm_measure,grip_stiffness=self.gains,pad_relative=self.pad_relative,
            vertical_hand_force_n=signed_hand_vertical_force(self.vertical_components),
            object_link_loads=self.link_loads,grouped_object_loads=self.link_loads@self.group_matrix,
            hand_object_loads=self.link_loads.sum(-1),joint_failure=failure)

    def advance(self, task, allowed, live):
        target = task.palm_actions*self.palm_limit
        movement = (target-self.palms).clamp(-.002*self.dt,.002*self.dt)
        movement[:,:,2] = torch.maximum(movement[:,:,2],-(self.clearances-.0021).clamp_min(0))
        self.palms = torch.where(allowed[:,None,None],
            torch.maximum(torch.minimum(self.palms+movement,self.palm_limit),-self.palm_limit),self.palms)
        self.gains = torch.where(live[:,None],(self.gains+self.parameters.stiffness_rate*self.dt)
                                .clamp_max(self.parameters.stiffness_max),self.gains)
        pose = self.root_seed.clone()
        pose[:,:,:3] += self.palms
        pose[:,:,2] += task.offset[:,None]
        self.root_targets = torch.where(live[:,None,None],pose,self.root_targets)
        self.commands = torch.where(live[:,None,None],task.command.reshape(self.n,2,6),self.commands)
        for hi, hand in enumerate(self.hands):
            if getattr(self.w,'mounts',None) is None:
                self.write_root_pose(hi,self.root_targets[:,hi])
            else:
                self.w.mounts[hi].move(self.root_targets[:,hi])
            hand.set_joint_position_target(self.commands[:,hi].float(),joint_ids=self.w.ids[hi])
            gains = self.gains[:,hi,None].expand(-1,6).float()
            hand.write_joint_stiffness_to_sim(gains,joint_ids=self.w.ids[hi])
            hand.actuators['active'].stiffness[:] = gains
            hand.write_data_to_sim()
        self.w.sim.step(render=not self.w.args.headless)
        measurement=self.refresh()
        if getattr(self.w,'mounts',None) is not None:
            from .tabletop_palm_mount import check_transport
            check_transport(self.hands,self.w.mounts,self.root_targets)
        return measurement

    def reset(self, ids):
        if ids.numel()==0: return self.measure()
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
        self.w.sim.physics_sim_view.update_articulations_kinematic()
        return self.refresh(sensors=False)
