"""Cloned GPU physics with batched I/O and the existing measured task controller.

Import after AppLauncher. The host controller is intentionally retained; this is
not an end-to-end GPU task. No reset advances the world clock.
"""
import copy
import time
import os
import json
import numpy as np
import torch
from scipy.spatial import ConvexHull
from isaacsim.core.cloner import GridCloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from .tabletop_vec_isaac import spawn_scene
from .tabletop_batch_geometry import collision_clearance, rotated_points, normalized_convex
from .linear_coupling import ACTIVE, JOINTS, MIMIC
from .tabletop_adaptive_backend import adaptive_step, select_grasp_command
from .tabletop_task import joint_safety_failure
from .tabletop_load_parameters import LEGACY_LOAD_PARAMETERS
from .tabletop_control import moving_goal_error
from .native_scene import _PAD_LOCAL_POINTS


class GpuWorld:
    def __init__(self, sim, args, layout, sources):
        if args.control_mode != 'adaptive_grasp':
            raise ValueError('GPU batch currently supports adaptive grasp only')
        self.sim, self.args = sim, args
        self.parameters = getattr(args,'load_parameters',None) or LEGACY_LOAD_PARAMETERS
        self.n, self.dt = args.num_envs, 1/args.physics_hz
        self.device = args.device
        self.layout = copy.deepcopy(layout)
        
        self.full_hand_contact = bool(getattr(args,'full_hand_contact',False))
        
        
        self.cloner = GridCloner(spacing=args.env_spacing_m)
        self.cloner.define_base_env('/World/envs')
        self.paths = self.cloner.generate_paths('/World/envs/env', self.n)
        self.source = spawn_scene(sim, args, layout, sources, self.paths[0], np.zeros(3))
        # PhysX cooks GPU convex data before applying mesh scale. The original
        # half-meter, half-millimeter wall prisms exceed its aspect-ratio limit.
        # Normalize cooking coordinates and restore their EXACT world vertices;
        # no change to bore, thickness, surface, mass, inertia or material.
        for prim in (Usd.PrimRange(sim.stage.GetPrimAtPath(self.paths[0]+'/Wire'))):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            mesh = UsdGeom.Mesh(prim)
            local, center, scale = normalized_convex(np.asarray(mesh.GetPointsAttr().Get()))
            mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in local])
            mesh.GetExtentAttr().Set([Gf.Vec3f(-1.),Gf.Vec3f(1.)])
            mesh.AddTranslateOp().Set(Gf.Vec3d(*center))
            mesh.AddScaleOp().Set(Gf.Vec3f(*scale))
        self.origins = np.asarray(self.cloner.clone(source_prim_path=self.paths[0],
            prim_paths=self.paths, replicate_physics=True, copy_from_source=False,
            enable_env_ids=True))
        # Keep the authored source wrappers alive, but use only these batched
        # wrappers for writes/reads after initialization.
        self.hands = [Articulation(h.cfg.replace(prim_path=f'/World/envs/env_.*/{side}Hand', spawn=None))
                      for h, side in zip(self.source['hands'], ('left','right'))]
        self.wire = RigidObject(self.source['wire'].cfg.replace(prim_path='/World/envs/env_.*/Wire', spawn=None))
        self.object_contact_links = ()
        if self.full_hand_contact:
            from .tabletop_object_contact import link_groups
            per_hand=[]
            for side in ('left','right'):
                base=f'{self.paths[0]}/{side}Hand'
                links=set()
                for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(base), Usd.TraverseInstanceProxies()):
                    path=str(prim.GetPath())
                    # Match the scene's collision geometry traversal, including
                    # instance proxies whose schemas may live on prototypes.
                    if '/collisions/' in path and prim.IsA(UsdGeom.Mesh):
                        links.add(path[len(base)+1:].split('/')[0])
                    elif prim.HasAPI(UsdPhysics.CollisionAPI):
                        body=prim
                        while body and str(body.GetPath()).startswith(base+'/'):
                            if body.HasAPI(UsdPhysics.RigidBodyAPI):
                                links.add(str(body.GetPath())[len(base)+1:]);break
                            body=body.GetParent()
                per_hand.append(tuple(sorted(links)))
            if per_hand[0] != per_hand[1]:raise ValueError('hand collision rigid link order mismatch')
            self.object_contact_links=per_hand[0]
            link_groups(self.object_contact_links)
        self.sensor_links=self.object_contact_links if self.full_hand_contact else ('finger14_link','finger23_link')
        self.sensors = [ContactSensor(ContactSensorCfg(
            prim_path=f'/World/envs/env_.*/{side}Hand/{link}',
            track_friction_forces=bool(getattr(args,'tensor_pipeline',False)),
            track_contact_points=bool(getattr(args,'tensor_pipeline',False)),
            max_contact_data_count_per_prim=16 if getattr(args,'tensor_pipeline',False) else 4,
            filter_prim_paths_expr=['/World/envs/env_.*/Wire'], update_period=0.))
            for side in ('left','right') for link in self.sensor_links]
        self.backends = []

    def tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    def bind(self, seed):
        self.seed = copy.deepcopy(seed)
        self.mounts=None
        if getattr(self.args,'palm_transport','fixed_pose')=='driven_carriage':
            from .tabletop_palm_mount import MountView
            self.mounts=[]
            for hand in self.hands:
                if not hand.is_fixed_base:raise ValueError('wrist carriage requires fixed rail anchor')
                self.mounts.append(MountView(hand))
        if any(h.num_instances != self.n for h in self.hands) or (self.wire is not None and self.wire.num_instances != self.n):
            raise ValueError('GPU articulation/object instance count mismatch')
        # The view ordering must match the cloner's numerical environment order.
        for hand, side in zip(self.hands, ('left','right')):
            actual = [p.split(f'/{side}Hand')[0] for p in hand.root_physx_view.prim_paths]
            if actual != self.paths:
                raise ValueError('GPU articulation view order differs from environment order')
        if self.wire is not None and [p.rsplit('/Wire',1)[0] for p in self.wire.root_physx_view.prim_paths] != self.paths:
            raise ValueError('GPU object view order differs from environment order')
        for sensor, (side,link) in zip(self.sensors,[(s,l) for s in ('left','right')
                                                   for l in self.sensor_links]):
            expected = [f'{p}/{side}Hand/{link}' for p in self.paths]
            if list(sensor.body_physx_view.prim_paths) != expected:
                raise ValueError('GPU contact sensor view order differs from environment order')
        if self.full_hand_contact:
            for hand in self.hands:
                if any(link not in hand.body_names for link in self.object_contact_links):
                    raise ValueError('collision contact links must each be a physics rigid body')
        self.ids = [[h.joint_names.index(n) for n in ACTIVE] for h in self.hands]
        self.named_ids = [[h.joint_names.index(n) for n in JOINTS] for h in self.hands]
        self.pad_ids = [[h.body_names.index(n) for n in ('finger14_link','finger23_link')] for h in self.hands]
        self.pad_points = np.asarray([_PAD_LOCAL_POINTS[s][d] for s in ('left','right') for d in ('thumb','index')])
        self.shapes = []
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for hi, side in enumerate(('left','right')):
            base = f'{self.paths[0]}/{side}Hand'
            for prim in Usd.PrimRange(self.sim.stage.GetPrimAtPath(base), Usd.TraverseInstanceProxies()):
                path = str(prim.GetPath())
                if '/collisions/' not in path or not prim.IsA(UsdGeom.Mesh):
                    continue
                link = path[len(base)+1:].split('/')[0]
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                relative = cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(
                    self.sim.stage.GetPrimAtPath(base+'/'+link)).GetInverse()
                points = (np.c_[points,np.ones(len(points))] @ np.asarray(relative))[:,:3]
                points = np.unique(points,axis=0)
                self.shapes.append((hi,self.hands[hi].body_names.index(link),points[ConvexHull(points).vertices]))
        if len(self.shapes) != 34:
            raise ValueError('GPU geometry requires all 34 hand collision meshes')
        self.limits = np.stack([h.data.joint_pos_limits.cpu().numpy() for h in self.hands],axis=1)
        self.force = np.zeros((self.n,4))
        self.force_vectors = np.zeros((self.n,4,3))
        if self.full_hand_contact:
            from .tabletop_object_contact import aggregate_object_contacts
            self.object_link_vectors=np.zeros((self.n,2,len(self.object_contact_links),3))
            for key,value in aggregate_object_contacts(self.object_link_vectors,self.object_contact_links).items():setattr(self,key,value)
        self.root_targets = np.zeros((self.n,2,7))
        self.commands = np.zeros((self.n,2,6))
        self.gains = np.full((self.n,2),10.)
        self.palms = np.zeros((self.n,2,3))
        self.previous_gains = self.gains.copy()
        
        self.backends = [GpuBackend(self,i) for i in range(self.n)]
        self.refresh(sensors=False)
        return self.backends

    def refresh(self, *, sensors=True, geometry_indices=None):
        for hand in self.hands:
            hand.update(self.dt)
        if self.wire is not None:
            self.wire.update(self.dt)
        # Partial reset reads only the changed slot. Normal physics steps read
        # each whole tensor once, independent of environment count.
        selection = slice(None) if geometry_indices is None else geometry_indices
        def store(name,value):
            if geometry_indices is None or not hasattr(self,name):
                setattr(self,name,value)
            else:
                getattr(self,name)[selection] = value
        for name,attribute in (('poses','body_link_pose_w'),('q','joint_pos'),
                ('qd','joint_vel'),('torque','applied_torque'),('roots','root_pose_w')):
            store(name,torch.stack([getattr(h.data,attribute)[selection] for h in self.hands],dim=1).cpu().numpy())
        if self.mounts is not None:
            store('roots',torch.stack([h.data.body_link_pose_w[selection,m.base_id] for h,m in zip(self.hands,self.mounts)],1).cpu().numpy())
        for name,attribute in (('wp','root_pos_w'),('wq','root_quat_w'),('wv','root_vel_w')):
            store(name,getattr(self.wire.data,attribute)[selection].cpu().numpy())
        if sensors:
            for sensor in self.sensors:
                sensor.update(self.dt,force_recompute=True)
            if self.full_hand_contact:
                vectors=[]
                for sensor in self.sensors:
                    matrix=sensor.data.force_matrix_w
                    if matrix is None or tuple(matrix.shape)!=(self.n,1,1,3):
                        raise ValueError('one rigid body and one per-environment Wire filter required')
                    vectors.append(matrix[:,0,0,:])
                vectors=torch.stack(vectors,dim=1).cpu().numpy()
                from .tabletop_object_contact import aggregate_object_contacts
                self.object_link_vectors[:]=vectors.reshape(self.n,2,len(self.sensor_links),3)
                for key,value in aggregate_object_contacts(self.object_link_vectors,self.object_contact_links).items():getattr(self,key)[:]=value
                pad_indices=[h*len(self.sensor_links)+self.sensor_links.index(link)
                             for h in range(2) for link in ('finger14_link','finger23_link')]
                self.force_vectors[:]=vectors[:,pad_indices]
                self.force[:]=np.linalg.norm(self.force_vectors,axis=2)
            else:
                self.force[:] = torch.stack([torch.linalg.vector_norm(s.data.force_matrix_w.flatten(1),dim=1)
                                              for s in self.sensors],dim=1).cpu().numpy()
                self.force_vectors[:] = torch.stack([s.data.force_matrix_w.reshape(self.n,-1,3).sum(dim=1)
                                                     for s in self.sensors],dim=1).cpu().numpy()
        poses, q, wp, quat = self.poses[selection],self.q[selection],self.wp[selection],self.wq[selection]
        clear, lowest = collision_clearance(poses,self.shapes)
        store('clearances',clear); store('lowest',lowest)
        axis_z = np.abs(2*(quat[:,2]*quat[:,3]+quat[:,0]*quat[:,1]))
        store('bottom',wp[:,2]-.25*axis_z-.008*np.sqrt(np.maximum(0.,1-axis_z**2)))
        store('errors',np.stack([np.stack([q[:,hi,h.joint_names.index(f)]-
            r*q[:,hi,h.joint_names.index(m)]-o for f,m,r,o in MIMIC],axis=1)
            for hi,h in enumerate(self.hands)],axis=1))
        pads = np.concatenate([poses[:,hi,self.pad_ids[hi]] for hi in range(2)],axis=1)
        pad_positions=np.stack([rotated_points(pads[:,j],self.pad_points[j:j+1])[:,0] for j in range(4)],axis=1)
        store('pad_positions',pad_positions)
        relative_to=wp[:,None]
        store('pad_relative',pad_positions-relative_to)

    def reset_slot(self, i, offset):
        self.reset_slots([i], [offset])

    def reset_slots(self, indices, offsets):
        """Reset selected slots with one articulation write/readback per hand.

        Isaac's selected stiffness writes and state refresh still touch full
        articulation tensors. Grouping slots avoids repeating those costs for
        every environment, without advancing the shared physics clock.
        """
        indices = list(indices)
        offsets = np.asarray(offsets, dtype=float)
        if (offsets.shape != (len(indices),) or not np.isfinite(offsets).all()
                or any(not isinstance(i, (int, np.integer)) or not 0 <= i < self.n for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('unique valid reset indices and matching finite offsets required')
        if not indices:
            return
        ids = self.tensor(indices,torch.long)
        count = len(indices)
        for hi, (hand, side) in enumerate(zip(self.hands,('left','right'))):
            hand.reset(ids)
            q = hand.data.default_joint_pos[ids].clone()
            qd = torch.zeros_like(q)
            q[:,self.named_ids[hi]] = self.tensor(self.seed['joint_positions'][hi])
            qd[:,self.named_ids[hi]] = self.tensor(self.seed['joint_velocities'][hi])
            hand.write_joint_state_to_sim(q,qd,env_ids=ids)
            pose = np.tile(self.layout['hand_roots'][side], (count, 1))
            pose[:,:3] += self.origins[indices]
            pose[:,:2] += self.seed['root_xy']
            pose[:,2] += offsets
            if self.mounts is not None:self.mounts[hi].reset(self.tensor(pose),ids)
            hand.write_root_pose_to_sim(self.tensor(pose),env_ids=ids)
            hand.write_root_velocity_to_sim(self.tensor(np.zeros((count,6))),env_ids=ids)
            hand.set_joint_position_target(q[:,self.ids[hi]],joint_ids=self.ids[hi],env_ids=ids)
            hand.write_joint_stiffness_to_sim(self.tensor(np.full((count,6),10.)),joint_ids=self.ids[hi],env_ids=ids)
            hand.actuators['active'].stiffness[ids] = 10.
            hand.write_data_to_sim()
            self.root_targets[indices,hi] = pose
            self.commands[indices,hi] = q[:,self.ids[hi]].cpu().numpy()
        self.gains[indices] = self.previous_gains[indices] = 10.
        self.palms[indices] = 0.
        self.wire.reset(ids)
        pose = np.tile(self.seed['wire_pose'], (count, 1)); pose[:,:3] += self.origins[indices]
        self.wire.write_root_pose_to_sim(self.tensor(pose),env_ids=ids)
        self.wire.write_root_velocity_to_sim(self.tensor(np.tile(self.seed['wire_velocity'], (count, 1))),env_ids=ids)
        for sensor in self.sensors:
            sensor.reset(ids)
        self.sim.physics_sim_view.update_articulations_kinematic()
        self.refresh(sensors=False,geometry_indices=indices)
        
        if self.full_hand_contact:
            from .tabletop_object_contact import clear_object_contact_slots
            clear_object_contact_slots(self,indices)
        self.force[indices] = 0.
        self.force_vectors[indices] = 0.

    def advance(self, requests):
        for backend, request in zip(self.backends,requests):
            if request is not None:
                offset, command, kwargs = request
                backend.apply(offset,command,**kwargs)
        self._step()

    def advance_batch(self, offset, command, allowed, live, palm_actions):
        target=np.asarray(palm_actions).reshape(self.n,2,3)*np.array([.003,.003,.004])
        movement=np.clip(target-self.palms,-.002*self.dt,.002*self.dt)
        movement[:,:,2]=np.maximum(movement[:,:,2],-np.maximum(self.clearances-.0021,0.))
        self.palms[allowed]=np.clip(self.palms+movement,-np.array([.003,.003,.004]),np.array([.003,.003,.004]))[allowed]
        self.gains[live]=np.minimum(self.parameters.stiffness_max,self.gains[live]+self.parameters.stiffness_rate*self.dt)
        for hi,side in enumerate(('left','right')):
            pose=np.tile(self.layout['hand_roots'][side],(self.n,1))
            pose[:,:3]+=self.origins+self.palms[:,hi]
            pose[:,:2]+=self.seed['root_xy']
            pose[:,2]+=offset
            self.root_targets[live,hi]=pose[live]
        self.commands[live]=command[live].reshape(-1,2,6)
        self._step()

    def measure_batch(self):
        failure=np.full(self.n,None,dtype=object)
        for hi in range(2):
            q,bounds,errors=self.q[:,hi],self.limits[:,hi],self.errors[:,hi]
            nonfinite=~np.isfinite(q).all(axis=1)|~np.isfinite(errors).all(axis=1)
            outside=np.any((q<bounds[:,:,0]-.02)|(q>bounds[:,:,1]+.02),axis=1)
            mimic=np.max(abs(errors),axis=1)>.02
            for reason,condition in (('nonfinite_joint_state',nonfinite),('joint_limit',outside),('mimic_error',mimic)):
                failure[(failure==None)&condition]=reason
        goals=np.stack([self.layout['hand_roots'][s][:3] for s in ('left','right')])[None]+self.origins[:,None]
        delta=self.wp-(self.layout['wire_center']+self.origins)
        goals[:,:,:2]+=delta[:,None,:2]
        return dict(**({k:getattr(self,k).copy() for k in ('object_link_loads','grouped_object_loads','hand_object_loads')} if self.full_hand_contact else {}),
            **({}),
            q=np.concatenate([self.q[:,h,self.ids[h]] for h in range(2)],axis=1),
            qd=np.concatenate([self.qd[:,h,self.ids[h]] for h in range(2)],axis=1),
            forces=self.force.copy(),clearance=self.clearances.min(axis=1),
            object_relative=self.wp.astype(np.float64)-self.roots[:,:,:3].mean(axis=1,dtype=np.float64),object_quat=self.wq.copy(),
            object_velocity=self.wv.copy(),bottom=self.bottom.copy(),
            approach_error=np.linalg.norm(self.roots[:,:,:3]-goals,axis=2).max(axis=1),
            mimic_errors=self.errors.reshape(self.n,10).copy(),hand_clearance=self.clearances.copy(),
            adaptive_palm_offset=(self.palms.copy() if self.mounts is None else
                self.palms+self.roots[:,:,:3]-self.root_targets[:,:,:3]),grip_stiffness=self.gains.copy(),
            commanded_palm_offset_m=self.palms.copy(),
            pad_relative=self.pad_relative.copy(),joint_failure=failure,
            _diagnostics={i:self.backends[i].measure()['diagnostics'] for i in sorted(set(range(min(8,self.n))) | set(np.flatnonzero(failure != None)) | set(np.flatnonzero(np.max(abs(self.errors),axis=(1,2))>=.004)[:8]))},
            _lowest_links={i:f'{("left","right")[self.shapes[self.lowest[i]][0]]}/'
                +self.hands[self.shapes[self.lowest[i]][0]].body_names[self.shapes[self.lowest[i]][1]]
                for i in range(min(8,self.n))})

    def _step(self):
        started=time.monotonic()
        for hi,hand in enumerate(self.hands):
            if self.mounts is None:
                hand.write_root_pose_to_sim(self.tensor(self.root_targets[:,hi]))
            else:
                self.mounts[hi].move(self.tensor(self.root_targets[:,hi]))
            hand.set_joint_position_target(self.tensor(self.commands[:,hi]),joint_ids=self.ids[hi])
            if not np.array_equal(self.gains[:,hi],self.previous_gains[:,hi]):
                gains = self.tensor(np.repeat(self.gains[:,hi,None],6,axis=1))
                hand.write_joint_stiffness_to_sim(gains,joint_ids=self.ids[hi])
                hand.actuators['active'].stiffness[:] = gains
            hand.write_data_to_sim()
        self.previous_gains[:] = self.gains
        written=time.monotonic()
        
        advanced=time.monotonic()
        self.sim.step(render=not self.args.headless)
        stepped=time.monotonic()
        self.refresh()
        if self.mounts is not None:
            from .tabletop_palm_mount import check_transport
            check_transport(self.hands,self.mounts,self.tensor(self.root_targets))
        self.last_step_timing=dict(hand_write=written-started,
            physx=stepped-advanced,readback=time.monotonic()-stepped)
        if os.environ.get('DH116_PHYSICS_PROFILE') == '1':
            print('PHYSICS_TIMING',json.dumps(self.last_step_timing),flush=True)


class GpuBackend:
    def __init__(self, world, index):
        self.world, self.index = world,index
        self.full_hand_contact = world.full_hand_contact
        self.object_contact_links = world.object_contact_links
        self.dt, self.decimation = world.dt,world.args.physics_hz//60
        self.grasp = select_grasp_command(np.concatenate([world.layout['active_joints'][s]
            for s in ('left','right')]),world.seed)
        self.initial_command = world.seed['initial_command'].copy()
        self.adaptive_palm_offset = np.zeros((2,3))
        self.grip_stiffness = np.full(2,10.)

    def reset(self, offset):
        self.adaptive_palm_offset[:] = 0.
        self.grip_stiffness[:] = 10.
        prepared = self.__dict__.pop('_prepared_reset_offset', None)
        if prepared is None or prepared != offset:
            self.world.reset_slot(self.index,offset)

    def apply(self, offset, command, *, phase, palm_motion_allowed, palm_action=None,grip_action=None):
        w,i = self.world,self.index
        self.adaptive_palm_offset,self.grip_stiffness = adaptive_step(
            palm_offset=self.adaptive_palm_offset,stiffness=self.grip_stiffness,
            palm_action=palm_action,grip_action=grip_action,hand_clearance=w.clearances[i],
            dt=self.dt,palm_motion_allowed=palm_motion_allowed,parameters=w.parameters)
        for hi,side in enumerate(('left','right')):
            pose = w.layout['hand_roots'][side].copy()
            pose[:3] += w.origins[i]+self.adaptive_palm_offset[hi]
            pose[:2] += w.seed['root_xy']
            pose[2] += offset
            w.root_targets[i,hi] = pose
        w.commands[i] = np.asarray(command).reshape(2,6)
        w.gains[i] = self.grip_stiffness

    def measure(self):
        w,i = self.world,self.index
        diagnostics, failure = [],None
        failure = next((reason for hi in range(2) if (reason := joint_safety_failure(w.q[i,hi],w.limits[i,hi],w.errors[i,hi])) is not None),None)
        for hi,hand in enumerate(w.hands):
            if (i < 8 or failure is not None or np.max(abs(w.errors[i])) >= .004
                    or np.max(w.object_link_loads[i] if w.full_hand_contact else w.force[i]) > 30.):
                diagnostics.append(dict(joints_rad=dict(zip(hand.joint_names,map(float,w.q[i,hi]))),
                    mimic_error_rad=dict(zip([m[0] for m in MIMIC],map(float,w.errors[i,hi]))),
                    joint_vel_rad_s=w.qd[i,hi].tolist(),applied_torque_nm=w.torque[i,hi].tolist(),
                    torque_measurement='implicit_pd_estimate_not_measured_drive_torque',
                    force_on_pads_from_object_normal_w_n=w.force_vectors[i,hi*2:hi*2+2].tolist(),
                    root_pose=w.roots[i,hi].tolist(),command_rad=w.commands[i,hi].tolist(),
                    actual_stiffness=w.gains[i,hi].item(),
                    object_link_loads_n=(w.object_link_loads[i,hi].tolist() if w.full_hand_contact else None),
                    object_contact_links=(list(w.object_contact_links) if w.full_hand_contact else None)))
        hi,bi,_ = w.shapes[w.lowest[i]]
        goals = np.stack([w.layout['hand_roots'][s][:3]+w.origins[i] for s in ('left','right')])
        return dict(**({k:getattr(w,k)[i].copy() for k in ('object_link_loads','grouped_object_loads','hand_object_loads')} if self.full_hand_contact else {}),
            **({}),
            q=np.concatenate([w.q[i,h,w.ids[h]] for h in range(2)]),
            qd=np.concatenate([w.qd[i,h,w.ids[h]] for h in range(2)]),
            forces=w.force[i].copy(),clearance=float(w.clearances[i].min()),
            object_relative=w.wp[i]-w.roots[i,:,:3].mean(axis=0),object_quat=w.wq[i].copy(),
            object_velocity=w.wv[i].copy(),bottom=float(w.bottom[i]),
            approach_error=moving_goal_error(w.roots[i,:,:3],goals,w.wp[i],w.layout['wire_center']+w.origins[i]),
            joint_failure=failure,diagnostics=diagnostics,mimic_errors=w.errors[i].ravel().copy(),
            palm_adjustment=np.zeros(4),hand_clearance=w.clearances[i].copy(),
            lowest_link=f'{("left","right")[hi]}/{w.hands[hi].body_names[bi]}',
            adaptive_palm_offset=self.adaptive_palm_offset.copy(),grip_stiffness=self.grip_stiffness.copy(),
            pad_relative=w.pad_relative[i].copy())
