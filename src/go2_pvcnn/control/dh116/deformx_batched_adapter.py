"""Flexible object boundary for the existing hand/action/PPO scene.

Rod coordinates are local to each cloned environment. PhysX wrenches are world
axis forces at the body's COM; observation forces include normal contact only.
"""
import numpy as np
import time
import os
import json
import torch
from scipy.spatial.transform import Rotation
from .deformx_wire_runtime import GraspRegions, paper_rod_config, outward_hull


class MaterialSampling:
    """Fixed material samples, vectorized over environments (never reselected)."""
    def __init__(self, initial, grasp_points):
        regions = GraspRegions(initial,grasp_points)
        self.intervals = regions.intervals
        self.weights=[]; self.segments=[]; self.mean_weights=[]
        for lo,hi in self.intervals:
            # Keep both adjoining element radii at a material boundary.
            weights=[]; segments=[]; mean=np.zeros(len(regions.s))
            for i in range(len(regions.s)-1):
                a,b=max(lo,regions.s[i]),min(hi,regions.s[i+1])
                if a>b: continue
                for station in (a,b):
                    f=(station-regions.s[i])/(regions.s[i+1]-regions.s[i])
                    row=np.zeros(len(regions.s));row[i]=1-f;row[i+1]=f
                    weights.append(row);segments.append(i)
                    mean += row*(b-a)/(2*(hi-lo))
            self.weights.append(np.array(weights));self.segments.append(np.array(segments));self.mean_weights.append(mean)

    def measure(self,p,v,r):
        centers=[];bottoms=[];speeds=[];velocities=[]
        for weights,segments,mean in zip(self.weights,self.segments,self.mean_weights):
            points=np.einsum('bcn,sn->bsc',p,weights)
            velocity=np.einsum('bcn,sn->bsc',v,weights)
            centers.append(np.einsum('bcn,n->bc',p,mean));velocities.append(np.einsum('bcn,n->bc',v,mean))
            bottoms.append((points[:,:,2]-r[:,segments]).min(axis=1))
            speeds.append(np.linalg.norm(velocity,axis=2).max(axis=1))
        return dict(centers=np.stack(centers,axis=1),velocities=np.stack(velocities,axis=1),
            bottom=np.stack(bottoms,axis=1).min(axis=1),speed=np.stack(speeds,axis=1),
            global_bottom=(np.minimum(p[:,2,:-1],p[:,2,1:])-r).min(axis=1))


def mesh_wrenches(impulse,angular_origin,com,dt):
    return impulse/dt,(angular_origin-torch.cross(com,impulse,dim=-1))/dt


class FlexibleObject:
    def __init__(self,world):
        from vendor.DeformX.co_sim.batched_cosserat import BatchedCosserat
        self.world=world
        self.config=paper_rod_config(world.seed['wire_pose'][:3],world.layout['table_center'],world.layout['table_size'])
        # The original wire axis is +Y. Never silently rotate a different seed.
        if not np.allclose(world.seed['wire_pose'][3:],[1,0,0,0],atol=1e-7):
            raise ValueError('flexible object requires original Y-axis wire seed')
        self.engine=BatchedCosserat(self.config,world.n,world.device)
        meshes=[outward_hull(shape[2]) for shape in world.shapes]
        self.engine.register_meshes([x[0] for x in meshes],[x[1] for x in meshes])
        self.hi=world.tensor([x[0] for x in world.shapes],torch.long)
        self.bi=world.tensor([x[1] for x in world.shapes],torch.long)
        self.origins=world.tensor(world.origins,torch.float64)
        self.regions=None
        self.normal=np.zeros((world.n,4,3))
        self.visual=None
        if not world.args.headless:
            from .deformx_wire_asset import SkeletonWire
            self.visual=SkeletonWire(world.sim.stage,'/World/FlexibleWirePreview',world.args.object_usd,
                                     render_mode='mesh_points')
        self.last=None

    def initialize_regions(self,pads):
        if self.regions is None:
            # Called once after the original seed hand poses have been written.
            grasp_points=pads[0].reshape(2,2,3).mean(axis=1)-self.world.origins[0]
            self.regions=MaterialSampling(self.engine.position[0].cpu().numpy(),grasp_points)

    def reset(self,indices):
        self.engine.reset(indices);self.normal[indices]=0.

    def advance(self):
        started=time.monotonic()
        w=self.world
        from isaaclab.utils.math import matrix_from_quat
        pose=torch.stack([h.data.body_link_pose_w for h in w.hands],dim=1)[:,self.hi,self.bi].double()
        comp=torch.stack([h.data.body_com_pos_w for h in w.hands],dim=1)[:,self.hi,self.bi].double()
        vel=torch.stack([h.data.body_com_vel_w for h in w.hands],dim=1)[:,self.hi,self.bi].double()
        pos=pose[:,:,:3]-self.origins[:,None];com=comp-self.origins[:,None]
        omega=vel[:,:,3:]
        link_velocity=vel[:,:,:3]+torch.cross(omega,pos-com,dim=-1)
        gathered=time.monotonic()
        self.last=self.engine.advance(w.dt,pos,matrix_from_quat(pose[:,:,3:]),link_velocity,omega)
        simulated=time.monotonic()
        force,torque=mesh_wrenches(self.last['linear_impulse'],self.last['angular_origin_impulse'],com,w.dt)
        # Aggregate all collision meshes belonging to a body before applying.
        for hi,hand in enumerate(w.hands):
            mask=self.hi==hi;indices=self.bi[mask]
            f=torch.zeros((w.n,len(hand.body_names),3),device=w.device)
            t=torch.zeros_like(f)
            f.index_add_(1,indices,force[:,mask].float());t.index_add_(1,indices,torque[:,mask].float())
            if not bool(torch.isfinite(f).all() & torch.isfinite(t).all()) or float(torch.linalg.vector_norm(f,dim=-1).max())>100.:
                raise FloatingPointError('invalid flexible-object body reaction (100 N guard)')
            hand.root_physx_view.apply_forces_and_torques_at_position(force_data=f,torque_data=t,
                position_data=hand.data.body_com_pos_w.contiguous(),
                indices=torch.arange(w.n,device=w.device,dtype=torch.int32),is_global=True)
        normal=self.last['normal_impulse']/w.dt
        self.normal=np.stack([normal[:,(self.hi==hi)&(self.bi==bi)].sum(dim=1).cpu().numpy()
            for hi in range(2) for bi in w.pad_ids[hi]],axis=1)
        if max(float(self.last[k].abs().max()) for k in ('linear_balance','angular_origin_balance'))>1e-7:
            raise FloatingPointError('flexible contact reaction imbalance')
        self.render()
        if os.environ.get('DH116_FLEX_PROFILE') == '1':
            print('ROD_TIMING',json.dumps(dict(gather=gathered-started,solver=simulated-gathered,reaction=time.monotonic()-simulated)),flush=True)

    def render(self):
        if self.visual is not None:
            self.visual.update(self.engine.position[0].cpu().numpy()+self.world.origins[0,:,None],
                               self.engine.director[0].cpu().numpy())

    def measure(self,selection):
        p=self.engine.position[selection].cpu().numpy();v=self.engine.velocity[selection].cpu().numpy()
        r=self.engine.radius[selection].cpu().numpy()
        lengths=np.linalg.norm(np.diff(p,axis=2),axis=1)
        if (not np.isfinite(p).all() or not np.isfinite(v).all() or lengths.min()<.025
                or lengths.max()>.1 or np.abs(v).max()>100.):
            raise FloatingPointError('divergent flexible rod state')
        if self.regions is None:
            # Only bind() before first reset; regions are set using reset hand poses.
            points=np.repeat(p.mean(axis=2)[:,None],2,axis=1)
            result=dict(centers=points,velocities=np.repeat(v.mean(axis=2)[:,None],2,axis=1),
                bottom=(np.minimum(p[:,2,:-1],p[:,2,1:])-r).min(axis=1),speed=np.zeros((len(p),2)))
        else: result=self.regions.measure(p,v,r)
        centers=result['centers']+self.world.origins[selection,None]
        result['centers']=centers
        frames=self.engine.director[selection,:,:,10].cpu().numpy()[:,[1,2,0],:].transpose(0,2,1)
        q=Rotation.from_matrix(frames).as_quat()[:,[3,0,1,2]]
        speeds=np.linalg.norm(result['velocities'],axis=2)
        velocity=result['velocities'][np.arange(len(p)),speeds.argmax(axis=1)]
        result.update(wp=centers.mean(axis=1),wq=q,wv=np.c_[velocity,np.zeros_like(velocity)])
        return result
