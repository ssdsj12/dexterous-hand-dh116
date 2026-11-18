"""Differentiable DH116 USD-frame FK and sampled convex geometry, not physics.

Only six active angles per hand are optimized. All eleven finger joint angles
are derived from the existing affine mimic model. Geometric samples never
stand in for measured contact loads.
"""
import numpy as np
import torch
from scipy.spatial import ConvexHull
from .linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits

CONTRACT=dict(version='dh116_surface_fk_v2',active_per_hand=6,passive_per_hand=5,
    geometry='source_USD_collision_convex_hulls_and_surface_samples',
    patch='spread_inner_surface_samples_within15mm_reference_normal_cosine_gt0.5',
    collision_approximation='sampled_points_and_convex_halfspaces;physical_validation_required',
    self_exclusions='same_fixed_rigid_cluster_and_directly_adjacent_clusters_only',
    measured_contact=False,hardware_calibrated=False)


def full_angles(active):
    if active.ndim!=2 or active.shape[1]!=6 or not active.is_floating_point() or not torch.isfinite(active).all():
        raise ValueError('six finite active angles per hand required')
    values={name:active[:,i] for i,name in enumerate(ACTIVE)}
    for follower,master,ratio,offset in MIMIC:values[follower]=values[master]*ratio+offset
    return torch.stack([values[name] for name in JOINTS],-1)


def pose_matrix(pose):
    if pose.ndim!=2 or pose.shape[1]!=7 or not torch.isfinite(pose).all():
        raise ValueError('finite N x 7 poses required')
    q=pose[:,3:];scale=q.abs().amax(-1,keepdim=True)
    if (scale<=0).any():raise ValueError('nonzero quaternion required')
    q=q/scale;q=q/q.norm(dim=-1,keepdim=True)
    w,x,y,z=q.unbind(-1)
    r=torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                   2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                   2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)
    result=torch.eye(4,device=pose.device,dtype=pose.dtype)[None].repeat(len(pose),1,1)
    result[:,:3,:3]=r;result[:,:3,3]=pose[:,:3]
    return result


def transform(pose,points):
    return points@pose[:,:3,:3].transpose(-1,-2)+pose[:,:3,3,None].transpose(-1,-2)


def _surface(vertices,count=96,seed=0):
    hull=ConvexHull(vertices);tri=vertices[hull.simplices]
    areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)/2
    rng=np.random.default_rng(seed)
    choices=rng.choice(len(tri),size=count,p=areas/areas.sum())
    uv=rng.random((count,2));s=np.sqrt(uv[:,0]);v=uv[:,1]
    points=(1-s[:,None])*tri[choices,0]+(s*(1-v))[:,None]*tri[choices,1]+(s*v)[:,None]*tri[choices,2]
    return points,hull.equations[choices,:3],hull.equations


def _inscribed_proxy(vertices,count=48):
    if len(vertices)<=count:return vertices
    selected=[0];distance=np.full(len(vertices),np.inf)
    for _ in range(count-1):
        distance=np.minimum(distance,((vertices-vertices[selected[-1]])**2).sum(-1))
        selected.append(int(distance.argmax()))
    return vertices[selected]


class SurfaceHandModel:
    def __init__(self,spec,device='cpu',dtype=torch.float64):
        self.spec=spec;self.device=torch.device(device);self.dtype=dtype
        self.root=spec['root'];self.side=spec['side'];self.edges=[]
        pending=list(spec['edges']);known={self.root};self.neighbors={self.root:set()}
        moving=[e['name'] for e in pending if e['axis'] is not None]
        if len(moving)!=11 or set(moving)!=set(JOINTS):raise ValueError('exact eleven named finger joints required')
        while pending:
            ready=[e for e in pending if e['parent'] in known]
            if not ready:raise ValueError('disconnected/cyclic hand geometry')
            for e in ready:
                if e['child'] in known:raise ValueError('duplicate child in hand geometry')
                axis=None if e['axis'] is None else self.t(e['axis'])
                if axis is not None and (axis.shape!=(3,) or not torch.isfinite(axis).all() or abs(float(axis.norm())-1)>1e-6):
                    raise ValueError('unit joint axis required')
                self.edges.append((e['name'],e['parent'],e['child'],self.t(e['t0']),self.t(e['t1']),axis))
                self.neighbors.setdefault(e['parent'],set()).add(e['child'])
                self.neighbors.setdefault(e['child'],set()).add(e['parent'])
                known.add(e['child']);pending.remove(e)
        self.bodies=known;self.shapes=[]
        for row in spec['shapes']:
            if row['body'] not in known:raise ValueError('unknown collision body')
            vertices=np.asarray(row['vertices'],dtype=float)
            surface,_,planes=_surface(vertices)
            proxy=ConvexHull(_inscribed_proxy(vertices))
            self.shapes.append(dict(body=row['body'],points=self.t(np.r_[vertices,surface]),
                probe_points=self.t(surface),planes=self.t(planes),optimization_planes=self.t(proxy.equations),
                lower=self.t(vertices.min(0)),upper=self.t(vertices.max(0))))
        self.patches=[]
        if len(spec['patches'])!=2:raise ValueError('thumb and index patches required')
        for p in spec['patches']:
            if p['body'] not in known:raise ValueError('unknown patch body')
            points,normals=self.t(p['points']),self.t(p['normals'])
            if points.ndim!=2 or points.shape[-1]!=3 or normals.shape!=points.shape or len(points)<3:
                raise ValueError('multiple finite surface points and normals required')
            if not torch.isfinite(points).all() or not torch.isfinite(normals).all() or (normals.norm(dim=-1)<=0).any():
                raise ValueError('finite surface points and nonzero normals required')
            self.patches.append(dict(body=p['body'],points=points,normals=normals/normals.norm(dim=-1,keepdim=True)))
        if self.patches[0]['points'].shape!=self.patches[1]['points'].shape:raise ValueError('matching patch sample counts required')
        clusters={body:body for body in known}
        def representative(body):
            while clusters[body]!=body:body=clusters[body]
            return body
        for _,parent,child,_,_,axis in self.edges:
            if axis is None:clusters[representative(child)]=representative(parent)
        adjacent=set()
        for _,parent,child,_,_,axis in self.edges:
            a,b=representative(parent),representative(child)
            if a!=b:adjacent.update(((a,b),(b,a)))
        self.self_pairs=[]
        for i,a in enumerate(self.shapes):
            cluster=representative(a['body'])
            for j,b in enumerate(self.shapes[:i]):
                other=representative(b['body'])
                if other!=cluster and (cluster,other) not in adjacent:self.self_pairs.append((i,j))
        self.lower,self.upper=map(self.t,active_limits())

    def t(self,value):return torch.as_tensor(value,device=self.device,dtype=self.dtype)

    def fk(self,active,root_pose):
        active=active.to(device=self.device,dtype=self.dtype);root_pose=root_pose.to(device=self.device,dtype=self.dtype)
        angles=full_angles(active)
        if len(root_pose)!=len(active):raise ValueError('matching root and active batches required')
        poses={self.root:pose_matrix(root_pose)}
        for name,parent,child,t0,t1,axis in self.edges:
            before=poses[parent]@t0
            if axis is None:poses[child]=before@t1;continue
            a=angles[:,JOINTS.index(name)];x,y,z=axis
            cross=torch.stack((x*0,-z,y,z,x*0,-x,-y,x,x*0)).reshape(3,3)
            r=torch.eye(3,device=self.device,dtype=self.dtype)+a.sin()[:,None,None]*cross+(1-a.cos())[:,None,None]*(cross@cross)
            motion=torch.eye(4,device=self.device,dtype=self.dtype)[None].repeat(len(active),1,1)
            motion[:,:3,:3]=r
            poses[child]=before@motion@t1
        return poses

    def sample(self,active,root_pose):
        return self.sample_poses(self.fk(active,root_pose))

    def sample_poses(self,poses):
        """Sample actual SDK link poses without assuming perfect mimic tracking."""
        collision=[transform(poses[s['body']],s['points']) for s in self.shapes]
        patches=[transform(poses[p['body']],p['points']) for p in self.patches]
        normals=[p['normals']@poses[p['body']][:,:3,:3].transpose(-1,-2) for p in self.patches]
        return dict(patches=torch.stack(patches,1),patch_normals=torch.stack(normals,1),
                    collision_points=torch.cat(collision,1),poses=poses,shape_points=collision)

    def _inside(self,points,pose,shape,fast=False):
        local=(points-pose[:,:3,3][:,None])@pose[:,:3,:3]
        within=((local>=shape['lower'])&(local<=shape['upper'])).all(-1)
        result=points.sum((1,2))*0
        if not within.any():return result
        indices=within.nonzero();planes=shape['optimization_planes'] if fast else shape['planes']
        signed=local[within]@planes[:,:3].T+planes[:,3]
        depth=(-signed.amax(-1)).clamp_min(0)
        return result.scatter_reduce(0,indices[:,0],depth,reduce='amax',include_self=True)

    def object_penetration(self,active,root_pose,object_points,fast=False):
        return self.object_penetration_poses(self.fk(active,root_pose),object_points,fast=fast)

    def object_penetration_poses(self,poses,object_points,fast=False):
        return torch.stack([self._inside(object_points,poses[s['body']],s,fast) for s in self.shapes],-1).amax(-1)

    def self_penetration(self,active,root_pose,fast=False):
        return self.self_penetration_poses(self.fk(active,root_pose),fast=fast)

    def self_penetration_poses(self,poses,fast=False):
        if not self.self_pairs:return poses[self.root].sum((1,2))*0
        s=self.sample_poses(poses);values=[]
        for i,j in self.self_pairs:
            a,b=self.shapes[i],self.shapes[j]
            values.append(self._inside(transform(s['poses'][a['body']],a['probe_points']),s['poses'][b['body']],b,fast))
            values.append(self._inside(transform(s['poses'][b['body']],b['probe_points']),s['poses'][a['body']],a,fast))
        return torch.stack(values,-1).amax(-1)

    @classmethod
    def from_usd(cls,source,side,device='cpu',dtype=torch.float64,patch_count=24):
        from pxr import Usd,UsdGeom
        from .hand_point_kinematics import HandPointFK
        from .native_scene import _PAD_LOCAL_POINTS
        fk=HandPointFK(source,side,[0.,0.,0.,1.,0.,0.,0.])
        stage=Usd.Stage.Open(str(source));cache=UsdGeom.XformCache()
        bodies={fk.root}|{e[2] for e in fk.edges};shapes=[];patches=[];surface={}
        for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
            if '/collisions/' not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh):continue
            body=prim.GetParent()
            while body and str(body.GetPath()) not in bodies:body=body.GetParent()
            if not body:raise ValueError('collision mesh is outside hand tree')
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            relative=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
            points=(np.c_[points,np.ones(len(points))]@relative)[:,:3]
            points=np.unique(points,axis=0);points=points[ConvexHull(points).vertices]
            name=str(body.GetPath());shapes.append(dict(body=name,vertices=points.tolist()))
            ps,ns,_=_surface(points,count=512,seed=len(shapes))
            surface.setdefault(name,[]).append((ps,ns))
        for name,digit in (('finger14_link','thumb'),('finger23_link','index')):
            body=next(b for b in bodies if b.endswith('/'+name))
            points=np.concatenate([p[0] for p in surface[body]])
            normals=np.concatenate([p[1] for p in surface[body]])
            reference=_PAD_LOCAL_POINTS[side][digit]
            distance=np.linalg.norm(points-reference,axis=-1)
            nearest=int(distance.argmin())
            inner=(normals@normals[nearest]>.5)&(distance<.015)
            eligible=np.flatnonzero(inner)
            if len(eligible)<patch_count:eligible=np.argsort(distance)[:max(patch_count,len(eligible))]
            selected=[nearest];spacing=np.full(len(eligible),np.inf)
            for _ in range(patch_count-1):
                spacing=np.minimum(spacing,((points[eligible]-points[selected[-1]])**2).sum(-1))
                selected.append(int(eligible[spacing.argmax()]))
            patches.append(dict(body=body,points=points[selected].tolist(),normals=normals[selected].tolist()))
        if len(shapes)!=17:raise ValueError('expected all seventeen source collision hulls per hand')
        spec=dict(side=side,root=fk.root,source=str(source),edges=[dict(name=n,parent=a,child=b,
            t0=t0.tolist(),t1=t1.tolist(),axis=None if axis is None else axis.tolist()) for n,a,b,t0,t1,axis in fk.edges],
            shapes=shapes,patches=patches)
        return cls(spec,device=device,dtype=dtype)
