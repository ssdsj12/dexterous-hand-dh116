"""Independent, float64 CUDA Cosserat rods with exact convex mesh contact.

The explicit midpoint split, constitutive law, material spin/contact couples,
quadrature and damping match ``TorchCudaCosseratRod``. CUDA kernels have an
explicit environment axis; Python loops only over integration substeps while
building a reusable CUDA graph. Mesh queries use exact triangle distances and
validated outward convex face planes (equivalent to winding inside a convex
closed hull). No collision proxy or contact-force clipping is introduced.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import warp as wp
from .gpu_cosserat import GpuRodConfig, TorchCudaCosseratRod

D = wp.float64
V = wp.vec3d
R = wp.mat33d

@wp.struct
class Parameters:
    n: int
    samples: int
    length: D
    radius: D
    shear: V
    bend: V
    inertia_inv: V
    mass: D
    damping: D
    gravity: V
    table: int
    center: V
    half: V
    stiffness: D
    normal_damping: D
    friction: D
    tangent_damping: D

@wp.func
def rotation(w: V, dt: D):
    length = wp.length(w)
    axis = w / wp.max(length, D(1.e-15))
    x = axis[0]; y = axis[1]; z = axis[2]
    c = wp.cos(length*dt); s = wp.sin(length*dt); o = D(1.)-c
    return R(c+x*x*o, x*y*o-z*s, x*z*o+y*s,
             x*y*o+z*s, c+y*y*o, y*z*o-x*s,
             x*z*o-y*s, y*z*o+x*s, c+z*z*o)

@wp.kernel
def drift(p: wp.array2d(dtype=V), v: wp.array2d(dtype=V),
          q: wp.array2d(dtype=R), w: wp.array2d(dtype=V), c: Parameters, h: D):
    b,i=wp.tid()
    p[b,i] = p[b,i]+D(.5)*h*v[b,i]
    if i<c.n:
        q[b,i] = rotation(w[b,i],-D(.5)*h)*q[b,i]

@wp.kernel
def geometry(p: wp.array2d(dtype=V), q: wp.array2d(dtype=R),
             radius: wp.array2d(dtype=D), dilation: wp.array2d(dtype=D),
             tangent: wp.array2d(dtype=V), stress: wp.array2d(dtype=V),
             couple: wp.array2d(dtype=V), weighted: wp.array2d(dtype=V), c: Parameters):
    b,i=wp.tid()
    delta=p[b,i+1]-p[b,i]; length=wp.length(delta)
    t=delta/wp.max(length,D(1e-12)); dil=length/c.length
    tangent[b,i]=t; dilation[b,i]=dil
    radius[b,i]=wp.sqrt(c.radius*c.radius*c.length/wp.max(length,D(1e-12)))
    local=q[b,i]*t
    sigma=dil*local-V(D(0.),D(0.),D(1.))
    stress[b,i]=wp.cw_mul(c.shear,sigma)
    if i<c.n-1:
        rel=q[b,i+1]*wp.transpose(q[b,i])
        vee=V(rel[2,1]-rel[1,2],rel[0,2]-rel[2,0],rel[1,0]-rel[0,1])
        cosine=wp.clamp((rel[0,0]+rel[1,1]+rel[2,2]-D(1.))*D(.5),D(-1.),D(1.))
        angle=wp.acos(cosine); sine=wp.sin(angle)
        if wp.abs(sine)<D(1e-8): sine=D(1e-8)
        kappa=vee*(-D(.5)*angle/sine/c.length)
        cp=wp.cw_mul(c.bend,kappa)
        vor=(length+wp.length(p[b,i+2]-p[b,i+1]))*D(.5)/c.length
        iv3=D(1.)/wp.pow(wp.max(vor,D(1e-9)),D(3.))
        couple[b,i]=cp*iv3
        weighted[b,i]=wp.cross(kappa,cp)*(c.length*iv3)

@wp.kernel
def internal(p: wp.array2d(dtype=V),v: wp.array2d(dtype=V),q: wp.array2d(dtype=R),
             w: wp.array2d(dtype=V),dilation: wp.array2d(dtype=D),tangent: wp.array2d(dtype=V),
             stress: wp.array2d(dtype=V),couple: wp.array2d(dtype=V),weighted: wp.array2d(dtype=V),
             force: wp.array2d(dtype=V),torque: wp.array2d(dtype=V),c: Parameters):
    b,i=wp.tid()
    mass=c.mass
    if i==0 or i==c.n: mass=mass*D(.5)
    f=mass*c.gravity
    if i<c.n:
        f=f+wp.transpose(q[b,i])*stress[b,i]/wp.max(dilation[b,i],D(1e-12))
    if i>0:
        f=f-wp.transpose(q[b,i-1])*stress[b,i-1]/wp.max(dilation[b,i-1],D(1e-12))
    force[b,i]=f
    if i<c.n:
        tau=V(D(0.))
        if i<c.n-1: tau=tau+couple[b,i]+D(.5)*weighted[b,i]
        if i>0: tau=tau-couple[b,i-1]+D(.5)*weighted[b,i-1]
        local=q[b,i]*tangent[b,i]
        jw=wp.cw_div(w[b,i],c.inertia_inv)
        dil=dilation[b,i]
        rate=wp.dot(tangent[b,i],v[b,i+1]-v[b,i])/c.length
        tau=tau+wp.cross(local,stress[b,i])*c.length
        tau=tau+wp.cross(jw/dil,w[b,i])+jw/dil*(rate/wp.max(dil,D(1e-12)))
        torque[b,i]=tau

@wp.func
def fraction(i: int, sample: int, c: Parameters):
    f=(D(sample)+D(.5))/D(c.samples)
    if c.samples==1:
        if i==0: f=D(2.)/D(3.)
        if i==c.n-1: f=D(1.)/D(3.)
    return f

@wp.func
def contact_force(velocity: V, normal: V, penetration: D, c: Parameters):
    speed=wp.dot(velocity,normal)
    normal_force=wp.max(c.stiffness*penetration-c.normal_damping*speed,D(0.))
    if penetration<=D(0.):normal_force=D(0.)
    tangent=velocity-speed*normal
    slip=wp.length(tangent)
    magnitude=wp.min(c.tangent_damping*slip,c.friction*normal_force)
    return normal_force*normal-magnitude*tangent/(slip+D(1e-14)), normal_force*normal

@wp.kernel
def table(p: wp.array2d(dtype=V),v: wp.array2d(dtype=V),q: wp.array2d(dtype=R),
          w: wp.array2d(dtype=V),radius: wp.array2d(dtype=D),force: wp.array2d(dtype=V),
          torque: wp.array2d(dtype=V),c: Parameters):
    b,s=wp.tid(); i=s/c.samples; sample=s%c.samples
    f=fraction(i,sample,c)
    point=(D(1.)-f)*p[b,i]+f*p[b,i+1]
    local=point-c.center
    fd=V(wp.abs(local[0]),wp.abs(local[1]),wp.abs(local[2]))-c.half
    axis=int(0)
    if fd[1]>fd[axis]:axis=1
    if fd[2]>fd[axis]:axis=2
    outside=V(wp.max(fd[0],D(0.)),wp.max(fd[1],D(0.)),wp.max(fd[2],D(0.)))
    od=wp.length(outside); signed=od+wp.min(fd[axis],D(0.))
    closest=c.center+V(wp.clamp(local[0],-c.half[0],c.half[0]),
                         wp.clamp(local[1],-c.half[1],c.half[1]),wp.clamp(local[2],-c.half[2],c.half[2]))
    normal=(point-closest)/wp.max(od,D(1e-15))
    if fd[axis]<=D(0.):
        normal=V(D(0.)); normal[axis]=D(1.)
        if local[axis]<D(0.):normal[axis]=D(-1.)
        closest=point-signed*normal
    offset=closest-point
    velocity=(D(1.)-f)*v[b,i]+f*v[b,i+1]+wp.cross(wp.transpose(q[b,i])*w[b,i],offset)
    force_world,normal_world=contact_force(velocity,normal,radius[b,i]-signed,c)
    fw=force_world/D(c.samples)
    wp.atomic_add(force,b,i,(D(1.)-f)*fw);wp.atomic_add(force,b,i+1,f*fw)
    wp.atomic_add(torque,b,i,q[b,i]*wp.cross(offset,fw))

@wp.func
def closest_segment(p: V,a: V,b: V):
    ab=b-a
    f=wp.clamp(wp.dot(p-a,ab)/wp.max(wp.dot(ab,ab),D(1e-20)),D(0.),D(1.))
    return a+f*ab

@wp.func
def closest_triangle(p: V,a: V,b: V,c: V):
    ab=b-a;ac=c-a;ap=p-a
    d00=wp.dot(ab,ab); d01=wp.dot(ab,ac); d11=wp.dot(ac,ac)
    d20=wp.dot(ap,ab);d21=wp.dot(ap,ac)
    denom=wp.max(d00*d11-d01*d01,D(1e-20))
    v=(d11*d20-d01*d21)/denom;w=(d00*d21-d01*d20)/denom
    best=closest_segment(p,a,b);d2=wp.length_sq(p-best)
    bc=closest_segment(p,b,c);bc2=wp.length_sq(p-bc)
    if bc2<d2:best=bc;d2=bc2
    ca=closest_segment(p,c,a);ca2=wp.length_sq(p-ca)
    if ca2<d2:best=ca;d2=ca2
    if v>=D(0.) and w>=D(0.) and v+w<=D(1.):
        projection=a+v*ab+w*ac
        if wp.length_sq(p-projection)<d2:best=projection
    return best

@wp.kernel
def mesh_contact(p: wp.array2d(dtype=V),v: wp.array2d(dtype=V),q: wp.array2d(dtype=R),
                 w: wp.array2d(dtype=V),radius: wp.array2d(dtype=D),force: wp.array2d(dtype=V),
                 torque: wp.array2d(dtype=V),positions: wp.array2d(dtype=V),rotations: wp.array2d(dtype=R),
                 velocities: wp.array2d(dtype=V),omegas: wp.array2d(dtype=V),
                 triangles: wp.array2d(dtype=V),normals: wp.array(dtype=V),
                 offsets: wp.array(dtype=wp.int32),bounds_lo: wp.array(dtype=V),bounds_hi: wp.array(dtype=V),
                 tree_lo: wp.array(dtype=V),tree_hi: wp.array(dtype=V),
                 tree_begin: wp.array(dtype=wp.int32),tree_end: wp.array(dtype=wp.int32),
                 tree_escape: wp.array(dtype=wp.int32),roots: wp.array(dtype=wp.int32),
                 reaction: wp.array2d(dtype=V),angular: wp.array2d(dtype=V),normal_reaction: wp.array2d(dtype=V),
                 rod_linear: wp.array2d(dtype=V),rod_angular: wp.array2d(dtype=V),
                 c: Parameters,time: D):
    b,m,i=wp.tid()
    origin=positions[b,m]+time*velocities[b,m]
    rot=rotation(omegas[b,m],time)*rotations[b,m]
    start=wp.transpose(rot)*(p[b,i]-origin)
    end=wp.transpose(rot)*(p[b,i+1]-origin)
    rad=radius[b,i];lo=bounds_lo[m];hi=bounds_hi[m]
    if wp.max(start[0],end[0])<lo[0]-rad or wp.min(start[0],end[0])>hi[0]+rad or wp.max(start[1],end[1])<lo[1]-rad or wp.min(start[1],end[1])>hi[1]+rad or wp.max(start[2],end[2])<lo[2]-rad or wp.min(start[2],end[2])>hi[2]+rad:
        return
    for sample in range(c.samples):
        f=fraction(i,sample,c)
        point=(D(1.)-f)*p[b,i]+f*p[b,i+1]
        local=(D(1.)-f)*start+f*end
        if local[0]<lo[0]-rad or local[0]>hi[0]+rad or local[1]<lo[1]-rad or local[1]>hi[1]+rad or local[2]<lo[2]-rad or local[2]>hi[2]+rad:
            continue
        # Classify before nearest-point search. Exterior points farther than
        # one rod radius cannot contact, so their exact search may be bounded
        # by that radius. Interior points retain an unbounded search: a deeply
        # embedded centerline still needs its nearest exit surface and force.
        inside=bool(True)
        separated=bool(False)
        for face in range(offsets[m],offsets[m+1]):
            plane_distance=wp.dot(normals[face],local-triangles[face,0])
            if plane_distance>D(0.):
                inside=False
                separated=plane_distance>rad
                break
        if separated:continue
        best=V(D(0.));best_d=rad*rad;best_face=int(-1)
        if inside:best_d=D(1e30)
        node=roots[m]
        while node<roots[m+1]:
            node_lo=tree_lo[node];node_hi=tree_hi[node]
            box_delta=V(wp.max(wp.max(node_lo[0]-local[0],local[0]-node_hi[0]),D(0.)),
                        wp.max(wp.max(node_lo[1]-local[1],local[1]-node_hi[1]),D(0.)),
                        wp.max(wp.max(node_lo[2]-local[2],local[2]-node_hi[2]),D(0.)))
            if wp.length_sq(box_delta)>best_d:
                node=tree_escape[node]
            else:
                for face in range(tree_begin[node],tree_end[node]):
                    candidate=closest_triangle(local,triangles[face,0],triangles[face,1],triangles[face,2])
                    d2=wp.length_sq(local-candidate)
                    if d2<best_d:best_d=d2;best=candidate;best_face=face
                node=node+1
        if best_face<0:continue
        distance=wp.sqrt(wp.max(best_d,D(0.)))
        signed=distance
        if inside:signed=-distance
        if rad-signed<=D(0.):continue
        closest=rot*best+origin
        normal=(local-best)/wp.max(distance,D(1e-12))
        if inside:normal=-normal
        normal=rot*normal
        offset=closest-point
        velocity=(D(1.)-f)*v[b,i]+f*v[b,i+1]+wp.cross(wp.transpose(q[b,i])*w[b,i],offset)
        velocity=velocity-velocities[b,m]-wp.cross(omegas[b,m],closest-origin)
        fw,nw=contact_force(velocity,normal,rad-signed,c)
        fw=fw/D(c.samples);nw=nw/D(c.samples)
        left=(D(1.)-f)*fw;right=f*fw
        tw=wp.cross(offset,fw)
        wp.atomic_add(force,b,i,left);wp.atomic_add(force,b,i+1,right)
        material=q[b,i]*tw
        wp.atomic_add(torque,b,i,material)
        wp.atomic_add(reaction,b,m,-fw)
        wp.atomic_add(angular,b,m,wp.cross(closest,-fw))
        wp.atomic_add(normal_reaction,b,m,-nw)
        # Independently accumulated rod wrench uses nodal moments + material couple.
        wp.atomic_add(rod_linear,b,m,left+right)
        wp.atomic_add(rod_angular,b,m,wp.cross(p[b,i],left)+wp.cross(p[b,i+1],right)+wp.transpose(q[b,i])*material)

@wp.kernel
def kick(p: wp.array2d(dtype=V),v: wp.array2d(dtype=V),q: wp.array2d(dtype=R),
         w: wp.array2d(dtype=V),dilation: wp.array2d(dtype=D),force: wp.array2d(dtype=V),
         torque: wp.array2d(dtype=V),c: Parameters,h: D,bad: wp.array(dtype=wp.int32)):
    b,i=wp.tid();mass=c.mass
    if i==0 or i==c.n:mass=mass*D(.5)
    vel=(v[b,i]+h*force[b,i]/mass)*wp.exp(-c.damping*h)
    pos=p[b,i]+D(.5)*h*vel
    v[b,i]=vel;p[b,i]=pos
    for k in range(3):
        if not wp.isfinite(pos[k]) or not wp.isfinite(vel[k]):wp.atomic_max(bad,b,1)
    if i<c.n:
        dil=dilation[b,i]
        omega=w[b,i]+h*wp.cw_mul(torque[b,i],c.inertia_inv)*dil
        rates=c.damping*c.mass*c.inertia_inv*h*dil
        omega=wp.cw_mul(omega,V(wp.exp(-rates[0]),wp.exp(-rates[1]),wp.exp(-rates[2])))
        frame=rotation(omega,-D(.5)*h)*q[b,i]
        w[b,i]=omega;q[b,i]=frame
        for k in range(3):
            if not wp.isfinite(omega[k]):wp.atomic_max(bad,b,1)
            for j in range(3):
                if not wp.isfinite(frame[k,j]):wp.atomic_max(bad,b,1)

@wp.kernel
def accumulate(reaction: wp.array2d(dtype=V),angular: wp.array2d(dtype=V),normal: wp.array2d(dtype=V),
               rod_linear: wp.array2d(dtype=V),rod_angular: wp.array2d(dtype=V),
               impulse: wp.array2d(dtype=V),angular_impulse: wp.array2d(dtype=V),normal_impulse: wp.array2d(dtype=V),
               linear_balance: wp.array2d(dtype=V),angular_balance: wp.array2d(dtype=V),peak: wp.array2d(dtype=D),h:D,
               bad: wp.array(dtype=wp.int32)):
    b,m=wp.tid()
    impulse[b,m]=impulse[b,m]+h*reaction[b,m]
    angular_impulse[b,m]=angular_impulse[b,m]+h*angular[b,m]
    normal_impulse[b,m]=normal_impulse[b,m]+h*normal[b,m]
    linear_balance[b,m]=linear_balance[b,m]+h*(reaction[b,m]+rod_linear[b,m])
    angular_balance[b,m]=angular_balance[b,m]+h*(angular[b,m]+rod_angular[b,m])
    peak[b,m]=wp.max(peak[b,m],wp.length(reaction[b,m]))
    for k in range(3):
        if not wp.isfinite(reaction[b,m][k]) or not wp.isfinite(angular[b,m][k]):wp.atomic_max(bad,b,1)
    reaction[b,m]=V(D(0.));angular[b,m]=V(D(0.));normal[b,m]=V(D(0.))
    rod_linear[b,m]=V(D(0.));rod_angular[b,m]=V(D(0.))

class BatchedCosserat:
    """Fixed-interface, CUDA-resident collection of independent flexible rods."""
    backend='warp_cuda_batched_cosserat'

    def __init__(self, config: GpuRodConfig, num_envs: int, device='cuda:0'):
        TorchCudaCosseratRod._validate_config(config)
        if not torch.cuda.is_available() or not str(device).startswith('cuda'):
            raise RuntimeError('BatchedCosserat requires CUDA')
        if not isinstance(num_envs,int) or num_envs<1:raise ValueError('num_envs must be positive')
        self.cfg=config;self.num_envs=num_envs;self.n_elem=config.n_elem
        self.device=torch.device(device);self.dtype=torch.float64
        wp.init();self.wp_device=wp.get_device(str(self.device));self._graphs={}
        self._torch={};self._wp={}
        self._stream=torch.cuda.Stream(device=self.device)
        b=num_envs;n=config.n_elem
        # Reuse only initialization/parameter definitions, never single-env stepping.
        reference=TorchCudaCosseratRod(config,device=str(self.device))
        self.node_mass=reference.node_mass
        self._allocate('p',reference.rod_position.T[None].repeat(b,1,1),V)
        self._allocate('v',reference.rod_velocity.T[None].repeat(b,1,1),V)
        self._allocate('q',reference.rod_director.permute(2,0,1)[None].repeat(b,1,1,1),R)
        self._allocate('w',reference.rod_omega.T[None].repeat(b,1,1),V)
        self.position=self._torch['p'].transpose(1,2);self.velocity=self._torch['v'].transpose(1,2)
        self.director=self._torch['q'].permute(0,2,3,1);self.omega=self._torch['w'].transpose(1,2)
        self._allocate('radius',torch.full((b,n),config.base_radius,device=self.device,dtype=self.dtype),D)
        self.radius=self._torch['radius']
        self._initial={key:self._torch[key][0].clone() for key in ('p','v','q','w','radius')}
        for key in ('tangent','stress','couple','weighted','torque'):
            self._zeros(key,(b,n,3),V)
        self._zeros('force',(b,n+1,3),V);self._zeros('dilation',(b,n),D)
        self._allocate('bad',torch.zeros(b,device=self.device,dtype=torch.int32),wp.int32)
        c=Parameters();c.n=n;c.samples=config.contact_subsamples;c.length=config.base_length/n
        c.radius=config.base_radius;c.mass=float(reference.node_mass[1].item());c.damping=config.damping
        c.shear=V(*reference._shear_diag[:,0].cpu().tolist());c.bend=V(*reference._bend_diag[:,0].cpu().tolist())
        c.inertia_inv=V(*reference._inverse_inertia_diag[:,0].cpu().tolist());c.gravity=V(*config.gravity)
        c.table=int(config.table_center is not None);c.center=V(*(config.table_center or (0.,0.,0.)))
        c.half=V(*(np.array(config.table_size or (0.,0.,0.))*.5))
        c.stiffness=config.contact_stiffness;c.normal_damping=config.contact_damping
        c.friction=config.friction_coefficient;c.tangent_damping=config.tangential_damping
        self._c=c;self.time=torch.zeros(b,device=self.device,dtype=self.dtype)
        self.register_meshes([],[])

    def _allocate(self,name,value,dtype):
        value=value.to(device=self.device).contiguous()
        self._torch[name]=value;self._wp[name]=wp.from_torch(value,dtype=dtype)

    def _zeros(self,name,shape,dtype):
        self._allocate(name,torch.zeros(shape,device=self.device,dtype=self.dtype),dtype)

    def register_meshes(self,vertices,faces):
        """Upload closed convex hull triangles; reject nonconvex/open geometry.

        Face winding is normalized once using the convex interior centroid.
        Convex plane classification is exact for the accepted hulls; distances
        still use triangles, including their edges and vertices.
        """
        if len(vertices)!=len(faces):raise ValueError('vertices/faces length mismatch')
        tris=[];normals=[];offsets=[0];lows=[];highs=[]
        tree_lo=[];tree_hi=[];tree_begin=[];tree_end=[];tree_escape=[];roots=[0]
        for v,f in zip(vertices,faces):
            v=np.asarray(v,dtype=np.float64);f=np.asarray(f,dtype=np.int32)
            if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or len(f)==0:
                raise ValueError('mesh must contain vertices (V,3), faces (F,3)')
            if not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v):raise ValueError('invalid mesh')
            t=v[f].copy();normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
            lengths=np.linalg.norm(normal,axis=1)
            if np.any(lengths<1e-18):raise ValueError('degenerate mesh face')
            normal/=lengths[:,None]
            flip=np.einsum('ij,ij->i',normal,t[:,0]-v.mean(0))<0
            t[flip]=t[flip][:,[0,2,1]];normal[flip]*=-1
            tolerance=max(np.ptp(v,axis=0).max()*1e-8,1e-10)
            if np.max(np.einsum('fvc,fc->fv',v[None]-t[:,0,None],normal))>tolerance:
                raise ValueError('mesh is not convex; convex plane classification would be invalid')
            edges=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
            if np.any(np.unique(edges,axis=0,return_counts=True)[1]!=2):raise ValueError('mesh is not closed')
            def build_tree(ids):
                node=len(tree_lo)
                tree_lo.append(t[ids].min(axis=(0,1)));tree_hi.append(t[ids].max(axis=(0,1)))
                tree_begin.append(0);tree_end.append(0);tree_escape.append(0)
                if len(ids)<=8:
                    tree_begin[node]=len(tris);tris.extend(t[ids]);normals.extend(normal[ids]);tree_end[node]=len(tris)
                else:
                    centers=t[ids].mean(axis=1);axis=int(np.ptp(centers,axis=0).argmax())
                    ordered=ids[np.argsort(centers[:,axis],kind='stable')];mid=len(ids)//2
                    build_tree(ordered[:mid]);build_tree(ordered[mid:])
                tree_escape[node]=len(tree_lo)
            build_tree(np.arange(len(t)))
            roots.append(len(tree_lo));offsets.append(len(tris));lows.append(v.min(0));highs.append(v.max(0))
        self.num_meshes=len(vertices);m=self.num_meshes;b=self.num_envs
        for name,data,dtype in [('triangles',np.asarray(tris).reshape(-1,3,3),V),('normals',np.asarray(normals).reshape(-1,3),V),
                ('offsets',np.asarray(offsets,np.int32),wp.int32),('lo',np.asarray(lows).reshape(-1,3),V),('hi',np.asarray(highs).reshape(-1,3),V)]:
            self._allocate(name,torch.as_tensor(data,device=self.device,dtype=torch.int32 if dtype==wp.int32 else self.dtype),dtype)
        for name,data,dtype in [('tree_lo',np.asarray(tree_lo).reshape(-1,3),V),('tree_hi',np.asarray(tree_hi).reshape(-1,3),V),
                ('tree_begin',tree_begin,wp.int32),('tree_end',tree_end,wp.int32),('tree_escape',tree_escape,wp.int32),('roots',roots,wp.int32)]:
            self._allocate(name,torch.as_tensor(data,device=self.device,dtype=torch.int32 if dtype==wp.int32 else self.dtype),dtype)
        for name in ('positions','velocities','omegas','reaction','angular','normal','rod_linear','rod_angular',
                     'linear_impulse','angular_origin_impulse','normal_impulse','linear_balance','angular_origin_balance'):
            self._zeros(name,(b,m,3),V)
        self._zeros('rotations',(b,m,3,3),R);self._zeros('peak_force',(b,m),D)
        self._graphs.clear()

    def reset(self,indices):
        indices=torch.as_tensor(indices,device=self.device,dtype=torch.long).flatten()
        for name,initial in self._initial.items():self._torch[name][indices]=initial
        for name in ('bad','force','torque','dilation','tangent','stress','couple','weighted','reaction','angular',
                     'normal','rod_linear','rod_angular','linear_impulse','angular_origin_impulse','normal_impulse',
                     'linear_balance','angular_origin_balance','peak_force'):
            self._torch[name][indices]=0
        self.time[indices]=0

    def _launch(self,kernel,dim,names,extra=()):
        wp.launch(kernel,dim,inputs=[self._wp[name] for name in names]+list(extra),device=self.wp_device)

    def _simulate(self,steps,h):
        b=self.num_envs;n=self.n_elem;m=self.num_meshes;c=self._c
        for key in ('bad','linear_impulse','angular_origin_impulse','normal_impulse','linear_balance','angular_origin_balance','peak_force'):
            self._wp[key].zero_()
        for step in range(steps):
            self._launch(drift,(b,n+1),('p','v','q','w'),(c,h))
            self._launch(geometry,(b,n),('p','q','radius','dilation','tangent','stress','couple','weighted'),(c,))
            self._launch(internal,(b,n+1),('p','v','q','w','dilation','tangent','stress','couple','weighted','force','torque'),(c,))
            if c.table:self._launch(table,(b,n*c.samples),('p','v','q','w','radius','force','torque'),(c,))
            if m:
                self._launch(mesh_contact,(b,m,n),('p','v','q','w','radius','force','torque','positions','rotations','velocities','omegas',
                    'triangles','normals','offsets','lo','hi','tree_lo','tree_hi','tree_begin','tree_end','tree_escape','roots','reaction','angular','normal','rod_linear','rod_angular'),(c,(step+.5)*h))
                self._launch(accumulate,(b,m),('reaction','angular','normal','rod_linear','rod_angular','linear_impulse','angular_origin_impulse',
                    'normal_impulse','linear_balance','angular_origin_balance','peak_force'),(h,self._wp['bad']))
            self._launch(kick,(b,n+1),('p','v','q','w','dilation','force','torque'),(c,h,self._wp['bad']))

    @torch.no_grad()
    def advance(self,duration,positions,rotations,velocities,omegas):
        duration=float(duration)
        if not math.isfinite(duration) or duration<=0:raise ValueError('duration must be finite and positive')
        for name,value in (('positions',positions),('rotations',rotations),('velocities',velocities),('omegas',omegas)):
            if tuple(value.shape)!=tuple(self._torch[name].shape):raise ValueError(f'{name} has incorrect shape')
            self._torch[name].copy_(value)
        # One boundary validation, no per-substep host synchronization.
        if not all(bool(torch.isfinite(self._torch[name]).all()) for name in ('p','v','q','w','positions','rotations','velocities','omegas')):
            raise FloatingPointError('non-finite batched rod state or prescribed mesh motion')
        steps=max(1,math.ceil(duration/self.cfg.dt));h=duration/steps;key=(steps,h)
        current=torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)
        stream=wp.stream_from_torch(self._stream)
        with wp.ScopedStream(stream):
            if key not in self._graphs:
                wp.load_module(module=__name__,device=self.wp_device)
                with wp.ScopedCapture(device=self.wp_device) as capture:self._simulate(steps,h)
                self._graphs[key]=capture.graph
            wp.capture_launch(self._graphs[key])
        current.wait_stream(self._stream)
        if bool(self._torch['bad'].any()):
            raise FloatingPointError('batched Cosserat integration/contact became non-finite')
        self.time+=duration
        return {name:self._torch[name] for name in ('linear_impulse','angular_origin_impulse','normal_impulse','peak_force',
                                                   'linear_balance','angular_origin_balance')}
