"""Read-only native object-filtered manifold sampling; never synthesizes contact.

All loads are sums of native nonnegative normal scalars. Vectors are sums of
native scalar * native normal (force ON HAND), without renormalizing the raw
normal. Actual point is normal-load-weighted. Unloaded actuals/depth are zero,
body id is -1; the separately named FK reference always remains geometry_only.
"""
import torch
from .tabletop_effective_pad import build_pad_atlas, DISTAL_LINKS


def _rotation(pose):
    """Safe working copy only; raw invalid poses remain available to TensorEnv."""
    q = torch.nan_to_num(pose[...,3:], nan=0., posinf=0., neginf=0.)
    q = q/q.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    w,x,y,z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def _owners(count, start, capacity):
    """Map flat native slots to env with sorted intervals, no per-env .item().

    Native start offsets need not follow env order. Empty/invalid intervals do
    not own slots. Overlapping intervals mark both implicated environments bad.
    """
    invalid = (count<0)|(start<0)|(start>capacity)|(count>capacity-start)
    eligible = ~invalid & (count>0)
    ordered, perm = torch.sort(torch.where(eligible,start,capacity+1))
    ends = ordered + torch.where(eligible[perm],count[perm],0)
    running_end, furthest = torch.cummax(ends,0)
    overlap = (ordered[1:]<running_end[:-1]) & eligible[perm[1:]]
    conflicts = torch.zeros_like(count)
    conflicts.scatter_add_(0,perm[1:],overlap.long())
    conflicts.scatter_add_(0,perm[furthest[:-1]],overlap.long())
    invalid = invalid | (conflicts>0)
    slots = torch.arange(capacity,device=count.device)
    index = (torch.searchsorted(ordered,slots,right=True)-1).clamp_min(0)
    env = perm[index]
    active = eligible[env] & ~invalid[env] & (slots>=start[env]) & (slots<start[env]+count[env])
    return env,active,invalid


def _on_patch(point, normal, row):
    rel = point-row['center_local_m']
    uv = torch.stack(((rel*row['tangent_u']).sum(-1),(rel*row['tangent_v']).sum(-1)),-1)
    region = ((uv/row['half_widths_m'])**2).sum(-1)<=1.
    # Chunk actual CAD plane envelope to bound memory for large native buffers.
    best = torch.full(point.shape[:-1],-torch.inf,device=point.device,dtype=point.dtype)
    direction = torch.zeros_like(point)
    planes = row['planes_local']
    for begin in range(0,len(planes),32):
        group = planes[begin:begin+32]
        signed = point@group[:,:3].T+group[:,3]
        distance, index = signed.max(-1)
        take = distance>best
        direction = torch.where(take[:,None],group[index,:3],direction)
        best = torch.maximum(best,distance)
    cosine = (normal*direction).sum(-1)/normal.norm(dim=-1).clamp_min(1e-30)
    return region & (best.abs()<=row['surface_tolerance_m']) & (cosine<=-row['normal_cosine_min'])


class PadContactSampler:
    def __init__(self, world):
        self.world = world
        self.n, self.device = world.n, world.device
        links = tuple(world.w.sensor_links)
        if (len(world.hands)!=2 or len(links)!=17 or len(set(links))!=17 or
                len(world.w.sensors)!=34 or not set(DISTAL_LINKS).issubset(links)):
            raise ValueError('full seventeen object-contact sensor links per hand required')
        self.bindings=[]
        for hi,hand in enumerate(world.hands):
            if any(link not in hand.body_names for link in links):
                raise ValueError('sensor body missing from articulation')
            for li,link in enumerate(links):
                sensor = world.w.sensors[hi*17+li]
                if getattr(sensor.contact_physx_view,'filter_count',1)!=1:
                    raise ValueError('one object-only native contact filter required')
                self.bindings.append((hi,hand.body_names.index(link),
                                      DISTAL_LINKS.index(link) if link in DISTAL_LINKS else None,sensor))
        self.atlas = build_pad_atlas(world)
        self.rows={}
        for hi,side in enumerate(('left','right')):
            for di,link in enumerate(DISTAL_LINKS):
                row=self.atlas['sides'][side][link]
                self.rows[hi,di]={key:torch.as_tensor(row[key],device=self.device,dtype=torch.float64)
                    for key in ('center_local_m','tangent_u','tangent_v','half_widths_m','planes_local')}
                for key in ('surface_tolerance_m','normal_cosine_min'): self.rows[hi,di][key]=row[key]

    @torch.no_grad()
    def sample(self,*,contacts=True):
        if contacts and getattr(self,'use_batched_native',False):
            from .tabletop_effective_pad_batch import sample_native_batched
            result=sample_native_batched(self)
            if result is not None:return result
        w=self.world; n=self.n
        loads=torch.zeros(n,2,5,device=self.device,dtype=torch.float64)
        vectors=torch.zeros(n,2,5,3,device=self.device,dtype=torch.float64)
        points=torch.zeros_like(vectors); references=torch.zeros_like(vectors)
        depth=torch.zeros_like(loads)
        body_ids=torch.full((n,2,5),-1,device=self.device,dtype=torch.long)
        total=torch.zeros(n,2,device=self.device,dtype=torch.float64)
        signed_force=torch.zeros(n,2,3,device=self.device,dtype=torch.float64)
        bad=torch.zeros(n,device=self.device,dtype=torch.bool)
        poses=[hand.data.body_link_pose_w.double() for hand in w.hands]
        rotations=[];positions=[]
        for hi,pose in enumerate(poses):
            bad |= (~torch.isfinite(pose).all((-1,-2)) |
                    ((pose[...,3:].norm(dim=-1)-1).abs()>1e-3).any(-1))
            rotations.append(_rotation(pose))
            positions.append(torch.nan_to_num(pose[...,:3],nan=0.,posinf=0.,neginf=0.))
            for di,link in enumerate(DISTAL_LINKS):
                bi=w.hands[hi].body_names.index(link)
                references[:,hi,di]=(rotations[hi][:,bi]@self.rows[hi,di]['center_local_m'])+positions[hi][:,bi]
        for hi,bi,di,sensor in (self.bindings if contacts else ()):
            force,p,normal,separation,count,start=sensor.contact_physx_view.get_contact_data(dt=w.dt)
            f=force.reshape(-1).double();p=p.reshape(-1,3).double();normal=normal.reshape(-1,3).double()
            sep=separation.reshape(-1).double()
            if count.numel()!=n or start.numel()!=n or not (len(f)==len(p)==len(normal)==len(sep)):
                bad.fill_(True)
                continue
            count=count.reshape(n).long();start=start.reshape(n).long()
            env,active,invalid=_owners(count,start,len(f))
            bad |= invalid
            if not len(f): continue
            finite=(torch.isfinite(f)&torch.isfinite(p).all(-1)&torch.isfinite(normal).all(-1)&torch.isfinite(sep))
            nn=normal.norm(dim=-1)
            # Native compliant-contact reports can contain finite signed
            # separation/release terms. Sign alone is not an invalid state.
            # Only positive compression may qualify as gripping contact.
            sane=finite&((nn-1).abs()<=.05)
            faults=torch.zeros(n,device=self.device,dtype=torch.long)
            faults.scatter_add_(0,env,(active&~sane).long())
            bad |= faults>0
            safe_f=torch.where(active&finite,f,0.)
            total[:,hi].scatter_add_(0,env,safe_f.abs())
            signed_force[:,hi].scatter_add_(0,env[:,None].expand(-1,3),torch.nan_to_num(normal)*safe_f[:,None])
            if di is None: continue
            safe_p=torch.nan_to_num(p,nan=0.,posinf=0.,neginf=0.)
            safe_normal=torch.nan_to_num(normal,nan=0.,posinf=0.,neginf=0.)
            r=rotations[hi][env,bi]
            local_p=torch.einsum('ki,kij->kj',safe_p-positions[hi][env,bi],r)
            local_n=torch.einsum('ki,kij->kj',safe_normal,r)
            accept=active&sane&(f>0)&_on_patch(local_p,local_n,self.rows[hi,di])
            accepted_f=torch.where(accept,f,0.)
            loads[:,hi,di].scatter_add_(0,env,accepted_f)
            vectors[:,hi,di].scatter_add_(0,env[:,None].expand(-1,3),safe_normal*accepted_f[:,None])
            points[:,hi,di].scatter_add_(0,env[:,None].expand(-1,3),safe_p*accepted_f[:,None])
            accepted_depth=torch.where(accept,(-sep).clamp_min(0),0.)
            depth[:,hi,di].scatter_reduce_(0,env,accepted_depth,reduce='amax',include_self=True)
            body_ids[:,hi,di]=bi
        bad |= (~torch.isfinite(signed_force).all((1,2)) | ~torch.isfinite(loads).all((1,2)) | ~torch.isfinite(vectors).all((1,2,3)) |
                ~torch.isfinite(points).all((1,2,3)) | ~torch.isfinite(references).all((1,2,3)))
        valid=(loads>0)&~bad[:,None,None]
        points=torch.where(valid[...,None],points/loads.clamp_min(1e-30)[...,None],0.)
        loads=torch.where(valid,loads,0.)
        vectors=torch.where(valid[...,None],vectors,0.)
        depth=torch.where(valid,depth,0.)
        references=torch.where(bad[:,None,None,None],0.,references)
        body_ids=torch.where(valid,body_ids,-1)
        rejected=torch.nan_to_num((total-loads.sum(-1)).clamp_min(0),nan=0.,posinf=0.,neginf=0.)
        self.last_signed_force_w=signed_force
        return dict(effective_pad_loads_n=loads,effective_pad_vectors_w=vectors,
                    effective_pad_points_w=points,effective_pad_valid=valid,effective_pad_body_ids=body_ids,
                    rejected_contact_loads_n=rejected,effective_pad_depth_m=depth,
                    pad_reference_points_w=references,effective_pad_bad_env=bad)
