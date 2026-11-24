"""Owned single-writer batch control; legacy audited transactions stay available."""
import numpy as np
import torch

from .tabletop_deformable_bimanual_step import (
    BimanualStepConnection,_capture,_clock,_bindings,_owned,SIDES,GROUPS,DT)
from .tabletop_owned_memory_geometry import freeze_owned
from .tabletop_command_kernels import finger_pre_guard,surface_final_guard,wrist_requested_targets
from .tabletop_surface_contract import ACTION_CONTRACT,validate_actions
from .linear_coupling import ACTIVE,MIMIC,active_limits


def containers(value):
    """Fresh metadata containers; retain owned read-only arrays without deepcopy."""
    if isinstance(value,dict):return {k:containers(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [containers(v) for v in value]
    return value


class CapturePlan:
    def __init__(self,bindings,lane,read_clock):
        self.bindings,self.lane,self.read_clock=tuple(bindings),lane,read_clock
        self.template=self.post=None
        self.measurements=0

    def _clock(self):
        return _clock(lambda:[self.read_clock()],(self.lane,))

    def mapping(self):
        _bindings(self.bindings,(self.lane,))
        if self.template is not None:
            for b in self.bindings:
                row=self.template["hands"][b.side]
                if (id(b.hand)!=row["instance_object_id"] or id(b.hand.root_physx_view)!=row["view_object_id"]
                        or tuple(b.hand.joint_names)!=tuple(row["joint_names"])
                        or tuple(b.hand.root_physx_view.prim_paths)!=tuple(row["prim_paths"])
                        or b.mount is None or b.mount.ids!=[b.hand.joint_names.index(n) for n in ("wrist_x","wrist_y","wrist_z")]
                        or b.mount.angular_ids!=[b.hand.joint_names.index(n) for n in ("wrist_roll","wrist_pitch","wrist_yaw")]):
                    raise ValueError("cached static SDK topology changed")

    def pre(self):
        self.mapping();now=self._clock()
        if self.post is not None:
            old=self.post["identity"][0]
            if all(old[k]==now[0][k] for k in old if k not in ("policy_step","substep")):
                result=containers(self.post);result["identity"]=now
                result["capture_mode"]="same_physical_boundary_previous_post"
                return result
        # Startup and each local Reset recertify types and topology.
        return self.capture(full=True)

    def capture(self,full=False):
        self.mapping();before=self._clock()
        if full or self.template is None:
            result=_capture(self.bindings,(self.lane,),lambda:self._clock())
            if any(not h["mounted"] for h in result["hands"].values()):
                raise ValueError("batched native path requires complete mounted hands")
            self.template=freeze_owned(result)
        else:
            result={"identity":before,"hands":{}}
            for b in self.bindings:
                h=b.hand;row=containers(self.template["hands"][b.side]);view=h.root_physx_view
                for key,getter in (("q",view.get_dof_positions),("qd",view.get_dof_velocities),
                                   ("backend_targets",view.get_dof_position_targets)):
                    value=_owned(getter())
                    if value.shape!=(1,len(row["joint_names"])) or value.dtype.kind!="f" or not np.isfinite(value).all():
                        raise ValueError("actual finite complete SDK dynamic state required")
                    row[key]=value
                cache=_owned(h.data.joint_pos_target)
                if cache.shape!=row["q"].shape or cache.dtype.kind!="f" or not np.isfinite(cache).all():
                    raise ValueError("finite actual command cache required")
                row["command_cache_targets"]=cache
                root=_owned(h.data.root_pose_w)
                native=_owned(view.get_root_transforms())[:,[0,1,2,6,3,4,5]]
                if (root.shape!=(1,7) or not np.isfinite(root).all() or not np.isfinite(native).all()
                        or not np.allclose(np.linalg.norm(root[:,3:],axis=1),1.,rtol=0,atol=1e-5)
                        or not np.allclose(root,native,rtol=0,atol=1e-6)):
                    raise ValueError("actual SDK/cache mount anchor differs")
                row["mount_anchor_pose_w"]=root;row["native_anchor_pose_w"]=native
                result["hands"][b.side]=row
        if self._clock()!=before:raise ValueError("SDK boundary changed during batch capture")
        self.measurements+=1
        return result

    def remember(self,post):
        self.post=freeze_owned(post)


class OwnedLaneConnection(BimanualStepConnection):
    def __init__(self,plan,pre,request):
        super().__init__(env_ids=(plan.lane,),solver_substeps=12)
        self.plan=plan
        self._pre,self._request,self._state=pre,request,"prepared"

    def _same_bindings(self,bindings):
        if len(bindings)!=len(self.plan.bindings) or any(a is not b for a,b in zip(bindings,self.plan.bindings)):
            raise ValueError("transaction bindings differ from owned lane")

    def submit(self,bindings,*,read_clock,projected_root_poses=None,projected_angles=None,projection_source=None):
        if self._state!="prepared":raise ValueError("prepared transaction required")
        self._same_bindings(bindings)
        self.plan.mapping()
        if _clock(read_clock,self.env_ids)!=list(self._pre["identity"]):
            raise ValueError("pre/action clock changed before submission")
        # Explicit owned single-writer mode requested by user. Legacy submit keeps full recapture.
        return self._submit_from_capture(bindings,self._pre,read_clock=read_clock,
            projected_root_poses=projected_root_poses,projected_angles=projected_angles,
            projection_source=projection_source,copy_result=False)

    def finish(self,bindings,*,read_clock):
        if self._state!="submitted":raise ValueError("submitted transaction required")
        self._state="failed"
        self._same_bindings(bindings)
        post=self.plan.capture()
        if _clock(read_clock,self.env_ids)!=post["identity"]:
            raise ValueError("finish caller clock differs from actual owned lane")
        result=self._finish_from_capture(post,copy_result=False)
        result["capture_mode"]="owned_single_writer_clock_only_submit"
        self.plan.remember(post)
        return result


class BatchedControl:
    def __init__(self,groups,clock,*,device="cuda:0"):
        self.plans=[CapturePlan(g,i,lambda i=i:clock(i)) for i,g in enumerate(groups)]
        self.device=torch.device(device)

    @property
    def capture_count(self):
        return sum(p.measurements for p in self.plans)

    def prepare(self,actions,palms,angles):
        validate_actions(actions)
        n=len(self.plans)
        if actions.shape!=(n,24) or palms.shape!=(n,2,3) or angles.shape!=(n,2,3):
            raise ValueError("complete batch action and original controller state required")
        packets=[p.pre() for p in self.plans]
        names={s:packets[0]["hands"][s]["joint_names"] for s in SIDES}
        for packet in packets:
            if any(packet["hands"][s]["joint_names"]!=names[s] for s in SIDES):
                raise ValueError("one actual original joint ordering per side required")
        action=actions.detach().to(device=self.device,dtype=torch.float64)
        command=[];q=[];mimic=[]
        for s in SIDES:
            rows=[packet["hands"][s] for packet in packets]
            ids=rows[0]["active_indices"]
            command.append(np.concatenate([r["backend_targets"][:,ids] for r in rows]))
            q.append(np.concatenate([r["q"][:,ids] for r in rows]))
            full=np.concatenate([r["q"] for r in rows])
            mimic.extend(full[:,names[s].index(f)]-ratio*full[:,names[s].index(m)]-offset
                         for f,m,ratio,offset in MIMIC)
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float64,device=self.device)
        command=tensor(np.concatenate(command,axis=1));q=tensor(np.concatenate(q,axis=1))
        mimic=tensor(np.stack(mimic,axis=1))
        lo,hi=(tensor(np.tile(v,2)) for v in active_limits())
        rates=dict(joint_rate_rad_s=ACTION_CONTRACT["joint_rate_rad_s"],
            wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"],
            rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"])
        target=finger_pre_guard(command,action,q,mimic,lo,hi,DT,rates,policy_owned=True)
        _,target=surface_final_guard(command,target,lo,hi,DT)
        palm,angle=wrist_requested_targets(palms.to(self.device).double(),angles.to(self.device).double(),action,rates,DT)
        packed=torch.cat((action,target,palm.flatten(1),angle.flatten(1)),1).cpu()
        output=[]
        for i,plan in enumerate(self.plans):
            a=packed[i:i+1,:24]
            req=dict(action24=a,action_contract=ACTION_CONTRACT,
                action_groups={k:a[:,low:high].numpy() for k,(low,high) in GROUPS.items()},
                requested_finger_targets_rad=packed[i:i+1,24:36],
                requested_palm_offsets_m=packed[i:i+1,36:42].reshape(1,2,3),
                requested_angles_rad=packed[i:i+1,42:48].reshape(1,2,3),
                command_baseline="actual PhysX backend target, copied before this action",
                controller_route="original policy_owned surface kernels; no force redistribution",
                wrist_projection_status="required_from_existing_caller; requested_offsets_are_not_world_poses")
            output.append(OwnedLaneConnection(plan,packets[i],req))
        return output


def batch_projection(connections,roots,bases,clearance,palms,angles,*,device):
    """One original projection kernel invocation for the complete owned batch."""
    from .tabletop_deformable_wrist_projection import project_policy_wrist_targets
    n=len(connections);dev=torch.device(device)
    conv=lambda t:t.to(device=dev,dtype=torch.float64)
    result=project_policy_wrist_targets(root_seed_w=conv(roots),measured_base_w=conv(bases),
        common_lift_offset_m=torch.zeros(n,device=dev,dtype=torch.float64),
        measured_clearance_m=conv(clearance),palm_offsets_m=conv(palms),
        requested_palm_offsets_m=conv(torch.cat([c._request["requested_palm_offsets_m"] for c in connections])),
        angles_rad=conv(angles),requested_angles_rad=conv(torch.cat([c._request["requested_angles_rad"] for c in connections])),
        allowed=torch.ones(n,device=dev,dtype=torch.bool),live=torch.ones(n,device=dev,dtype=torch.bool))
    result={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in result.items()}
    return [{k:v[i:i+1] if isinstance(v,torch.Tensor) else v for k,v in result.items()} for i in range(n)]


class BatchReactions:
    """Original mesh-origin to hand-COM world wrenches, vectorized over lanes."""
    def __init__(self,bindings,mesh_names,measured,*,device):
        from .tabletop_power_shared_rod import route_world_wrenches
        self.bindings=bindings;self.names=tuple(mesh_names);self.device=torch.device(device)
        self.keys={};self.ids={}
        for names,(poses,centers) in zip(bindings,measured,strict=True):
            old,_=route_world_wrenches({k:np.zeros(6) for k in (*names,"table")},names,poses,centers)
            for s in SIDES:
                id_to_key={v["body_index"]:k for k,v in names.items() if v["side"]==s}
                ordered=[id_to_key[j] for j in old[s]["body_ids"]]
                if names is bindings[0]:
                    self.keys[s]=ordered;self.ids[s]=old[s]["body_ids"]
                if ordered!=self.keys[s] or old[s]["body_ids"]!=self.ids[s]:
                    raise ValueError("same actual contact layout across lanes required")
        self.slots={s:[self.names.index(k) for k in self.keys[s]] for s in SIDES}

    def prepare(self,wrenches,measured):
        w=torch.as_tensor(wrenches,device=self.device,dtype=torch.float64)
        if w.shape!=(len(self.bindings),len(self.names),6) or not bool(torch.isfinite(w).all()):
            raise ValueError("finite complete actual batch mesh wrenches required")
        gpu={};cpu={}
        for s in SIDES:
            origins=np.stack([[row[0][k][0] for k in self.keys[s]] for row in measured]).astype(np.float32).astype(np.float64)
            com=np.stack([[row[1][k] for k in self.keys[s]] for row in measured])
            delta=torch.as_tensor(origins-com,device=self.device,dtype=torch.float64)
            if not bool(torch.isfinite(delta).all()):raise ValueError("finite actual origins/COM required")
            values=w[:,self.slots[s]]
            f=values[...,:3]
            tau=values[...,3:]+torch.linalg.cross(delta,f,dim=-1)
            if bool((torch.linalg.vector_norm(f,dim=-1)>30).any()):
                raise ValueError("original 30 N reaction guard failed")
            gpu[s]=dict(forces=f.float(),torques=tau.float())
            cpu[s]=torch.cat((f,tau),-1).cpu().numpy()
        rows=[]
        for i in range(len(self.bindings)):
            rows.append({s:dict(body_ids=list(self.ids[s]),forces=cpu[s][i:i+1,:,:3],
                                torques=cpu[s][i:i+1,:,3:],is_global=True) for s in SIDES})
        return rows,gpu

    def write(self,lane,composers,tensors):
        if (set(composers)!=set(SIDES) or composers["left"] is composers["right"]
                or any(not callable(getattr(c,name,None)) for c in composers.values()
                       for name in ("reset","set_forces_and_torques"))):
            raise ValueError("both independent original composers required")
        for s in SIDES:
            composers[s].reset()
            composers[s].set_forces_and_torques(forces=tensors[s]["forces"][lane:lane+1],
                torques=tensors[s]["torques"][lane:lane+1],body_ids=list(self.ids[s]),is_global=True)

def checked_links(value,n):
    """Retain LiveGeometryReader complete-link source guards for every style."""
    a=np.asarray(value)
    if (a.shape!=(n,2,23,7) or a.dtype.kind!="f" or not np.isfinite(a).all()
            or not np.allclose(np.linalg.norm(a[...,3:],axis=-1),1.,rtol=0,atol=1e-5)):
        raise ValueError("complete finite actual [N,2,23,7] unit XYZW link source required")
    return a
