"""Fit native hands to the recorded right-hand pinch; geometry is not force."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import compose_rotation
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import SurfaceHandModel,pose_matrix,CONTRACT
from go2_pvcnn.control.dh116.tabletop_surface_objectives import finite_cylinder_sdf
from go2_pvcnn.control.dh116.tabletop_surface_candidates import object_surface_points
from go2_pvcnn.control.dh116.tabletop_pinch_reference import reflect_y,proper_rigid_fit,reframe_transforms,gravity_cylinder_frame


def entry():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--starts',type=int,default=32);p.add_argument('--steps',type=int,default=320)
    p.add_argument('--wall-seconds',type=float,default=180.);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();root=a.root.resolve()
    if a.output.exists():raise FileExistsError('preserve existing search evidence')
    if not 2<=a.starts<=128 or not 1<=a.steps<=1000 or not 1<=a.wall_seconds<=600:raise ValueError('bounded search required')
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    c=json.loads((a.recording/'resolved_grip_lift.json').read_text())
    initial=json.loads((a.recording/'initial_cases.json').read_text())[35]
    raw=torch.load(a.recording/'raw_physics.pt',map_location='cpu',weights_only=False)
    recorded=raw['full_link_pose_w'][-1,35].clone();recorded_q=raw['q_measured'][-1,35].reshape(2,6).clone()
    obj=raw['object_pose_w'][-1,35].clone();origin=obj.new_tensor(initial['environment_origin_w'])
    recorded[...,:3]-=origin;obj[:3]-=origin
    obj_initial=obj.new_tensor([.43,0.,.308,1.,0.,0.,0.])
    body_transforms=pose_matrix(recorded.reshape(-1,7)).reshape(1,2,-1,4,4)
    measured_object=pose_matrix(obj[None])
    source_frame=gravity_cylinder_frame(obj[None,:3],measured_object[:,:3,1])
    target_frame=gravity_cylinder_frame(obj_initial[None,:3],obj.new_tensor([[0.,1.,0.]]))
    canonical=reframe_transforms(body_transforms,source_frame,target_frame)[0]
    del raw
    paths=[root/'assets/fixed_dual_panda_dh116_v1/dependencies'/n for n in ('d134680ab619_DH116_left.usd','3046ad2ed855_DH116_right.usd')]
    assets={s:hashlib.sha256(path.read_bytes()).hexdigest() for s,path in zip(('left','right'),paths)}
    if assets!=c['source_asset_sha256']:raise ValueError('reference source asset identity mismatch')
    models=[SurfaceHandModel.from_usd(path,side,a.device) for side,path in zip(('left','right'),paths)]
    rmodel=models[1];right_poses={body:canonical[1,c['link_body_names'][1].index(body.rsplit('/',1)[-1])][None].to(a.device) for body in rmodel.bodies}
    right_samples=rmodel.sample_poses(right_poses)
    center=rmodel.t([[.43,0.,.308]]);quat=rmodel.t([[1.,0.,0.,0.]])
    def distal_points(model,samples):
        return [torch.cat([points for shape,points in zip(model.shapes,samples['shape_points'])
            if shape['body'].endswith('/'+name)],1) for name in ('finger14_link','finger23_link')]
    reference_distance_mm=[];target_parts=[]
    for points in distal_points(rmodel,right_samples):
        distance=finite_cylinder_sdf(points,center,quat).abs()
        values,indices=distance.topk(3,dim=-1,largest=False)
        target_parts.append(points.gather(1,indices[...,None].expand(-1,-1,3)))
        reference_distance_mm.append((values.mean(-1)*1000).cpu().tolist())
    targets=torch.stack(target_parts,1)
    anchor_names=['base_link','finger11_link','finger12_link','finger13_link','finger14_link','finger21_link','finger22_link','finger23_link']
    anchor_target=torch.stack([right_poses[next(body for body in right_poses if body.endswith('/'+name))][:,:3,3] for name in anchor_names],1)
    low,high=(rmodel.t(x) for x in active_limits());n=a.starts
    all_sides=[];searches=[]
    for h,(side,model) in enumerate(zip(('left','right'),models)):
        contact_target=reflect_y(targets,center) if h==0 else targets
        anchors=reflect_y(anchor_target,center) if h==0 else anchor_target
        seed=recorded_q[h].to(a.device).maximum(low).minimum(high)
        qseed=seed[None].expand(n,-1).clone()
        qseed[::2]=recorded_q[1].to(a.device).maximum(low).minimum(high)
        generator=torch.Generator(device=a.device);generator.manual_seed(31+h)
        qseed+=(torch.rand(n,6,device=a.device,dtype=torch.float64,generator=generator)-.5)*.35
        qseed[:,3:]=.2+torch.rand(n,3,device=a.device,dtype=torch.float64,generator=generator)*.8
        qseed[1::4,2]=.15;qseed[2::4,2]=1.;qseed[3::4,0]=.6
        qseed[::8,3:]=.2
        qseed=qseed.maximum(low).minimum(high)
        local=model.fk(qseed,model.t([[0,0,0,1,0,0,0]]).expand(n,-1))
        anchor_keys=[next(body for body in local if body.endswith('/'+name)) for name in anchor_names]
        source=torch.stack([local[key][:,:3,3] for key in anchor_keys],1)
        rotation,translation=proper_rigid_fit(source,anchors.expand(n,-1,-1))
        xyzw=Rotation.from_matrix(rotation.cpu().numpy()).as_quat()
        root_quat=model.t(np.c_[xyzw[:,3],xyzw[:,:3]])
        x=torch.zeros(n,12,device=a.device,dtype=torch.float64)
        x[:,:6]=2*(qseed-low)/(high-low)-1
        x[:,6:9]=torch.randn(n,3,device=a.device,dtype=torch.float64,generator=generator)*.08
        x[:,8]+=.15
        x[:,9:]=torch.randn(n,3,device=a.device,dtype=torch.float64,generator=generator)*.12
        x=torch.nn.Parameter(x.clamp(-1,1));optimizer=torch.optim.Adam([x],lr=.01)
        begin=time.monotonic();best=torch.full((n,),float('inf'),device=a.device);best_x=x.detach().clone()
        def evaluate(v,fast=True):
            q=low+(v[:,:6]+1)/2*(high-low)
            pose=torch.cat((translation+v[:,6:9]*.04,compose_rotation(root_quat,v[:,9:]*.9)),1)
            actual=model.fk(q,pose);samples=model.sample_poses(actual)
            patches=distal_points(model,samples);points=samples['collision_points']
            match=torch.stack([torch.cdist(patch,contact_target[:,digit].expand(n,-1,-1)).amin(-2).square().mean(1)
                for digit,patch in enumerate(patches)],1).mean(1)/(.001**2)
            bone=(torch.stack([actual[key][:,:3,3] for key in anchor_keys],1)-anchors).square().mean((1,2))/(.01**2)
            sd=finite_cylinder_sdf(points,center.expand(n,-1),quat.expand(n,-1))
            hand_depth=(-sd).clamp_min(0).amax(-1)
            rod_depth=model.object_penetration_poses(actual,object_surface_points(center.expand(n,-1),quat.expand(n,-1)),fast=fast)
            self_depth=model.self_penetration_poses(actual,fast=fast)
            clearance=points[...,2].amin(-1)-.30
            table=(.0003-clearance).clamp_min(0)
            penalties=torch.stack((hand_depth,rod_depth,self_depth,table),1)/.001
            score=2*match+.1*bone+30*(penalties+penalties.square()).sum(-1)
            return score,dict(q=q,pose=pose,hand=hand_depth,rod=rod_depth,self=self_depth,clearance=clearance,
                match_mm=match.sqrt(),patch_distance=torch.stack([finite_cylinder_sdf(patch,center.expand(n,-1),quat.expand(n,-1)).abs().topk(3,largest=False).values.mean(-1)
                    for patch in patches],1))
        completed=0
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True);score,metrics=evaluate(x)
            better=score.detach()<best;best=torch.minimum(best,score.detach());best_x=torch.where(better[:,None],x.detach(),best_x)
            score.sum().backward()
            if x.grad is None or not torch.isfinite(x.grad).all():raise RuntimeError('invalid reference-fit gradient')
            torch.nn.utils.clip_grad_norm_([x],10.);optimizer.step()
            with torch.no_grad():x.clamp_(-1,1)
            completed=step+1
            if step%40==0:print('PINCH_FIT',side,completed,float(best.min()),time.monotonic()-begin,flush=True)
            if time.monotonic()-begin>a.wall_seconds:break
        with torch.no_grad():score,metrics=evaluate(best_x,False)
        rows=[]
        for i in score.argsort().tolist():
            safe=(max(float(metrics[key][i]) for key in ('hand','rod','self'))<=.0003 and float(metrics['clearance'][i])>=.0003)
            rows.append(dict(side=side,score=float(score[i]),safe=safe,angles=metrics['q'][i].cpu().tolist(),
                root=metrics['pose'][i].cpu().tolist(),hand_mm=float(metrics['hand'][i]*1000),rod_mm=float(metrics['rod'][i]*1000),
                self_mm=float(metrics['self'][i]*1000),minimum_clearance_m=float(metrics['clearance'][i]),
                match_mm=float(metrics['match_mm'][i]),patch_distance_m=metrics['patch_distance'][i].cpu().tolist()))
        all_sides.append(rows);searches.append(dict(side=side,steps=completed,elapsed_s=time.monotonic()-begin,safe=sum(row['safe'] for row in rows)))
        print('PINCH_FIT_DONE',json.dumps(searches[-1]),flush=True)
    valid=[[row for row in rows if row['safe'] and row['match_mm']<2.] for rows in all_sides]
    pairs=sorted(((left,right) for left in valid[0] for right in valid[1]),key=lambda pair:sum(row['score'] for row in pair))[:8]
    candidates=[]
    for i,pair in enumerate(pairs):
        with torch.no_grad():
            points=[model.sample(model.t([row['angles']]),model.t([row['root']]))['collision_points'][0]
                for model,row in zip(models,pair)]
            axis_gaps=torch.maximum(points[1].amin(0)-points[0].amax(0),points[0].amin(0)-points[1].amax(0))
            if not (axis_gaps>0).any():continue
        geometry=dict(penetration=max(row['hand_mm'] for row in pair),object_in_hand=max(row['rod_mm'] for row in pair),
            self=max(row['self_mm'] for row in pair),table=max(0.,.3-min(row['minimum_clearance_m'] for row in pair)*1000),reach=max(max(row['patch_distance_m']) for row in pair)/.008,
            minimum_clearance_m=min(row['minimum_clearance_m'] for row in pair),patch_distance_m=[row['patch_distance_m'] for row in pair])
        candidates.append(dict(candidate_index=i,active_angles=pair[0]['angles']+pair[1]['angles'],wrist_pose_w=[row['root'] for row in pair],
            geometry=geometry,score=sum(row['score'] for row in pair),qualified=False,reference_match_mm=[row['match_mm'] for row in pair],
            interhand_aabb_axis_gaps_mm=(axis_gaps*1000).cpu().tolist(),interhand_aabb_separated=True))
    source_files=[Path(__file__),root/'src/go2_pvcnn/control/dh116/tabletop_pinch_reference.py']
    out=dict(geometry_only=True,qualified=False,surface_model_contract=dict(CONTRACT,patch='recorded_closest_surface_points_on_full_distal_thumb14_and_index23_hulls'),source_hashes=assets,
        geometry_feasible_count=len(candidates),candidates=candidates,searches=searches,per_hand_candidates=all_sides,
        reference=dict(recording=str(a.recording),case=35,frame=4799,right_patch_distance_mm=reference_distance_mm,
            object_frame_alignment='gravity_aligned_roll_free_target_frame @ inverse(source_frame) @ recorded_link',
            target_points_w=targets.cpu().tolist(),force_target=None),
        producer_lf_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for path in source_files})
    a.output.write_text(json.dumps(out,indent=2)+'\n');print('REFERENCE_CANDIDATES',len(candidates),flush=True)


if __name__=='__main__':entry()
