"""Audit the original pinch seed, without optimizing or claiming a grasp."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import SurfaceHandModel,CONTRACT
from go2_pvcnn.control.dh116.tabletop_surface_candidates import object_surface_points
from go2_pvcnn.control.dh116.tabletop_surface_objectives import finite_cylinder_sdf
from go2_pvcnn.control.dh116.tabletop_surface_teacher import implementation_identity

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--placement',type=Path,help='Optional audited palm offsets; force values are not used')
args=parser.parse_args();root=args.root.resolve();target=args.output.resolve()
if target.exists():raise FileExistsError('preserve existing candidate evidence')
torch.set_num_threads(6)
seed=json.loads((root/'configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json').read_text())
if args.placement is not None:
    placement=args.placement if args.placement.is_absolute() else root/args.placement
    offsets=torch.as_tensor(json.loads(placement.read_text())['palm_offsets_m'],dtype=torch.float64)
    if offsets.shape!=(2,3) or not torch.isfinite(offsets).all() or offsets.abs().max()>.003:
        raise ValueError('finite two-hand placement offsets within3mm required')
    for h in range(2):
        for axis in range(3):seed['roots'][h][axis]+=float(offsets[h,axis])
sources=[root/'assets/fixed_dual_panda_dh116_v1/dependencies'/name for name in
    ('d134680ab619_DH116_left.usd','3046ad2ed855_DH116_right.usd')]
metrics=[];active=[]
for h,(path,side) in enumerate(zip(sources,('left','right'))):
    model=SurfaceHandModel.from_usd(path,side,device='cpu')
    q=model.t([[seed['joint_positions'][h][JOINTS.index(name)] for name in ACTIVE]])
    pose=model.t([seed['roots'][h]]);obj=model.t([seed['wire_pose']])
    poses=model.fk(q,pose);s=model.sample_poses(poses);points=object_surface_points(obj[:,:3],obj[:,3:])
    distance=finite_cylinder_sdf(s['patches'],obj[:,:3],obj[:,3:]).abs().topk(3,largest=False).values.mean(-1)[0]
    geo=dict(penetration=float((-finite_cylinder_sdf(s['collision_points'],obj[:,:3],obj[:,3:])).clamp_min(0).max()*1000),
        object_in_hand=float(model.object_penetration_poses(poses,points)*1000),
        self=float(model.self_penetration_poses(poses)*1000),table=0.,reach=float(distance.mean()/.008),
        patch_distance_m=distance.tolist(),
        minimum_clearance_m=float(s['collision_points'][...,2].min()-.30))
    metrics.append(geo);active+=q[0].tolist()
print('ORIGINAL_PINCH_GEOMETRY',json.dumps(metrics),flush=True)
safe=all(max(g['penetration'],g['object_in_hand'],g['self'])<=.3 and g['minimum_clearance_m']>=.00029 for g in metrics)
geo={k:max(g[k] for g in metrics) for k in ('penetration','object_in_hand','self','table','reach')}
geo.update(minimum_clearance_m=min(g['minimum_clearance_m'] for g in metrics),patch_distance_m=[g['patch_distance_m'] for g in metrics])
out=dict(geometry_only=True,qualified=False,surface_model_contract=dict(CONTRACT),
    source_hashes={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in zip(('left','right'),sources)},
    implementation_hashes=implementation_identity(root),geometry_feasible_count=0,
    candidates=[dict(candidate_index=0,active_angles=active,wrist_pose_w=seed['roots'],geometry=geo,score=0.,qualified=False)] if safe else [])
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(out,indent=2)+'\n');print('SAFE_ORIGINAL_PINCH',safe)
