#!/usr/bin/env python3
"""Offline native grasp generation. No Isaac app, physics or teacher labels."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import SurfaceHandModel,CONTRACT as SURFACE_MODEL
from go2_pvcnn.control.dh116.tabletop_surface_candidates import optimize
from go2_pvcnn.control.dh116.tabletop_surface_teacher import implementation_identity


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=Path,default=root/'configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--starts',type=int,default=24)
    parser.add_argument('--steps',type=int,default=240)
    parser.add_argument('--wall-seconds',type=float,default=240.)
    parser.add_argument('--maximum-candidates',type=int,default=8)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=6)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve existing candidate evidence')
    torch.set_num_threads(args.threads)
    source_identity=implementation_identity(root)
    seed=json.loads(args.seed.read_text())
    names=['d134680ab619_DH116_left.usd','3046ad2ed855_DH116_right.usd']
    sources=[root/'assets/fixed_dual_panda_dh116_v1/dependencies'/name for name in names]
    models=[SurfaceHandModel.from_usd(source,side,device=args.device) for source,side in zip(sources,('left','right'))]
    report=optimize(models,seed['roots'],seed['grasp_command'],seed['wire_pose'],
        starts=args.starts,steps=args.steps,maximum=args.maximum_candidates,wall_seconds=args.wall_seconds,
        progress=lambda row:print('SURFACE_OPT',json.dumps(row),flush=True))
    report.update(implementation_hashes=source_identity,surface_model_contract=dict(SURFACE_MODEL),seed_path=str(args.seed),source_hashes={side:hashlib.sha256(source.read_bytes()).hexdigest()
        for source,side in zip(sources,('left','right'))},seed_sha256=hashlib.sha256(args.seed.read_bytes()).hexdigest(),
        mechanical_counts=dict(active=12,passive=10,finger_total=22,external_wrist=12),
        object_pose=seed['wire_pose'],seed_roots=seed['roots'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('SURFACE_CANDIDATES',json.dumps(dict(output=str(args.output),iterations=report['iterations'],
        elapsed_s=report['elapsed_s'],geometry_feasible_count=report['geometry_feasible_count'],
        scores=[c['score'] for c in report['candidates']],qualified=False)),flush=True)


if __name__=='__main__':main()
