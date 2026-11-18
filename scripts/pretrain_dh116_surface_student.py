"""Fit the surface student actor from qualified native replay pairs."""
import argparse,json
from pathlib import Path
import torch
from go2_pvcnn.control.dh116.tabletop_surface_imitation import pretrain,make_actor

p=argparse.ArgumentParser()
p.add_argument('--demonstrations',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--steps',type=int,default=2000)
p.add_argument('--batch-size',type=int,default=256)
p.add_argument('--device',default='cpu')
p.add_argument('--seed',type=int,default=42)
p.add_argument('--warm-start',type=Path,help='immutable compatible actor for supervised refinement')
p.add_argument('--learning-rate',type=float,default=1e-3)
p.add_argument('--holding-weight',type=float,default=0.,help='extra weight for teacher zero-velocity retention samples')
p.add_argument('--wrist-weight',type=float,default=1.,help='weight of wrist translation/rotation reconstruction')
p.add_argument('--extend-contact-history',action='store_true',help='explicitly initialize new causal inputs with zero weights')
args=p.parse_args()
if args.output.exists():raise FileExistsError(args.output)
if not 1<=args.steps<=20000:raise ValueError('explicit finite BC budget required')
data=torch.load(args.demonstrations,map_location='cpu',weights_only=False)
metrics=pretrain(data,args.output,steps=args.steps,batch_size=args.batch_size,seed=args.seed,device=args.device,
    warm_start=args.warm_start,learning_rate=args.learning_rate,holding_weight=args.holding_weight,
    wrist_weight=args.wrist_weight,extend_contact_history=args.extend_contact_history)
payload=torch.load(args.output,map_location='cpu',weights_only=True)
actor=make_actor(payload['contract']['observation_size']);actor.load_state_dict(payload['actor'])
with torch.no_grad():
    error=(actor(data['observations'].float())-data['actions'].float()).square().mean(0).sqrt()
metrics.update(per_action_rmse=error.tolist(),assessment='training_reconstruction_only;native_student_success_not_yet_evaluated',
    teacher_successful_replays=data['qualification']['successful_replays'],
    actor_path=str(args.output),demonstrations=str(args.demonstrations))
args.output.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2))
print('SURFACE_STUDENT_BC',json.dumps(metrics),flush=True)
