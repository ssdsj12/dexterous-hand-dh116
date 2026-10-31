"""Bounded real PPO update smoke from the validated static pregrasp seed."""
from pathlib import Path
import argparse,hashlib,json,os,shlex,sys
import torch

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import full_angles

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=16)
p.add_argument('--updates',type=int,default=2)
p.add_argument('--rollout-steps',type=int,default=16)
args=p.parse_args()
if not 2<=args.num_envs<=64 or not 1<=args.updates<=4 or not 8<=args.rollout_steps<=32:
    raise ValueError('bounded smoke only: 2..64 envs, 1..4 updates, 8..32 rollout steps')
os.chdir(root)
output=args.output.resolve()
initial_path=output.with_name(output.name+'_initialization.json')
if output.exists() or initial_path.exists():raise FileExistsError(output)
output.parent.mkdir(parents=True,exist_ok=True)
candidates=root/'configs/dh116_training/effective_pad_triad_pregrasp_v1.json'
payload=json.loads(candidates.read_text());row=payload['candidates'][0]
assert row['static_wrench']['static_feasible'] and payload['support_clearance_m']==.06
initial=json.loads((root/'configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json').read_text())
roots=torch.tensor(row['wrist_pose_w'],dtype=torch.float64);roots[:,2]-=.06
initial.update(roots=roots.tolist(),grasp_command=row['active_angles'],
    joint_positions=full_angles(torch.tensor(row['active_angles'],dtype=torch.float64).reshape(2,6)).tolist(),
    surface_candidate=dict(geometry_only=True,qualified=False,requires_end_support_clearance_m=.06,
        input_sha256=hashlib.sha256(candidates.read_bytes()).hexdigest()))
initial_path.write_text(json.dumps(initial,indent=2))
flags=(root/'scripts/smoke_dh116_surface_grasp.sh').read_text()
prefix='exec bash scripts/python.sh '
cmd=shlex.split(flags[flags.index(prefix)+len(prefix):].replace('\\\n',' '))
for flag,value in (('--num-envs',args.num_envs),('--max-iterations',args.updates),
    ('--rollout-steps',args.rollout_steps),('--surface-diagnostic-steps',0),
    ('--grasp-initialization',initial_path),('--cpu-threads',2),('--output',output)):
    cmd[cmd.index(flag)+1]=str(value)
cmd.append('--native-sensor-batch')
print('BOUNDED_PPO_SMOKE',json.dumps(dict(output=str(output),num_envs=args.num_envs,
    updates=args.updates,rollout_steps=args.rollout_steps,teacher_policy_loaded=False,
    initialization='static_pregrasp_only;original_support_height',command=cmd)),flush=True)
os.execv('/bin/bash',['bash','scripts/python.sh',*cmd])
