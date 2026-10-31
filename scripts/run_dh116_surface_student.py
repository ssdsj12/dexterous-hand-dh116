"""Native surface teacher replay or student PPO with the same physical reset.

Default: --updates 128 (absolute PPO iteration target, capped at 2048).
Episode mode: --num-envs 4096 --episodes-per-env 2048 --resume saved.pt
uses cumulative completed counters and has no iteration cap. Optional repair
modes: --imitation-mode stable_hold_v2 --noise-mode holding_v2, with native
--demonstrations and repeated --validation-report paths.
"""
from pathlib import Path
import argparse,hashlib,json,os,shlex,sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=16)
p.add_argument('--updates',type=int,default=128)
p.add_argument('--episodes-per-env',type=int,
    help='absolute completed-episode minimum for every env; replaces the update budget')
p.add_argument('--rollout-steps',type=int,default=32)
p.add_argument('--teacher-replay',type=Path)
p.add_argument('--student-actor',type=Path)
p.add_argument('--demonstrations',type=Path)
p.add_argument('--imitation-mode',choices=('legacy','stable_hold_v2'),default='legacy')
p.add_argument('--noise-mode',choices=('legacy','holding_v2'),default='legacy')
p.add_argument('--validation-report',type=Path,action='append',default=[])
p.add_argument('--require-native-student-validation',action='store_true',default=False,
    help='opt in to native admission or the cumulative diagnostic pilot; strict checkpoints stay strict')
p.add_argument('--first-episode-only',action='store_true')
p.add_argument('--stop-on-success',action='store_true')
p.add_argument('--contact-history',action='store_true')
p.add_argument('--pregrasp-candidates',type=Path,default=root/'configs/dh116_training/effective_pad_triad_pregrasp_v1.json')
p.add_argument('--candidate-index',type=int,default=0)
p.add_argument('--resume',type=Path)
p.add_argument('--eval-steps',type=int,default=0)
p.add_argument('--mpc',action='store_true')
args=p.parse_args()
if (not 2<=args.num_envs<=4096 or not 8<=args.rollout_steps<=128
        or (args.episodes_per_env is None and not 1<=args.updates<=2048)
        or (args.episodes_per_env is not None and args.episodes_per_env<1)):
    raise ValueError('explicit bounded environment/update/rollout budget required')
if args.teacher_replay and (args.student_actor or args.demonstrations):
    raise ValueError('teacher collection and student learning are distinct runs')
import torch
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import full_angles
os.chdir(root)
output=args.output.resolve();initial_path=output.with_name(output.name+'_initialization.json')
if output.exists() or initial_path.exists():raise FileExistsError(output)
output.parent.mkdir(parents=True,exist_ok=True)
candidates=args.pregrasp_candidates.resolve()
payload=json.loads(candidates.read_text());row=payload['candidates'][args.candidate_index]
if not row['static_wrench']['static_feasible'] or payload['support_clearance_m']!=.06:
    raise ValueError('validated static pregrasp/support configuration required')
initial=json.loads((root/'configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json').read_text())
roots=torch.tensor(row['wrist_pose_w'],dtype=torch.float64);roots[:,2]-=.06
initial.update(roots=roots.tolist(),grasp_command=row['active_angles'],
    joint_positions=full_angles(torch.tensor(row['active_angles'],dtype=torch.float64).reshape(2,6)).tolist(),
    surface_candidate=dict(geometry_only=True,qualified=False,requires_end_support_clearance_m=.06,
        input_sha256=hashlib.sha256(candidates.read_bytes()).hexdigest()))
initial_path.write_text(json.dumps(initial,indent=2))
flags=(root/'scripts/smoke_dh116_surface_grasp.sh').read_text();prefix='exec bash scripts/python.sh '
cmd=shlex.split(flags[flags.index(prefix)+len(prefix):].replace('\\\n',' '))
for flag,value in (('--num-envs',args.num_envs),('--max-iterations',args.updates),
    ('--rollout-steps',args.rollout_steps),('--surface-diagnostic-steps',10000 if args.teacher_replay else 0),
    ('--grasp-initialization',initial_path),('--cpu-threads',2),('--output',output)):
    cmd[cmd.index(flag)+1]=str(value)
if args.episodes_per_env is not None:
    index=cmd.index('--max-iterations');del cmd[index:index+2]
    cmd.extend(['--episodes-per-env',str(args.episodes_per_env)])
cmd.append('--native-sensor-batch')
for value,flag in ((args.teacher_replay,'--surface-teacher-replay'),
                   (args.student_actor,'--surface-student-actor'),(args.demonstrations,'--surface-demonstrations'),
                   (args.resume,'--resume')):
    if value:cmd.extend([flag,str(value.resolve())])
if args.eval_steps:cmd.extend(['--surface-student-eval-steps',str(args.eval_steps)])
if args.mpc:cmd.append('--surface-mpc')
if args.first_episode_only:cmd.append('--surface-student-first-episode-only')
if args.stop_on_success:cmd.append('--surface-student-stop-on-success')
if args.contact_history:cmd.append('--surface-contact-history')
for report in args.validation_report:cmd.extend(['--surface-student-validation-report',str(report.resolve())])
if args.require_native_student_validation:cmd.append('--require-native-student-validation')
if args.imitation_mode!='legacy':
    if not args.demonstrations:raise ValueError('stable hold imitation requires native demonstrations')
    cmd.extend(['--surface-imitation-mode',args.imitation_mode])
cmd.extend(['--surface-noise-mode',args.noise_mode])
print('SURFACE_STUDENT_LAUNCH',json.dumps(dict(command=cmd)),flush=True)
os.execv('/bin/bash',['bash','scripts/python.sh',*cmd])
