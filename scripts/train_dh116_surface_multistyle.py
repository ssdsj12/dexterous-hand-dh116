"""Sequential PPO blocks over an explicitly qualified reset family.

Each block builds one physical scene from its own pregrasp. Actor, critic,
PPO/auxiliary optimizers and completed-episode counters continue via --resume.
Auxiliary imitation samples both styles in every block. Style changes do not
zero saved episode counters; each native child constructs its physical scene.

Default finite mode: --updates-per-style 64 --cycles 1 (2048 iteration cap).
Episode mode: --num-envs 4096 --episodes-per-env 2048 --episodes-per-style 128
--resume saved.pt, with absolute minimum goals 128, 256, ..., 2048. Already
completed goals are skipped without changing the original style rotation.
This is a cumulative goal, not 2048 additional episodes after the checkpoint;
PPO updates can overshoot the minimum. There is no iteration cap in this mode.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--demonstrations',type=Path,required=True)
source=p.add_mutually_exclusive_group(required=True)
source.add_argument('--student-actor',type=Path)
source.add_argument('--resume',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=64)
p.add_argument('--updates-per-style',type=int,default=64)
p.add_argument('--cycles',type=int,default=1)
p.add_argument('--episodes-per-env',type=int,
    help='absolute completed-episode minimum for every env; replaces updates/cycles')
p.add_argument('--episodes-per-style',type=int,default=128,
    help='spacing between absolute minimum episode goals (default 128)')
p.add_argument('--rollout-steps',type=int,default=32)
p.add_argument('--imitation-mode',choices=('legacy','stable_hold_v2'),default='legacy')
p.add_argument('--noise-mode',choices=('legacy','holding_v2'),default='legacy')
p.add_argument('--contact-history',action='store_true')
p.add_argument('--validation-report',type=Path,action='append',default=[])
p.add_argument('--require-native-student-validation',action='store_true',default=False,
    help='opt in to native admission or the cumulative diagnostic pilot; strict checkpoints stay strict')
args=p.parse_args()
if args.resume and args.validation_report:
    raise ValueError('resume validation reports require exact saved-actor binding; '
                     'use a checkpoint with validated admission evidence')
if (not 2<=args.num_envs<=4096
        or (args.episodes_per_env is None
            and (not 1<=args.updates_per_style<=1024 or not 1<=args.cycles<=16))
        or (args.episodes_per_env is not None
            and (args.episodes_per_env<1 or args.episodes_per_style<1))):
    raise ValueError('explicit finite multistyle budget required')
import torch
from go2_pvcnn.control.dh116.tabletop_surface_imitation import validate_dataset
from go2_pvcnn.control.dh116.tabletop_surface_episode_schedule import episode_schedule


def checkpoint_episode_min(saved):
    """Read every saved counter; a sum/mean cannot establish a per-env budget."""
    episodes=saved.get('episodes')
    if episodes is None or tuple(episodes.shape)!=(args.num_envs,):
        raise ValueError('episode mode requires one saved episode counter per environment')
    values=episodes.tolist()
    if any(type(value) is not int or value<0 for value in values):
        raise ValueError('saved episode counters must be nonnegative integers')
    return min(values)


library=args.demonstrations.resolve()
data=torch.load(library,map_location='cpu',weights_only=False);validate_dataset(data)
family=data.get('reset_family')
if not family:raise ValueError('explicit qualified reset family required')
templates=dict(pinch_thumb_index=root/'configs/dh116_training/effective_pad_center_pregrasp_v1.json',
    tripod_thumb_index_middle=root/'configs/dh116_training/effective_pad_triad_pregrasp_v1.json')
styles=[member['label'] for member in family]
if len(set(styles))!=len(styles) or any(label not in templates for label in styles):
    raise ValueError('each declared style needs exactly one supported pregrasp template')
output=args.output.resolve()
if output.exists():raise FileExistsError(output)
checkpoint=args.resume.resolve() if args.resume else None
saved=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint else None
if saved is not None:
    inherited_strict=saved.get('require_native_student_validation',False)
    if type(inherited_strict) is not bool:
        raise ValueError('checkpoint require_native_student_validation must be a boolean')
    args.require_native_student_validation |= inherited_strict
iteration=int(saved['iteration']) if saved else 0
current_min=checkpoint_episode_min(saved) if saved and args.episodes_per_env is not None else 0
if args.episodes_per_env is not None:
    blocks=[(block.index,block.style,block.episode_goal) for block in episode_schedule(styles,
        current_min=current_min,episodes_per_env=args.episodes_per_env,
        episodes_per_style=args.episodes_per_style)]
else:
    blocks=[(index,label,None) for index,label in enumerate(styles*args.cycles)]
if args.episodes_per_env is None and iteration+len(styles)*args.cycles*args.updates_per_style>2048:
    raise ValueError('total explicit PPO iteration budget exceeds 2048')
output.mkdir(parents=True)
started=time.monotonic();history=[]
report=dict(status='initializing',num_envs=args.num_envs,styles=styles,cycles=args.cycles,
    updates_per_style=args.updates_per_style,initial_iteration=iteration,blocks=history,
    budget_mode='episodes' if args.episodes_per_env is not None else 'iterations',
    episodes_per_env=args.episodes_per_env,episodes_per_style=args.episodes_per_style,
    overall_episode_goal=args.episodes_per_env,
    initial_episodes_per_env_min=current_min if args.episodes_per_env is not None else None,
    episodes_per_env_min=current_min if args.episodes_per_env is not None else None,
    current_output=None,current_block=None,current_style=None,current_episode_goal=None,
    current_iteration=iteration,latest_checkpoint=str(checkpoint) if checkpoint else None,
    imitation_mode=args.imitation_mode,noise_mode=args.noise_mode,
    require_native_student_validation=args.require_native_student_validation,
    validation_reports=[str(path.resolve()) for path in args.validation_report],
    reset_distribution='one_declared_style_per_native_block;auxiliary_imitation_balanced_each_update')
def publish():
    report['elapsed_wall_s']=time.monotonic()-started
    tmp=output/'multistyle_status.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(output/'multistyle_status.json')


def refresh_child_status(child):
    try:
        monitor=json.loads((child/'monitor.json').read_text())
    except (OSError,json.JSONDecodeError):
        return
    if 'iteration' in monitor:report['current_iteration']=monitor['iteration']
    if 'episodes_per_env_min' in monitor:report['episodes_per_env_min']=monitor['episodes_per_env_min']
    if (child/'tensor_ppo.pt').is_file():report['latest_checkpoint']=str(child/'tensor_ppo.pt')


def fail(message):
    report.update(status='failed',error=message);publish()
    raise SystemExit(2)


publish()
for block_index,label,episode_goal in blocks:
    # A full PPO update can finish several goals. Never relaunch a completed one.
    if episode_goal is not None and current_min>=episode_goal:continue
    target=iteration+args.updates_per_style if episode_goal is None else None
    child=output/f'block_{block_index:02d}_{label}'
    command=['bash','scripts/python.sh','scripts/run_dh116_surface_student.py',
        '--output',str(child),'--num-envs',str(args.num_envs),
        '--rollout-steps',str(args.rollout_steps),'--demonstrations',str(library),
        '--pregrasp-candidates',str(templates[label]),
        '--imitation-mode',args.imitation_mode,'--noise-mode',args.noise_mode]
    command.extend(['--episodes-per-env',str(episode_goal)] if episode_goal is not None
                   else ['--updates',str(target)])
    command.extend(['--resume',str(checkpoint)] if checkpoint else ['--student-actor',str(args.student_actor.resolve())])
    if args.contact_history:command.append('--contact-history')
    if args.require_native_student_validation:command.append('--require-native-student-validation')
    if checkpoint is None:
        for path in args.validation_report:command.extend(['--validation-report',str(path.resolve())])
    report.update(status='training',current_block=block_index,current_style=label,
        current_episode_goal=episode_goal,target_iteration=target,current_output=str(child))
    publish()
    with (output/f'block_{block_index:02d}.log').open('x') as log:
        try:
            process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONUNBUFFERED='1'))
        except OSError as error:
            fail(f'could not launch native child: {error}')
        try:
            while True:
                try:
                    returncode=process.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    refresh_child_status(child);publish()
        except BaseException:
            process.terminate();process.wait()
            refresh_child_status(child)
            report.update(status='interrupted',process_exit_code=process.returncode);publish()
            raise
    refresh_child_status(child)
    if returncode:
        report.update(status='failed',process_exit_code=returncode);publish()
        raise SystemExit(returncode)
    try:
        monitor=json.loads((child/'monitor.json').read_text())
    except (OSError,json.JSONDecodeError) as error:
        fail(f'native child has no valid final monitor: {error}')
    if monitor.get('status')!='training_budget_complete':
        fail('native child did not complete its explicit budget')
    if episode_goal is not None:
        observed=monitor.get('episodes_per_env_min')
        if type(observed) is not int or observed<episode_goal:
            fail('native child did not complete its minimum per-env episode budget')
    elif monitor.get('iteration')!=target:
        fail('native child did not complete its explicit update budget')
    checkpoint=child/'tensor_ppo.pt'
    if not checkpoint.is_file():fail('native child has no continuation checkpoint')
    if episode_goal is not None:
        try:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            current_min=checkpoint_episode_min(saved)
        except Exception as error:
            fail(f'invalid episode continuation checkpoint: {error}')
        if current_min!=observed or saved.get('iteration')!=monitor.get('iteration'):
            fail('continuation checkpoint counters/iteration disagree with the final monitor')
    iteration=int(monitor['iteration'])
    history.append(dict(block=block_index,style=label,cycle=block_index//len(styles),
        episode_goal=episode_goal,episodes_per_env_min=monitor.get('episodes_per_env_min'),
        iteration=iteration,checkpoint=str(checkpoint),
        timesteps=monitor['timesteps'],successes=monitor['successes']))
    report.update(current_iteration=iteration,latest_checkpoint=str(checkpoint));publish()
report.update(status='training_budget_complete',process_exit_code=0);publish()
print('SURFACE_MULTISTYLE_PPO',json.dumps(report),flush=True)
