"""Native surface teacher replay or student PPO with the same physical reset.

Default: --updates 128 (absolute PPO iteration target, capped at 2048).
Episode mode: --num-envs 4096 --episodes-per-env 2048 --resume saved.pt
uses cumulative completed counters and has no iteration cap. Optional repair
modes: --imitation-mode stable_hold_v2 --noise-mode holding_v2, with native
--demonstrations and repeated --validation-report paths.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,shlex,sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import PROFILES

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=16)
p.add_argument('--updates',type=int,default=128)
p.add_argument('--episodes-per-env',type=int,
    help='absolute completed-episode minimum for every env; replaces the update budget')
p.add_argument('--rollout-steps',type=int,default=32)
p.add_argument('--teacher-replay',type=Path)
p.add_argument('--surface-action-audit',type=Path)
p.add_argument('--surface-dagger-collect',action='store_true')
p.add_argument('--surface-reward-profile',choices=PROFILES,default='baseline')
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
p.add_argument('--surface-feedback',action='store_true')
p.add_argument('--surface-feedback-telemetry',action='store_true')
p.add_argument('--pregrasp-candidates',type=Path,default=root/'configs/dh116_training/effective_pad_triad_pregrasp_v1.json')
p.add_argument('--mixed-pregrasp-candidates',type=Path,
    help='use this second pregrasp for the upper half of a single PPO environment batch')
p.add_argument('--depth-coordinate-probe',action='store_true',
    help='capture real rendered depth coordinates without PPO updates')
p.add_argument('--depth-camera-config',type=Path)
p.add_argument('--vision-student',action='store_true',
    help='fresh RGB-D PointNet actor with privileged critic; legacy actor transfer is forbidden')
p.add_argument('--vision-action-distribution',choices=('gaussian_clip','tanh_gaussian'),default='gaussian_clip')
p.add_argument('--vision-wrist-quaternion-preprocessing',
               choices=('raw','wrist_quaternion_hemisphere_v1'),default='raw')
p.add_argument('--vision-lift-reward-experiment',choices=('R0','R1','R2','R3','R4','R5'),default='R0')
p.add_argument('--vision-warm-start-optimizer',choices=('preserve','reset'),default='preserve')
p.add_argument('--vision-checkpoint-interval',type=int,default=10)
p.add_argument('--vision-experiment-initial-state-audit',action='store_true')
p.add_argument('--vision-experiment-initial-state-reference',type=Path)
p.add_argument('--vision-width',type=int,default=320)
p.add_argument('--vision-height',type=int,default=240)
p.add_argument('--vision-camera-batch-size',type=int,default=64)
p.add_argument('--vision-ppo-microbatch-size',type=int,default=128)
p.add_argument('--vision-inference-batch-size',type=int,default=128)
p.add_argument('--vision-visibility-mode',choices=('all','batch'),default='all')
p.add_argument('--vision-camera-mode',choices=('bounded','fixed'),default='bounded',
    help='fixed: one stationary real camera per environment, without sequential camera reuse')
p.add_argument('--physics-clone-mode',choices=('replicate','usd_parse'),default='replicate',
    help='usd_parse loads original clone physics directly and explicitly filters environments')
p.add_argument('--vision-resume-downsize',action='store_true',
    help='explicitly convert a compatible 4096 visual checkpoint to fixed 2048/1024 environments')
p.add_argument('--pose-summary',type=Path,default=root/'configs/dh116_training/pose_reference.json',
    help='explicit pose provenance; use the source checkpoint assets when running from a staging tree')
p.add_argument('--load-profile',type=Path,default=root/'configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json',
    help='explicit load-profile provenance; use the source checkpoint assets when running from a staging tree')
p.add_argument('--grasp-initialization',type=Path,
    help='reuse an authored initialization file when evaluating a checkpoint')
p.add_argument('--candidate-index',type=int,default=0)
p.add_argument('--resume',type=Path)
p.add_argument('--eval-steps',type=int,default=0)
p.add_argument('--mpc',action='store_true')
args=p.parse_args()
from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
try:validate_vision_action_options(args)
except ValueError as exc:p.error(str(exc))
if (not 2<=args.num_envs<=4096 or not 8<=args.rollout_steps<=128
        or (args.episodes_per_env is None and not 1<=args.updates<=2048)
        or (args.episodes_per_env is not None and args.episodes_per_env<1)):
    raise ValueError('explicit bounded environment/update/rollout budget required')
if args.depth_coordinate_probe and not args.depth_camera_config:
    p.error('--depth-coordinate-probe requires --depth-camera-config')
if args.vision_camera_mode!='bounded' and not args.vision_student:
    p.error('--vision-camera-mode requires --vision-student')
if args.vision_resume_downsize and (not args.vision_student or not args.resume or args.vision_camera_mode!='fixed'):
    p.error('--vision-resume-downsize requires a fixed visual resume')
if args.vision_student:
    if args.vision_camera_mode=='fixed' and (args.num_envs>2048 or args.vision_visibility_mode!='all'):
        p.error('fixed cameras require at most 2048 environments with all scene roots visible')
    if not ((args.vision_camera_mode=='fixed' or 1<=args.vision_camera_batch_size<=256) and 1<=args.vision_ppo_microbatch_size<=1024
            and 1<=args.vision_inference_batch_size<=1024):
        p.error('bounded camera/PPO/inference batch limits required')
    if (args.num_envs%2 or not args.depth_camera_config
            or not args.mixed_pregrasp_candidates or not args.surface_feedback
            or args.episodes_per_env is not None or args.depth_coordinate_probe
            or args.teacher_replay or args.student_actor or args.demonstrations
            or args.eval_steps or args.surface_dagger_collect):
        p.error('vision student requires 2..4096 even mixed RGB-D scenes, camera config, surface feedback and update budget; legacy inputs excluded')
if args.teacher_replay and (args.student_actor or args.demonstrations):
    raise ValueError('teacher collection and student learning are distinct runs')
import torch
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import full_angles
os.chdir(root)
output=args.output.resolve();initial_path=output.with_name(output.name+'_initialization.json')
mixed_path=output.with_name(output.name+'_mixed_initialization.json')
if output.exists() or initial_path.exists() or (args.mixed_pregrasp_candidates and mixed_path.exists()):
    raise FileExistsError(output)

def initialization_payload(candidates):
    candidates=candidates.resolve()
    payload=json.loads(candidates.read_text());row=payload['candidates'][args.candidate_index]
    if not row['static_wrench']['static_feasible'] or payload['support_clearance_m']!=.06:
        raise ValueError('validated static pregrasp/support configuration required')
    initial=json.loads((root/'configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json').read_text())
    roots=torch.tensor(row['wrist_pose_w'],dtype=torch.float64);roots[:,2]-=.06
    initial.update(roots=roots.tolist(),grasp_command=row['active_angles'],
        joint_positions=full_angles(torch.tensor(row['active_angles'],dtype=torch.float64).reshape(2,6)).tolist(),
        surface_candidate=dict(geometry_only=True,qualified=False,requires_end_support_clearance_m=.06,
            input_sha256=hashlib.sha256(candidates.read_bytes()).hexdigest()))
    return json.dumps(initial,indent=2)

initial_payload=mixed_payload=None
if args.grasp_initialization:
    initial_path=args.grasp_initialization.resolve()
else:
    initial_payload=initialization_payload(args.pregrasp_candidates)
if args.mixed_pregrasp_candidates:
    mixed_payload=initialization_payload(args.mixed_pregrasp_candidates)
if args.vision_action_distribution=='tanh_gaussian':
    # Validate the exact JSON bytes before creating either initialization file.
    reset_sha256={}
    for field,payload in [('primary_reset_sha256',initial_payload),
                          ('secondary_reset_sha256',mixed_payload)]:
        if payload is not None:reset_sha256[field]=hashlib.sha256(payload.encode('utf-8')).hexdigest()
    try:validate_vision_action_options(args,reset_sha256=reset_sha256)
    except ValueError as exc:p.error(str(exc))
output.parent.mkdir(parents=True,exist_ok=True)
if initial_payload is not None:initial_path.write_bytes(initial_payload.encode('utf-8'))
if mixed_payload is not None:mixed_path.write_bytes(mixed_payload.encode('utf-8'))
flags=(root/'scripts/smoke_dh116_surface_grasp.sh').read_text();prefix='exec bash scripts/python.sh '
cmd=shlex.split(flags[flags.index(prefix)+len(prefix):].replace('\\\n',' '))
if args.vision_student and args.vision_camera_mode=='fixed':
    kit_index=next(i for i,value in enumerate(cmd) if value.startswith('--kit_args='))
    cmd[kit_index],changed=re.subn(r'--/rtx/viewTile/limit=\d+',
        f'--/rtx/viewTile/limit={args.num_envs}',cmd[kit_index])
    if changed!=1:raise RuntimeError('fixed camera launch requires one explicit native view limit')
for flag,value in (('--num-envs',args.num_envs),('--max-iterations',args.updates),
    ('--rollout-steps',args.rollout_steps),('--surface-diagnostic-steps',10000 if args.teacher_replay else 0),
    ('--grasp-initialization',initial_path),('--cpu-threads',2),('--output',output)):
    cmd[cmd.index(flag)+1]=str(value)
cmd[cmd.index('--pose-summary')+1]=str(args.pose_summary.resolve())
cmd[cmd.index('--load-profile')+1]=str(args.load_profile.resolve())
if args.episodes_per_env is not None:
    index=cmd.index('--max-iterations');del cmd[index:index+2]
    cmd.extend(['--episodes-per-env',str(args.episodes_per_env)])
cmd.append('--native-sensor-batch')
cmd.extend(['--physics-clone-mode',args.physics_clone_mode])
if args.depth_coordinate_probe:
    cmd.extend(['--depth-coordinate-probe','--enable_cameras',
                '--depth-camera-config',str(args.depth_camera_config.resolve())])
if args.vision_student:
    cmd.extend(['--vision-student','--enable_cameras','--depth-camera-config',str(args.depth_camera_config.resolve()),
                '--vision-width',str(args.vision_width),'--vision-height',str(args.vision_height),
                '--vision-camera-batch-size',str(args.vision_camera_batch_size),
                '--vision-ppo-microbatch-size',str(args.vision_ppo_microbatch_size),
                '--vision-inference-batch-size',str(args.vision_inference_batch_size),
                '--vision-visibility-mode',args.vision_visibility_mode,
                '--vision-camera-mode',args.vision_camera_mode,
                '--vision-action-distribution',args.vision_action_distribution,
                '--vision-wrist-quaternion-preprocessing',args.vision_wrist_quaternion_preprocessing,
                '--vision-lift-reward-experiment',args.vision_lift_reward_experiment,
                '--vision-warm-start-optimizer',args.vision_warm_start_optimizer,
                '--vision-checkpoint-interval',str(args.vision_checkpoint_interval)])
    if args.vision_experiment_initial_state_audit:cmd.append('--vision-experiment-initial-state-audit')
    if args.vision_experiment_initial_state_reference:
        cmd.extend(['--vision-experiment-initial-state-reference',str(args.vision_experiment_initial_state_reference.resolve())])
    if args.vision_resume_downsize:cmd.append('--vision-resume-downsize')
if args.mixed_pregrasp_candidates:
    cmd.extend(['--mixed-grasp-initialization',str(mixed_path)])
for value,flag in ((args.teacher_replay,'--surface-teacher-replay'),
                   (args.surface_action_audit,'--surface-action-audit'),
                   (args.student_actor,'--surface-student-actor'),(args.demonstrations,'--surface-demonstrations'),
                   (args.resume,'--resume')):
    if value:cmd.extend([flag,str(value.resolve())])
if args.eval_steps:cmd.extend(['--surface-student-eval-steps',str(args.eval_steps)])
if args.mpc:cmd.append('--surface-mpc')
if args.first_episode_only:cmd.append('--surface-student-first-episode-only')
if args.stop_on_success:cmd.append('--surface-student-stop-on-success')
if args.contact_history:cmd.append('--surface-contact-history')
if args.surface_feedback:cmd.append('--surface-feedback')
if args.surface_feedback_telemetry:cmd.append('--surface-feedback-telemetry')
if args.surface_dagger_collect:cmd.append('--surface-dagger-collect')
for report in args.validation_report:cmd.extend(['--surface-student-validation-report',str(report.resolve())])
if args.require_native_student_validation:cmd.append('--require-native-student-validation')
if args.imitation_mode!='legacy':
    if not args.demonstrations:raise ValueError('stable hold imitation requires native demonstrations')
    cmd.extend(['--surface-imitation-mode',args.imitation_mode])
cmd.extend(['--surface-noise-mode',args.noise_mode])
if args.surface_reward_profile!='baseline':cmd.extend(['--surface-reward-profile',args.surface_reward_profile])
print('SURFACE_STUDENT_LAUNCH',json.dumps(dict(command=cmd)),flush=True)
os.execv('/bin/bash',['bash','scripts/python.sh',*cmd])
