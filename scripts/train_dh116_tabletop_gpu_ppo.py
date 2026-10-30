"""Cloned GPU physics for adaptive close-to-lift PPO, with host task control."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import traceback
import time
from go2_pvcnn.control.dh116.tabletop_training import write_json, canonical_contract
from go2_pvcnn.control.dh116.tabletop_vec_config import (add_mimic_arguments,
    validate_mimic_options, add_protected_mimic_arguments, validate_protected_mimic_options)
from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import (
    validate_adaptive_options, adaptive_initialization, adaptive_contract_fields, object_spec)
from go2_pvcnn.control.dh116.tabletop_material import (material_contract, verify_material_properties,
    hand_material_contract, verify_hand_material_properties)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pose-summary', type=Path, required=True)
parser.add_argument('--close-reset-snapshot', type=Path)
parser.add_argument('--adaptive-grasp', action='store_true',
                    help='PPO controls 12 finger targets, 6 palm offsets and 2 grip stiffness targets')
parser.add_argument('--grasp-initialization', type=Path,
                    help='explicit authored geometry for fresh adaptive grasp training')
parser.add_argument('--force-feedback', action='store_true',
                    help='PPO targets measured hand reaction load; lift requires stable measured contact')
parser.add_argument('--full-hand-contact', action='store_true',
                    help='versioned rigid-object contact on all hand links, 148D observation')
parser.add_argument('--joint-envelope', type=Path,
                    help='explicit per-joint lower/center/upper targets for free grasp')
parser.add_argument('--load-profile', type=Path, help='explicit estimated force/drive parameters; strict checkpoint provenance')
parser.add_argument('--object-kind', choices=('solid_cylinder','hollow_straw'),default='solid_cylinder')
parser.add_argument('--solid-cylinder-radius-m', type=float, default=.008,
                    help='radius for the built-in solid cylinder; non-default radii require fresh geometry/checkpoints')
parser.add_argument('--solid-cylinder-density-kg-m3', type=float, default=None,
                    help='density for the built-in solid cylinder; omitted uses 200 kg/m^3 for non-default radii')
parser.add_argument('--object-material', choices=('default','high_friction','rough_rod','extreme_grip'),default='default')
parser.add_argument('--hand-material', choices=('default','grip','grip_compliant'),default='default',
                    help='explicit hand collider friction; grip is an uncalibrated training profile')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=4)
parser.add_argument('--cpu-threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', '1')),
                    help='PyTorch host threads; defaults to the launch environment')
parser.add_argument('--env-spacing-m', type=float, default=2.)
budget = parser.add_mutually_exclusive_group()
budget.add_argument('--timesteps', type=int, default=65536)
budget.add_argument('--episodes-per-env', type=int,
                    help='train until every env completes this many episodes, checked after full PPO updates')
parser.add_argument('--eval-episodes', type=int, default=8)
parser.add_argument('--checkpoint-every', type=int, default=2048)
parser.add_argument('--rollout-steps', type=int,
                    help='per-env steps before each fresh-policy PPO update (default 64); resume uses the checkpoint value')
parser.add_argument('--tensor-pipeline', action='store_true',
                    help='device-resident rigid task and RSL-RL PPO; defaults to 24 steps per iteration')
parser.add_argument('--pressure-relief', action='store_true',
                    help='tensor controller: measured overload overrides policy closing targets; explicit native checkpoint migration allowed')
parser.add_argument('--palm-transport',choices=('fixed_pose','driven_carriage'),default='fixed_pose',
                    help='diagnostic moving wrist support with real surface velocity')
parser.add_argument('--grasp-controller',choices=('legacy','coordinated','policy'),default='legacy',
                    help='opt-in measured-motion grasp governor; requires its own physics qualification')
parser.add_argument('--release-reposition',action='store_true',
                    help='release measured contact before downward reposition during close, then reclose')
parser.add_argument('--force-direction-reposition',action='store_true',
                    help='use signed vertical-force gated release, downward reposition, and reclose')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', type=Path)
parser.add_argument('--contact-effort-feedback',action='store_true',help='bounded contact torque feedback sharing the existing motor effort budget')
parser.add_argument('--teacher-search-cases',type=Path)
parser.add_argument('--manual-control',action='store_true',help='interactive force/finger/wrist/lift controls; no PPO training')
parser.add_argument('--teacher-episodes',type=int,default=1)
parser.add_argument('--teacher-control-steps',type=int,default=1000)
parser.add_argument('--actor-initialization',type=Path)
parser.add_argument('--student-validation-only',action='store_true')
parser.add_argument('--student-validation-steps',type=int,default=2700)
parser.add_argument('--control-probe-checkpoint',type=Path,
                    help='8-env inference-only control comparison; no PPO updates')
parser.add_argument('--control-probe-mode',choices=('grasp','grasp_fixed','grasp_z_sweep','grasp_teacher_z_sweep','grasp_teacher_z_fine','grasp_teacher_z_confirm','grasp_teacher_z_down_sweep','grasp_teacher_contact_z_sweep','grasp_teacher_xy_sweep','lift_gate','bounded_thumb','coordinated','wrist_motion','loaded_palm','force_direction'),default='grasp')
parser.add_argument('--control-probe-episodes',type=int,default=3)
parser.add_argument('--control-probe-swap',action='store_true',help='counterbalance paired loaded-palm candidate slots')
parser.add_argument('--play-checkpoint', type=Path,
                    help='single-environment GUI policy inference; no learning')
parser.add_argument('--allow-playback-material-change', action='store_true',
                    help='explicit object-friction-only inference comparison; never training resume')
parser.add_argument('--play-episodes', type=int, default=0,
                    help='playback episode count; zero repeats until the GUI closes')
parser.add_argument('--allow-env-expansion', action='store_true',
                    help='resume with strictly more environments, preserving policy and optimizer')
parser.add_argument('--allow-contact-reward-change', action='store_true',
                    help='explicitly resume v3 with the 10 N / 5 point contact reward only')
parser.add_argument('--resume-episode-progress', type=Path,
                    help='matching final monitor for an older checkpoint without saved episode counters')
parser.add_argument('--initialize-from', type=Path,
                    help='transfer policy weights with a fresh optimizer and step count')
parser.add_argument('--allow-physics-refinement', action='store_true',
                    help='allow a validated 240-to-480 Hz weights-only transfer')
add_mimic_arguments(parser)
add_protected_mimic_arguments(parser)
parser.add_argument('--probe-only', action='store_true')
parser.add_argument('--startup-only', action='store_true',help='exit after reset isolation and bounded startup checks')
parser.add_argument('--capacity-steps', type=int, default=0,
                    help='additional timed policy steps in startup-only mode; no optimization')
parser.add_argument('--training-episode-resets', action='store_true',
                    help='training-only isolated resets for bounded faults after independent guard validation')
parser.add_argument('--direct-training', action='store_true',
                    help='skip three full admission batches; retain startup contact and runtime safety checks')
parser.add_argument('--physics-hz', type=int, choices=(240, 480), default=240)
parser.add_argument('--solver-type', type=int, choices=(0,), default=0)
parser.add_argument('--solve-articulation-contact-last', action='store_true',
                    help='solve contact and joint velocity constraints last; changes physics checkpoint contract')
parser.add_argument('--position-iterations', type=int, default=16)
parser.add_argument('--velocity-iterations', type=int, choices=(4,16,64), default=4,
                    help='author contact velocity solver iterations before PhysX scene creation')
parser.add_argument('--passive-velocity-limit-rad-s',type=float,choices=(3.,2.97),default=3.,
                    help='opt-in 2.97 solver headroom candidate; measured velocity guard stays 3')
parser.add_argument('--grasp-readiness',choices=('measured_load',),default=None,
                    help='trial lift after stable measured hand loads, independent of stochastic force goals')
parser.add_argument('--skip-hold',action='store_true',
                    help='enter lift directly after 0.2 seconds of measured close readiness')
parser.add_argument('--close-downward-penalty',action='store_true',
                    help='add instantaneous -3 when close-to-lift force on object is downward')
parser.add_argument('--grasp-shaping',action='store_true',
                    help='add discounted force-direction potential and physical 3/10 mm milestones')
parser.add_argument('--palm-exploration-axis',choices=('z_down','xyz'),default='z_down',
                    help='training palm exploration: below-axis Z only, or legacy XYZ')
parser.add_argument('--contact-reward-window',choices=('first_contact','close_start'),default=None,
                    help='single one-second force reward window per episode, at most 5 points')
parser.add_argument('--force-reward-start',choices=('episode_start','after_close','downward_after_close'),default='episode_start',
                    help='after_close gates both signs; downward_after_close rewards upward force already during close')
parser.add_argument('--vertical-force-reward-mode',choices=('window','continuous'),default='window',
                    help='continuous removes positive reward time window and episode cap')
parser.add_argument('--vertical-force-reward',action='store_true',
                    help='reward upward object contact only; penalize downward signed force throughout the episode')
parser.add_argument('--reset-height-jitter-m', type=float, default=.0003)
parser.add_argument('--grasp-acquisition-shaping', action='store_true',
                    help='potential feedback for real fingertip approach and bilateral object contact')
parser.add_argument('--policy-lift-hold', action='store_true',
                    help='retain accepted grasp during policy-requested lift, with bounded grip recovery')
parser.add_argument('--stable-lift-reward', action='store_true',
                    help='gate actual height progress by stable bilateral grip and add bounded low-slip holding credit')
parser.add_argument('--force-region-control', action='store_true',
                    help='policy requests 2-12N per hand with measured feedback and below-axis contact admission')
parser.add_argument('--phase-force-reward',action='store_true',
                    help='separate upward hand-force rewards in grasp and lift; lift includes actual friction')
parser.add_argument('--free-grasp-lift',action='store_true',
                    help='continuous policy grasp and lift, without stages, admission gates or pose latching')
parser.add_argument('--continuous-motion',action='store_true',
                    help='finite mimic errors are diagnostic; isolate and validate recoverable invalid-state resets')
parser.add_argument('--manual-fast-force',action='store_true',help='manual-only bounded micro-closing contact acquisition and faster force adjustment')
parser.add_argument('--manual-run',action='store_true',help='start the explicitly requested manual demonstration running')
parser.add_argument('--manual-placement',type=Path,help='manual-only initial palm translation settings JSON')
parser.add_argument('--manual-auto-lift-mm',type=float,help='manual demonstration: lift once measured bilateral forces stabilize, not necessarily at target')
parser.add_argument('--require-initial-below-axis',action='store_true',
                    help='verify measured reset pad references below object axis with table clearance')
parser.add_argument('--max-iterations', type=int, default=None,
                    help='stop after this many completed PPO updates; takes priority over episode/timestep budgets')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.continuous_motion and (not args.tensor_pipeline or args.grasp_controller!='policy'
        or not args.training_episode_resets or args.palm_transport!='driven_carriage'):
    parser.error('--continuous-motion requires policy tensor pipeline, driven_carriage and training-episode-resets')
if (args.manual_fast_force or args.manual_run or args.manual_placement or args.manual_auto_lift_mm is not None) and not args.manual_control:
    parser.error('--manual-fast-force and --manual-run require --manual-control')
if args.manual_auto_lift_mm is not None and not (args.manual_fast_force and 0<args.manual_auto_lift_mm<=50):
    parser.error('--manual-auto-lift-mm requires manual-fast-force and a height in (0,50]mm')
if args.free_grasp_lift and (args.grasp_controller!='policy' or not args.force_region_control
        or not args.phase_force_reward or args.policy_lift_hold or args.manual_control
        or args.teacher_search_cases or args.control_probe_checkpoint):
    parser.error('--free-grasp-lift requires policy force-region phase-force rewards and no staged hold/manual/teacher/probe')
if args.phase_force_reward and (not args.force_region_control or not args.grasp_shaping):
    parser.error('--phase-force-reward requires force-region control and grasp-shaping')
if args.manual_control and (args.headless or args.num_envs!=1 or not args.force_region_control
        or args.resume or args.teacher_search_cases or args.actor_initialization):
    parser.error('--manual-control requires GUI, one env, force-region control and no checkpoint/teacher mode')
if args.force_region_control and (args.grasp_controller!='policy' or not args.tensor_pipeline
        or not args.vertical_force_reward or args.grasp_readiness!='measured_load'
        or not (args.policy_lift_hold or getattr(args,'free_grasp_lift',False))):
    parser.error('--force-region-control requires policy tensor measured_load vertical force rewards and --policy-lift-hold')
if args.max_iterations is not None and (args.max_iterations<=0 or not args.tensor_pipeline):
    parser.error('--max-iterations requires tensor pipeline and a positive update count')
if args.stable_lift_reward and (not args.tensor_pipeline or args.grasp_readiness!='measured_load'
        or not args.vertical_force_reward):
    parser.error('--stable-lift-reward requires tensor measured_load and vertical force reward')
if args.policy_lift_hold and (not args.tensor_pipeline or args.grasp_controller!='policy'):
    parser.error('--policy-lift-hold requires policy tensor control')
if args.grasp_acquisition_shaping and (not args.tensor_pipeline or args.grasp_controller!='policy'):
    parser.error('--grasp-acquisition-shaping requires policy tensor training')
if args.grasp_controller=='policy':
    from go2_pvcnn.control.dh116.tabletop_policy_control import validate_configuration
    try:
        validate_configuration(args)
    except ValueError as exc:
        parser.error(str(exc))
if args.contact_effort_feedback and (not args.tensor_pipeline or args.grasp_controller!='coordinated'):
    parser.error('contact effort feedback requires coordinated tensor control')
if args.student_validation_only and not args.actor_initialization:
    parser.error('student validation requires an initialized actor')
if not 1<=args.student_validation_steps<=2700:
    parser.error('student validation budget must be between 1 and 2700')
if args.teacher_search_cases and (not args.tensor_pipeline or not args.training_episode_resets or args.resume or args.initialize_from or args.control_probe_checkpoint or args.play_checkpoint or args.actor_initialization):
    parser.error('teacher search requires the production tensor task with ordinary episode resets and no checkpoint mode')
if args.teacher_episodes<1 or args.teacher_control_steps<1:
    parser.error('teacher episode and control-step budgets must be positive')
if args.actor_initialization and (not args.tensor_pipeline or args.resume or args.initialize_from or args.control_probe_checkpoint or args.play_checkpoint):
    parser.error('actor initialization requires a fresh tensor PPO run')
if args.control_probe_swap and (not args.control_probe_checkpoint or args.control_probe_mode not in ('loaded_palm','force_direction')):
    parser.error('probe swap is only valid for paired palm reposition probes')
if args.release_reposition and (args.grasp_controller!='coordinated' or args.control_probe_checkpoint):
    parser.error('release-reposition requires coordinated tensor training; probes select candidate by mode')
if args.force_direction_reposition and (args.release_reposition or args.grasp_controller!='coordinated'
        or (args.control_probe_checkpoint and args.control_probe_mode!='force_direction')
        or args.resume or args.initialize_from):
    parser.error('force-direction-reposition requires a fresh coordinated tensor run or its probe mode')
if args.palm_transport=='driven_carriage' and not args.control_probe_checkpoint and (
        not args.tensor_pipeline or args.grasp_controller not in ('coordinated','policy')):
    parser.error('driven wrist carriage requires bounded probe or coordinated tensor training')
if args.grasp_controller=='coordinated' and (not args.tensor_pipeline or not args.pressure_relief
        or not args.skip_hold or args.grasp_readiness!='measured_load' or args.reset_height_jitter_m!=0
        or (args.control_probe_checkpoint and args.control_probe_mode not in ('loaded_palm','force_direction','grasp_teacher_contact_z_sweep')) or args.play_checkpoint):
    parser.error('coordinated controller requires tensor measured-load direct lift, pressure relief and zero reset jitter; probes select it through probe mode')
if args.force_direction_reposition and args.palm_transport!='driven_carriage':
    parser.error('force-direction-reposition requires driven carriage palm transport')
if args.grasp_shaping and not args.close_downward_penalty:
    parser.error('--grasp-shaping requires --close-downward-penalty configuration')
if args.close_downward_penalty and (not args.skip_hold or not args.vertical_force_reward
        or args.vertical_force_reward_mode!='continuous' or args.force_reward_start!='downward_after_close'):
    parser.error('--close-downward-penalty requires --skip-hold and continuous downward_after_close force rewards')
if args.skip_hold and (not args.tensor_pipeline or args.grasp_readiness!='measured_load'):
    parser.error('--skip-hold requires --tensor-pipeline and --grasp-readiness measured_load')
if args.force_reward_start!='episode_start' and (not args.vertical_force_reward or args.vertical_force_reward_mode!='continuous'):
    parser.error('phase-gated force rewards require continuous vertical force reward')
if args.vertical_force_reward_mode=='continuous' and not args.vertical_force_reward:
    parser.error('--vertical-force-reward-mode continuous requires --vertical-force-reward')
if args.vertical_force_reward and (not args.tensor_pipeline or args.contact_reward_window!='close_start'):
    parser.error('--vertical-force-reward requires --tensor-pipeline and --contact-reward-window close_start')
if args.control_probe_episodes<1 or args.control_probe_episodes>3:
    parser.error('control probe episodes must be 1..3')
if not args.control_probe_checkpoint and (args.control_probe_mode!='grasp' or args.control_probe_episodes!=3):
    parser.error('probe settings require --control-probe-checkpoint')
probe_envs = (8, 16) if args.control_probe_mode == 'force_direction' else (6,) if args.control_probe_mode in ('grasp_fixed','grasp_z_sweep','grasp_teacher_z_sweep','grasp_teacher_z_fine','grasp_teacher_z_confirm','grasp_teacher_z_down_sweep','grasp_teacher_contact_z_sweep','grasp_teacher_xy_sweep') else (8,)
if args.control_probe_checkpoint and (not args.tensor_pipeline or args.num_envs not in probe_envs
        or not args.grasp_shaping or args.resume or args.initialize_from or args.play_checkpoint
        or args.startup_only or args.probe_only or args.episodes_per_env is not None):
    parser.error(f'control probe requires {probe_envs}-env shaped tensor inference, without training/resume/play modes')
if (args.control_probe_checkpoint and args.palm_exploration_axis!='xyz'
        and args.control_probe_mode!='grasp_teacher_contact_z_sweep'):
    parser.error('control probes use the checkpoint legacy XYZ palm contract; set --palm-exploration-axis xyz')
if args.cpu_threads < 1:
    parser.error('--cpu-threads must be positive')
if args.pressure_relief and not args.tensor_pipeline:
    parser.error('--pressure-relief requires --tensor-pipeline')
if args.passive_velocity_limit_rad_s!=3. and not args.tensor_pipeline:
    parser.error('passive velocity headroom candidate requires --tensor-pipeline')
if args.grasp_readiness is not None and not args.tensor_pipeline:
    parser.error('--grasp-readiness requires --tensor-pipeline')
if args.contact_reward_window is not None and not args.tensor_pipeline:
    parser.error('--contact-reward-window requires --tensor-pipeline')
if args.rollout_steps is not None and args.rollout_steps < 2:
    parser.error('--rollout-steps must be at least 2')
if args.rollout_steps is not None and args.resume and not args.tensor_pipeline:
    parser.error('--resume retains checkpoint rollout length; omit --rollout-steps')
if args.training_episode_resets and (not args.full_hand_contact or args.reset_height_jitter_m != 0):
    parser.error('--training-episode-resets requires full-hand contact and zero reset jitter')
if args.tensor_pipeline and (not args.full_hand_contact or not args.force_feedback
        or args.reset_height_jitter_m != 0
        or args.initialize_from or args.play_checkpoint or args.probe_only or args.startup_only
        or args.allow_env_expansion or args.resume_episode_progress):
    parser.error('--tensor-pipeline requires rigid full-hand force feedback, zero jitter, and training mode; legacy transfer is unsupported')
if args.full_hand_contact and (not args.force_feedback):
    parser.error('full-hand contact requires rigid-object force feedback')
if args.joint_envelope and not args.full_hand_contact:
    parser.error('joint-envelope requires full-hand contact')
args.joint_envelope_data = None
if args.joint_envelope:
    from go2_pvcnn.control.dh116.tabletop_joint_envelope import bounds
    args.joint_envelope_data = json.loads(args.joint_envelope.read_text())
    bounds(args.joint_envelope_data)

if args.allow_playback_material_change and not args.play_checkpoint:
    parser.error('--allow-playback-material-change requires --play-checkpoint')
if args.play_episodes < 0 or (args.play_episodes and not args.play_checkpoint):
    parser.error('--play-episodes must be nonnegative and requires --play-checkpoint')
if args.play_checkpoint and (args.headless or args.num_envs != 1 or args.resume
        or args.initialize_from or args.startup_only or args.probe_only
        or args.episodes_per_env is not None or args.direct_training):
    parser.error('playback requires one GUI environment and excludes training/resume/probe modes')
if args.allow_env_expansion and (not args.resume or args.allow_contact_reward_change):
    parser.error('--allow-env-expansion requires --resume and unchanged reward')
if args.capacity_steps < 0 or (args.capacity_steps and not args.startup_only):
    parser.error('--capacity-steps must be nonnegative and requires --startup-only')
if args.direct_training and args.probe_only:
    parser.error('--direct-training and --probe-only are mutually exclusive')
if args.episodes_per_env is not None and (args.episodes_per_env < 1 or args.probe_only):
    parser.error('--episodes-per-env must be positive and cannot use --probe-only')
if args.resume_episode_progress and not args.resume:
    parser.error('--resume-episode-progress requires --resume')
if args.allow_contact_reward_change and not args.resume:
    parser.error('--allow-contact-reward-change requires --resume')
from go2_pvcnn.control.dh116.tabletop_load_parameters import read_load_parameters
try:
    args.load_parameters = read_load_parameters(args.load_profile)
    if args.load_parameters is not None and not args.force_feedback:
        raise ValueError('load profile requires force feedback')
except (ValueError, TypeError, KeyError, OSError) as error:
    parser.error(str(error))
if not args.force_feedback or not args.adaptive_grasp or args.protected_mimic_resets:
    parser.error('GPU batch requires adaptive force feedback; protected resets are unsupported')
if args.force_feedback and not args.adaptive_grasp:
    parser.error('force feedback requires --adaptive-grasp')
if (args.output.exists() or not 1 <= args.num_envs <= 10240 or args.env_spacing_m < 2.
        or args.timesteps <= 0 or not 0 <= args.reset_height_jitter_m <= .002
        or args.eval_episodes < 2 or args.checkpoint_every <= 0 or args.device != 'cuda:0'):
    parser.error('new output,1..10240 envs,spacing>=2,positive budget,CUDA and bounded jitter required')
if args.resume and args.initialize_from:
    parser.error('resume and initialize-from are mutually exclusive')
if args.allow_physics_refinement and (not args.initialize_from or args.resume):
    parser.error('physics refinement requires initialize-from, not resume')
try:
    validate_adaptive_options(adaptive_grasp=args.adaptive_grasp,
        grasp_initialization=args.grasp_initialization,close_reset_snapshot=args.close_reset_snapshot,
        # Same-schema weights are checked against the full contract below.
        initialize_from=None,physics_hz=args.physics_hz)
    validate_mimic_options(physics_hz=args.physics_hz, mimic_frequency=args.mimic_frequency,
        allow_rigid=args.load_parameters is not None,
        initialize_from=args.initialize_from, resume=args.resume,
        allow_mimic_refinement=args.allow_mimic_refinement,
        allow_physics_refinement=args.allow_physics_refinement)
    validate_protected_mimic_options(physics_hz=args.physics_hz, mimic_frequency=args.mimic_frequency,
        protected_mimic_resets=args.protected_mimic_resets,
        allow_protected_mimic_init=args.allow_protected_mimic_init,
        initialize_from=args.initialize_from, resume=args.resume,
        allow_physics_refinement=args.allow_physics_refinement,
        allow_mimic_refinement=args.allow_mimic_refinement)
except ValueError as error:
    parser.error(str(error))
args.control_mode = 'adaptive_grasp' if args.adaptive_grasp else 'joint_target'
try:
    args.object_usd, args.object_mass_kg, object_contract = object_spec(
        Path(__file__).resolve().parents[1], args.object_kind,
        solid_radius_m=args.solid_cylinder_radius_m,
        solid_density_kg_m3=args.solid_cylinder_density_kg_m3)
except (ValueError, OSError, KeyError) as error:
    parser.error(str(error))
args.output.mkdir(parents=True)
write_json(args.output/'monitor.json',dict(status='initializing',num_envs=args.num_envs,
    timesteps=0,updates=0,object_kind=args.object_kind,
    episodes_per_env_target=None if args.max_iterations is not None else args.episodes_per_env,
    iteration_budget=args.max_iterations))
app = None
report = dict(passed=False, grasp_qualified=False, num_envs=args.num_envs,
              status='initializing', rigid_proxy=True, pipeline_scope='close_to_lift',
              continuous_motion=args.continuous_motion)
vec = None
from go2_pvcnn.control.dh116.tabletop_run_status import finalize_run
try:
    app = AppLauncher(args).app
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from go2_pvcnn.control.dh116.hand_only_scene import native_pose_hand_only_layout
    from go2_pvcnn.control.dh116.tabletop_direct import snapshot_state
    from go2_pvcnn.control.dh116.tabletop_vec_config import validate_seed_provenance
    from go2_pvcnn.control.dh116.tabletop_env import TabletopEnv
    from go2_pvcnn.control.dh116.tabletop_vec_env import SharedPhysicsVecEnv
    from go2_pvcnn.control.dh116.tabletop_vec_isaac import spawn_scene, bind_scene, advance_world, isolate_scenes
    from go2_pvcnn.control.dh116.tabletop_vec_training import run_vector_training, collect_vector_episodes
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(args.cpu_threads)
    report['cpu_runtime'] = dict(torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        omp_threads=os.environ.get('OMP_NUM_THREADS'),
        openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    layout = native_pose_hand_only_layout(json.loads(args.pose_summary.read_text()),roll_degrees=dict(left=76.,right=82.))
    manifest = json.loads((root/'assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
    sources = {x['side']:x['path'] for x in manifest['sources'] if x['kind']=='dh116'}
    provenance = dict(pose_sha256=sha(args.pose_summary),source_sha256={s:sha(p) for s,p in sources.items()},
        physics_hz=args.physics_hz,control_hz=60,solver_type=args.solver_type,position_iterations=args.position_iterations)
    if args.adaptive_grasp:
        layout, seed_state = adaptive_initialization(
            json.loads(args.grasp_initialization.read_text()),layout,provenance,
            object_radius_m=args.solid_cylinder_radius_m)
        seed_provenance = {}
    else:
        snapshot = json.loads(args.close_reset_snapshot.read_text())
        seed_provenance = validate_seed_provenance(snapshot['source_contract'], provenance)
        seed_state = snapshot_state(snapshot['row'],layout)
    contract = dict(provenance, observation_version='dh116-direct-close-v1',observation_size=108,
        action='12D_direct_joint_target_v1',target_radius_rad=.08,command_rate_rad_s=.05,
        nominal='fixed_target_center_no_scripted_closure',nominal_rate_rad_s=0.,
        reward='object_support_progress_v4_10n_5',grasp_mode='object_stability',
        coupling_pause_rad=.004,coupling_warning='follower_consistent_recovery_v2',
        active_velocity_limit_rad_s=.3,passive_velocity_limit_rad_s=args.passive_velocity_limit_rad_s,
        mimic_natural_frequency_s_inv=float(args.mimic_frequency),mimic_damping_ratio=1.,
        lift_attempt_after_s=2.3,success_bottom_m=.33,success_dwell_s=.5,success_slip_m=.02,
        reset_phase='close',close_seed_sha256=sha(args.grasp_initialization if args.adaptive_grasp
                                               else args.close_reset_snapshot),reset_settle_steps=0,
        reset_height_jitter_m=args.reset_height_jitter_m,
        palm_motion='bounded_per_hand_close_adjustment_v1',palm_xy_limit_m=.003,palm_z_limit_m=.002,palm_z_rate_m_s=.001,
        policy='bounded_gaussian_mean_v1',force_observation='log1p_force_over_01N',
        num_envs=args.num_envs,env_spacing_m=args.env_spacing_m,vector_backend='cloned_gpu_normalized_convex_batched_task_v2')
    contract.update(seed_provenance)
    if args.velocity_iterations != 4:
        contract['velocity_iterations'] = args.velocity_iterations
    if args.solve_articulation_contact_last:
        from go2_pvcnn.control.dh116.tabletop_solver_order import CONTRACT as SOLVER_ORDER
        contract['articulation_solver_order'] = dict(SOLVER_ORDER)
    if args.passive_velocity_limit_rad_s==2.97:
        from go2_pvcnn.control.dh116.tabletop_velocity_limits import VELOCITY_HEADROOM_CONTRACT
        contract['velocity_headroom']=VELOCITY_HEADROOM_CONTRACT
    contract.update(hand_material_contract(args.hand_material))
    contract.update(object_contract)
    contract.update(material_contract(args.object_material))
    if args.adaptive_grasp:
        for key in ('palm_xy_limit_m','palm_z_limit_m','palm_z_rate_m_s'):
            contract.pop(key)
        contract.update(adaptive_contract_fields(sha(args.grasp_initialization)))
    force_direction = None
    force_palm_compensation = None
    if args.force_feedback:
        from go2_pvcnn.control.dh116.tabletop_force_feedback import closure_direction, force_feedback_contract
        payload = json.loads(args.grasp_initialization.read_text())
        force_direction = closure_direction([payload['diagnostics']['hands'][side]['closure_direction_active_unit']
                                             for side in ('left', 'right')])
        contract.update(force_feedback_contract(args.load_parameters))
        from go2_pvcnn.control.dh116.tabletop_pressure_geometry import seed_pressure_compensation
        force_palm_compensation = seed_pressure_compensation(payload)
        if force_palm_compensation is not None:
            contract['pressure_geometry'] = payload['pressure_geometry']
    if args.training_episode_resets:
        from go2_pvcnn.control.dh116.tabletop_gpu_recovery import VERSION
        contract['training_failure_handling'] = VERSION
    if args.protected_mimic_resets:
        contract['training_failure_handling'] = 'validated_mimic_reset_v1'
    if args.joint_envelope_data is not None:
        contract.pop('target_radius_rad', None)
        contract.update(joint_envelope=args.joint_envelope_data,
            joint_envelope_sha256=sha(args.joint_envelope),
            action='20D_piecewise_joint_envelope_palm_load_v1')
    contract = canonical_contract(contract)
    report['contract'] = contract
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/args.physics_hz,device=args.device,
        physx=sim_utils.PhysxCfg(solver_type=args.solver_type,
            solve_articulation_contact_last=args.solve_articulation_contact_last)))
    from go2_pvcnn.control.dh116.tabletop_solver_order import verify_solver_order
    report['articulation_solver_order_readback'] = verify_solver_order(
        sim.get_physics_context(), args.solve_articulation_contact_last)
    from go2_pvcnn.control.dh116.tabletop_gpu_isaac import GpuWorld
    world = GpuWorld(sim,args,layout,sources)
    if args.full_hand_contact:
        from go2_pvcnn.control.dh116.tabletop_object_contact import object_contact_contract
        contact_contract = object_contact_contract(world.object_contact_links)
        contract.update(object_contact=contact_contract,
            observation_version=contact_contract['observation_version'],
            observation_size=contact_contract['observation_size'],
            reward='full_hand_object_support_progress_v1_10n_5',
            force_feedback_measurement='sum_of_object_filtered_link_loads_per_hand',
            full_hand_contact=True)
        contract['force_feedback']['measurement'] = 'sum_of_object_filtered_link_loads_per_hand'
        if args.joint_envelope_data is not None:
            contract['force_feedback'].pop('combined_joint_radius_rad', None)
            contract['force_feedback']['combined_joint_bounds'] = 'joint_envelope'
        contract = canonical_contract(contract)
        report['contract'] = contract
    if args.initialize_from:
        metadata = json.loads(args.initialize_from.with_suffix('.json').read_text())
        layout_keys = {'num_envs','env_spacing_m','vector_backend'}
        comparable = lambda c: {k:v for k,v in c.items() if k not in layout_keys}
        if comparable(metadata['contract']) != comparable(contract):
            raise ValueError('GPU initialization requires identical observation/action/reward/physics contract')
    if not args.headless:
        light=sim_utils.DomeLightCfg(intensity=1800.)
        light.func('/World/DemoLight',light)
        sim.set_camera_view(eye=np.array((1.1,-1.3,1.05))+world.origins[0],
                            target=np.array((.43,0.,.32))+world.origins[0])
    sim.reset()
    signal.signal(signal.SIGINT,signal.default_int_handler)
    backends = world.bind(seed_state)
    for backend in backends:
        backend.joint_envelope = args.joint_envelope_data
    counts = [0]*args.num_envs
    def sink(index):
        directory=args.output/f'env_{index:03d}'
        directory.mkdir()
        def save(summary,rows):
            number=counts[index];counts[index]+=1
            if index < 8:
                write_json(directory/f'episode_{number:05d}.json',rows)
            write_json(directory/'latest_episode.json',dict(summary,env_index=index,episode=number))
        return save
    envs=[TabletopEnv(b,stage='full',reset_phase='close',control_mode=args.control_mode,
        grasp_mode='object_stability',initial_offset=seed_state['root_offset'],reset_jitter=args.reset_height_jitter_m,
        episode_sink=sink(i),force_direction=force_direction,trace_steps=i<8,force_palm_compensation=force_palm_compensation,load_parameters=args.load_parameters) for i,b in enumerate(backends)]
    from go2_pvcnn.control.dh116.tabletop_gpu_vec_env import GpuTaskVecEnv
    vec = GpuTaskVecEnv(envs, world, recover_training_failures=args.training_episode_resets)
    # Actual isolation probe: a selected reset must not tick physics or change
    # another slot's physical state/clock; repeated local resets must agree.
    reset_started=time.monotonic()
    vec.seed(args.seed);first=vec.reset()
    report['first_reset_elapsed_seconds']=time.monotonic()-reset_started
    other=[b.measure() for b in backends[1:]]
    envs[0].reset(seed=args.seed)
    world.refresh(sensors=False)
    after=[b.measure() for b in backends[1:]]
    independent=all(np.array_equal(a['q'],b['q']) and np.array_equal(a['object_relative'],b['object_relative'])
                    for a,b in zip(other,after))
    vec.seed(args.seed);second=vec.reset()
    repeatable=bool(np.allclose(first,second,atol=1e-4,rtol=0))
    report['isolation']=dict(reset_does_not_change_other_envs=independent,reset_repeatable=repeatable,
        origins=world.origins.tolist(), sensor_filters=('Wire-only, GPU environment-ID isolation'),
        collision_groups=args.num_envs)
    if not independent or not repeatable:
        raise ValueError('physical per-environment reset isolation failed')
    if not False and args.object_material != 'default':
        properties=world.wire.root_physx_view.get_material_properties().cpu().numpy()
        report['object_material_verification']=[verify_material_properties(properties[i:i+1],
            args.object_material,len(world.source['object_material_colliders'])) for i in range(args.num_envs)]
    if args.hand_material != 'default':
        report['hand_material_verification'] = {
            side: verify_hand_material_properties(hand.root_physx_view.get_material_properties().cpu().numpy(),
                args.hand_material,args.num_envs,len(world.source['hand_material_colliders'][side]))
            for side,hand in zip(('left','right'),world.hands)}
        report['hand_material_bindings'] = world.source['hand_material_bindings']
        if args.hand_material=='grip_compliant':
            if not args.full_hand_contact:
                raise ValueError('compliant pads require rigid full-hand contact sensors')
            from go2_pvcnn.control.dh116.tabletop_material import verify_hand_link_material
            report['hand_material_link_verification']=[verify_hand_link_material(
                sensor.body_physx_view.get_material_properties().cpu().numpy(),args.hand_material,link,args.num_envs)
                for sensor,link in zip(world.sensors,list(world.sensor_links)*2)]
    write_json(args.output/'material_verification.json',dict(
        object_material=report.get('object_material_verification'),
        hands=report.get('hand_material_verification'),hand_bindings=report.get('hand_material_bindings'),
        hand_links=report.get('hand_material_link_verification'),
        contract={k:contract[k] for k in ('object_contact_material','hand_contact_material') if k in contract}))
    report['initial_physics'] = [dict(object_mass_kg=world.wire.root_physx_view.get_masses().cpu().tolist(),
        object_inertia_kg_m2=world.wire.root_physx_view.get_inertias().cpu().tolist(),
        pad_relative_m=world.pad_relative.tolist(),hand_clearance_m=world.clearances.tolist())]
    if args.num_envs >= 12 and not False:
        # Asymmetric physical state catches env_2/env_10 ordering mistakes that
        # identical resets cannot detect. Do not tick physics in this audit.
        original = world.wire.data.root_pose_w.clone()
        shifted = original.clone()
        shifted[:,2] += torch.arange(args.num_envs,device=args.device)*1e-5
        world.wire.write_root_pose_to_sim(shifted)
        sim.physics_sim_view.update_articulations_kinematic()
        world.refresh(sensors=False)
        if not np.allclose(world.wp,shifted[:,:3].cpu().numpy(),atol=1e-6,rtol=0.):
            raise ValueError('asymmetric GPU object mapping failed')
        before = world.wp.copy()
        selected = args.num_envs//2
        envs[selected].reset(seed=args.seed+selected)
        world.refresh(sensors=False)
        others = np.arange(args.num_envs)!=selected
        if not np.array_equal(world.wp[others],before[others]):
            raise ValueError('selected GPU reset changed other object states')
        report['isolation']['asymmetric_object_mapping'] = True
        vec.seed(args.seed);vec.reset()
    from go2_pvcnn.control.dh116.tabletop_preflight import contact_preflight
    report['contact_preflight'] = contact_preflight(vec,world,direct=bool(args.direct_training or args.play_checkpoint),
        progress=lambda row: write_json(args.output/'monitor.json',row))
    write_json(args.output/'contact_preflight.json',report['contact_preflight'])
    from go2_pvcnn.control.dh116.tabletop_load_parameters import physics_load_readback
    report['physical_parameter_readback'] = physics_load_readback(world.hands,args.load_parameters,
        passive_velocity_limit_rad_s=args.passive_velocity_limit_rad_s)
    if args.tensor_pipeline and args.full_hand_contact:
        from go2_pvcnn.control.dh116.tabletop_tensor_training import (
            tensor_contract_metadata, apply_continuous_motion_contract)
        contract = tensor_contract_metadata(contract, args.load_parameters.max_force_n)
        apply_continuous_motion_contract(contract,args.continuous_motion)
        report['contract'] = contract
    write_json(args.output/'configuration.json',report)
    if args.play_checkpoint:
        from go2_pvcnn.control.dh116.tabletop_playback import play_policy
        report.update(play_policy(vec,app,args.play_checkpoint,contract,args.output,
            device=args.device,seed=args.seed,episodes=args.play_episodes,
            allow_material_change=args.allow_playback_material_change))
    elif args.startup_only:
        if args.capacity_steps:
            from go2_pvcnn.control.dh116.tabletop_training import load_checkpoint
            probe_model=None
            if args.resume:
                probe_model,metadata=load_checkpoint(args.resume,vec,contract,device=args.device,
                    allow_env_expansion=args.allow_env_expansion)
                report['capacity_resume']=dict(timesteps=probe_model.num_timesteps,
                    updates=probe_model._n_updates,n_envs=probe_model.n_envs,
                    rollout_buffer_envs=probe_model.rollout_buffer.n_envs,
                    optimizer_state_entries=len(probe_model.policy.optimizer.state),
                    env_expansion=metadata.get('env_expansion'))
            obs = vec.reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            step_times=[]
            memory_samples=[]
            timing_samples=[]
            for tick in range(args.capacity_steps):
                free,total=torch.cuda.mem_get_info()
                memory_samples.append(dict(free_bytes=free,total_bytes=total,used_bytes=total-free))
                if free < 1536*1024**2:
                    raise RuntimeError('capacity probe has less than 1.5 GiB free GPU memory')
                started=time.monotonic()
                action=(probe_model.predict(obs,deterministic=False)[0] if probe_model is not None
                        else np.zeros((args.num_envs,20),dtype=np.float32))
                obs,rewards,dones,infos=vec.step(action)
                torch.cuda.synchronize()
                step_times.append(time.monotonic()-started)
                timing_samples.append(dict(getattr(world,'last_step_timing',{})))
                if (not np.isfinite(obs).all() or not np.isfinite(rewards).all()
                        or any(info.get('safety_failure') for info in infos)):
                    raise ValueError('capacity probe numerical/hand safety failure')
                write_json(args.output/'monitor.json',dict(status='capacity_probe',num_envs=args.num_envs,
                    measured_vector_steps=tick+1,step_seconds=step_times,
                    transitions_per_second=args.num_envs*len(step_times)/sum(step_times),
                    gpu_memory_samples=memory_samples))
            free,total=torch.cuda.mem_get_info()
            memory_samples.append(dict(free_bytes=free,total_bytes=total,used_bytes=total-free))
            if free < 1536*1024**2:
                raise RuntimeError('capacity probe ended with less than 1.5 GiB free GPU memory')
            report['capacity']=dict(num_envs=args.num_envs,vector_steps=args.capacity_steps,
                step_seconds=step_times,transitions_per_second=args.num_envs*len(step_times)/sum(step_times),
                gpu_memory_samples=memory_samples,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),last_physics_step_timings=timing_samples,
                minimum_free_bytes=min(x['free_bytes'] for x in memory_samples),
                scope='short startup physics plus policy inference; excludes training callback and optimization',
                reset_elapsed_seconds=report['first_reset_elapsed_seconds'])
            write_json(args.output/'capacity.json',report['capacity'])
        report.update(status='startup_validated',startup_validated=True)
    elif args.tensor_pipeline:
        from go2_pvcnn.control.dh116.tabletop_tensor_training import run_tensor_training
        vec.reset()
        report.update(run_tensor_training(world,envs[0],args,contract))
    else:
        training,model=run_vector_training(vec,args.output,contract,timesteps=args.timesteps,
            eval_episodes=args.eval_episodes,checkpoint_every=args.checkpoint_every,seed=args.seed,
            resume=args.resume,initialize_from=args.initialize_from,
            allow_physics_refinement=args.allow_physics_refinement,
            allow_mimic_refinement=args.allow_mimic_refinement,
            allow_protected_mimic_init=args.allow_protected_mimic_init,probe_only=args.probe_only,
            direct_training=args.direct_training,
            episodes_per_env=args.episodes_per_env,resume_episode_progress=args.resume_episode_progress,
            allow_contact_reward_change=args.allow_contact_reward_change,
            allow_env_expansion=args.allow_env_expansion,
            ppo_kwargs=dict(device=args.device,n_steps=args.rollout_steps or 64,
                            batch_size=min(2048,(args.rollout_steps or 64)*args.num_envs)))
        report.update(training)

except KeyboardInterrupt:
    report.update(status='interrupted',interrupted=True)
except (Exception,SystemExit):
    report.update(status='error',error=traceback.format_exc())
    print(report['error'],flush=True)
finally:
    # Isaac fast shutdown may terminate the process without returning. Publish
    # after fallible episode flushing but before the final application shutdown.
    code=2
    try:
        code=finalize_run(args.output,report,[vec.close] if vec is not None else [])
        print('VECTOR_PPO_RESULT',json.dumps(report),flush=True)
    finally:
        if app is not None:
            try:
                import omni.kit.app
                omni.kit.app.get_app().post_quit(code)
                app.close(wait_for_replicator=False)
            except Exception:
                report.update(status='error',error=report.get('error','')+'\n'+traceback.format_exc())
                code=finalize_run(args.output,report,[])
raise SystemExit(code)
