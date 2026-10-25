"""Multiple independent close-to-lift scenes sharing a single Isaac clock."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import traceback
from go2_pvcnn.control.dh116.tabletop_training import write_json
from go2_pvcnn.control.dh116.tabletop_vec_config import (add_mimic_arguments,
    validate_mimic_options, add_protected_mimic_arguments, validate_protected_mimic_options)
from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import (
    validate_adaptive_options, adaptive_initialization, adaptive_contract_fields, object_spec)
from go2_pvcnn.control.dh116.tabletop_material import material_contract, verify_material_properties
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
parser.add_argument('--load-profile', type=Path, help='explicit estimated force/drive parameters; strict checkpoint provenance')
parser.add_argument('--object-kind', choices=('solid_cylinder','hollow_straw'),default='solid_cylinder')
parser.add_argument('--object-material', choices=('default','high_friction'),default='default')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=4)
parser.add_argument('--env-spacing-m', type=float, default=2.)
parser.add_argument('--timesteps', type=int, default=65536)
parser.add_argument('--eval-episodes', type=int, default=8)
parser.add_argument('--checkpoint-every', type=int, default=2048)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', type=Path)
parser.add_argument('--initialize-from', type=Path,
                    help='transfer policy weights with a fresh optimizer and step count')
parser.add_argument('--allow-physics-refinement', action='store_true',
                    help='allow a validated 240-to-480 Hz weights-only transfer')
add_mimic_arguments(parser)
add_protected_mimic_arguments(parser)
parser.add_argument('--probe-only', action='store_true')
parser.add_argument('--physics-hz', type=int, choices=(240, 480), default=240)
parser.add_argument('--solver-type', type=int, choices=(0,), default=0)
parser.add_argument('--position-iterations', type=int, default=16)
parser.add_argument('--reset-height-jitter-m', type=float, default=.0003)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from go2_pvcnn.control.dh116.tabletop_load_parameters import read_load_parameters
try:
    args.load_parameters = read_load_parameters(args.load_profile)
    if args.load_parameters is not None and not args.force_feedback:
        raise ValueError('load profile requires force feedback')
except (ValueError, TypeError, KeyError, OSError) as error:
    parser.error(str(error))
if args.force_feedback and not args.adaptive_grasp:
    parser.error('force feedback requires --adaptive-grasp')
if (args.output.exists() or not 1 <= args.num_envs <= 16 or args.env_spacing_m < 2.
        or args.timesteps <= 0 or not 0 <= args.reset_height_jitter_m <= .002
        or args.eval_episodes < 2 or args.checkpoint_every <= 0 or args.device != 'cpu'):
    parser.error('new output,1..16 envs,spacing>=2,positive budget,CPU and bounded jitter required')
if args.resume and args.initialize_from:
    parser.error('resume and initialize-from are mutually exclusive')
if args.allow_physics_refinement and (not args.initialize_from or args.resume):
    parser.error('physics refinement requires initialize-from, not resume')
try:
    validate_adaptive_options(adaptive_grasp=args.adaptive_grasp,
        grasp_initialization=args.grasp_initialization,close_reset_snapshot=args.close_reset_snapshot,
        initialize_from=args.initialize_from,physics_hz=args.physics_hz)
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
        Path(__file__).resolve().parents[1], args.object_kind)
except (ValueError, OSError, KeyError) as error:
    parser.error(str(error))
args.output.mkdir(parents=True)
app = AppLauncher(args).app
report = dict(passed=False, grasp_qualified=False, num_envs=args.num_envs,
              status='initializing', rigid_proxy=True, pipeline_scope='close_to_lift')
vec = None
try:
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
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    layout = native_pose_hand_only_layout(json.loads(args.pose_summary.read_text()),roll_degrees=dict(left=76.,right=82.))
    manifest = json.loads((root/'assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
    sources = {x['side']:x['path'] for x in manifest['sources'] if x['kind']=='dh116'}
    provenance = dict(pose_sha256=sha(args.pose_summary),source_sha256={s:sha(p) for s,p in sources.items()},
        physics_hz=args.physics_hz,control_hz=60,solver_type=args.solver_type,position_iterations=args.position_iterations)
    if args.adaptive_grasp:
        layout, seed_state = adaptive_initialization(
            json.loads(args.grasp_initialization.read_text()),layout,provenance)
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
        active_velocity_limit_rad_s=.3,passive_velocity_limit_rad_s=3.,
        mimic_natural_frequency_s_inv=float(args.mimic_frequency),mimic_damping_ratio=1.,
        lift_attempt_after_s=2.3,success_bottom_m=.33,success_dwell_s=.5,success_slip_m=.02,
        reset_phase='close',close_seed_sha256=sha(args.grasp_initialization if args.adaptive_grasp
                                               else args.close_reset_snapshot),reset_settle_steps=0,
        reset_height_jitter_m=args.reset_height_jitter_m,
        palm_motion='bounded_per_hand_close_adjustment_v1',palm_xy_limit_m=.003,palm_z_limit_m=.002,palm_z_rate_m_s=.001,
        policy='bounded_gaussian_mean_v1',force_observation='log1p_force_over_01N',
        num_envs=args.num_envs,env_spacing_m=args.env_spacing_m,vector_backend='shared_physics_cpu_v1')
    contract.update(seed_provenance)
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
    if args.protected_mimic_resets:
        contract['training_failure_handling'] = 'validated_mimic_reset_v1'
    report['contract'] = contract
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/args.physics_hz,device='cpu',
        physx=sim_utils.PhysxCfg(solver_type=args.solver_type)))
    scenes = [spawn_scene(sim,args,layout,sources,f'/World/envs/env_{i}',
              np.array([i%4,i//4,0.])*args.env_spacing_m) for i in range(args.num_envs)]
    isolate_scenes(sim.stage,scenes)
    sim.reset()
    signal.signal(signal.SIGINT,signal.default_int_handler)
    backends = [bind_scene(sim,args,scene,seed_state) for scene in scenes]
    counts = [0]*args.num_envs
    def sink(index):
        directory=args.output/f'env_{index:03d}'
        directory.mkdir()
        def save(summary,rows):
            number=counts[index];counts[index]+=1
            write_json(directory/f'episode_{number:05d}.json',rows)
            write_json(directory/'latest_episode.json',dict(summary,env_index=index,episode=number))
        return save
    envs=[TabletopEnv(b,stage='full',reset_phase='close',control_mode=args.control_mode,
        grasp_mode='object_stability',initial_offset=seed_state['root_offset'],reset_jitter=args.reset_height_jitter_m,
        episode_sink=sink(i),force_direction=force_direction,force_palm_compensation=force_palm_compensation,load_parameters=args.load_parameters) for i,b in enumerate(backends)]
    advance = lambda requests: advance_world(sim,backends,requests,render=not args.headless)
    if args.protected_mimic_resets:
        from go2_pvcnn.control.dh116.tabletop_protected_vec_env import ProtectedMimicVecEnv, make_joint_guard
        vec = ProtectedMimicVecEnv(envs, advance, joint_guard=make_joint_guard(scenes))
    else:
        vec = SharedPhysicsVecEnv(envs, advance)
    # Actual isolation probe: a selected reset must not tick physics or change
    # another slot's physical state/clock; repeated local resets must agree.
    vec.seed(args.seed);first=vec.reset()
    other=[b.measure() for b in backends[1:]]
    envs[0].reset(seed=args.seed)
    after=[b.measure() for b in backends[1:]]
    independent=all(np.array_equal(a['q'],b['q']) and np.array_equal(a['object_relative'],b['object_relative'])
                    for a,b in zip(other,after))
    vec.seed(args.seed);second=vec.reset()
    repeatable=bool(np.allclose(first,second,atol=1e-4,rtol=0))
    report['isolation']=dict(reset_does_not_change_other_envs=independent,reset_repeatable=repeatable,
        origins=[s['origin'].tolist() for s in scenes],sensor_filters=[[s['prefix']+'/Wire']*4 for s in scenes],
        collision_groups=args.num_envs)
    if not independent or not repeatable:
        raise ValueError('physical per-environment reset isolation failed')
    if args.object_material != 'default':
        report['object_material_verification']=[]
        for scene in scenes:
            properties=scene['wire'].root_physx_view.get_material_properties().cpu().numpy()
            audit=verify_material_properties(properties,args.object_material,
                len(scene['object_material_colliders']))
            prim=sim.stage.GetPrimAtPath(scene['prefix']+'/ObjectContactMaterial')
            for attr,expected in (('frictionCombineMode','max'),('restitutionCombineMode','min')):
                actual=prim.GetAttribute('physxMaterial:'+attr).Get()
                if actual != expected:
                    raise ValueError('object material combination mode mismatch')
                audit[attr]=actual
            report['object_material_verification'].append(audit)
    if args.adaptive_grasp:
        # Audit loaded PhysX mass/inertia and live link geometry, independently
        # of the authored asset's nominal values. No extra physics step.
        report['initial_physics'] = []
        for scene,backend in zip(scenes,backends):
            measured = backend.measure()
            report['initial_physics'].append(dict(
                object_mass_kg=scene['wire'].root_physx_view.get_masses().cpu().tolist(),
                object_inertia_kg_m2=scene['wire'].root_physx_view.get_inertias().cpu().tolist(),
                pad_relative_m=measured['pad_relative'].tolist(),
                hand_clearance_m=measured['hand_clearance'].tolist(),
                grip_stiffness=measured['grip_stiffness'].tolist()))
    from go2_pvcnn.control.dh116.tabletop_load_parameters import physics_load_readback, LoadParameters
    expected_load = args.load_parameters if args.control_mode == 'adaptive_grasp' else LoadParameters(effort_limit_nm=.15)
    report['physical_parameter_readback'] = physics_load_readback([h for scene in scenes for h in scene['hands']],expected_load)
    write_json(args.output/'configuration.json',report)
    training,model=run_vector_training(vec,args.output,contract,timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,checkpoint_every=args.checkpoint_every,seed=args.seed,
        resume=args.resume,initialize_from=args.initialize_from,
        allow_physics_refinement=args.allow_physics_refinement,
        allow_mimic_refinement=args.allow_mimic_refinement,
        allow_protected_mimic_init=args.allow_protected_mimic_init,probe_only=args.probe_only)
    report.update(training)
except KeyboardInterrupt:
    report.update(status='interrupted',interrupted=True)
except Exception:
    report.update(status='error',error=traceback.format_exc())
    print(report['error'],flush=True)
finally:
    if vec is not None:
        vec.close()
    code=2 if (report.get('error') or report.get('interrupted') or report.get('training_skipped')
        or report.get('training_stopped')=='physical_safety_failure') else 0
    report['process_exit_code']=code
    write_json(args.output/'summary.json',report)
    print('VECTOR_PPO_RESULT',json.dumps(report),flush=True)
    import omni.kit.app
    omni.kit.app.get_app().post_quit(code)
    app.close(wait_for_replicator=False,skip_cleanup=True)
raise SystemExit(code)
