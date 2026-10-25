"""Cloned GPU physics for adaptive close-to-lift PPO, with host task control."""
from __future__ import annotations
import argparse
import hashlib
import json
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
parser.add_argument('--object-kind', choices=('solid_cylinder','hollow_straw','deformx_wire'),default='solid_cylinder')
parser.add_argument('--object-material', choices=('default','high_friction','rough_rod'),default='default')
parser.add_argument('--hand-material', choices=('default','grip'),default='default',
                    help='explicit hand collider friction; grip is an uncalibrated training profile')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=4)
parser.add_argument('--env-spacing-m', type=float, default=2.)
budget = parser.add_mutually_exclusive_group()
budget.add_argument('--timesteps', type=int, default=65536)
budget.add_argument('--episodes-per-env', type=int,
                    help='train until every env completes this many episodes, checked after full PPO updates')
parser.add_argument('--eval-episodes', type=int, default=8)
parser.add_argument('--checkpoint-every', type=int, default=2048)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', type=Path)
parser.add_argument('--play-checkpoint', type=Path,
                    help='single-environment GUI policy inference; no learning')
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
parser.add_argument('--direct-training', action='store_true',
                    help='skip three full admission batches; retain startup contact and runtime safety checks')
parser.add_argument('--physics-hz', type=int, choices=(240, 480), default=240)
parser.add_argument('--solver-type', type=int, choices=(0,), default=0)
parser.add_argument('--position-iterations', type=int, default=16)
parser.add_argument('--reset-height-jitter-m', type=float, default=.0003)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.full_hand_contact and (args.object_kind == 'deformx_wire' or not args.force_feedback):
    parser.error('full-hand contact requires rigid-object force feedback')
if args.joint_envelope and not args.full_hand_contact:
    parser.error('joint-envelope requires full-hand contact')
args.joint_envelope_data = None
if args.joint_envelope:
    from go2_pvcnn.control.dh116.tabletop_joint_envelope import bounds
    args.joint_envelope_data = json.loads(args.joint_envelope.read_text())
    bounds(args.joint_envelope_data)
if args.object_kind == 'deformx_wire' and (args.hand_material != 'default' or args.object_material == 'rough_rod'):
    parser.error('explicit rigid friction profiles require a rigid object; external rod contact uses its own material')
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
if (args.output.exists() or not 1 <= args.num_envs <= 4096 or args.env_spacing_m < 2.
        or args.timesteps <= 0 or not 0 <= args.reset_height_jitter_m <= .002
        or args.eval_episodes < 2 or args.checkpoint_every <= 0 or args.device != 'cuda:0'):
    parser.error('new output,1..4096 envs,spacing>=2,positive budget,CUDA and bounded jitter required')
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
    if args.object_kind == 'deformx_wire':
        from dataclasses import asdict
        from go2_pvcnn.control.dh116.deformx_wire_runtime import paper_rod_config
        asset_dir=Path(__file__).resolve().parents[1]/'assets/deformx_wire_v1'
        manifest=json.loads((asset_dir/'manifest.json').read_text())
        args.object_usd=asset_dir/'wire.usda'
        if not args.object_usd.is_file(): raise ValueError('DeformX asset bundle is missing')
        args.object_mass_kg=700.*3.141592653589793*.005**2
        object_contract=dict(object_kind='deformx_wire',object_asset=manifest,
            object_mass_kg=args.object_mass_kg,object_backend='batched_cuda_cosserat_v1',
            grasp_geometry='fixed_material_regions_40mm_v1',
            object_observation='two_region_center_mean_and_max_region_speed_v1',
            visual='original_skin_mesh_points_env0',rope_self_contact=False)
    else:
        args.object_usd, args.object_mass_kg, object_contract = object_spec(
            Path(__file__).resolve().parents[1], args.object_kind)
except (ValueError, OSError, KeyError) as error:
    parser.error(str(error))
args.output.mkdir(parents=True)
write_json(args.output/'monitor.json',dict(status='initializing',num_envs=args.num_envs,
    timesteps=0,updates=0,object_kind=args.object_kind,episodes_per_env_target=args.episodes_per_env))
app = AppLauncher(args).app
report = dict(passed=False, grasp_qualified=False, num_envs=args.num_envs,
              status='initializing', rigid_proxy=args.object_kind != 'deformx_wire', pipeline_scope='close_to_lift')
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
        num_envs=args.num_envs,env_spacing_m=args.env_spacing_m,vector_backend='cloned_gpu_normalized_convex_batched_task_v2')
    contract.update(seed_provenance)
    contract.update(hand_material_contract(args.hand_material))
    contract.update(object_contract)
    if args.object_kind == 'deformx_wire':
        contract['flexible_physics']=asdict(paper_rod_config(seed_state['wire_pose'][:3],layout['table_center'],layout['table_size']))
        contract['object_contact_material']=dict(model='penalty_regularized_coulomb',friction=1.,
            normal_stiffness_n_m=10000.,normal_damping_ns_m=10.,tangential_damping_ns_m=10.)
    else:
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
    if args.joint_envelope_data is not None:
        contract.pop('target_radius_rad', None)
        contract.update(joint_envelope=args.joint_envelope_data,
            joint_envelope_sha256=sha(args.joint_envelope),
            action='20D_piecewise_joint_envelope_palm_load_v1')
    contract = canonical_contract(contract)
    report['contract'] = contract
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/args.physics_hz,device=args.device,
        physx=sim_utils.PhysxCfg(solver_type=args.solver_type)))
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
    vec = GpuTaskVecEnv(envs, world)
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
        origins=world.origins.tolist(), sensor_filters=('external rod contacts, explicit per-env CUDA indexing'
            if world.flexible else 'Wire-only, GPU environment-ID isolation'),
        collision_groups=args.num_envs)
    if not independent or not repeatable:
        raise ValueError('physical per-environment reset isolation failed')
    if not world.flexible and args.object_material != 'default':
        properties=world.wire.root_physx_view.get_material_properties().cpu().numpy()
        report['object_material_verification']=[verify_material_properties(properties[i:i+1],
            args.object_material,len(world.source['object_material_colliders'])) for i in range(args.num_envs)]
    if args.hand_material != 'default':
        report['hand_material_verification'] = {
            side: verify_hand_material_properties(hand.root_physx_view.get_material_properties().cpu().numpy(),
                args.hand_material,args.num_envs,len(world.source['hand_material_colliders'][side]))
            for side,hand in zip(('left','right'),world.hands)}
        report['hand_material_bindings'] = world.source['hand_material_bindings']
    write_json(args.output/'material_verification.json',dict(
        object_material=report.get('object_material_verification'),
        hands=report.get('hand_material_verification'),hand_bindings=report.get('hand_material_bindings'),
        contract={k:contract[k] for k in ('object_contact_material','hand_contact_material') if k in contract}))
    if world.flexible:
        local_centers=world.flex.engine.position.mean(dim=2).cpu().numpy()
        center_error=float(np.max(np.abs(local_centers-seed_state['wire_pose'][:3])))
        expected_roots=np.stack([layout['hand_roots'][s][:3] for s in ('left','right')])
        expected_roots[:,:2]+=seed_state['root_xy']
        expected_roots[:,2]+=seed_state['root_offset']
        root_error=float(np.max(np.abs(world.roots[:,:,:3]-world.origins[:,None]-expected_roots)))
        if center_error>1e-9 or (args.reset_height_jitter_m==0 and root_error>5e-6):
            raise ValueError('flexible replacement altered the original reset position')
        report['initial_physics']=dict(object_mass_kg=args.object_mass_kg,
            seed_center_error_m=center_error,hand_root_position_error_m=root_error,
            seed_wire_pose=seed_state['wire_pose'].tolist(),hand_root_poses=world.roots[:8].tolist(),
            material_intervals_m=world.flex.regions.intervals,
            pad_relative_m=world.pad_relative[:8].tolist(),hand_clearance_m=world.clearances[:8].tolist(),
            position_preserved=True,grasp_point_offset_m=0.,height_adjustment_m=0.)
    else:
        report['initial_physics'] = [dict(object_mass_kg=world.wire.root_physx_view.get_masses().cpu().tolist(),
            object_inertia_kg_m2=world.wire.root_physx_view.get_inertias().cpu().tolist(),
            pad_relative_m=world.pad_relative.tolist(),hand_clearance_m=world.clearances.tolist())]
    if args.num_envs >= 12 and not world.flexible:
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
    if world.flexible:
        # A thinner replacement can initially miss the pads. Contact failure
        # is learnable; it is not a numerical fault or an admission criterion.
        before=world.flex.engine.position.clone()
        if args.num_envs > 1:
            world.flex.engine.position[1,2] += .0001
            untouched=world.flex.engine.position[1:].clone()
            envs[0].reset(seed=args.seed)
            if not torch.equal(untouched,world.flex.engine.position[1:]):
                raise ValueError('flexible reset changed another material state')
            world.flex.engine.position.copy_(before)
            world.refresh(sensors=False)
        report['isolation']['asymmetric_object_mapping']=True
        started=time.monotonic()
        for tick in range(2):
            obs,rewards,dones,infos=vec.step(np.zeros((args.num_envs,20),dtype=np.float32))
            if not np.isfinite(obs).all() or any(info.get('safety_failure') for info in infos):
                raise ValueError('flexible startup numerical/hand safety failure')
            write_json(args.output/'monitor.json',dict(status='startup_preflight',num_envs=args.num_envs,
                vector_steps=tick+1,elapsed_seconds=time.monotonic()-started))
        report['contact_preflight']=dict(mode='finite_bounded_exploration_contact_not_required',
            vector_steps=2,elapsed_seconds=time.monotonic()-started,
            peak_hand_load_n=world.force.reshape(args.num_envs,2,2).sum(axis=2).tolist())
        write_json(args.output/'contact_preflight.json',report['contact_preflight'])
    else:
        from go2_pvcnn.control.dh116.tabletop_preflight import contact_preflight
        report['contact_preflight'] = contact_preflight(vec,world,direct=bool(args.direct_training or args.play_checkpoint),
            progress=lambda row: write_json(args.output/'monitor.json',row))
        write_json(args.output/'contact_preflight.json',report['contact_preflight'])
    from go2_pvcnn.control.dh116.tabletop_load_parameters import physics_load_readback
    report['physical_parameter_readback'] = physics_load_readback(world.hands,args.load_parameters)
    write_json(args.output/'configuration.json',report)
    if args.play_checkpoint:
        from go2_pvcnn.control.dh116.tabletop_playback import play_policy
        report.update(play_policy(vec,app,args.play_checkpoint,contract,args.output,
            device=args.device,seed=args.seed,episodes=args.play_episodes))
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
            ppo_kwargs=dict(device=args.device,n_steps=64,batch_size=min(2048,64*args.num_envs)))
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
