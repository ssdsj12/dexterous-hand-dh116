"""Single-scene, measured DH116 tabletop MPC-residual PPO experiment.

Rigid cylinder proxy only. No attachment constraints or object pose writes
after reset. Direct-close mode learns contact topology from object lift/stability;
legacy residual mode retains its four-pad gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
import hashlib
from go2_pvcnn.control.dh116.tabletop_training import report_exit_code

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pose-summary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--timesteps', type=int, default=1024)
parser.add_argument('--evaluate-only', action='store_true')
parser.add_argument('--probe-only', action='store_true', help='Collect current-stage exploration admission evidence only')
parser.add_argument('--stage', choices=('approach', 'close', 'lift', 'full'), default=None)
parser.add_argument('--curriculum', action='store_true', help='Train approach, close/hold, then lift with measured promotion gates')
parser.add_argument('--initialize-from', type=Path, help='Reuse policy weights across pregrasp handoff change; discard optimizer and qualifications')
parser.add_argument('--resume', type=Path, help='Resume a checkpoint with matching observation/action/physics metadata')
parser.add_argument('--checkpoint', type=Path, help='Evaluate a saved checkpoint; requires --evaluate-only')
parser.add_argument('--eval-episodes', type=int, default=5)
parser.add_argument('--probe-episodes', type=int, default=3)
parser.add_argument('--checkpoint-every', type=int, default=512)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--reset-height-jitter-m', type=float, default=.001)
parser.add_argument('--residual-scale-rad', type=float, default=.02,
                    help='Initial exploration radius; changing it requires fresh admission and a new checkpoint contract')
parser.add_argument('--control-mode', choices=('residual', 'joint_target'), default='residual')
parser.add_argument('--close-reset-snapshot', type=Path, help='Measured close seed with source contract and trajectory row')
parser.add_argument('--joint-target-radius-rad', type=float, default=.08)
parser.add_argument('--replay-episode', type=Path, help='Diagnostic recorded-action replay; no PPO or qualification')
parser.add_argument('--replay-zero-steps', type=int, default=0, help='Continue replay with zero actions')

parser.add_argument('--approach-only', action='store_true',
                    help='Execute approach only, stopping before close or at the clearance guard')
parser.add_argument('--hold-window-s', type=float, default=0.)
parser.add_argument('--preview-only', action='store_true', help='Show the initial scene without motion or training')
parser.add_argument('--solver-type', type=int, choices=(0, 1), default=0,
                    help='PhysX solver: 0 PGS, 1 TGS; for controlled stability comparisons')
parser.add_argument('--position-iterations', type=int, choices=range(1, 256), default=16)
parser.add_argument('--physics-hz', type=int, choices=(240, 480, 960), default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists() or args.timesteps < 0 or args.hold_window_s < 0:
    raise ValueError('new output directory and nonnegative timesteps required')
if (args.eval_episodes < 2 or args.probe_episodes < 3 or args.checkpoint_every <= 0
        or not 0 <= args.reset_height_jitter_m <= .002 or not 0 < args.residual_scale_rad <= .1):
    parser.error('require >=2 evaluations, >=3 probes, positive checkpoint interval, jitter in [0,.002], radius in (0,.1]')
if args.checkpoint and not args.evaluate_only:
    parser.error('--checkpoint requires --evaluate-only')
if args.initialize_from and (args.resume or args.checkpoint or args.evaluate_only or args.probe_only):
    parser.error('--initialize-from is a separate training-only weights transfer')
if args.resume and (args.checkpoint or args.evaluate_only or args.probe_only):
    parser.error('--resume is for training; use --checkpoint for evaluation')
if args.curriculum and (args.stage or args.evaluate_only or args.probe_only):
    parser.error('--curriculum is a training mode; use --stage for individual probes/evaluation')
if bool(args.close_reset_snapshot) != (args.control_mode == 'joint_target'):
    parser.error('joint_target mode requires --close-reset-snapshot, and vice versa')
if args.close_reset_snapshot and (args.approach_only or args.preview_only or args.initialize_from or args.stage == 'approach'):
    parser.error('direct close cannot run approach or import the old residual policy')
if args.replay_zero_steps < 0 or (args.replay_zero_steps and not args.replay_episode):
    parser.error('nonnegative replay extension requires --replay-episode')
if args.replay_episode and (not args.close_reset_snapshot or args.curriculum or args.resume
        or args.checkpoint or args.initialize_from or args.evaluate_only or args.probe_only):
    parser.error('replay requires direct close and is separate from training/evaluation')
if args.resume and args.stage is None and not args.curriculum:
    args.stage = json.loads(args.resume.with_suffix('.json').read_text())['stage']
args.stage = args.stage or ('full' if args.evaluate_only or args.close_reset_snapshot else 'approach')
args.output.mkdir(parents=True)
app = AppLauncher(args).app
# Kit installs its own SIGINT handler; restore Python unwinding so checkpoint
# and final report cleanup execute when the user interrupts training.
import signal
signal.signal(signal.SIGINT, signal.default_int_handler)
report = dict(passed=False, grasp_qualified=False, rigid_proxy=True, episodes=[], ppo_updates=0,
              scene_mode='hand_only', arm_state_included=False,
              hand_paths=['/World/leftHand', '/World/rightHand'])
env = None
try:
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from scipy.spatial import ConvexHull
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import Usd, UsdGeom
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, expand, active_limits
    from go2_pvcnn.control.dh116.hand_only_scene import native_pose_hand_only_layout
    from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure
    from go2_pvcnn.control.dh116.tabletop_control import OBSERVATION_VERSION, OBSERVATION_SIZE, prepared_approach_reference, track_root_xy, moving_goal_error
    from go2_pvcnn.control.dh116.tabletop_env import TabletopEnv
    from go2_pvcnn.control.dh116.tabletop_training import (
        run_training, evaluate, run_episode, admission_report, load_checkpoint, write_json)
    from go2_pvcnn.control.dh116.runtime_mimic import install_hand_mimic
    from go2_pvcnn.control.dh116.tabletop_direct import snapshot_state, adjust_palms

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    layout = native_pose_hand_only_layout(json.loads(args.pose_summary.read_text()),
                                         roll_degrees=dict(left=76., right=82.))
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
    sources = {x['side']: x['path'] for x in manifest['sources'] if x['kind'] == 'dh116'}
    dt, decimation = 1/args.physics_hz, args.physics_hz//60
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device,
        physx=sim_utils.PhysxCfg(solver_type=args.solver_type)))
    report['solver_type'] = args.solver_type
    report['position_iterations'] = args.position_iterations
    if not args.headless:
        sim.set_camera_view(eye=(0.85, -0.72, 0.88), target=(0.43, 0.0, 0.34))
        light = sim_utils.DomeLightCfg(intensity=1800)
        light.func('/World/Light', light)
    table = sim_utils.CuboidCfg(size=layout['table_size'],
                               collision_props=sim_utils.CollisionPropertiesCfg())
    table.func('/World/Table', table, translation=layout['table_center'])
    hands, sensors = [], []
    for side in ('left', 'right'):
        pose = layout['hand_roots'][side].copy()
        pose[2] += .04
        cfg = ArticulationCfg(prim_path=f'/World/{side}Hand',
            spawn=sim_utils.UsdFileCfg(usd_path=sources[side], activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                    enabled_self_collisions=False, solver_position_iteration_count=args.position_iterations,
                    solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:]),
                joint_pos=dict(zip(JOINTS, map(float, expand(
                    prepared_approach_reference(layout['active_joints'][side])))))),
            actuators={'active': ImplicitActuatorCfg(joint_names_expr=list(ACTIVE),
                stiffness=10., damping=.2, effort_limit_sim=.15, velocity_limit_sim=.3),
                'passive': ImplicitActuatorCfg(joint_names_expr=[n for n in JOINTS if n not in ACTIVE],
                    stiffness=0., damping=0., effort_limit_sim=0.,
                    # A low solver brake on passive followers competes with
                    # mimic/contact correction. Keep active drives at .3.
                    velocity_limit_sim=3. if args.control_mode == 'joint_target' else .3)})
        hands.append(Articulation(cfg))
        install_hand_mimic(sim.stage, f'/World/{side}Hand')
        for link in ('finger14_link', 'finger23_link'):
            sensors.append(ContactSensor(ContactSensorCfg(prim_path=f'/World/{side}Hand/{link}',
                filter_prim_paths_expr=['/World/Wire'], update_period=0.)))
    wire = RigidObject(RigidObjectCfg(prim_path='/World/Wire',
        spawn=sim_utils.CylinderCfg(radius=.008, height=.5, axis='Y',
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
            mass_props=sim_utils.MassPropertiesCfg(mass=.12),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1., .35, .02))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=layout['wire_center'])))
    sim.reset()
    # SimulationContext also installs a stop handler after AppLauncher.
    # Restore unwinding here, after context creation, for checkpoint cleanup.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    ids = [[h.joint_names.index(n) for n in ACTIVE] for h in hands]
    # Cache convex-hull vertices in each link frame. Runtime transforms come
    # from PhysX link poses, never stale USD world transforms or COM poses.
    shapes = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for hi, side in enumerate(('left', 'right')):
        base = f'/World/{side}Hand'
        for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(base), Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if '/collisions/' not in path or not prim.IsA(UsdGeom.Mesh):
                continue
            link = path[len(base)+1:].split('/')[0]
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(
                sim.stage.GetPrimAtPath(base+'/'+link)).GetInverse()
            points = (np.c_[points, np.ones(len(points))] @ np.asarray(relative))[:, :3]
            points = np.unique(points, axis=0)
            points = points[ConvexHull(points).vertices]
            shapes.append((hi, hands[hi].body_names.index(link), points))
    if len(shapes) != 34:
        raise ValueError(f'expected 34 collision meshes, found {len(shapes)}')

    def rotation(q):
        return Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()

    class IsaacBackend:
        def __init__(self):
            self.dt, self.decimation = dt, decimation
            self.grasp = np.concatenate([layout['active_joints'][s] for s in ('left', 'right')])
            self.forces = np.zeros(4)
            self.seed = None
            self.palm_z = np.zeros(2)
            self.clearance = .005
            if args.close_reset_snapshot:
                snapshot = json.loads(args.close_reset_snapshot.read_text())
                provenance = snapshot['source_contract']
                expected = dict(pose_sha256=hashlib.sha256(args.pose_summary.read_bytes()).hexdigest(),
                    source_sha256={s:hashlib.sha256(Path(p).read_bytes()).hexdigest() for s,p in sources.items()},
                    physics_hz=args.physics_hz, control_hz=60, solver_type=args.solver_type,
                    position_iterations=args.position_iterations)
                if any(provenance.get(k) != v for k,v in expected.items()):
                    raise ValueError('close seed assets/pose/physics provenance mismatch')
                self.seed = snapshot_state(snapshot['row'], layout)
                self.initial_command = self.seed['initial_command'].copy()

        def reset(self, offset):
            self.root_xy = np.zeros(2) if self.seed is None else self.seed['root_xy'].copy()
            self.palm_z = np.zeros(2)
            for hi, (hand, side) in enumerate(zip(hands, ('left', 'right'))):
                hand.reset()
                q = hand.data.default_joint_pos.clone()
                qd = torch.zeros_like(q)
                named_ids = [hand.joint_names.index(n) for n in JOINTS]
                q[:, named_ids] = torch.tensor(
                    expand(prepared_approach_reference(layout['active_joints'][side])) if self.seed is None
                    else self.seed['joint_positions'][hi], device=args.device, dtype=torch.float32)
                if self.seed is not None:
                    qd[:, named_ids] = torch.tensor(self.seed['joint_velocities'][hi], device=args.device, dtype=torch.float32)
                hand.write_joint_state_to_sim(q, qd)
                hand.set_joint_position_target(q[:, ids[hi]], joint_ids=ids[hi])
                pose = layout['hand_roots'][side].copy()
                pose[:2] += self.root_xy
                pose[2] += offset
                hand.write_root_pose_to_sim(torch.tensor(pose[None], device=args.device, dtype=torch.float32))
                hand.write_root_velocity_to_sim(torch.zeros((1, 6), device=args.device))
                hand.write_data_to_sim()
            wire.reset()
            wire.write_root_pose_to_sim(wire.data.default_root_state[:, :7].clone() if self.seed is None
                else torch.tensor(self.seed['wire_pose'][None],device=args.device,dtype=torch.float32))
            wire.write_root_velocity_to_sim(torch.zeros((1, 6), device=args.device) if self.seed is None
                else torch.tensor(self.seed['wire_velocity'][None],device=args.device,dtype=torch.float32))
            for sensor in sensors:
                sensor.reset()
            sim.physics_sim_view.update_articulations_kinematic()
            for hand in hands:
                hand.update(dt)
            wire.update(dt)
            self.forces[:] = 0.
            if self.seed is not None:
                # Rebuild real solver contacts; no task timers/reward during
                # settling. Restoring logged forces would fabricate contact.
                for _ in range(12):
                    self.advance(offset, self.initial_command, phase='close', palm_motion_allowed=False)
                    measured = self.measure()
                    if (measured['joint_failure'] or measured['clearance'] < -.001
                            or measured['bottom'] < .29 or np.max(measured['forces']) > 30):
                        raise ValueError('unsafe direct-close initialization')

        def advance(self, offset, command, *, phase, palm_motion_allowed):
            # Track measured cylinder translation only before a stable grasp.
            # Root motion remains scripted; no object pose is assigned here.
            self.root_xy = track_root_xy(self.root_xy,
                wire.data.root_pos_w[0].detach().cpu().numpy(), layout['wire_center'], phase, dt, allowed=palm_motion_allowed)
            if self.seed is not None:
                self.root_xy = np.clip(self.root_xy, self.seed['root_xy']-.003, self.seed['root_xy']+.003)
                self.palm_z = adjust_palms(self.palm_z, self.forces, self.clearance,
                                          phase, dt, allowed=palm_motion_allowed)
            for hi, (hand, side) in enumerate(zip(hands, ('left', 'right'))):
                pose = layout['hand_roots'][side].copy()
                pose[:2] += self.root_xy
                pose[2] += offset + self.palm_z[hi]
                hand.write_root_pose_to_sim(torch.tensor(pose[None], device=args.device, dtype=torch.float32))
                hand.set_joint_position_target(torch.tensor(command[hi*6:hi*6+6][None],
                    device=args.device, dtype=torch.float32), joint_ids=ids[hi])
                hand.write_data_to_sim()
            sim.step(render=not args.headless)
            for hand in hands:
                hand.update(dt)
            wire.update(dt)
            for i, sensor in enumerate(sensors):
                sensor.update(dt, force_recompute=True)
                self.forces[i] = float(torch.linalg.vector_norm(sensor.data.force_matrix_w).item())

        def measure(self):
            poses = [h.data.body_link_pose_w[0].detach().cpu().numpy() for h in hands]
            heights = [float((p @ rotation(poses[i][j, 3:]).T + poses[i][j, :3])[:, 2].min())
                       for i, j, p in shapes]
            lowest = int(np.argmin(heights))
            self.clearance = heights[lowest]-.30
            hi, bi, _ = shapes[lowest]
            diagnostics = []
            joint_failure = None
            for hand in hands:
                q = hand.data.joint_pos[0].detach().cpu().numpy()
                named = dict(zip(hand.joint_names, map(float, q)))
                errors = {f: named[f]-r*named[m]-o for f, m, r, o in MIMIC}
                diagnostics.append(dict(joints_rad=named, mimic_error_rad=errors,
                    joint_vel_rad_s=hand.data.joint_vel[0].detach().cpu().tolist(),
                    applied_torque_nm=hand.data.applied_torque[0].detach().cpu().tolist(),
                    root_pose=hand.data.root_pose_w[0].detach().cpu().tolist()))
                failure = joint_safety_failure(q,
                    hand.data.joint_pos_limits[0].detach().cpu().numpy(), list(errors.values()))
                joint_failure = joint_failure or failure
            wp = wire.data.root_pos_w[0].detach().cpu().numpy()
            wq = wire.data.root_quat_w[0].detach().cpu().numpy()
            axis_z = abs(rotation(wq)[2, 1])
            bottom = float(wp[2] - .25*axis_z - .008*np.sqrt(max(0., 1-axis_z**2)))
            roots = np.stack([h.data.root_pos_w[0].detach().cpu().numpy() for h in hands])
            goal = np.stack([layout['hand_roots'][s][:3] for s in ('left', 'right')])
            return dict(q=np.concatenate([h.data.joint_pos[0,k].detach().cpu().numpy()
                                          for h,k in zip(hands,ids)]),
                qd=np.concatenate([h.data.joint_vel[0,k].detach().cpu().numpy() for h,k in zip(hands,ids)]),
                forces=self.forces.copy(), clearance=heights[lowest]-.30,
                object_relative=wp-roots.mean(axis=0), object_quat=wq.copy(),
                object_velocity=wire.data.root_vel_w[0].detach().cpu().numpy().copy(),
                bottom=bottom, approach_error=moving_goal_error(roots, goal, wp, layout['wire_center']),
                joint_failure=joint_failure, diagnostics=diagnostics,
                mimic_errors=np.array([e for h in diagnostics for e in h['mimic_error_rad'].values()]),
                palm_adjustment=np.r_[self.root_xy-self.seed['root_xy'],self.palm_z]
                    if self.seed is not None else np.zeros(4),
                lowest_link=f'{("left", "right")[hi]}/{hands[hi].body_names[bi]}')

    def save_episode(summary, rows):
        summary['index'] = len(report['episodes'])
        report['episodes'].append(summary)
        write_json(args.output/f"episode_{summary['index']:04d}.json", rows)

    backend = IsaacBackend()
    env = TabletopEnv(backend, stage=args.stage,
        reset_jitter=args.reset_height_jitter_m, episode_sink=save_episode,
        residual_scale=args.residual_scale_rad,
        reset_phase='close' if backend.seed is not None else 'approach',
        control_mode=args.control_mode, target_radius=args.joint_target_radius_rad,
        grasp_mode='object_stability' if backend.seed is not None else 'four_pad',
        initial_offset=backend.seed['root_offset'] if backend.seed is not None else .04)

    def sha256(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    contract = dict(observation_version=OBSERVATION_VERSION, observation_size=OBSERVATION_SIZE,
        action='12D_bounded_nominal_residual_v2', residual_scale_rad=args.residual_scale_rad,
        residual_rate_rad_s=.05,
        nominal='calibrated_pose_rate_limited_v1', nominal_rate_rad_s=.15,
        palm_motion='prepared_object_tracking_reacquisition_v3', coupling_pause_rad=.008, root_xy_rate_m_s=.01,
        approach_reference='calibrated_minus_004_rad', approach_pregrasp_radius_m=.005,
        command_rate_rad_s=.3, reward='measured_progress_v2',
        pose_sha256=sha256(args.pose_summary), source_sha256={s:sha256(p) for s,p in sources.items()},
        solver_type=args.solver_type, position_iterations=args.position_iterations,
        physics_hz=args.physics_hz, control_hz=60, reset_height_jitter_m=args.reset_height_jitter_m)
    if backend.seed is not None:
        contract.update(observation_version='dh116-direct-close-v1', observation_size=108,
            action='12D_direct_joint_target_v1', target_radius_rad=args.joint_target_radius_rad,
            nominal='fixed_target_center_no_scripted_closure', nominal_rate_rad_s=0.,
            command_rate_rad_s=.05, reward='object_support_progress_v4_10n_5', coupling_pause_rad=.004,
            grasp_mode='object_stability', lift_attempt_after_s=2.3,
            success_bottom_m=.33, success_dwell_s=.5, success_slip_m=.02,
            reset_phase='close', close_seed_sha256=sha256(args.close_reset_snapshot),
            reset_settle_steps=12, palm_motion='bounded_per_hand_close_adjustment_v1',
            palm_xy_limit_m=.003, palm_z_limit_m=.002, palm_z_rate_m_s=.001,
            policy='bounded_gaussian_mean_v1', force_observation='log1p_force_over_01N',
            coupling_warning='follower_consistent_recovery_v2',
            active_velocity_limit_rad_s=.3, passive_velocity_limit_rad_s=3.)
    report.update(contract=contract, collision_meshes=len(shapes), control_hz=60,
                  physics_hz=args.physics_hz, stage=args.stage, seed=args.seed,
                  approach_completion_is_grasp_success=False)
    report['pipeline_scope'] = 'close_to_lift' if backend.seed is not None else 'approach_to_lift'
    write_json(args.output/'configuration.json', report)
    if args.replay_episode:
        from go2_pvcnn.control.dh116.tabletop_replay import replay_episode
        env.set_mode('replay')
        replay = replay_episode(env, json.loads(args.replay_episode.read_text()),
            seed=args.seed, extend_zero_steps=args.replay_zero_steps)
        report.update(status='diagnostic_only', replay=replay,
            replay_source_sha256=sha256(args.replay_episode))
        if replay['safety_failure']:
            report['training_stopped'] = 'physical_safety_failure'
    elif args.approach_only or args.preview_only:
        env.set_stage('approach')
        env.set_mode('preview')
        obs, _ = env.reset(seed=args.seed)
        if not args.preview_only:
            for _ in range(1000):
                obs, reward, done, _, info = env.step(np.zeros(12))
                if done:
                    break
            report['approach'] = info
        report['preview_only'] = args.preview_only
        report['final_clearance_m'] = env.metrics['clearance_m']
        if not args.headless:
            import time
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            sim.pause()
            sim.render()
            capture_viewport_to_file(get_active_viewport(), str(args.output/'approach.png'))
            deadline = time.monotonic() + max(3., args.hold_window_s)
            while time.monotonic() < deadline and app.is_running():
                app.update()
    elif args.probe_only:
        env.set_mode('admission')
        probes = [run_episode(env, seed=args.seed+i, random_actions=i>0,
                              probe_stage=args.stage, check_reset=True)
                  for i in range(args.probe_episodes)]
        report['admission'] = admission_report(args.stage, probes)
    elif args.evaluate_only or args.timesteps == 0:
        model = load_checkpoint(args.checkpoint, env, contract)[0] if args.checkpoint else None
        assessment = evaluate(env, model, args.stage, episodes=args.eval_episodes, seed=args.seed)
        report['evaluation'] = assessment
        report['passed'] = assessment['physically_safe'] and assessment['grasp_success_rate'] >= .8
        report['grasp_qualified'] = report['passed']
    else:
        training, model = run_training(env, args.output, contract,
            stages=(['full'] if backend.seed is not None else ['approach', 'close', 'lift'])
                if args.curriculum else [args.stage],
            timesteps=args.timesteps, eval_episodes=args.eval_episodes,
            probe_episodes=args.probe_episodes, checkpoint_every=args.checkpoint_every,
            resume=args.resume, initialize_from=args.initialize_from, seed=args.seed)
        report.update(training)
except Exception:
    report['error'] = traceback.format_exc()
    print(report['error'], flush=True)
except KeyboardInterrupt:
    report['interrupted'] = True
    report['status'] = 'interrupted'
finally:
    if env is not None:
        try:
            env.flush_episode()
        except Exception:
            report['episode_flush_error'] = traceback.format_exc()
            report.setdefault('error', report['episode_flush_error'])
    training_path = args.output/'training_report.json'
    if training_path.exists():
        report.update(json.loads(training_path.read_text()))
    exit_code = report_exit_code(report)
    report['process_exit_code'] = exit_code
    (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    print('TABLETOP_PPO_RESULT', json.dumps(report), flush=True)
    # Kit can terminate the process inside close(); code after it is not a
    # reliable way to propagate a rejected admission to the calling shell.
    import omni.kit.app
    omni.kit.app.get_app().post_quit(exit_code)
    app.close(wait_for_replicator=False, skip_cleanup=True)

raise SystemExit(exit_code)
