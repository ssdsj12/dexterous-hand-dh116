"""Single-scene, measured DH116 tabletop MPC-residual PPO experiment.

Rigid cylinder proxy only. No attachment constraints or object pose writes
after reset; success requires four measured pad contacts and loaded lift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pose-summary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--timesteps', type=int, default=1024)
parser.add_argument('--evaluate-only', action='store_true')
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
args.output.mkdir(parents=True)
app = AppLauncher(args).app
report = dict(passed=False, grasp_qualified=False, rigid_proxy=True, episodes=[], ppo_updates=0,
              scene_mode='hand_only', arm_state_included=False,
              hand_paths=['/World/leftHand', '/World/rightHand'])
try:
    import gymnasium as gym
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from scipy.spatial import ConvexHull
    from stable_baselines3 import PPO
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import Usd, UsdGeom
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, expand, active_limits
    from go2_pvcnn.control.dh116.hand_only_scene import native_pose_hand_only_layout
    from go2_pvcnn.control.dh116.hand_only_env import HandOnlyState
    from go2_pvcnn.control.dh116.parallel_mpc import ParallelJointMPC
    from go2_pvcnn.control.dh116.tabletop_task import TabletopTask, task_reward, REWARD_PARAMETERS
    from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure, baseline_allows_training
    from go2_pvcnn.control.dh116.tabletop_task import advance_root_offset
    from go2_pvcnn.control.dh116.tabletop_task import guarded_descent
    from go2_pvcnn.control.dh116.tabletop_task import approach_joint_reference
    from go2_pvcnn.control.dh116.runtime_mimic import install_hand_mimic

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
                    approach_joint_reference(layout['active_joints'][side], .04)))))),
            actuators={'active': ImplicitActuatorCfg(joint_names_expr=list(ACTIVE),
                stiffness=10., damping=.2, effort_limit_sim=.15, velocity_limit_sim=.3),
                'passive': ImplicitActuatorCfg(joint_names_expr=[n for n in JOINTS if n not in ACTIVE],
                    stiffness=0., damping=0., effort_limit_sim=0., velocity_limit_sim=.3)})
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

    class TabletopEnv(gym.Env):
        action_space = gym.spaces.Box(-1., 1., (12,), dtype=np.float32)
        observation_space = gym.spaces.Box(-np.inf, np.inf, (47,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.task = TabletopTask()
            self.offset, self.close_progress = .04, 0.
            self.descent_stop = None
            self.previous = np.concatenate([approach_joint_reference(layout['active_joints'][s], .04)
                                            for s in ('left', 'right')])
            self.command = self.previous.copy()
            self.planner = ParallelJointMPC(grasp_fraction=.35)
            self.grip_relative = None
            self.log = []
            self.return_ = 0.
            for hi, (hand, side) in enumerate(zip(hands, ('left', 'right'))):
                hand.reset()
                q = hand.data.default_joint_pos.clone()
                q[:, [hand.joint_names.index(n) for n in JOINTS]] = torch.tensor(
                    expand(approach_joint_reference(layout['active_joints'][side], .04)),
                    device=args.device, dtype=torch.float32)
                hand.write_joint_state_to_sim(q, torch.zeros_like(q))
                hand.set_joint_position_target(q[:, ids[hi]], joint_ids=ids[hi])
                pose = layout['hand_roots'][side].copy()
                pose[2] += self.offset
                hand.write_root_pose_to_sim(torch.tensor(pose[None], device=args.device, dtype=torch.float32))
                hand.write_root_velocity_to_sim(torch.zeros((1, 6), device=args.device))
                hand.write_data_to_sim()
            wire.reset()
            wire.write_root_pose_to_sim(wire.data.default_root_state[:, :7].clone())
            wire.write_root_velocity_to_sim(torch.zeros((1, 6), device=args.device))
            for sensor in sensors:
                sensor.reset()
            sim.physics_sim_view.update_articulations_kinematic()
            for hand in hands:
                hand.update(dt)
            wire.update(dt)
            self.forces = np.zeros(4)
            self.measure()
            return self.observe(), {}

        def measure(self):
            poses = [h.data.body_link_pose_w[0].detach().cpu().numpy() for h in hands]
            heights = [float((p @ rotation(poses[i][j, 3:]).T + poses[i][j, :3])[:, 2].min())
                       for i, j, p in shapes]
            lowest = int(np.argmin(heights))
            hi, bi, _ = shapes[lowest]
            self.lowest_link = f'{("left", "right")[hi]}/{hands[hi].body_names[bi]}'
            self.clearance = heights[lowest] - .30
            self.joint_diagnostics = []
            for hand in hands:
                q = hand.data.joint_pos[0].detach().cpu().numpy()
                named = dict(zip(hand.joint_names, map(float, q)))
                self.joint_diagnostics.append(dict(joints_rad=named,
                    mimic_error_rad={f: named[f]-r*named[m]-o for f, m, r, o in MIMIC},
                    root_pose=hand.data.root_pose_w[0].detach().cpu().tolist()))
            wp = wire.data.root_pos_w[0].detach().cpu().numpy()
            wq = wire.data.root_quat_w[0].detach().cpu().numpy()
            axis_z = abs(rotation(wq)[2, 1])
            self.bottom = float(wp[2] - .25*axis_z - .008*np.sqrt(max(0., 1-axis_z**2)))
            self.speed = float(torch.linalg.vector_norm(wire.data.root_lin_vel_w[0]).item())
            roots = np.stack([h.data.root_pos_w[0].detach().cpu().numpy() for h in hands])
            relative = wp - roots.mean(axis=0)
            if self.task.phase == 'hold' and self.grip_relative is None:
                self.grip_relative = relative.copy()
            self.slip = 0. if self.grip_relative is None else float(np.linalg.norm(relative-self.grip_relative))
            goal = np.stack([layout['hand_roots'][s][:3] for s in ('left', 'right')])
            self.approach_error = float(np.max(np.linalg.norm(roots-goal, axis=1)))
            self.metrics = dict(approach_error_m=self.approach_error, pad_forces_n=self.forces,
                clearance_m=self.clearance, object_bottom_m=self.bottom, object_speed_m_s=self.speed,
                slip_m=self.slip, max_force_n=float(self.forces.max()))

        def observe(self):
            q = np.concatenate([h.data.joint_pos[0, k].detach().cpu().numpy() for h, k in zip(hands, ids)])
            qd = np.concatenate([h.data.joint_vel[0, k].detach().cpu().numpy() for h, k in zip(hands, ids)])
            phase = np.eye(4)[('approach', 'close', 'hold', 'lift').index(self.task.phase)]
            return np.r_[q, qd, self.previous, self.forces, phase,
                         self.approach_error, self.bottom-.30, self.slip].astype(np.float32)

        def step(self, residual):
            phase = self.task.phase
            if phase == 'approach':
                reference = np.concatenate([approach_joint_reference(layout['active_joints'][s], self.offset)
                                            for s in ('left', 'right')])
                if self.descent_stop is not None:
                    reference = self.command.copy()
            elif phase == 'close':
                self.close_progress = min(1., self.close_progress+dt*decimation/4.)
                state = HandOnlyState(np.stack([layout['hand_roots'][s] for s in ('left', 'right')]),
                    self.command.reshape(2, 6), np.array([self.close_progress, 0.]), self.command)
                reference = self.planner.plan(state)[0][0]
            else:
                reference = self.command.copy()
            low, high = active_limits()
            target = np.clip(reference+.10*np.asarray(residual), np.tile(low, 2), np.tile(high, 2))
            event = False
            for _ in range(decimation):
                if phase == 'approach':
                    if self.descent_stop is None:
                        if np.any(self.forces >= .1):
                            self.descent_stop = 'pad_contact'
                        elif self.clearance <= .002:
                            self.descent_stop = 'table_margin'
                    if self.descent_stop is None:
                        self.offset = guarded_descent(self.offset, dt, self.clearance, self.forces)
                else:
                    self.offset = advance_root_offset(self.offset, phase, dt)
                self.command += np.clip(target-self.command, -.3*dt, .3*dt)
                for hi, (hand, side) in enumerate(zip(hands, ('left', 'right'))):
                    pose = layout['hand_roots'][side].copy()
                    pose[2] += self.offset
                    hand.write_root_pose_to_sim(torch.tensor(pose[None], device=args.device, dtype=torch.float32))
                    hand.set_joint_position_target(torch.tensor(self.command[hi*6:hi*6+6][None],
                        device=args.device, dtype=torch.float32), joint_ids=ids[hi])
                    hand.write_data_to_sim()
                sim.step(render=not args.headless)
                for hand in hands:
                    hand.update(dt)
                wire.update(dt)
                for i, sensor in enumerate(sensors):
                    sensor.update(dt, force_recompute=True)
                    self.forces[i] = float(torch.linalg.vector_norm(sensor.data.force_matrix_w).item())
                self.measure()
                for hand, diagnostic in zip(hands, self.joint_diagnostics):
                    failure = joint_safety_failure(
                        list(diagnostic['joints_rad'].values()),
                        hand.data.joint_pos_limits[0].detach().cpu().numpy(),
                        list(diagnostic['mimic_error_rad'].values()))
                    if failure:
                        self.task.failure = failure
                        break
                event = self.task.update(dt, **self.metrics)
                if self.task.failure or self.task.success:
                    break
            reward, terms = task_reward(phase=phase, action=self.command, reference=reference,
                previous=self.previous, dt=dt*decimation, success_event=event,
                failed=bool(self.task.failure), **self.metrics)
            self.return_ += reward
            self.previous = self.command.copy()
            self.log.append(dict(t=self.task.elapsed, phase=phase, reward=reward, terms=terms,
                clearance_m=self.clearance, object_bottom_m=self.bottom,
                pad_forces_n=self.forces.tolist(), slip_m=self.slip,
                lowest_link=self.lowest_link, hands=self.joint_diagnostics,
                command_rad=self.command.tolist(), approach_error_m=self.approach_error))
            self.log[-1]['descent_stop'] = self.descent_stop
            done = bool(self.task.failure or self.task.success)
            info = dict(is_success=self.task.success, failure=self.task.failure)
            if done:
                episode = dict(index=len(report['episodes']), success=self.task.success,
                    failure=self.task.failure, phases=sorted(set(r['phase'] for r in self.log)),
                    return_=self.return_, final_bottom_m=self.bottom,
                    min_clearance_m=min(r['clearance_m'] for r in self.log))
                report['episodes'].append(episode)
                (args.output/f"episode_{episode['index']:04d}.json").write_text(json.dumps(self.log, indent=2))
            return self.observe(), reward, done, False, info

    env = TabletopEnv()
    ppo_parameters = dict(learning_rate=3e-4, n_steps=128, batch_size=64, n_epochs=5,
        gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.005, vf_coef=.5,
        max_grad_norm=.5, target_kl=.02, seed=42, device='cpu',
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
    report.update(reward_parameters=REWARD_PARAMETERS, ppo_parameters=ppo_parameters,
                  collision_meshes=len(shapes), reference='ParallelJointMPC_reference_generator',
                  residual_scale_rad=.10, control_hz=60, physics_hz=args.physics_hz)
    (args.output/'configuration.json').write_text(json.dumps(report, indent=2))
    obs, _ = env.reset(seed=42)
    if not args.headless:
        sim.render()
    info = dict(is_success=False, failure=None)
    for _ in range(0 if args.preview_only else 1000):
        if args.approach_only and (env.task.phase != 'approach' or env.clearance < .01):
            info = dict(is_success=False, failure=env.task.failure)
            report['approach_stop'] = ('approach_complete' if env.task.phase != 'approach'
                                       else 'clearance_guard_10mm')
            break
        obs, reward, done, _, info = env.step(np.zeros(12))
        if done:
            break
    report['baseline'] = info
    if args.approach_only or args.preview_only:
        report['preview_only'] = args.preview_only
        report['approach_elapsed_s'] = env.task.elapsed
        report['final_clearance_m'] = env.clearance
        report['approach_error_m'] = env.approach_error
        (args.output/'approach.json').write_text(json.dumps(env.log, indent=2))
        if not args.headless:
            import time
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            sim.pause()
            sim.render()
            capture_viewport_to_file(get_active_viewport(), str(args.output/'approach.png'))
            deadline = time.monotonic() + max(3., args.hold_window_s)
            while time.monotonic() < deadline and app.is_running():
                app.update()
    elif not args.evaluate_only and args.timesteps and baseline_allows_training(info):
        model = PPO('MlpPolicy', env, verbose=1, **ppo_parameters)
        model.learn(total_timesteps=args.timesteps)
        model.save(str(args.output/'ppo_checkpoint'))
        report['ppo_updates'] = model._n_updates
        report['ppo_timesteps'] = model.num_timesteps
        obs, _ = env.reset(seed=43)
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            if done:
                break
        report['evaluation'] = info
        report['passed'] = bool(info['is_success'])
    else:
        if not args.evaluate_only and args.timesteps:
            report['training_skipped'] = 'baseline_not_qualified'
        report['passed'] = bool(info['is_success'])
except Exception:
    report['error'] = traceback.format_exc()
    print(report['error'], flush=True)
finally:
    (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    print('TABLETOP_PPO_RESULT', json.dumps(report), flush=True)
    app.close(wait_for_replicator=False, skip_cleanup=True)
