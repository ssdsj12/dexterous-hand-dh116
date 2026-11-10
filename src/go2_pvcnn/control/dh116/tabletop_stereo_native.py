"""Opt-in bridge to the unchanged DH116 native task and launcher.

The original launcher creates the scene, controller, guards and reward. This
module substitutes only its final training callback in the current process.
It never patches a task/control/physics method or a historical artifact.
"""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch

from .tabletop_stereo_contract import (
    ROOT, STYLES, EpisodeIdentities, file_sha256, require_opt_in, write_json_new, physical_initial_sha256,
)
from .tabletop_surface_contract import validate_contract as validate_surface


@contextmanager
def native_instance_lock(path='/tmp/dh116-isaac-single-instance.lock'):
    """Linux advisory lock plus conservative detection of existing native jobs."""
    if sys.platform != 'linux':
        raise RuntimeError('native Isaac execution requires the existing Linux server environment')
    import fcntl
    current = os.getpid()
    names = {'train_dh116_tabletop_gpu_ppo.py', 'collect_dh116_hand_only_mpc_isaac.py', 'kit', 'isaacsim'}
    for candidate in Path('/proc').iterdir():
        if not candidate.name.isdigit() or int(candidate.name) == current:
            continue
        try:
            command = (candidate / 'cmdline').read_bytes().split(b'\0')
        except (OSError, PermissionError):
            continue
        if any(Path(part.decode(errors='replace')).name in names for part in command if part):
            raise RuntimeError(f'existing Isaac/native task process {candidate.name}; refusing a second instance')
    with Path(path).open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another DH116 native experiment owns the instance lock') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    return value


class NativeTensorBridge:
    """Read actual named links/encoders and preserve terminal samples before reset."""
    evidence_scope = 'native_runtime_capture_unqualified'
    def __init__(self, world, device_world, task, env, args, contract, *, output):
        validate_surface(contract)
        if task.observe().shape != (env.n, 371) or getattr(task, 'action_size', None) != 24:
            raise ValueError('exact audited 371D/24D native task required')
        if getattr(device_world, 'decimation', None) != 8 or abs(float(device_world.dt) - 1/480) > 1e-12:
            raise ValueError('unchanged 480Hz physics / 60Hz control required')
        self.world, self.device_world, self.task, self.env = world, device_world, task, env
        self.args, self.contract, self.output = args, contract, Path(output)
        self.n, self.device = env.n, torch.device(env.device)
        self.identities = EpisodeIdentities(self.n)
        self.policy_tick = 0
        self.previous_issued = torch.zeros(self.n, 24, device=self.device)
        self._terminal_rows, self._last_live_rows = {}, {}
        self._initial_conditions, self._initial_sha, self._initial_clock = {}, {}, {}
        self._initial_policy_tick = {}
        self._in_step = False
        self._cold_trials, self._trial_reset_counts = None, [0]*self.n
        self._seen_trial_ids = set()
        self.runtime_descriptor = None
        self._headers_written = set()
        self.execution_owner = 'teacher'
        self.action_owners = ['teacher'] * self.n
        self.checkpoint_sha256 = None
        styles = getattr(args, '_stereo_grasp_styles', ['pinch_thumb_index'])
        if len(styles) == 2:
            if not getattr(args, 'mixed_grasp_initialization', None) or self.n % 2:
                raise ValueError('bilateral style labels require the actual mixed reset initializer')
            self.grasp_styles = [styles[0]]*(self.n//2) + [styles[1]]*(self.n//2)
        elif len(styles) == 1:
            self.grasp_styles = styles * self.n
        else:
            raise ValueError('one real reset style or an explicit two-style mixed reset required')
        self._raw = (self.output / 'native_physics.jsonl').open('x', encoding='utf-8')
        self._palm_ids = []
        for hand in device_world.hands:
            # DH116's measured base_link is the explicitly declared palm FK
            # reference when no separate palm_link is authored. Never a target.
            name = 'palm_link' if 'palm_link' in hand.body_names else 'base_link'
            if name not in hand.body_names:
                raise ValueError('native named palm/base link readback unavailable')
            self._palm_ids.append((name, hand.body_names.index(name)))
        write_json_new(self.output / 'native_task_contract.json', dict(
            inherited_contract=contract, inherited_launcher_sha256=file_sha256(
                ROOT / 'scripts/train_dh116_tabletop_gpu_ppo.py'),
            palm_fk_links=[name for name, _ in self._palm_ids],
            observation_source='unchanged_TensorTask.observe',
            physics_evidence='runtime_capture_not_automatic_qualification',
            force_qualification=None, hardware_feedback_qualification=None))
        from .tabletop_evaluation_telemetry import EvaluationTelemetry
        owner = self

        class StereoEvidenceObserver(EvaluationTelemetry):
            def reset(observer, ids, measurement):
                super().reset(ids, measurement)
                owner._physical_reset(ids, measurement)

        # Independent diagnostic subclass only: no Task/World/Env control patch.
        env.evaluation_telemetry = StereoEvidenceObserver(self.n, self.device, self._capture)
        self.critic = self.reset()

    def close(self):
        self._raw.flush()
        self._raw.close()

    def reset(self):
        self.previous_issued.zero_()
        self.critic = self.env.reset().detach().float().clone()
        if self.critic.shape != (self.n, 371) or not torch.isfinite(self.critic).all():
            raise RuntimeError('invalid native reset observation')
        return self.critic

    def _physical_reset(self, ids, measurement):
        """Observe the actual reset callback, before another physical substep."""
        indices = ids.detach().cpu().tolist()
        self.identities.reset(indices)
        w = self.device_world
        palms = self.palm_poses()
        for i in indices:
            uid = self.identities.uid(i)
            self._trial_reset_counts[i] += 1
            initial = _json_safe(dict(coordinate_frame='world', environment_origin=w.origins[i],
                object_pose=torch.cat((w.wp[i], w.wq[i])), object_velocity=w.wv[i],
                palm_pose=palms[i], joint_position=measurement['q'][i],
                joint_velocity=measurement['qd'][i]))
            self._initial_conditions[uid] = initial
            self._initial_sha[uid] = physical_initial_sha256(initial)
            self._initial_clock[uid] = (int(w._physics_steps), float(self.world.sim.current_time))
            self._initial_policy_tick[uid] = self.policy_tick + int(self._in_step)

    def bind_cold_trials(self, trials):
        """Bind already-audited reset-buffer trials; this method moves no body."""
        if self._in_step or not isinstance(trials, list) or len(trials) != self.n:
            raise ValueError('one declared cold trial per slot at a control boundary required')
        names = [row.get('trial_id') if isinstance(row, dict) else None for row in trials]
        if (any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names) or set(names) & self._seen_trial_ids
                or any(row.get('grasp_style') not in STYLES for row in trials)):
            raise ValueError('unique unvisited cold trial IDs and real grasp style labels required')
        self._cold_trials = [dict(row) for row in trials]
        self._trial_reset_counts = [0]*self.n
        self._seen_trial_ids.update(names)
        self.grasp_styles = [row['grasp_style'] for row in trials]

    def set_runtime_descriptor(self, descriptor):
        """Freeze evidence about the actual driver before this episode starts."""
        if not isinstance(descriptor, dict):
            raise ValueError('explicit runtime descriptor required')
        value = json.loads(json.dumps(descriptor, allow_nan=False))
        current_uids = {self.identities.uid(i) for i in range(self.n)}
        if self.runtime_descriptor != value and current_uids & self._headers_written:
            raise ValueError('runtime descriptor cannot change after an episode has physical evidence')
        self.runtime_descriptor = value

    def set_execution_context(self, *, owner, checkpoint_sha256=None):
        if owner not in ('teacher', 'student', 'dagger', 'bc', 'mpc', 'diagnostic_zero'):
            raise ValueError('explicit independent execution category required')
        self.execution_owner = owner
        self.checkpoint_sha256 = checkpoint_sha256

    def set_action_owners(self, owners):
        if len(owners) != self.n or any(owner not in ('teacher', 'student', 'diagnostic_zero') for owner in owners):
            raise ValueError('one explicit complete action owner per environment required')
        self.action_owners = list(owners)

    def palm_poses(self):
        return torch.stack([hand.data.body_link_pose_w[:, i] for hand, (_, i) in
            zip(self.device_world.hands, self._palm_ids)], 1).detach().float().clone()

    def proprioception(self):
        w = self.device_world
        q = torch.cat([hand.data.joint_pos[:, ids] for hand, ids in zip(w.hands, w.w.ids)], -1)
        qd = torch.cat([hand.data.joint_vel[:, ids] for hand, ids in zip(w.hands, w.w.ids)], -1)
        result = torch.cat((q, qd, self.palm_poses().flatten(1), self.previous_issued), -1).float()
        if result.shape != (self.n, 62) or not torch.isfinite(result).all():
            raise RuntimeError('finite actual encoder/FK/issued-action proprioception required')
        return result

    def pre_action_identities(self, calibration_sha256):
        tick = int(self.device_world._physics_steps)
        timestamp = float(self.world.sim.current_time)
        return [dict(run_uuid=self.identities.run_uuid, env_id=i,
            episode_uid=self.identities.uid(i), reset_id=self.identities.reset_ids[i],
            policy_tick=self.policy_tick, physics_tick_start=tick, physics_tick_end=tick,
            simulation_timestamp=timestamp, calibration_sha256=calibration_sha256,
            physical_initial_condition_sha256=self._initial_sha[self.identities.uid(i)],
            source_episode_group=(self._cold_trials[i]['trial_id'] if self._cold_trials else self.identities.uid(i)),
            episode_policy_tick=self.policy_tick-self._initial_policy_tick[self.identities.uid(i)],
            trial_episode_index=self._trial_reset_counts[i],
            trial_id=self._cold_trials[i]['trial_id'] if self._cold_trials else None,
            grasp_style=self.grasp_styles[i])
            for i in range(self.n)]

    def _capture(self, packet):
        w, task = self.device_world, self.task
        tracker = task.surface_lift_success
        live = packet['live'].nonzero().flatten().tolist()
        for i in live:
            key = self.identities.uid(i)
            reference = torch.as_tensor(task.surface_reference_bottom_m,
                device=task.m['bottom'].device).expand(self.n)[i]
            if hasattr(self, '_headers_written') and key not in self._headers_written:
                initial_tick, initial_time = self._initial_clock[key]
                header = dict(kind='episode_start', run_uuid=self.identities.run_uuid,
                    env_id=i, episode_uid=key, reset_id=self.identities.reset_ids[i],
                    source_scope='unqualified', owner=self.execution_owner,
                    grasp_style=self.grasp_styles[i], checkpoint_sha256=self.checkpoint_sha256,
                    source_episode_group=self._cold_trials[i]['trial_id'] if self._cold_trials else key,
                    trial_id=self._cold_trials[i]['trial_id'] if self._cold_trials else None,
                    physical_initial_condition=self._initial_conditions[key],
                    initial_measurement_valid=True, physical_initial_condition_source='actual_sdk_readback',
                    initial_physics_step=initial_tick, initial_timestamp=initial_time,
                    first_episode=self._trial_reset_counts[i] == 1, resumed=False, recovered=False,
                    runtime=self.runtime_descriptor)
                self._raw.write(json.dumps(header, allow_nan=False, separators=(',', ':')) + '\n')
                self._headers_written.add(key)
            row = dict(run_uuid=self.identities.run_uuid, env_id=i, episode_uid=key,
                reset_id=self.identities.reset_ids[i], policy_tick=self.policy_tick,
                physics_step=int(w._physics_steps), timestamp=float(self.world.sim.current_time),
                evidence_scope='native_runtime_capture_unqualified',
                native_success=bool(packet['native_success'][i]), terminal=bool(packet['done'][i]),
                failure_code=int(task.failure[i]),
                native_metrics={k: (v[i] if isinstance(v, torch.Tensor) and v.ndim else v)
                    for k, v in packet.items() if k not in ('live', 'action')},
                actor_issued_action=packet['action'][i],
                formal_drift_m=tracker.drift_m[i], formal_hold_s=tracker.duration[i],
                formal_window_good=tracker.window_good[i],
                object_net_lift_m=task.m['bottom'][i] - reference,
                actual_palm_pose=self.palm_poses()[i],
                active_joint_position=task.m['q'][i], active_joint_velocity=task.m['qd'][i],
                target_joint_position=task.command[i],
                effective_pad_loads_n=task.m.get('effective_pad_loads_n', [None]*self.n)[i],
                rejected_contact_loads_n=task.m.get('rejected_contact_loads_n', [None]*self.n)[i],
                force_qualification=None, support_qualification=None)
            row.update(kind='sample', source_scope='unqualified',
                owner=getattr(self, 'execution_owner', 'teacher'),
                action_owner=getattr(self, 'action_owners', ['teacher']*self.n)[i],
                grasp_style=getattr(self, 'grasp_styles', ['pinch_thumb_index']*self.n)[i],
                measurement_valid=bool(packet['measurement_valid'][i]),
                contact_valid=bool(packet.get('contact_valid', torch.zeros(self.n, dtype=torch.bool))[i]),
                bilateral_opposed_contact=bool(packet.get('bilateral_opposed_contact', torch.zeros(self.n, dtype=torch.bool))[i]),
                formal_tracker_source='surface_lift_success_30mm_5s_0.5mm',
                formal_tracker_valid=bool(getattr(tracker, '_physical_window_valid',
                    torch.zeros(self.n, dtype=torch.bool))[i]),
                formal_window_completed=abs(float(getattr(tracker, '_elapsed', 1.))) < 1e-10,
                formal_window_good=bool(tracker.window_good[i]),
                formal_anchor_latched=bool(getattr(tracker, '_latched',
                    torch.zeros(self.n, dtype=torch.bool))[i]),
                formal_drift_source='last_completed_window_max_over_raw_pose_points',
                force_source_qualified=False, support_source_qualified=False,
                external_support_force_n=None, external_upward_force_n=None, attachment_count=None,
                checkpoint_sha256=getattr(self, 'checkpoint_sha256', None))
            row['runtime'] = getattr(self, 'runtime_descriptor', None)
            formal_keys = ('object_relative', 'object_quat', 'wrist_pose', 'effective_pad_loads_n',
                'effective_pad_vectors_w', 'effective_pad_depth_m', 'rejected_contact_loads_n', 'object_velocity')
            if all(name in task.m for name in formal_keys):
                wrist = task.m['wrist_pose'][i]
                row.update(formal_object_pose_w=torch.cat((task.m['object_relative'][i]
                        + wrist[..., :3].mean(0), task.m['object_quat'][i])),
                    formal_wrist_pose_w=wrist,
                    formal_pad_loads_n=task.m['effective_pad_loads_n'][i],
                    formal_pad_vectors_w=task.m['effective_pad_vectors_w'][i],
                    formal_pad_depth_m=task.m['effective_pad_depth_m'][i],
                    formal_rejected_load_n=task.m['rejected_contact_loads_n'][i].sum(),
                    formal_object_bottom_m=task.m['bottom'][i],
                    formal_object_velocity=task.m['object_velocity'][i],
                    formal_reference_bottom_m=reference,
                    formal_task_active=bool(packet['measurement_valid'][i]) and int(task.failure[i]) == 0)
            row = _json_safe(row)
            if key in self._terminal_rows:
                raise RuntimeError('live sample after terminal for unchanged episode identity')
            self._raw.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
            self._last_live_rows[key] = row
            if row['terminal']:
                self._terminal_rows[key] = row
        self._raw.flush()

    def step(self, issued_actions):
        if (issued_actions.shape != (self.n, 24) or not torch.isfinite(issued_actions).all()
                or bool((issued_actions.abs() > 1).any())):
            raise ValueError('finite normalized canonical issued actions required')
        before = self.pre_action_identities('0'*64)
        self._in_step = True
        try:
            obs, reward, done, extra = self.env.step(issued_actions)
        finally:
            self._in_step = False
        for i in done.nonzero().flatten().tolist():
            if before[i]['episode_uid'] not in self._terminal_rows:
                raise RuntimeError('terminal lacks a preserved pre-reset 480Hz native sample')
        self.previous_issued.copy_(issued_actions)
        self.previous_issued[done] = 0
        self.policy_tick += 1
        self.critic = obs.detach().float().clone()
        return self.critic, reward.detach().float(), done, extra


def launch_native(callback, *, native_arguments, output, num_envs=2, seed=116,
                  rollout_steps=32, enabled=False):
    """Execute via scripts/python.sh on server; callback owns only new models."""
    require_opt_in(enabled)
    spec = json.loads(Path(native_arguments).read_text(encoding='utf-8'))
    if spec.get('version') != 'dh116-stereo-native-bootstrap-v1':
        raise ValueError('versioned native task bootstrap configuration required')
    arguments = spec.get('arguments')
    if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
        raise ValueError('explicit immutable native launcher arguments required')
    forbidden = {'--resume', '--initialize-from', '--vision-teacher-source', '--surface-mpc',
        '--vision-evaluation-checkpoint', '--vision-m2-replay', '--surface-corrections'}
    if any(x.split('=', 1)[0] in forbidden for x in arguments):
        raise ValueError('native bootstrap cannot modify/replay historical models or enable MPC')
    overrides = {'--num-envs', '--seed', '--rollout-steps', '--output', '--max-iterations'}
    if any(x.split('=', 1)[0] in overrides for x in arguments):
        raise ValueError('native bootstrap dynamic options must be set by the new CLI')
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    result = {}
    callback_completed = False

    def isolated_callback(world, device_world, task, env, args, contract, **unused):
        nonlocal callback_completed
        args._stereo_grasp_styles = spec.get('grasp_styles', ['pinch_thumb_index'])
        bridge = NativeTensorBridge(world, device_world, task, env, args, contract, output=output)
        try:
            value = callback(bridge)
            result.update(value)
        finally:
            bridge.close()
        callback_completed = True
        return value

    with native_instance_lock():
        # Import after the native wrapper has established the actual environment.
        module = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_training')
        original, old_argv, old_cwd = module.run_vision_training, sys.argv[:], Path.cwd()
        module.run_vision_training = isolated_callback
        try:
            os.chdir(ROOT)
            sys.argv = [str(ROOT / 'scripts/train_dh116_tabletop_gpu_ppo.py'), *arguments,
                '--num-envs', str(num_envs), '--seed', str(seed), '--rollout-steps', str(rollout_steps),
                '--max-iterations', '1', '--output', str(output / 'scene')]
            try:
                runpy.run_path(sys.argv[0], run_name='__main__')
            except SystemExit as error:
                # The unchanged launcher exits after its cleanup/publication.
                # Noninteger exit payloads represent failures even when == 0.
                if error.code is not None and (type(error.code) is not int or error.code != 0):
                    raise
            if not callback_completed:
                raise RuntimeError('native training callback was not completed')
        finally:
            module.run_vision_training, sys.argv = original, old_argv
            os.chdir(old_cwd)
    return result
