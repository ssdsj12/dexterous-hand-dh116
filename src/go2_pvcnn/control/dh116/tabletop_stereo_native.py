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
    ROOT, EpisodeIdentities, file_sha256, require_opt_in, write_json_new,
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
        env.enable_evaluation_telemetry(self._capture)
        self.critic = self.reset()

    def close(self):
        self._raw.flush()
        self._raw.close()

    def reset(self):
        self.identities.reset(list(range(self.n)))
        self.previous_issued.zero_()
        self._terminal_rows.clear()
        self.critic = self.env.reset().detach().float().clone()
        if self.critic.shape != (self.n, 371) or not torch.isfinite(self.critic).all():
            raise RuntimeError('invalid native reset observation')
        return self.critic

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
            simulation_timestamp=timestamp, calibration_sha256=calibration_sha256)
            for i in range(self.n)]

    def _capture(self, packet):
        w, task = self.device_world, self.task
        tracker = task.surface_lift_success
        live = packet['live'].nonzero().flatten().tolist()
        for i in live:
            key = self.identities.uid(i)
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
                object_net_lift_m=task.m['bottom'][i] - task.surface_reference_bottom_m[i],
                actual_palm_pose=self.palm_poses()[i],
                active_joint_position=task.m['q'][i], active_joint_velocity=task.m['qd'][i],
                target_joint_position=task.command[i],
                effective_pad_loads_n=task.m.get('effective_pad_loads_n', [None]*self.n)[i],
                rejected_contact_loads_n=task.m.get('rejected_contact_loads_n', [None]*self.n)[i],
                force_qualification=None, support_qualification=None)
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
        obs, reward, done, extra = self.env.step(issued_actions)
        for i in done.nonzero().flatten().tolist():
            if before[i]['episode_uid'] not in self._terminal_rows:
                raise RuntimeError('terminal lacks a preserved pre-reset 480Hz native sample')
        self.previous_issued.copy_(issued_actions)
        self.previous_issued[done] = 0
        self.identities.reset(done.nonzero().flatten().tolist())
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

    def isolated_callback(world, device_world, task, env, args, contract, **unused):
        bridge = NativeTensorBridge(world, device_world, task, env, args, contract, output=output)
        try:
            value = callback(bridge)
            result.update(value)
            return value
        finally:
            bridge.close()

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
            runpy.run_path(sys.argv[0], run_name='__main__')
            if not result:
                raise RuntimeError('native training callback was not reached')
        finally:
            module.run_vision_training, sys.argv = original, old_argv
            os.chdir(old_cwd)
    return result
