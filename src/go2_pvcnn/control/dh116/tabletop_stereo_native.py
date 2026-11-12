"""Opt-in bridge to the unchanged DH116 native task and launcher.

The original launcher creates the scene, controller, guards and reward. This
module substitutes only its final training callback in the current process.
It never patches a task/control/physics method or a historical artifact.
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch

from .tabletop_stereo_contract import (
    ROOT, STYLES, EpisodeIdentities, digest, file_sha256, require_opt_in, write_json_new, physical_initial_sha256,
    require_sha, validate_identity,
)
from .tabletop_surface_contract import validate_contract as validate_surface


def native_task_identity(spec):
    """Resolve authored task identity from the same object argument the launcher uses."""
    if not isinstance(spec, dict) or spec.get('version') != 'dh116-stereo-native-bootstrap-v1':
        raise ValueError('versioned native task bootstrap configuration required')
    arguments = spec.get('arguments')
    if not isinstance(arguments, list) or not all(isinstance(arg, str) for arg in arguments):
        raise ValueError('explicit immutable native launcher arguments required')
    object_kinds = []
    for index, token in enumerate(arguments):
        if token.startswith('--object-kind='):
            object_kinds.append(token.split('=', 1)[1])
        elif token == '--object-kind':
            object_kinds.append(arguments[index + 1] if index + 1 < len(arguments) else None)
    if len(object_kinds) > 1:
        raise ValueError('duplicate --object-kind arguments have ambiguous physical identity')

    kind = spec.get('task_kind', 'rigid_proxy')
    if kind == 'rigid_proxy':
        object_kind = object_kinds[0] if object_kinds else 'solid_cylinder'
        if object_kind not in ('solid_cylinder', 'hollow_straw'):
            raise ValueError('rigid_proxy requires solid_cylinder or hollow_straw --object-kind')
        if spec.get('physical_object_kind', object_kind) != object_kind:
            raise ValueError('physical_object_kind does not match actual --object-kind')
        source = 'rigid_body_sdk_pose_velocity'
    elif kind == 'deformable_rod':
        if spec.get('physical_object_kind') != 'cosserat_rod' or object_kinds:
            raise ValueError('deformable_rod requires cosserat_rod with no rigid --object-kind')
        object_kind = 'cosserat_rod'
        source = 'cosserat_solver_node_readback'
    else:
        raise ValueError('task_kind must be rigid_proxy or deformable_rod')

    identity = dict(version='dh116-stereo-native-task-identity-v1', task_kind=kind,
        physical_object_kind=object_kind, state_source=source, rod_state_available=False,
        task_profile_sha256=digest(spec))
    identity['task_identity_sha256'] = digest(identity)
    return identity


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
        object_kind = getattr(args, 'object_kind', 'solid_cylinder')
        identity = getattr(args, '_stereo_task_identity', None)
        if identity is None:
            identity = native_task_identity(dict(version='dh116-stereo-native-bootstrap-v1',
                arguments=['--object-kind', object_kind]))
            identity['task_profile_sha256'] = None
            identity['task_identity_sha256'] = digest({key: val for key, val in identity.items()
                if key != 'task_identity_sha256'})
        if (identity['task_kind'] != 'rigid_proxy' or identity['physical_object_kind'] != object_kind
                or identity['rod_state_available'] is not False):
            raise ValueError('actual TensorTask launcher object does not match rigid_proxy task identity')
        self._task_identity = deepcopy(identity)
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
        self.camera = None
        self._teacher_checkpoint = None
        self._teacher_policy_version = None
        self._teacher_epoch_by_episode = {}
        self._teacher_pre_action = None
        self._teacher_actions = None
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
            force_qualification=None, hardware_feedback_qualification=None, **self._task_binding()))
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

    @property
    def task_identity(self):
        return deepcopy(getattr(self, '_task_identity', None))

    def _task_binding(self):
        identity = self.task_identity
        if identity is None:
            return {}  # Compatibility for schema-only bridges that bypass __init__.
        return dict(task_identity=identity, task_identity_sha256=identity['task_identity_sha256'])

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

    def bind_stereo_candidate_frame(self, packet, identities):
        """Bind the already sampled stereo frame to the next Teacher-owned step."""
        if (self._in_step or self.execution_owner != 'teacher'
                or getattr(self, '_teacher_checkpoint', None) is not None):
            raise ValueError('candidate stereo binding requires the independent Teacher collector')
        calibration = require_sha(self.camera.calibration.sha256, 'native calibration SHA')
        checkpoint = require_sha(self.checkpoint_sha256, 'candidate Teacher checkpoint SHA')
        frame = packet.get('frame_id')
        if (calibration == '0'*64 or checkpoint == '0'*64
                or packet.get('calibration_sha256') != calibration
                or type(frame) is not int or frame < 0 or frame != self.camera.last_frame
                or packet.get('identity') != identities or len(identities) != self.n):
            raise ValueError('candidate camera packet identity is absent or stale')
        for row, actual in zip(identities,self.pre_action_identities(calibration)):
            validate_identity(row)
            if (row['frame_id'] != frame or row['simulation_timestamp'] != packet.get('simulation_time')
                    or any(row.get(key) != value for key,value in actual.items())):
                raise ValueError('candidate camera frame does not match the live control boundary')
        self._stereo_candidate_pre_action = deepcopy(identities)

    def bind_camera(self, camera):
        """Bind the real camera object; configuration hashes are not readbacks."""
        calibration = require_sha(getattr(getattr(camera, 'calibration', None), 'sha256', None),
            'native calibration SHA256')
        if calibration == '0'*64 or not callable(getattr(camera, 'sample', None)):
            raise ValueError('nonzero native camera calibration and synchronized reader required')
        if self._in_step:
            raise ValueError('camera binding requires a control boundary')
        self.camera = camera
        self._teacher_pre_action = None
        self._teacher_actions = None

    def bind_teacher_checkpoint(self, checkpoint, *, policy_version=None):
        """Hash saved policy bytes before any physical Teacher action."""
        if policy_version is not None and (type(policy_version) is not int or policy_version < 0):
            raise ValueError('nonnegative integer Teacher policy version required')
        if getattr(self, '_teacher_pre_action', None) is not None:
            raise ValueError('cannot change Teacher policy epoch during a pending pre-action frame')
        previous_version = getattr(self, '_teacher_policy_version', None)
        if previous_version is not None and (policy_version is None or policy_version <= previous_version):
            raise ValueError('Teacher policy epoch must advance at the next clean control boundary')
        if (getattr(self, '_teacher_checkpoint', None) is None
                and any(self.identities.uid(i) in self._headers_written for i in range(self.n))):
            raise ValueError('Teacher checkpoint must be bound before episode physical evidence starts')
        self.bind_camera(getattr(self, 'camera', None))
        path = Path(checkpoint).resolve(strict=True)
        sha = require_sha(file_sha256(path), 'native Teacher checkpoint SHA256')
        if sha == '0'*64:
            raise ValueError('nonzero actual Teacher checkpoint SHA256 required')
        self.set_execution_context(owner='teacher', checkpoint_sha256=sha)
        self.set_action_owners(['teacher']*self.n)
        self._teacher_checkpoint = path
        self._teacher_policy_version = policy_version
        if not hasattr(self, '_teacher_epoch_by_episode'):
            self._teacher_epoch_by_episode = {}

    def _validate_teacher_binding(self):
        if (self.execution_owner != 'teacher' or self.action_owners != ['teacher']*self.n
                or file_sha256(self._teacher_checkpoint) != self.checkpoint_sha256):
            raise ValueError('Teacher checkpoint/execution identity changed after binding')
        calibration = require_sha(self.camera.calibration.sha256, 'native calibration SHA256')
        if calibration == '0'*64:
            raise ValueError('nonzero actual native calibration required')
        return calibration

    def teacher_pre_action_identities(self):
        """Use the camera-returned frame identity and compare every live clock."""
        if getattr(self, '_teacher_checkpoint', None) is None:
            raise ValueError('actual Teacher checkpoint identity must be bound before capture')
        calibration = self._validate_teacher_binding()
        requested = self.pre_action_identities(calibration)
        previous_frame = self.camera.last_frame
        packet = self.camera.sample(identity=deepcopy(requested))
        actual = packet.get('identity')
        frame = packet.get('frame_id')
        if (not isinstance(actual, list) or len(actual) != self.n
                or type(frame) is not int or frame < 0 or frame <= previous_frame
                or packet.get('calibration_sha256') != calibration):
            raise ValueError('actual camera frame/calibration identity required')
        current = self.pre_action_identities(calibration)
        for i, row in enumerate(actual):
            validate_identity(row)
            if (row['frame_id'] != frame or packet.get('simulation_time') != row['simulation_timestamp']
                    or any(row.get(key) != value for key, value in requested[i].items())
                    or current[i] != requested[i]):
                raise ValueError('camera identity must be exactly aligned with live Teacher pre-action state')
            row['checkpoint_sha256'] = self.checkpoint_sha256
            if self._teacher_policy_version is not None:
                row['policy_version'] = self._teacher_policy_version
        self._teacher_pre_action = deepcopy(actual)
        self._teacher_actions = None
        return deepcopy(actual)

    def set_teacher_action_diagnostics(self, *, sampled_action, mean_action, issued_action,
                                       policy_audit=None, diagnostic_mode=None):
        if getattr(self, '_teacher_pre_action', None) is None:
            raise ValueError('aligned Teacher pre-action identity required before diagnostics')
        values = dict(teacher_sampled_action=sampled_action, teacher_mean_action=mean_action,
            actually_issued_action=issued_action)
        for value in values.values():
            if (value is not None and (not isinstance(value, torch.Tensor) or value.shape != (self.n, 24)
                    or not bool(torch.isfinite(value).all()) or bool((value.abs() > 1).any()))):
                raise ValueError('finite normalized Teacher action diagnostics required')
        if mean_action is None or issued_action is None:
            raise ValueError('explicit Teacher mean and actually issued actions required')
        if diagnostic_mode not in (None, 'teacher_sampled', 'teacher_mean'):
            raise ValueError('unknown Teacher diagnostic mode')
        if diagnostic_mode is not None and policy_audit is None:
            raise ValueError('paired diagnostic mode requires the frozen policy distribution')
        if diagnostic_mode == 'teacher_sampled' and (sampled_action is None
                or not torch.equal(sampled_action, issued_action)):
            raise ValueError('sampled diagnostic requires the actually issued sample')
        if diagnostic_mode == 'teacher_mean' and (sampled_action is not None
                or not torch.equal(mean_action, issued_action)):
            raise ValueError('mean diagnostic requires the actually issued mean and no sample')
        captured_policy = None
        if policy_audit is not None:
            fields = {'latent_mean', 'latent_std', 'latent_actions'}
            deterministic = diagnostic_mode == 'teacher_mean'
            if (not isinstance(policy_audit, dict) or set(policy_audit) != fields
                    or any(not isinstance(policy_audit[key], torch.Tensor)
                        or policy_audit[key].shape != (self.n, 24)
                        or not bool(torch.isfinite(policy_audit[key]).all())
                        for key in (fields-{'latent_actions'} if deterministic else fields))
                    or not bool((policy_audit['latent_std'] > 0).all())
                    or (deterministic and policy_audit['latent_actions'] is not None)
                    or (not deterministic and (sampled_action is None
                        or not torch.allclose(policy_audit['latent_actions'].tanh(), sampled_action)))
                    or not torch.allclose(policy_audit['latent_mean'].tanh(), mean_action)):
                raise ValueError('joint action audit requires actual aligned 24D policy distribution and action')
            captured_policy = {key: None if value is None else value.detach().clone()
                for key, value in policy_audit.items()}
        self._teacher_policy_audit = captured_policy
        self._teacher_action_diagnostic_mode = diagnostic_mode
        self._teacher_actions = {key: None if value is None else value.detach().clone()
            for key, value in values.items()}

    def _teacher_capture_identity(self, i, packet):
        """Verify telemetry really belongs to this action before writing raw rows."""
        if not self._in_step or self._teacher_actions is None:
            raise ValueError('Teacher raw identity requires an active aligned action')
        row = self._teacher_pre_action[i]
        if (row['episode_uid'] != self.identities.uid(i) or row['reset_id'] != self.identities.reset_ids[i]
                or row['policy_tick'] != self.policy_tick
                or row['checkpoint_sha256'] != self.checkpoint_sha256
                or row['calibration_sha256'] != self.camera.calibration.sha256
                or int(self.device_world._physics_steps) <= row['physics_tick_start']
                or float(self.world.sim.current_time) <= row['simulation_timestamp']
                or not torch.equal(packet['action'][i], self._teacher_actions['actually_issued_action'][i])):
            raise ValueError('native raw sample identity/action differs from bound Teacher execution')
        result = dict(calibration_sha256=row['calibration_sha256'], frame_id=row['frame_id'],
            checkpoint_sha256=row['checkpoint_sha256'], pre_action_identity=deepcopy(row))
        if self._teacher_policy_version is not None:
            if row.get('policy_version') != self._teacher_policy_version:
                raise ValueError('native raw sample differs from bound Teacher policy epoch')
            result['policy_version'] = self._teacher_policy_version
        return result

    def _record_teacher_policy_epoch(self, i, identity):
        """Append one epoch boundary for a real first use without rewriting episode_start."""
        version = identity.get('policy_version')
        if version is None:
            return
        before = identity['pre_action_identity']
        uid, sha = before['episode_uid'], identity['checkpoint_sha256']
        previous = self._teacher_epoch_by_episode.get(uid)
        if previous == (version, sha):
            return
        event = dict(kind='teacher_policy_epoch_start', run_uuid=self.identities.run_uuid, env_id=i,
            episode_uid=uid, reset_id=before['reset_id'], policy_tick=before['policy_tick'],
            first_physics_step=int(self.device_world._physics_steps),
            previous_policy_version=previous[0] if previous is not None else None,
            previous_checkpoint_sha256=previous[1] if previous is not None else None,
            **identity)
        path = self.output / 'teacher_policy_epochs.jsonl'
        with path.open('a' if self._teacher_epoch_by_episode else 'x', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False, separators=(',', ':')) + '\n')
        self._teacher_epoch_by_episode[uid] = (version, sha)

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
            grasp_style=self.grasp_styles[i], **self._task_binding())
            for i in range(self.n)]

    def _capture(self, packet):
        w, task = self.device_world, self.task
        tracker = task.surface_lift_success
        live = packet['live'].nonzero().flatten().tolist()
        for i in live:
            key = self.identities.uid(i)
            teacher_identity = (self._teacher_capture_identity(i, packet)
                if getattr(self, '_teacher_checkpoint', None) is not None else {})
            candidate = getattr(self, '_stereo_candidate_pre_action', None)
            candidate_identity = {}
            if candidate is not None:
                before = candidate[i]
                if (before['episode_uid'] != key or before['policy_tick'] != self.policy_tick
                        or self.execution_owner != 'teacher' or self.action_owners[i] != 'teacher'):
                    raise ValueError('native candidate execution no longer matches the bound stereo frame')
                candidate_identity = dict(frame_id=before['frame_id'],
                    calibration_sha256=before['calibration_sha256'], pre_action_identity=deepcopy(before))
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
                    runtime=self.runtime_descriptor, policy_tick=self.policy_tick)
                header.update(self._task_binding())
                header.update(teacher_identity)
                header.update(candidate_identity)
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
            row.update(self._task_binding())
            row.update(candidate_identity)
            if teacher_identity:
                self._record_teacher_policy_epoch(i, teacher_identity)
                row.update(teacher_identity)
                row.update({name: None if value is None else value[i]
                    for name, value in self._teacher_actions.items()})
                sdk = [getattr(hand.data, 'joint_pos_target', None) for hand in w.hands]
                sdk = (torch.cat([value[:, ids] for value, ids in zip(sdk, w.w.ids)], -1)[i]
                    if all(value is not None for value in sdk) else None)
                row.update(applied_action=None,
                    applied_action_source='normalized_applied_action_unavailable;native_controller_and_SDK_readback',
                    applied_targets=dict(controller_joint_command_rad=task.command[i],
                        actual_sdk_joint_target_rad=sdk,
                        source='pre_reset_physics_sample;not_normalized_applied_action'))
                # Palm projection has actual before/after telemetry. The joint
                # controller provides accepted targets and a pressure mask, but
                # no unconstrained/projected joint pair at this sample boundary.
                diagnostics = getattr(w, 'last_action_diagnostics', None)
                if (isinstance(diagnostics, dict)
                        and diagnostics.get('physics_step') == row['physics_step']
                        and diagnostics.get('timestamp_s') == row['timestamp']
                        and all(name in diagnostics for name in ('pre_projection_palm_target_m',
                            'projected_palm_target_m', 'projection_reasons'))):
                    row['palm_safety_projection'] = {
                        name: diagnostics[name][i] for name in ('pre_projection_palm_target_m',
                            'projected_palm_target_m', 'projection_reasons')
                    }
                else:
                    row['palm_safety_projection'] = None
                pressure_limited = getattr(task, 'pressure_limited', None)
                row['joint_pressure_limited'] = None if pressure_limited is None else pressure_limited[i]
                row['joint_safety_projection'] = None
                policy_audit = getattr(self, '_teacher_policy_audit', None)
                if policy_audit is not None:
                    from .tabletop_privileged_action_audit import TeacherJointActionAudit
                    if not hasattr(self, '_teacher_joint_audit'):
                        self._teacher_joint_audit = TeacherJointActionAudit(w)
                    row['action_audit'] = self._teacher_joint_audit.capture(
                        w, task, self.env, i, row, policy_audit,
                        diagnostic_mode=getattr(self, '_teacher_action_diagnostic_mode', None))
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
            if teacher_identity:
                self._teacher_step_rows[i] = row
            if row['terminal']:
                self._terminal_rows[key] = row
        self._raw.flush()

    def step(self, issued_actions):
        if (issued_actions.shape != (self.n, 24) or not torch.isfinite(issued_actions).all()
                or bool((issued_actions.abs() > 1).any())):
            raise ValueError('finite normalized canonical issued actions required')
        strict_teacher = getattr(self, '_teacher_checkpoint', None) is not None
        if strict_teacher:
            calibration = self._validate_teacher_binding()
            if self._teacher_pre_action is None:
                raise ValueError('aligned Teacher pre-action identity required before physical action')
            current = self.pre_action_identities(calibration)
            if any(any(row.get(key) != value for key, value in now.items())
                    for row, now in zip(self._teacher_pre_action, current)):
                raise ValueError('Teacher pre-action identity became stale before physical action')
            if (self._teacher_actions is None
                    or not torch.equal(issued_actions, self._teacher_actions['actually_issued_action'])):
                raise ValueError('Teacher action diagnostics must match actually issued action')
            before = deepcopy(self._teacher_pre_action)
            self._teacher_step_rows = {}
        else:
            camera = getattr(self, 'camera', None)
            calibration = getattr(getattr(camera, 'calibration', None), 'sha256', None)
            before = self.pre_action_identities(calibration)
        self._in_step = True
        try:
            obs, reward, done, extra = self.env.step(issued_actions)
        finally:
            self._in_step = False
            self._stereo_candidate_pre_action = None
        if strict_teacher:
            if len(self._teacher_step_rows) != self.n:
                raise RuntimeError('Teacher execution lacks actual aligned native raw identity samples')
            self._teacher_pre_action = None
            self._teacher_actions = None
            self._teacher_policy_audit = None
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
    identity = native_task_identity(spec)
    if identity['task_kind'] == 'deformable_rod':
        raise RuntimeError('deformable_rod requires an independent Cosserat/DeformX launcher and verified '
            'node state provider; this TensorTask launcher only implements rigid_proxy')
    arguments = spec['arguments']
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
        args._stereo_task_identity = deepcopy(identity)
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
