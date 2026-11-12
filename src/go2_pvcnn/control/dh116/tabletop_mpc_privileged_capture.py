"""Diagnostic join of actual canonical TensorEnv observations and executed MPC intervals."""

from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from .tabletop_surface_contract import ACTION_CONTRACT, validate_contract


VERSION = 'dh116-native-mpc-privileged-capture-v1'
ACTION_OWNER = 'legacy_teacher_mpc'
STEP_FIELDS = ('world_step', 'sdk_step')
TIME_FIELDS = ('world_time_s', 'sdk_time_s')
SUBSTEPS = 8
TIME_TOLERANCE_S = 1e-7


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(name=path.name, size_bytes=path.stat().st_size, sha256=digest.hexdigest())


def _index(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer')
    return value


def _valid_clock(clock):
    for name in STEP_FIELDS:
        _index(clock[name], name)
    for name in TIME_FIELDS:
        if type(clock[name]) not in (int, float) or not math.isfinite(clock[name]) or clock[name] < 0:
            raise ValueError(f'{name} must be a finite nonnegative timestamp')
    return {name: clock[name] for name in (*STEP_FIELDS, *TIME_FIELDS)}


def _same_clock(current, before, delta=0):
    current = _valid_clock(current)
    return all(current[k] - before[k] == delta for k in STEP_FIELDS) and all(
        math.isclose(current[k] - before[k], delta / 480, rel_tol=0, abs_tol=TIME_TOLERANCE_S)
        for k in TIME_FIELDS)


def _tensor(value, shape, label, *, bounded=False):
    if not isinstance(value, torch.Tensor) or value.shape != shape or not value.is_floating_point():
        raise ValueError(f'{label} must be floating-point with shape {shape}')
    if not bool(torch.isfinite(value).all()) or (bounded and not bool((value.abs() <= 1).all())):
        raise ValueError(f'{label} contains nonfinite or out-of-range values')
    return value.detach().to(device='cpu', copy=True)


@dataclass(frozen=True)
class PreActionSample:
    state: torch.Tensor
    valid: torch.Tensor
    action: torch.Tensor
    clock: dict


class PreActionCapture:
    """Keep original values even where an independent validity source is absent."""

    def __init__(self, environments, max_steps, *, t5_first_episodes=False):
        if type(environments) is not int or environments not in (2, 4, 6, 8):
            raise ValueError('use 2 to 8 even native environments')
        if t5_first_episodes and (environments != 8 or max_steps != 1800):
            raise ValueError('T5 capture requires exactly 8 environments and a 1800-step cap')
        max_budget = 1800 if t5_first_episodes else 180
        if type(max_steps) is not int or not 1 <= max_steps <= max_budget:
            raise ValueError(f'use at most {max_budget} policy steps')
        self.environments = environments
        self.max_steps = max_steps
        self.samples = []
        self._current = None

    def observe(self, state, task, clock):
        observed = _tensor(state, (self.environments, 371), 'canonical 371D TensorEnv observation')
        q, qd = task.m['q'], task.m['qd']
        # These are the first 24 fields of the original observe() with no running normalization.
        expected = torch.cat((2 * (q - task.low) / (task.high - task.low) - 1, qd / .3), dim=1)
        # TensorEnv rounds its normalized observation to the exported dtype.
        expected = expected.detach().to(device=observed.device, dtype=observed.dtype)
        if expected.shape != observed[:, :24].shape:
            raise ValueError('measured q/qd does not match the canonical joint layout')
        valid = torch.zeros_like(observed, dtype=torch.bool)
        valid[:, :24] = torch.isclose(observed[:, :24], expected, atol=1e-6, rtol=1e-6)
        self._current = observed, valid, _valid_clock(clock)

    def prepare(self, action, clock):
        if self._current is None or len(self.samples) >= self.max_steps:
            raise ValueError('missing pre-action state or capture budget exceeded')
        state, valid, before = self._current
        if not _same_clock(clock, before):
            raise ValueError('physics advanced after the recorded pre-action observation')
        action = _tensor(action, (self.environments, 24), 'canonical 24D issued action', bounded=True)
        return PreActionSample(state, valid, action, before)


@contextmanager
def capture_environment(env, recorder, read_clock):
    """Observe the existing reset/step calls. The collector owns all physics."""
    original_reset, original_step = env.reset, env.step
    originals = {key: vars(env)[key] for key in ('reset', 'step') if key in vars(env)}
    saw_reset = False

    def reset():
        nonlocal saw_reset
        if saw_reset:
            raise ValueError('a second whole-environment reset would reuse first-episode identities')
        state = original_reset()
        recorder.observe(state, env.task, read_clock())
        saw_reset = True
        return state

    def step(actions):
        sample = recorder.prepare(actions, read_clock())
        result = original_step(actions)
        after = read_clock()
        if not _same_clock(after, sample.clock, SUBSTEPS):
            raise ValueError('original TensorEnv step did not preserve the eight-substep clock')
        recorder.samples.append(sample)
        recorder.observe(result[0], env.task, after)
        return result

    env.reset, env.step = reset, step
    try:
        yield
    finally:
        for key in ('reset', 'step'):
            if key in originals:
                setattr(env, key, originals[key])
            else:
                delattr(env, key)


def join_execution(samples, visual, physical_path, *, run_uuid):
    """Match by environment, control tick, clock, episode, and exactly issued 24D action."""
    metadata = visual['sample_metadata']
    if not metadata or not isinstance(run_uuid, str) or not run_uuid:
        raise ValueError('nonempty visual metadata and actual physical run identity required')
    count = len(metadata)
    issued = _tensor(visual['actions'], (count, 24), 'visual issued actions', bounded=True)
    references = _tensor(visual['reference_actions'], (count, 24), 'historical reference actions', bounded=True)
    expected = {}
    states, masks = [], []
    records = []
    camera_frames = {}
    for index, row in enumerate(metadata):
        tick, env_id = _index(row['tick'], 'tick'), _index(row['env_id'], 'env_id')
        if tick >= len(samples) or env_id >= samples[tick].state.shape[0]:
            raise ValueError('visual metadata points outside recorded control samples')
        key = env_id, tick
        if key in expected:
            raise ValueError('duplicate environment/policy step in visual metadata')
        sample = samples[tick]
        timestamp = row['simulation_time_s']
        frame = _index(row['frame_id'], 'camera frame')
        if (camera_frames.setdefault(tick, frame) != frame
                or row.get('run_uuid', run_uuid) != run_uuid or row['episode_id'] != env_id
                or row['physics_step'] != sample.clock['sdk_step']
                or type(timestamp) not in (int, float)
                or not math.isclose(timestamp, sample.clock['sdk_time_s'], rel_tol=0, abs_tol=TIME_TOLERANCE_S)
                or not torch.equal(issued[index], sample.action[env_id])):
            raise ValueError('visual clock or actually issued action disagrees with pre-action observation')
        result_row = dict(row, policy_clock_before=sample.clock)
        result_row['camera_episode_uid'] = row.get('episode_uid')
        states.append(sample.state[env_id])
        masks.append(sample.valid[env_id])
        records.append(result_row)
        expected[key] = sample, result_row
    latest = {}
    episodes = {}
    owners = {}
    terminal_intervals = {}
    with gzip.open(physical_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):
                raise ValueError('incomplete physical JSONL line')
            packet = json.loads(line)
            physical = packet['state']['clock']
            env_id, tick = _index(packet['env_id'], 'physical env_id'), _index(physical['policy_step'], 'physical policy_step')
            key = env_id, tick
            if key not in expected:
                continue
            sample, row = expected[key]
            episode = packet['episode_uid']
            generation = _index(physical['reset_generation'], 'reset generation')
            if (not isinstance(episode, str) or not episode
                    or packet['run_uuid'] != run_uuid or physical['run_uuid'] != run_uuid
                    or physical['env_id'] != env_id or physical['episode_uid'] != episode
                    or packet['action_owner'] != ACTION_OWNER or packet['counts_as_Student_success'] is not False
                    or physical['reset'] is not False or type(physical['terminal']) is not bool
                    or episodes.setdefault(env_id, (episode, generation)) != (episode, generation)
                    or owners.setdefault(episode, env_id) != env_id or tick > terminal_intervals.get(env_id, tick)
                    or not _same_clock(packet['policy_clock_before'], sample.clock)
                    or packet['issued_action24'] != sample.action[env_id].tolist()):
                raise ValueError('physical ownership, episode, clock or issued action mismatch')
            previous, terminal = latest.get(key, (0, False))
            delta = previous + 1
            if terminal or delta > SUBSTEPS or not _same_clock(physical, sample.clock, delta):
                raise ValueError('missing, extra or out-of-order 480Hz evidence')
            latest[key] = delta, physical['terminal']
            if physical['terminal']:
                terminal_intervals[env_id] = tick
            row.update(episode_uid=episode, reset_generation=generation,
                       physical_substeps=delta, last_physics_step=physical['sdk_step'],
                       terminal_in_interval=physical['terminal'])
    for key in expected:
        if key[1] > terminal_intervals.get(key[0], key[1]):
            raise ValueError('visual record appears after the first terminal sample')
        count, terminal = latest.get(key, (0, False))
        if not terminal and count != SUBSTEPS:
            raise ValueError('complete 480Hz evidence missing for issued action')
    return dict(version=VERSION, observations=torch.stack(states), privileged_valid=torch.stack(masks),
                actions=issued, reference_actions=references, sample_metadata=records, run_uuid=run_uuid,
                action_owner=ACTION_OWNER, action_contract=ACTION_CONTRACT,
                qualified_for_bc=False, new_expert_qualification_issued=False,
                counts_as_Student_success=False)


def bind_camera_frames(directory, rows, environments):
    """Verify and bind the original raw RGB-D frames and per-environment calibration."""
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest['version'] != 'dh116-real-rgbd-chunks-v1'
            or manifest['source'] != 'native_same_pre_action_frame' or manifest['sensor_only'] is not True):
        raise ValueError('original pre-action sensor-only camera evidence required')
    expected = {}
    for row in rows:
        expected.setdefault(row['frame_id'], []).append(row)
    seen = []
    matched = set()
    for entry in manifest['chunks']:
        chunk = directory / entry['path']
        if Path(entry['path']).name != entry['path'] or chunk.is_symlink():
            raise ValueError('raw RGB-D chunk must belong to the capture directory')
        frames = entry['frame_ids']
        seen.extend(frames)
        if not expected.keys() & set(frames):
            continue
        identity = file_identity(chunk)
        if identity['sha256'] != entry['sha256']:
            raise ValueError('raw RGB-D chunk SHA differs')
        content = torch.load(chunk, map_location='cpu', weights_only=True)
        if [frame['frame_id'] for frame in content] != frames:
            raise ValueError('raw RGB-D frame identities disagree with their manifest')
        for frame in content:
            frame_id = frame['frame_id']
            if frame_id not in expected:
                continue
            k = _tensor(frame['K'], (environments, 3, 3), 'fixed-camera intrinsics')
            transform = _tensor(frame['T_world_camera'], (environments, 4, 4), 'fixed-camera pose')
            rgb, depth = frame['rgb'], frame['depth']
            if (rgb.dtype != torch.uint8 or rgb.shape != (environments, 96, 224, 3)
                    or not depth.is_floating_point() or depth.shape != (environments, 96, 224, 1)):
                raise ValueError('raw RGB-D frames disagree with the bounded camera contract')
            for row in expected[frame_id]:
                if (frame['physics_step'] != row['physics_step']
                        or any(not math.isclose(frame[name], row['simulation_time_s'],
                                               abs_tol=TIME_TOLERANCE_S, rel_tol=0)
                               for name in ('simulation_time', 'rendered_simulation_time'))):
                    raise ValueError('raw RGB-D frame is not from the recorded pre-action clock')
                index = row['env_id']
                calibration = dict(K=k[index].tolist(), T_world_camera=transform[index].tolist(),
                                   resolution_wh=[224, 96], convention='native_fixed_camera_world_optical')
                encoded = json.dumps(calibration, sort_keys=True, separators=(',', ':'), allow_nan=False)
                row.update(camera_calibration_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                           raw_rgbd_chunk_sha256=identity['sha256'])
            matched.add(frame_id)
    if seen != sorted(set(seen)) or len(seen) != manifest['frames'] or matched != set(expected):
        raise ValueError('missing, duplicate or unordered raw RGB-D frame evidence')
    return file_identity(manifest_path)


def read_terminal_snapshots(path, native_identity):
    identity = file_identity(path)
    if (native_identity['path'] != str(path.resolve())
            or native_identity['sha256'] != identity['sha256']):
        raise ValueError('original native first-episode terminal archive has changed')
    return json.loads(path.read_text(encoding='utf-8')), identity


def partition_first_episodes(archive, report, terminals, *, environments):
    """Cross-check the native collector's first-episode outcomes against the joined ledger."""
    rows = archive['sample_metadata']
    episodes = report['episodes']
    expected_keys = {str(env) for env in range(environments)}
    if (set(episodes) != expected_keys or not set(terminals) <= expected_keys
            or report['status'] != 'completed' or report['mode'] != 'teacher_reference_mpc'
            or report['mpc_profile'] != 'pressure_geometry_constrained'
            or report['native_runtime_verified'] is not True or report['ppo_updates'] != 0
            or report['samples'] != len(rows)):
        raise ValueError('matching completed original native MPC collection required')
    max_ticks = _index(report['control_steps'], 'native collection steps')
    by_env = [[] for _ in range(environments)]
    for index, row in enumerate(rows):
        env_id = _index(row['env_id'], 'joined environment')
        if env_id >= environments:
            raise ValueError('joined environment is absent from the collection')
        by_env[env_id].append(index)
    result = {key: [] for key in ('successful', 'failed', 'incomplete')}
    native_successes = 0
    for env_id, indexes in enumerate(by_env):
        episode = episodes[str(env_id)]
        terminal = episode['terminal']
        physical = terminals.get(str(env_id))
        count = max_ticks if terminal is None else _index(terminal['control_steps'], 'terminal control steps')
        if (episode['env_id'] != env_id or episode['first_episode'] is not True
                or episode['native_physics'] is not True or type(episode['success']) is not bool
                or not 1 <= count <= max_ticks or len(indexes) != count
                or (terminal is None) != (physical is None)):
            raise ValueError('incomplete or inconsistent first-episode coverage')
        selected = [rows[index] for index in indexes]
        uid = selected[0]['episode_uid']
        if (not uid or any(row['tick'] != tick or row['episode_id'] != env_id
                or row['episode_uid'] != uid or row['grasp_style'] != episode['grasp_style']
                or row['terminal_in_interval'] is not (terminal is not None and tick == count - 1)
                for tick, row in enumerate(selected))):
            raise ValueError('first-episode tick, outcome, style, or identity differs')
        if physical is not None:
            clock = physical['state']['clock']
            if (physical['run_uuid'] != archive['run_uuid'] or physical['env_id'] != env_id
                    or physical['episode_uid'] != uid or clock['terminal'] is not True
                    or physical['action_owner'] != ACTION_OWNER
                    or physical['counts_as_Student_success'] is not False
                    or physical['issued_action24'] != archive['actions'][indexes[-1]].tolist()
                    or clock['policy_step'] != count - 1
                    or clock['sdk_step'] != selected[-1]['last_physics_step']):
                raise ValueError('physical terminal does not match the joined final issued action')
        if episode['success']:
            style_audit = episode['native_contact_style_audit']
            if (terminal is None or terminal['success'] is not True or terminal['failure'] != 0
                    or terminal['recovered'] is not False or episode['recovered'] is not False
                    or terminal['recorded_execution_complete'] is not True
                    or not isinstance(style_audit, dict) or style_audit.get('style') != episode['grasp_style']):
                raise ValueError('native success lacks a complete audited success terminal')
            status = 'successful'
            native_successes += 1
        else:
            status = 'incomplete' if terminal is None else 'failed'
        result[status].extend(indexes)
    if report['successful_episodes'] != native_successes:
        raise ValueError('native successful first-episode count disagrees with its audit')
    return {status: sorted(indexes) for status, indexes in result.items()}


def _write_outcome_archives(output, archive, report, terminals, environments):
    partitions = partition_first_episodes(archive, report, terminals, environments=environments)
    metadata = archive['sample_metadata']
    files = {}
    for status, indexes in partitions.items():
        path = output / f'mpc_{status}_first_episodes_unqualified.pt'
        rows = [metadata[index] for index in indexes]
        partition = dict(archive, sample_metadata=rows, first_episode_outcome=status,
                         source_row_indices=indexes)
        for name in ('observations', 'privileged_valid', 'actions', 'reference_actions'):
            partition[name] = archive[name][indexes]
        with path.open('xb') as stream:
            torch.save(partition, stream)
        files[status] = dict(archive=file_identity(path), rows=len(rows),
                             episodes=len({row['episode_uid'] for row in rows}))
    return files


def finish_capture(output, recorder, contract, *, run_plan):
    """Write a separate, nonexpert artifact only after the genuine archives agree."""
    validate_contract(contract)
    output = Path(output)
    raw_dir = output / 'mpc_state_capture'
    summary = json.loads((raw_dir / 'capture_summary.json').read_text())
    physical = raw_dir / 'states_480hz.jsonl.gz'
    identity = file_identity(physical)
    if (summary['native_runtime_verified'] is not True or summary['error'] is not None
            or summary['action_owner'] != ACTION_OWNER or summary['states']['sha256'] != identity['sha256']):
        raise ValueError('original native 480Hz archive is incomplete or has changed')
    visual_path = output / 'all_executed_supervision.pt'
    visual = torch.load(visual_path, map_location='cpu', weights_only=True)
    if any(row.get('run_uuid') != summary['run_uuid'] or not row.get('episode_uid')
           for row in visual['sample_metadata']):
        raise ValueError('original sequence sidecar camera/run identity required')
    archive = join_execution(recorder.samples, visual, physical, run_uuid=summary['run_uuid'])
    camera = bind_camera_frames(output / 'raw_rgbd', archive['sample_metadata'], recorder.environments)
    sensor_path = raw_dir / 'sensor_evidence_480hz.jsonl.gz'
    sensor_identity = file_identity(sensor_path)
    raw_report = json.loads((raw_dir / 'native_state_capture_report.json').read_text())
    if raw_report['run_uuid'] != summary['run_uuid'] or raw_report['raw_evidence']['sha256'] != sensor_identity['sha256']:
        raise ValueError('original native 480Hz sensor evidence is incomplete or has changed')
    visual_contract = visual['visual_contract']
    if visual_contract['privileged_critic_contract'] != contract or visual_contract['action_contract'] != ACTION_CONTRACT:
        raise ValueError('legacy visual archive disagrees with the resolved canonical contracts')
    archive.update(
        privileged_observation_contract=contract,
        source_sha256=visual['teacher_source_sha256'],
        fixed_camera_config_sha256=visual_contract['camera_config_sha256'],
        run_plan=run_plan,
        initial_native_state=file_identity(output / 'initial_native_state.pt'),
        legacy_visual_archive=file_identity(visual_path), physical_archive=identity,
        physical_sensor_archive=sensor_identity, raw_rgbd_manifest=camera,
        action_scope='canonical actually issued TensorEnv command; SDK targets are separate native evidence',
        validity_source='TensorEnv joint q/qd prefix independently checked against task.m; other fields unqualified')
    target = output / 'mpc_privileged_pre_action_unqualified.pt'
    outcomes = {}
    if run_plan.get('capture_scope') == 't5_first_episodes':
        terminal_path = raw_dir / 'terminal_snapshots.json'
        terminals, terminal_identity = read_terminal_snapshots(terminal_path, summary['terminals'])
        report_path = output / 'teacher_collection_report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        archive.update(legacy_collection_report=file_identity(report_path),
                       physical_terminal_archive=terminal_identity)
        outcomes['first_episode_outcomes'] = _write_outcome_archives(
            output, archive, report, terminals, recorder.environments)
    with target.open('xb') as stream:
        torch.save(archive, stream)
    return dict(archive=file_identity(target), rows=len(archive['sample_metadata']),
                valid_fields=archive['privileged_valid'].sum(dim=0).tolist(), qualified_for_bc=False,
                **outcomes)
