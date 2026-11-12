import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import (
    PreActionCapture,
    capture_environment,
    join_execution,
)


def clock(step):
    return dict(world_step=step, sdk_step=step + 2,
                world_time_s=step / 480, sdk_time_s=(step + 2) / 480)


class Env:
    def __init__(self):
        self.current = torch.arange(371, dtype=torch.float32).repeat(2, 1)
        self.current[:, :24] = 0
        self.physical_step = 16
        self.calls = 0
        self.task = SimpleNamespace(low=torch.full((12,), -1.0), high=torch.ones(12),
                                    m=dict(q=torch.zeros(2, 12), qd=torch.zeros(2, 12)))

    def reset(self):
        return self.current

    def step(self, action):
        self.calls += 1
        self.physical_step += 8
        self.current[:, 24:] += 1
        return self.current, torch.zeros(2), torch.zeros(2, dtype=torch.bool), {}


def test_preserves_original_pre_action_state_and_restores_hooks():
    env = Env()
    recorder = PreActionCapture(2, 2)
    with capture_environment(env, recorder, lambda: clock(env.physical_step)):
        assert env.reset() is env.current
        env.step(torch.full((2, 24), .25))
        env.step(torch.full((2, 24), .50))
    assert env.calls == 2
    assert 'reset' not in vars(env) and 'step' not in vars(env)
    first, second = recorder.samples
    assert first.state[0, 24] == 24
    assert second.state[0, 24] == 25
    assert first.action[1, 7] == .25
    assert first.valid[:, :24].all() and not first.valid[:, 24:].any()
    assert first.clock == clock(16) and second.clock == clock(24)


@pytest.mark.parametrize('fault', ['stale_clock', 'action_shape', 'action_bounds', 'visual_state'])
def test_refuses_incompatible_capture_before_step(fault):
    env = Env()
    recorder = PreActionCapture(2, 2)
    action = torch.zeros(2, 24)
    with capture_environment(env, recorder, lambda: clock(env.physical_step)):
        if fault == 'visual_state':
            env.current = torch.zeros(2, 3844)
            with pytest.raises(ValueError, match='371D'):
                env.reset()
        else:
            env.reset()
            if fault == 'stale_clock':
                env.physical_step += 1
            elif fault == 'action_shape':
                action = torch.zeros(2, 23)
            else:
                action[0, 17] = 1.001
            with pytest.raises(ValueError):
                env.step(action)
    assert env.calls == 0 and not recorder.samples


def make_evidence(path, before, action, *, failure=None):
    with gzip.open(path, 'wt') as stream:
        for delta in range(1, 9):
            if delta == 4 and failure == 'hole':
                continue
            packet = dict(
                run_uuid='recorded-run', env_id=0, episode_uid='episode-0',
                action_owner='legacy_teacher_mpc', issued_action24=action[0].tolist(),
                policy_clock_before=before, counts_as_Student_success=False,
                state=dict(clock=dict(clock(before['world_step'] + delta),
                    run_uuid='recorded-run', env_id=0, episode_uid='episode-0',
                    policy_step=0, reset_generation=1, reset=False, terminal=False)))
            if delta == 4 and failure == 'physics_action':
                packet['issued_action24'][7] = .5
            stream.write(json.dumps(packet) + '\n')


@pytest.mark.parametrize('fault', [None, 'hole', 'physics_action', 'vision_action', 'camera_clock'])
def test_joins_actual_issued_action_and_contiguous_physics(tmp_path, fault):
    env = Env()
    recorder = PreActionCapture(2, 2)
    action = torch.full((2, 24), .25)
    with capture_environment(env, recorder, lambda: clock(env.physical_step)):
        env.reset()
        env.step(action)
    first = recorder.samples[0]
    raw = tmp_path / 'states_480hz.jsonl.gz'
    make_evidence(raw, first.clock, action, failure=fault)
    issued = action[:1].clone()
    if fault == 'vision_action':
        issued[0, 7] = .75
    metadata = [dict(env_id=0, episode_id=0, tick=0, frame_id=7, physics_step=18,
                     simulation_time_s=18 / 480, grasp_style='pinch_thumb_index')]
    if fault == 'camera_clock':
        metadata[0]['simulation_time_s'] += 1
    visual = dict(actions=issued, reference_actions=torch.zeros(1, 24), sample_metadata=metadata)
    if fault is not None:
        with pytest.raises(ValueError):
            join_execution(recorder.samples, visual, raw, run_uuid='recorded-run')
    else:
        result = join_execution(recorder.samples, visual, raw, run_uuid='recorded-run')
        assert result['observations'].shape == (1, 371)
        assert result['actions'].equal(action[:1])
        assert result['reference_actions'].equal(torch.zeros(1, 24))
        assert result['sample_metadata'][0]['episode_uid'] == 'episode-0'
        assert result['sample_metadata'][0]['physical_substeps'] == 8
        assert not result['qualified_for_bc']


def test_joint_measurement_mismatch_is_invalid_without_replacing_observation():
    env = Env()
    env.current[0, 2] = .2
    recorder = PreActionCapture(2, 1)
    with capture_environment(env, recorder, lambda: clock(env.physical_step)):
        env.reset()
        env.step(torch.zeros(2, 24))
        with pytest.raises(ValueError, match='budget'):
            env.step(torch.zeros(2, 24))
    sample = recorder.samples[0]
    assert sample.state[0, 2] == pytest.approx(.2)
    assert not sample.valid[0, 2] and sample.valid[1, 2]
    assert env.calls == 1


def test_preserves_early_terminal_without_fabricated_noop_substeps(tmp_path):
    env = Env()
    recorder = PreActionCapture(2, 1)
    action = torch.zeros(2, 24)
    with capture_environment(env, recorder, lambda: clock(env.physical_step)):
        env.reset()
        env.step(action)
    raw = tmp_path / 'states_480hz.jsonl.gz'
    make_evidence(raw, recorder.samples[0].clock, action)
    with gzip.open(raw, 'rt') as stream:
        physical = [json.loads(line) for line in stream]
    physical[2]['state']['clock']['terminal'] = True
    visual = dict(actions=action[:1], reference_actions=action[:1],
                  sample_metadata=[dict(env_id=0, episode_id=0, tick=0, frame_id=7, physics_step=18,
                                        simulation_time_s=18 / 480)])
    with gzip.open(raw, 'wt') as stream:
        for row in physical[:3]:
            stream.write(json.dumps(row) + '\n')
    result = join_execution(recorder.samples, visual, raw, run_uuid='recorded-run')
    assert result['sample_metadata'][0]['terminal_in_interval']
    assert result['sample_metadata'][0]['physical_substeps'] == 3
    with gzip.open(raw, 'at') as stream:
        stream.write(json.dumps(physical[3]) + '\n')
    with pytest.raises(ValueError, match='extra or out-of-order'):
        join_execution(recorder.samples, visual, raw, run_uuid='recorded-run')


def test_baseline_preflight_only_accepts_original_mpc_route(tmp_path):
    from scripts.collect_dh116_mpc_privileged import FLAGS, INPUTS, REQUIRED, baseline_arguments

    inputs = {k: dict(path=str(tmp_path / k[2:]), sha256='a' * 64) for k in INPUTS}
    output = tmp_path / 'new'
    args = [*FLAGS, *(f'{k}={v}' for k, v in REQUIRED.items()),
            *(f'{k}={v["path"]}' for k, v in inputs.items()),
            '--num-envs=8', '--vision-teacher-max-control-steps=180', '--kit_args=--allow-root']
    native = baseline_arguments(args, inputs, output)
    assert '--num-envs=2' in native
    assert '--vision-teacher-max-control-steps=32' in native
    assert f'--output={output}' in native
    assert not output.exists()
    with pytest.raises(ValueError, match='unsupported'):
        baseline_arguments([*args, '--resume=old.pt'], inputs, output)
    with pytest.raises(ValueError, match='original'):
        baseline_arguments([item for item in args if item != '--vision-teacher-mpc'], inputs, output)


@pytest.mark.parametrize('fault', [None, 'sha', 'render_clock'])
def test_raw_rgbd_frame_binds_calibration_to_the_same_pre_action_clock(tmp_path, fault):
    from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import bind_camera_frames, file_identity

    frame = dict(frame_id=26, simulation_time=.0375, rendered_simulation_time=.0375,
                 physics_step=18, K=torch.eye(3).repeat(2, 1, 1),
                 T_world_camera=torch.eye(4).repeat(2, 1, 1),
                 rgb=torch.zeros(2, 96, 224, 3, dtype=torch.uint8),
                 depth=torch.zeros(2, 96, 224, 1))
    if fault == 'render_clock':
        frame['rendered_simulation_time'] += 1 / 480
    chunk = tmp_path / 'chunk_00000.pt'
    torch.save([frame], chunk)
    sha = file_identity(chunk)['sha256'] if fault != 'sha' else '0' * 64
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        version='dh116-real-rgbd-chunks-v1', source='native_same_pre_action_frame',
        frames=1, chunks=[dict(path=chunk.name, sha256=sha, frame_ids=[26])], sensor_only=True)))
    rows = [dict(frame_id=26, env_id=1, physics_step=18, simulation_time_s=.0375)]
    if fault:
        with pytest.raises(ValueError):
            bind_camera_frames(tmp_path, rows, 2)
    else:
        bind_camera_frames(tmp_path, rows, 2)
        assert len(rows[0]['camera_calibration_sha256']) == 64
        assert rows[0]['raw_rgbd_chunk_sha256'] == sha


def test_joint_validity_compares_physical_float64_to_observation_float32():
    env = Env()
    for name in ('low', 'high'):
        setattr(env.task, name, getattr(env.task, name).double())
    env.task.m['q'] = torch.full((2, 12), .123456789, dtype=torch.float64)
    env.task.m['qd'] = torch.full((2, 12), .043212345, dtype=torch.float64)
    env.current[:, :12] = (2 * (env.task.m['q'] - env.task.low) /
                           (env.task.high - env.task.low) - 1).float()
    env.current[:, 12:24] = (env.task.m['qd'] / .3).float()
    env.current[1, 12] += .002
    original = env.current.clone()
    recorder = PreActionCapture(2, 1)
    with capture_environment(env, recorder, lambda: clock(env.physical_step)):
        assert env.reset() is env.current
        env.step(torch.zeros(2, 24))

    sample, = recorder.samples
    assert torch.equal(sample.state, original)
    assert sample.state.dtype is torch.float32
    assert env.task.m['q'].dtype is torch.float64
    assert not sample.valid[1, 12]
    assert sample.valid[:, :24].sum().item() == 47
    assert not sample.valid[:, 24:].any()


def test_reviewed_t5_first_episode_preflight_keeps_control_inputs(tmp_path, monkeypatch):
    import scripts.collect_dh116_mpc_privileged as cli

    bound_inputs = {}
    for flag in cli.INPUTS:
        f = tmp_path / flag[2:]
        f.write_text(flag)
        bound_inputs[flag] = dict(path=str(f), sha256=cli.file_identity(f)['sha256'])
    legacy = [flag for flag in sorted(cli.FLAGS)
              if flag not in ('--vision-sequence-context-sidecar', '--vision-mpc-state-capture')]
    historical_tokens = [
        *legacy, *(f'{k}={v}' for k, v in cli.REQUIRED.items()),
        *(f'{k}={v["path"]}' for k, v in bound_inputs.items()),
        '--bimanual-teacher-physics-audit', '--num-envs=8', '--vision-teacher-max-control-steps=1900',
        '--vision-teacher-hold-tail-control-steps=600', '--seed=42', '--kit_args=--allow-root',
        '--output=historical',
    ]
    historical = tmp_path / 't5_teacher_reference_launch.json'
    historical.write_text(json.dumps(dict(mode='teacher', command=[
        'bash', '/archive/scripts/python.sh', '/archive/scripts/train_dh116_tabletop_gpu_ppo.py',
        *historical_tokens])))
    monkeypatch.setattr(cli, 'T5_LAUNCH_SHA256', cli.file_identity(historical)['sha256'])
    source = tmp_path / 'command.json'
    source.write_text(json.dumps(dict(version='dh116-stage7-private-native-state-launch-v1',
        command=['bash', 'scripts/python.sh', 'scripts/train_dh116_tabletop_gpu_ppo.py',
            *legacy, '--vision-sequence-context-sidecar', '--vision-mpc-state-capture',
            *(f'{k}={v}' for k, v in cli.REQUIRED.items()),
            *(f'{k}={v["path"]}' for k, v in bound_inputs.items())],
        bound_inputs=bound_inputs, bound_asset_dependencies=[])))
    new = tmp_path / 'new'
    args, _inputs, plan = cli.preflight(source, cli.file_identity(source)['sha256'], new,
                                       t5_first_episodes=historical)
    assert set(args) == ((set(historical_tokens) - {'--output=historical',
        '--vision-teacher-max-control-steps=1900', '--bimanual-teacher-physics-audit'}) |
        {f'--output={new}', '--vision-teacher-max-control-steps=1800',
         '--vision-sequence-context-sidecar', '--vision-mpc-state-capture'})
    assert plan['environments'] == 8 and plan['max_policy_steps'] == 1800
    assert plan['capture_scope'] == 't5_first_episodes' and not plan['qualified_for_bc']
    assert PreActionCapture(8, 1800, t5_first_episodes=True).max_steps == 1800
    with pytest.raises(ValueError, match='180'):
        PreActionCapture(8, 1800)
    with pytest.raises(ValueError):
        PreActionCapture(2, 1800, t5_first_episodes=True)
    historical.write_text(historical.read_text() + ' ')
    with pytest.raises(ValueError, match='T5.*SHA'):
        cli.preflight(source, cli.file_identity(source)['sha256'], new,
                      t5_first_episodes=historical)


@pytest.mark.parametrize('fault', [None, 'raw_terminal', 'unaudited_success', 'missing_policy_tick', 'raw_tick'])
def test_first_episode_partition_uses_native_outcome_and_complete_join(fault):
    from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import partition_first_episodes

    style = ('pinch_thumb_index', 'tripod_thumb_index_middle', 'pinch_thumb_index')
    rows = [dict(env_id=env, episode_uid=f'run:{env}:0', episode_id=env, tick=tick,
                 physical_substeps=8, last_physics_step=18 + 8 * (tick + 1),
                 terminal_in_interval=bool(tick == 1 and env in (0, 1)), grasp_style=style[env])
            for tick in range(2) for env in range(3)]
    report = dict(mode='teacher_reference_mpc', mpc_profile='pressure_geometry_constrained',
                  status='completed', native_runtime_verified=True, successful_episodes=1,
                  control_steps=2, samples=6, ppo_updates=0,
                  episodes={str(env): dict(env_id=env, first_episode=True, native_physics=True,
                      success=env == 0, recovered=False, grasp_style=style[env],
                      terminal=(dict(control_steps=2, success=env == 0, failure=0 if env == 0 else 7,
                                     recovered=False, recorded_execution_complete=True) if env < 2 else None),
                      native_contact_style_audit=dict(style=style[env]) if env == 0 else {})
                      for env in range(3)})
    terminals = {str(env): dict(env_id=env, run_uuid='run', episode_uid=f'run:{env}:0',
                 action_owner='legacy_teacher_mpc', issued_action24=[0.] * 24,
                 counts_as_Student_success=False, state=dict(clock=dict(policy_step=1, sdk_step=34, terminal=True)))
                 for env in range(2)}
    if fault == 'raw_terminal':
        terminals.pop('1')
    if fault == 'unaudited_success':
        report['episodes']['0']['native_contact_style_audit'] = {}
    if fault == 'missing_policy_tick':
        rows.pop(0)
        report['samples'] -= 1
    if fault == 'raw_tick':
        terminals['0']['state']['clock']['sdk_step'] += 1
    archive = dict(run_uuid='run', sample_metadata=rows, actions=torch.zeros(len(rows), 24))
    if fault:
        with pytest.raises(ValueError):
            partition_first_episodes(archive, report, terminals, environments=3)
    else:
        expected = {'successful': [0, 3], 'failed': [1, 4], 'incomplete': [2, 5]}
        assert partition_first_episodes(archive, report, terminals, environments=3) == expected



def test_native_terminal_identity_matches_existing_archive_format(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import (
        file_identity, read_terminal_snapshots,
    )

    path = tmp_path / 'terminal_snapshots.json'
    path.write_text('{}')
    native_identity = dict(path=str(path.resolve()), sha256=file_identity(path)['sha256'])
    assert read_terminal_snapshots(path, native_identity) == ({}, file_identity(path))
    path.write_text('{"2":{}}')
    with pytest.raises(ValueError, match='has changed'):
        read_terminal_snapshots(path, native_identity)
