import json

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from go2_pvcnn.control.dh116 import tabletop_vertical_force as force_module
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
from test_dh116_close_downward_penalty import task
from test_dh116_tensor_contact_window import loads
from test_dh116_tensor_env import World


def read_events(path):
    accumulator = EventAccumulator(str(path), size_guidance={'scalars': 0})
    accumulator.Reload()
    return {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()['scalars']}


def test_individual_endpoint_forces_are_never_averaged_and_ids_survive_resume(tmp_path):
    with SummaryWriter(str(tmp_path / 'tb')) as writer:
        recorder = force_module.CloseCompletionForceEvents(writer, tmp_path / 'events.jsonl', event_index=7)
        components = torch.tensor([[[.4, -100.], [.6, -100.]],
                                   [[2., 100.], [3., 100.]],
                                   [[900., 0.], [900., 0.]]], dtype=torch.float64)
        recorder.record(components, components[:, :, 0].sum(1), torch.tensor([4, 9, 0]),
                        torch.tensor([.25, .32, 0.]), torch.tensor([True, True, False]), 3)
        components.zero_()  # Later samples/reset must not change either endpoint.
        assert recorder.event_index == 9
    rows = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert [r['event_index'] for r in rows] == [8, 9]
    assert [r['env_index'] for r in rows] == [0, 1]
    assert [r['episode_index'] for r in rows] == [5, 10]
    assert [r['force_z_n'] for r in rows] == [1., 5.]
    assert rows[0]['hand_normal_z_n'] == [.4, .6]
    assert rows[0]['physics_substep'] == 3
    assert all(r['friction_included'] is False for r in rows)
    scalars = read_events(tmp_path / 'tb')
    points = scalars['force_N/close完成_竖直合力_瞬时']
    assert [(p.step, p.value) for p in points] == [(8, 1.), (9, 5.)]
    assert not any('平均' in tag for tag in scalars)


def test_no_completed_close_has_no_force_point(tmp_path):
    with SummaryWriter(str(tmp_path / 'tb')) as writer:
        recorder = force_module.CloseCompletionForceEvents(writer, tmp_path / 'events.jsonl')
        recorder.record(torch.zeros(2, 2, 2), torch.zeros(2), torch.zeros(2, dtype=torch.long),
                        torch.zeros(2), torch.zeros(2, dtype=torch.bool), 1)
        assert recorder.event_index == 0
    assert not (tmp_path / 'events.jsonl').exists()
    assert not any(tag.startswith('force_N/') for tag in read_events(tmp_path / 'tb'))


@pytest.mark.parametrize('later', ['force_change', 'reset', 'fatal'])
def test_env_records_gate_sample_before_later_change_reset_or_fault(tmp_path, later):
    t = task(2)
    class EndpointWorld(World):
        decimation = 3
        def __init__(self):
            super().__init__(t)
            self.m = loads(t, 4.)
            self.object_weight_n = torch.ones(2)
            self.vertical_components = torch.tensor([[[.4, -5.], [.6, -5.]],
                                                     [[-.4, 5.], [-.6, 5.]]], dtype=torch.float64)
        def advance(self, *args):
            self.tick += 1
            if self.tick > 1:
                self.vertical_components[:, :, 0] = -4.
                if later == 'reset' and self.tick == 2: t.elapsed[0] = 15.01
                if later == 'fatal': self.qd[0] = float('nan')
            return self.m
        def reset(self, ids):
            m = super().reset(ids)
            m['bottom'][ids] = .3
            self.vertical_components[ids] = 0.
            return m
    w = EndpointWorld()
    env = TensorEnv(w, t)
    t.contact_time[:] = .2 - w.dt
    env.episodes[:] = torch.tensor([3, 6])
    with SummaryWriter(str(tmp_path / 'tb')) as writer:
        env.close_completion_events = force_module.CloseCompletionForceEvents(writer, tmp_path / 'events.jsonl')
        if later == 'fatal':
            with pytest.raises(RuntimeError, match='nonfinite_velocity'):
                env.step(torch.zeros(2, 20))
        else:
            env.step(torch.zeros(2, 20))
    rows = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['force_z_n'] == 1.
    assert rows[0]['env_index'] == 0 and rows[0]['episode_index'] == 4
    assert rows[0]['physics_substep'] == 1
    assert rows[0]['episode_time_s'] == pytest.approx(w.dt)
    assert read_events(tmp_path / 'tb')['force_N/close完成_竖直合力_瞬时'][0].value == 1.


def test_rollout_average_has_an_explicit_diagnostic_namespace():
    class Writer:
        def __init__(self): self.points = {}
        def add_scalar(self, tag, value, step): self.points[tag] = value
    stats = force_module.VerticalForceStats('cpu')
    stats.add(torch.tensor([[[1., 100.], [2., 100.]]]), torch.ones(1), torch.tensor([True]))
    writer = Writer()
    force_module.write_directional_force_scalars(writer, stats.summary(), 5)
    assert not any(tag.startswith('force_N/') for tag in writer.points)
    assert writer.points['rollout_force_N/竖直合力_向上为正'] == 3.


def test_instant_force_metadata_migration_is_exact_and_one_way():
    from go2_pvcnn.control.dh116 import tabletop_tensor_training as training
    from go2_pvcnn.control.dh116.tabletop_tensor_task import CLOSE_COMPLETION_FORCE_CONTRACT
    from types import SimpleNamespace
    old = dict(num_envs=4096, vertical_force_reward={
        'source': 'object_filtered_normal_on_hand_negated', 'friction_excluded': True},
        grasp_readiness={'phase_flow': 'close_to_lift'})
    new = dict(old)
    training.apply_close_completion_contract(new, SimpleNamespace(vertical_force_reward=True, skip_hold=True))
    assert new['close_completion_force_gate'] == CLOSE_COMPLETION_FORCE_CONTRACT
    assert new['close_completion_force_telemetry'] == force_module.CLOSE_EVENT_CONTRACT
    assert training.checkpoint_contract_changes(old, new) == [
        'instantaneous_close_completion_force_v1', 'individual_close_completion_force_v1']
    assert training.checkpoint_contract_changes(new, new) == []
    for a, b in ((new, old), (old, dict(new, num_envs=4)),
                 (old, dict(new, close_completion_force_gate=dict(CLOSE_COMPLETION_FORCE_CONTRACT, condition='force > -1')))):
        with pytest.raises(ValueError, match='contract mismatch'):
            training.checkpoint_contract_changes(a, b)
    unchanged = dict(old)
    training.apply_close_completion_contract(unchanged, SimpleNamespace(vertical_force_reward=False, skip_hold=True))
    assert unchanged == old


def test_checkpoint_preserves_exact_endpoint_event_counter(tmp_path):
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_tensor_training import save_checkpoint
    class State:
        def state_dict(self): return {}
    algorithm = SimpleNamespace(policy=State(), optimizer=State(), learning_rate=.0003)
    env = SimpleNamespace(**{name: torch.zeros(2) for name in (
        'episodes', 'successes', 'recovered', 'reached_counts', 'transition_counts')},
        close_completion_events=SimpleNamespace(event_index=123))
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, algorithm, {}, 7, 20, env)
    assert torch.load(path, weights_only=False)['close_completion_event_index'] == 123


def test_instant_gate_metadata_composes_with_existing_normal_force_migration():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_tensor_training import apply_close_completion_contract, checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_tensor_task import CLOSE_DOWNWARD_PENALTY_CONTRACT, measured_readiness_contract
    current_force = force_module.vertical_reward_contract('continuous', 'downward_after_close', skip_hold=True)
    legacy_force = dict(current_force, version='signed_vertical_close_upward_v4',
                        source='object_filtered_normal_plus_friction_on_hand_negated')
    old = dict(num_envs=4096, reward=legacy_force['version'], phase_reward_version=legacy_force['version'],
               vertical_force_reward=legacy_force, grasp_readiness=measured_readiness_contract(True),
               close_downward_penalty=dict(CLOSE_DOWNWARD_PENALTY_CONTRACT,
                   version='close_downward_impulse_reset_v3', impulse_limit_n_s=2., force_semantics=None))
    current = dict(old, reward=current_force['version'], phase_reward_version=current_force['version'],
                   vertical_force_reward=current_force, close_downward_penalty=CLOSE_DOWNWARD_PENALTY_CONTRACT)
    apply_close_completion_contract(current, SimpleNamespace(vertical_force_reward=True, skip_hold=True))
    assert checkpoint_contract_changes(old, current) == ['instantaneous_close_completion_force_v1',
        'individual_close_completion_force_v1', 'normal_force_excludes_friction_v2']
    with pytest.raises(ValueError, match='contract mismatch'):
        checkpoint_contract_changes(old, dict(current, num_envs=16))


@pytest.mark.parametrize('failed_sink', ['journal', 'tensorboard'])
def test_endpoint_io_failure_keeps_healthy_physics_running_and_reports_gap(tmp_path, monkeypatch, failed_sink):
    from pathlib import Path
    t = task(2)
    class EndpointWorld(World):
        def __init__(self):
            super().__init__(t)
            self.m = loads(t, 4.)
            self.object_weight_n = torch.ones(2)
            self.vertical_components = torch.tensor([[[.4, 0.], [.6, 0.]],
                                                     [[-.4, 0.], [-.6, 0.]]], dtype=torch.float64)
        def advance(self, *args):
            self.tick += 1
            return self.m
    class Writer:
        def __init__(self): self.points = []; self.failures = 0
        def add_scalar(self, *point):
            if failed_sink == 'tensorboard':
                self.failures += 1
                raise OSError('injected TensorBoard write failure')
            self.points.append(point)
        def flush(self): pass
        def close(self): pass
    path = tmp_path / 'events.jsonl'
    original_open = Path.open
    def fail_journal(self, *args, **kwargs):
        if self == path and args and args[0] == 'a':
            raise OSError('injected journal write failure')
        return original_open(self, *args, **kwargs)
    if failed_sink == 'journal': monkeypatch.setattr(Path, 'open', fail_journal)
    w = EndpointWorld(); env = TensorEnv(w, t); writer = Writer()
    env.close_completion_events = force_module.CloseCompletionForceEvents(writer, path)
    t.contact_time[:] = .2 - w.dt
    _, _, done, _ = env.step(torch.zeros(2, 20))
    assert w.tick == 8 and not done.any() and not t.failure.any()
    assert t.phase.tolist() == [3, 1] and env.episodes.tolist() == [0, 0]
    diagnostic = env.close_completion_events.diagnostics()
    assert diagnostic['status'] == 'degraded'
    assert failed_sink in diagnostic['errors']
    assert diagnostic['first_affected_event_index'][failed_sink] == 1
    assert diagnostic['observed_events'] == 1
    if failed_sink == 'tensorboard':
        assert writer.failures == 1  # Do not repeatedly call a failed writer.
        assert json.loads(path.read_text())['force_z_n'] == 1.
    else:
        assert not path.exists()
        assert any(point[0] == 'force_N/close完成_竖直合力_瞬时' for point in writer.points)


def test_endpoint_writer_failure_never_masks_later_physical_stop(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_tensor_training import publish_stopped_monitor
    t = task(2)
    class EndpointWorld(World):
        def __init__(self):
            super().__init__(t)
            self.m = loads(t, 4.)
            self.object_weight_n = torch.ones(2)
            self.vertical_components = torch.ones(2, 2, 2)
        def advance(self, *args):
            self.tick += 1
            if self.tick == 2: self.qd[0] = float('nan')
            return self.m
    class Writer:
        def add_scalar(self, *args): raise OSError('injected endpoint write failure')
        def flush(self): raise OSError('injected writer flush failure')
        def close(self): raise OSError('injected writer close failure')
    w = EndpointWorld(); env = TensorEnv(w, t)
    recorder = env.close_completion_events = force_module.CloseCompletionForceEvents(Writer(), tmp_path / 'events.jsonl')
    t.contact_time[:] = .2 - w.dt
    with pytest.raises(RuntimeError, match='nonfinite_velocity') as stopped:
        env.step(torch.zeros(2, 20))
    assert w.tick == 2 and not w.resets
    recorder.writer.flush()  # The rollout logger shares this writer.
    recorder.writer.close()  # Shutdown must preserve the physical exception.
    diagnostic = recorder.diagnostics()
    path = tmp_path / 'monitor.json'
    publish_stopped_monitor(path, stopped.value, env.physics_peaks, 0, 0, 2,
                            close_completion_telemetry=diagnostic)
    monitor = json.loads(path.read_text())
    assert 'nonfinite_velocity' in monitor['error']
    assert monitor['close_completion_telemetry']['status'] == 'degraded'
    assert 'injected endpoint write failure' in monitor['close_completion_telemetry']['errors']['tensorboard']
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 2


def test_journal_failure_preserves_prior_rows_and_other_sink(tmp_path, monkeypatch):
    from pathlib import Path
    class Writer:
        def __init__(self): self.points = []
        def add_scalar(self, *point): self.points.append(point)
        def flush(self): pass
    writer = Writer(); path = tmp_path / 'events.jsonl'
    recorder = force_module.CloseCompletionForceEvents(writer, path)
    args = (torch.ones(1, 2, 2), torch.tensor([2.]), torch.tensor([0]),
            torch.tensor([.2]), torch.tensor([True]), 1)
    recorder.record(*args)
    prior = path.read_bytes()
    original_open = Path.open
    def failed_open(self, *args, **kwargs):
        if self == path and args and args[0] == 'a': raise OSError('journal unavailable')
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', failed_open)
    recorder.record(*args)
    recorder.record(*args)
    assert path.read_bytes() == prior
    assert recorder.event_index == 3
    assert recorder.diagnostics()['first_affected_event_index']['journal'] == 2
    assert [point[2] for point in writer.points if point[0] == 'force_N/close完成_竖直合力_瞬时'] == [1, 2, 3]


def test_endpoint_programming_and_nonfinite_data_errors_are_not_downgraded(tmp_path):
    class Writer:
        def add_scalar(self, *args): raise ValueError('invalid scalar format')
        def flush(self): pass
    recorder = force_module.CloseCompletionForceEvents(Writer(), tmp_path / 'events.jsonl')
    args = (torch.ones(1, 2, 2), torch.tensor([2.]), torch.tensor([0]),
            torch.tensor([.2]), torch.tensor([True]), 1)
    with pytest.raises(ValueError, match='invalid scalar format'):
        recorder.record(*args)
    assert recorder.diagnostics()['status'] == 'ok'
    with pytest.raises(ValueError, match='JSON compliant'):
        recorder.record(args[0], torch.tensor([float('nan')]), *args[2:])
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('operation', ['flush', 'close', 'add_text'])
def test_shared_writer_io_errors_are_explicit_without_inventing_endpoint_events(tmp_path, operation):
    class Writer:
        def flush(self): raise OSError('flush unavailable')
        def close(self): raise OSError('close unavailable')
        def add_text(self, *args): raise OSError('text unavailable')
    recorder = force_module.CloseCompletionForceEvents(Writer(), tmp_path / 'events.jsonl')
    getattr(recorder.writer, operation)(*(['tag', 'text'] if operation == 'add_text' else []))
    diagnostic = recorder.diagnostics()
    assert diagnostic['status'] == 'degraded'
    assert diagnostic['observed_events'] == diagnostic['event_index'] == 0
    assert diagnostic['first_affected_event_index']['tensorboard'] is None
    assert operation in diagnostic['errors']['tensorboard']
    assert not (tmp_path / 'events.jsonl').exists()


@pytest.mark.parametrize('later_physical_fault', [False, True])
def test_tensor_training_publishes_degraded_telemetry_and_keeps_completed_checkpoint(tmp_path, monkeypatch, later_physical_fault):
    from types import SimpleNamespace
    from test_dh116_tensor_task import template
    from go2_pvcnn.control.dh116 import tabletop_tensor_training as training
    from go2_pvcnn.control.dh116 import tabletop_tensor_world as world_module
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
    import torch.utils.tensorboard
    seed_task = task(2)
    class SafeWorld(World):
        def __init__(self):
            super().__init__(seed_task)
            self.m = loads(seed_task, 4.)
            self.object_weight_n = torch.ones(2)
            self.vertical_components = torch.ones(2, 2, 2)
        def reset(self, ids): return self.m
        def advance(self, *args):
            self.tick += 1
            if later_physical_fault and self.tick == 130: self.qd[0] = float('nan')
            return self.m
    class BrokenEndpointWriter:
        def __init__(self, *args): self.failed = False
        def add_text(self, *args): pass
        def add_scalar(self, tag, *args):
            if tag.startswith('force_N/'):
                self.failed = True
                raise OSError('endpoint sink unavailable')
        def flush(self):
            if self.failed: raise OSError('writer remains unavailable')
        def close(self):
            if later_physical_fault:
                # Publish the physical stop before optional logger cleanup.
                stopped = json.loads((tmp_path / 'monitor.json').read_text())
                assert stopped['status'] == 'stopped' and 'nonfinite_velocity' in stopped['error']
            if self.failed: raise OSError('writer remains unavailable')
    world = SafeWorld()
    monkeypatch.setattr(world_module, 'TensorWorld', lambda _: world)
    monkeypatch.setattr(training, 'validate_world_adapter', lambda *args: {'synthetic_world': True})
    monkeypatch.setattr(torch.utils.tensorboard, 'SummaryWriter', BrokenEndpointWriter)
    e = template(); e.force_controller.parameters = LoadParameters(max_force_n=12., force_rate_n_s=None)
    args = SimpleNamespace(seed=0, num_envs=2, device='cpu', output=tmp_path,
        training_episode_resets=True, rollout_steps=16, pressure_relief=False,
        contact_reward_window='close_start', grasp_readiness='measured_load',
        vertical_force_reward=True, vertical_force_reward_mode='continuous',
        force_reward_start='downward_after_close', skip_hold=True, close_downward_penalty=True,
        control_probe_checkpoint=None, resume=None, episodes_per_env=None,
        timesteps=64 if later_physical_fault else 32)
    if later_physical_fault:
        with pytest.raises(RuntimeError, match='nonfinite_velocity'):
            training.run_tensor_training(world, e, args, {})
    else:
        result = training.run_tensor_training(world, e, args, {})
        assert result['status'] == 'training_budget_complete'
    monitor = json.loads((tmp_path / 'monitor.json').read_text())
    checkpoint = torch.load(tmp_path / 'tensor_ppo.pt', weights_only=False)
    assert monitor['iteration'] == checkpoint['iteration'] == 1
    assert monitor['close_completion_telemetry']['status'] == 'degraded'
    assert 'endpoint sink unavailable' in monitor['close_completion_telemetry']['errors']['tensorboard']
    assert len((tmp_path / 'close_completion_events.jsonl').read_text().splitlines()) == 2
    if later_physical_fault:
        assert monitor['status'] == 'stopped' and 'nonfinite_velocity' in monitor['error']
        assert world.tick == 130
    else:
        assert monitor['status'] == 'training_budget_complete' and world.tick == 128
