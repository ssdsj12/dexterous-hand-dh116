import json
from types import SimpleNamespace

import pytest
import torch

from test_dh116_tensor_env import World
from test_dh116_tensor_task import template
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv


def test_fault_trace_keeps_bounded_ordered_snapshots_not_tensor_aliases(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_physics_fault_trace import PhysicsFaultTrace
    task = TensorTask(template(), 2, 'cpu')
    world = World(task)
    trace = PhysicsFaultTrace(tmp_path / 'fault.json', capacity=2)
    env = TensorEnv(world, task)
    task.palm_reposition = SimpleNamespace(state=torch.tensor([[1, 0], [0, 2]]))
    world.palms = torch.zeros((2, 2, 3))
    for i in range(3):
        world.qd.fill_(i)
        task.command.fill_(i / 10)
        env.substep = i + 1
        trace.record(env, world.m)
    world.qd.fill_(99)
    trace.write(torch.tensor([1]), 'nonfinite_velocity')
    result = json.loads((tmp_path / 'fault.json').read_text())
    assert result['env_indices'] == [1]
    assert len(result['frames']) == 2
    assert [f['physics_substep'] for f in result['frames']] == [2, 3]
    assert [f['joint_velocity_rad_s'][0][0][0] for f in result['frames']] == [1, 2]
    assert result['frames'][-1]['command_rad'][0][0] == pytest.approx(.2)
    assert result['frames'][-1]['reposition_state'] == [[0, 2]]
    assert result['frames'][-1]['commanded_palm_offset_m'] == [[[0, 0, 0], [0, 0, 0]]]


def test_nonfinite_velocity_writes_trace_before_exception(tmp_path):
    task = TensorTask(template(), 2, 'cpu')
    world = World(task)
    env = TensorEnv(world, task, fault_trace_path=tmp_path / 'fault.json')
    world.qd[0, 0, 1] = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite_velocity'):
        env.step(torch.zeros((2, 20)))
    result = json.loads((tmp_path / 'fault.json').read_text())
    assert result['context'] == 'nonfinite_velocity'
    assert result['env_indices'] == [0]
    assert result['frames'][-1]['joint_velocity_rad_s'][0][0][1] is None
    assert world.tick == 1 and not world.resets
    assert not task.elapsed.any() and not env.episodes.any()


def test_trace_write_error_does_not_hide_original_safety_failure(tmp_path):
    task = TensorTask(template(), 2, 'cpu')
    world = World(task)
    env = TensorEnv(world, task, fault_trace_path=tmp_path / 'missing' / 'fault.json')
    world.qd[0] = float('inf')
    with pytest.raises(RuntimeError, match='nonfinite_velocity') as caught:
        env.step(torch.zeros((2, 20)))
    assert 'fault_trace_error' in str(caught.value)
    assert not world.resets


def test_trace_record_error_does_not_bypass_velocity_guard(tmp_path, monkeypatch):
    task = TensorTask(template(), 2, 'cpu')
    world = World(task)
    env = TensorEnv(world, task, fault_trace_path=tmp_path / 'fault.json')
    def broken_record(*args):
        raise RuntimeError('trace allocation failed')
    monkeypatch.setattr(env.fault_trace, 'record', broken_record)
    world.qd[0] = float('inf')
    with pytest.raises(RuntimeError, match='nonfinite_velocity') as caught:
        env.step(torch.zeros((2, 20)))
    assert 'trace allocation failed' in str(caught.value)
    assert world.tick == 1 and not world.resets and not task.elapsed.any()


def test_extended_trace_copies_bounded_snapshots_in_selected_env_order(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_physics_fault_trace import PhysicsFaultTrace
    task = TensorTask(template(), 2, 'cpu')
    world = World(task)
    env = TensorEnv(world, task)
    trace = PhysicsFaultTrace(tmp_path / 'fault.json', capacity=2)
    task.f_target[:] = torch.tensor([[2., 3.], [4., 5.]])
    task.f_requested[:] = task.f_target + 1
    task.f_filtered[:] = task.f_target - 1
    task.f_offset[:] = torch.tensor([[.01, .02], [.03, .04]])
    task.f_blocked[:] = torch.tensor([[True, False], [False, True]])
    task.pressure_limited[:] = ~task.f_blocked
    task.paused[:] = torch.tensor([False, True])
    world.gains = torch.tensor([[10., 11.], [12., 13.]])
    world.previous_gains = world.gains - 1
    world.commands = torch.arange(24.).reshape(2, 2, 6)
    world.wp = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    world.wq = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    world.wv = torch.arange(12.).reshape(2, 6)
    world.m.update(
        contact_center_w=torch.arange(12.).reshape(2, 2, 3),
        contact_center_relative_m=torch.arange(12.).reshape(2, 2, 3) / 10,
        contact_normal_w=torch.tensor([[[1., 0., 0.], [0., 1., 0.]],
                                      [[0., 0., 1.], [-1., 0., 0.]]]),
        contact_normal_load_n=torch.tensor([[1., 2.], [3., 4.]]),
        contact_valid=torch.tensor([[True, False], [False, True]]),
        contact_all_below_centerline=torch.tensor([[False, True], [True, False]]),
        contact_count=torch.tensor([[1, 2], [3, 4]]),
    )
    sources = {key: getattr(task, key) for key in
               ('f_target', 'f_requested', 'f_filtered', 'f_offset',
                'f_blocked', 'pressure_limited', 'paused')}
    sources.update(grip_stiffness=world.gains,
                   previous_grip_stiffness=world.previous_gains,
                   applied_command_rad=world.commands,
                   object_position_w=world.wp, object_quat_w=world.wq,
                   object_velocity_w=world.wv)
    sources.update({key: value for key, value in world.m.items()
                    if key.startswith('contact_')})
    expected = []
    ids = torch.tensor([1, 0])
    for step in range(3):
        env.substep = step + 1
        expected.append({key: value[ids].clone().tolist() for key, value in sources.items()})
        trace.record(env, world.m)
        for value in sources.values():
            if value.dtype == torch.bool:
                value.logical_not_()
            else:
                value.add_(1)
    trace.write(ids, 'nonfinite_velocity')
    result = json.loads(trace.path.read_text())
    assert result['env_indices'] == [1, 0]
    assert [frame['sequence'] for frame in result['frames']] == [1, 2]
    assert [frame['physics_substep'] for frame in result['frames']] == [2, 3]
    for frame, snapshot in zip(result['frames'], expected[-2:]):
        for key, value in snapshot.items():
            assert frame[key] == value
    assert all(buffer.shape[0] == 2 for buffer in trace.buffers.values())


def test_trace_accepts_minimal_fixture_without_optional_diagnostics(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_physics_fault_trace import PhysicsFaultTrace
    q = torch.zeros((2, 2, 6))
    task = SimpleNamespace(command=torch.zeros((2, 12)), phase=torch.ones(2),
                           elapsed=torch.zeros(2), offset=torch.zeros(2))
    world = SimpleNamespace(q=q, qd=q.clone())
    env = SimpleNamespace(world=world, task=task, episodes=torch.zeros(2),
                          pending=torch.zeros(2, dtype=torch.bool), substep=1)
    trace = PhysicsFaultTrace(tmp_path / 'fault.json')
    trace.record(env, {'q': q.flatten(1), 'qd': q.flatten(1)})
    trace.write(torch.tensor([1]), 'nonfinite_velocity')
    result = json.loads(trace.path.read_text())
    assert result['joint_names'] is None
    assert set(result['frames'][0]) == {
        'sequence', 'physics_substep', 'joint_position_rad', 'joint_velocity_rad_s',
        'command_rad', 'phase', 'elapsed_s', 'episode', 'pending_reset',
        'lift_offset_m', 'active_q', 'active_qd',
    }


def test_extended_trace_serializes_invalid_contact_geometry_as_null(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_physics_fault_trace import PhysicsFaultTrace
    task = TensorTask(template(), 2, 'cpu')
    world = World(task)
    env = TensorEnv(world, task)
    world.wp = torch.tensor([[float('nan'), float('inf'), -float('inf')],
                             [1., 2., 3.]])
    world.m['contact_center_w'] = world.wp[:, None].expand(-1, 2, -1)
    world.m['contact_normal_w'] = world.m['contact_center_w'].clone()
    world.m['contact_valid'] = torch.tensor([[False, False], [True, True]])
    task.f_target[0, 0] = float('nan')
    trace = PhysicsFaultTrace(tmp_path / 'fault.json')
    trace.record(env, world.m)
    trace.write(torch.tensor([0]), 'independent_guard')
    def reject_nonfinite(token):
        pytest.fail(f'non-standard JSON number: {token}')
    result = json.loads(trace.path.read_text(), parse_constant=reject_nonfinite)
    frame = result['frames'][0]
    assert frame['object_position_w'] == [[None, None, None]]
    assert frame['contact_center_w'] == [[[None, None, None], [None, None, None]]]
    assert frame['contact_normal_w'] == frame['contact_center_w']
    assert frame['contact_valid'] == [[False, False]]
    assert frame['f_target'][0][0] is None
    assert not trace.path.with_suffix('.tmp').exists()
