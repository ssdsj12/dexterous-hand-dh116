"""CPU contract tests for batched reset scheduling; no Isaac physics evidence."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_gpu_vec_env import GpuTaskVecEnv
from test_dh116_tabletop_vec_env import World, TabletopEnv


def isaac_classes():
    # Execute the production classes without importing/starting Isaac.
    path = Path(__file__).parents[1] / 'src/go2_pvcnn/control/dh116/tabletop_gpu_isaac.py'
    tree = ast.parse(path.read_text())
    scope = {'np': np, 'torch': torch}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)],
                            type_ignores=[]), str(path), 'exec'), scope)
    return scope['GpuWorld'], scope['GpuBackend']


def make_vector(jitter=0):
    _, backend_type = isaac_classes()
    world = World(4)
    world.advance = world.advance_all
    world.batch_calls = []
    physical_reset = [b.reset for b in world.backends]

    def reset_slots(indices, offsets):
        world.batch_calls.append(list(indices))
        for i, offset in zip(indices, offsets):
            physical_reset[i](offset)

    world.reset_slots = reset_slots
    world.reset_slot = lambda i, offset: reset_slots([i], [offset])
    for backend in world.backends:
        backend.adaptive_palm_offset = np.zeros((2, 3))
        backend.grip_stiffness = np.full(2, 10.)
        backend.reset = backend_type.reset.__get__(backend)
    envs = [TabletopEnv(b, stage='full', reset_jitter=jitter) for b in world.backends]
    return world, GpuTaskVecEnv(envs, world)


def test_full_reset_batches_physics_preserves_rng_and_flushes_before_writes():
    world, vec = make_vector()
    vec.seed(42)
    vec.reset()
    assert world.batch_calls == [[0, 1, 2, 3]]
    for env in vec.envs:
        env.log = [{'clearance_m': .01}]
        env.episode_sink = lambda summary, rows: world.events.append(('flush',))
    world.events.clear()
    vec.seed(42)
    repeated = vec.reset()
    assert world.events[:4] == [('flush',)] * 4
    assert repeated.shape == (4, 104) and world.ticks == 0
    for i, env in enumerate(vec.envs):
        expected = np.random.default_rng(42+i)
        expected.uniform(0, 0)  # Normal env.reset must still consume its draw.
        assert env.np_random.random() == expected.random()
    assert all(not hasattr(b, '_prepared_reset_offset') for b in world.backends)
    vec.envs[2].reset()
    assert world.batch_calls[-1] == [2]  # Prepared reset cannot be consumed twice.


def test_autoreset_batches_only_done_slots_and_preserves_terminal_samples():
    world, vec = make_vector()
    vec.reset()
    world.batch_calls.clear()
    before = world.backends[1].q.copy()
    results = []
    for i, env in enumerate(vec.envs):
        env.metrics.update(object_bottom_m=.301+i*.001, max_force_n=float(i))
        results.append((np.full(104, i+5., np.float32), float(i), i in (0, 2), False, {}))
    obs, _, dones, infos = vec.finish_results(results)
    assert world.batch_calls == [[0, 2]] and world.ticks == 0
    assert dones.tolist() == [True, False, True, False]
    assert np.array_equal(infos[2]['terminal_observation'], np.full(104, 7.))
    assert infos[2]['physical']['object_bottom_m'] == pytest.approx(.303)
    assert not np.array_equal(obs[2], infos[2]['terminal_observation'])
    assert np.array_equal(world.backends[1].q, before)
    assert [b.resets for b in world.backends] == [2, 1, 2, 1]


def test_jitter_falls_back_to_normal_per_slot_resets():
    world, vec = make_vector(jitter=.0003)
    vec.seed(42)
    vec.reset()
    assert world.batch_calls == [[0], [1], [2], [3]]
    assert len({b.offset for b in world.backends}) == 4


def test_prepared_token_is_cleared_if_normal_reset_raises():
    world, vec = make_vector()
    vec.envs[1].reset = Mock(side_effect=ValueError('unsafe reset'))
    with pytest.raises(ValueError, match='unsafe reset'):
        vec.reset()
    assert all(not hasattr(b, '_prepared_reset_offset') for b in world.backends)


def make_physical_world():
    world_type, _ = isaac_classes()
    world = world_type.__new__(world_type)
    world.n, world.device = 4, 'cpu'
    world.full_hand_contact = False
    world.mounts = None  # This fixture exercises the fixed-root reset path.
    world.origins = np.arange(12).reshape(4, 3).astype(float)
    world.layout = {'hand_roots': {side: np.array([.1, .2, .3, 1., 0, 0, 0])
                                  for side in ('left', 'right')}}
    world.seed = dict(joint_positions=np.ones((2, 6)), joint_velocities=np.full((2, 6), .2),
                      root_xy=np.array([.01, .02]), wire_pose=np.array([0., 0., .3, 1, 0, 0, 0]),
                      wire_velocity=np.zeros(6))
    world.ids = world.named_ids = [list(range(6)), list(range(6))]
    world.hands = []
    for _ in range(2):
        hand = Mock()
        hand.data.default_joint_pos = torch.zeros((4, 6))
        hand.actuators = {'active': SimpleNamespace(stiffness=torch.full((4, 6), 25.))}
        world.hands.append(hand)
    for name, shape in [('root_targets', (4, 2, 7)), ('commands', (4, 2, 6)),
                        ('gains', (4, 2)), ('previous_gains', (4, 2)), ('palms', (4, 2, 3)),
                        ('force', (4, 4)), ('force_vectors', (4, 4, 3))]:
        setattr(world, name, np.full(shape, 7.))
    world.flex = Mock(regions=object())
    world.wire = None
    world.sensors = []
    world.sim = SimpleNamespace(physics_sim_view=Mock())
    world.refresh = Mock()
    world.pad_positions = np.zeros((4, 4, 3))
    return world


def test_physical_subset_is_one_batch_and_leaves_other_targets_and_gains_untouched():
    world = make_physical_world()
    world.reset_slots([0, 2], [.01, .02])
    for hand in world.hands:
        assert hand.reset.call_count == hand.write_data_to_sim.call_count == 1
        assert hand.write_joint_stiffness_to_sim.call_count == 1
        assert hand.write_joint_state_to_sim.call_args.args[0].shape == (2, 6)
        assert hand.write_root_pose_to_sim.call_args.args[0].shape == (2, 7)
        assert torch.equal(hand.actuators['active'].stiffness[[1, 3]], torch.full((2, 6), 25.))
    world.sim.physics_sim_view.update_articulations_kinematic.assert_called_once()
    world.refresh.assert_called_once_with(sensors=False, geometry_indices=[0, 2])
    world.flex.reset.assert_called_once_with([0, 2])
    for name in ('root_targets', 'commands', 'gains', 'previous_gains', 'palms', 'force', 'force_vectors'):
        assert np.all(getattr(world, name)[[1, 3]] == 7.)
    assert np.all(world.gains[[0, 2]] == 10.)
    assert np.all(world.force[[0, 2]] == 0.)


def test_material_regions_initialize_once_when_slot_zero_is_in_batch():
    world = make_physical_world()
    world.flex.regions = None
    world.flex.initialize_regions.side_effect = lambda _: setattr(world.flex, 'regions', object())
    world.reset_slots([1, 3], [.01, .01])
    world.flex.initialize_regions.assert_not_called()
    world.reset_slots([0, 2], [.01, .01])
    world.flex.initialize_regions.assert_called_once_with(world.pad_positions)
    world.reset_slot(0, .01)
    assert world.flex.initialize_regions.call_count == 1
