"""Flat contact ABI reductions and source-owned read-once CPU contracts."""
import numpy as np
import pytest
import torch
from dh116.physics.solver import flat_cuda_pair_capture as flat_cuda_pair_capture
from dh116.physics.solver import TensorRodOwner

def flat(time=0.1):
    vectors = torch.zeros(4, 12)
    vectors[:3, 3] = torch.tensor([1.0, 1.0, -1.0])
    vectors[:3, 6] = torch.tensor([16777216.0, 1.0, -16777216.0])
    vectors[3, 4] = 1.0
    vectors[3, 7] = 3.0
    nodes = torch.zeros(2, 41, 3, dtype=torch.float64)
    nodes[0, 7, 0] = 1 / 12
    nodes[1, 2, 1] = 0.25
    wrenches = torch.zeros(2, 2, 6, dtype=torch.float64)
    wrenches[0, 0, 0] = -1 / 12
    wrenches[1, 1, 1] = -0.25
    return dict(schema='dh116.sdf_pair_flat_cuda.v1', substeps=12, num_envs=2, mesh_count=2, owner_time_s_before=time, owner_time_s_after=time, indices_gpu=torch.tensor([[0, 0, 7, 0], [0, 1, 7, 0], [0, 2, 7, 0], [1, 4, 2, 1]], dtype=torch.int32), vectors_gpu=vectors, lane_meta_gpu=torch.tensor([[0, 0, 3, 3], [0, 0, 3, 1]], dtype=torch.int32), node_forces_gpu=nodes, mesh_wrenches_gpu=wrenches)

def capture(raw):
    return flat_cuda_pair_capture(raw, n=2, vertices=41, meshes=2, substeps=12, device='cpu', expected_owner_time_s=0.1)

def test_flat_pair_fp64_scatter_preserves_cancellation_and_owns_buffers():
    raw = flat()
    packet = capture(raw)
    assert packet['node_forces'].dtype == torch.float64
    torch.testing.assert_close(packet['node_forces'][0, 7, 0], torch.tensor(1 / 12, dtype=torch.float64), atol=1e-09, rtol=0)
    torch.testing.assert_close(packet['mesh_wrenches'][0, 0, 0], torch.tensor(-1 / 12, dtype=torch.float64), atol=1e-09, rtol=0)
    torch.testing.assert_close(packet['node_forces'].sum(1) + packet['mesh_wrenches'][..., :3].sum(1), torch.zeros(2, 3, dtype=torch.float64), atol=0, rtol=0)
    raw['vectors_gpu'].zero_()
    raw['indices_gpu'].zero_()
    raw['lane_meta_gpu'].zero_()
    assert packet['vectors'].abs().sum() > 0 and packet['indices'][0, 1] == 7 and (packet['pair_lane_meta'][0, 3] == 3)

def test_native_training_flat_capture_reuses_bound_reductions():
    raw = flat()
    packet = flat_cuda_pair_capture(
        raw, n=2, vertices=41, meshes=2, substeps=12, device='cpu',
        expected_owner_time_s=0.1, strict_audit=False,
    )
    assert packet['node_forces'] is raw['node_forces_gpu']
    assert packet['mesh_wrenches'] is raw['mesh_wrenches_gpu']
    assert packet['valid'].all()
    raw['lane_meta_gpu'][1, 1] = 1
    rejected = flat_cuda_pair_capture(
        raw, n=2, vertices=41, meshes=2, substeps=12, device='cpu',
        expected_owner_time_s=0.1, strict_audit=False,
    )
    assert rejected['valid'].tolist() == [True, False]
    raw['lane_meta_gpu'][1, 1] = 0
    raw['indices_gpu'][0, 0] = -1
    malformed = flat_cuda_pair_capture(
        raw, n=2, vertices=41, meshes=2, substeps=12, device='cpu',
        expected_owner_time_s=0.1, strict_audit=False,
    )
    assert malformed['valid'].tolist() == [False, False]
    raw['indices_gpu'][0, 0] = 0
    raw['node_forces_gpu'][1, 0, 0] += 0.002
    rejected = flat_cuda_pair_capture(
        raw, n=2, vertices=41, meshes=2, substeps=12, device='cpu',
        expected_owner_time_s=0.1, strict_audit=False,
    )
    assert rejected['valid'].tolist() == [True, False]


def test_empty_flat_lane_is_measured_zero_without_fabricated_pairs():
    raw = flat()
    raw['indices_gpu'] = torch.empty(0, 4, dtype=torch.int32)
    raw['vectors_gpu'] = torch.empty(0, 12)
    raw['lane_meta_gpu'][:, 3] = 0
    raw['node_forces_gpu'].zero_()
    raw['mesh_wrenches_gpu'].zero_()
    packet = capture(raw)
    assert packet['counts'].tolist() == [0, 0] and packet['vectors'].shape == (0, 12)
    assert not packet['node_forces'].any() and (not packet['mesh_wrenches'].any())

@pytest.mark.parametrize('fault', ['overflow', 'duplicate', 'index', 'time', 'nonfinite', 'native_total'])
def test_invalid_contact_evidence_cannot_become_a_reaction(fault):
    raw = flat()
    if fault == 'overflow':
        raw['lane_meta_gpu'][0, 1] = 1
    elif fault == 'duplicate':
        raw['indices_gpu'][1] = raw['indices_gpu'][0]
    elif fault == 'index':
        raw['indices_gpu'][0, 2] = 41
    elif fault == 'time':
        raw['owner_time_s_after'] += 1 / 480
    elif fault == 'nonfinite':
        raw['vectors_gpu'][0, 3] = float('nan')
    else:
        raw['node_forces_gpu'][0, 7, 0] += 0.01
    with pytest.raises(ValueError):
        capture(raw)

class Solver:
    num_envs = 2
    last_error = 0

    def __init__(self):
        self.time = 0.0
        self.initial = torch.arange(246).reshape(2, 41, 3).float() / 100
        self.p = self.initial.clone()
        self.v = torch.zeros_like(self.p)
        self.reads = 0

    def positions(self):
        return self.p

    def velocities(self):
        return self.v

    def step_free(self, dt, substeps):
        self.time = float(np.float32(self.time) + np.float32(dt))
        self.p += 0.001

    def contact_pair_samples_flat_cuda_v1(self):
        self.reads += 1
        return flat(self.time)

    def reset_envs(self, ids):
        self.p[ids] = self.initial[ids]

def owner():
    solver = Solver()
    return (solver, TensorRodOwner(solver, run_uuid='fixture', physics_step=2, episodes=('a', 'b'), mesh_names=('left.pad', 'right.pad'), device='cpu'))

def test_read_once_selected_reset_preserves_peer_and_rebinds_only_selected_source():
    (solver, rod) = owner()
    snapshot = rod.advance(physics_step=3)
    snapshot['vectors'].zero_()
    peer = solver.p[1].clone()
    time = solver.time
    rod.reset_lanes((0,), episode_uids=('new-a',))
    packet = rod.consume(physics_step=4)
    assert solver.reads == 1 and solver.time == time and torch.equal(solver.p[1], peer)
    assert packet['source_solver_id'] == id(solver) and (packet['source_physics_step'], packet['applied_physics_step']) == (3, 4)
    assert packet['rod_ticks'] == (0, 12) and packet['episode_uids'] == ('new-a', 'b') and (packet['reset_ids'] == (2, 1))
    assert packet['counts'].tolist() == [0, 1] and packet['vectors'].abs().sum() > 0
    with pytest.raises(ValueError):
        rod.consume(physics_step=4)
    with pytest.raises(ValueError):
        rod.advance(physics_step=4)
    assert solver.reads == 1

@pytest.mark.parametrize('fault', ['skip', 'unconsumed', 'clock'])
def test_invalid_owned_boundary_poisoning_prevents_reuse(fault):
    (solver, rod) = owner()
    if fault == 'skip':
        with pytest.raises(ValueError):
            rod.advance(physics_step=4)
    else:
        rod.advance(physics_step=3)
        if fault == 'clock':
            solver.time += 0.1
            with pytest.raises(ValueError):
                rod.consume(physics_step=4)
        else:
            with pytest.raises(ValueError):
                rod.advance(physics_step=4)
    with pytest.raises(ValueError):
        rod.initial_reaction(physics_step=3)

def test_grasp_contact_requires_each_substep_and_counts_uncancelled_support_load():
    from dh116.physics.runtime import GraspContactReadout

    names = (
        'left.finger14_link', 'left.finger23_link', 'left.finger33_link',
        'right.finger14_link', 'right.finger23_link', 'right.finger33_link',
        'table', 'support_0', 'support_1',
    )
    rows, vectors = [], []

    def pair(env, step, node, slot, normal, load):
        rows.append((env, step, node, slot))
        vectors.append([0, 0, 0, *normal, *(x * load for x in normal), 0, 0, 0])

    for env in range(2):
        for step in range(12):
            for side, node in enumerate((8, 32)):
                for finger, direction in enumerate((1., -1., -1.)):
                    if (env, step, side, finger) != (1, 11, 0, 2):
                        pair(env, step, node, side * 3 + finger, (direction, 0., 0.), .12)
    # These forces cancel as a net vector; both are still external contact loads.
    pair(0, 0, 14, 7, (1., 0., 0.), .1)
    pair(0, 0, 15, 7, (-1., 0., 0.), .1)
    keys = torch.tensor(rows)
    raw = torch.tensor(vectors, dtype=torch.float64)
    packet = dict(env_ids=keys[:, 0], indices=keys[:, 1:], vectors=raw,
                  normal_force=(raw[:, 3:6] * raw[:, 6:9]).sum(1))
    nodes = torch.zeros(2, 41, 3, dtype=torch.float64)
    nodes[:, :, 1] = torch.linspace(-0.25, 0.25, 41)
    readout = GraspContactReadout('tripod', names, device='cpu')
    result = readout(packet, nodes)
    torch.testing.assert_close(result['coverage_left'], torch.tensor([1., 2/3], dtype=torch.float64))
    assert result['opposed_left'].tolist() == [True, False]
    assert result['contact_nodes'][0].tolist() == [8, 32]
    assert result['bilateral_support'].tolist() == [True, False]
    assert result['off_table'].tolist() == [True, True]
    assert result['no_external_support'].tolist() == [False, True]


def test_flat_reward_readout_separates_light_contact_from_vertical_carry():
    from dh116.physics.runtime import GraspContactReadout

    names = ('left.finger14_link', 'left.finger23_link',
             'right.finger14_link', 'right.finger23_link', 'table', 'support_0', 'support_1')
    indices = [(0, 0, 8, 0), (0, 0, 20, 5), (1, 0, 32, 0)]
    vectors = [[0, 0, 0, 1, 0, 0, 0.025, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0.100, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0.025, 0, 0, 0, 0, 0]]
    for step in range(12):
        indices.extend(((2, step, 8, 0), (2, step, 32, 2), (2, step, 20, 5)))
        vectors.extend(([0, 0, 0, 1, 0, 0, 0.1, 0, 0.04, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0.1, 0, 0.02, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0.12, 0, 0, 0]))
    indices, vectors = torch.tensor(indices), torch.tensor(vectors, dtype=torch.float64)
    packet = dict(env_ids=indices[:, 0], indices=indices[:, 1:], vectors=vectors,
                  normal_force=(vectors[:, 3:6] * vectors[:, 6:9]).sum(-1))
    nodes = torch.zeros((3, 41, 3), dtype=torch.float64)
    nodes[:, :, 1] = torch.linspace(-0.25, 0.25, 41)
    readout = GraspContactReadout('pinch', names, device='cpu')
    result = readout(packet, nodes)
    assert 0 < result['load_progress_left'][0] < 1
    assert result['load_progress_left'][1] == 0
    assert not result['bilateral_support'].any()
    assert result['no_external_support'].tolist() == [False, True, False]
    torch.testing.assert_close(result['support_vertical_n'], nodes.new_tensor([0, 0, 0.12]))
    torch.testing.assert_close(result['hand_vertical_n'], nodes.new_tensor([0, 0, 0.06]))
    nodes[0, 8, 1] = 0.15  # A left material node displaced into the right world half.
    assert readout(packet, nodes)['load_progress_left'][0] == 0

def test_grasp_contact_progress_prefers_load_shared_across_required_pads():
    from dh116.physics.runtime import GraspContactReadout

    names = ('left.finger14_link', 'left.finger23_link', 'right.finger14_link',
             'right.finger23_link', 'table', 'support_0', 'support_1')
    keys = torch.tensor([(env, step, 8, pad) for env in range(2)
                         for step in range(12) for pad in range(2)])
    loads = torch.tensor([0.05, 0.0] * 12 + [0.025, 0.025] * 12, dtype=torch.float64)
    vectors = torch.zeros((len(keys), 12), dtype=loads.dtype)
    vectors[:, 3] = 1 - 2 * keys[:, 3]
    vectors[:, 6] = vectors[:, 3] * loads
    nodes = torch.zeros(2, 41, 3, dtype=loads.dtype)
    nodes[..., 1] = torch.linspace(-0.25, 0.25, 41)
    grasp = GraspContactReadout('pinch', names, device='cpu')
    result = grasp(dict(env_ids=keys[:, 0], indices=keys[:, 1:],
                        vectors=vectors, normal_force=loads), nodes)

    assert result['load_progress_left'][1] > result['load_progress_left'][0] > 0
    assert not result['bilateral_support'].any()


def test_quarantine_parks_only_the_failed_rod_without_opening_a_new_episode():
    solver, rod = owner()
    ledger = rod.advance(physics_step=3)
    peer = solver.p[1].clone()
    rod.park_rods((0,))
    assert torch.equal(solver.p[0], solver.initial[0])
    assert torch.equal(solver.p[1], peer)
    next_reaction = rod.consume(physics_step=4)
    assert ledger['counts'].tolist() == [3, 1]
    assert next_reaction['counts'].tolist() == [0, 1]
    assert next_reaction['episode_uids'] == ('a', 'b')
    assert next_reaction['reset_ids'] == (1, 1)
    assert next_reaction['rod_ticks'] == (12, 12)
    assert solver.time == ledger['pair_owner_time_s']


def test_full_rod_observations_retain_the_last_finite_row_without_hiding_validity():
    from dh116.physics.runtime import NativeTensorPhysicsRuntime

    solver, rod = owner()
    physical = NativeTensorPhysicsRuntime.__new__(NativeTensorPhysicsRuntime)
    physical.runtime, physical.device, physical.n = rod, rod.device, solver.num_envs
    physical.nodes, physical.velocity = solver.positions().clone(), solver.velocities().clone()

    solver.p[0, 20, 2] = float('nan')
    solver.p[1, 20, 2] += 0.01
    solver.v[1, 20, 2] = 0.05
    valid = physical._read_full_rod_state()

    assert valid.tolist() == [False, True]
    torch.testing.assert_close(physical.nodes[0], solver.initial[0], atol=0, rtol=0)
    torch.testing.assert_close(physical.nodes[1], solver.p[1], atol=0, rtol=0)
    torch.testing.assert_close(physical.velocity[1], solver.v[1], atol=0, rtol=0)
    solver.v[1, 20, 2] = float('inf')
    assert physical._read_full_rod_state().tolist() == [False, False]
    assert torch.isfinite(physical.nodes).all() and torch.isfinite(physical.velocity).all()


def test_native_float_clock_uses_exact_advance_once_ulp_is_large():
    from dh116.physics.solver import solver_clock_matches
    old = np.float32(8192)
    dt = np.float32(1 / 480)
    known = old + dt
    assert solver_clock_matches(float(old), float(known), steps=1)
    assert not solver_clock_matches(float(old), float(known + dt), steps=1)
    assert not solver_clock_matches(float(old), float(old), steps=1)
