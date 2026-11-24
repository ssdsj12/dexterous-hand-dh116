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
        self.reads = 0

    def positions(self):
        return self.p

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
