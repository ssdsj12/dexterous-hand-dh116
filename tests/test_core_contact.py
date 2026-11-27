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
    with pytest.raises(ValueError, match='indices'):
        flat_cuda_pair_capture(
            raw, n=2, vertices=41, meshes=2, substeps=12, device='cpu',
            expected_owner_time_s=0.1, strict_audit=False,
        )
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
    torch.testing.assert_close(result['balanced_hand_vertical_n'], nodes.new_tensor([0, 0, 0.04]))
    # A lateral normal with upward friction and a tilted normal are different mechanisms.
    readout.trace_pad_vertical = True
    vectors[(indices[:, 0] == 2) & (indices[:, 3] == 2), 3:6] = nodes.new_tensor([0.8, 0, 0.6])
    packet['normal_force'] = (vectors[:, 3:6] * vectors[:, 6:9]).sum(-1)
    readout(packet, nodes)
    diag = readout.last_diagnostics
    torch.testing.assert_close(diag['pad_normal_z_mean'][2, :, 0], nodes.new_tensor([0, 0.6]))
    torch.testing.assert_close(diag['pad_normal_vertical_n'][2, :, 0], nodes.new_tensor([0, 0.0552]))
    torch.testing.assert_close(diag['pad_tangential_vertical_n'][2, :, 0], nodes.new_tensor([0.04, -0.0352]))
    assert not diag['pad_normal_z_mean'][:, :, 1].any()
    assert not readout(packet, nodes)['bilateral_support'].any()
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


def test_pinch_opposition_progress_uses_loaded_contact_directions():
    from dh116.physics.runtime import GraspContactReadout

    names = ('left.finger14_link', 'left.finger23_link',
             'right.finger14_link', 'right.finger23_link', 'table', 'support_0', 'support_1')
    directions = torch.tensor(((1., 0., 0.), (0., 1., 0.), (-1., 0., 0.)), dtype=torch.float64)
    keys = torch.tensor([(env, step, node, slot) for env in range(3) for step in range(12)
                         for node, slot in ((8, 0), (8, 1), (32, 2), (32, 3))])
    normals = directions[keys[:, 0]].clone()
    normals[keys[:, 3] % 2 == 0] = directions[0]
    vectors = torch.zeros(len(keys), 12, dtype=normals.dtype)
    vectors[:, 3:6] = normals
    vectors[:, 6:9] = normals * .16
    nodes = torch.zeros(3, 41, 3, dtype=normals.dtype)
    nodes[..., 1] = torch.linspace(-.25, .25, 41)
    readout = GraspContactReadout('pinch', names, device='cpu')
    result = readout(dict(env_ids=keys[:, 0], indices=keys[:, 1:], vectors=vectors,
                          normal_force=vectors.new_full((len(keys),), .16)), nodes)

    parallel, partial, opposed = result['load_progress_left']
    assert 0 < parallel < partial < opposed <= 1
    assert result['bilateral_support'].tolist() == [False, False, True]


def test_power_opposition_needs_a_loaded_palm_and_finger_face():
    from dh116.physics.runtime import GraspContactReadout

    pads = (*GraspContactReadout.PADS, 'base_link')
    names = tuple(f'{side}.{pad}' for side in ('left', 'right') for pad in pads)
    names += GraspContactReadout.SUPPORTS
    cases = ((5,), (0, 1), (0, 5), tuple(range(6)))
    keys = torch.tensor([(env, step, node, 6 * side + pad)
                         for env, loaded in enumerate(cases) for step in range(12)
                         for side, node in enumerate((8, 32)) for pad in loaded])
    vectors = torch.zeros(len(keys), 12, dtype=torch.float64)
    vectors[:, 3] = torch.where(keys[:, 3] % 6 % 2 == 0, 1., -1.)
    vectors[:, 6] = vectors[:, 3] * .12
    nodes = torch.zeros(len(cases), 41, 3, dtype=vectors.dtype)
    nodes[..., 1] = torch.linspace(-.25, .25, 41)
    readout = GraspContactReadout('power', names, device='cpu')
    result = readout(dict(env_ids=keys[:, 0], indices=keys[:, 1:], vectors=vectors,
                          normal_force=vectors.new_full((len(keys),), .12)), nodes)

    palm_only, fingers_only, palm_and_finger, complete = result['load_progress_left']
    assert max(palm_only, fingers_only) < palm_and_finger < complete <= 1
    assert result['bilateral_support'].tolist() == [False, False, False, True]


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



def test_flat_global_count_mismatch_and_local_nonfinite_are_distinct():
    raw = flat()
    options = dict(n=2, vertices=41, meshes=2, substeps=12, device='cpu',
                   expected_owner_time_s=0.1, strict_audit=False)
    raw['indices_gpu'][3, 0] = 0
    with pytest.raises(ValueError, match='lane counts'):
        flat_cuda_pair_capture(raw, **options)
    raw['indices_gpu'][3, 0] = 1
    raw['vectors_gpu'][3, 7] = float('nan')
    packet = flat_cuda_pair_capture(raw, **options)
    assert packet['valid'].tolist() == [True, False]
    assert packet['lane_failure_bits'].tolist() == [0, 2]
    assert packet['env_ids'].tolist() == [0, 0, 0]
    assert not packet['mesh_wrenches'][1].any()
    assert not packet['node_forces'][1].any()


@pytest.mark.parametrize('force_band_n', (None, (.4, 2.5)))
def test_grip_balance_uses_substep_normal_loads_and_grasp_specific_ratios(force_band_n):
    from dh116.physics.runtime import GraspContactReadout

    for grasp, target in (('pinch', (.4, .4)), ('tripod', (.4, .2, .2)),
                          ('power', (.06, .12, .12, .12, .12, .26))):
        count = len(target)
        pads = (*GraspContactReadout.PADS, 'base_link')[:count]
        names = tuple(f'{side}.{pad}' for side in ('left', 'right') for pad in pads)
        names += GraspContactReadout.SUPPORTS
        loads = torch.tensor(target, dtype=torch.float64).expand(9, 12, 2, count).clone()
        loads[1, :, 1] *= .5
        loads[2, :, :, 0] += .05
        loads[2, :, :, 1] -= .05
        loads[3, :, :, -1] = 0
        loads[4] *= 5
        loads[5, 1:] = 0
        loads[6] = 0
        if grasp == 'tripod':
            loads[7, :, :, 1] += .09
            loads[7, :, :, 2] -= .09
        keys = torch.tensor([(env, step, node, side * count + pad)
                             for env in range(len(loads)) for step in range(12)
                             for side, node in enumerate((8, 32)) for pad in range(count)])
        force = loads.flatten()
        vectors = force.new_zeros((len(keys), 12))
        positive = (keys[:, 3] % count == 0)
        if grasp == 'power':
            positive |= keys[:, 3] % count == 5
        vectors[:, 3] = torch.where(positive | (keys[:, 0] == 8), 1., -1.)
        vectors[:, 6] = force * vectors[:, 3]
        nodes = force.new_zeros((len(loads), 41, 3))
        nodes[..., 1] = torch.linspace(-.25, .25, 41)
        readout = GraspContactReadout(grasp, names, device='cpu', balance_enabled=True,
                                      force_band_n=force_band_n)
        result = readout(dict(env_ids=keys[:, 0], indices=keys[:, 1:],
                              vectors=vectors, normal_force=force), nodes)
        score = result['grip_balance']
        assert score[0] == 1
        assert 0 < score[1] < score[0]
        if grasp != 'power':
            assert 0 < score[2] < score[0]
        else:
            assert score[2] == score[0]
        if grasp == 'tripod':
            assert 0 < score[7] < score[0]
        assert score[3] == score[6] == score[8] == 0
        assert result['bilateral_support'][4]
        torch.testing.assert_close(score[4], score.new_tensor(1.0 if force_band_n is None else .4))
        torch.testing.assert_close(score[5], score[0] / 12)


def test_power_wrap_is_independent_of_load_shares_and_rejects_empty_and_wrong_side_grasps():
    from dh116.physics.runtime import GraspContactReadout

    names = tuple(side + '.' + pad for side in ('left', 'right')
                  for pad in (*GraspContactReadout.PADS, 'base_link', 'finger52_link'))
    readout = GraspContactReadout('power', (*names, 'table', 'support_0', 'support_1'), device='cpu')
    nodes = torch.zeros(4, 41, 3, dtype=torch.float64)
    nodes[..., 1] = torch.linspace(-.25, .25, 41)
    # Lanes have two real middle-phalanx contacts, no right contact, a wrong-side
    # right contact, or a fist around empty space with both hands brushing the wire.
    keys = torch.tensor([(0, 0, 7, 6), (0, 0, 32, 13), (1, 0, 7, 6),
                         (2, 0, 7, 6), (2, 0, 7, 13), (3, 0, 7, 6), (3, 0, 32, 13)])
    enclosed = torch.ones(4, 2, dtype=torch.bool)
    enclosed[3] = False
    for f in (.001, 2.0):
        w = nodes.new_zeros(len(keys), 12)
        w[:, 3], w[:, 6] = 1., f
        out = readout(dict(env_ids=keys[:, 0], indices=keys[:, 1:], vectors=w,
                           normal_force=w[:, 6]), nodes, wrap=(enclosed, enclosed.double()))
        assert out['bilateral_support'].tolist() == [True, False, False, False]
        assert out['contact_nodes'][0].tolist() == [7, 32]
        assert not out['non_target_normal_n'].any()


def test_power_wrap_requires_measured_closure_and_circumferential_enclosure():
    from dh116.physics.runtime import GraspContactReadout

    nodes = torch.zeros(4, 41, 3)
    nodes[..., 1] = torch.linspace(-.25, .25, 41)
    indices = torch.tensor([7, 32]).view(1, 2, 1).expand(4, 2, 6)
    theta = torch.arange(6) * (torch.pi / 3)
    normals = torch.stack((theta.cos(), torch.zeros(6), theta.sin()), -1).repeat(4, 2, 1, 1)
    tangent = nodes.new_tensor([0, 1, 0]).expand(4, 2, 3)
    near = torch.ones(4, 2, 6)
    error = torch.zeros(4, 2)
    normals[1] = normals[1, :, :1].clone()
    near[2, :, 5] = 0.
    error[3] = .3
    enclosed, _ = GraspContactReadout.power_wrap(nodes, tangent, indices, near * 0, normals,
                                                  near, torch.full((4,), .005), error)
    assert enclosed.tolist() == [[True, True], [False, False], [False, False], [False, False]]
