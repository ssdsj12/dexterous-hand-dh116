"""CPU contracts at the mounted Articulation and tensor command boundaries."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from dh116.env.contracts import JOINTS
from dh116.env.contracts import expand
from dh116.hand.kinematics import JOINT_NAMES
from dh116.hand.kinematics import ANGULAR_JOINT_NAMES
from dh116.hand.articulation import RegexArticulationSide
from dh116.hand.safety import TensorControl
from dh116.rl.teacher import SharedSourceIdentity as SharedSourceIdentity
from dh116.env.contracts import surfa_contr__ACTION_CONTRACT as ACTION_CONTRACT
NAMES = list(JOINTS + JOINT_NAMES + ANGULAR_JOINT_NAMES)

class Composer:

    def __init__(self):
        self.force = torch.full((3, 23, 3), 7.0)
        self.torque = self.force.clone()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.force[ids] = 0
        self.torque[ids] = 0

class View:

    def __init__(self):
        self.prim_paths = [f'/World/envs/env_{i}/leftHand/wrist_fixed' for i in (10, 0, 2)]
        self.q = torch.tensor([10.0, 0.0, 2.0])[:, None].repeat(1, 17)
        self.qd = self.q + 0.25
        self.targets = self.q + 100
        self.velocity = self.q + 200
        self.effort = self.q + 300
        self.root = torch.zeros(3, 7)
        self.root[:, 0] = torch.tensor([10.0, 0.0, 2.0])
        self.root[:, 6] = 1
        self.root_velocity = torch.ones(3, 6)
        self.calls = []

    def get_dof_positions(self):
        return self.q

    def get_dof_velocities(self):
        return self.qd

    def get_dof_position_targets(self):
        return self.targets

    def get_root_transforms(self):
        return self.root

    def get_root_velocities(self):
        return self.root_velocity

    def get_link_transforms(self):
        return self.root[:, None].repeat(1, 23, 1)

    def get_link_velocities(self):
        return self.root_velocity[:, None].repeat(1, 23, 1)

    def write(self, key, data, indices):
        target = getattr(self, key)
        assert data.shape == target.shape, 'SDK selected setters require a full-view payload'
        target[indices] = data[indices]
        self.calls.append((key, indices.clone()))

    def set_dof_position_targets(self, data, indices):
        self.write('targets', data, indices)

    def set_dof_velocity_targets(self, data, indices):
        self.write('velocity', data, indices)

    def set_dof_actuation_forces(self, data, indices):
        self.write('effort', data, indices)

    def apply_forces_and_torques_at_position(self, force_data, torque_data, position_data, indices, is_global):
        assert force_data.shape == torque_data.shape == (3 * 23, 3)
        self.calls.append(('force', indices.clone()))

class Hand:
    device = 'cpu'
    joint_names = NAMES
    body_names = ['base_link'] + [f'link_{i}' for i in range(22)]
    num_instances = 3
    is_initialized = True

    def __init__(self):
        self.root_physx_view = View()
        v = self.root_physx_view
        self.data = SimpleNamespace(joint_pos_target=v.targets.clone(), joint_vel_target=v.velocity.clone(), joint_effort_target=v.effort.clone())
        self.permanent_wrench_composer = Composer()
        self.instantaneous_wrench_composer = Composer()
        self.actuators = {'active': SimpleNamespace(stiffness=torch.ones(3, 6))}

    def reset(self, env_ids):
        self.permanent_wrench_composer.reset(env_ids)
        self.instantaneous_wrench_composer.reset(env_ids)

    def cache(self, key, value, joint_ids, env_ids):
        ids = torch.arange(17) if joint_ids is None else torch.as_tensor(joint_ids)
        getattr(self.data, key)[env_ids[:, None], ids] = value

    def set_joint_position_target(self, value, joint_ids=None, env_ids=None):
        self.cache('joint_pos_target', value, joint_ids, env_ids)

    def set_joint_velocity_target(self, value, joint_ids=None, env_ids=None):
        self.cache('joint_vel_target', value, joint_ids, env_ids)

    def set_joint_effort_target(self, value, joint_ids=None, env_ids=None):
        self.cache('joint_effort_target', value, joint_ids, env_ids)

    def write_joint_state_to_sim(self, q, qd, env_ids):
        self.root_physx_view.q[env_ids] = q
        self.root_physx_view.qd[env_ids] = qd

    def write_root_pose_to_sim(self, pose, env_ids):
        self.root_physx_view.root[env_ids] = pose[:, [0, 1, 2, 4, 5, 6, 3]]

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.root_physx_view.root_velocity[env_ids] = velocity

    def write_joint_stiffness_to_sim(self, *args, **kwargs):
        pass

    def write_joint_effort_limit_to_sim(self, *args, **kwargs):
        pass

    def write_data_to_sim(self):
        raise AssertionError('selected reset must preserve other instantaneous wrenches')

def articulation():
    hand = Hand()
    ids = (0, 2, 10)
    return (RegexArticulationSide(hand, ids, tuple((f'/World/envs/env_{i}' for i in ids)), '/wrist_fixed'), hand)

def test_actual_regex_row_mapping_owns_reads_and_orders_selected_targets():
    (side, hand) = articulation()
    state = side.read()
    assert state['q'][:, 0].tolist() == [0, 2, 10] and state['root_pose_wxyz'][:, 3].tolist() == [1, 1, 1]
    hand.root_physx_view.q.add_(40)
    assert state['q'][:, 0].tolist() == [0, 2, 10]
    side.set_joint_position_target(torch.tensor([[31.0, 32.0], [41.0, 42.0]]), joint_ids=[0, 3], env_ids=[10, 0])
    assert hand.data.joint_pos_target[:2, [0, 3]].tolist() == [[31, 32], [41, 42]]
    for invalid in ([2, 2], [3], [2.5]):
        with pytest.raises(ValueError):
            side.rows(invalid)

def test_selected_reset_preserves_peer_state_targets_and_pending_reactions():
    (side, hand) = articulation()
    before = side.read()
    force = hand.instantaneous_wrench_composer.force.clone()
    q = torch.zeros(1, 17)
    q[:, :11] = torch.tensor(expand([0.2] * 6))
    qd = torch.zeros_like(q)
    pose = torch.tensor([[0.3, 0.4, 0.5, 1.0, 0.0, 0.0, 0.0]])
    side.reset(env_ids=[2], root_pose_wxyz=pose, joint_pos=q, joint_vel=qd, active_effort_limit_nm=0.2)
    after = side.read()
    for key in ('q', 'qd', 'backend_targets', 'root_pose_wxyz', 'root_velocity'):
        torch.testing.assert_close(after[key][[0, 2]], before[key][[0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(after['backend_targets'][1], q[0], rtol=0, atol=0)
    torch.testing.assert_close(after['root_pose_wxyz'][1], pose[0], rtol=0, atol=0)
    assert torch.equal(hand.instantaneous_wrench_composer.force[:2], force[:2]) and (not hand.instantaneous_wrench_composer.force[2].any())
    assert all((rows.tolist() == [2] for (_, rows) in hand.root_physx_view.calls))

def control_fixture(*, soft_constraints=False):
    writes = []
    sides = {}
    for name in ('left', 'right'):
        sides[name] = SimpleNamespace(joint_names=NAMES, hand=SimpleNamespace(permanent_wrench_composer=SimpleNamespace(reset=lambda : writes.append('reset'))), set_wrenches=lambda **kw: writes.append(kw), set_joint_position_target=lambda *a, **kw: writes.append('target'))
    if soft_constraints:
        limits = torch.tensor([-1.0, 2.0]).expand(2, 17, 2)
        for side in sides.values():
            side.row_indices = torch.arange(2)
            side.hand.root_physx_view = SimpleNamespace(get_dof_limits=lambda: limits)
    batch = SimpleNamespace(sides=sides, write_data_to_sim=lambda : writes.append('write'))
    roots = torch.zeros(2, 2, 7, dtype=torch.float64)
    roots[:, :, 3] = 1
    roots[:, :, 2] = 0.6
    roots[:, 0, 0] = -0.1
    roots[:, 1, 0] = 0.1
    q = torch.zeros(2, 2, 17, dtype=torch.float32)
    q[:, :, :11] = torch.tensor(expand([0.2] * 6))
    state = dict(q=q, qd=torch.zeros_like(q), backend_targets=q.clone(), root_pose_wxyz=roots.clone(), link_pose_xyzw=torch.zeros(2, 2, 1, 7, dtype=torch.float64), body_com_pose_w=torch.zeros(2, 2, 1, 7, dtype=torch.float64))
    control = TensorControl(batch, roots, {'left.pad': dict(side='left', body_index=0),
                                           'right.pad': dict(side='right', body_index=0)},
                            ('left.pad', 'right.pad', 'table'), isolate_constraints=soft_constraints,
                            soft_constraints=soft_constraints)
    stamp = SharedSourceIdentity(1, 2, (0, 1), 'run', ('a', 'b'), ('rod0', 'rod1'), (1, 1), (0, 0), 2, 2 / 480, 1 / 480, 1 / 480, 2 / 480, 4.0, 'a' * 64, 'cpu_fixture')
    action = torch.full((2, 24), 0.2)
    action[1] *= -1
    tr = control.prepare(action, state, torch.full((2, 2), 0.01), roots, stamp=stamp)
    w = torch.zeros(2, 3, 6, dtype=torch.float64)
    source = dict(source_solver_id=1, parent_num_envs=2, run_uuid='run', source_physics_step=2, applied_physics_step=3, rod_ticks=(0, 0), episode_uids=('a', 'b'), reset_ids=(1, 1), solver_time_s=4.0, mesh_wrenches=w)
    return (control, stamp, state, tr, w, source, writes)

def test_all24_commands_and_com_wrench_reach_one_owned_sdk_transaction():
    (control, stamp, state, tr, w, source, writes) = control_fixture()
    finger = 0.2 + torch.tensor([0.2, -0.2])[:, None] * ACTION_CONTRACT['joint_rate_rad_s'] / 480
    for expected in tr['expected']:
        torch.testing.assert_close(expected[:, :6], finger.expand(-1, 6).float(), atol=1e-07, rtol=0)
        for (lo, rate) in ((6, 'translation_rate_m_s'), (9, 'rotation_rate_rad_s')):
            delta = torch.tensor([0.2, -0.2])[:, None].expand(-1, 3) * ACTION_CONTRACT[rate] / 480
            torch.testing.assert_close(expected[:, lo:lo + 3], delta, atol=1e-09, rtol=1e-05)
    state['link_pose_xyzw'][..., 0] = 48.125
    state['body_com_pose_w'][..., 0] = 48.13500000023
    w[:, :, 1] = 2.0
    w[:, :, 5] = 0.125
    control.submit(tr, stamp=stamp, wrenches=w, state=state, reaction_source=source)
    issued = [row for row in writes if isinstance(row, dict)]
    for row in issued:
        torch.testing.assert_close(row['torques'][:, :, 2], torch.full((2, 1), 0.10499999954).float(), rtol=0, atol=0)
    assert writes.count('write') == 1 and len(issued) == 2
    for s in (0, 1):
        state['backend_targets'][:, s, control.write_ids[s]] = tr['expected'][s]
    after = replace(stamp, sdk_step=3, rod_ticks=(12, 12), solver_time_s=4 + 1 / 480)
    assert not control.finish(tr, state, stamp=after).any()
    with pytest.raises(ValueError):
        control.submit(tr, stamp=stamp, wrenches=w, state=state, reaction_source=source)

@pytest.mark.parametrize('fault', ['source', 'buffer', 'force_limit', 'nonfinite_com'])
def test_reaction_faults_are_rejected_before_any_sdk_write(fault):
    (control, stamp, state, tr, w, source, writes) = control_fixture()
    if fault == 'source':
        source['reset_ids'] = (2, 1)
    elif fault == 'buffer':
        source['mesh_wrenches'] = w.clone()
    elif fault == 'force_limit':
        w[0, 0, 0] = 30 + 1e-07
    else:
        state['body_com_pose_w'][0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        control.submit(tr, stamp=stamp, wrenches=w, state=state, reaction_source=source)
    assert not writes

def test_full_control_rejects_only_the_unsafe_reaction_row_before_sdk_write():
    control, stamp, state, tr, w, source, writes = control_fixture()
    control.isolate_constraints = True
    old = [row[1].clone() for row in tr['expected']]
    w[0, 0, 0] = 30 + 1e-7
    w[1, 0, 0] = 7
    control.submit(tr, stamp=stamp, wrenches=w, state=state, reaction_source=source)
    assert tr['constraint_failure'].tolist() == [True, False]
    issued = [row for row in writes if isinstance(row, dict)]
    assert issued[0]['forces'][:, 0, 0].tolist() == [0, 7]
    for side, expected in enumerate(tr['expected']):
        torch.testing.assert_close(expected[0], state['backend_targets'][0, side, control.write_ids[side]])
        torch.testing.assert_close(expected[1], old[side], atol=0, rtol=0)
    assert w[0, 0, 0] > 30


def test_soft_training_caps_submitted_force_without_altering_evidence_or_terminating():
    control, stamp, state, transaction, wrench, source, writes = control_fixture(soft_constraints=True)
    wrench[0, 0, :3] = torch.tensor([60.0, 80.0, 0.0])
    wrench[0, 0, 5] = 5.0
    wrench[1, 0, 1] = 17.0
    measured_wrench = wrench.clone()
    control.submit(transaction, stamp=stamp, wrenches=wrench, state=state, reaction_source=source)
    left = next(row for row in writes if isinstance(row, dict))
    torch.testing.assert_close(left['forces'][:, 0], torch.tensor([[24., 32., 0.], [0., 17., 0.]]))
    torch.testing.assert_close(left['torques'][:, 0, 2], torch.tensor([2.0, 0.0]))
    assert torch.equal(wrench, measured_wrench)
    assert not transaction['constraint_failure'].any()
    after = replace(stamp, sdk_step=3, rod_ticks=(12, 12), solver_time_s=4 + 1 / 480)
    joint = control.active[0][1]
    state['q'][:, 0, joint] = torch.tensor([-1.24, -1.26])
    assert control.finish(transaction, state, stamp=after).tolist() == [False, True]
    assert transaction['soft_joint_limit'].tolist() == [1.0, 1.0]


def test_wrist_projection_matches_issued_mount_targets_at_rotated_hard_stop():
    control, stamp, state, _, _, _, _ = control_fixture(soft_constraints=True)
    control.invalidate()
    # 45 degrees about Y means that independently clipping the local X
    # drive reduces world Z for this target.
    angle = torch.tensor(torch.pi / 8, dtype=torch.float64)
    anchors = state['root_pose_wxyz']
    anchors[..., 3] = angle.cos()
    anchors[..., 5] = angle.sin()
    control.palms[:] = torch.tensor([-0.089, 0., 0.089])
    base = anchors.clone()
    base[..., :3] += control.palms
    transaction = control.prepare(torch.zeros(2, 24), state, torch.full((2, 2), 0.0021, dtype=torch.float64),
                                  base, stamp=stamp)
    local = torch.stack([x[:, 6:9].double() for x in transaction['expected']], dim=1)
    c = s = 2 ** -0.5
    world = anchors[..., :3] + torch.stack((local[..., 0] * c + local[..., 2] * s, local[..., 1],
                                           -local[..., 0] * s + local[..., 2] * c), dim=-1)
    projection = transaction['projection']
    assert local.abs().max() <= 0.100000002
    assert not transaction['constraint_failure'].any()
    assert (world[..., 2] >= base[..., 2] - 1e-8).all()
    torch.testing.assert_close(projection['projected_root_poses'][..., :3], world, atol=1e-9, rtol=0)
    torch.testing.assert_close(projection['palm_offsets_m'], world - control.roots[..., :3], atol=1e-9, rtol=0)



def test_clamp_lift_dwell_is_consecutive_and_lane_local():
    from dh116.hand.safety import stable_clamp_step
    ticks = torch.zeros(2, dtype=torch.long)
    for _ in range(95):
        ticks, unlocked = stable_clamp_step(ticks, torch.tensor([True, True]), required_steps=96)
    assert not unlocked.any()
    ticks, unlocked = stable_clamp_step(ticks, torch.tensor([True, False]), required_steps=96)
    assert unlocked.tolist() == [True, False]
    assert ticks.tolist() == [96, 0]
    ticks, unlocked = stable_clamp_step(ticks, torch.tensor([False, True]), required_steps=96)
    assert ticks.tolist() == [0, 1] and not unlocked.any()



def test_reduced_dof_wrist_alignment_does_not_need_finger_arrival():
    from dh116.physics.runtime import NativeTensorPhysicsRuntime
    fields = dict(wrist_goal_m=torch.tensor([0.009, 0.001, 0.012]),
                  wrist_goal_rad=torch.tensor([0.1, 0.2, 0.1]))
    assert NativeTensorPhysicsRuntime.wrist_aligned(fields).tolist() == [True, False, False]


def test_coordinated_lift_preserves_clamp_until_lost_and_shares_mount_headroom():
    control, stamp, state, _, _, _, _ = control_fixture(soft_constraints=True)
    control.invalidate()
    control.fixed_fingers = (3, 4, 5, 9, 10, 11)
    anchors = state['root_pose_wxyz']
    angle = torch.tensor(torch.pi / 8, dtype=torch.float64)
    anchors[0, 0, 3] = angle.cos()
    anchors[0, 0, 5] = angle.sin()
    root = control.roots
    control.palms[0, 0, 0] = -0.085
    control.palms[0, 0, 2] = 0.1 * 2 ** .5 - 0.085 - 0.000025
    measured = root.clone()
    measured[..., :3] += control.palms
    pose = torch.cat([state['q'][:, side, control.active[side]].double()
                      for side in (0, 1)], dim=1)
    action = torch.zeros(2, 24)
    action[:, :12] = -1
    action[:, (14, 17)] = torch.tensor([1., -.25])
    actor_request = action.clone()

    def prepare(clamped, reference, ready, clearance=.01):
        control.pending = None
        return control.prepare(action, state, torch.full((2, 2), clearance, dtype=torch.float64),
                               measured, stamp=stamp, grasp_lift=(clamped, reference, ready))

    clamped = torch.tensor([True, False])
    ready = torch.tensor([True, False])
    reference = pose.clone()
    reference[1] = torch.nan
    good = prepare(clamped, reference, ready)
    dz = good['projection']['projected_root_poses'][..., 2] - measured[..., 2]
    torch.testing.assert_close(dz[0, 0], dz[0, 1], rtol=0, atol=1e-8)
    assert 0 < dz[0, 0] <= .000026
    assert dz[1, 0] > 0 > dz[1, 1]
    for side in (0, 1):
        first = state['backend_targets'][0, side, control.active[side][0]]
        assert 0 < first - good['expected'][side][0, 0] < .001
        torch.testing.assert_close(good['expected'][side][:, 3:6],
                                   state['backend_targets'][:, side, control.active[side][3:]])
    assert torch.equal(action, actor_request)
    assert not good['constraint_failure'].any()

    # Losing either side freezes wrist motion immediately and releases the
    # narrow finger target envelope so the next contact can be found.
    reference[0] = torch.nan
    stopped = prepare(clamped, reference, torch.zeros_like(ready))
    dz = stopped['projection']['projected_root_poses'][..., 2] - measured[..., 2]
    torch.testing.assert_close(dz[0], torch.zeros_like(dz[0]), atol=1e-8, rtol=0)
    assert (state['backend_targets'][0, 0, control.active[0][0]]
            - stopped['expected'][0][0, 0]) > .002

    # On re-acquisition the measured shape can differ from the old command.
    # Bring the target back at its normal rate before permitting lift.
    reference[0] = pose[0]
    state['backend_targets'][0, 0, control.active[0][0]] += .09
    catchup = prepare(clamped, reference, ready)
    dz = catchup['projection']['projected_root_poses'][..., 2] - measured[..., 2]
    torch.testing.assert_close(dz[0], torch.zeros_like(dz[0]), atol=1e-8, rtol=0)
    drift = state['backend_targets'][0, 0, control.active[0][0]] - catchup['expected'][0][0, 0]
    assert .002 < drift <= .002501

    # Infeasible hold at the table must go through the original safe projection.
    blocked = prepare(clamped, reference, torch.zeros_like(ready), clearance=.0019)
    assert blocked['constraint_failure'][0]
    required_z = root[..., 2] + blocked['projection']['safe_z_offset_m']
    assert (blocked['projection']['projected_root_poses'][0, :, 2] >= required_z[0] - 1e-8).all()


def test_coordinated_lift_recaptures_measured_pose_only_after_full_reacquisition():
    from dh116.physics.runtime import NativeTensorPhysicsRuntime

    runtime = NativeTensorPhysicsRuntime.__new__(NativeTensorPhysicsRuntime)
    control, _, state, _, _, _, _ = control_fixture(soft_constraints=True)
    runtime.ctrl, runtime.state = control, state
    runtime.gated_lift = True
    runtime._clamp_dwell_steps = 2
    runtime._clamp_steps = torch.zeros(2, dtype=torch.long)
    runtime._stable_clamp_s = torch.zeros(2, dtype=torch.float64)
    runtime._lift_unlocked = torch.zeros(2, dtype=torch.bool)
    runtime._grasp_latched = torch.zeros(2, dtype=torch.bool)
    runtime._grasp_pose = torch.full((2, 12), torch.nan, dtype=torch.float64)

    def advance(left, right):
        runtime._count_clamp(torch.tensor([left, right]))

    advance(True, True)
    assert runtime._grasp_pose.isnan().all()
    advance(True, True)
    original = runtime._grasp_pose.clone()
    assert runtime._lift_unlocked.tolist() == [True, True]
    state['q'][0, 0, control.active[0][0]] = .3
    state['backend_targets'][0, 0, control.active[0][0]] = .6
    advance(False, True)
    assert runtime._grasp_latched.tolist() == [True, True]
    assert runtime._grasp_pose[0].isnan().all()
    advance(True, True)
    assert not runtime._lift_unlocked[0]
    advance(True, True)
    assert runtime._lift_unlocked.tolist() == [True, True]
    torch.testing.assert_close(runtime._grasp_pose[0, 0], torch.tensor(.3, dtype=torch.float64))
    assert torch.equal(runtime._grasp_pose[1], original[1])

@pytest.mark.parametrize('style,digits', [('pinch', 2), ('tripod', 3), ('power', 5)])
def test_reference_closure_preserves_authority_and_regulates_individual_pads(style, digits):
    from dh116.hand.safety import guided_closure_action
    q = torch.zeros((2, 12), dtype=torch.float64)
    reference = q[0].reshape(2, 6)
    request = torch.zeros((2, 24), dtype=q.dtype)
    loads = q.new_zeros(2, 2, digits + (style == 'power'))
    enabled = torch.tensor([True, False])
    guided = guided_closure_action(request, q, reference, loads, style, enabled)
    assert (guided[0, 1:digits + 1] > 0).all() and guided[0, 0] == 0
    assert guided[1].count_nonzero() == 0
    assert torch.equal(guided[:, 12:], request[:, 12:])
    loads[0, 0, 0] = 2.0
    guided = guided_closure_action(request, q, reference, loads, style, enabled)
    assert guided[0, 1] < 0 and guided[0, 2] > 0
    request[:, 1::2] = -1
    request[:, ::2] = 1
    assert torch.equal(request, guided_closure_action(request, q, reference, loads, style, enabled))
