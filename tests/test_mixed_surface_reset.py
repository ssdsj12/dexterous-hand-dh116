"""Mixed native reset seeds, using real TensorTask/TensorWorld reset code on CPU."""
import copy
import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_free_wrist_grasp import setup
from go2_pvcnn.control.dh116.linear_coupling import JOINTS, ACTIVE, expand
from go2_pvcnn.control.dh116.tabletop_tensor_world import TensorWorld
from go2_pvcnn.control.dh116.tabletop_surface_exploration import enable

MODULE = 'go2_pvcnn.control.dh116.tabletop_mixed_reset'


def module():
    assert importlib.util.find_spec(MODULE) is not None, 'mixed reset installation is missing'
    return importlib.import_module(MODULE)


class Hand:
    def __init__(self, n):
        self.q = torch.zeros(n, 17)
        self.qd = torch.zeros_like(self.q)
        self.pose = torch.zeros(n, 7)
        self.target = torch.zeros(n, 6)
        self.actuators = {'active': SimpleNamespace(stiffness=torch.ones(n, 6))}
    def reset(self, ids): pass
    def write_joint_state_to_sim(self, q, qd, env_ids):
        self.q[env_ids] = q; self.qd[env_ids] = qd
    def write_root_pose_to_sim(self, pose, env_ids): self.pose[env_ids] = pose
    def write_root_velocity_to_sim(self, velocity, env_ids): pass
    def set_joint_position_target(self, q, joint_ids, env_ids): self.target[env_ids] = q
    def write_joint_stiffness_to_sim(self, *args, **kwargs): pass
    def write_data_to_sim(self): pass


class Wire:
    def __init__(self, n): self.pose = torch.zeros(n, 7); self.velocity = torch.zeros(n, 6)
    def reset(self, ids): pass
    def write_root_pose_to_sim(self, pose, env_ids): self.pose[env_ids] = pose
    def write_root_velocity_to_sim(self, velocity, env_ids): self.velocity[env_ids] = velocity


def fixture(n=4, supports=False):
    t, old = setup(n)
    t.force_reward_start = 'downward_after_close'
    t.force_scoring_started.zero_()
    t._seed['force_scoring_started'] = t.force_scoring_started[0].clone()
    w = object.__new__(TensorWorld)
    w.__dict__.update(vars(old))
    w.device = torch.device('cpu')
    command = t.command[0].clone()
    other = command.clone()
    other[3] += .1; other[9] += .1
    def seed(c):
        return dict(root_xy=[.001, -.002], root_offset=.004,
            joint_positions=[expand(c[:6].tolist()).tolist(), expand(c[6:].tolist()).tolist()],
            joint_velocities=[[0.] * 11] * 2, initial_command=c.tolist(),
            grasp_command=c.tolist(), wire_pose=[0., 0., .308, 1., 0., 0., 0.],
            wire_velocity=[0.] * 6)
    layout = dict(hand_roots=dict(left=[-.1, 0., .304, 1., 0., 0., 0.],
                                  right=[.1, 0., .304, 1., 0., 0., 0.]),
        wire_center=[0., 0., .308], wire_length_m=.5, wire_radius_m=.008, wire_axis='Y',
        table_center=[0., 0., .15], table_size=[1., 1., .3])
    secondary = copy.deepcopy(layout)
    secondary['hand_roots']['left'][0] -= .02
    secondary['hand_roots']['right'][0] += .02
    second_seed = seed(other)
    primary_seed = seed(command)
    scene = None
    if supports:
        from go2_pvcnn.control.dh116.tabletop_end_supported_scene import plan_end_support_scene, translate_end_support_seed
        layout, scene = plan_end_support_scene(layout, .06)
        primary_seed = translate_end_support_seed(primary_seed, scene)
    w.origins = torch.tensor([[i * 2., (i % 2) * 3., i * .01] for i in range(n)], dtype=torch.float64)
    w.root_seed = torch.tensor(np.asarray([layout['hand_roots'][s] for s in ('left', 'right')]), dtype=torch.float64).repeat(n, 1, 1)
    w.root_seed[:, :, :3] += w.origins[:, None]
    w.root_seed[:, :, :2] += torch.tensor(primary_seed['root_xy'])
    w.goal_roots = torch.tensor(np.asarray([layout['hand_roots'][s][:3] for s in ('left', 'right')]), dtype=torch.float64)[None] + w.origins[:, None]
    w.initial_offset = .004
    named = list(range(6, 17))
    active = [6 + JOINTS.index(x) for x in ACTIVE]
    w.hands = [Hand(n), Hand(n)]
    w.seed_q = [h.q.clone() for h in w.hands]
    w.seed_qd = [h.qd.clone() for h in w.hands]
    for i in range(2): w.seed_q[i][:, named] = torch.tensor(primary_seed['joint_positions'][i], dtype=torch.float32)
    w.seed_wire = torch.tensor(primary_seed['wire_pose'], dtype=torch.float64).repeat(n, 1)
    w.seed_wire[:, :3] += w.origins
    w.seed_velocity = torch.zeros(n, 6, dtype=torch.float64)
    w.root_targets = w.root_seed.clone()
    w.commands = command.reshape(1, 2, 6).repeat(n, 1, 1)
    w.gains = torch.ones(n, 2, dtype=torch.float64) * 10.
    w.wire = Wire(n)
    w._root_pose_written = w.root_seed.float().clone()
    w.w = SimpleNamespace(layout=layout, seed=primary_seed, end_support_scene=scene,
        named_ids=[named, named], ids=[active, active], mounts=None, sensors=[],
        sim=SimpleNamespace(physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda: None)))
    for name, tail in [('link_loads', (2, 5)), ('forces', (4,)), ('vertical_components', (2, 2)),
                       ('contact_center_w', (2, 3)), ('contact_normal_w', (2, 3)),
                       ('contact_normal_load_n', (2,)), ('contact_count', (2,))]:
        setattr(w, name, torch.zeros((n, *tail), dtype=torch.float64))
    w.contact_valid = torch.zeros(n, 2, dtype=torch.bool)
    w.contact_all_below_centerline = w.contact_valid.clone()
    base = {k: v.clone() for k, v in t.m.items()}
    def refresh(sensors=False):
        w.roots = torch.stack([h.pose for h in w.hands], 1).double()
        w.q = torch.stack([h.q for h in w.hands], 1).double()
        w.qd = torch.stack([h.qd for h in w.hands], 1).double()
        w.wire_pose = w.wire.pose.double().clone()
        w.wp = w.wire_pose[:, :3]
        w.wq = w.wire_pose[:, 3:]
        w.wv = w.wire.velocity.double().clone()
        m = {k: v.clone() for k, v in base.items()}
        m.update(q=w.q[:, :, active].reshape(n, 12).clone(),
            qd=w.qd[:, :, active].reshape(n, 12).clone(), wrist_pose=w.roots.clone(),
            wrist_velocity=torch.zeros(n, 2, 6, dtype=torch.float64),
            wrist_joint_angles=torch.zeros(n, 2, 3, dtype=torch.float64))
        m['adaptive_palm_offset'] = w.palms.clone()
        return m
    w.refresh = refresh
    w.measure = lambda: refresh()
    t.free_wrist_world = w
    for name, tail, dtype in [('effective_pad_loads_n', (2, 5), torch.float64),
        ('effective_pad_vectors_w', (2, 5, 3), torch.float64),
        ('effective_pad_points_w', (2, 5, 3), torch.float64),
        ('effective_pad_valid', (2, 5), torch.bool),
        ('rejected_contact_loads_n', (2,), torch.float64)]:
        base[name] = torch.zeros((n, *tail), dtype=dtype)
        t.m[name] = base[name].clone()
    w.lift_reference_bottom_m = .36 if supports else .30
    enable(t, w)
    base.update({k: v.clone() for k, v in t.m.items()})
    ids = torch.arange(n)
    t.reset(ids, w.reset(ids))
    return t, w, secondary, second_seed


@pytest.mark.parametrize('supports', [False, True])
def test_install_keeps_first_half_and_uses_secondary_local_frame_once(supports):
    t, w, layout, seed = fixture(supports=supports)
    primary_roots = w.root_seed[:2].clone()
    primary_q = w.seed_q[0][:2].clone()
    before = copy.deepcopy((layout, seed))
    report = module().install_mixed_reset(w, t, layout, seed)
    assert report['status'] == 'passed'
    assert report['counts'] == [2, 2]
    torch.testing.assert_close(w.root_seed[:2], primary_roots)
    torch.testing.assert_close(w.seed_q[0][:2], primary_q)
    expected = torch.tensor(np.asarray([layout['hand_roots'][s] for s in ('left', 'right')]), dtype=torch.float64)
    expected[:, :2] += torch.tensor(seed['root_xy'])
    if supports: expected[:, 2] += .06
    torch.testing.assert_close(w.root_seed[2:] - torch.nn.functional.pad(w.origins[2:, None], (0, 4)), expected.expand(2, -1, -1))
    assert (layout, seed) == before
    assert t.observe().shape == (4, 351)
    assert torch.isfinite(t.observe()).all()


def test_repeated_selective_reset_restores_style_specific_commands_and_leaves_other_rows():
    t, w, layout, seed = fixture()
    module().install_mixed_reset(w, t, layout, seed)
    expected = t.command.clone()
    for ids in [torch.tensor([3, 0]), torch.tensor([2]), torch.tensor([1, 3])]:
        t.command += .02; t.previous += .03; t.reference += .04
        t.elapsed += 2.; t.joint_requested_target += .05
        other = torch.tensor([i for i in range(4) if i not in ids.tolist()])
        before = {k: getattr(t, k)[other].clone() for k in ['command', 'previous', 'reference', 'elapsed', 'joint_requested_target']}
        t.reset(ids, w.reset(ids))
        for name in ['command', 'previous', 'reference', 'old_command', 'joint_requested_target', 'joint_governed_target']:
            torch.testing.assert_close(getattr(t, name)[ids], expected[ids])
        torch.testing.assert_close(t.m['q'][ids], expected[ids], atol=1e-6, rtol=0)
        assert not t.elapsed[ids].any()
        for name, value in before.items(): torch.testing.assert_close(getattr(t, name)[other], value)
    assert not torch.equal(expected[0], expected[3])


@pytest.mark.parametrize('field,value', [('root_offset', .005), ('wire_pose', [0., 0., .4, 1., 0., 0., 0.]),
    ('wire_velocity', [1., 0., 0., 0., 0., 0.])])
def test_incompatible_object_or_offset_fails_before_mutation(field, value):
    t, w, layout, seed = fixture()
    seed[field] = value
    before = w.root_seed.clone()
    with pytest.raises(ValueError, match='wire|offset'):
        module().install_mixed_reset(w, t, layout, seed)
    assert torch.equal(w.root_seed, before)
    assert not hasattr(t, '_per_env_reset_seed')


def test_odd_count_is_rejected_without_partial_application():
    t, w, layout, seed = fixture(n=3)
    before = w.root_seed.clone()
    with pytest.raises(ValueError, match='even'):
        module().install_mixed_reset(w, t, layout, seed)
    assert torch.equal(w.root_seed, before)


def test_bad_joint_command_rejected_before_mutation():
    t, w, layout, seed = fixture()
    seed['initial_command'][0] += .05
    before = w.root_seed.clone()
    with pytest.raises(ValueError, match='joint|command'):
        module().install_mixed_reset(w, t, layout, seed)
    assert torch.equal(w.root_seed, before)


def test_validation_detects_wrong_native_readback():
    t, w, layout, seed = fixture()
    module().install_mixed_reset(w, t, layout, seed)
    original = w.refresh
    def corrupt(sensors=False):
        m = original(sensors)
        w.roots[2, 0, 0] += .01
        m['wrist_pose'][2, 0, 0] += .01
        return m
    w.refresh = corrupt
    with pytest.raises(RuntimeError, match='pose|root'):
        module().validate_mixed_reset(w, t)


def test_reference_remains_authored_grasp_while_command_matches_initial_joints():
    t, w, layout, seed = fixture()
    seed['grasp_command'][0] += .02
    module().install_mixed_reset(w, t, layout, seed)
    ids = torch.tensor([3, 1])
    t.reference.fill_(0.); t.command.fill_(0.)
    t.reset(ids, w.reset(ids))
    torch.testing.assert_close(t.reference[3], torch.tensor(seed['grasp_command'], dtype=torch.float64))
    torch.testing.assert_close(t.command[3], torch.tensor(seed['initial_command'], dtype=torch.float64))
    assert t.reference[3, 0] != t.command[3, 0]


@pytest.mark.parametrize('n', [2, 4])
def test_371d_feedback_reset_remains_finite_and_style_specific(n):
    from go2_pvcnn.control.dh116.tabletop_surface_feedback import SurfaceFeedback
    t, w, layout, seed = fixture(n=n)
    t.surface_feedback = SurfaceFeedback(n, 'cpu', dtype=torch.float64)
    t.last_observation = t.observe().clone()
    report = module().install_mixed_reset(w, t, layout, seed)
    assert report['counts'] == [n // 2, n // 2]
    assert t.observe().shape == (n, 371)
    assert torch.isfinite(t.observe()).all()
    assert not t.surface_feedback.features().any()
