"""Bounded adaptive grasp controls; these are not Isaac grasp evidence."""
import ast
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits, expand
from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure


def helper():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_adaptive_backend')


def step(**overrides):
    values = dict(palm_offset=np.zeros((2, 3)), stiffness=np.full(2, 10.),
                  palm_action=np.ones((2, 3)), grip_action=np.ones(2),
                  hand_clearance=np.full(2, .01), dt=1/480,
                  palm_motion_allowed=True)
    values.update(overrides)
    return helper().adaptive_step(**values)


def test_rate_limits_each_axis_and_each_hand_stiffness():
    palm, stiffness = step(palm_action=np.array([[1, -1, 1], [-1, 1, -1]]),
                           grip_action=np.array([1., -1.]))
    assert np.allclose(palm, np.array([[1, -1, 1], [-1, 1, -1]])*.002/480)
    assert stiffness == pytest.approx([10+10/480, 10])


def test_long_command_reaches_absolute_seed_bounds_without_integrator_drift():
    palm, stiffness = np.zeros((2, 3)), np.full(2, 10.)
    for _ in range(1500):
        palm, stiffness = step(palm_offset=palm, stiffness=stiffness)
    assert palm == pytest.approx(np.tile([.003, .003, .004], (2, 1)))
    assert stiffness == pytest.approx([20., 20.])
    for _ in range(1500):
        palm, stiffness = step(palm_offset=palm, stiffness=stiffness,
                               palm_action=-np.ones((2, 3)), grip_action=-np.ones(2))
    assert np.all(palm >= -np.array([.003, .003, .004]))
    assert stiffness == pytest.approx([10., 10.])


def test_palm_gate_freezes_position_but_force_action_remains_controllable():
    initial = np.array([[.001, -.001, .002], [-.002, .001, -.003]])
    palm, stiffness = step(palm_offset=initial, palm_motion_allowed=False)
    assert np.array_equal(palm, initial)
    assert stiffness == pytest.approx([10+10/480, 10+10/480])
    assert not np.shares_memory(palm, initial)


def test_descent_uses_each_hand_clearance_and_never_blocks_up_or_xy():
    palm, _ = step(palm_action=np.array([[1., 1., -1.], [1., 1., -1.]]),
                   hand_clearance=np.array([.002101, .002]))
    assert palm[:, :2] == pytest.approx(np.full((2, 2), .002/480))
    assert palm[:, 2] == pytest.approx([-1e-6, 0.])
    up, _ = step(hand_clearance=np.array([-.0005, 0.]))
    assert up[:, 2] == pytest.approx([.002/480, .002/480])


def test_missing_optional_actions_hold_existing_state():
    initial = np.array([[.001, -.001, .002], [-.002, .001, -.003]])
    palm, stiffness = step(palm_offset=initial, stiffness=np.array([12., 18.]),
                           palm_action=None, grip_action=None)
    assert np.array_equal(palm, initial)
    assert stiffness == pytest.approx([12., 18.])


@pytest.mark.parametrize('override', [
    dict(palm_action=np.zeros(6)), dict(palm_action=np.full((2, 3), np.nan)),
    dict(palm_action=np.full((2, 3), 1.01)), dict(grip_action=[0.]),
    dict(grip_action=[0., np.inf]), dict(grip_action=[-1.01, 0.]),
    dict(palm_offset=np.full((2, 3), .005)), dict(stiffness=[9., 10.]),
    dict(stiffness=[20.01, 10.]), dict(hand_clearance=[np.nan, .01]),
    dict(hand_clearance=[.01]), dict(dt=0.), dict(dt=-.1), dict(dt=np.inf),
    dict(dt=.1), dict(palm_motion_allowed='yes'),
])
def test_invalid_control_state_or_action_is_rejected_before_step(override):
    with pytest.raises(ValueError):
        step(**override)


def test_pad_relative_uses_measured_link_rotation_and_cancels_env_translation():
    poses = np.tile([0., 0., 0., 1., 0., 0., 0.], (4, 1))
    poses[0] = [1., 2., 3., np.sqrt(.5), 0., 0., np.sqrt(.5)]
    points = np.zeros((4, 3)); points[0] = [1., 0., 0.]
    wire = np.array([1., 2., 3.])
    before = helper().pad_relative_positions(poses, points, wire)
    assert before[0] == pytest.approx([0., 1., 0.])
    poses[:, :3] += [4., 6., 0.]
    after = helper().pad_relative_positions(poses, points, wire+[4., 6., 0.])
    assert after == pytest.approx(before)


@pytest.mark.parametrize('which', ['pose_shape', 'quaternion', 'local_nan', 'wire_nan'])
def test_pad_geometry_rejects_invalid_physical_data(which):
    poses = np.tile([0., 0., 0., 1., 0., 0., 0.], (4, 1))
    local, wire = np.zeros((4, 3)), np.zeros(3)
    if which == 'pose_shape': poses = poses[:3]
    if which == 'quaternion': poses[0, 3:] = 0.
    if which == 'local_nan': local[0, 0] = np.nan
    if which == 'wire_nan': wire[0] = np.nan
    with pytest.raises(ValueError):
        helper().pad_relative_positions(poses, local, wire)


def test_seed_grasp_override_is_independent_and_coupled_limit_checked():
    low, high = active_limits()
    default = np.tile((low+high)/2, 2)
    override = default+.001
    assert np.array_equal(helper().select_grasp_command(default, {}), default)
    actual = helper().select_grasp_command(default, {'grasp_command': override})
    assert np.array_equal(actual, override) and not np.shares_memory(actual, override)
    with pytest.raises(ValueError):
        helper().select_grasp_command(default, {'grasp_command': np.tile(high, 2)+.01})


def backend_class(namespace):
    """Execute the actual nested backend class without importing/starting Isaac."""
    source = Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'IsaacBackend')
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(code, str(source), 'exec'), namespace)
    return namespace['IsaacBackend']


def test_backend_apply_accepts_adaptive_controls_and_preserves_legacy_signature():
    namespace = {}
    backend = backend_class(namespace)
    parameters = __import__('inspect').signature(backend.apply).parameters
    assert parameters['palm_action'].default is None
    assert parameters['grip_action'].default is None


def fake_backend(mode='adaptive_grasp'):
    """Real backend logic with tensor-backed stand-ins for the Isaac boundary."""
    low, high = active_limits(); active = (low+high)/2
    q = torch.tensor(expand(active)[None], dtype=torch.float32)

    class Hand:
        joint_names = list(JOINTS)
        body_names = ['finger14_link', 'finger23_link']
        def __init__(self):
            root = torch.tensor([[.1, .2, .3, 1., 0., 0., 0.]])
            bodies = root[:, None, :].repeat(1, 2, 1); bodies[:, :, 2] = .305
            self.data = SimpleNamespace(default_joint_pos=q.clone(), joint_pos=q.clone(),
                joint_vel=torch.zeros_like(q), root_pose_w=root, root_pos_w=root[:, :3],
                body_link_pose_w=bodies, applied_torque=torch.zeros_like(q),
                joint_pos_limits=torch.tensor([[[-10., 10.]]*len(JOINTS)]),
                joint_stiffness=torch.full_like(q, 10.))
            self.actuators = {'active': SimpleNamespace(stiffness=torch.full((1, 6), 10.))}
            self.gain_writes = []; self.pose_writes = []; self.targets = None
        def reset(self): pass
        def write_joint_state_to_sim(self, position, velocity):
            self.data.joint_pos = position.clone(); self.data.joint_vel = velocity.clone()
        def set_joint_position_target(self, position, joint_ids): self.targets = position.clone()
        def write_root_pose_to_sim(self, pose):
            self.pose_writes.append(pose.clone())
            self.data.root_pose_w = pose.clone(); self.data.root_pos_w = pose[:, :3].clone()
        def write_root_velocity_to_sim(self, velocity): pass
        def write_data_to_sim(self): pass
        def update(self, dt): pass
        def write_joint_stiffness_to_sim(self, stiffness, joint_ids):
            self.gain_writes.append((torch.as_tensor(stiffness).clone(), list(joint_ids)))
            self.data.joint_stiffness[:, joint_ids] = stiffness

    class Wire:
        def __init__(self):
            root = torch.tensor([[.43, 0., .308, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
            self.data = SimpleNamespace(default_root_state=root, root_pos_w=root[:, :3].clone(),
                root_quat_w=root[:, 3:7].clone(), root_vel_w=root[:, 7:].clone())
        def reset(self): pass
        def write_root_pose_to_sim(self, pose):
            self.data.root_pos_w = pose[:, :3].clone(); self.data.root_quat_w = pose[:, 3:].clone()
        def write_root_velocity_to_sim(self, velocity): self.data.root_vel_w = velocity.clone()
        def update(self, dt): pass

    hands, wire = [Hand(), Hand()], Wire()
    layout = dict(active_joints={'left': active, 'right': active},
        hand_roots={'left': np.array([.1, .2, .3, 1., 0., 0., 0.]),
                    'right': np.array([.2, -.2, .3, 1., 0., 0., 0.])},
        wire_center=np.array([.43, 0., .308]))
    seed = dict(wire_pose=np.array([.43, 0., .308, 1., 0., 0., 0.]),
        wire_velocity=np.zeros(6), root_xy=np.array([.001, -.001]),
        joint_positions=np.tile(expand(active), (2, 1)), joint_velocities=np.zeros((2, len(JOINTS))),
        initial_command=np.tile(active, 2), grasp_command=np.tile(active+.001, 2))
    calls = []
    def old_xy(current, *unused, **kwargs): calls.append('old_xy'); return current.copy()
    def old_z(current, *unused, **kwargs): calls.append('old_z'); return current.copy()
    namespace = dict(np=np, torch=torch, copy=copy, args=SimpleNamespace(control_mode=mode, device='cpu'),
        dt=1/480, decimation=8, hands=hands, wire=wire, sensors=[], layout=layout,
        scene={'origin': np.zeros(3)}, seed_state=seed,
        sim=SimpleNamespace(physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda: None)),
        ids=[[list(JOINTS).index(n) for n in ACTIVE]]*2, JOINTS=JOINTS, ACTIVE=ACTIVE, MIMIC=MIMIC,
        expand=expand, prepared_approach_reference=lambda value: value,
        track_root_xy=old_xy, adjust_palms=old_z, rotation=lambda quat: np.eye(3),
        moving_goal_error=lambda *args: 0., joint_safety_failure=joint_safety_failure,
        shapes=[(0, 0, np.array([[0., 0., -.001]])), (0, 1, np.array([[0., 0., -.003]])),
                (1, 0, np.zeros((1, 3))), (1, 1, np.zeros((1, 3)))],
        pad_body_ids=[[0, 1], [0, 1]], pad_local_points=np.zeros((4, 3)),
        adaptive_step=helper().adaptive_step, pad_relative_positions=helper().pad_relative_positions,
        select_grasp_command=helper().select_grasp_command)
    backend = backend_class(namespace)()
    return backend, hands, wire, calls, namespace


def test_adaptive_backend_reset_clears_offsets_gain_and_uses_seed_center():
    backend, hands, _, _, namespace = fake_backend()
    backend.adaptive_palm_offset = np.full((2, 3), .002)
    backend.grip_stiffness = np.full(2, 18.)
    backend.reset(.005)
    assert np.array_equal(backend.adaptive_palm_offset, np.zeros((2, 3)))
    assert np.array_equal(backend.grip_stiffness, [10., 10.])
    assert np.array_equal(backend.grasp, namespace['seed_state']['grasp_command'])
    for hand in hands:
        assert hand.gain_writes[-1][0].numpy() == pytest.approx(np.full((1, 6), 10.))
        assert hand.actuators['active'].stiffness.numpy() == pytest.approx(np.full((1, 6), 10.))


def test_adaptive_backend_writes_bounded_roots_and_real_and_estimator_gains():
    backend, hands, _, calls, namespace = fake_backend(); backend.reset(.005)
    backend.hand_clearance = np.array([.002, .01])
    backend.apply(.005, backend.initial_command, phase='close', palm_motion_allowed=True,
        palm_action=np.array([[1., -1., -1.], [-1., 1., 1.]]), grip_action=np.array([1., -1.]))
    assert calls == []
    expected = np.array([[.002/480, -.002/480, 0.], [-.002/480, .002/480, .002/480]])
    assert backend.adaptive_palm_offset == pytest.approx(expected)
    for i, side in enumerate(('left', 'right')):
        base = namespace['layout']['hand_roots'][side][:3]+np.r_[namespace['seed_state']['root_xy'], .005]
        assert hands[i].pose_writes[-1][0, :3].numpy() == pytest.approx(base+expected[i], abs=2e-8)
        actual_gain = 10+10/480 if i == 0 else 10
        assert hands[i].data.joint_stiffness[:, namespace['ids'][i]].numpy() == pytest.approx(np.full((1, 6), actual_gain))
        assert hands[i].actuators['active'].stiffness.numpy() == pytest.approx(np.full((1, 6), actual_gain))


def test_legacy_backend_keeps_original_controller_and_never_writes_gain():
    backend, hands, _, calls, _ = fake_backend('joint_target'); backend.reset(.005)
    backend.apply(.005, backend.initial_command, phase='close', palm_motion_allowed=True)
    assert calls == ['old_xy', 'old_z']
    assert all(hand.gain_writes == [] for hand in hands)


def test_backend_measure_exposes_each_hand_clearance_and_actual_relative_pads():
    backend, hands, wire, _, _ = fake_backend(); backend.reset(.005)
    result = backend.measure()
    assert result['hand_clearance'] == pytest.approx([.002, .005], abs=2e-8)
    expected = np.concatenate([h.data.body_link_pose_w[0, :, :3].numpy() for h in hands])-wire.data.root_pos_w[0].numpy()
    assert result['pad_relative'] == pytest.approx(expected)
    assert np.array_equal(result['adaptive_palm_offset'], np.zeros((2, 3)))
    assert np.array_equal(result['grip_stiffness'], [10., 10.])


def object_config(args):
    """Exercise the real spawn configuration without importing Isaac."""
    source = Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py'
    tree = ast.parse(source.read_text())
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'wire_spawn_configuration'), None)
    assert node is not None, 'wire spawn must support a supplied hollow asset'
    def config(kind):
        return lambda **kwargs: SimpleNamespace(kind=kind, **kwargs)
    sim_utils = SimpleNamespace(**{name: config(name) for name in ('CylinderCfg', 'UsdFileCfg',
        'RigidBodyPropertiesCfg', 'CollisionPropertiesCfg', 'MassPropertiesCfg', 'PreviewSurfaceCfg')})
    namespace = dict(np=np, sim_utils=sim_utils)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['wire_spawn_configuration'](args)


def test_default_wire_retains_solid_shape_mass_and_original_collision_configuration():
    cfg = object_config(SimpleNamespace())
    assert cfg.kind == 'CylinderCfg' and cfg.radius == .008 and cfg.height == .5 and cfg.axis == 'Y'
    assert cfg.mass_props.mass == .12 and cfg.activate_contact_sensors
    assert cfg.rigid_props.disable_gravity is False
    assert cfg.collision_props.contact_offset == .001 and cfg.collision_props.rest_offset == 0.


def test_hollow_wire_uses_supplied_usd_and_mass_without_creating_a_solid_cylinder():
    cfg = object_config(SimpleNamespace(object_usd=Path('assets/dh116_hollow_straw_v1/straw.usda'),
                                        object_mass_kg=.010956))
    assert cfg.kind == 'UsdFileCfg'
    assert cfg.usd_path == 'assets/dh116_hollow_straw_v1/straw.usda'
    assert cfg.mass_props.mass == pytest.approx(.010956)
    assert cfg.activate_contact_sensors and cfg.rigid_props.disable_gravity is False
    assert cfg.collision_props.contact_offset == .001 and cfg.collision_props.rest_offset == 0.
    assert not hasattr(cfg, 'radius')


@pytest.mark.parametrize('mass', [None, 0., -.01, np.inf, np.nan])
def test_external_object_mass_must_be_explicit_positive_and_finite(mass):
    with pytest.raises(ValueError):
        object_config(SimpleNamespace(object_usd='straw.usda', object_mass_kg=mass))


def configured_spawn_joints(mode, grasp):
    """Evaluate the actual articulation init joint expression before Isaac reset."""
    from go2_pvcnn.control.dh116.tabletop_control import prepared_approach_reference
    source = Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py'
    tree = ast.parse(source.read_text())
    spawn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'spawn_scene')
    initial = next(n for n in ast.walk(spawn) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == 'InitialStateCfg'
                   and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ArticulationCfg')
    expression = next(k.value for k in initial.keywords if k.arg == 'joint_pos')
    return eval(compile(ast.Expression(expression), str(source), 'eval'),
        dict(args=SimpleNamespace(control_mode=mode), layout={'active_joints': {'right': grasp}},
             side='right', JOINTS=JOINTS, expand=expand,
             prepared_approach_reference=prepared_approach_reference))


def test_adaptive_spawn_keeps_legal_low_angle_grasp_before_seed_reset():
    low, high = active_limits(); grasp = (low+high)/2
    grasp[ACTIVE.index('finger12')] = .030153
    actual = configured_spawn_joints('adaptive_grasp', grasp)
    assert actual['finger12'] == pytest.approx(.030153)
    assert actual['finger12'] >= .008
    assert [actual[name] for name in ACTIVE] == pytest.approx(grasp)


@pytest.mark.parametrize('mode', ['joint_target', 'residual'])
def test_legacy_spawn_retains_approach_reference(mode):
    from go2_pvcnn.control.dh116.tabletop_control import prepared_approach_reference
    low, high = active_limits(); grasp = (low+high)/2
    actual = configured_spawn_joints(mode, grasp)
    assert [actual[name] for name in JOINTS] == pytest.approx(expand(prepared_approach_reference(grasp)))
