"""CPU contracts for external wrist drives; no Isaac app or solver is started."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pxr import Usd, UsdGeom, UsdPhysics

from go2_pvcnn.control.dh116 import tabletop_palm_mount as mount
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
from go2_pvcnn.control.dh116.tabletop_load_parameters import physics_load_readback


ANGULAR = ('wrist_roll', 'wrist_pitch', 'wrist_yaw')


def stage_with_hand():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World/Hand')
    base = UsdGeom.Xform.Define(stage, '/World/Hand/base_link').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    root = UsdPhysics.FixedJoint.Define(stage, '/World/Hand/root_joint')
    root.CreateBody1Rel().SetTargets([base.GetPath()])
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    return stage


def test_rotation_contract_is_simulation_estimate_for_external_support_only():
    assert mount.ANGULAR_JOINT_NAMES == ANGULAR
    contract = mount.ROTATION_CONTRACT
    assert contract['angle_limit_rad'] == math.pi / 2
    assert contract['stiffness_nm_rad'] == 40.
    assert contract['damping_nms_rad'] == 4.
    assert contract['effort_limit_nm'] == 2.
    assert contract['velocity_limit_rad_s'] == 1.
    assert contract['calibration_status'] == 'simulation_estimate'


def test_optional_rotation_authors_serial_translation_then_intrinsic_xyz():
    stage = stage_with_hand()
    mount.author_mount(stage, '/World/Hand', rotation=True)
    previous = '/World/Hand/wrist_anchor'
    for index, (name, axis) in enumerate(zip(mount.JOINT_NAMES + ANGULAR, ('X', 'Y', 'Z') * 2)):
        prim = stage.GetPrimAtPath('/World/Hand/' + name)
        joint = (UsdPhysics.PrismaticJoint if index < 3 else UsdPhysics.RevoluteJoint)(prim)
        assert joint and joint.GetAxisAttr().Get() == axis
        assert str(joint.GetBody0Rel().GetTargets()[0]) == previous
        previous = str(joint.GetBody1Rel().GetTargets()[0])
        assert stage.GetPrimAtPath(previous).HasAPI(UsdPhysics.RigidBodyAPI)
        assert joint.GetLocalPos0Attr().Get() == joint.GetLocalPos1Attr().Get()
        assert joint.GetLocalRot0Attr().Get() == joint.GetLocalRot1Attr().Get()
        if index >= 3:
            assert joint.GetLowerLimitAttr().Get() == -90.
            assert joint.GetUpperLimitAttr().Get() == 90.
            drive = UsdPhysics.DriveAPI(prim, 'angular')
            assert drive.GetTypeAttr().Get() == 'force'
            assert drive.GetStiffnessAttr().Get() == 40.
            assert drive.GetDampingAttr().Get() == 4.
            assert drive.GetMaxForceAttr().Get() == 2.
    assert previous == '/World/Hand/base_link'
    assert not stage.GetPrimAtPath('/World/Hand/root_joint').IsActive()


def test_default_authoring_retains_three_axis_chain():
    stage = stage_with_hand()
    mount.author_mount(stage, '/World/Hand')
    assert all(not stage.GetPrimAtPath('/World/Hand/' + name) for name in ANGULAR)
    assert str(UsdPhysics.PrismaticJoint(stage.GetPrimAtPath('/World/Hand/wrist_z'))
               .GetBody1Rel().GetTargets()[0]) == '/World/Hand/base_link'


def target_hand(rotation=True):
    names = list(reversed(list(JOINTS) + list(mount.JOINT_NAMES) + (list(ANGULAR) if rotation else [])))
    targets = torch.full((3, len(names)), .017)
    root = torch.tensor([[1., 2., 3., 2**-.5, 0., 0., 2**-.5]]).repeat(3, 1)

    def write(values, joint_ids, env_ids=None):
        rows = torch.arange(3) if env_ids is None else env_ids
        targets[rows[:, None], torch.tensor(joint_ids)] = values

    return SimpleNamespace(joint_names=names, body_names=['base_link'],
        data=SimpleNamespace(root_pose_w=root), set_joint_position_target=write), targets


def test_mount_infers_angular_ids_and_submits_explicit_local_xyz_radians():
    hand, targets = target_hand()
    view = mount.MountView(hand)
    assert view.ids == [hand.joint_names.index(n) for n in mount.JOINT_NAMES]
    assert view.angular_ids == [hand.joint_names.index(n) for n in ANGULAR]
    pose = hand.data.root_pose_w.clone()
    pose[:, 0] += .01
    angles = torch.tensor([[.2, -.3, .4], [-.4, .1, 0.], [math.pi/2, 0., -math.pi/2]])
    view.move(pose, angles=angles)
    torch.testing.assert_close(targets[:, view.ids], torch.tensor([[0., -.01, 0.]]).repeat(3, 1), atol=1e-6, rtol=0)
    torch.testing.assert_close(targets[:, view.angular_ids], angles)
    finger_ids = [hand.joint_names.index(n) for n in JOINTS]
    assert targets[:, finger_ids].eq(.017).all()
    view.move(pose)
    assert targets[:, view.angular_ids].eq(0.).all()


@pytest.mark.parametrize('rotation', [False, True])
def test_selected_reset_zeros_all_external_axes_and_preserves_fingers(rotation):
    hand, targets = target_hand(rotation)
    view = mount.MountView(hand)
    view.reset(hand.data.root_pose_w[[2, 0]], torch.tensor([2, 0]))
    external = [hand.joint_names.index(n) for n in mount.JOINT_NAMES + (ANGULAR if rotation else ())]
    assert targets[[2, 0]][:, external].eq(0.).all()
    assert targets[1].eq(.017).all()
    assert targets[:, [hand.joint_names.index(n) for n in JOINTS]].eq(.017).all()


@pytest.mark.parametrize('partial', [ANGULAR[:1], ANGULAR[:2]])
def test_mount_rejects_incomplete_angular_joint_set(partial):
    hand, _ = target_hand(False)
    hand.joint_names += list(partial)
    with pytest.raises(ValueError, match='angular|rotation'):
        mount.MountView(hand)


@pytest.mark.parametrize('angles', [torch.zeros(3), torch.zeros(2, 3),
    torch.full((3, 3), float('nan')), torch.full((3, 3), math.pi)])
def test_mount_rejects_invalid_angular_commands_before_writing(angles):
    hand, targets = target_hand()
    before = targets.clone()
    with pytest.raises(ValueError, match='angle'):
        mount.MountView(hand).move(hand.data.root_pose_w, angles=angles)
    torch.testing.assert_close(targets, before)


def test_legacy_mount_rejects_explicit_rotation_instead_of_silently_dropping_it():
    hand, _ = target_hand(False)
    with pytest.raises(ValueError, match='angular|rotation'):
        mount.MountView(hand).move(hand.data.root_pose_w, angles=torch.zeros(3, 3))


def readback_hand(external=mount.JOINT_NAMES + ANGULAR):
    names = list(reversed(list(JOINTS) + list(external)))
    def row(active, linear, angular, passive=0.):
        return np.tile([linear if n in mount.JOINT_NAMES else angular if n in ANGULAR
                        else active if n in ACTIVE else passive for n in names], (3, 1))
    limits = np.zeros((3, len(names), 2))
    for name in ANGULAR:
        if name in names:
            limits[:, names.index(name)] = [-math.pi/2, math.pi/2]
    values = dict(get_dof_stiffnesses=row(10., 100000., 40.),
        get_dof_dampings=row(.2, 1000., 4.), get_dof_max_forces=row(.2, 100., 2.),
        get_dof_max_velocities=row(.3, .05, 1., 2.97), get_dof_limits=limits,
        get_masses=np.ones((3, 2)), get_inertias=np.ones((3, 2, 9)), get_coms=np.zeros((3, 2, 7)))
    sdk = SimpleNamespace(**{key: lambda key=key: torch.tensor(values[key], dtype=torch.float32)
                             for key in values})
    return SimpleNamespace(joint_names=names, body_names=['base_link', 'finger'], root_physx_view=sdk), values


@pytest.mark.parametrize('passive_limit', [None, 2.97])
def test_six_axis_readback_accepts_complete_set_and_reports_radians(passive_limit):
    hand, _ = readback_hand()
    report = physics_load_readback([hand, hand], passive_velocity_limit_rad_s=passive_limit)
    info = report['hands'][0]
    assert info['wrist_rotation'] == mount.ROTATION_CONTRACT
    assert info['joint_units']['wrist_roll'] == 'angular_rad_Nm'
    np.testing.assert_allclose(np.asarray(info['joint_limits_rad'])[0, hand.joint_names.index('wrist_roll')], [-math.pi/2, math.pi/2], atol=1e-6)
    assert report['dof_accounting'] == dict(finger_active=12, finger_passive=10,
        external_wrist_drives=12, policy_command_size=26, force_command_channels=2)
    assert report['hardware_calibration_complete'] is False


def test_surface_mode_readback_counts_motion_commands_without_force_channels():
    hand, _ = readback_hand()
    report = physics_load_readback([hand, hand], force_command_channels=0)
    assert report['dof_accounting'] == dict(finger_active=12, finger_passive=10,
        external_wrist_drives=12, policy_command_size=24, force_command_channels=0)


@pytest.mark.parametrize('external', [mount.JOINT_NAMES[:1], ANGULAR,
    mount.JOINT_NAMES + ANGULAR[:1], mount.JOINT_NAMES + ANGULAR[:2]])
@pytest.mark.parametrize('passive_limit', [None, 2.97])
def test_readback_rejects_partial_support_even_without_passive_velocity_check(external, passive_limit):
    hand, _ = readback_hand(external)
    with pytest.raises(ValueError, match='wrist|carriage|angular'):
        physics_load_readback([hand], passive_velocity_limit_rad_s=passive_limit)


@pytest.mark.parametrize('method,bad', [('get_dof_stiffnesses', 41.), ('get_dof_dampings', 0.),
    ('get_dof_max_forces', 3.), ('get_dof_max_velocities', 2.),
    ('get_dof_max_velocities', float('nan')), ('get_dof_limits', 90.)])
@pytest.mark.parametrize('passive_limit', [None, 2.97])
def test_readback_checks_angular_drive_velocity_and_limits_in_unsampled_clone(method, bad, passive_limit):
    hand, values = readback_hand()
    values[method][-1, hand.joint_names.index('wrist_pitch')] = bad
    with pytest.raises(ValueError, match='wrist|angular|velocity'):
        physics_load_readback([hand], passive_velocity_limit_rad_s=passive_limit)


@pytest.mark.parametrize('free_wrist', [False, True])
def test_spawn_config_keeps_finger_actuators_and_opts_into_separate_external_rotation(free_wrist):
    # Execute the real hand configuration block, with SDK config containers only.
    # No Isaac imports, articulation initialization, or simulation are needed.
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LEGACY_LOAD_PARAMETERS
    from go2_pvcnn.control.dh116.tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S, passive_velocity_limit
    source = Path(__file__).parents[1] / 'src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py'
    spawn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'spawn_scene')
    loop = next(n for n in spawn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'side')
    first = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cfg' for t in n.targets))
    last = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If) and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == 'author_mount' for c in ast.walk(n)))
    cfg_type = type('Config', (SimpleNamespace,), {'InitialStateCfg': SimpleNamespace})
    stage = stage_with_hand()
    scope = dict(__package__='go2_pvcnn.control.dh116', args=SimpleNamespace(
        control_mode='adaptive_grasp', position_iterations=16, palm_transport='driven_carriage', free_wrist_grasp=free_wrist),
        layout=dict(active_joints={'Hand': np.zeros(6)}), side='Hand', prefix='/World', pose=np.array([0., 0., 0., 1., 0., 0., 0.]),
        sources={'Hand': 'unchanged.usd'}, ArticulationCfg=cfg_type, sim_utils=SimpleNamespace(
            UsdFileCfg=SimpleNamespace, RigidBodyPropertiesCfg=SimpleNamespace, ArticulationRootPropertiesCfg=SimpleNamespace),
        ImplicitActuatorCfg=SimpleNamespace, Articulation=lambda cfg: cfg, hands=[], sim=SimpleNamespace(stage=stage),
        JOINTS=JOINTS, ACTIVE=ACTIVE, expand=lambda q: np.zeros(11),
        LEGACY_LOAD_PARAMETERS=LEGACY_LOAD_PARAMETERS, ACTIVE_VELOCITY_LIMIT_RAD_S=ACTIVE_VELOCITY_LIMIT_RAD_S,
        passive_velocity_limit=passive_velocity_limit)
    # The actual hand path in spawn_scene is prefix + '/' + side + 'Hand'.
    scope['side'] = ''
    scope['sources'][''] = scope['sources']['Hand']
    scope['layout']['active_joints'][''] = np.zeros(6)
    exec(compile(ast.Module(body=copy.deepcopy(loop.body[first:last+1]), type_ignores=[]), str(source), 'exec'), scope)
    actuators = scope['cfg'].actuators
    assert actuators['active'].joint_names_expr == list(ACTIVE)
    assert actuators['passive'].joint_names_expr == [n for n in JOINTS if n not in ACTIVE]
    assert actuators['active'].effort_limit_sim == .2
    angular = [cfg for cfg in actuators.values() if cfg.joint_names_expr == list(ANGULAR)]
    assert len(angular) == int(free_wrist)
    if free_wrist:
        assert (angular[0].stiffness, angular[0].damping, angular[0].effort_limit_sim, angular[0].velocity_limit_sim) == (40., 4., 2., 1.)
    assert bool(stage.GetPrimAtPath('/World/Hand/wrist_roll')) == free_wrist
