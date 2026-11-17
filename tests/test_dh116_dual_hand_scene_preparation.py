"""Actual USD/SDF CPU checks; live-view fixtures do not certify native physics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from pxr import UsdGeom, UsdPhysics

from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import (
    prepare_scene, author_source_stage, inspect_source_stage, load_sdf_arrays,
    bind_actual_geometry, spawn_parameters, validate_simulation_settings,
)
from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
from go2_pvcnn.control.dh116.tabletop_palm_mount import JOINT_NAMES, ANGULAR_JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def prepared():
    return prepare_scene(ROOT)


def test_preparation_pins_two_original_assets_and_all_ten_locked_sdfs(prepared):
    assert set(prepared['hands']) == {'left', 'right'}
    assert len(prepared['pads']) == 10
    assert prepared['task_unknown_fields']['rod']['gravity_world_m_s2'] is None
    assert prepared['candidate_physics']['gravity_world_m_s2'] == [0., 0., -9.81]
    assert prepared['native_qualified'] is False
    assert prepared['teacher_admitted'] is False
    for side in ('left', 'right'):
        assert len(prepared['hands'][side]['source_colliders']) == 17
        assert not Path(prepared['hands'][side]['usd_path']).is_absolute()
        for link in DISTAL_PAD_LINKS:
            pad = prepared['pads'][side+'.'+link]
            assert pad['side'] == side and pad['body_name'] == link
            assert pad['source_geometry_sha256'] == prepared['hands'][side]['source_colliders'][link]['geometry_sha256']
            assert len(pad['capture_sha256']) == 64


def test_actual_usd_authoring_reuses_original_mount_mimic_and_materials(prepared):
    stage = author_source_stage(ROOT, prepared)
    audit = inspect_source_stage(stage, prepared)
    assert audit['original_colliders'] == 34
    assert audit['compliant_pad_colliders'] == 10
    assert audit['wrist_joints'] == 12
    assert audit['usd_only'] and not audit['sdk_readback_performed']
    for side in ('left', 'right'):
        root = prepared['prefix']+'/'+side+'Hand'
        assert not stage.GetPrimAtPath(root+'/root_joint').IsActive()
        assert stage.GetPrimAtPath(root+'/wrist_fixed').HasAPI(UsdPhysics.ArticulationRootAPI)
        for name in JOINT_NAMES+ANGULAR_JOINT_NAMES:
            assert stage.GetPrimAtPath(root+'/'+name)
    gravity = UsdPhysics.Scene(stage.GetPrimAtPath('/World/physicsScene'))
    assert gravity.GetGravityDirectionAttr().Get() == (0., 0., -1.)
    assert gravity.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81)
    assert not stage.GetPrimAtPath(prepared['prefix']+'/Wire')


def test_source_inspection_rejects_changed_original_geometry(prepared):
    stage = author_source_stage(ROOT, prepared)
    root = prepared['prefix']+'/leftHand'
    stage.GetPrimAtPath(root+'/finger14_link').SetActive(False)
    with pytest.raises(ValueError):
        inspect_source_stage(stage, prepared)


def test_spawn_parameters_record_authoring_offset_and_override_legacy_effort(prepared):
    args, layout, sources = spawn_parameters(ROOT, prepared)
    assert args.palm_transport == 'driven_carriage' and args.free_wrist_grasp is True
    assert args.load_parameters.effort_limit_nm == .15
    assert args.position_iterations == 64 and args.velocity_iterations == 16
    assert sources['left'] != sources['right']
    for side in ('left', 'right'):
        assert np.allclose(layout['hand_roots'][side], prepared['hands'][side]['source_root_pose_w'])
        expected = np.array(layout['hand_roots'][side]); expected[2] += .04
        assert np.allclose(expected, prepared['hands'][side]['builder_initial_root_pose_w'])
    assert 'isaaclab.sim' not in sys.modules


def test_real_sdf_arrays_are_owned_and_bound(prepared):
    grids = load_sdf_arrays(prepared)
    assert set(grids) == set(prepared['pads'])
    for row in grids.values():
        assert row['values'].ndim == 3 and row['values'].dtype == np.float32
        assert row['origin'].shape == (3,)
        assert row['spacing'] == pytest.approx(.001)
    grids['left.finger14_link']['values'][:] = 123.
    assert not np.all(load_sdf_arrays(prepared)['left.finger14_link']['values'] == 123)


def test_changed_sdf_bytes_fail_closed(prepared, tmp_path):
    altered = deepcopy(prepared)
    key = 'right.finger14_link'
    bad = tmp_path/'bad.npz'
    bad.write_bytes(Path(altered['pads'][key]['sdf_path']).read_bytes()+b'changed')
    altered['pads'][key]['sdf_path'] = str(bad)
    with pytest.raises(ValueError, match='SHA256'):
        load_sdf_arrays(altered)


@pytest.mark.parametrize('gravity,hz', [([0.,0.,0.],480), ([0.,0.,-9.81],240)])
def test_simulation_owner_must_supply_actual_matching_gravity_and_dt(prepared, gravity, hz):
    sim = SimpleNamespace(get_physics_dt=lambda:1/hz,
                          physics_sim_view=SimpleNamespace(get_gravity=lambda:gravity))
    with pytest.raises(ValueError):
        validate_simulation_settings(sim, prepared)


def views(prepared, stage):
    hands = []
    for side in ('left', 'right'):
        names = list(reversed(list(prepared['hands'][side]['source_colliders']))) + [
            'wrist_anchor','wrist_slide_x','wrist_slide_y','wrist_slide_z','wrist_rotate_x','wrist_rotate_y']
        paths = [[prepared['prefix']+'/'+side+'Hand/'+name for name in names]]
        poses = np.tile([.5, 0., .55, 0., 0., 0., 1.], (1, len(names), 1)).astype(np.float32)
        view = SimpleNamespace(prim_paths=[prepared['prefix']+'/'+side+'Hand'],
            link_paths=paths, get_link_transforms=lambda p=poses:p)
        hands.append(SimpleNamespace(body_names=names, root_physx_view=view))
    clock = dict(run_uuid='cpu-run',env_id=0,episode_uid='cpu-episode',reset_id=1,rod_uid='rod0',
        physics_step=2,rod_tick=0,policy_step=0,substep=0,timestamp_s=2/480,
        solver_time_s=0.,clock_valid=True)
    return hands,clock


def test_actual_view_paths_and_link_pose_supply_original_clearance_input(prepared):
    stage = author_source_stage(ROOT, prepared)
    hands,clock = views(prepared,stage)
    packet = bind_actual_geometry(ROOT,prepared,stage,hands,read_clock=lambda:[deepcopy(clock)])
    assert packet['body_poses_wxyz'].shape == (1,2,23,7)
    assert len(packet['shapes']) == 34 and len(packet['pad_bindings']) == 10
    assert packet['clearance_m'].shape == (1,2)
    assert np.all(packet['clearance_m'] > .1)
    assert not packet['all_34_live_cooked_available']
    assert not packet['native_qualified']
    for key,row in packet['pad_bindings'].items():
        side,link = key.split('.')
        hi = ('left','right').index(side)
        assert row['body_index'] == hands[hi].body_names.index(link)
        assert row['prim_path'] == hands[hi].root_physx_view.link_paths[0][row['body_index']]
    before = packet['body_poses_wxyz'].copy()
    hands[0].root_physx_view.get_link_transforms()[:] = -1
    assert np.array_equal(before,packet['body_poses_wxyz'])


@pytest.mark.parametrize('fault', ['path','same_view','nonfinite','invalid_clock','moving_clock'])
def test_live_binding_rejects_wrong_instance_or_stale_clock(prepared,fault):
    stage = author_source_stage(ROOT, prepared)
    hands,clock = views(prepared,stage)
    read = lambda:[deepcopy(clock)]
    if fault == 'path':
        hands[1].root_physx_view.link_paths[0][0] = hands[0].root_physx_view.link_paths[0][0]
    elif fault == 'same_view':
        hands[1].root_physx_view = hands[0].root_physx_view
    elif fault == 'nonfinite':
        hands[0].root_physx_view.get_link_transforms()[0,0,0] = np.nan
    elif fault == 'invalid_clock':
        clock['clock_valid'] = False
    else:
        def read():
            clock['physics_step'] += 1
            clock['timestamp_s'] += 1/480
            return [deepcopy(clock)]
    with pytest.raises(ValueError):
        bind_actual_geometry(ROOT,prepared,stage,hands,read_clock=read)


def test_mesh_body_frame_change_cannot_hide_behind_same_triangle_sha(prepared):
    from pxr import Gf
    stage = author_source_stage(ROOT, prepared)
    prim = stage.GetPrimAtPath(prepared['prefix']+'/leftHand/finger14_link/collisions')
    UsdGeom.Xformable(prim).AddTranslateOp(opSuffix='changed').Set(Gf.Vec3d(.001,0.,0.))
    with pytest.raises(ValueError,match='geometry'):
        inspect_source_stage(stage,prepared)


def test_table_resize_is_rejected_by_actual_usd_bounds(prepared):
    stage = author_source_stage(ROOT, prepared)
    UsdGeom.Cube(stage.GetPrimAtPath(prepared['prefix']+'/Table')).GetSizeAttr().Set(2.)
    with pytest.raises(ValueError,match='table bounds'):
        inspect_source_stage(stage,prepared)


def test_actual_simulation_settings_are_read_not_filled_from_configuration(prepared):
    sim = SimpleNamespace(get_physics_dt=lambda:1/480,
        physics_sim_view=SimpleNamespace(get_gravity=lambda:(0.,0.,-9.81)))
    result = validate_simulation_settings(sim,prepared)
    assert result['gravity_world_m_s2'] == [0.,0.,-9.81]
    assert result['source'].endswith('SimulationView.get_gravity')


def test_changed_prepared_source_fingerprint_fails_before_authoring(prepared):
    altered = deepcopy(prepared)
    altered['hands']['left']['builder_initial_root_pose_w'][2] += .1
    with pytest.raises(ValueError,match='fingerprint'):
        author_source_stage(ROOT,altered)


def test_articulation_root_view_path_is_distinct_from_hand_container(prepared):
    stage = author_source_stage(ROOT,prepared)
    hands,clock = views(prepared,stage)
    for hand in hands:
        hand.root_physx_view.prim_paths[0] += '/wrist_fixed'
    result = bind_actual_geometry(ROOT,prepared,stage,hands,read_clock=lambda:[deepcopy(clock)])
    for side in ('left','right'):
        row = result['articulation_paths'][side]
        assert row['actual_articulation_view_path'] == row['authored_articulation_root_path']
        assert row['actual_articulation_view_path'] == row['hand_container_path']+'/wrist_fixed'
        assert result['pad_bindings'][side+'.finger14_link']['prim_path'] == row['hand_container_path']+'/finger14_link'
