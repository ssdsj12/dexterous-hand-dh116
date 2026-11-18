"""Synthetic CPU arithmetic; these fixtures are not DH116/native evidence."""
import importlib
import importlib.util
import hashlib
import copy
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation


MODULE = 'go2_pvcnn.control.dh116.tabletop_centerline_grasp'


def api():
    assert importlib.util.find_spec(MODULE) is not None, 'centerline screening API is missing'
    return importlib.import_module(MODULE)


def test_arc_interpolation_endpoints_unit_tangent_and_bent_proxy():
    g = api()
    frames = g.sample_centerline_frames([[0, 0, 0], [0, 1, 0], [1, 1, 0]], [0, .5, 1, 2])
    np.testing.assert_allclose([f['point_w_m'] for f in frames], [[0, 0, 0], [0, .5, 0], [0, 1, 0], [1, 1, 0]])
    np.testing.assert_allclose(np.linalg.norm([f['tangent_w'] for f in frames], axis=1), 1)
    assert all(f['bent_centerline_proxy'] for f in frames)
    assert frames[2]['at_vertex'] and frames[-1]['at_endpoint']


@pytest.mark.parametrize('nodes,arcs', [([[0, 0, 0], [0, 0, 0]], [0]),
    ([[0, 0, 0], [np.nan, 0, 0]], [0]), ([[0, 0, 0], [0, 1, 0]], [-.1]),
    ([[0, 0, 0], [0, 1, 0]], [1.1])])
def test_reject_invalid_polyline_or_arc(nodes, arcs):
    with pytest.raises(ValueError):
        api().sample_centerline_frames(nodes, arcs)


def fixture():
    from go2_pvcnn.control.dh116.tabletop_surface_kinematics import SurfaceHandModel
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS, active_limits, expand
    from go2_pvcnn.control.dh116.tabletop_effective_pad import DISTAL_LINKS
    q = np.mean(active_limits(), axis=0)
    full = expand(q)
    models, atlas = [], {'sides': {}}
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    children = {j: j + '_joint_body' for j in JOINTS}
    for j, body in zip(('finger13', 'finger22', 'finger32', 'finger42', 'finger52'), DISTAL_LINKS):
        children[j] = body
    for side in ('left', 'right'):
        edges = [dict(name=j, parent='root', child=children[j], t0=np.eye(4).tolist(),
                      t1=np.eye(4).tolist(), axis=[0, 0, 1]) for j in JOINTS]
        shapes, patches = [], {}
        for k, (j, body) in enumerate(zip(('finger13', 'finger22', 'finger32', 'finger42', 'finger52'), DISTAL_LINKS)):
            center = np.array([-.012 if k == 0 else .012 if k == 1 else .1 + .02*k, 0, .2])
            r = Rotation.from_rotvec([0, 0, full[JOINTS.index(j)]]).as_matrix()
            verts = (center + corners*np.array([.002, .003, .003])) @ r
            shapes.append(dict(body=body, vertices=verts.tolist()))
            outward = np.array([1 if k == 0 else -1, 0, 0])
            point = center + outward*.002
            patches[body] = dict(center_local_m=(point @ r).tolist(),
                reference_outward_normal_local=(outward @ r).tolist(),
                tangent_u=(np.array([0, 1, 0]) @ r).tolist(), tangent_v=[0, 0, 1],
                half_widths_m=[.004, .0035], planes_local=[], surface_tolerance_m=.001,
                geometry_only=True, hull_vertices_sha256=hashlib.sha256(
                    np.unique(verts, axis=0).astype('<f8').tobytes()).hexdigest())
        old = [dict(body=b, points=[patches[b]['center_local_m']]*3,
                    normals=[patches[b]['reference_outward_normal_local']]*3) for b in DISTAL_LINKS[:2]]
        models.append(SurfaceHandModel(dict(root='root', side=side, edges=edges, shapes=shapes, patches=old)))
        atlas['sides'][side] = patches
    scene = dict(schema_version='dh116_centerline_scene_v1', centerline_w_m=[[0, -.3, .2], [0, .3, .2]],
        radius_m=.01, mass_kg=.01, friction=.8, drive_caps_nm=[.2]*12,
        gravity_w_m_s2=[0, 0, -9.81], gravity_torques_nm=[0]*12, center_of_mass_w_m=[0, 0, .2],
        table={'top_z_m': 0}, obstacles=[], obstacles_complete=True,
        source_usd={'left': 'synthetic-left', 'right': 'synthetic-right'},
        source_provenance={'fixture': 'synthetic CPU only'}, native_object_model='rigid_cylinder')
    seed = dict(roots=[[0, .1, 0, 1, 0, 0, 0], [0, -.1, 0, 1, 0, 0, 0]],
                grasp_command=np.r_[q, q].tolist(), wire_pose=[0, 0, .2, 1, 0, 0, 0],
                provenance={'fixture': 'synthetic CPU only'})
    candidate = dict(style='pinch', active_angles=seed['grasp_command'], wrist_pose_w=seed['roots'],
                     arc_positions_m=[.4, .2], azimuth_rad=0.)
    return models, atlas, scene, seed, candidate


def test_opposed_pads_full_hulls_static_fields_and_native_false():
    models, atlas, scene, _, c = fixture()
    r = api().screen_candidate(models, atlas, scene, c)
    assert r['geometry_feasible'], r['rejection_reasons']
    assert r['static_feasible']
    assert r['static_proposal_ready_for_native_verification'] is True
    assert r['native_contact_verified'] is False
    assert r['native_experiment_eligible'] is False, 'synthetic fixtures never admit native experiments'
    assert r['continuous_contact_verified'] is False
    assert len(r['hand_diagnostics']) == 2
    assert r['clearance']['full_source_hulls_checked']
    assert r['hand_diagnostics'][0]['required_digits'] == ['thumb', 'index']
    assert r['hand_diagnostics'][0]['pads'][0]['normal_convention'] == 'CAD outward ON ROD; native force ON HAND is opposite'


def test_wrong_facing_is_rejected_without_replacing_pad():
    models, atlas, scene, _, c = fixture()
    p = atlas['sides']['left']['finger14_link']
    p['reference_outward_normal_local'] = (-np.array(p['reference_outward_normal_local'])).tolist()
    r = api().screen_candidate(models, atlas, scene, c)
    assert not r['geometry_feasible']
    assert 'left/thumb:WRONG_NORMAL' in r['rejection_reasons']


def test_tripod_requires_independent_middle_atlas():
    models, atlas, scene, _, c = fixture()
    c['style'] = 'tripod'
    del atlas['sides']['left']['finger33_link']
    r = api().screen_candidate(models, atlas, scene, c)
    assert not r['geometry_feasible']
    assert 'left/middle:MISSING_SOURCE_PAD' in r['rejection_reasons']


def test_full_cable_support_and_table_screening_retains_all_reasons():
    models, atlas, scene, _, c = fixture()
    scene['table']['top_z_m'] = .198
    obstacle = np.array(models[0].spec['shapes'][0]['vertices']) + [0, .1, 0]
    # A very large original-support proxy intersects both hands regardless of seed rotation.
    scene['obstacles'] = [dict(id='original_support', vertices_w_m=(np.array(
        [[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (.15, .25)])).tolist())]
    r = api().screen_candidate(models, atlas, scene, c)
    assert not r['geometry_feasible']
    assert any('TABLE_CLEARANCE' in v for v in r['rejection_reasons'])
    assert any('OBSTACLE_CLEARANCE_UNCERTIFIED:original_support' in v for v in r['rejection_reasons'])
    assert r['clearance']['full_cable_segments_checked'] == 1


def test_static_unknown_gravity_compensation_is_separate_failure():
    models, atlas, scene, _, c = fixture()
    scene['gravity_torques_nm'] = None
    r = api().screen_candidate(models, atlas, scene, c)
    assert r['geometry_feasible']
    assert not r['static_feasible']
    assert 'STATIC_GRAVITY_COMPENSATION_UNKNOWN' in r['rejection_reasons']


def test_bent_polyline_is_not_admitted_to_native_rigid_cylinder():
    models, atlas, scene, _, c = fixture()
    scene['centerline_w_m'] = [[0, -.3, .2], [0, 0, .2], [.01, .3, .2]]
    r = api().screen_candidate(models, atlas, scene, c)
    assert r['bent_centerline_proxy']
    assert not r['native_experiment_eligible']


def test_no_failed_candidate_fallback_and_distinct_left_right_arcs():
    models, atlas, scene, seed, _ = fixture()
    scene['table']['top_z_m'] = 1.
    report = api().generate_candidates(models, atlas, scene, seed, scene['obstacles'],
        dict(arc_positions_m=[.2, .4], azimuths_rad=[0], styles=['pinch', 'tripod'],
             steps=0, wall_seconds=10, wrist_translation_bound_m=.02, wrist_rotation_bound_rad=.2))
    assert report['accepted'] == []
    assert len(report['rejected']) == 4
    assert all(r['arc_positions_m'][0] != r['arc_positions_m'][1] for r in report['rejected'])
    assert report['native_contact_verified'] is False


def test_frozen_input_requires_physics_and_complete_obstacles():
    _, _, scene, seed, _ = fixture()
    del scene['mass_kg']
    with pytest.raises(ValueError, match='mass_kg'):
        api().validate_frozen_inputs(scene, seed)
    _, _, scene, seed, _ = fixture()
    scene['obstacles_complete'] = False
    with pytest.raises(ValueError, match='obstacles_complete'):
        api().validate_frozen_inputs(scene, seed)


def test_cli_refuses_existing_output_before_loading_usd(tmp_path):
    from runpy import run_path
    path = __import__('pathlib').Path(__file__).resolve().parents[1]/'scripts/generate_dh116_centerline_candidates.py'
    assert path.exists(), 'independent geometry CLI is missing'
    args = run_path(str(path))
    output = tmp_path/'report.json'
    output.write_text('{}')
    with pytest.raises(FileExistsError):
        args['main'](['--scene', 'missing.json', '--seed', 'missing.json', '--output', str(output)])


def test_pad_hash_cannot_be_substituted_from_a_different_source_hull():
    models, atlas, scene, _, c = fixture()
    atlas['sides']['left']['finger14_link']['hull_vertices_sha256'] = 'wrong-source'
    r = api().screen_candidate(models, atlas, scene, c)
    assert not r['geometry_feasible']
    assert 'left/thumb:SOURCE_PAD_IDENTITY' in r['rejection_reasons']


def test_screen_cannot_omit_frozen_support_obstacles():
    models, atlas, scene, _, c = fixture()
    scene['obstacles'] = [dict(id='support', vertices_w_m=np.array(
        [[x, y, z] for x in (.5, .6) for y in (.5, .6) for z in (.0, .1)]).tolist())]
    with pytest.raises(ValueError, match='obstacles'):
        api().screen_candidate(models, atlas, scene, c, obstacles=[])


def test_one_optimizer_step_keeps_exact_active_and_wrist_bounds():
    from go2_pvcnn.control.dh116.linear_coupling import expand
    models, atlas, scene, seed, _ = fixture()
    report = api().generate_candidates(models, atlas, scene, seed, scene['obstacles'],
        dict(arc_positions_m=[.2, .4], azimuths_rad=[0], styles=['pinch'], steps=1,
             wall_seconds=30, wrist_translation_bound_m=.001, wrist_rotation_bound_rad=.01))
    rows = report['accepted']+report['rejected']
    assert len(rows) == 2
    for row in rows:
        assert row['screening_complete']
        q = np.array(row['active_angles']).reshape(2, 6)
        expand(q[0]); expand(q[1])
        assert np.max(np.abs(row['wrist_translation_delta_m'])) <= .001
        assert np.max(np.linalg.norm(row['wrist_rotation_vector_w_rad'], axis=1)) <= .0100000001


def test_self_and_interhand_overlap_are_uncertified_not_asserted_collision():
    models, atlas, scene, _, c = fixture()
    c['wrist_pose_w'][1] = c['wrist_pose_w'][0]
    r = api().screen_candidate(models, atlas, scene, c)
    assert not r['geometry_feasible']
    assert any(v.startswith('INTERHAND_CLEARANCE_UNCERTIFIED') for v in r['rejection_reasons'])
    assert 'proven collision' in r['clearance']['semantics']


def test_nonrequired_hull_checks_whole_cable_beyond_target():
    models, atlas, scene, _, c = fixture()
    # Extend the full object into a nonrequired distal hull, away from target arcs.
    scene['centerline_w_m'] = [[0, -.3, .2], [0, .3, .2], [.14, .3, .2], [.14, -.3, .2]]
    r = api().screen_candidate(models, atlas, scene, c)
    assert not r['geometry_feasible']
    assert any('finger33_link:FULL_CABLE_CLEARANCE_UNCERTIFIED' in v for v in r['rejection_reasons'])
    assert r['clearance']['full_cable_segments_checked'] == 3


def test_forged_seventeen_hulls_and_source_metadata_never_admit_native():
    from go2_pvcnn.control.dh116.tabletop_surface_kinematics import SurfaceHandModel
    models, atlas, scene, _, c = fixture()
    forged = []
    for model in models:
        spec = copy.deepcopy(model.spec)
        spec['source'] = 'fake-but-nonempty-source.usd'
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        for k in range(12):
            spec['shapes'].append(dict(body='root', vertices=(corners*.002+[1+.02*k, 0, .2]).tolist()))
        forged.append(SurfaceHandModel(spec))
    atlas['version'] = 'dh116_centerline_source_pad_v1'
    r = api().screen_candidate(forged, atlas, scene, c)
    assert r['geometry_feasible'] and r['static_feasible'], r['rejection_reasons']
    assert r['native_experiment_eligible'] is False, 'metadata is not a native admission gate'
    assert r['static_proposal_ready_for_native_verification'] is True
    assert r['native_candidate_gate_implemented'] is False


@pytest.mark.parametrize('mutation,error', [
    ('wire_translation', 'wire_pose'), ('wire_rotation', 'wire_pose'),
    ('wire_nonunit', 'unit quaternion'), ('root_nonunit', 'unit quaternion'),
    ('com_offset', 'center_of_mass'), ('native_length', 'length'), ('bent_native', 'proxy')])
def test_frozen_native_cylinder_pose_and_centerline_must_match(mutation, error):
    _, _, scene, seed, _ = fixture()
    if mutation == 'wire_translation':
        seed['wire_pose'][:3] = [40, 30, 20]
    elif mutation == 'wire_rotation':
        seed['wire_pose'][3:] = [2**-.5, 2**-.5, 0, 0]
    elif mutation == 'wire_nonunit':
        seed['wire_pose'][3:] = [2, 0, 0, 0]
    elif mutation == 'root_nonunit':
        seed['roots'][0][3:] = [2, 0, 0, 0]
    elif mutation == 'com_offset':
        scene['center_of_mass_w_m'][0] = .01
    elif mutation == 'native_length':
        scene['source_provenance']['object_parameters'] = dict(native_model='rigid_cylinder', length_m=.5)
    elif mutation == 'bent_native':
        scene['centerline_w_m'] = [[0, -.3, .2], [.01, 0, .2], [0, .3, .2]]
    with pytest.raises(ValueError, match=error):
        api().validate_frozen_inputs(scene, seed)


def test_frozen_cylinder_allows_reversed_axis_and_consistent_rotated_geometry():
    _, _, scene, seed, _ = fixture()
    scene['centerline_w_m'].reverse()
    assert api().validate_frozen_inputs(scene, seed)
    # Rotate native local Y axis onto world Z, translating every geometric field.
    seed['wire_pose'] = [1, 2, 3, 2**-.5, 2**-.5, 0, 0]
    scene['centerline_w_m'] = [[1, 2, 2.7], [1, 2, 3.3]]
    scene['center_of_mass_w_m'] = [1, 2, 3]
    scene['source_provenance']['object_parameters'] = dict(native_model='rigid_cylinder', axis='Y', length_m=.6)
    assert api().validate_frozen_inputs(scene, seed)


def test_bent_centerline_requires_independent_proxy_declaration():
    _, _, scene, seed, _ = fixture()
    scene['centerline_w_m'] = [[0, -.3, .2], [.01, 0, .2], [0, .3, .2]]
    scene['native_object_model'] = 'bent_centerline_proxy'
    assert api().validate_frozen_inputs(scene, seed)


def test_nonzero_gravity_compensation_needs_configuration_dependent_model():
    models, atlas, scene, _, c = fixture()
    scene['gravity_torques_nm'] = [.001]*12
    r = api().screen_candidate(models, atlas, scene, c)
    assert r['geometry_feasible']
    assert not r['static_feasible']
    assert r['static_support']['status'] == 'compensation_model_required'
    assert 'STATIC_COMPENSATION_MODEL_REQUIRED' in r['rejection_reasons']
    assert r['static_proposal_ready_for_native_verification'] is False
