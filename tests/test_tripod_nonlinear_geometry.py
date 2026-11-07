"""Independent nonlinear source geometry; these tests never start Isaac."""
import copy
import importlib
import importlib.util
import inspect
import numpy as np
import pytest
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import JOINTS, active_limits, expand, coupling_matrix

NAME = 'go2_pvcnn.control.dh116.tabletop_tripod_nonlinear_geometry'


def api():
    assert importlib.util.find_spec(NAME) is not None, 'independent nonlinear geometry module missing'
    return importlib.import_module(NAME)


def fixture(side='right'):
    q = np.mean(active_limits(), axis=0)
    vertices = np.array([[x, y, z] for x in (-.002, .002)
                        for y in (-.003, .003) for z in (-.003, .003)])
    t = np.eye(4); t[0, 3] = .02
    edges = [dict(name=j, parent='root', child=j+'_body', t0=t.tolist(),
                  t1=np.eye(4).tolist(), axis=[0., 0., 1.]) for j in JOINTS]
    shapes = [dict(body=j+'_body', vertices=vertices.tolist()) for j in JOINTS]
    patches = {}
    for digit, j in zip(('thumb', 'index', 'middle'), ('finger13', 'finger22', 'finger32')):
        patches[digit] = dict(body=j+'_body', center_local_m=[-.002, 0., 0.],
            reference_outward_normal_local=[-1., 0., 0.], tangent_u=[0., 1., 0.],
            tangent_v=[0., 0., 1.], half_widths_m=[.0025, .0025],
            planes_local=[[-1., 0., 0., -.002]], geometry_only=True)
    spec = dict(root='root', side=side, edges=edges, shapes=shapes)
    full = expand(q)
    poses = {'root': np.eye(4)}
    for edge, a in zip(edges, full):
        r = np.eye(4); r[:3, :3] = Rotation.from_rotvec([0, 0, a]).as_matrix()
        poses[edge['child']] = t@r
    return spec, patches, full, poses


def test_zero_increment_preserves_every_measured_link_pose_and_full_follower_error():
    m = api(); spec, patches, full, poses = fixture()
    full[2] += .023
    poses['finger13_body'][:3, 3] += [.0001, -.0002, .0003]
    model = m.SourceHandGeometry(spec, patches)
    snapshot = model.bind(full, poses)
    result = model.predict(snapshot, np.zeros(6))
    np.testing.assert_array_equal(result['full_q'], full)
    for body in poses:
        np.testing.assert_allclose(result['poses'][body], poses[body], atol=1e-15, rtol=0)
    delta = np.array([0., .01, 0., 0., 0., 0.])
    moved = model.predict(snapshot, delta)
    np.testing.assert_allclose(moved['full_q'], full+coupling_matrix()@delta, atol=1e-15)
    assert moved['full_q'][2] - expand(full[[0, 1, 3, 5, 7, 9]]+delta)[2] == pytest.approx(.023)


def test_increment_matches_independent_standard_fk_in_ideal_closed_state():
    m = api(); spec, patches, full, poses = fixture()
    model = m.SourceHandGeometry(spec, patches)
    delta = np.linspace(-.003, .004, 6)
    result = model.predict(model.bind(full, poses), delta)
    for edge, angle in zip(spec['edges'], full+coupling_matrix()@delta):
        motion = np.eye(4)
        motion[:3, :3] = Rotation.from_rotvec(np.array(edge['axis'])*angle).as_matrix()
        expected = np.array(edge['t0'])@motion@np.array(edge['t1'])
        np.testing.assert_allclose(result['poses'][edge['child']], expected, atol=2e-15)


def test_full_hull_and_selected_facet_normals_rotate_at_every_candidate():
    m = api(); spec, patches, full, poses = fixture()
    model = m.SourceHandGeometry(spec, patches); snapshot = model.bind(full, poses)
    a = model.predict(snapshot, np.zeros(6))
    b = model.predict(snapshot, np.array([0., 0., .1, 0., 0., 0.]))
    assert not np.allclose(a['hulls'][3]['vertices'], b['hulls'][3]['vertices'])
    assert not np.allclose(a['pad_facets']['index'][0]['normal_w'], b['pad_facets']['index'][0]['normal_w'])


def test_patch_region_finds_off_center_same_witness_with_its_normal():
    m = api()
    vertices = np.array([[x, y, z] for x in (-.002, .002) for y in (-.004, .004) for z in (-.004, .004)])
    hull = ConvexHull(vertices)
    patch = dict(center_local_m=[-.002, 0., 0.], reference_outward_normal_local=[-1., 0., 0.],
        tangent_u=[0., 1., 0.], tangent_v=[0., 0., 1.], half_widths_m=[.0035, .0035],
        planes_local=[[-1., 0., 0., -.002]], geometry_only=True)
    pose = np.eye(4); pose[:3, 3] = [.012, 0., 0.]
    rod = dict(pose_w=[0., .0025, 0., 1., 0., 0., 0.], radius_m=.01, length_m=.004)
    result = m.audit_pad_region(hull.equations, patch, pose, rod, contact_distance_m=.0001, normal_cosine_min=.8)
    assert result['complete'] and result['distance_pass'] and result['direction_pass']
    witness = result['witness']
    assert witness['distance_m'] == pytest.approx(0., abs=1e-11)
    assert abs(witness['point_local_m'][1]) >= .0005-1e-10
    assert witness['ellipse_squared_radius'] <= 1.+1e-9
    assert witness['source_surface_residual_m'] < 1e-10
    assert witness['facing_cosine'] >= .8


def test_wrong_facing_facet_is_not_replaced_with_reference_direction():
    m = api(); spec, patches, _, _ = fixture()
    patch = patches['index']; patch['reference_outward_normal_local'] = [1., 0., 0.]
    patch['center_local_m'] = [.002, 0., 0.]; patch['planes_local'] = [[1., 0., 0., -.002]]
    hull = ConvexHull(np.array(spec['shapes'][0]['vertices']))
    pose = np.eye(4); pose[0, 3] = .008
    result = m.audit_pad_region(hull.equations, patch, pose,
        dict(pose_w=[0., 0., 0., 1., 0., 0., 0.], radius_m=.01, length_m=.1),
        contact_distance_m=.0001, normal_cosine_min=.8)
    assert result['complete'] and result['distance_pass'] and not result['direction_pass']


def test_missing_explicit_limit_sources_is_rejected():
    m = api()
    with pytest.raises(ValueError, match='source'):
        m.GeometryThresholds(contact_distance_m=.001, normal_cosine_min=.8,
            object_overlap_m=.0003, noncontact_object_clearance_m=0., table_clearance_m=0.,
            self_clearance_m=0., interhand_clearance_m=0., obstacle_clearance_m=0.,
            max_command_step_rad=.005, command_lead_rad=.1, sources={})


def test_insufficient_substep_coverage_is_failclosed_before_screening():
    m = api()
    report = m.validate_substep_coverage(np.zeros((4, 7, 6)), np.zeros(6))
    assert not report['passed'] and 'EIGHT_SUBSTEPS_REQUIRED' in report['reasons']
    report = m.validate_substep_coverage(np.zeros((4, 8, 6)), np.ones(6)*.001)
    assert not report['passed'] and 'ENCODED_FIRST_MISMATCH' in report['reasons']


def test_convex_certificate_never_uses_sampled_point_inside_fallback():
    m = api()
    corners = np.array([[x, y, z] for x in (-.02, .02) for y in (-.001, .001) for z in (-.001, .001)])
    a = m.world_hull(corners, np.eye(4), 'a')
    pose = np.eye(4); pose[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    b = m.world_hull(corners, pose, 'b')
    report = m.convex_clearance_certificate(a, b, 0.)
    assert not report['passed'] and report['status'] == 'UNCERTIFIED'
    assert report['sampled_fallback'] is False


def test_rod_middle_and_original_support_are_checked_on_complete_hulls():
    m = api()
    vertices = np.array([[x, y, z] for x in (-.003, .003) for y in (-.003, .003) for z in (-.003, .003)])
    hull = m.world_hull(vertices, np.eye(4), 'source')
    rod = dict(pose_w=[0., 0., 0., 1., 0., 0., 0.], radius_m=.001, length_m=.5)
    report = m.hull_rod_certificate(hull, rod, 0.)
    assert not report['passed'] and report['status'] == 'UNCERTIFIED'
    obstacle = m.world_hull(vertices, np.eye(4), 'original_support')
    assert not m.convex_clearance_certificate(hull, obstacle, .0003)['passed']


def test_invalid_sdk_frame_or_missing_body_fails_initialization():
    m = api(); spec, patches, full, poses = fixture()
    model = m.SourceHandGeometry(spec, patches)
    del poses['finger22_body']
    with pytest.raises(ValueError, match='body'):
        model.bind(full, poses)


def test_quadratic_region_solver_covers_circle_and_polygon_extrema():
    m = api()
    result = m.quadratic_disk_polygon_extrema(np.eye(2), np.array([-.3, .2]),
        np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]), np.array([-.4, -.4, -.8, -.8]))
    assert result['complete']
    np.testing.assert_allclose(result['minimum_point'], [.3, -.2], atol=1e-10)
    assert result['maximum_value'] >= .8


def test_good_pad_witness_does_not_exempt_other_distal_surface_contact():
    m = api(); spec, patches, _, _ = fixture()
    vertices = np.array(spec['shapes'][0]['vertices']); planes = ConvexHull(vertices).equations
    patches['index']['half_widths_m'] = [.004, .0025]
    pose = np.eye(4); pose[0, 3] = .01199
    rod = dict(pose_w=[0., 0., 0., 1., 0., 0., 0.], radius_m=.01, length_m=.5)
    check = m.nonpad_contact_certificate(vertices, planes, patches['index'], pose, rod,
        normal_cosine_min=.8)
    # The rod intersects the two Y side faces too; a good central witness
    # cannot exempt those non-pad intersections on this same distal hull.
    assert not check['passed'] and check['complete'], check
    assert check['source_facets_checked'] == 6
    pose[2, 3] = .003
    check = m.nonpad_contact_certificate(vertices, planes, patches['index'], pose, rod,
        normal_cosine_min=.8)
    assert not check['passed'] and check['status'] == 'UNCERTIFIED'


def test_far_distance_and_wrong_normal_are_diagnostics_not_motion_collision():
    m = api(); spec, patches, full, poses = fixture('left')
    for p in poses.values(): p[0, 3] -= 2.
    left = m.SourceHandGeometry(spec, patches); ls = left.bind(full, poses)
    spec, patches, full, poses = fixture('right')
    for p in poses.values(): p[0, 3] += 2.
    right = m.SourceHandGeometry(spec, patches); rs = right.bind(full, poses)
    fields = dict(contact_distance_m=.001, normal_cosine_min=.8, object_overlap_m=.0003,
        noncontact_object_clearance_m=0., table_clearance_m=0., self_clearance_m=0.,
        interhand_clearance_m=0., obstacle_clearance_m=0., max_command_step_rad=.005, command_lead_rad=.1)
    threshold = m.GeometryThresholds(**fields, sources={k:'explicit fixture' for k in fields})
    audit = m.TripodNonlinearGeometry(left, right, dict(rod=dict(pose_w=[0., 0., 0., 1., 0., 0., 0.],
        radius_m=.01, length_m=.5), table_top_z_m=-1., obstacles=[], obstacles_complete=True,
        provenance={'fixture':True}), threshold)
    result = audit.audit_state([ls, rs], np.zeros(6))
    # Coincident synthetic sibling hulls intentionally fail self clearance;
    # isolate the distance/facing classification without changing collision data.
    assert all(c['passed'] for c in result['collision']['object'])
    assert not result['contact_geometry_qualified']
    assert not any('PAD_DISTANCE' in r or 'WRONG_NORMAL' in r for r in result['rejection_reasons'])


def test_actual_full_delta_and_measured_next_root_are_independent_of_mimic_model():
    m = api(); spec, patches, full, poses = fixture()
    model = m.SourceHandGeometry(spec, patches); snapshot = model.bind(full, poses)
    delta = np.zeros(11); delta[2] = .04
    root = np.eye(4); root[:3, :3] = Rotation.from_euler('x', .2).as_matrix(); root[2, 3] = .003
    moved = model.predict_full_delta(snapshot, delta, root_pose_next=root)
    np.testing.assert_allclose(moved['full_q'], full+delta)
    np.testing.assert_allclose(moved['poses']['root'], root)
    assert not np.allclose(moved['poses']['finger13_body'], root@poses['finger13_body'])
    np.testing.assert_allclose(moved['poses']['finger22_body'], root@poses['finger22_body'], atol=1e-14)


def safe_audit_fixture():
    m = api(); models=[]; snapshots=[]
    for side, offset in [('left', -5.), ('right', 5.)]:
        spec, patches, full, poses = fixture(side)
        for k, edge in enumerate(spec['edges']):
            edge['t0'][0][3] += k*.04
            poses[edge['child']][0, 3] += k*.04
        for k in range(6):
            name='fixed'+str(k); t=np.eye(4);t[0, 3]=.6+k*.04
            spec['edges'].append(dict(name=name,parent='root',child=name,t0=t.tolist(),t1=np.eye(4).tolist(),axis=None))
            spec['shapes'].append(dict(body=name,vertices=spec['shapes'][0]['vertices']))
            poses[name]=t.copy()
        for p in poses.values(): p[0,3]+=offset
        model=m.SourceHandGeometry(spec,patches); models.append(model);snapshots.append(model.bind(full,poses))
    fields=dict(contact_distance_m=.001,normal_cosine_min=.8,object_overlap_m=.0003,
        noncontact_object_clearance_m=0.,table_clearance_m=0.,self_clearance_m=0.,
        interhand_clearance_m=0.,obstacle_clearance_m=0.,max_command_step_rad=.005,command_lead_rad=.1)
    threshold=m.GeometryThresholds(**fields,sources={k:'explicit fixture' for k in fields})
    audit=m.TripodNonlinearGeometry(*models,dict(rod=dict(pose_w=[0.,0.,0.,1.,0.,0.,0.],radius_m=.01,length_m=.5),
        table_top_z_m=-1.,obstacles=[dict(id='original_support',vertices_w_m=np.array(spec['shapes'][0]['vertices'])+[0,0,-.5])],
        obstacles_complete=True,provenance={'fixture':True}),threshold)
    return audit,snapshots


def test_all_horizon_eight_substeps_swept_full_34_hulls_and_fixed_left_are_audited():
    m=api();audit,snapshots=safe_audit_fixture()
    q=snapshots[1]['full_q'][[0,1,3,5,7,9]];commands=np.stack((q+1e-4,q+2e-4))
    x=m.build_command_trajectory(snapshots[1],commands,response=.5)
    report=audit.audit_trajectory(snapshots,x,x[0,-1],command_trajectory=commands,initial_command=q,
        active_lower=np.zeros(6),active_upper=np.ones(6)*2,full_lower=np.zeros((2,11)),full_upper=np.ones((2,11))*2,
        limit_sources={k:'fixture source' for k in ('active','full','command','lead','response')},
        response=.5,encoded_first_command=commands[0])
    assert report['accepted'] and report['motion_safe'],report['rejection_reasons']
    assert not report['contact_geometry_qualified']
    assert report['coverage']['states']==16 and report['coverage']['swept_intervals']==16
    for state in report['states']:
        assert state['source_hull_counts']==[17,17]
        assert len(state['collision']['interhand'])==289
        assert len(state['collision']['obstacles'])==34
        np.testing.assert_array_equal(state['predictions'][0]['full_q'],snapshots[0]['full_q'])


def test_command_and_encoded_action_cannot_borrow_unrelated_safe_delta_certificate():
    m=api();audit,snapshots=safe_audit_fixture();q=snapshots[1]['full_q'][[0,1,3,5,7,9]]
    report=audit.audit_trajectory(snapshots,np.zeros((1,8,6)),np.zeros(6),
        command_trajectory=q[None]+.001,initial_command=q,active_lower=np.zeros(6),active_upper=np.ones(6)*2,
        full_lower=np.zeros((2,11)),full_upper=np.ones((2,11))*2,
        limit_sources={k:'fixture source' for k in ('active','full','command','lead','response')},
        response=.5,encoded_first_command=q+.001)
    assert not report['accepted']
    assert 'COMMAND_STATE_MISMATCH' in report['rejection_reasons']


def test_actual_active_guard_is_checked_separately_from_wider_full_joint_limits():
    m=api();audit,snapshots=safe_audit_fixture();q=snapshots[1]['full_q'][[0,1,3,5,7,9]]
    x=m.build_command_trajectory(snapshots[1],(q-.002)[None],response=.5)
    report=audit.audit_trajectory(snapshots,x,x[0,-1],
        command_trajectory=(q-.002)[None],initial_command=q-.002,active_lower=np.zeros(6),active_upper=q-.001,
        full_lower=np.zeros((2,11)),full_upper=np.ones((2,11))*2,
        limit_sources={k:'fixture source' for k in ('active','full','command','lead','response')},
        response=.5,encoded_first_command=q-.002)
    assert not report['joint_audit']['actual_active_limits']
    assert not report['accepted']


def test_response_trajectory_starts_from_actual_and_tracks_absolute_target():
    m=api();assert hasattr(m,'build_command_trajectory'),'shared absolute command response helper missing'
    _,snapshots=safe_audit_fixture();q=snapshots[1]['full_q'][[0,1,3,5,7,9]]
    commands=np.stack((q+.02,q+.03))
    x=m.build_command_trajectory(snapshots[1],commands,response=.5)
    np.testing.assert_allclose(x[0,-1],.01,atol=1e-14)
    np.testing.assert_allclose(x[1,-1],.02,atol=1e-14)


def test_lead_guard_uses_each_future_action_actual_start_instead_of_fixed_snapshot():
    m=api();audit,snapshots=safe_audit_fixture();q=snapshots[1]['full_q'][[0,1,3,5,7,9]]
    commands=np.stack((q+.1,q-.1));x=m.build_command_trajectory(snapshots[1],commands,response=.5)
    report=audit.audit_trajectory(snapshots,x,x[0,-1],command_trajectory=commands,initial_command=commands[0],
        active_lower=np.zeros(6),active_upper=np.ones(6)*2,full_lower=np.zeros((2,11)),full_upper=np.ones((2,11))*2,
        limit_sources={k:'fixture source' for k in ('active','full','command','lead','response')},
        response=.5,encoded_first_command=commands[0])
    assert not report['joint_audit']['command_lead']


def test_swept_contact_qualification_explicitly_bounds_normal_and_radial_changes():
    m=api()
    assert 'normal_motion_bound' in inspect.signature(m.nonpad_contact_certificate).parameters
    _,snapshots=safe_audit_fixture();spec,patches,_,_=fixture();vertices=np.array(spec['shapes'][0]['vertices'])
    pose=np.eye(4);pose[0,3]=.03
    report=m.nonpad_contact_certificate(vertices,ConvexHull(vertices).equations,patches['index'],pose,
        dict(pose_w=[0.,0.,0.,1.,0.,0.,0.],radius_m=.01,length_m=.5),normal_cosine_min=.8,
        motion_bound=.001,normal_motion_bound=.01)
    assert report['swept_required_facing_cosine'] > .8
    assert report['normal_motion_bound'] == .01


def test_source_loading_api_requires_independent_usd_manifest():
    assert hasattr(api().SourceHandGeometry,'from_usd'),'audited source USD loader missing'


def test_bearing_overlap_requires_same_snapshot_native_contact_provenance():
    m=api();audit,snapshots=safe_audit_fixture()
    assert 'bearing_mask' in inspect.signature(audit.audit_state).parameters
    with pytest.raises(ValueError,match='bearing provenance'):
        audit.audit_state(snapshots,np.zeros(6),bearing_mask=np.ones((2,5),dtype=bool))


def test_degenerate_normal_cone_selects_positive_ray_for_either_ellipse_axis_sign():
    m=api();n=np.array([.8,.6,0.]);u=np.array([0.,0.,1.]);v=np.array([.6,-.8,0.])
    vertices=np.array([x*n+y*u+z*v for x in (-.002,0.) for y in (-.004,.004) for z in (-.004,.004)])
    planes=np.array([[*n,0.],[*(-n),-.002],[*u,-.004],[*(-u),-.004],[*v,-.004],[*(-v),-.004]])
    for sign in (1,-1):
        patch=dict(center_local_m=[0.,0.,0.],reference_outward_normal_local=n,tangent_u=u,tangent_v=sign*v,
            half_widths_m=[.0035,.0035],planes_local=[[*n,0.]],geometry_only=True)
        result=m.audit_pad_region(planes,patch,np.eye(4),dict(pose_w=[0.,0.,0.,1.,0.,0.,0.],
            radius_m=.001,length_m=.02),contact_distance_m=1e-9,normal_cosine_min=.8)
        assert result['contact_geometry_qualified'],result


def test_explicit_left_tracking_delta_is_audited_with_both_complete_hands():
    m=api();audit,snapshots=safe_audit_fixture()
    assert 'left_delta' in inspect.signature(audit.audit_state).parameters
    left=np.linspace(-1e-4,1e-4,6);right=np.zeros(6)
    result=audit.audit_state(snapshots,right,left_delta=left)
    np.testing.assert_allclose(result['predictions'][0]['full_q'],snapshots[0]['full_q']+coupling_matrix()@left)
    np.testing.assert_array_equal(result['predictions'][1]['full_q'],snapshots[1]['full_q'])
    assert result['source_hull_counts']==[17,17]
    assert len(result['collision']['interhand'])==289


@pytest.mark.parametrize('side,filename', [('left','d134680ab619_DH116_left.usd'),
                                         ('right','3046ad2ed855_DH116_right.usd')])
def test_actual_source_usd_all_hulls_match_independent_fk_and_five_regions(side, filename):
    from go2_pvcnn.control.dh116.tabletop_surface_kinematics import SurfaceHandModel
    from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
    from go2_pvcnn.control.dh116.tabletop_effective_pad import _patch, DISTAL_LINKS, DIGITS
    m=api(); source=Path(__file__).resolve().parents[1]/'assets/fixed_dual_panda_dh116_v1/dependencies'/filename
    surface=SurfaceHandModel.from_usd(source,side)
    atlas={s['body'].split('/')[-1]:_patch(s['vertices'],side,s['body'].split('/')[-1])
        for s in surface.spec['shapes'] if s['body'].split('/')[-1] in DISTAL_LINKS}
    model=m.SourceHandGeometry(surface.spec,atlas); q=np.mean(active_limits(),axis=0)
    audited=m.SourceHandGeometry.from_usd(source,side)
    assert audited.source_provenance_verified
    assert audited.spec['source_manifest']['source_collision_count']==17
    assert audited.spec['source_manifest']['layer_sha256']
    assert audited.spec['source_manifest']['meters_per_unit']==1.
    assert len(audited.spec['source_manifest']['collision_meshes'])==17
    independent=HandPointFK(source,side,[.4,-.2,.5,1.,0.,0.,0.]); poses=independent.poses(q)
    snapshot=model.bind(expand(q),poses);delta=np.linspace(-.001,.002,6)
    predicted=model.predict(snapshot,delta);expected=independent.poses(q+delta)
    assert len(predicted['hulls'])==17
    for body in expected: np.testing.assert_allclose(predicted['poses'][body],expected[body],atol=1e-13)
    for digit in DIGITS:
        patch=model.patches[digit]; pose=predicted['poses'][patch['body']]
        center=pose[:3,:3]@np.array(patch['center_local_m'])+pose[:3,3]
        normal=pose[:3,:3]@np.array(patch['reference_outward_normal_local'])
        tangent=pose[:3,:3]@np.array(patch['tangent_u'])
        # Build a Y-cylinder pose whose axis is this region's own tangent.
        rotation=Rotation.align_vectors([tangent],[np.array([0.,1.,0.])])[0]
        quat=rotation.as_quat();rod=dict(pose_w=[*(center+.015*normal),quat[3],*quat[:3]],radius_m=.015,length_m=.5)
        shape=next(s for s in model.shapes if s['body']==patch['body'])
        report=m.audit_pad_region(shape['planes'],patch,pose,rod,contact_distance_m=.00001,normal_cosine_min=.8)
        assert report['complete'] and report['contact_geometry_qualified'],(side,digit,report)
