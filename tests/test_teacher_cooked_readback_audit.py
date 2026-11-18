"""Offline validation of saved packets; never live shape or grasp evidence."""
import copy
import importlib
import importlib.util
import json

import pytest

NAME='scripts.audit_dh116_teacher_cooked_readback'


def api():
    assert importlib.util.find_spec(NAME) is not None, 'read-only saved-hull auditor missing'
    return importlib.import_module(NAME)


def cube():
    from test_tripod_cooked_geometry import cube as fixture
    return fixture()[0]


def test_exact_cube_is_reported_without_modifying_raw():
    raw=cube();before=copy.deepcopy(raw)
    row=api().analyze_hull(raw)
    assert row['closed_convex'] is True
    assert row['bad_plane_count']==0 and row['polygons_examined']==6
    assert row['max_face_residual_native_units']==0.
    assert raw==before


def test_inconsistent_plane_is_measured_without_refit_or_tolerance_change():
    raw=cube();raw['polygons'][0]['plane'][3]+=.0001
    before=copy.deepcopy(raw)
    row=api().analyze_hull(raw)
    assert row['closed_convex'] is False and row['bad_plane_count']==1
    assert row['bad_planes'][0]['polygon_index']==0
    assert row['bad_planes'][0]['max_face_residual_native_units']==pytest.approx(.0001)
    assert row['numeric_topology_tolerance_native_units']==pytest.approx(64*2**-23*.002)
    assert raw==before


@pytest.mark.parametrize('bad', ['missing','tagged_nonfinite','index_out_of_bounds'])
def test_invalid_payload_does_not_become_valid_from_partial_plane_analysis(bad):
    raw=cube()
    if bad=='missing':del raw['polygons']
    elif bad=='tagged_nonfinite':raw['vertices'][0][0]={'nonfinite':'nan'}
    else:raw['indices'][0]=999
    result=api().analyze_hull(raw)
    assert result['closed_convex'] is False
    assert result['reasons']


def test_saved_validation_flags_cannot_override_recomputed_result_or_unique_identity():
    raw=cube();raw['polygons'][0]['plane'][3]+=.0001
    row=dict(binding=dict(side='right',body_id=15,actual_body_path='/World/envs/env_0/rightHand/finger42_link'),
        complete=True,cooked_hulls=[dict(raw=raw,validation=dict(closed_convex=True))])
    packet=dict(capture=dict(complete=True,colliders=[row,copy.deepcopy(row)]))
    result=api().analyze_capture(packet)
    assert result['collider_rows']==2 and result['unique_body_count']==1
    assert result['duplicate_body_identities']
    assert result['hulls_returned']==2 and result['hulls_passing_existing_validator']==0
    assert result['native_qualification'] is False
    assert result['bodies'][0]['actual_body_path'].endswith('rightHand/finger42_link')


def test_cli_refuses_overwrite_and_keeps_input_sha(tmp_path):
    module=api()
    source=tmp_path/'capture.json';source.write_text(json.dumps(dict(capture=dict(complete=True,colliders=[]))))
    output=tmp_path/'audit.json'
    assert module.main(['--capture',str(source),'--output',str(output)])==0
    saved=json.loads(output.read_text())
    assert len(saved['input_sha256'])==64 and saved['new_native_runs']==0
    with pytest.raises(FileExistsError):
        module.main(['--capture',str(source),'--output',str(output)])


def surface(raw):
    report=api().analyze_hull(raw)
    assert 'surface_diagnostics' in report, 'independent surface diagnostics missing'
    return report,report['surface_diagnostics']


def test_cube_has_single_closed_vertex_rings_and_centered_signed_volume():
    report,diagnostic=surface(cube())
    assert diagnostic['available'] is True
    assert diagnostic['combinatorial_surface_checks_passed'] is True
    assert diagnostic['vertex_links_single_cycle'] is True
    assert diagnostic['connected_components']==1 and diagnostic['euler_characteristic']==2
    assert diagnostic['signed_fan_volume_native_units_cubed']==pytest.approx(8e-9)
    assert diagnostic['nonplanar_polygon_count']==0
    assert diagnostic['self_intersections_tested'] is False
    assert diagnostic['native_qualification'] is False


def test_nonplanar_closed_surface_never_overrides_protected_rejection():
    raw=cube();raw['vertices'][0][2]+=.0001
    before=copy.deepcopy(raw)
    report,diagnostic=surface(raw)
    assert report['closed_convex'] is False
    assert diagnostic['combinatorial_surface_checks_passed'] is True
    assert diagnostic['nonplanar_polygon_count']==1
    assert diagnostic['max_best_fit_face_residual_native_units']>0.
    assert diagnostic['diagnostic_plane_fits_applied_to_input'] is False
    assert raw==before


def test_open_surface_is_not_hidden_by_positive_volume():
    raw=cube();raw['polygons']=raw['polygons'][1:];raw['num_polygons']-=1
    raw['indices']=raw['indices'][4:]
    for polygon in raw['polygons']:polygon['index_base']-=4
    _,diagnostic=surface(raw)
    assert diagnostic['available'] is True
    assert diagnostic['directed_edges_pair_once'] is False
    assert diagnostic['combinatorial_surface_checks_passed'] is False


def test_reversed_face_detects_winding_and_edge_mismatch():
    raw=cube();raw['indices'][:4]=raw['indices'][:4][::-1]
    _,diagnostic=surface(raw)
    assert diagnostic['winding_matches_stored_normals'] is False
    assert diagnostic['directed_edges_pair_once'] is False
    assert diagnostic['combinatorial_surface_checks_passed'] is False


def test_overlapping_index_slots_are_reported_separately():
    raw=cube();raw['polygons'][1]['index_base']=0
    _,diagnostic=surface(raw)
    assert diagnostic['index_ranges_complete_once'] is False
    assert diagnostic['combinatorial_surface_checks_passed'] is False


def test_pinched_vertex_is_detected_even_when_each_edge_has_an_opposite():
    raw=cube();second=cube();mapping={0:6}
    for i,vertex in enumerate(second['vertices']):
        if i:
            mapping[i]=len(raw['vertices'])
            raw['vertices'].append([v+.002 for v in vertex])
    for polygon in second['polygons']:
        ids=second['indices'][polygon['index_base']:polygon['index_base']+polygon['num_vertices']]
        polygon=copy.deepcopy(polygon)
        polygon['plane'][3]-=sum(n*.002 for n in polygon['plane'][:3])
        polygon['index_base']=len(raw['indices'])
        raw['indices'].extend(mapping[i] for i in ids)
        raw['polygons'].append(polygon)
    raw['num_vertices']=len(raw['vertices']);raw['num_polygons']=len(raw['polygons'])
    _,diagnostic=surface(raw)
    assert diagnostic['directed_edges_pair_once'] is True
    assert diagnostic['connected_components']==1
    assert diagnostic['vertex_links_single_cycle'] is False
    assert diagnostic['combinatorial_surface_checks_passed'] is False


def test_volume_diagnostic_is_invariant_to_local_origin_translation():
    raw=cube();shift=[.31,-.27,.45]
    for vertex in raw['vertices']:
        for i in range(3):vertex[i]+=shift[i]
    for polygon in raw['polygons']:
        polygon['plane'][3]-=sum(n*t for n,t in zip(polygon['plane'][:3],shift))
    _,diagnostic=surface(raw)
    assert diagnostic['combinatorial_surface_checks_passed'] is True
    assert diagnostic['signed_fan_volume_native_units_cubed']==pytest.approx(8e-9)


def test_nonempty_cli_report_serializes_diagnostic_numpy_comparisons(tmp_path):
    row=dict(binding=dict(side='left',body_id=0,actual_body_path='/World/envs/env_0/leftHand/base_link'),
        complete=True,cooked_hulls=[dict(raw=cube())])
    source=tmp_path/'capture.json'
    source.write_text(json.dumps(dict(capture=dict(complete=True,colliders=[row]))))
    output=tmp_path/'result.json'
    assert api().main(['--capture',str(source),'--output',str(output)])==0
    result=json.loads(output.read_text())
    assert result['bodies'][0]['hulls'][0]['surface_diagnostics']['positive_volume_at_original_numeric_scale'] is True
    assert result['native_qualification'] is False
