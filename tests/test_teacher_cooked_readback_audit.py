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
