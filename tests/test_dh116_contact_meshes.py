from fixture_paths import ARCHIVE, ASSET, BACKEND, CONTACT, CONTACT_SHA, SURFACE_REFS
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import trimesh
import open3d as o3d
pytest.importorskip('pxr.Usd')
from scripts.build_dh116_contact_meshes import build

REF=SURFACE_REFS


def test_actual_four_hulls_com_and_identity(tmp_path):
    if not REF.exists():pytest.skip('local references absent')
    out=tmp_path/'meshes'; report=build(REF,out)
    assert set(report['meshes'])=={'left_thumb','left_index','right_thumb','right_index'}
    assert not report['physics_contact_qualified']
    for channel,rec in report['meshes'].items():
        path=out/rec['path'];mesh=trimesh.load(path,process=False)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
        backend_mesh=o3d.io.read_triangle_mesh(str(path))
        assert backend_mesh.is_watertight() and backend_mesh.is_orientable()
        assert np.linalg.norm(mesh.center_mass)<1e-6
        assert rec['body_name']==('finger14_link' if channel.endswith('thumb') else 'finger23_link')
        assert abs(rec['reference_surface_gap_m'])<1e-4
        assert rec['convex_hull_displacement_bound_m']<=np.sqrt(3)*.5e-7+1e-15
    with pytest.raises(ValueError):build(REF,out)


def test_rejects_changed_reference_source(tmp_path):
    if not REF.exists():pytest.skip('local references absent')
    r=json.loads(REF.read_text());r['asset_sha256']='0'*64
    changed=tmp_path/'bad.json';changed.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='identity'):build(changed,tmp_path/'out')
