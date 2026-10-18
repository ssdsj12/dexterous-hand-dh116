from fixture_paths import ARCHIVE, ASSET, BACKEND, CONTACT, CONTACT_SHA, SURFACE_REFS
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('pxr.Usd')
from scripts.calibrate_dh116_surfaces import build

ASSET=ASSET

def test_real_calibration_and_no_overwrite(tmp_path):
    if not ASSET.exists():pytest.skip('mounted asset missing')
    r=build(ASSET,tmp_path/'calibration')
    assert len(r['references'])==12
    assert r['hardware_calibrated'] is False
    for ref in r['references']:
        assert abs(sum(ref['weights'])-1)<1e-9
        assert np.isclose(np.linalg.norm(ref['normal']),1.)
        assert len(ref['geometry_sha256'])==64
        if ref['digit']!='palm': assert abs(ref['ideal_hull_ray_gap_m'])<1e-4
    for frame in r['palm_frames'].values():
        R=np.asarray(frame)[:3,:3]
        np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-8)
        assert np.linalg.det(R)>0
    with pytest.raises(ValueError):build(ASSET,tmp_path/'calibration')
