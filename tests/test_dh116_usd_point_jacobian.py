from fixture_paths import ARCHIVE, ASSET, BACKEND, CONTACT, CONTACT_SHA, SURFACE_REFS
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('pxr.Usd')
from pxr import Usd
from go2_pvcnn.control.dh116.usd_kinematics import SideKinematics

ASSET=ASSET

@pytest.fixture(scope='module')
def calibrated_references(tmp_path_factory):
    if not ASSET.exists():pytest.skip('asset missing')
    from scripts.calibrate_dh116_surfaces import build
    return build(ASSET,tmp_path_factory.mktemp('dh116')/'references')['references']

@pytest.mark.parametrize('side',['left','right'])
def test_actual_usd_point_jacobian(side,calibrated_references):
    if not ASSET.exists():pytest.skip('asset missing')
    model=SideKinematics(Usd.Stage.Open(str(ASSET)),side)
    for h in (.1,.3,.7):
        q=np.array([.1,-.6,.1,-2.2,.1,1.8,.7]+[h]*6)
        for ref in calibrated_references:
            if ref['side']!=side:continue
            body=ref['body_name'];local=np.array(ref['point'])
            j=model.point_jacobian(q,body,local)
            assert j.shape==(3,13)
            for eps in (1e-4,1e-5,1e-6):
                fd=np.column_stack([(model.point(q+np.eye(13)[i]*eps,body,local)-
                    model.point(q-np.eye(13)[i]*eps,body,local))/(2*eps) for i in range(13)])
                np.testing.assert_allclose(j,fd,rtol=0.,atol=1e-6)
    with pytest.raises(ValueError):model.point(np.zeros(12),body,local)
    with pytest.raises(ValueError):model.point(q,'missing',local)


def test_measured_point_error_contract():
    from go2_pvcnn.control.dh116.point_probe import compare_point
    # Rotation-free actor with its COM displaced along X. Pure Z angular
    # velocity at COM produces +Y speed at the reference point.
    names=tuple(f'left_panda_joint{i}' for i in range(1,8))+(
        'finger11','finger12','finger13','finger21','finger22','finger31',
        'finger32','finger41','finger42','finger51','finger52')
    raw=np.zeros((6,18));raw[5,0]=1.
    reduced=np.zeros((3,13));reduced[1,0]=.1
    velocity=np.zeros(13);velocity[0]=.01
    r=compare_point(raw,names,'left',[0.,0.,0.,1.,0.,0.,0.],
        [.1,0.,0.],[0.,0.,0.,0.,0.,.01],[.2,0.,0.],
        [.2,0.,0.],reduced,velocity)
    assert r['position_error_m']==0
    assert r['jacobian_error_m_rad']==0
    assert r['velocity_error_m_s']<1e-12
