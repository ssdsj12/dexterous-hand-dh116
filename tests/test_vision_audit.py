from types import SimpleNamespace
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_vision_audit import audit_initial_points


def test_initial_sensor_geometry_and_camera_slot_swap_detection():
    n,p=2,8
    center=np.array([[.48,0,.368],[2.48,0,.368]])
    roots=np.zeros((n,2,7));roots[...,:3]=center[:,None];roots[...,3]=1.
    dw=SimpleNamespace(roots=roots,wp=center,wq=np.tile([1.,0,0,0],(n,1)))
    points=np.zeros((n,2,p,3));points[...,0]=.008
    points[...,1]=np.linspace(-.2,.2,p)
    packet=dict(points=points,point_valid=np.ones((n,p),bool))
    assert audit_initial_points(packet,dw)['passed']
    # Recovering env 1 camera points with env 0's origin must be caught.
    dw.roots[1,...,:3]=dw.roots[0,...,:3]
    with pytest.raises(RuntimeError,match='3mm'):
        audit_initial_points(packet,dw)


def test_failed_geometry_audit_exposes_full_bad_environment_report():
    from go2_pvcnn.control.dh116.tabletop_vision_audit import VisionGeometryError
    n,p=3,8
    center=np.array([[.48,0,.368],[2.48,0,.368],[4.48,0,.368]])
    roots=np.zeros((n,2,7));roots[...,:3]=center[:,None];roots[...,3]=1.
    dw=SimpleNamespace(roots=roots,wp=center,wq=np.tile([1.,0,0,0],(n,1)))
    points=np.zeros((n,2,p,3));points[...,0]=.008
    points[...,1]=np.linspace(-.2,.2,p)
    dw.roots[1:,...,2]-=.07
    with pytest.raises(VisionGeometryError) as caught:
        audit_initial_points(dict(points=points,point_valid=np.ones((n,p),bool)),dw)
    assert caught.value.report['failed_env_ids']==[1,2]
    assert len(caught.value.report['environments'])==3
    assert caught.value.report['passed'] is False
