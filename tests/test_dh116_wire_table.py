import numpy as np
import pytest
from go2_pvcnn.control.dh116.wire_table import settled_metrics


def test_free_wire_settled_on_table():
    p=np.array([[.43]*41,np.linspace(-.25,.25,41),[.308]*41])
    result=settled_metrics(p,np.zeros_like(p),[])
    assert result['passed']
    assert result['min_surface_z_m']==pytest.approx(.3)


@pytest.mark.parametrize('fault',['floating','penetrating','moving','outside','bound','nan'])
def test_rejects_false_table_support(fault):
    p=np.array([[.43]*41,np.linspace(-.25,.25,41),[.308]*41]);v=np.zeros_like(p);constraints=[]
    if fault=='floating':p[2]+=.01
    if fault=='penetrating':p[2]-=.01
    if fault=='moving':v[0]=.1
    if fault=='outside':p[0]=2.
    if fault=='bound':constraints=['endpoint']
    if fault=='nan':p[0,0]=np.nan
    assert not settled_metrics(p,v,constraints)['passed']
