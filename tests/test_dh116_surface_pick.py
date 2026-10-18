import numpy as np
import pytest
from go2_pvcnn.control.dh116.surface_pick import ray_pick


def test_pick_first_triangle_and_barycentric():
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                     [0.,0.,-1.],[1.,0.,-1.],[0.,1.,-1.]])
    r=ray_pick(points, np.array([[0,1,2],[3,4,5]]), [.2,.3,2.], [0.,0.,-1.])
    assert r['triangle_index']==0
    np.testing.assert_allclose(r['weights'],[.5,.2,.3])
    np.testing.assert_allclose(r['point'],[.2,.3,0.])
    np.testing.assert_allclose(r['normal'],[0.,0.,1.])


@pytest.mark.parametrize('origin,direction', [([2.,2.,1.],[0.,0.,-1.]),
    ([.2,.3,1.],[0.,0.,0.]),([.2,.3,np.nan],[0.,0.,-1.])])
def test_miss_or_invalid_rejected(origin,direction):
    with pytest.raises(ValueError):
        ray_pick(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
                 np.array([[0,1,2]]),origin,direction)
