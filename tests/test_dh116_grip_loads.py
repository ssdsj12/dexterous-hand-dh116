import numpy as np
import pytest
from go2_pvcnn.control.dh116 import grip_loads


def test_symmetric_pinch_balances_weight_and_moment_inside_friction_cones():
    points=np.array([[-.01,0,0],[.01,0,0]])
    normals=np.array([[1.,0,0],[-1.,0,0]])
    report=grip_loads.solve_support(points,normals,[0,0,0],weight_n=1.,mu=.5,max_load_n=2.)
    assert report['feasible']
    assert np.allclose(report['normal_loads_n'],[1.,1.],atol=1e-7)
    assert report['force_error_n']<1e-7 and report['moment_error_nm']<1e-7
    assert report['max_cone_violation_n']<1e-7
    assert report['grasp_qualified'] is False


def test_insufficient_load_limit_does_not_claim_support():
    report=grip_loads.solve_support([[-.01,0,0],[.01,0,0]],[[1,0,0],[-1,0,0]],
        [0,0,0],weight_n=1.,mu=.5,max_load_n=.6)
    assert report['feasible'] is False
    assert report['forces_w_n'] is None


def test_parallel_downward_normals_cannot_lift_even_with_large_normal_load():
    report=grip_loads.solve_support([[-.01,0,0],[.01,0,0]],[[0,0,-1],[0,0,-1]],
        [0,0,0],weight_n=1.,mu=.5,max_load_n=20.)
    assert report['feasible'] is False


@pytest.mark.parametrize('kwargs',[{'mu':-1},{'weight_n':np.nan},{'max_load_n':0}])
def test_invalid_support_request_rejected(kwargs):
    options=dict(weight_n=1.,mu=.5,max_load_n=2.);options.update(kwargs)
    with pytest.raises(ValueError):
        grip_loads.solve_support([[-.01,0,0],[.01,0,0]],[[1,0,0],[-1,0,0]],[0,0,0],**options)
