import numpy as np
import pytest
from go2_pvcnn.control.dh116.closure import closure_increment


def test_low_load_closes_high_load_retracts_with_rate_limit():
    J=np.array([[.03,.02,0],[0,0,.04]])
    delta=closure_increment(J,[.2,1.],.0001)
    velocity=J@(delta/.0001)
    assert velocity[0]>0 and velocity[1]<0
    assert np.abs(delta/.0001).max()<=.02


def test_force_deadband_holds_without_position_jump():
    J=np.array([[.03,.02,0],[0,0,.04]])
    np.testing.assert_array_equal(closure_increment(J,[.58,.62],.0001),np.zeros(3))


def test_dt_scaling_and_invalid_geometry():
    J=np.array([[.03,.02,0],[0,0,.04]])
    np.testing.assert_allclose(closure_increment(J,[0,0],.0001),2*closure_increment(J,[0,0],.00005))
    with pytest.raises(ValueError):closure_increment(np.zeros((2,3)),[0,0],.0001)
    with pytest.raises(ValueError):closure_increment(J,[np.nan,0],.0001)
    with pytest.raises(ValueError):closure_increment(J,[0,0],.01)


def test_explicit_asymmetric_load_targets_and_response_rate():
    J=np.array([[.03,.02,0],[0,0,.04]])
    delta=closure_increment(J,[.7,.7],.0001,target_loads_n=[.5,1.5],
                            force_to_speed=.01,max_normal_speed=.002)
    speed=J@(delta/.0001)
    assert speed[0]<0 and speed[1]>0
    assert np.abs(delta/.0001).max()<=.02+1e-12
    with pytest.raises(ValueError):closure_increment(J,[0,0],.0001,target_loads_n=[-1,1])
