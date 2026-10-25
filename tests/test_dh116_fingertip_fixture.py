import numpy as np
import pytest


def test_fixture_near_face_has_positive_gap_for_both_directions():
    from go2_pvcnn.control.dh116.fingertip_fixture import fixture_center
    points=np.array([[-.01,-.02,-.03],[.04,.03,.02]])
    for sign in (-1,1):
        center=fixture_center(points,sign,gap=.0005,thickness=.004)
        near_face=sign*center[0]-.002
        assert near_face==pytest.approx(np.max(sign*points[:,0])+.0005)


def test_sustained_capacity_does_not_count_peak_or_short_or_unsafe_hold():
    from go2_pvcnn.control.dh116.fingertip_fixture import summarize_hold
    times=np.arange(0,1.01,.01)
    forces=np.full(times.shape,15.)
    result=summarize_hold(times,forces,np.zeros_like(times))
    assert result['sustained_14n'] is True
    forces[:90]=0
    assert summarize_hold(times,forces,np.zeros_like(times))['sustained_14n'] is False
    assert summarize_hold(times[:40],np.full(40,15.),np.zeros(40))['sustained_14n'] is False
    assert summarize_hold(times,np.full(times.shape,15.),np.full(times.shape,.021))['sustained_14n'] is False
    assert summarize_hold([0,1],[15,15],[0,0])['sustained_14n'] is False
    # Full-rate sampling must expose dropouts that 60Hz snapshots could miss.
    t=np.arange(481)/480;f=np.full(481,15.);f[1::8]=0.
    assert summarize_hold(t,f,np.zeros(481),max_sample_gap_s=1/480+1e-9)['sustained_14n'] is False


def test_nonfinite_measurements_are_rejected_and_empty_hold_is_not_success():
    from go2_pvcnn.control.dh116.fingertip_fixture import fixture_center, summarize_hold
    assert summarize_hold([],[],[])['sustained_14n'] is False
    with pytest.raises(ValueError):fixture_center([[float('nan'),0,0]],1)
    with pytest.raises(ValueError):summarize_hold([0,1],[0,float('nan')],[0,0])
