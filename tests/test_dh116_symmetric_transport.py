import numpy as np
import pytest
from go2_pvcnn.control.dh116.symmetric_transport import SynchronizedTransport


def tick(c,t,rises=(0.,0.),direction=1,valid=(True,True)):
    return c.update(t,rises_m=rises,contacts_valid=valid,direction=direction)


def test_equal_hands_ramp_vertically_and_bound_target_lead():
    c=SynchronizedTransport()
    first=tick(c,0.)
    np.testing.assert_array_equal(first['velocity_w'],0.)
    for t in np.arange(.01,.51,.01):
        r=tick(c,float(t))
        np.testing.assert_allclose(r['velocity_w'][0],r['velocity_w'][1])
        assert 0<=r['velocity_w'][0,2]<=.005
        assert 0<=r['target_progress_m']<=.001+1e-12
        np.testing.assert_array_equal(r['velocity_w'][:,:2],0.)


@pytest.mark.parametrize('leader',[0,1])
def test_leader_waits_with_hysteresis_and_resumes(leader):
    c=SynchronizedTransport();tick(c,0.)
    heights=np.zeros(2);heights[leader]=.0012
    r=tick(c,.01,heights)
    assert r['waiting_side']==leader
    assert r['velocity_w'][leader,2]==0.
    assert r['velocity_w'][1-leader,2]>0.
    heights[leader]=.0007
    assert tick(c,.02,heights)['waiting_side']==leader
    heights[leader]=.0004
    assert tick(c,.03,heights)['waiting_side'] is None


def test_direction_change_resets_wait_and_never_commands_reverse_motion():
    c=SynchronizedTransport();tick(c,0.)
    assert tick(c,.01,[.0012,0])['waiting_side']==0
    r=tick(c,.02,[.0012,0],direction=-1)
    assert r['waiting_side']==1
    np.testing.assert_array_equal(r['velocity_w'],0.)
    r=tick(c,.03,[.0012,0],direction=-1)
    assert r['velocity_w'][0,2]<0
    assert r['velocity_w'][1,2]==0


@pytest.mark.parametrize('heights',[[.002,0],[0,-.002],[np.nan,0],[0]])
def test_rejects_unsafe_or_invalid_measured_height(heights):
    c=SynchronizedTransport()
    with pytest.raises(ValueError):tick(c,0.,heights)


@pytest.mark.parametrize('valid',[[True,False],[True],[1,1]])
def test_both_real_contacts_required(valid):
    with pytest.raises(ValueError):tick(SynchronizedTransport(),0.,valid=valid)


@pytest.mark.parametrize('t',[0.,-.01,.02,np.nan])
def test_time_must_advance_contiguously(t):
    c=SynchronizedTransport();tick(c,0.)
    with pytest.raises(ValueError):tick(c,t)


def test_synthetic_unequal_tracking_stays_synchronized():
    c=SynchronizedTransport();h=np.zeros(2);maximum=0.
    for i in range(1001):
        r=tick(c,i*.01,h)
        h+=r['velocity_w'][:,2]*np.array([1.,.7])*.01
        maximum=max(maximum,float(abs(h[0]-h[1])))
    assert min(h)>.01
    assert maximum<.0011


@pytest.mark.parametrize('direction',[0,2,True,float('nan')])
def test_invalid_direction_rejected(direction):
    with pytest.raises(ValueError):tick(SynchronizedTransport(),0.,direction=direction)
