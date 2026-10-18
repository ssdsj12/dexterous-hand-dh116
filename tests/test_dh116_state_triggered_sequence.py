import numpy as np
import pytest
from go2_pvcnn.control.dh116.grasp_sequence import TabletopGraspSequence
from test_dh116_grasp_sequence import observation


def sequence():
    return TabletopGraspSequence(material_ids=(36,36,4,4),target_loads_n=[.8]*4,state_triggered=True)


def advance(seq,start,end,**changes):
    data=observation(joint_speeds_rad_s=np.zeros((2,18)),loads_n=[0.]*4,valid_contacts=[False]*4)
    data.update(changes)
    for i in range(start,end+1):result=seq.update(i*.01,**data)
    return result


def test_state_triggered_close_before_four_seconds():
    seq=sequence()
    assert advance(seq,1,54)['stage']=='approach'
    r=advance(seq,55,55)
    assert r['stage']=='close' and r['touch_first']['phases']==['touch','touch']
    assert r['approach_ready_since']==pytest.approx(.5)


def test_high_speed_or_position_waits_past_four_seconds_then_closes():
    seq=sequence()
    assert advance(seq,1,450,joint_speeds_rad_s=np.full((2,18),.021))['stage']=='approach'
    assert advance(seq,451,455)['stage']=='approach'
    assert advance(seq,456,456)['stage']=='close'


def test_missing_speeds_fail_closed_and_timeout_never_authorizes_close():
    seq=sequence()
    with pytest.raises(ValueError):seq.update(.01,**observation())
    r=advance(seq,1,1050,approach_position_errors_m=[.003]*4)
    assert r['stage']=='abort' and '10 s' in r['reason'] and not r['permit_close']


def test_large_loads_do_not_skip_touch_opposition_and_preload():
    seq=sequence();advance(seq,1,55)
    r=advance(seq,56,80,loads_n=[.8]*4,valid_contacts=[True]*4)
    assert r['stage']=='close' and not r['permit_lift']
    assert r['touch_first']['phases']==['preload','preload']
    r=advance(seq,81,160,loads_n=[.8]*4,valid_contacts=[True]*4)
    assert r['stage']=='hold'
    r=advance(seq,161,270,loads_n=[.8]*4,valid_contacts=[True]*4)
    assert r['stage']=='lift' and r['permit_lift']
    r=advance(seq,271,380,loads_n=[.8]*4,valid_contacts=[True]*4,
              station_heights_m=[.344,.344],min_surface_clearance_m=.003)
    assert r['grasp_lift_passed']


def test_handoff_moves_toward_measured_pose_without_command_jump():
    from go2_pvcnn.control.dh116.touch_first import bounded_handoff
    command=np.ones((2,13));measured=command-.001
    q,ready=bounded_handoff(command,measured,1e-4)
    assert not ready
    np.testing.assert_allclose(command[:,:7]-q[:,:7],.05e-4)
    np.testing.assert_allclose(command[:,7:]-q[:,7:],.02e-4)
    for _ in range(500):q,ready=bounded_handoff(q,measured,1e-4)
    assert ready
    np.testing.assert_array_equal(q,measured)


@pytest.mark.parametrize('bad',['shape','nan','dt'])
def test_handoff_rejects_unvalidated_measurements(bad):
    from go2_pvcnn.control.dh116.touch_first import bounded_handoff
    q=np.zeros((2,13));measured=q.copy();dt=1e-4
    if bad=='shape':measured=np.zeros(13)
    if bad=='nan':measured[0,7]=np.nan
    if bad=='dt':dt=.001
    with pytest.raises(ValueError):bounded_handoff(q,measured,dt)
