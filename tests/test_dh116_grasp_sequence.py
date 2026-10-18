import numpy as np
import pytest
from go2_pvcnn.control.dh116.grasp_sequence import TabletopGraspSequence


def sequence():return TabletopGraspSequence(material_ids=(36,36,4,4),target_loads_n=[.8]*4)


def observation(**changes):
    d=dict(settled=True,approach_position_errors_m=np.zeros(4),approach_normal_errors_rad=np.zeros(4),
        station_heights_m=np.full(2,.308),loads_n=np.ones(4),valid_contacts=[True]*4,
        normals=np.array([[1.,0,0],[-1.,0,0],[1.,0,0],[-1.,0,0]]),anchors_local=np.zeros((4,3)),
        min_surface_clearance_m=0.,safety_ok=True,active_constraints=[])
    d.update(changes);return d


def advance(s,start,end,**changes):
    for i in range(start,end+1):r=s.update(i*.01,**observation(**changes))
    return r


def test_complete_measured_sequence_and_transport_requires_hold():
    s=sequence();assert advance(s,1,49)['stage']=='settle'
    assert advance(s,50,399)['stage']=='approach'
    np.testing.assert_array_equal(s.transport_velocity(3.99),np.zeros(3))
    assert advance(s,400,409)['stage']=='close'
    assert advance(s,410,509)['stage']=='hold'
    np.testing.assert_array_equal(s.transport_velocity(5.09),np.zeros(3))
    assert advance(s,510,510)['stage']=='lift'
    v=s.transport_velocity(5.11);assert 0<v[2]<.005
    np.testing.assert_array_equal(v[:2],np.zeros(2))
    r=advance(s,511,611,station_heights_m=[.344,.344],min_surface_clearance_m=.003)
    assert r['grasp_lift_passed'] and r['whole_wire_clear']
    np.testing.assert_array_equal(s.transport_velocity(6.11),np.zeros(3))


@pytest.mark.parametrize('bad',[
    {'settled':False}, {'safety_ok':False}, {'active_constraints':['bound_endpoint']},
    {'approach_position_errors_m':[.003,0,0,0]},
    {'approach_normal_errors_rad':[0,.1,0,0]},
])
def test_failed_settle_or_approach_never_commands_closure_or_lift(bad):
    s=sequence();r=advance(s,1,400,**bad)
    assert r['stage']=='abort' and not r['permit_close'] and not r['grasp_lift_passed']
    np.testing.assert_array_equal(s.transport_velocity(4.),np.zeros(3))


def test_low_contact_load_never_starts_hold_or_lift():
    s=sequence();r=advance(s,1,600,loads_n=[.01]*4)
    assert r['stage']=='close'
    np.testing.assert_array_equal(s.transport_velocity(6.),np.zeros(3))


def test_both_stations_must_rise_and_hold_loss_is_terminal():
    s=sequence();advance(s,1,510)
    r=advance(s,511,700,station_heights_m=[.344,.328])
    assert r['stage']=='lift' and not r['grasp_lift_passed']
    r=advance(s,701,701,valid_contacts=[True,True,False,True])
    assert r['stage']=='abort'
    np.testing.assert_array_equal(s.transport_velocity(7.01),np.zeros(3))
    assert advance(s,702,750)['stage']=='abort'


def test_reaching_both_stop_heights_stops_motion_before_finish_dwell():
    s=sequence();advance(s,1,510)
    r=advance(s,511,511,station_heights_m=[.344,.344])
    assert r['stage']=='lift' and not r['grasp_lift_passed']
    np.testing.assert_array_equal(s.transport_velocity(5.12),np.zeros(3))


def test_nonfinite_or_missing_time_interval_is_rejected():
    s=sequence();s.update(.01,**observation())
    with pytest.raises(ValueError):s.update(.03,**observation())
    with pytest.raises(ValueError):s.update(.02,**observation(station_heights_m=[np.nan,.308]))
    with pytest.raises(ValueError):s.transport_velocity(np.nan)


def test_cannot_start_at_end_of_approach_or_command_from_stale_lift_evidence():
    s=sequence()
    with pytest.raises(ValueError):s.update(4.,**observation())
    advance(s,1,510)
    with pytest.raises(ValueError):s.transport_velocity(6.)


def test_stage_summary_never_promotes_unrun_phases():
    s=sequence()
    assert s.stage_summary(final_passed=False)==dict.fromkeys(('settle','approach','close','hold','lift'),'NOT_RUN')
    advance(s,1,400,loads_n=[.01]*4)
    assert s.stage_summary(final_passed=False)==dict(settle='PASS',approach='PASS',close='FAIL',hold='NOT_RUN',lift='NOT_RUN')
    with pytest.raises(ValueError):s.stage_summary(final_passed=True)


def test_stage_summary_keeps_completed_phases_but_rejects_failed_hold():
    s=sequence();advance(s,1,410)
    advance(s,411,411,valid_contacts=[False,True,True,True])
    assert s.stage_summary(final_passed=False)==dict(settle='PASS',approach='PASS',close='PASS',hold='FAIL',lift='NOT_RUN')


def test_stage_summary_requires_final_validation_for_lift_pass():
    s=sequence();advance(s,1,510)
    advance(s,511,611,station_heights_m=[.344,.344],min_surface_clearance_m=.003)
    assert s.stage_summary(final_passed=True)==dict.fromkeys(('settle','approach','close','hold','lift'),'PASS')
    assert s.stage_summary(final_passed=False)['lift']=='FAIL'
    with pytest.raises(ValueError):s.stage_summary(final_passed='yes')
