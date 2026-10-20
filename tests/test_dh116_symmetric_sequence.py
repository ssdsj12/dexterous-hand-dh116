import numpy as np
from test_dh116_grasp_sequence import advance
from go2_pvcnn.control.dh116.grasp_sequence import TabletopGraspSequence


def sequence():
    return TabletopGraspSequence(material_ids=(36,36,4,4),target_loads_n=[.8]*4,symmetric_lift=True)


def test_sync_only_operates_in_lift_and_pauses_leader():
    s=sequence();advance(s,1,509)
    np.testing.assert_array_equal(s.transport_velocity(5.09),np.zeros((2,3)))
    advance(s,510,510)
    r=advance(s,511,511,station_heights_m=[.3092,.308])
    assert r['lift_synchronization']['waiting_side']==0
    v=s.transport_velocity(5.12)
    assert v[0,2]==0 and v[1,2]>0


def test_height_mismatch_does_not_qualify_lift_then_resets_dwell():
    s=sequence();advance(s,1,510)
    r=advance(s,511,650,station_heights_m=[.340,.3388])
    assert r['stage']=='lift' and not r['grasp_lift_passed']
    r=advance(s,651,750,station_heights_m=[.340,.340])
    assert not r['grasp_lift_passed']
    r=advance(s,751,751,station_heights_m=[.340,.340])
    assert r['grasp_lift_passed']
    np.testing.assert_array_equal(s.transport_velocity(7.51),np.zeros((2,3)))


def test_large_mismatch_aborts_before_more_transport():
    s=sequence();advance(s,1,510)
    r=advance(s,511,511,station_heights_m=[.3101,.308])
    assert r['stage']=='abort' and not r['grasp_lift_passed']
    np.testing.assert_array_equal(s.transport_velocity(5.12),np.zeros((2,3)))
