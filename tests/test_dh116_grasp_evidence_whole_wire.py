import numpy as np

from go2_pvcnn.control.dh116.grasp_evidence import GraspEvidenceGate


def sample(**changes):
    value=dict(
        loads_n=np.full(6, .6),
        valid_contacts=np.ones(6, dtype=bool),
        normals=np.array([[1.,0,0],[-1.,0,0],[0,0,1.],
                          [-1.,0,0],[1.,0,0],[0,0,1.]]),
        anchors_local=np.zeros((6,3)),
        station_rise_m=np.zeros(2),
        min_surface_clearance_m=.0021,
        safety_ok=True, active_constraints=[],
    )
    value.update(changes)
    return value


def gate():
    return GraspEvidenceGate(material_ids=(38,34,5,3,2,1),
                             approach_qualified=True)


def advance(g, start, end, **changes):
    result=None
    for i in range(start, end + 1):
        result=g.update(i*.001, **sample(**changes))
    return result


def test_finish_requires_whole_wire_clearance_during_lift_dwell():
    g=gate()
    advance(g,0,99)
    advance(g,100,1099)
    assert advance(g,1100,1100)['stage']=='lift'
    result=advance(g,1101,2100,station_rise_m=[.031,.032],min_surface_clearance_m=.001)
    assert result['stage']=='lift'
    assert not result['grasp_lift_passed']
    result=advance(g,2101,3101,station_rise_m=[.031,.032],min_surface_clearance_m=.0021)
    assert result['stage']=='finish'
    assert result['grasp_lift_passed'] and result['whole_wire_clear']


def test_clearance_loss_resets_lift_dwell():
    g=gate()
    advance(g,0,99); advance(g,100,1099); advance(g,1100,1100)
    advance(g,1101,1500,station_rise_m=[.031,.032])
    result=g.update(1.501, **sample(station_rise_m=[.031,.032],min_surface_clearance_m=.0019))
    assert result['stage']=='lift' and not result['grasp_lift_passed']
