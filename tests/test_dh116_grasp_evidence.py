import numpy as np
import pytest
from go2_pvcnn.control.dh116.grasp_evidence import GraspEvidenceGate


def evidence(**kw):
    x=dict(loads_n=np.full(4,.6),valid_contacts=[True]*4,
        normals=np.array([[1,0,0],[-1,0,0],[1,0,0],[-1,0,0]]),
        anchors_local=np.zeros((4,3)),station_rise_m=[0,0],
        min_surface_clearance_m=0.,safety_ok=True,active_constraints=[])
    x.update(kw);return x


def run(g,start,end,**kw):
    last=None
    for i in range(start,end+1):last=g.update(i*.001,**evidence(**kw))
    return last


def gate():return GraspEvidenceGate(material_ids=(38,34,5,3),approach_qualified=True)


def test_requires_measured_approach_not_just_target():
    with pytest.raises(ValueError):GraspEvidenceGate(material_ids=(38,34,5,3),approach_qualified=False)


def test_all_contacts_hold_then_actual_rise_and_dwell_required():
    g=gate();r=run(g,0,99)
    assert r['stage']=='close' and not r['permit_lift']
    r=run(g,100,1099)
    assert r['stage']=='hold' and not r['permit_lift']
    r=run(g,1100,1110)
    assert r['stage']=='lift' and r['permit_lift'] and not r['grasp_lift_passed']
    r=run(g,1111,2111,station_rise_m=[.031,.032],min_surface_clearance_m=.001)
    assert r['grasp_lift_passed'] and not r['whole_wire_clear']


def test_proximity_or_one_hand_cannot_qualify():
    g=gate();r=run(g,0,500,loads_n=[.6,.6,0,0],valid_contacts=[True,True,False,False])
    assert r['stage']=='close' and not r['permit_lift']


@pytest.mark.parametrize('changes',[
    {'anchors_local':np.array([[.0021,0,0],[0,0,0],[0,0,0],[0,0,0]])},
    {'valid_contacts':[True,True,True,False]},
    {'safety_ok':False},{'active_constraints':['bound_endpoint']},
])
def test_hold_slip_loss_safety_or_binding_abort(changes):
    g=gate();run(g,0,120)
    r=g.update(.121,**evidence(**changes))
    assert r['stage']=='abort' and not r['permit_lift'] and not r['grasp_lift_passed']


def test_one_station_rise_or_short_dwell_not_success():
    g=gate();run(g,0,1110)
    r=run(g,1111,2211,station_rise_m=[.031,.029])
    assert not r['grasp_lift_passed']
    r=run(g,2212,2712,station_rise_m=[.031,.032])
    assert not r['grasp_lift_passed']


def test_large_time_gap_or_nan_not_credited_as_dwell():
    g=gate();g.update(0.,**evidence())
    with pytest.raises(ValueError):g.update(2.,**evidence())
    with pytest.raises(ValueError):g.update(.001,**evidence(loads_n=[np.nan,1,1,1]))


def test_same_section_pinch_still_requires_four_contacts_and_full_hold():
    g=GraspEvidenceGate(material_ids=(36,36,4,4),approach_qualified=True)
    r=run(g,0,200,valid_contacts=[True,False,True,True])
    assert r['stage']=='close'
    r=run(g,201,1300)
    assert r['stage']=='hold' and not r['permit_lift']
    r=run(g,1301,1301)
    assert r['permit_lift'] and not r['grasp_lift_passed']


@pytest.mark.parametrize('ids',[(36,36,36,4),(38,34,34,3),(4,4,4,4)])
def test_material_grasp_regions_must_be_separate_between_hands(ids):
    with pytest.raises(ValueError):GraspEvidenceGate(material_ids=ids,approach_qualified=True)


def test_shared_material_id_cannot_report_different_coordinates_in_same_hand():
    g=GraspEvidenceGate(material_ids=(36,36,4,4),approach_qualified=True)
    anchors=np.zeros((4,3));anchors[1,0]=1e-5
    with pytest.raises(ValueError):g.update(0.,**evidence(anchors_local=anchors))


def test_same_section_material_slip_still_aborts_hold():
    g=GraspEvidenceGate(material_ids=(36,36,4,4),approach_qualified=True)
    run(g,0,120)
    anchors=np.zeros((4,3));anchors[:2,0]=.0021
    r=g.update(.121,**evidence(anchors_local=anchors))
    assert r['stage']=='abort' and not r['permit_lift']


def test_explicit_preload_must_precede_full_contact_and_hold_dwells():
    g=GraspEvidenceGate(material_ids=(36,36,4,4),approach_qualified=True,
        minimum_hold_loads_n=[.8,.7,.8,.85])
    r=run(g,0,2000,loads_n=[.1]*4)
    assert r['stage']=='close' and not r['permit_lift']
    r=run(g,2001,2100,loads_n=[.8,.7,.8,.85])
    assert r['stage']=='close'
    r=run(g,2101,3100,loads_n=[.8,.7,.8,.85])
    assert r['stage']=='hold' and not r['permit_lift']
    r=run(g,3101,3101,loads_n=[.8,.7,.8,.85])
    assert r['stage']=='lift' and r['permit_lift']


@pytest.mark.parametrize('threshold',[[.5]*3,[.5,.5,.5,np.nan],[.004]*4,[3.01]*4])
def test_invalid_preload_cannot_weaken_noise_floor_or_exceed_force_target_cap(threshold):
    with pytest.raises(ValueError):
        GraspEvidenceGate(material_ids=(36,36,4,4),approach_qualified=True,
            minimum_hold_loads_n=threshold)
