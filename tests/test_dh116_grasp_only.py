"""Grasp-local initialization and a pinch that does not require a third contact."""
from types import SimpleNamespace
import numpy as np

from go2_pvcnn.control.dh116.grasp_sequence import TabletopGraspSequence, validate_continuous_options
from go2_pvcnn.control.dh116.grasp_evidence import GraspEvidenceGate


def measured():
    return dict(settled=True, approach_position_errors_m=np.zeros(6),
        approach_normal_errors_rad=np.zeros(6), station_heights_m=np.full(2,.308),
        loads_n=np.array([1.,1.,0.,1.,1.,0.]),
        valid_contacts=np.array([True,True,False,True,True,False]),
        normals=np.array([[1.,0,0],[-1.,0,0],[0.,0,1],[1.,0,0],[-1.,0,0],[0.,0,1]]),
        anchors_local=np.zeros((6,3)), min_surface_clearance_m=0.,
        safety_ok=True, active_constraints=())


def test_grasp_only_initializes_at_close_without_recording_an_approach_pass():
    seq=TabletopGraspSequence(material_ids=(36,36,36,4,4,4), target_loads_n=[.8]*6,
        entry_stage='close', require_middle_support=False)
    assert seq.stage=='close'
    row=seq.update(.0001,**measured())
    assert row['stage']=='close' and row['permit_close'] and not row['permit_lift']
    status=seq.stage_summary(final_passed=False)
    assert status['settle']=='INITIALIZED' and status['approach']=='NOT_RUN'


def test_bimanual_pinch_can_hold_and_lift_without_a_middle_finger_contact():
    seq=TabletopGraspSequence(material_ids=(36,36,36,4,4,4), target_loads_n=[.8]*6,
        entry_stage='close', require_middle_support=False)
    data=measured()
    for i in range(112):row=seq.update(i*.01,**data)
    assert row['stage']=='lift'
    data['station_heights_m']=np.full(2,.344)
    data['min_surface_clearance_m']=.003
    for i in range(112,214):row=seq.update(i*.01,**data)
    assert row['grasp_lift_passed'] and row['whole_wire_clear']
    assert seq.stage_summary(final_passed=True)['approach']=='NOT_RUN'


def test_legacy_middle_hook_mode_still_requires_actual_upward_support():
    gate=GraspEvidenceGate(material_ids=(36,36,36,4,4,4), approach_qualified=True)
    data=measured()
    args={k:data[k] for k in ('loads_n','valid_contacts','normals','anchors_local',
        'min_surface_clearance_m','safety_ok','active_constraints')}
    for i in range(30):row=gate.update(i*.01,station_rise_m=np.zeros(2),**args)
    assert row['stage']=='close'


def test_grasp_only_options_allow_short_trials_without_approach_certificate():
    args=SimpleNamespace(grasp_sequence=True,grasp_entry='close',closure_pilot=False,
        active_pose='pose',active_pose_sha256='sha',approach_certificate=None,
        approach_certificate_sha256=None,point_references='refs',contact_manifest='mesh',
        contact_manifest_sha256='sha',disable_contact_reaction=False,coordinated_closure=False,
        hand_mpc=True,gravity_hold=True,table_wire=True,physics_dt=.0001,steps=500,
        closure_loads=[.8]*6,probe_channel=None)
    validate_continuous_options(args)


def test_synchronized_pinch_lift_does_not_require_the_parked_middle_contacts():
    seq=TabletopGraspSequence(material_ids=(36,36,36,4,4,4),target_loads_n=[.05]*6,
        entry_stage='close',require_middle_support=False,state_triggered=True,symmetric_lift=True)
    data=measured();data['joint_speeds_rad_s']=np.zeros((2,18))
    for i in range(150):row=seq.update(i*.01,**data)
    assert row['stage']=='lift',row['reason']
    assert seq.transport_velocity(1.49)[:,2].min()>0.
    # Required pinch contact still cannot be lost during transport.
    data['valid_contacts'][0]=False
    row=seq.update(1.50,**data)
    assert row['stage']=='abort'


def test_mpc_goal_sync_retains_actual_grip_and_one_second_hold_requirements():
    seq=TabletopGraspSequence(material_ids=(36,36,36,4,4,4),target_loads_n=[1.]*6,
        entry_stage='close',require_middle_support=False,state_triggered=True)
    data=measured();data['joint_speeds_rad_s']=np.zeros((2,18))
    for i in range(20):row=seq.update(i*.01,**data,controller_target_loads_n=np.ones(6))
    assert row['stage']=='close'  # The original100ms physical gate is still incomplete.
    row=seq.update(.20,**data,controller_target_loads_n=np.ones(6))
    assert row['stage']=='hold' and not row['permit_lift']
    assert row['touch_first']['preload_target_source']=='controller'
    for i in range(21,120):
        row=seq.update(i*.01,**data,controller_target_loads_n=np.ones(6))
        assert row['stage']=='hold'
    row=seq.update(1.20,**data,controller_target_loads_n=np.ones(6))
    assert row['stage']=='lift' and not row['grasp_lift_passed']


def test_mpc_goal_sync_cannot_qualify_insufficient_actual_preload():
    seq=TabletopGraspSequence(material_ids=(36,36,36,4,4,4),target_loads_n=[1.]*6,
        entry_stage='close',require_middle_support=False,state_triggered=True)
    data=measured();data['joint_speeds_rad_s']=np.zeros((2,18));data['loads_n']*=.05
    for i in range(100):row=seq.update(i*.01,**data,controller_target_loads_n=np.ones(6))
    assert row['touch_first']['ready_for_hold']
    assert row['stage']=='close' and not row['permit_lift']
