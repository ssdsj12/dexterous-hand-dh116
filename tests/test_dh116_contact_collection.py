import copy

import numpy as np
import pytest

from go2_pvcnn.control.dh116.contact_collection import ContactDecisionCollector


def observation():
    return dict(hand_q=np.full((2,6),.2),hand_qd=np.full((2,6),.01),
        palm_poses_w=np.zeros((2,7)),contact_normals_w=np.tile([0.,0.,1.],(2,3,1)),
        pad_positions_w=np.zeros((2,3,3)),target_points_w=np.ones((2,3,3))*.001,
        pad_jacobians=np.zeros((2,3,3,6)),contact_loads_n=np.ones(6)*.5,
        contact_gaps_m=np.full((2,3),-.0002),contact_valid=np.ones(6,dtype=bool),
        contact_valid_known=np.ones(6,dtype=bool),target_loads_n=np.ones(6),
        rod_position=np.zeros((3,5)),rod_velocity=np.ones((3,5))*.001,
        rod_director=np.repeat(np.eye(3)[:,:,None],4,axis=2),rod_omega=np.zeros((3,4)),
        rod_radius_m=np.full(4,.008),clearances_m=np.array([.0015]),
        clearance_jacobians=np.zeros((1,12)),clearance_body_names=['left_finger13'],
        goal=dict(lift_height_m=.035,contact_mode='pinch'))


def begin(collector,obs,step,phase='close',owner='hand_mpc'):
    return collector.begin(step=step,time_s=step*.0001,token=step*3,phase=phase,owner=owner,
        observation=obs,previous_action=np.full(12,.2),action=np.full(12,.201),
        prediction=dict(trajectory=np.full((16,12),.201),model='linearized_pad_kinematics')
            if owner=='hand_mpc' else None)


def test_collection_binds_an_action_to_the_next_measured_control_interval():
    collector=ContactDecisionCollector(control_dt_s=.01)
    obs=observation()
    assert begin(collector,obs,0) is None
    obs['hand_q'][:]=.202;obs['rod_velocity'][:]=.002
    result=begin(collector,obs,100)
    assert result['schema']=='dh116-contact-decision-v2'
    np.testing.assert_array_equal(result['observation']['hand_q'],np.full((2,6),.2))
    np.testing.assert_array_equal(result['outcome']['state']['hand_q'],obs['hand_q'])
    np.testing.assert_allclose(result['outcome']['state']['rod_velocity'],.002)
    assert result['step']==0 and result['outcome']['step']==100
    assert result['outcome']['elapsed_s']==pytest.approx(.01)
    assert result['outcome']['token']==300 and not result['outcome']['terminal']
    assert result['prediction']['model']=='linearized_pad_kinematics'
    assert np.asarray(result['prediction']['trajectory']).shape==(16,12)
    assert result['arm_state_included'] is False


def test_collection_retains_hold_lift_and_partial_terminal_outcomes_without_success_credit():
    collector=ContactDecisionCollector(control_dt_s=.01);obs=observation()
    begin(collector,obs,0)
    close=begin(collector,obs,20,phase='hold',owner='hand_mpc_hold')
    hold=begin(collector,obs,120,phase='lift',owner='hand_mpc_hold')
    lift=collector.finish(state=obs,step=160,time_s=.016,token=481,passed=False,
        terminal=True,grasp_lift_passed=False)
    assert [close['phase'],hold['phase'],lift['phase']]==['close','hold','lift']
    assert hold['prediction'] is None and lift['prediction'] is None
    assert not lift['outcome']['passed'] and lift['outcome']['terminal']
    assert not lift['outcome']['grasp_lift_passed']
    assert lift['outcome']['elapsed_s']==pytest.approx(.004)
    assert collector.finish(state=obs,step=161,time_s=.0161,token=482,passed=False,
        terminal=True,grasp_lift_passed=False) is None


@pytest.mark.parametrize('change',[
    lambda o:o.pop('hand_qd'),
    lambda o:o.update(hand_q=np.zeros((2,13))),
    lambda o:o.update(pad_jacobians=np.zeros((2,3,3,13))),
    lambda o:o.update(rod_omega=np.zeros((3,5))),
    lambda o:o.update(contact_valid=[0,1,0,1,0,1]),
    lambda o:o['rod_velocity'].__setitem__((0,0),np.nan),
])
def test_collection_rejects_missing_feedback_nonfinite_or_panda_width_inputs(change):
    obs=observation();change(obs)
    with pytest.raises(ValueError):begin(ContactDecisionCollector(control_dt_s=.01),obs,0)


def test_bad_new_decision_does_not_consume_the_pending_valid_interval():
    collector=ContactDecisionCollector(control_dt_s=.01);obs=observation()
    begin(collector,obs,0)
    bad=copy.deepcopy(obs);bad.pop('contact_gaps_m')
    with pytest.raises(ValueError):begin(collector,bad,100)
    result=begin(collector,obs,100)
    assert result['step']==0


def test_mounted_runner_collects_measured_intervals_and_explicit_hold_actions():
    from pathlib import Path
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    for interface in ('ContactDecisionCollector',"'contact_decisions.jsonl'",
                      'contact_collector.begin(', 'contact_collector.finish(',
                      'contact_valid_known', 'observed_active_velocities'):
        assert interface in source


def test_mounted_preload_hold_keeps_the_physical_observation_and_records_command_mask():
    from pathlib import Path
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    for interface in ('PreloadTargetHold','preload_target_hold.update(',
                      'held_coordinates=hand_preload_held','preload_held_hands'):
        assert interface in source


def test_mounted_preload_observes_each_physical_feedback_before_qualification():
    from pathlib import Path
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert 'preload_target_hold.observe(' in source
    observer=source.index('preload_target_hold.observe(')
    assert source.index("previous_loads=np.array([samples[c]['normal_load_n']")<observer
    assert observer<source.index('evidence=grasp_sequence.update(now')
    assert 'preload_window=hand_preload_window_report' in source
    assert 'completed_physics_window_v1' in source


def test_collection_copies_used_preload_window_without_changing_policy_feature_contract():
    from go2_pvcnn.control.dh116.grip_load_reference import PreloadTargetHold
    from go2_pvcnn.control.dh116.complete_contact_policy import complete_features
    hold=PreloadTargetHold(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
    gaps=np.full((2,3),-.0003);known=np.ones((2,3),dtype=bool)
    low=np.full((2,3),.901);low[0,0]=.899
    hold.observe(low,gaps,known)
    hold.update(np.full((2,3),.901),gaps,known)
    obs=observation();obs['contact_loads_n'][:]=.901
    obs['goal']['preload_window']=dict(hold.last_window,decision_step=100,
        first_physics_step=99,last_physics_step=99)
    collector=ContactDecisionCollector(control_dt_s=.01)
    begin(collector,obs,100)
    baseline=copy.deepcopy(collector.pending)
    baseline['observation']['goal'].pop('preload_window')
    a,b=complete_features(collector.pending,[],rod_nodes=5)
    old_a,old_b=complete_features(baseline,[],rod_nodes=5)
    np.testing.assert_array_equal(a,old_a);np.testing.assert_array_equal(b,old_b)
    assert a.shape==(267,) and b.shape==(581,)
    obs['goal']['preload_window']['minimum_loads_n'][0][0]=99.
    row=begin(collector,obs,200)
    saved=row['observation']['goal']['preload_window']
    assert saved['minimum_loads_n'][0][0]==.899
    assert saved['sample_count']==1 and saved['last_physics_step']==99
