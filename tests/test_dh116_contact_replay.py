import copy
import json
import numpy as np
import pytest

from go2_pvcnn.control.dh116.contact_replay import contact_features, load_contact_run


def sample_row():
    return dict(step=100,passed=True,control_ownership=dict(phase='close',hand_owner='hand_mpc'),
        grasp_sequence=dict(stage='close',station_rise_m=[0.,0.]),
        contact=dict(reaction_applied=True,normal_samples={
            c:dict(normal_load_n=.2) for c in ('left_thumb','left_index','right_thumb','right_index')}),
        hand_mpc_training_sample=dict(observed_active_joints=np.full((2,6),.2).tolist(),
            previous_action=np.full(12,.21).tolist(),mpc_action=np.full(12,.212).tolist(),
            pad_positions_w=np.zeros((2,3,3)).tolist(),
            target_points_w=np.full((2,3,3),.001).tolist(),rod_backend='torch_cuda',
            arm_state_included=False,planner_diagnostics=dict(model='linearized_pad_kinematics')))


def write_run(path,rows,success=True):
    path.mkdir()
    (path/'run_manifest.json').write_text(json.dumps(dict(run_config=dict(
        rod_backend='torch_cuda',hand_mpc=True,physics_dt=.0001,hand_mpc_update_steps=100))))
    (path/'summary.json').write_text(json.dumps(dict(passed=success,
        grasp_only=dict(grasp_lift_passed=success))))
    (path/'trajectory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def test_contact_features_use_previous_action_and_relative_geometry_without_arm_state():
    row=sample_row();sample=row['hand_mpc_training_sample']
    student,teacher=contact_features(sample)
    assert student.shape==(42,) and teacher.shape==(78,)
    np.testing.assert_allclose(student[:12],.2)
    np.testing.assert_allclose(student[12:24],.21)
    np.testing.assert_allclose(student[24:],1.)  # metre residual converted to mm
    sample['arm_joints']=[1000.]*14
    a,b=contact_features(sample)
    np.testing.assert_array_equal(a,student);np.testing.assert_array_equal(b,teacher)


def test_replay_reads_actual_expert_actions_and_skips_unvalidated_or_student_rows(tmp_path):
    valid=sample_row();invalid=copy.deepcopy(valid);invalid['passed']=False
    student=copy.deepcopy(valid)
    student['hand_mpc_training_sample']['planner_diagnostics']['model']='learned_contact_imitation'
    run=write_run(tmp_path/'run',[valid,invalid,student])

    transitions,metadata=load_contact_run(run)

    assert len(transitions)==1 and transitions[0].physics_valid
    assert transitions[0].phase=='grasp' and transitions[0].done
    assert len(transitions[0].obs_student)==42
    assert len(transitions[0].obs_teacher_privileged)==78
    np.testing.assert_allclose(transitions[0].previous_action,.21)
    np.testing.assert_allclose(transitions[0].mpc_action,.212)
    assert metadata['grasp_lift_passed'] and metadata['rod_backend']=='torch_cuda'
    assert metadata['control_dt_s']==pytest.approx(.01)


def test_incomplete_motion_is_never_silently_promoted_to_successful_training(tmp_path):
    run=write_run(tmp_path/'run',[sample_row()],success=False)
    with pytest.raises(ValueError,match='successful|grasp'):
        load_contact_run(run)
    data,metadata=load_contact_run(run,require_success=False)
    assert len(data)==1 and data[0].reward==0.
    assert not metadata['grasp_lift_passed']


def complete_run(path,*,success=True,phases=('close','hold','lift')):
    from go2_pvcnn.control.dh116.contact_collection import ContactDecisionCollector
    c=ContactDecisionCollector(control_dt_s=.01)
    obs=dict(hand_q=np.full((2,6),.2),hand_qd=np.zeros((2,6)),
        palm_poses_w=np.zeros((2,7)),contact_normals_w=np.tile([0.,0.,1.],(2,3,1)),
        pad_positions_w=np.zeros((2,3,3)),target_points_w=np.ones((2,3,3))*.001,
        pad_jacobians=np.zeros((2,3,3,6)),contact_loads_n=np.ones(6),
        contact_gaps_m=np.full((2,3),-.0001),contact_valid=np.ones(6,dtype=bool),
        contact_valid_known=np.ones(6,dtype=bool),target_loads_n=np.ones(6),
        rod_position=np.zeros((3,5)),rod_velocity=np.zeros((3,5)),
        rod_director=np.repeat(np.eye(3)[:,:,None],4,axis=2),rod_omega=np.zeros((3,4)),
        rod_radius_m=np.full(4,.008),clearances_m=np.array([.0015]),
        clearance_jacobians=np.zeros((1,12)),clearance_body_names=['left_finger13'],
        goal=dict(lift_height_m=.035,contact_mode='pinch'))
    rows=[]
    for i,phase in enumerate(phases):
        r=c.begin(step=i*100,time_s=i*.01,token=i*300,phase=phase,
            owner='hand_mpc' if phase=='close' else 'hand_mpc_hold',observation=obs,
            previous_action=np.full(12,.2),action=np.full(12,.201),
            prediction=dict(model='linearized_pad_kinematics',trajectory=np.full((16,12),.201))
                if phase=='close' else None)
        if r:rows.append(r)
    rows.append(c.finish(state=obs,step=len(phases)*100,time_s=len(phases)*.01,
        token=len(phases)*300,passed=success,terminal=True,grasp_lift_passed=success))
    write_run(path,[],success=success)
    summary=json.loads((path/'summary.json').read_text())
    summary['grasp_only']['whole_wire_clear']=success
    (path/'summary.json').write_text(json.dumps(summary))
    (path/'contact_decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def test_complete_replay_retains_physics_predictions_and_all_three_actual_phases(tmp_path):
    from go2_pvcnn.control.dh116.contact_replay import load_complete_contact_run
    rows,meta=load_complete_contact_run(complete_run(tmp_path/'complete'))
    assert meta['grasp_lift_passed']
    assert meta['phase_counts']=={'close':1,'hold':1,'lift':1}
    assert meta['action_dim']==12 and not meta['arm_state_included']
    assert np.asarray(rows[0]['observation']['rod_director']).shape==(3,3,4)
    assert np.asarray(rows[0]['prediction']['trajectory']).shape==(16,12)
    assert rows[-1]['outcome']['terminal'] and rows[-1]['outcome']['grasp_lift_passed']


def test_complete_replay_never_promotes_close_only_even_with_a_success_summary(tmp_path):
    from go2_pvcnn.control.dh116.contact_replay import load_complete_contact_run
    path=complete_run(tmp_path/'close_only',phases=('close',))
    with pytest.raises(ValueError,match='hold|lift|complete'):
        load_complete_contact_run(path)


def test_complete_replay_rejects_missing_feedback_broken_alignment_and_learned_labels(tmp_path):
    from go2_pvcnn.control.dh116.contact_replay import load_complete_contact_run
    path=complete_run(tmp_path/'bad')
    original=[json.loads(line) for line in (path/'contact_decisions.jsonl').read_text().splitlines()]
    for change in (lambda r:r[0]['observation'].pop('hand_qd'),
                   lambda r:r[1].update(step=101),
                   lambda r:r[0].update(owner='hand_student')):
        rows=copy.deepcopy(original);change(rows)
        (path/'contact_decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with pytest.raises(ValueError):load_complete_contact_run(path)


def test_complete_failed_physics_is_available_only_as_explicit_diagnostic(tmp_path):
    from go2_pvcnn.control.dh116.contact_replay import load_complete_contact_run
    path=complete_run(tmp_path/'failed',success=False)
    with pytest.raises(ValueError,match='successful|grasp'):
        load_complete_contact_run(path)
    rows,meta=load_complete_contact_run(path,require_success=False)
    assert len(rows)==3 and not meta['grasp_lift_passed']
    assert meta['training_scope']=='partial_action_diagnostic'


@pytest.mark.parametrize('whole_clear',[False,None])
def test_complete_training_requires_whole_wire_clear_not_just_raised_grasp_stations(tmp_path,whole_clear):
    from go2_pvcnn.control.dh116.contact_replay import load_complete_contact_run
    path=complete_run(tmp_path/'station_lift')
    summary=json.loads((path/'summary.json').read_text())
    summary['grasp_only']['whole_wire_clear']=whole_clear
    (path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='whole|successful|grasp'):
        load_complete_contact_run(path)
    _,meta=load_complete_contact_run(path,require_success=False)
    assert not meta['grasp_lift_passed']
