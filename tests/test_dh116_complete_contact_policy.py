import copy
import numpy as np
import pytest
import torch

from test_dh116_contact_collection import observation


def decision(value=.5):
    o=observation();o['contact_loads_n'][:]=value
    return dict(observation=o,phase='close',previous_action=[.2]*12,action=[.201]*12)


def test_complete_features_retain_feedback_history_and_full_privileged_rod():
    from go2_pvcnn.control.dh116.complete_contact_policy import complete_features
    row=decision();student,teacher=complete_features(row,[],rod_nodes=5)
    assert student.shape==(267,) and teacher.shape==(581,)
    np.testing.assert_array_equal(student[-3:],[0,0,1])
    changed=copy.deepcopy(row);changed['observation']['hand_qd'][:]=.1
    a,b=complete_features(changed,[row],rod_nodes=5)
    assert not np.array_equal(a,student)
    np.testing.assert_array_equal(a[-3:],[0,1,1])
    changed['observation']['rod_omega'][0,0]=2.
    a2,b2=complete_features(changed,[row],rod_nodes=5)
    np.testing.assert_array_equal(a2,a)
    assert not np.array_equal(b2,b)
    changed['observation']['arm_joints']=np.ones(14)*1000
    a3,b3=complete_features(changed,[row],rod_nodes=5)
    np.testing.assert_array_equal(a3,a2);np.testing.assert_array_equal(b3,b2)
    changed['observation'].pop('contact_valid_known')
    with pytest.raises(ValueError):complete_features(changed,[],rod_nodes=5)


def test_complete_pair_fits_and_loads_as_an_explicit_new_checkpoint(tmp_path):
    from go2_pvcnn.control.dh116.complete_contact_policy import fit_complete_pair
    from go2_pvcnn.control.dh116.contact_imitation import load_mounted_policy
    rows=[decision(v) for v in (.2,.4,.6,.8)]
    models,metrics=fit_complete_pair([rows],control_dt=.01,epochs=60)
    assert metrics['teacher_loss_after']<metrics['teacher_loss_before']*.1
    assert metrics['student_loss_after']<metrics['student_loss_before']*.1
    file=tmp_path/'student.pt';models['student'].save(file)
    loaded=load_mounted_policy(file,control_dt=.01,role='student')
    assert loaded.requires_complete_observation and loaded.input_dim==267
    from go2_pvcnn.control.dh116.integrated_hand_mpc import IntegratedHandMPC
    obs=observation()
    kwargs=dict(pad_positions=obs['pad_positions_w'],pad_jacobians=np.ones((2,3,3,6)),
        target_points=obs['target_points_w'],complete_observation=obs)
    action=IntegratedHandMPC(loaded).plan(obs['palm_poses_w'],obs['hand_q'],.5,
        np.full(12,.2),**kwargs)
    assert action.shape==(2,6) and np.isfinite(action).all()
    assert np.max(np.abs(action-.2))<=.003001
    with pytest.raises(ValueError,match='complete'):
        loaded.plan_contact(active_joints=obs['hand_q'],previous_action=np.full(12,.2))


def test_complete_policy_masks_parked_joints_and_never_takes_future_outcomes():
    from go2_pvcnn.control.dh116.complete_contact_policy import CompleteContactPolicy,complete_features
    p=CompleteContactPolicy(role='student',control_dt=.01,rod_nodes=5)
    for param in p.net.parameters():param.data.zero_()
    p.net[-2].bias.data.fill_(1.)
    obs=observation();jac=np.zeros((2,3,3,6));jac[:,0,0,0]=.02
    action,_=p.plan_contact(active_joints=obs['hand_q'],previous_action=np.full(12,.2),
        complete_observation=obs,pad_jacobians=jac)
    np.testing.assert_array_equal(action.numpy()[:,1:],np.full((2,5),.2,dtype=np.float32))
    row=decision();before=complete_features(row,[],rod_nodes=5)
    row['outcome']={'state':{'hand_q':np.full((2,6),12345)}}
    after=complete_features(row,[],rod_nodes=5)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)


def test_training_cli_uses_complete_intervals_by_default(tmp_path):
    import json,subprocess,sys
    from pathlib import Path
    from test_dh116_contact_replay import complete_run
    source=complete_run(tmp_path/'source')
    out=tmp_path/'model'
    command=[sys.executable,str(Path(__file__).parents[1]/'scripts/train_dh116_contact_imitation.py'),
        '--source-run',str(source),'--out',str(out),'--device','cpu','--epochs','4']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['feature_schema']=='dh116-contact-decision-v2'
    assert manifest['sources'][0]['phase_counts']=={'close':1,'hold':1,'lift':1}
    assert manifest['checkpoints']['student']['input_dim']==267
    assert manifest['student_isaac_validated'] is False
    checkpoint=torch.load(out/'student.pt',weights_only=True)
    assert checkpoint['schema']=='dh116-contact-imitation-v2'
