import numpy as np
import pytest

torch=pytest.importorskip('torch')

from go2_pvcnn.control.dh116.contact_imitation import ContactImitationPolicy,fit_contact_pair
from go2_pvcnn.control.dh116.contact_replay import contact_features
from go2_pvcnn.control.dh116.teacher_student import make_transition


def examples():
    rows=[]
    for value in np.linspace(-1.,1.,9):
        sample=dict(observed_active_joints=np.full((2,6),.2),previous_action=np.full(12,.2),
            pad_positions_w=np.zeros((2,3,3)),target_points_w=np.full((2,3,3),value*.001))
        student,teacher=contact_features(sample)
        action=np.full(12,.2)+.002*np.tanh(value)*np.tile([1,-1,1,0,0,0],2)
        rows.append(make_transition(obs_student=student,obs_teacher_privileged=teacher,
            previous_action=sample['previous_action'],mpc_action=action,action=action,
            reward=1.,done=False,phase='grasp',physics_valid=True))
    return rows


def test_teacher_and_student_learn_expert_residuals_and_reload(tmp_path):
    models,metrics=fit_contact_pair(examples(),control_dt=.01,epochs=120,device='cpu')
    assert metrics['teacher_loss_after'] < metrics['teacher_loss_before']*.2
    assert metrics['student_loss_after'] < metrics['student_loss_before']*.2
    assert models['teacher'].input_dim==78 and models['student'].input_dim==42
    file=tmp_path/'student.pt';models['student'].save(file)
    loaded=ContactImitationPolicy.load(file,expected_role='student')
    kwargs=dict(active_joints=np.full((2,6),.2),previous_action=np.full(12,.2),
        pad_positions=np.zeros((2,3,3)),target_points=np.full((2,3,3),.0008))
    action,diag=loaded.plan_contact(**kwargs)
    expected,_=models['student'].plan_contact(**kwargs)
    torch.testing.assert_close(action,expected)
    assert action.shape==(2,6)
    assert torch.max(torch.abs(action.reshape(12)-.2)) <= .003001
    assert diag['model']=='learned_contact_imitation'
    assert action[0,0] > .2 and action[0,1] < .2
    from go2_pvcnn.control.dh116.integrated_hand_mpc import IntegratedHandMPC
    routed=IntegratedHandMPC(loaded).plan(np.zeros((2,7)),kwargs['active_joints'],.5,
        kwargs['previous_action'],pad_positions=kwargs['pad_positions'],
        pad_jacobians=np.ones((2,3,3,6)),target_points=kwargs['target_points'])
    np.testing.assert_allclose(routed,action.numpy(),atol=1e-8)


def test_student_loader_rejects_a_teacher_checkpoint(tmp_path):
    policy=ContactImitationPolicy(role='teacher',control_dt=.01)
    file=tmp_path/'teacher.pt';policy.save(file)
    with pytest.raises(ValueError,match='student'):
        ContactImitationPolicy.load(file,expected_role='student')


def test_mounted_contact_policy_loader_checks_the_control_interval(tmp_path):
    from go2_pvcnn.control.dh116.contact_imitation import load_mounted_policy
    policy=ContactImitationPolicy(role='student',control_dt=.01)
    policy.training_metadata={'training_scope':'partial_action_diagnostic',
                              'validated_source_grasp_lift':False}
    file=tmp_path/'student.pt';policy.save(file)
    loaded=load_mounted_policy(file,control_dt=.01,device='cpu',role='student')
    assert loaded.training_metadata==policy.training_metadata
    with pytest.raises(ValueError,match='interval'):
        load_mounted_policy(file,control_dt=.1,device='cpu',role='student')


def test_contact_imitation_preserves_mpc_parked_joint_contract():
    policy=ContactImitationPolicy(role='student',control_dt=.01)
    for parameter in policy.net.parameters():parameter.data.zero_()
    policy.net[-2].bias.data.fill_(1.)
    jac=np.zeros((2,3,3,6));jac[:,0,0,0]=.02;jac[:,1,0,2]=.02
    action,_=policy.plan_contact(active_joints=np.full((2,6),.2),previous_action=np.full(12,.2),
        pad_positions=np.zeros((2,3,3)),target_points=np.ones((2,3,3))*.001,pad_jacobians=jac)
    np.testing.assert_array_equal(action.numpy()[:,[1,3,4,5]],np.full((2,4),.2,dtype=np.float32))
    assert float(action[0,0])>.2 and float(action[0,2])>.2


def test_mounted_policy_loader_preserves_explicit_legacy_json_support(tmp_path):
    import json
    from go2_pvcnn.control.dh116.contact_imitation import load_mounted_policy
    from go2_pvcnn.control.dh116.teacher_student import make_trajectory_checkpoint
    checkpoint=make_trajectory_checkpoint(progress_knots=[0.,1.],actions=[[.2]*12,[.3]*12])
    file=tmp_path/'student.json';file.write_text(json.dumps(checkpoint))
    loaded=load_mounted_policy(file,control_dt=.01,role='student')
    assert loaded.input_dim==25 and loaded.role=='student'
