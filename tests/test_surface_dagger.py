import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_surface_contract import bind_contract
from go2_pvcnn.control.dh116.tabletop_surface_dagger import (
    build_failure_relabels, failure_rows, contract_sha)


def _contract():
    return bind_contract(dict(observation_size=351,
        observation_version='dh116-surface-sharing-lift-objective-v2',
        action_size=24,policy_command_size=24,
        action='24D_free_fingers_worldXYZ_intrinsicXYZ_surface_sharing_v1'))


def test_dagger_uses_only_aligned_student_failure_rows_and_24d_labels():
    contract=_contract()
    evaluation=dict(observations=torch.zeros(3,2,371),
        nominal_actions=torch.zeros(3,2,24),failure=torch.tensor([[0,1],[0,0],[2,0]]))
    assert failure_rows(evaluation).sum().item()==2
    proof=dict(version='dh116-native-failure-correction-candidates-v1',native_correction_qualified=True,
        source_evaluation_sha256='a'*64,contract_sha256=contract_sha(contract),
        rows=[dict(tick=i,env=j,observation=evaluation['observations'][i,j],action=torch.zeros(24)) for i,j in [(0,1),(2,0)]])
    payload=build_failure_relabels(evaluation,proof,contract,
        source_sha256='a'*64)
    assert payload['sample_count']==2
    assert payload['teacher_servo_used'] is False
    assert payload['execution_owner']=='student'


def test_dagger_rejects_wrong_label_count_or_width():
    contract=_contract()
    evaluation=dict(observations=torch.zeros(2,1,371),
        nominal_actions=torch.zeros(2,1,24),failure=torch.tensor([[1],[0]]))
    with pytest.raises(ValueError):
        build_failure_relabels(evaluation,torch.zeros(1,23),contract,source_sha256='b'*64)


@pytest.mark.parametrize('width',[351,26,401])
def test_dagger_rejects_wrong_observation_contract(width):
    with pytest.raises(ValueError):
        failure_rows(dict(observations=torch.zeros(2,1,width),failure=torch.ones(2,1,dtype=torch.long)))


def test_unqualified_native_candidate_cannot_enter_bc():
    contract=_contract()
    evaluation=dict(observations=torch.zeros(2,1,371),nominal_actions=torch.zeros(2,1,24),
        failure=torch.tensor([[0],[1]]))
    proof=dict(version='dh116-native-failure-correction-candidates-v1',native_correction_qualified=False,
        source_evaluation_sha256='a'*64,contract_sha256=contract_sha(contract),rows=[])
    with pytest.raises(ValueError,match='native correction validation'):
        build_failure_relabels(evaluation,proof,contract,source_sha256='a'*64)


def test_corrective_auxiliary_step_leaves_ppo_optimizer_and_critic_untouched():
    from types import SimpleNamespace as NS
    from go2_pvcnn.control.dh116.tabletop_surface_dagger import CorrectiveRegularizer,DATASET_VERSION
    from go2_pvcnn.control.dh116.tabletop_surface_imitation import make_actor
    c=_contract()
    data=dict(version=DATASET_VERSION,native_correction_qualified=True,execution_owner='student',
        teacher_servo_used=False,contract=c,contract_sha256=contract_sha(c),
        observations=torch.randn(8,371),teacher_actions=torch.zeros(8,24))
    actor=make_actor(371);critic=torch.nn.Linear(371,1)
    critic_before={k:v.clone() for k,v in critic.state_dict().items()}
    ppo=torch.optim.Adam([*actor.parameters(),*critic.parameters()])
    aux=torch.optim.Adam(actor.parameters(),lr=1e-4)
    algorithm=NS(policy=NS(actor=actor,critic=critic),optimizer=ppo)
    demo=NS(independent_optimizer=True,optimizer_for=lambda a:aux)
    r=CorrectiveRegularizer(data,c,'cpu').update(algorithm,demo,batch_size=4)
    assert r['ppo_storage_used'] is False and r['samples']==4
    assert len(aux.state)>0 and len(ppo.state)==0
    for k,v in critic.state_dict().items():assert torch.equal(v,critic_before[k])
