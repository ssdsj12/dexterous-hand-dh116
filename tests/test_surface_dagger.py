import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_surface_contract import bind_contract
from go2_pvcnn.control.dh116.tabletop_surface_dagger import (
    build_failure_relabels, failure_rows)


def _contract():
    return bind_contract(dict(observation_size=351,
        observation_version='dh116-surface-sharing-lift-objective-v2',
        action_size=24,policy_command_size=24,
        action='24D_free_fingers_worldXYZ_intrinsicXYZ_surface_sharing_v1'))


def test_dagger_uses_only_aligned_student_failure_rows_and_24d_labels():
    contract=_contract()
    evaluation=dict(observations=torch.zeros(3,2,351),
        nominal_actions=torch.zeros(3,2,24),failure=torch.tensor([[0,1],[0,0],[2,0]]))
    assert failure_rows(evaluation).sum().item()==2
    payload=build_failure_relabels(evaluation,torch.zeros(2,24),contract,
        source_sha256='a'*64)
    assert payload['sample_count']==2
    assert payload['teacher_servo_used'] is False
    assert payload['execution_owner']=='student'


def test_dagger_rejects_wrong_label_count_or_width():
    contract=_contract()
    evaluation=dict(observations=torch.zeros(2,1,351),
        nominal_actions=torch.zeros(2,1,24),failure=torch.tensor([[1],[0]]))
    with pytest.raises(ValueError):
        build_failure_relabels(evaluation,torch.zeros(1,23),contract,source_sha256='b'*64)
