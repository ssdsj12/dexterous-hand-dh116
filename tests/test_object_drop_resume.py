import pytest
from go2_pvcnn.control.dh116.tabletop_tensor_env import OBJECT_DROP_RESET_CONTRACT,TERMINAL_RESET_CONTRACT
from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes


def test_only_drop_reset_contract_is_approved_for_existing_checkpoint():
    saved=dict(terminal_reset=TERMINAL_RESET_CONTRACT,observation_size=160)
    current=dict(saved,object_drop_reset=dict(OBJECT_DROP_RESET_CONTRACT))
    assert checkpoint_contract_changes(saved,current)==['isolated_object_drop_reset_v1']
    assert 'object_drop_reset' not in saved


def test_drop_resume_does_not_allow_other_changes():
    saved=dict(terminal_reset=TERMINAL_RESET_CONTRACT,observation_size=160)
    current=dict(saved,observation_size=161,object_drop_reset=dict(OBJECT_DROP_RESET_CONTRACT))
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(saved,current)
