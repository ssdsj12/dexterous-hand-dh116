import copy
import pytest


def contracts():
    saved=dict(observation_version='dh116-surface-sharing-lift-objective-v2',
        observation_size=351,policy_command_size=24,physics_hz=480,reward='same',
        close_seed_sha256='a'*64,grasp_initialization_sha256='a'*64)
    current=dict(saved,close_seed_sha256='b'*64,grasp_initialization_sha256='b'*64)
    family=[dict(label=label,contract=copy.deepcopy(c),source_dataset_sha256=digest*64,
                 proof_path=f'/native/{label}/proof.json')
            for label,c,digest in [('pinch_thumb_index',saved,'c'),('tripod_thumb_index_middle',current,'d')]]
    return saved,current,family


def test_only_declared_reset_family_can_resume_across_initial_grasps():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import surface_checkpoint_contract_changes
    saved,current,family=contracts();original=copy.deepcopy(saved)
    assert surface_checkpoint_contract_changes(saved,current,family)==['surface_reset_family_switch']
    assert saved==original
    with pytest.raises(ValueError):surface_checkpoint_contract_changes(saved,current,None)


def test_reset_family_resume_never_hides_physics_or_undeclared_seed_changes():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import surface_checkpoint_contract_changes
    saved,current,family=contracts()
    bad=dict(current,physics_hz=960)
    with pytest.raises(ValueError):surface_checkpoint_contract_changes(saved,bad,family)
    bad=dict(current,close_seed_sha256='e'*64,grasp_initialization_sha256='e'*64)
    with pytest.raises(ValueError):surface_checkpoint_contract_changes(saved,bad,family)


def test_feedback_reset_switch_rebinds_digest_without_mutating_source():
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_NAME, bind_contract
    from go2_pvcnn.control.dh116.tabletop_tensor_training import surface_checkpoint_contract_changes

    saved, current, family = contracts()
    saved, current = [bind_contract(dict(c, action=ACTION_NAME)) for c in (saved, current)]
    for member, contract in zip(family, (saved, current)):
        member['contract'] = copy.deepcopy(contract)
    original = copy.deepcopy(saved)
    assert saved['surface_contract_sha256'] != current['surface_contract_sha256']
    assert surface_checkpoint_contract_changes(saved, current, family) == ['surface_reset_family_switch']
    assert saved == original
    with pytest.raises(ValueError, match='SHA mismatch'):
        surface_checkpoint_contract_changes(saved, dict(current, surface_contract_sha256='0'*64), family)
