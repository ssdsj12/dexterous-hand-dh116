import hashlib
import json
import torch
import pytest
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.tabletop_surface_demonstrations import load_teacher_actions


def test_legacy_teacher_requires_explicit_sha_bound_action_audit_for_new_contract(tmp_path):
    path=tmp_path/'teacher.pt'; torch.save(dict(metadata=dict(qualified_lift=True,action_size=24),
        control=dict(action=torch.zeros(2,1,24))),path)
    with pytest.raises(ValueError,match='audit'):
        load_teacher_actions(path,require_action_contract=True)
    proof=tmp_path/'audit.json';proof.write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),action_contract=ACTION_CONTRACT)))
    actions,identity=load_teacher_actions(path,require_action_contract=True,action_audit=proof)
    assert actions.shape==(2,24) and len(identity['action_contract_sha256'])==64
    torch.save(dict(metadata=dict(qualified_lift=True,action_size=24),
        control=dict(action=torch.ones(2,1,24))),path)
    with pytest.raises(ValueError,match='SHA'):
        load_teacher_actions(path,require_action_contract=True,action_audit=proof)
