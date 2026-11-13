"""CPU admission and GAE contracts; no policy, optimizer, or solver instances."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_rollout_admission import (
    REQUEST_VERSION, audit_rollout_admission, gae_boundary, run_audit,
)
from go2_pvcnn.control.dh116.tabletop_privileged_ppo import OnPolicyRollout, PPOConfig

ROOT=Path(__file__).resolve().parents[1]


def request():
    return {"version":REQUEST_VERSION,"receipts":[],"common_packets":{}}


def endpoint(**changes):
    return dict(env_id=0,episode_uid="fixture",reset_id=1,rod_uid="rod0",
                physics_step=10,rod_tick=96,timestamp_s=10/480,
                phase="post_action_pre_reset",**changes)


def test_denied_three_styles_do_not_construct_models_or_optimizers(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_deformable_privileged_teacher as teacher
    def forbidden(*args,**kwargs):pytest.fail("denied admission constructed policy or optimizer")
    monkeypatch.setattr(teacher.DeformablePrivilegedTeacher,"__init__",forbidden)
    monkeypatch.setattr(torch.optim.Adam,"__init__",forbidden)
    before=torch.get_rng_state().clone()
    result=audit_rollout_admission(ROOT,request())
    assert list(result["styles"]) == ["pinch","tripod","power"]
    assert len({r["checkpoint_namespace"] for r in result["styles"].values()}) == 3
    for row in result["styles"].values():
        assert row["admitted"] is False and row["blocking_reasons"]
        assert row["common_source_gate"]["passed"] is False
    assert result["policy_instances_created"] == result["optimizer_instances_created"] == 0
    assert result["new_physics_steps"] == result["optimizer_updates"] == 0
    assert torch.equal(before,torch.get_rng_state())


def test_smoke_budget_is_32_by_2_controls_and_512_measured_physics_not_64():
    result=audit_rollout_admission(ROOT,request())
    plan=result["smoke_budget"]
    assert plan["control_steps_per_env"] == 32
    assert plan["environments"] == 2 and plan["updates"] == 1
    assert plan["measured_physics_steps_per_env"] == 256
    assert plan["total_measured_ledger_rows"] == 512
    assert plan["setup_physics_steps_per_env"] is None
    assert plan["total_physics_steps_across_envs"] is None
    assert plan["measured_solver_substeps_per_env"] is None
    assert result["planned_control_transitions"] == 64
    assert result["planned_updates_are_not_executed"] is True


def test_scoped_receipt_claims_cannot_grant_qualification(tmp_path):
    receipt=tmp_path/"fake.json"
    receipt.write_text(json.dumps(dict(scope="cpu_fixtures_only_no_native_or_training",
        native_ready=True,native_qualification=True,qualified_flexible_r0=True,receipts=[])))
    case=request()
    case["receipts"]=[dict(role="r0_cpu",styles=["pinch","tripod","power"],path=str(receipt),
                          sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())]
    result=audit_rollout_admission(ROOT,case)
    assert all(not r["admitted"] for r in result["styles"].values())
    assert all("flexible_r0_native_qualification_missing" in r["blocking_reasons"]
               for r in result["styles"].values())
    assert result["scoped_receipts"][0]["grants_native_qualification"] is False


def test_forged_full_mask_still_uses_actual_source_gate():
    from go2_pvcnn.control.dh116.tabletop_teacher_preaction_readback import VERSION,CONTRACT_SHA256,SOURCE
    case=request()
    case["common_packets"]["pinch"]=dict(version=VERSION,contract_sha256=CONTRACT_SHA256,
        identity=dict(run_uuid="fixture",env_id=0,episode_uid="fixture",reset_id=1,rod_uid="rod0",
                      physics_step=2,rod_tick=0,policy_step=0,timestamp_s=2/480,
                      phase="pre_action_before_target_write"),
        values=[0.]*371,valid=[True]*371,sources=[SOURCE+":left:joint_position"]*371)
    result=audit_rollout_admission(ROOT,case)
    assert not result["styles"]["pinch"]["common_source_gate"]["passed"]
    assert "implemented source" in result["styles"]["pinch"]["common_source_gate"]["reason"]


def test_input_sha_replay_and_exclusive_outputs(tmp_path):
    manifest=tmp_path/"request.json";manifest.write_text(json.dumps(request()))
    sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    first,second=tmp_path/"first.json",tmp_path/"second.json"
    assert run_audit(ROOT,manifest,sha,first) == 3
    assert run_audit(ROOT,manifest,sha,second) == 3
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):run_audit(ROOT,manifest,sha,first)
    with pytest.raises(ValueError,match="SHA256"):run_audit(ROOT,manifest,"0"*64,tmp_path/"bad.json")
    assert not (tmp_path/"bad.json").exists()


def test_no_self_attested_approval_channel():
    case=request();case["native_approved"]=True
    with pytest.raises(ValueError,match="request"):audit_rollout_admission(ROOT,case)


@pytest.mark.parametrize("terminal,truncated,expected", [
    (False,False,(False,True,True)),(True,False,(True,False,False)),(False,True,(True,True,False)),
])
def test_boundary_masks_preserve_original_gae_semantics(terminal,truncated,expected):
    masks=gae_boundary(terminal,truncated,endpoint_identity=endpoint(),
                       next_value_identity=None if terminal else endpoint())
    assert (masks["done"],masks["bootstrap_mask"],masks["gae_carry"]) == expected


@pytest.mark.parametrize("field,value",[("episode_uid","new"),("reset_id",2),
                                       ("physics_step",11),("phase","after_reset")])
def test_truncation_cannot_bootstrap_from_reset_or_wrong_endpoint(field,value):
    changed=endpoint();changed[field]=value
    with pytest.raises(ValueError,match="endpoint"):
        gae_boundary(False,True,endpoint_identity=endpoint(),next_value_identity=changed)


@pytest.mark.parametrize("terminal,truncated",[(True,True),(1,False),(False,0)])
def test_ambiguous_or_nonboolean_terminal_flags_rejected(terminal,truncated):
    with pytest.raises(ValueError):gae_boundary(terminal,truncated,endpoint_identity=endpoint())


def test_original_gae_stops_carry_at_terminal_and_bootstraps_truncation_only():
    capability=object()
    rollout=OnPolicyRollout(2,2,session_id="cpu",policy_version=0,capability=capability)
    first=[gae_boundary(True,False,endpoint_identity=endpoint()),
           gae_boundary(False,True,endpoint_identity=endpoint(),next_value_identity=endpoint())]
    rollout._append(dict(session_id="cpu",policy_version=0,rewards=torch.tensor([1.,2.]),
        values=torch.tensor([.5,.5]),next_value=torch.tensor([100.,3.]),
        bootstrap_mask=torch.tensor([m["bootstrap_mask"] for m in first]),
        done=torch.tensor([m["done"] for m in first])),capability)
    rollout._append(dict(session_id="cpu",policy_version=0,rewards=torch.tensor([999.,999.]),
        values=torch.tensor([0.,0.]),next_value=torch.tensor([0.,0.]),
        bootstrap_mask=torch.tensor([False,False]),done=torch.tensor([True,True])),capability)
    cfg=PPOConfig()
    returns,_=rollout.returns_and_advantages(cfg)
    torch.testing.assert_close(returns[:2],torch.tensor([1.,2.+cfg.gamma*3.]))


def test_inventory_replay_uses_the_bytes_whose_sha_was_checked(tmp_path,monkeypatch):
    from test_dh116_teacher_preaction_readback import legacy_row
    ledger=tmp_path/"physics.jsonl"
    raw=(json.dumps(legacy_row())+"\n").encode()
    ledger.write_bytes(raw)
    inventory=tmp_path/"inventory.json"
    inventory.write_text(json.dumps(dict(version="dh116-teacher-legacy-source-inventory-v1",
        rows=1,admitted_371d_channels=0,unknown_indices=list(range(371)),
        inputs={"ledger":{"path":str(ledger),"sha256":hashlib.sha256(raw).hexdigest()}})))
    case=request()
    case["receipts"]=[dict(role="preaction_inventory",styles=["pinch"],path=str(inventory),
                         sha256=hashlib.sha256(inventory.read_bytes()).hexdigest())]
    original=Path.read_bytes
    reads=[]
    def read(path):
        if path == ledger:
            reads.append(path)
            return raw if len(reads)==1 else b'{"corrupted_after_hash":true}\n'
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",read)
    result=audit_rollout_admission(ROOT,case)
    assert len(reads) == 1
    assert result["scoped_receipts"][0]["summary"]["rows"] == 1
    assert result["styles"]["pinch"]["admitted"] is False
