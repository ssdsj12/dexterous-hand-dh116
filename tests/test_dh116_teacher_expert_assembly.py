"""CPU-only probes of expert packing and exact physical/action joins."""
import json

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_privileged_expert_assembly as build
from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as promotion
from go2_pvcnn.control.dh116.tabletop_privileged_checkpoint import canonical_sha256
from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT, ACTION_NAME, LEGACY_OBSERVATION_VERSION, bind_contract


def artifacts(path, fault=None):
    identity = dict(state_contract_sha256='a'*64, action_contract_sha256=canonical_sha256(ACTION_CONTRACT))
    source = path / 'source.pt'
    proof = path / 'qualification.json'
    contract = bind_contract(dict(action_size=24, action=ACTION_NAME, observation_size=351,
        observation_version=LEGACY_OBSERVATION_VERSION))
    actions = torch.arange(48, dtype=torch.float32).view(2, 24) / 100
    valid = torch.ones(2, 371, dtype=torch.bool)
    if fault == 'invalid_371':
        valid[0, 70] = False
    torch.save(dict(action_owner='legacy_teacher_mpc', action_contract=ACTION_CONTRACT,
        privileged_observation_contract=contract, observations=torch.zeros(2, 371), actions=actions,
        privileged_valid=valid, sample_metadata=[dict(episode_uid='cpu-fixture', run_uuid='cpu',
            env_id=0, tick=i, physics_step=20+i*8, physical_substeps=(8 if i == 0 else 5),
            terminal_in_interval=i==1, policy_clock_before={'world_step':10+i*8})
            for i in range(2)]), source)
    rows = [dict(kind='episode_start', episode_uid='cpu-fixture')]
    for i in range(13):
        a = actions[i // 8].tolist()
        if i == 6 and fault == 'issued_mismatch':
            a[2] = .5
        rows.append(dict(kind='sample', episode_uid='cpu-fixture', env_id=0, run_uuid='cpu',
            physics_step=11+i, policy_tick=i//8, actor_issued_action=a, terminal=i==12,
            object_net_lift_m=.031, formal_hold_s=5., formal_drift_m=.0002))
    raw = path / 'cpu_raw.jsonl'
    raw.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    proof.write_text(json.dumps(dict(raw_artifact=dict(path=str(raw), sha256=file_sha256(raw)),
        producer=dict(config_artifact=dict(sha256='b'*64)))))
    manifest = dict(version='dh116-privileged-expert-assembly-v1',
        episodes=[dict(episode_uid='cpu-fixture', source_path=str(source), source_sha256=file_sha256(source),
            qualification_path=str(proof), qualification_sha256=file_sha256(proof))])
    return identity, manifest


def CPU_only_gate(monkeypatch, *, verified=True):
    """Mock the independent native gate to test serialization, never actual qualification."""
    def verify(path, *, expected_identity, subject=None):
        if not verified:
            raise ValueError('independent native qualification unavailable')
        return dict(episode_uid='cpu-fixture', owner='mpc', qualified_success=True,
            native_evidence_verified=True, sample_count=13)
    def produced(_proof, _base, binding, **_):
        path = build.Path(binding['path'])
        return path.read_bytes(), path
    monkeypatch.setattr(promotion, 'verify_native_episode', verify)
    monkeypatch.setattr(promotion, '_process_output', produced)


@pytest.mark.parametrize('fault', ['invalid_371', 'issued_mismatch', 'raw_gate', 'proof_sha'])
def test_expert_assembly_refuses_ineligible_rows_before_creating_output(tmp_path, monkeypatch, fault):
    identity, manifest = artifacts(tmp_path, fault)
    CPU_only_gate(monkeypatch, verified=fault!='raw_gate')
    if fault == 'proof_sha':
        manifest['episodes'][0]['qualification_sha256']='0'*64
    output = tmp_path / 'result'
    with pytest.raises(ValueError):
        build.assemble_experts(manifest, identity, output)
    assert not output.exists()


def test_exact_source_rows_and_existing_expert_schema_pack_after_gate(tmp_path, monkeypatch):
    identity, manifest = artifacts(tmp_path)
    CPU_only_gate(monkeypatch)
    output = tmp_path / 'result'
    result = build.assemble_experts(manifest, identity, output)
    payload = torch.load(output / 'privileged_native_expert.pt', weights_only=True)
    assert payload['version'] == 'dh116-privileged-native-expert-v1'
    assert payload['observations'].shape == (2,371) and payload['actions'].shape == (2,24)
    assert payload['valid_mask'].all() and result['episodes'] == 1
    assert payload['episodes'][0]['source_sha256'] == manifest['episodes'][0]['source_sha256']
    assert result['dataset_sha256'] == file_sha256(output / 'privileged_native_expert.pt')
    assert (output / 'episode_00000.json').is_file()
