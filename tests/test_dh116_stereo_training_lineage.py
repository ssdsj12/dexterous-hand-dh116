"""Artifact inventory fixtures do not establish native or heldout qualification."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_contract import digest,file_sha256,physical_initial_sha256,make_contract
from test_dh116_stereo_evaluation import records


def module():
    name='go2_pvcnn.control.dh116.tabletop_stereo_training_lineage'
    assert importlib.util.find_spec(name),'versioned training source registry is missing'
    return importlib.import_module(name)


def inputs(tmp_path,name='raw'):
    model=tmp_path/(name+'_model.pt');model.write_bytes(b'CPU fixture; no trained model')
    raw=tmp_path/(name+'.jsonl');rows=records(uid=name)
    rows[0].update(initial_measurement_valid=True,physical_initial_condition_source='actual_sdk_readback',source_episode_group='source:'+name)
    raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return model,raw,rows


def test_registry_recomputes_actual_initials_and_never_upgrades_cpu_scope(tmp_path):
    m=module();model,raw,rows=inputs(tmp_path)
    path=tmp_path/'manifest.json'
    result=m.register_training_lineage(path,model,[raw],complete_source_inventory=True)
    loaded=m.load_training_lineage(path,expected_model_checkpoint_sha256=file_sha256(model))
    assert loaded['model_checkpoint_sha256']==file_sha256(model)
    assert loaded['physical_initial_condition_sha256']==[physical_initial_sha256(rows[0]['physical_initial_condition'])]
    assert loaded['source_episode_groups']==['source:raw']
    assert loaded['complete_source_inventory'] is False and loaded['native_qualified'] is False
    assert loaded['raw_episode_artifacts'][0]['sha256']==file_sha256(raw)
    assert result['sha256']==file_sha256(path)
    with pytest.raises(ValueError,match='complete|inventory'):
        m.load_training_lineage(path,require_complete=True)
    with pytest.raises(FileExistsError):m.register_training_lineage(path,model,[raw])


def test_tampered_raw_and_handwritten_initial_lists_cannot_replace_registry(tmp_path):
    m=module();model,raw,rows=inputs(tmp_path);path=tmp_path/'manifest.json'
    m.register_training_lineage(path,model,[raw])
    value=json.loads(path.read_text());value['physical_initial_condition_sha256']=['0'*64]
    value['manifest_sha256']=digest({key:item for key,item in value.items() if key!='manifest_sha256'})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='raw|initial|derived|inventory'):
        m.load_training_lineage(path)
    path.unlink();m.register_training_lineage(path,model,[raw])
    rows[0]['physical_initial_condition']['object_pose'][0]=.01
    raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='SHA|artifact'):
        m.load_training_lineage(path)


def test_parent_union_is_reopened_and_bound_to_each_actual_model(tmp_path):
    m=module();model,raw,_=inputs(tmp_path,'parent');parent=tmp_path/'parent.json'
    m.register_training_lineage(parent,model,[raw])
    child=tmp_path/'child.pt';child.write_bytes(b'new CPU model')
    path=tmp_path/'child.json'
    m.register_training_lineage(path,child,[],parent_manifests=[parent])
    result=m.load_training_lineage(path,expected_model_checkpoint_sha256=file_sha256(child))
    assert result['source_episode_groups']==['source:parent'] and len(result['parents'])==1
    assert result['raw_episode_artifacts'][0]['sha256']==file_sha256(raw)
    assert result['complete_source_inventory'] is False
    with pytest.raises(ValueError,match='model|checkpoint'):
        m.load_training_lineage(path,expected_model_checkpoint_sha256='a'*64)
    parent.write_text(parent.read_text()+' ')
    with pytest.raises(ValueError,match='SHA|artifact'):
        m.load_training_lineage(path)


def test_raw_headers_require_real_initial_measurement_and_source_group(tmp_path):
    m=module();model,raw,rows=inputs(tmp_path)
    rows[0]['initial_measurement_valid']=False
    raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result=m.register_training_lineage(tmp_path/'bad.json',model,[raw],complete_source_inventory=True)
    value=m.load_training_lineage(result['path'])
    assert value['complete_source_inventory'] is False
    assert any('measurement' in reason for reason in value['incomplete_reasons'])


def test_actor_package_carries_only_metadata_training_sources_and_actual_identity(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_actor_package,load_deployment_actor
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor
    source=tmp_path/'dataset.pt';source.write_bytes(b'CPU source fixture')
    provenance={'dataset_artifact':dict(path=str(source),sha256=file_sha256(source)),
        'complete_source_inventory':False,'scope':'CPU_fixture_only'}
    network=dict(embed_dim=32,history=4,num_heads=4,num_layers=1,dropout=0.)
    actor=make_stereo_actor(network);path=tmp_path/'actor.pt'
    save_actor_package(path,actor,make_contract('bc',dict(network=network,provenance=provenance),calibration_sha256='c'*64),network_config=network)
    loaded=load_deployment_actor(path)
    assert loaded.deployment_contract['training_provenance']==provenance
    assert loaded.deployment_package_artifact==dict(path=str(path.resolve()),sha256=file_sha256(path))
    poisoned=deepcopy(provenance);poisoned['teacher_state_dict']={'weight':torch.zeros(24)}
    with pytest.raises(ValueError,match='metadata|tensor|provenance|Teacher'):
        module().validate_training_provenance(poisoned)


def test_registration_cli_is_explicit_exclusive_and_does_not_grant_complete_scope(tmp_path,capsys):
    m=module();model,raw,_=inputs(tmp_path);path=tmp_path/'manifest.json'
    assert m.main(['--model-checkpoint',str(model),'--raw-episode-artifact',str(raw),'--output',str(path)])==0
    value=json.loads(capsys.readouterr().out)
    assert value['complete_source_inventory'] is False and value['native_qualified'] is False


def test_export_and_resume_preserve_registered_runtime_sources_without_changing_config(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_student_trainer
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_student_checkpoint,load_student_checkpoint,export_actor,load_deployment_actor
    config=dict(network=dict(embed_dim=32,history=4,num_heads=4,num_layers=1,dropout=0.),ppo={},rollout_steps=1)
    trainer=make_student_trainer(config,1)
    trainer.training_provenance={'raw_episode_source_paths':[str(tmp_path/'unclosed_native.jsonl')],
        'source_inventory_complete':False}
    contract=make_contract('student',config,calibration_sha256='c'*64)
    checkpoint=tmp_path/'checkpoint.pt';save_student_checkpoint(checkpoint,trainer,contract,network_config=config['network'])
    restored=make_student_trainer(config,1);load_student_checkpoint(checkpoint,restored,contract)
    assert restored.training_provenance==trainer.training_provenance
    package=tmp_path/'package.pt';export_actor(checkpoint,package)
    deployed=load_deployment_actor(package)
    assert deployed.deployment_contract['training_provenance']['raw_episode_source_paths']==trainer.training_provenance['raw_episode_source_paths']
    assert deployed.deployment_contract['training_provenance']['exported_from_checkpoint_artifact']['sha256']==file_sha256(checkpoint)
