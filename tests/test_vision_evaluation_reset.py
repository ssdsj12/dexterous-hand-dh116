"""Authored float32 reset identity and actual write/readback adapter tests."""
import copy
import importlib
import importlib.util
from types import SimpleNamespace
import pytest
import torch
import json
from test_vision_evaluation_trials import physical_state, manifest
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
from go2_pvcnn.control.dh116.tabletop_evaluation_trials import GRASP_STYLES, state_sha256


def api():
    name='go2_pvcnn.control.dh116.tabletop_vision_evaluation_reset'
    assert importlib.util.find_spec(name), 'cold-reset adapter is missing'
    return importlib.import_module(name)


def templates():
    result={}
    for i, style in enumerate(GRASP_STYLES):
        state=physical_state(i)
        state['joint_positions']=[expand([.2+i*.01]*6).tolist() for _ in range(2)]
        result[style]=state
    return result


def reset_manifest():
    value=manifest()
    for index,row in enumerate(value['trials']):
        row['physical_state']['joint_positions']=[expand([.2+index*.00001]*6).tolist() for _ in range(2)]
        row['state_sha256']=state_sha256(row['physical_state'])
    return value


def test_frozen_generator_has_200_distinct_physical_states_and_heldout_seed():
    m=api(); provenance=dict(source_sha256='2'*64,physics_sha256='3'*64,camera_sha256='4'*64)
    before=torch.get_rng_state().clone()
    validation=m.generate_cold_reset_manifest(templates(),seed=129,split='validation',provenance=provenance)
    heldout=m.generate_cold_reset_manifest(templates(),seed=1439,split='test',provenance=provenance)
    assert torch.equal(before,torch.get_rng_state())
    assert len(validation['trials'])==200
    assert len({r['state_sha256'] for r in validation['trials']+heldout['trials']})==400
    assert validation==m.generate_cold_reset_manifest(templates(),seed=129,split='validation',provenance=provenance)
    assert validation['metadata']['validation_status']=='authored_unverified'
    for trial in validation['trials']:
        state=trial['physical_state']
        for names,q in zip(state['joint_names'],state['joint_positions']):
            coupled=expand([q[names.index(a)] for a in ACTIVE])
            torch.testing.assert_close(torch.tensor(q),torch.tensor(coupled,dtype=torch.float32),atol=1e-7,rtol=0)
    with pytest.raises(ValueError,match='100'):
        m.generate_cold_reset_manifest(templates(),seed=1,split='test',provenance=provenance,count_per_style=2)


def world_stub():
    n=2; names=list(JOINTS)
    q=torch.zeros(n,2,11); roots=torch.zeros(n,2,7); roots[...,3]=1
    dw=SimpleNamespace(n=n,device='cpu',origins=torch.tensor([[0.,0,0],[2.,0,0]]),
        root_seed=roots.clone(),initial_offset=.1,seed_q=[q[:,h].clone() for h in range(2)],
        seed_qd=[q[:,h].clone() for h in range(2)],seed_wire=torch.zeros(n,7),seed_velocity=torch.zeros(n,6),
        q=q,qd=q.clone(),roots=roots,goal_roots=roots[...,:3].clone(),
        hands=[SimpleNamespace(joint_names=names,data=SimpleNamespace(root_vel_w=torch.zeros(n,6))) for _ in range(2)],
        limits=torch.tensor([-2.,2.]).expand(n,2,11,2),w=SimpleNamespace(ids=[[names.index(a) for a in ACTIVE]]*2))
    return dw


def test_adapter_changes_real_reset_buffers_and_readback_evidence_is_not_a_flag():
    m=api(); dw=world_stub(); task=SimpleNamespace(n=2,_seed={'command':torch.zeros(12)})
    for name in m.TASK_RESET_FIELDS: setattr(task,name,torch.zeros(2,12))
    value=reset_manifest(); m.install_evaluation_reset(dw,task,value,min_trials_per_style=1)
    assert dw.seed_q[0][0,0].item()==pytest.approx(.2)
    assert dw.seed_q[0][1,0].item()==pytest.approx(.20001)
    assert dw.root_seed[0,0,2].item()+dw.initial_offset==pytest.approx(.4)
    assert dw.seed_wire[1,0].item()==2.
    assert task._per_env_reset_seed['command'][0,0].item()==pytest.approx(.2)
    # Simulate the existing native reset reading these buffers, without claiming physics.
    dw.q=torch.stack(dw.seed_q,1); dw.qd=torch.stack(dw.seed_qd,1)
    dw.roots=dw.root_seed.clone(); dw.roots[...,2]+=dw.initial_offset
    dw.wp=dw.seed_wire[:,:3]; dw.wq=dw.seed_wire[:,3:]; dw.wv=dw.seed_velocity
    evidence=m.readback_evaluation_reset(dw,value,min_trials_per_style=1)
    assert evidence['unique_readback_count']==2 and not evidence['native_validation_complete']
    assert len(evidence['actual_states'])==2
    dw.q[1]=dw.q[0]; dw.wp[1]=dw.wp[0]+dw.origins[1]
    with pytest.raises(RuntimeError,match='mismatch|duplicate'):
        m.readback_evaluation_reset(dw,value,min_trials_per_style=1)


def test_adapter_rejects_independently_authored_passive_angle_before_any_mutation():
    m=api(); dw=world_stub(); task=SimpleNamespace(n=2,_seed={'command':torch.zeros(12)})
    for name in m.TASK_RESET_FIELDS: setattr(task,name,torch.zeros(2,12))
    value=reset_manifest(); state=value['trials'][0]['physical_state']
    state['joint_positions'][0][2]+=.001
    value['trials'][0]['state_sha256']=state_sha256(state)
    before=dw.seed_q[0].clone()
    with pytest.raises(ValueError,match='coupling'):
        m.install_evaluation_reset(dw,task,value,min_trials_per_style=1)
    assert torch.equal(before,dw.seed_q[0])


def test_real_initialization_entrypoint_checks_checkpoint_template_hashes_and_never_overwrites(tmp_path):
    from test_vision_action_derivation import source_checkpoint
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import file_sha256
    source,saved=source_checkpoint(tmp_path)
    paths=[]
    for index in range(2):
        state=templates()[GRASP_STYLES[index]]
        path=tmp_path/f'initial{index}.json'
        path.write_text(json.dumps(dict(format='dh116-adaptive-grasp-initialization-v1',source_contract={},
            roots=state['roots'],joint_positions=state['joint_positions'],grasp_command=[.25]*12,
            wire_pose=state['wire_pose'])))
        paths.append(path)
    saved['contract']['primary_reset_sha256']=file_sha256(paths[0])
    saved['contract']['secondary_reset_sha256']=file_sha256(paths[1])
    torch.save(saved,source); digest=file_sha256(source)
    m=api(); assert hasattr(m,'freeze_cold_reset_manifest'), 'real-template freezing entrypoint missing'
    output=tmp_path/'validation.json'
    result=m.freeze_cold_reset_manifest(source,*paths,output,seed=19,split='validation',wrist_joint_names=())
    assert output.exists() and len(result['trials'])==200
    assert output.with_suffix('.provenance.json').exists()
    provenance=json.loads(output.with_suffix('.provenance.json').read_text())
    from go2_pvcnn.control.dh116.tabletop_evaluation_trials import _digest
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import evaluation_physics_sha256
    assert provenance['source_privileged_critic_contract_sha256']==_digest(saved['contract']['privileged_critic_contract'])
    assert result['metadata']['physics_sha256']==evaluation_physics_sha256(saved['contract']['privileged_critic_contract'])
    assert file_sha256(source)==digest
    with pytest.raises(FileExistsError):
        m.freeze_cold_reset_manifest(source,*paths,output,seed=19,split='validation',wrist_joint_names=())
    paths[0].write_text('{}')
    with pytest.raises(ValueError,match='hash'):
        m.freeze_cold_reset_manifest(source,*paths,tmp_path/'new.json',seed=20,split='test',wrist_joint_names=())
