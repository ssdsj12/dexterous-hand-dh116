"""CPU-only synthetic checkpoint fixtures; no simulator or native outcomes."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT, semantic_sha256
from go2_pvcnn.control.dh116.tabletop_vision_action_contract import upgrade_action_contract
from go2_pvcnn.control.dh116.tabletop_vision_checkpoint_state import VISUAL_POLICY_PARAMETERS


def module():
    name = 'go2_pvcnn.control.dh116.tabletop_vision_experiment_checkpoint'
    assert importlib.util.find_spec(name), 'explicit experiment checkpoint API is missing'
    return importlib.import_module(name)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(tmp_path, *, iteration=1439):
    n = 1024
    critic = dict(num_envs=n, minibatches=8, observation_size=371,
        success_bottom_m=[.39]*512+[.40]*512,
        surface_force_sharing=dict(lift_reference_bottom_m=[.36]*512+[.37]*512),
        object_contact_material={'profile': 'synthetic_test_only'})
    critic['surface_contract_sha256'] = semantic_sha256(critic)
    base = dict(version='dh116-vision-student-ppo-v2-bounded', actor_observation_size=3844,
        critic_observation_size=371, actor_uses_privileged_state=False,
        point_cloud_shape=[2,128,3], history_frames=4, point_frame='actual_palm_body_link',
        position_unit='m', num_envs=n, rollout_steps=32, action_contract=copy.deepcopy(ACTION_CONTRACT),
        groups=[dict(label='pinch_thumb_index',num_envs=512),
                dict(label='tripod_thumb_index_middle',num_envs=512)],
        camera_mode='fixed_per_env', camera_batch_size=n, camera_resolution_wh=[224,96],
        sensor_rendering={'verified_native_readback': {'aa_op':0}},
        ppo={'minibatches':8,'epochs':5},
        memory_contract={'logical_minibatches':8,'microbatch_size':512,'inference_batch_size':256},
        privileged_critic_contract=critic)
    policy, state = {}, {}
    for i, (name, shape) in enumerate(VUAL := VISUAL_POLICY_PARAMETERS):
        policy[name] = torch.full(shape, -.01 if name != 'log_std' else -2.3)
        state[i] = dict(step=torch.tensor(40.), exp_avg=torch.full(shape,.1),
                        exp_avg_sq=torch.full(shape,.2))
    saved = dict(version='dh116-vision-student-ppo-v3-squashed-gaussian',
        contract=upgrade_action_contract(base), policy=policy,
        optimizer=dict(state=state,param_groups=[dict(params=list(range(len(VUAL))),
            lr=.000417,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False)]),
        learning_rate=.000417,iteration=iteration,timesteps=47644672,
        episodes=torch.arange(n,dtype=torch.int64)+10,
        successes=torch.arange(n,dtype=torch.int64)//4,
        torch_rng=torch.Generator().manual_seed(91).get_state(),
        cuda_rng=[torch.arange(32,dtype=torch.uint8)],
        action_derivation={'source_sha256':'d'*64,'status':'warm_start_not_exact_resume'},
        historical_extra={'untouched':['fixture',3]})
    source=tmp_path/'source.pt'
    torch.save(saved,source)
    return source,saved


def same(a,b):
    assert type(a) is type(b)
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


@pytest.mark.parametrize('n',[8,16,128])
def test_balanced_projection_preserves_full_state_and_rng_without_source_writes(tmp_path,n):
    api=module(); source,before=checkpoint(tmp_path); source_bytes=source.read_bytes()
    rng=torch.get_rng_state().clone(); output=tmp_path/'projected.pt'
    report=api.project_experiment_checkpoint(source,output,num_envs=n,expected_source_sha256=sha(source))
    after=torch.load(output,map_location='cpu',weights_only=False)
    idx=list(range(n//2))+list(range(512,512+n//2))
    for key in before:
        if key not in ('contract','episodes','successes'):same(before[key],after[key])
    for key in ('episodes','successes'):same(before[key][idx],after[key])
    assert source.read_bytes()==source_bytes and torch.equal(rng,torch.get_rng_state())
    c=after['contract']; critic=c['privileged_critic_contract']
    assert c['num_envs']==n and c['camera_batch_size']==n
    assert c['ppo']['minibatches']==min(8,n) and c['memory_contract']['logical_minibatches']==min(8,n)
    assert c['sensor_rendering']==before['contract']['sensor_rendering']
    assert c['camera_resolution_wh']==[224,96]
    assert c['memory_contract']['microbatch_size']==512 and c['memory_contract']['inference_batch_size']==256
    assert critic['success_bottom_m']==[.39]*(n//2)+[.40]*(n//2)
    assert critic['surface_force_sharing']['lift_reference_bottom_m']==[.36]*(n//2)+[.37]*(n//2)
    assert critic['surface_contract_sha256']==semantic_sha256(critic)
    assert after['experiment_projection']['retained_indices']==idx
    assert report['source_sha256']==sha(source) and report['checkpoint_sha256']==sha(output)
    assert json.loads(output.with_suffix('.json').read_text())==report
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_squashed_checkpoint
    validate_squashed_checkpoint(after,c)
    with pytest.raises(ValueError,match='same complete v3 contract'):
        validate_squashed_checkpoint(before,c)


@pytest.mark.parametrize('bad',['physics','resolution','type','unknown_reference','reference_type','source_hash','iteration','nonfinite'])
def test_projection_fails_closed_before_output(tmp_path,bad):
    api=module(); source,saved=checkpoint(tmp_path); output=tmp_path/'projected.pt'
    options=dict(num_envs=16,expected_source_sha256=sha(source))
    if bad in ('physics','resolution','type'):
        target=copy.deepcopy(saved['contract']); target['num_envs']=16
        if bad=='physics':target['privileged_critic_contract']['object_contact_material']['profile']='changed'
        elif bad=='resolution':target['camera_resolution_wh']=[112,48]
        else:target['groups'][0]['num_envs']=8.
        options['target_contract']=target
    else:
        if bad=='unknown_reference':saved['contract']['new_per_env_field']=[0]*1024
        elif bad=='reference_type':saved['contract']['privileged_critic_contract']['success_bottom_m']=tuple([.39]*1024)
        elif bad=='iteration':saved['iteration']=0
        elif bad=='nonfinite':saved['policy']['log_std'][0]=float('nan')
        torch.save(saved,source)
        options['expected_source_sha256']='0'*64 if bad=='source_hash' else sha(source)
    with pytest.raises(ValueError):api.project_experiment_checkpoint(source,output,**options)
    assert not output.exists() and not output.with_suffix('.json').exists()


@pytest.mark.parametrize('collision',['checkpoint','sidecar','race_checkpoint','race_sidecar'])
def test_projection_exclusive_destinations_and_race_ownership(tmp_path,monkeypatch,collision):
    api=module(); source,_=checkpoint(tmp_path); output=tmp_path/'out.pt'
    blocked=output if collision.endswith('checkpoint') else output.with_suffix('.json')
    original=Path.open; raced=[]
    if collision.startswith('race'):
        def racing(path,mode='r',*a,**kw):
            if path==blocked and mode in ('xb','x') and not raced:
                raced.append(True)
                with original(path,'wb') as f:f.write(b'competing')
            return original(path,mode,*a,**kw)
        monkeypatch.setattr(Path,'open',racing)
    else:blocked.write_bytes(b'competing')
    with pytest.raises(FileExistsError):
        api.project_experiment_checkpoint(source,output,num_envs=16,expected_source_sha256=sha(source))
    assert blocked.read_bytes()==b'competing'
    other=output.with_suffix('.json') if blocked==output else output
    assert not other.exists()


def test_snapshot_keeps_all_iterations_full_bytes_and_manifest_idempotently(tmp_path):
    api=module(); source,saved=checkpoint(tmp_path); root=tmp_path/'snapshots'
    rng=torch.get_rng_state().clone()
    first=api.retain_checkpoint_bundle(source,root,provenance={'dataset_sha256':'a'*64})
    pt=Path(first['checkpoint_path']); manifest=Path(first['manifest_path'])
    assert pt.read_bytes()==source.read_bytes() and first['checkpoint_sha256']==sha(source)
    assert first['iteration']==1439 and first['timesteps']==47644672
    assert first['provenance']['dataset_sha256']=='a'*64
    assert first['dataset_sha256']=={'provenance.dataset_sha256':'a'*64}
    assert first['configuration_sha256']['contract_exact_sha256']==first['state_sha256']['contract']
    assert first['state_sha256']['learning_rate']
    assert {'policy','optimizer','torch_rng','cuda_rng','counts','contract'}<=first['state_sha256'].keys()
    assert api.retain_checkpoint_bundle(source,root,provenance={'dataset_sha256':'a'*64})==first
    assert manifest.exists() and torch.equal(rng,torch.get_rng_state())
    saved['iteration']=1440; torch.save(saved,source)
    second=api.retain_checkpoint_bundle(source,root)
    assert second['checkpoint_path']!=str(pt) and pt.exists() and manifest.exists()


def test_snapshot_rejects_conflict_incomplete_adam_and_source_alias(tmp_path):
    api=module(); source,saved=checkpoint(tmp_path); root=tmp_path/'snapshots'
    first=api.retain_checkpoint_bundle(source,root)
    saved['historical_extra']['untouched'][1]=4; torch.save(saved,source)
    with pytest.raises(FileExistsError):api.retain_checkpoint_bundle(source,root)
    assert sha(first['checkpoint_path'])==first['checkpoint_sha256']
    saved['optimizer']['state'].pop(0); torch.save(saved,source)
    with pytest.raises(ValueError):api.retain_checkpoint_bundle(source,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_same_checkpoint_bytes_from_another_path_reuse_original_immutable_manifest(tmp_path):
    api=module();source,_=checkpoint(tmp_path);root=tmp_path/'snapshots'
    provenance={'dataset_sha256':'a'*64}
    first=api.retain_checkpoint_bundle(source,root,provenance=provenance)
    meta=Path(first['manifest_path']);before=meta.read_bytes();pt=Path(first['checkpoint_path'])
    original_source=source.read_bytes();original_snapshot=pt.read_bytes()
    alias=tmp_path/'same_bytes_other_path.pt';alias.write_bytes(original_source)
    second=api.retain_checkpoint_bundle(alias,root,provenance=provenance)
    assert second==first and second['source_path']==str(source.absolute())
    assert meta.read_bytes()==before and pt.read_bytes()==original_snapshot
    assert alias.read_bytes()==source.read_bytes()==original_source
    assert len(list(root.glob('iteration_*')))==1
    with pytest.raises(FileExistsError):
        api.retain_checkpoint_bundle(alias,root,provenance={'dataset_sha256':'b'*64})
    assert meta.read_bytes()==before and pt.read_bytes()==original_snapshot


def test_snapshot_detects_source_change_during_copy_and_cleans_own_bundle(tmp_path,monkeypatch):
    api=module(); source,_=checkpoint(tmp_path); root=tmp_path/'snapshots'
    original=api._copy_bytes
    def changed(src,dst):
        original(src,dst)
        source.write_bytes(source.read_bytes()+b'changed')
    monkeypatch.setattr(api,'_copy_bytes',changed)
    with pytest.raises(RuntimeError,match='source checkpoint changed'):
        api.retain_checkpoint_bundle(source,root)
    assert not list(root.glob('iteration_*'))


@pytest.mark.parametrize('name',['checkpoint.pt','manifest.json'])
def test_snapshot_sidecar_race_never_deletes_competitor(tmp_path,monkeypatch,name):
    api=module(); source,_=checkpoint(tmp_path); root=tmp_path/'snapshots'
    blocked=root/'iteration_00001439'/name
    original=Path.open; raced=[]
    def racing(path,mode='r',*a,**kw):
        if path==blocked and mode=='xb' and not raced:
            raced.append(True)
            with original(path,'wb') as stream:stream.write(b'competitor')
        return original(path,mode,*a,**kw)
    monkeypatch.setattr(Path,'open',racing)
    with pytest.raises(FileExistsError):api.retain_checkpoint_bundle(source,root)
    assert blocked.read_bytes()==b'competitor'


def test_projection_never_changes_camera_physics_action_or_reward_hashes(tmp_path):
    api=module();source,before=checkpoint(tmp_path);out=tmp_path/'n16.pt'
    result=api.project_experiment_checkpoint(source,out,num_envs=16,expected_source_sha256=sha(source))
    changes=result['experiment_projection']['changes']
    assert all(c['path'] in api._ALLOWED and c['operation']=='changed'
               and type(c['from']) is type(c['to']) for c in changes)
    assert not any('sensor_rendering' in c['path'] or 'resolution' in c['path']
                   or 'object_contact_material' in c['path'] for c in changes)
