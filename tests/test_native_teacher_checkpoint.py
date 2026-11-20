"""CPU fixture train-state only; never a native qualification or physics snapshot."""
from copy import deepcopy
import importlib.util
import io
import hashlib
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from test_native_teacher_context import backend_profile
from test_native_teacher_collector import fixture as physical_fixture


def fixture(monkeypatch,tmp_path,profile,**kwargs):
    tmp_path.mkdir(parents=True,exist_ok=True)
    import test_deformable_r0_replay as replay
    from test_dh116_dual_hand_episode import setup as original_setup
    monkeypatch.setattr(replay,"episode_setup",original_setup)
    return physical_fixture(monkeypatch,tmp_path,profile,**kwargs)


def checkpoint():
    name="go2_pvcnn.control.dh116.tabletop_native_teacher_checkpoint"
    assert importlib.util.find_spec(name) is not None, "independent native Teacher checkpoint missing"
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_checkpoint as ck
    return ck


def contract(trainer):
    return checkpoint().build_training_contract(trainer, scope="cpu_fixture",
        reward_sha256="1"*64,target_sha256="2"*64,reset_sha256="3"*64,
        source_sha256={"synthetic_cpu_source":"4"*64},config_sha256="5"*64)


def trained(monkeypatch,tmp_path,profile):
    from test_native_teacher_training import runner
    c,*_=fixture(monkeypatch,tmp_path,profile,rollout_steps=2)
    runner().run_updates(c,additional_updates=1)
    return c


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b


def training_snapshot(trainer):
    """All live training state, including absent/present progress provenance."""
    names=('_native_accepted_minibatches','_native_completed_update',
        '_native_training_failed','_native_training_origin')
    markers={name:(name in trainer.__dict__,deepcopy(getattr(trainer,name,None))) for name in names}
    return (deepcopy(trainer.model.state_dict()),deepcopy(trainer.optimizer.state_dict()),
        deepcopy(trainer.counters),trainer.policy_version,torch.get_rng_state().clone(),
        random.getstate(),deepcopy(np.random.get_state()),markers,
        deepcopy(trainer._pending),len(trainer.rollout),id(trainer.rollout),
        id(trainer._pending),deepcopy(trainer.rollout._rows),
        (trainer.rollout.session_id,trainer.rollout.policy_version,
         id(trainer.rollout._capability),id(trainer.rollout.actor_storage)))


@pytest.mark.parametrize("loaded,extra,expected",[(0,1,1),(1,1,2),(3,2,5)])
def test_additional_target(loaded,extra,expected):
    assert checkpoint().stop_update_target(loaded,additional_updates=extra)==expected


@pytest.mark.parametrize("loaded,extra",[(True,1),(-1,1),(1,True),(1,0),(1,-1),(1.,1),(1,1.)])
def test_additional_target_exact_integers(loaded,extra):
    with pytest.raises(ValueError):checkpoint().stop_update_target(loaded,additional_updates=extra)


def test_clean_exclusive_restore_adam_rng_and_identical_next_update(monkeypatch,tmp_path,backend_profile):
    ck=checkpoint();c=trained(monkeypatch,tmp_path/'one',backend_profile);t=c.trainer
    path=tmp_path/'checkpoint.pt';identity=contract(t)
    saved=ck.save_checkpoint(path,t,identity)
    payload=torch.load(path,weights_only=True)
    assert payload['normalization'] is None and payload['accepted_minibatch_updates']>0
    assert payload['physical_state_saved'] is False and payload['restart_requires_verified_reset'] is True
    with pytest.raises(FileExistsError):ck.save_checkpoint(path,t,identity)
    clone,*_=fixture(monkeypatch,tmp_path/'two',backend_profile,rollout_steps=2)
    loaded=ck.load_checkpoint(path,clone.trainer,contract(clone.trainer))
    equal_tree(t.model.state_dict(),clone.trainer.model.state_dict())
    equal_tree(t.optimizer.state_dict(),clone.trainer.optimizer.state_dict())
    assert clone.trainer.counters==t.counters==dict(environment_steps=2,updates=1)
    assert clone.trainer.policy_version==1 and loaded['accepted_minibatch_updates']==payload['accepted_minibatch_updates']
    # Both trainers replay identical owned observations/RNG through original PPO.
    from test_native_teacher_model import packed
    _,obs=packed(backend_profile);obs={k:v[:1].clone() for k,v in obs.items()}
    rng=torch.get_rng_state().clone();py=random.getstate();npr=np.random.get_state()
    def update(tr):
        samples=[]
        for _ in range(2):
            sample=tr.collect_step(obs,obs);samples.append(sample['actions'].clone())
            tr.record_step(torch.tensor([.75]),torch.tensor([False]),torch.tensor([.2]),
                executed_actions=sample['actions'],collection_token=sample['collection_token'])
        metrics=tr.update();return samples,metrics
    a,ma=update(t)
    torch.set_rng_state(rng);random.setstate(py);np.random.set_state(npr)
    b,mb=update(clone.trainer)
    equal_tree(a,b);equal_tree(ma,mb)
    equal_tree(t.model.state_dict(),clone.trainer.model.state_dict())
    equal_tree(t.optimizer.state_dict(),clone.trainer.optimizer.state_dict())
    assert t.counters['updates']==clone.trainer.counters['updates']==2
    assert ck.stop_update_target(loaded['counters']['updates'],additional_updates=1)==2
    assert saved['version']!= 'dh116-flexible-measured-common-checkpoint-cpu-v1'


_PAYLOAD_DAMAGE=('reward','target','reset','source','config','context','backend','grasp','old_checkpoint',
    'noise','adam','adam_step','rng','policy_shape','policy_extra','normalization','accepted','origin')


def damage_payload(data,damage):
    if damage in ('reward','target','reset','source','config'):data['identity']['training_contract'][damage+'_sha256']='9'*64
    elif damage in ('context','backend','grasp'):data['identity']['model'][{'context':'context_schema_sha256','backend':'backend_sha256','grasp':'grasp'}[damage]]='damaged'
    elif damage=='old_checkpoint':data['version']='dh116-flexible-measured-common-checkpoint-cpu-v1'
    elif damage=='noise':data['policy']['log_std'].fill_(100.)
    elif damage=='adam':data['optimizer']['state'][0].pop('exp_avg')
    elif damage=='adam_step':data['optimizer']['state'][0]['step'].fill_(999.)
    elif damage=='rng':data['rng']['torch']=torch.zeros(2,dtype=torch.uint8)
    elif damage=='policy_shape':data['policy']['actor.0.weight']=torch.zeros(1)
    elif damage=='policy_extra':data['policy']['_extra_state']['grasp']='tripod'
    elif damage=='normalization':data['normalization']={}
    elif damage=='accepted':data['accepted_minibatch_updates']=True
    elif damage=='origin':data['origin']={'kind':'silent_smoke_to_full'}
    elif damage=='boolean_step':data['optimizer']['state'][0]['step']=torch.tensor(True)
    elif damage=='negative_second_moment':data['optimizer']['state'][0]['exp_avg_sq'].fill_(-1.)
    elif damage=='zero_updates':
        data['policy_version']=0;data['counters']['updates']=0;data['accepted_minibatch_updates']=0
    elif damage=='physical_saved':data['physical_state_saved']=True
    elif damage=='no_verified_reset':data['restart_requires_verified_reset']=False
    elif damage=='rollout':data['rollout_steps']+=1
    elif damage=='missing_field':data.pop('origin')
    elif damage=='numpy_rng':data['rng']['numpy'][1].fill_(-1)
    else:raise AssertionError('unknown payload damage')


@pytest.mark.parametrize('damage',_PAYLOAD_DAMAGE)
def test_reject_payload_atomically(monkeypatch,tmp_path,backend_profile,damage):
    ck=checkpoint();c=trained(monkeypatch,tmp_path/'src',backend_profile);path=tmp_path/'good.pt';identity=contract(c.trainer)
    ck.save_checkpoint(path,c.trainer,identity);data=torch.load(path,weights_only=True)
    clone,*_=fixture(monkeypatch,tmp_path/'dst',backend_profile,rollout_steps=2);t=clone.trainer
    t._native_training_origin=dict(kind='resume',checkpoint_sha256='a'*64)
    t._native_accepted_minibatches=7;t._native_completed_update=0;t._native_training_failed=True
    before=training_snapshot(t)
    damage_payload(data,damage)
    bad=tmp_path/'bad.pt';torch.save(data,bad)
    with pytest.raises((ValueError,RuntimeError)):ck.load_checkpoint(bad,t,contract(t))
    equal_tree(before,training_snapshot(t))


def test_authorized_resume_rejects_replaced_checkpoint_before_restore(monkeypatch,tmp_path,backend_profile):
    ck=checkpoint()
    collector,*_=fixture(monkeypatch,tmp_path/'live',backend_profile,rollout_steps=2)
    trainer=collector.trainer
    identity=contract(trainer)
    path=tmp_path/'replacement.pt'
    path.write_bytes(b'replaced after scheduler verification')
    before=training_snapshot(trainer)

    def deserialize(*args, **kwargs):
        pytest.fail("unauthorized checkpoint bytes must be rejected before deserialization")

    monkeypatch.setattr(ck.torch,'load',deserialize)
    with pytest.raises(ValueError,match='checkpoint SHA-256 differs'):
        ck.load_checkpoint(path,trainer,identity,expected_sha256='a'*64)
    with pytest.raises(ValueError,match='checkpoint SHA-256 differs'):
        ck.checkpoint_trainer_factory(lambda:trainer,path,contract,expected_sha256='a'*64)()
    equal_tree(before,training_snapshot(trainer))


def test_pending_and_rollout_and_never_updated_save_rejected(monkeypatch,tmp_path,backend_profile):
    ck=checkpoint();c,*_=fixture(monkeypatch,tmp_path,backend_profile,rollout_steps=2);identity=contract(c.trainer)
    with pytest.raises(RuntimeError):ck.save_checkpoint(tmp_path/'fresh.pt',c.trainer,identity)
    c.collect()
    with pytest.raises(RuntimeError):ck.save_checkpoint(tmp_path/'rollout.pt',c.trainer,identity)
    assert not list(tmp_path.glob('*.pt'))


def test_resume_factory_loads_before_collector_epoch_then_additional_one_reaches_two(monkeypatch,tmp_path,backend_profile):
    ck=checkpoint()
    assert hasattr(ck,'checkpoint_trainer_factory'), 'checkpoint trainer factory integration missing'
    c=trained(monkeypatch,tmp_path/'first',backend_profile);path=tmp_path/'resume.pt'
    ck.save_checkpoint(path,c.trainer,contract(c.trainer))
    from go2_pvcnn.control.dh116 import tabletop_deformable_privileged_ppo as ppo
    original=ppo.DeformablePPOTrainer
    wrapped=ck.checkpoint_trainer_factory(original,path,contract)
    monkeypatch.setattr(ppo,'DeformablePPOTrainer',wrapped)
    resumed,ep,rod,*_=fixture(monkeypatch,tmp_path/'second',backend_profile,rollout_steps=2)
    assert resumed.trainer.counters['updates']==1 and resumed.trainer.policy_version==1
    assert rod.rope.steps==0 and resumed.context.current.values[0,49]==0
    assert not resumed.history._previous.any()
    from test_native_teacher_training import runner
    result=runner().run_updates(resumed,additional_updates=1)
    assert result['loaded_updates']==1 and result['target_updates']==result['updates']==2
    assert rod.rope.steps==16
    second=tmp_path/'second-clean.pt';ck.save_checkpoint(second,resumed.trainer,contract(resumed.trainer))
    data=torch.load(second,weights_only=True)
    assert data['origin']['kind']=='resume' and data['accepted_minibatch_updates']==2


def test_historical_four_key_model_and_checkpoint_cross_rejected(monkeypatch,tmp_path,backend_profile):
    ck=checkpoint();c=trained(monkeypatch,tmp_path/'new',backend_profile)
    from test_dh116_flexible_measured_common_teacher import setup
    from go2_pvcnn.control.dh116.tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116 import tabletop_flexible_measured_checkpoint as old
    config,base,*_=setup('pinch')
    historical=DeformablePPOTrainer(MeasuredCommonTeacher(config,base),c.trainer.config,rollout_steps=2,num_envs=1)
    with pytest.raises(ValueError):contract(historical)
    new_path=tmp_path/'new.pt';ck.save_checkpoint(new_path,c.trainer,contract(c.trainer))
    with pytest.raises(ValueError):old.load_checkpoint(new_path,historical)
    old_path=tmp_path/'old.pt';old.save_checkpoint(old_path,historical)
    with pytest.raises(ValueError):ck.load_checkpoint(old_path,c.trainer,contract(c.trainer))


@pytest.mark.parametrize('damage',['boolean_step','negative_second_moment'])
def test_reject_numeric_adam_state_before_restore(monkeypatch,tmp_path,backend_profile,damage):
    ck=checkpoint();c=trained(monkeypatch,tmp_path/'train',backend_profile);identity=contract(c.trainer)
    path=tmp_path/'good.pt';ck.save_checkpoint(path,c.trainer,identity);data=torch.load(path,weights_only=True)
    if damage=='boolean_step':data['optimizer']['state'][0]['step']=torch.tensor(True)
    else:data['optimizer']['state'][0]['exp_avg_sq'].fill_(-1.)
    bad=tmp_path/'bad.pt';torch.save(data,bad)
    before=deepcopy(c.trainer.optimizer.state_dict())
    with pytest.raises(ValueError):ck.load_checkpoint(bad,c.trainer,identity)
    equal_tree(before,c.trainer.optimizer.state_dict())


@pytest.mark.parametrize('boundary',['after_read','after_validation'])
@pytest.mark.parametrize('change',['delete','replace'])
def test_resume_origin_and_state_bind_single_immutable_file_snapshot(monkeypatch,tmp_path,backend_profile,boundary,change):
    ck=checkpoint();source=trained(monkeypatch,tmp_path/'source',backend_profile)
    path=tmp_path/'snapshot.pt';ck.save_checkpoint(path,source.trainer,contract(source.trainer))
    original_bytes=path.read_bytes();expected_sha=hashlib.sha256(original_bytes).hexdigest()
    expected_payload=torch.load(path,weights_only=True)
    target,*_=fixture(monkeypatch,tmp_path/'target',backend_profile,rollout_steps=2)
    t=target.trainer;before=training_snapshot(t);changed=[]
    def mutate_path():
        if changed:return
        changed.append(change)
        if change=='delete':path.unlink()
        else:path.write_bytes(b'changed path after checkpoint bytes were read')
    if boundary=='after_read':
        original_load=ck.torch.load
        def deserialize(*args,**kwargs):
            payload=original_load(*args,**kwargs)
            mutate_path()
            return payload
        monkeypatch.setattr(ck.torch,'load',deserialize)
    else:
        validate=ck._validate_payload
        def after_candidate(*args,**kwargs):
            rng=validate(*args,**kwargs)
            mutate_path()
            return rng
        monkeypatch.setattr(ck,'_validate_payload',after_candidate)
    try:
        loaded=ck.load_checkpoint(path,t,contract(t))
    except BaseException as error:
        print('checkpoint mutation repro:',boundary,change,type(error).__name__,
            'live_updates',before[2]['updates'],'->',t.counters['updates'],
            'live_Adam_states',len(before[1]['state']),'->',len(t.optimizer.state_dict()['state']),
            'completed',getattr(t,'_native_completed_update',None),
            'origin',getattr(t,'_native_training_origin',None),
            'rng_restored',torch.equal(torch.get_rng_state(),expected_payload['rng']['torch']))
        raise
    assert changed==[change]
    assert loaded['counters']['updates']==1 and t.policy_version==1
    equal_tree(source.trainer.model.state_dict(),t.model.state_dict())
    equal_tree(source.trainer.optimizer.state_dict(),t.optimizer.state_dict())
    assert t._native_training_origin==dict(kind='resume',checkpoint_sha256=expected_sha)
    assert t._native_completed_update==1 and t._native_accepted_minibatches==1
    assert t._native_training_failed is False
    equal_tree(torch.get_rng_state(),expected_payload['rng']['torch'])
    assert (not path.exists()) if change=='delete' else path.read_bytes()!=original_bytes


def test_failed_snapshot_read_preserves_all_training_state(monkeypatch,tmp_path,backend_profile):
    ck=checkpoint();target,*_=fixture(monkeypatch,tmp_path,backend_profile,rollout_steps=2)
    t=target.trainer
    t._native_training_origin=dict(kind='resume',checkpoint_sha256='a'*64)
    t._native_accepted_minibatches=7;t._native_completed_update=0;t._native_training_failed=True
    before=training_snapshot(t)
    with pytest.raises(FileNotFoundError):ck.load_checkpoint(tmp_path/'missing.pt',t,contract(t))
    equal_tree(before,training_snapshot(t))


def test_deserialize_uses_exactly_one_read_bytes_snapshot(monkeypatch,tmp_path,backend_profile):
    ck=checkpoint();source=trained(monkeypatch,tmp_path/'source',backend_profile)
    path=tmp_path/'once.pt';ck.save_checkpoint(path,source.trainer,contract(source.trainer))
    original_bytes=path.read_bytes();expected_sha=hashlib.sha256(original_bytes).hexdigest()
    target,*_=fixture(monkeypatch,tmp_path/'target',backend_profile,rollout_steps=2)
    real_read=Path.read_bytes;real_load=ck.torch.load;reads=[];loads=[]
    def read_once(self):
        raw=real_read(self)
        if self==path:
            reads.append(self)
            path.write_bytes(b'replacement immediately after read_bytes')
        return raw
    def load_snapshot(stream,*args,**kwargs):
        assert isinstance(stream,io.BytesIO), 'checkpoint must deserialize immutable bytes rather than reopen path'
        assert stream.getvalue()==original_bytes
        loads.append(stream)
        return real_load(stream,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',read_once);monkeypatch.setattr(ck.torch,'load',load_snapshot)
    ck.load_checkpoint(path,target.trainer,contract(target.trainer))
    assert reads==[path] and len(loads)==1
    assert target.trainer._native_training_origin['checkpoint_sha256']==expected_sha
    equal_tree(source.trainer.model.state_dict(),target.trainer.model.state_dict())



def audit_checkpoint():
    ck=checkpoint()
    assert callable(getattr(ck,'audit_checkpoint_bytes',None)), 'read-only immutable checkpoint bytes audit missing'
    return ck


def audit_case(monkeypatch,tmp_path,profile,*,markers='present'):
    ck=audit_checkpoint();c=trained(monkeypatch,tmp_path/'source',profile);t=c.trainer
    path=tmp_path/'audit.pt';identity=contract(t);ck.save_checkpoint(path,t,identity)
    raw=path.read_bytes();payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    c.collect()
    row=t.rollout._rows[-1]
    t.collect_step(deepcopy(row['actor_obs']),deepcopy(row['critic_obs']))
    names=('_native_accepted_minibatches','_native_completed_update',
        '_native_training_failed','_native_training_origin')
    if markers=='absent':
        for name in names:t.__dict__.pop(name,None)
    else:
        t._native_accepted_minibatches=7;t._native_completed_update=0;t._native_training_failed=True
        t._native_training_origin=dict(kind='resume',checkpoint_sha256='a'*64)
    assert t._pending is not None and len(t.rollout)==1 and t.optimizer.state
    return ck,t,identity,raw,payload,path


@pytest.mark.parametrize('markers',['present','absent'])
def test_audit_checkpoint_bytes_returns_owned_metadata_without_any_live_mutation(monkeypatch,tmp_path,backend_profile,markers):
    ck,t,identity,raw,payload,path=audit_case(monkeypatch,tmp_path,backend_profile,markers=markers)
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    expected=dict(version=ck.VERSION,identity=deepcopy(payload['identity']),counters=deepcopy(payload['counters']),
        policy_version=payload['policy_version'],accepted_minibatch_updates=payload['accepted_minibatch_updates'],
        physical_state_saved=False,restart_requires_verified_reset=True,
        checkpoint_sha256=hashlib.sha256(raw).hexdigest(),policy_weights_sha256=_digest(payload['policy']))
    expected_payload=deepcopy(payload);expected_identity=deepcopy(identity)
    before=training_snapshot(t);loads=[];validated=[]
    real_load=ck.torch.load;real_open=Path.open;validate=ck._validate_payload
    def no_live_operation(*args,**kwargs):raise AssertionError('audit must not clean, restore RNG or update')
    def no_checkpoint_read(*args,**kwargs):raise AssertionError('audit must use supplied immutable bytes')
    def source_open(self,*args,**kwargs):
        assert self!=path and self.suffix=='.py', 'only existing source-identity reads are allowed'
        return real_open(self,*args,**kwargs)
    def load_same_bytes(stream,*args,**kwargs):
        assert isinstance(stream,io.BytesIO) and stream.getvalue()==raw
        assert kwargs==dict(map_location='cpu',weights_only=True)
        loads.append(stream);return real_load(stream,*args,**kwargs)
    def capture_payload(data,*args,**kwargs):
        validated.append(data);return validate(data,*args,**kwargs)
    path.unlink()
    monkeypatch.setattr(Path,'read_bytes',no_checkpoint_read);monkeypatch.setattr(Path,'open',source_open)
    monkeypatch.setattr(ck.torch,'load',load_same_bytes);monkeypatch.setattr(ck,'_validate_payload',capture_payload)
    monkeypatch.setattr(ck,'_clean',no_live_operation);monkeypatch.setattr(t,'update',no_live_operation)
    monkeypatch.setattr(torch,'set_rng_state',no_live_operation);monkeypatch.setattr(random,'setstate',no_live_operation)
    monkeypatch.setattr(np.random,'set_state',no_live_operation)
    metadata=ck.audit_checkpoint_bytes(raw,t,identity)
    assert len(loads)==len(validated)==1
    equal_tree(metadata,expected);equal_tree(before,training_snapshot(t))
    extended=ck.audit_checkpoint_bytes(raw,t,identity,include_rng_digest=True)
    equal_tree(extended,dict(expected,rng_state_sha256=_digest(payload['rng'])))
    equal_tree(before,training_snapshot(t))
    def plain(value):
        if type(value) is dict:
            assert all(type(key) is str for key in value)
            for item in value.values():plain(item)
        elif type(value) in (list,tuple):
            for item in value:plain(item)
        else:assert type(value) in (str,int,float,bool,type(None)), 'metadata must not expose live objects/Tensors'
    plain(metadata)
    metadata['identity']['model']['grasp']='metadata-only'
    metadata['identity']['training_contract']['source_sha256']['synthetic_cpu_source']='0'*64
    metadata['counters']['updates']=999
    equal_tree(expected_payload,validated[0]);equal_tree(expected_identity,identity)
    equal_tree(before,training_snapshot(t))


@pytest.mark.parametrize('damage',_PAYLOAD_DAMAGE+('boolean_step','negative_second_moment','zero_updates',
    'physical_saved','no_verified_reset','rollout','missing_field','numpy_rng'))
def test_audit_checkpoint_bytes_rejects_complete_invalid_payload_variants_without_mutation(monkeypatch,tmp_path,backend_profile,damage):
    ck,t,identity,raw,data,_=audit_case(monkeypatch,tmp_path,backend_profile)
    damage_payload(data,damage);stream=io.BytesIO();torch.save(data,stream);raw=stream.getvalue()
    before=training_snapshot(t)
    with pytest.raises((ValueError,RuntimeError)):ck.audit_checkpoint_bytes(raw,t,identity)
    equal_tree(before,training_snapshot(t))


@pytest.mark.parametrize('raw',[b'fake text checkpoint',b''])
def test_audit_checkpoint_bytes_rejects_noncheckpoint_text_without_mutation(monkeypatch,tmp_path,backend_profile,raw):
    ck,t,identity,*_=audit_case(monkeypatch,tmp_path,backend_profile);before=training_snapshot(t)
    with pytest.raises(ValueError):ck.audit_checkpoint_bytes(raw,t,identity)
    equal_tree(before,training_snapshot(t))


@pytest.mark.parametrize('kind',['none','text','bytearray','memoryview','bytes_subclass'])
def test_audit_checkpoint_bytes_requires_exact_bytes_without_mutation(monkeypatch,tmp_path,backend_profile,kind):
    ck,t,identity,raw,*_=audit_case(monkeypatch,tmp_path,backend_profile)
    class SubBytes(bytes):pass
    value={'none':None,'text':'checkpoint.pt','bytearray':bytearray(raw),
        'memoryview':memoryview(raw),'bytes_subclass':SubBytes(raw)}[kind]
    before=training_snapshot(t)
    with pytest.raises((TypeError,ValueError)):ck.audit_checkpoint_bytes(value,t,identity)
    equal_tree(before,training_snapshot(t))


def test_audit_checkpoint_bytes_rejects_wrong_expected_identity_without_mutation(monkeypatch,tmp_path,backend_profile):
    ck,t,identity,raw,*_=audit_case(monkeypatch,tmp_path,backend_profile);identity['target_sha256']='9'*64
    before=training_snapshot(t)
    with pytest.raises(ValueError):ck.audit_checkpoint_bytes(raw,t,identity)
    equal_tree(before,training_snapshot(t))


def test_audit_checkpoint_bytes_requires_cpu_before_deserialization_without_gpu_execution(monkeypatch,tmp_path,backend_profile):
    ck,t,identity,raw,*_=audit_case(monkeypatch,tmp_path,backend_profile);t.device=torch.device('cuda:0')
    before=training_snapshot(t)
    def never(*args,**kwargs):raise AssertionError('non-CPU audit must fail before deserialization')
    monkeypatch.setattr(ck.torch,'load',never)
    with pytest.raises(ValueError,match='CPU trainer'):ck.audit_checkpoint_bytes(raw,t,identity)
    assert str(t.device)=='cuda:0';equal_tree(before,training_snapshot(t))



def origin_payload_bytes(payload,origin):
    data=deepcopy(payload);data['origin']=deepcopy(origin)
    stream=io.BytesIO();torch.save(data,stream);return stream.getvalue()


@pytest.mark.parametrize('kind',['fresh','resume'])
def test_audit_expected_origin_preserves_default_metadata_and_matches(monkeypatch,tmp_path,backend_profile,kind):
    ck,t,identity,raw,payload,_=audit_case(monkeypatch,tmp_path,backend_profile)
    origin=dict(kind='fresh') if kind=='fresh' else dict(kind='resume',checkpoint_sha256=hashlib.sha256(raw).hexdigest())
    if kind=='resume':raw=origin_payload_bytes(payload,origin)
    before=training_snapshot(t);expected_origin=deepcopy(origin);loads=[];real_load=ck.torch.load
    def one_snapshot(stream,*args,**kwargs):
        assert isinstance(stream,io.BytesIO) and stream.getvalue()==raw
        assert kwargs==dict(map_location='cpu',weights_only=True)
        loads.append(stream);return real_load(stream,*args,**kwargs)
    monkeypatch.setattr(ck.torch,'load',one_snapshot)
    default=ck.audit_checkpoint_bytes(raw,t,identity)
    assert len(loads)==1;equal_tree(before,training_snapshot(t))
    explicit_none=ck.audit_checkpoint_bytes(raw,t,identity,expected_origin=None)
    assert len(loads)==2;equal_tree(before,training_snapshot(t))
    matched=ck.audit_checkpoint_bytes(raw,t,identity,expected_origin=expected_origin)
    assert len(loads)==3;equal_tree(before,training_snapshot(t))
    assert set(default)=={'version','identity','counters','policy_version','accepted_minibatch_updates',
        'physical_state_saved','restart_requires_verified_reset','checkpoint_sha256','policy_weights_sha256'}
    equal_tree(default,explicit_none);equal_tree(default,matched);equal_tree(expected_origin,origin)
    with pytest.raises(TypeError):ck.audit_checkpoint_bytes(raw,t,identity,origin)
    assert len(loads)==3;equal_tree(before,training_snapshot(t))


@pytest.mark.parametrize('case',['resume_as_fresh','fresh_as_resume','other_resume_sha'])
def test_audit_expected_origin_rejects_valid_but_different_origin_after_full_validation(monkeypatch,tmp_path,backend_profile,case):
    ck,t,identity,raw,payload,_=audit_case(monkeypatch,tmp_path,backend_profile)
    resumed=dict(kind='resume',checkpoint_sha256=hashlib.sha256(raw).hexdigest())
    if case!='fresh_as_resume':raw=origin_payload_bytes(payload,resumed)
    expected=dict(kind='fresh') if case=='resume_as_fresh' else resumed if case=='fresh_as_resume' else dict(kind='resume',checkpoint_sha256='9'*64)
    expected_before=deepcopy(expected);before=training_snapshot(t);validated=[];loads=[]
    validate=ck._validate_payload;real_load=ck.torch.load
    def complete_validation(*args,**kwargs):
        result=validate(*args,**kwargs);validated.append(True);return result
    def one_snapshot(stream,*args,**kwargs):
        assert isinstance(stream,io.BytesIO) and stream.getvalue()==raw
        assert kwargs==dict(map_location='cpu',weights_only=True)
        loads.append(stream);return real_load(stream,*args,**kwargs)
    monkeypatch.setattr(ck,'_validate_payload',complete_validation);monkeypatch.setattr(ck.torch,'load',one_snapshot)
    with pytest.raises(ValueError,match='origin'):ck.audit_checkpoint_bytes(raw,t,identity,expected_origin=expected)
    assert len(validated)==len(loads)==1
    equal_tree(before,training_snapshot(t));equal_tree(expected_before,expected)


@pytest.mark.parametrize('expected',[
    pytest.param('fresh',id='text'),pytest.param({},id='empty'),
    pytest.param(dict(kind='fresh',checkpoint_sha256='1'*64),id='fresh_extra'),
    pytest.param(dict(kind='resume'),id='resume_missing_sha'),
    pytest.param(dict(kind='resume',checkpoint_sha256='invalid'),id='resume_invalid_sha'),
    pytest.param(dict(kind='resume',checkpoint_sha256='1'*64,extra=True),id='resume_extra'),
    pytest.param(['fresh'],id='list')])
def test_audit_expected_origin_rejects_malformed_expectation_without_state_changes(monkeypatch,tmp_path,backend_profile,expected):
    ck,t,identity,raw,*_=audit_case(monkeypatch,tmp_path,backend_profile)
    before=training_snapshot(t);expected_before=deepcopy(expected)
    def no_deserialize(*args,**kwargs):raise AssertionError('malformed expected origin must fail before deserialization')
    monkeypatch.setattr(ck.torch,'load',no_deserialize)
    with pytest.raises(ValueError):ck.audit_checkpoint_bytes(raw,t,identity,expected_origin=expected)
    equal_tree(before,training_snapshot(t));equal_tree(expected_before,expected)
