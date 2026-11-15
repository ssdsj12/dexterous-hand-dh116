"""Full CPU artifact audit. Synthetic measurements never qualify a native run."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from scripts import run_dh116_work_packages as scheduler
from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_contract as worker
from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_artifacts as artifacts
from test_native_teacher_worker_artifacts import cpu_native_source_gate, write_jsonl
from test_native_teacher_worker_contract import recipe, pin_source_closure
from test_native_teacher_checkpoint import equal_tree, damage_payload, training_snapshot

ROOT=Path(__file__).resolve().parents[1]


def test_recipe_source_inventory_requires_reviewed_checkpoint_and_collector_dependencies():
    required={"src/go2_pvcnn/control/dh116/"+name for name in (
        'tabletop_native_teacher_checkpoint.py','tabletop_native_teacher_training.py',
        'tabletop_native_teacher_collector.py','tabletop_native_teacher_session.py')}
    assert required<=worker.SOURCE_FILES,'reviewed C04/C05 dependency closure missing'


def test_target_contract_has_one_shared_builder():
    assert callable(getattr(worker,'worker_training_target_contract',None)), 'shared actual target contract missing'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def binding(path):return dict(path=str(path),sha256=sha(path.read_bytes()))


def actual_reset(tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as pre
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned
    original_path=pre.EVIDENCE/('native-attempts/flex.f8-pinch-fast-evidence-native/'
        'f0b5bc8cd80210db27052f4f96f435c7ce48a7ff23d53fd533a53fd3ed63502f/attempt-2ynpy5si/receipt.json')
    original=json.loads(pre.checked_file(original_path,pre.ORIGINAL_RECEIPT_SHA));output=Path(original['output'])
    reset_path,numeric_path=output/'mesh_reset_state.json',output/'mesh_reset_numeric_state.json'
    saved=decode_owned(json.loads(reset_path.read_bytes()))['readback']['completed_reset_receipt']
    numeric=decode_owned(json.loads(numeric_path.read_bytes()))
    manifest=dict(schema=pre.RESET_MANIFEST_SCHEMA,style='pinch',pose_preview_only=False,
        target_schema=pre.TARGET_SCHEMA_VERSION,target_material_ids=list(range(41)),
        source_refs=dict(original_receipt=binding(original_path),reset_readback=binding(reset_path),numeric_reset=binding(numeric_path)),
        reset_seed=saved['seed'],rod_state={k:numeric['samples'][1]['state'][k].tolist()
            for k in ('positions','velocities','quaternions')},
        mesh_bindings_sha256=pre.digest(encode_owned(saved['rod_reset']['mesh_bindings'])))
    path=tmp_path/'resolved.json';path.write_text(json.dumps(manifest));return path


def build_full_case(tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile,digest
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as pre
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    from go2_pvcnn.control.dh116.tabletop_training_reward import load_config
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_checkpoint as ck
    profile=BackendProfile.load()
    spec=recipe(tmp_path);root=tmp_path/'source-project';root.mkdir()
    names={*spec['source_files'],'scripts/python.sh','scripts/run_dh116_work_packages.py',
        'tests/test_native_teacher_worker_checkpoint.py',source.CONFIG}
    # Actual backend local inventory and retained Reset replay readers; only
    # source/assets are copied. Historical PT/data stay at their original path.
    review=json.loads(pre.checked_file(pre.R08_REPORT,pre.R08_SHA))
    names.update(p for p in review['replay_source_sha256'] if not Path(p).is_absolute())
    names.update(str(Path(row.path).relative_to(ROOT)) for row in profile.inventory
        if Path(row.path).is_relative_to(ROOT))
    for name in names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    entry=root/spec['entrypoint'];entry.parent.mkdir(parents=True,exist_ok=True)
    entry.write_text('# CPU-only source-context fixture, never an execution provider\n')
    spec['source_files']=sorted(set(spec['source_files'])|{source.CONFIG})
    package=dict(status='pending',executor='native_bounded',title='CPU full artifact audit fixture',
        dependencies=[],tests=['tests/test_native_teacher_worker_checkpoint.py'],
        command_spec='configs/dh116_execution/cpu-checkpoint-audit.json',requires_approval=True,
        resource_class=scheduler.NATIVE_SCOPE,evidence_scope='cpu_fixture',dependency_scopes={},retry_policy=scheduler.NATIVE_RETRY)
    pin_source_closure(root,spec,package['tests'])
    reward_path=ROOT/'configs/dh116_training/approach_wiring_smoke_v1.json';reward=load_config(reward_path)
    reset=actual_reset(tmp_path);manifest=json.loads(reset.read_bytes())
    config=dict(schema='dh116-native-teacher-training-cpu-v1',preflight=pre.candidate_config(),
        model_config=json.loads((ROOT/'configs/dh116_training/privileged_deformable_teacher_pinch_v1.json').read_bytes()),
        episode_cap_s=2.,rollout_steps=32)
    config['preflight'].update(reward_config=binding(reward_path),resolved_reset_manifest=binding(reset),
        budget=dict(max_sdk_steps=258,max_resets=1,max_solver_calls=256,max_episode_count=1,wall_timeout_s=180))
    config_path=Path(spec['teacher_contract']['training_config']['path']);config_path.write_text(json.dumps(config))
    spec['teacher_contract']['training_config']=binding(config_path)
    spec['argv'][3]=binding(config_path)['sha256'];spec['inputs']={str(config_path):spec['argv'][3]}
    backend=dict(profile_sha256=profile.digest,binary_sha256=profile.binary_sha256,
        inventory_sha256=profile.inventory_sha256,source_tree_sha256=profile.source_tree_sha256,patch_sha256=profile.patch_sha256)
    physical=json.loads(pre.checked_file(ROOT/source.CONFIG,source.CONFIG_SHA))
    physics=dict(physics_hz=physical['physics_hz'],control_hz=physical['control_hz'],
        solver_substeps=physical['rod']['solver_substeps'],physics_config_sha256=source.CONFIG_SHA,
        action_contract_sha256=digest(ACTION_CONTRACT),target_schema=manifest['target_schema'])
    source_contract={'worker_source_closure':spec['teacher_contract']['source_sha256']}
    semantic=worker.training_semantic_digest(config,backend_identity=backend,source_identity=source_contract,
        physics_identity=physics,project=root)
    resolved=pre.load_reset_manifest(config['preflight'],project=root)
    spec['teacher_contract'].update(semantic_sha256=semantic,backend_sha256=profile.binary_sha256,
        target_sha256=digest(worker.worker_training_target_contract(profile,resolved)),
        reset_sha256=binding(reset)['sha256'],reward_sha256=reward.config_sha256)
    spec_path=root/package['command_spec'];spec_path.parent.mkdir(parents=True,exist_ok=True);spec_path.write_text(json.dumps(spec))
    registry=root/scheduler.REGISTRY;registry.parent.mkdir(parents=True,exist_ok=True)
    registry.write_text(json.dumps(dict(version=2,packages={'cpu-a':package})))
    out=tmp_path/'artifacts';out.mkdir()
    ppo=PPOConfig(lr=reward.ppo['learning_rate'],gamma=reward.gamma,lam=reward.ppo['gae_lambda'],
        epochs=reward.ppo['epochs'],minibatches=reward.ppo['minibatches'],target_kl=reward.ppo['target_kl'],
        kl_backtrack_steps=reward.ppo['kl_backtrack_steps'])
    from test_native_teacher_checkpoint import fixture as physical_fixture
    from go2_pvcnn.control.dh116.tabletop_native_teacher_training import run_updates
    collector,*_=physical_fixture(monkeypatch,tmp_path/'physical',profile,
        rollout_steps=32,ppo_config=ppo,physics_budget_end=258)
    trainer=collector.trainer;journal=collector.journal
    # The CPU source observes the original physical rows, then emits its honest
    # fixture distance sample; the collector owns every other raw event.
    observe=collector.sources.observe
    def measured_distance(row):
        endpoint=observe(row)
        journal.emit('policy_pinch_distance',dict(cpu_fixture=True,sdk_step=endpoint.stamp.physics_tick,
            distance_m=endpoint.fields['distance'].raw))
        return endpoint
    collector.sources.observe=measured_distance
    contract=ck.build_training_contract(trainer,scope='cpu_fixture',reward_sha256=reward.config_sha256,
        target_sha256=spec['teacher_contract']['target_sha256'],reset_sha256=binding(reset)['sha256'],
        source_sha256=source_contract,config_sha256=semantic)
    training=run_updates(collector,additional_updates=1)
    assert collector.consumed_sdk_steps==256 and training['updates']==1
    journal.close();shutil.copyfile(journal.path,out/'training_events.jsonl')
    saved=ck.save_checkpoint(out/'teacher_clean.pt',trainer,contract)
    result=dict(schema=worker.RESULT_SCHEMA,semantic_sha256=semantic,training=training,
        clean_checkpoint=dict(path='teacher_clean.pt',sha256=saved['sha256']))
    (out/'training_result.json').write_text(json.dumps(result))
    (out/'shutdown_complete.json').write_text(json.dumps(dict(close_returned=True,process_exit_verified=False,pending_simulation_callback_error=None)))
    rows=[json.loads(line) for line in (out/'training_events.jsonl').read_bytes().splitlines()]
    for name,operation in [('physics.jsonl','physical'),('transitions.jsonl','transition_recorded'),('ownership.jsonl',None)]:
        write_jsonl(out/name,[dict(raw_sequence=r['sequence'],payload=r['payload']) for r in rows if r['operation']==operation])
    return spec,out,scheduler.Repository(root),package,trainer,result


@pytest.fixture(scope='module')
def physical_artifacts(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        yield build_full_case(tmp_path_factory.mktemp('full-cpu-artifacts'),patch)


@pytest.fixture
def full_case(tmp_path,physical_artifacts,cpu_native_source_gate):
    spec,original,repo,package,trainer,result=physical_artifacts
    out=tmp_path/'copy';shutil.copytree(original,out)
    return deepcopy(spec),out,repo,deepcopy(package),trainer,deepcopy(result)


def test_public_scheduler_audits_intact_real_cpu_checkpoint_without_state_or_rng_changes(full_case):
    spec,out,repo,package,trainer,result=full_case
    before=training_snapshot(trainer)
    evidence=scheduler.validate_native_artifacts(spec,out,repository=repo,package=package)
    equal_tree(before,training_snapshot(trainer))
    assert evidence['checkpoint_audit']['checkpoint_sha256']==result['clean_checkpoint']['sha256']
    assert evidence['checkpoint_audit']['counters']==dict(environment_steps=32,updates=1)
    assert evidence['physical_evidence']['physical_nodes']==3 and evidence['model_nodes']==41
    assert evidence['physical_evidence']['measured_sdk_steps']==256
    assert evidence['physical_to_model_snapshot_binding_verified'] is False
    assert evidence['actual_setup_source_verified'] is False and evidence['cap_source_verified'] is False
    assert evidence['native_ready'] is False and evidence['native_qualification'] is False
    assert 'checkpoint_bytes' not in evidence


@pytest.mark.parametrize('damage',['policy_shape','adam','adam_step','rng','numpy_rng','origin','resume_origin','zero_updates','source','config','accepted','physical_saved','no_verified_reset'])
def test_public_scheduler_rejects_safe_loadable_invalid_checkpoints(full_case,damage):
    spec,out,repo,package,trainer,result=full_case
    path=out/'teacher_clean.pt';payload=torch.load(path,map_location='cpu',weights_only=True)
    if damage=='resume_origin':payload['origin']=dict(kind='resume',checkpoint_sha256='e'*64)
    else:damage_payload(payload,damage)
    torch.save(payload,path)
    result['clean_checkpoint']['sha256']=sha(path.read_bytes());(out/'training_result.json').write_text(json.dumps(result))
    before=training_snapshot(trainer)
    with pytest.raises((ValueError,RuntimeError)):
        scheduler.validate_native_artifacts(spec,out,repository=repo,package=package)
    equal_tree(before,training_snapshot(trainer))


def test_public_scheduler_rejects_recorded_policy_hash_different_from_checkpoint(full_case):
    spec,out,repo,package,trainer,result=full_case
    rows=[json.loads(line) for line in (out/'training_events.jsonl').read_bytes().splitlines()]
    next(r for r in rows if r['operation']=='policy_epoch')['payload']['weights']='9'*64
    write_jsonl(out/'training_events.jsonl',rows)
    with pytest.raises(ValueError,match='policy|weights'):
        scheduler.validate_native_artifacts(spec,out,repository=repo,package=package)


@pytest.mark.parametrize('bad',[False,True])
def test_reference_is_unchanged_and_checkpoint_bytes_are_read_once(full_case,monkeypatch,bad):
    spec,out,repo,package,trainer,result=full_case
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_checkpoint as ck
    path=out/'teacher_clean.pt';raw=path.read_bytes()
    if bad:
        payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        damage_payload(payload,'policy_shape');stream=io.BytesIO();torch.save(payload,stream)
        raw=stream.getvalue();path.write_bytes(raw)
        result['clean_checkpoint']['sha256']=sha(raw);(out/'training_result.json').write_text(json.dumps(result))
    real_read=Path.read_bytes;real_load=torch.load;reference=artifacts._cpu_reference
    original_artifacts={name:(out/name).read_bytes() for name in worker.ARTIFACTS}
    reads=[];loads=[];snapshots=[]
    def read_once(self):
        value=real_read(self)
        if self.parent==out and self.name in worker.ARTIFACTS:
            reads.append(self);self.write_bytes(b'replaced after immutable artifact snapshot')
        return value
    def safe_load(stream,*args,**kwargs):
        assert isinstance(stream,io.BytesIO) and stream.getvalue()==raw
        assert kwargs==dict(map_location='cpu',weights_only=True)
        loads.append(True);return real_load(stream,*args,**kwargs)
    @contextmanager
    def observed_reference(*args):
        with reference(*args) as (expected,contract):
            before=training_snapshot(expected)
            try:yield expected,contract
            finally:
                equal_tree(before,training_snapshot(expected));snapshots.append(True)
    def forbidden(*args,**kwargs):raise AssertionError('audit must not collect/update/restore')
    monkeypatch.setattr(Path,'read_bytes',read_once);monkeypatch.setattr(ck.torch,'load',safe_load)
    monkeypatch.setattr(artifacts,'_cpu_reference',observed_reference)
    monkeypatch.setattr(ck,'load_checkpoint',forbidden)
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    monkeypatch.setattr(DeformablePPOTrainer,'collect_step',forbidden);monkeypatch.setattr(DeformablePPOTrainer,'update',forbidden)
    global_before=(torch.get_rng_state().clone(),random.getstate(),deepcopy(np.random.get_state()))
    if bad:
        with pytest.raises(ValueError):scheduler.validate_native_artifacts(spec,out,repository=repo,package=package)
    else:
        evidence=scheduler.validate_native_artifacts(spec,out,repository=repo,package=package)
        assert evidence['checkpoint_audit']['checkpoint_sha256']==sha(raw)
        assert evidence['artifact_sha256']=={name:sha(value) for name,value in original_artifacts.items()}
    equal_tree(global_before,(torch.get_rng_state(),random.getstate(),np.random.get_state()))
    assert len(reads)==len(worker.ARTIFACTS) and {p.name for p in reads}==worker.ARTIFACTS
    assert loads==[True] and snapshots==[True]


@pytest.mark.parametrize('damage',['text','truncated','counter','semantic_recipe','source_bytes','ppo_numeric','missing_ppo','budget_bool','cpu_scope','backend_profile_swap','target_source','target_boolean'])
def test_actual_reference_and_checkpoint_reject_cross_bindings(full_case,monkeypatch,damage):
    spec,out,repo,package,trainer,result=full_case
    restored=None
    if damage in ('text','truncated'):
        path=out/'teacher_clean.pt'
        path.write_bytes(b'{}\n' if damage=='text' else path.read_bytes()[:32])
        result['clean_checkpoint']['sha256']=sha(path.read_bytes())
    elif damage=='counter':result['training']['accepted_minibatch_updates']+=1
    elif damage in ('semantic_recipe','backend_profile_swap'):
        if damage=='semantic_recipe':
            spec['teacher_contract']['semantic_sha256']='9'*64;result['semantic_sha256']='9'*64
        else:
            from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
            spec['teacher_contract']['backend_sha256']=BackendProfile.load().digest
        path=repo.root/package['command_spec'];restored=(path,path.read_bytes());path.write_text(json.dumps(spec))
    elif damage=='source_bytes':
        path=repo.root/'src/go2_pvcnn/control/dh116/tabletop_native_teacher_checkpoint.py'
        restored=(path,path.read_bytes());path.write_bytes(restored[1]+b'\n# altered source\n')
    elif damage in ('ppo_numeric','missing_ppo','budget_bool'):
        path=Path(spec['teacher_contract']['training_config']['path']);restored=(path,path.read_bytes())
        config=json.loads(restored[1])
        if damage=='ppo_numeric':config['preflight']['ppo']['lr']=.1
        elif damage=='missing_ppo':config['preflight']['ppo']={}
        else:config['preflight']['budget']['max_resets']=True
        path.write_text(json.dumps(config))
        spec['teacher_contract']['training_config']=binding(path);spec['argv'][3]=binding(path)['sha256'];spec['inputs']={str(path):spec['argv'][3]}
        recipe_path=repo.root/package['command_spec'];recipe_raw=recipe_path.read_bytes();recipe_path.write_text(json.dumps(spec))
    elif damage in ('target_source','target_boolean'):
        config_path=Path(spec['teacher_contract']['training_config']['path']);config_raw=config_path.read_bytes()
        config=json.loads(config_raw)
        path=Path(config['preflight']['resolved_reset_manifest']['path']);restored=(path,path.read_bytes())
        manifest=json.loads(restored[1]);manifest['target_material_ids'][0 if damage=='target_source' else 1]=40 if damage=='target_source' else True
        path.write_text(json.dumps(manifest))
        config['preflight']['resolved_reset_manifest']=binding(path);config_path.write_text(json.dumps(config))
        spec['teacher_contract']['training_config']=binding(config_path);spec['argv'][3]=binding(config_path)['sha256']
        spec['inputs']={str(config_path):spec['argv'][3]}
        recipe_path=repo.root/package['command_spec'];recipe_raw=recipe_path.read_bytes();recipe_path.write_text(json.dumps(spec))
    else:monkeypatch.setattr(artifacts,'_required_native_source_scope',lambda:'owner_source_unqualified')
    (out/'training_result.json').write_text(json.dumps(result))
    try:
        with pytest.raises((ValueError,RuntimeError)):
            scheduler.validate_native_artifacts(spec,out,repository=repo,package=package)
    finally:
        if restored:restored[0].write_bytes(restored[1])
        if damage in ('ppo_numeric','missing_ppo','budget_bool'):recipe_path.write_bytes(recipe_raw)
        if damage in ('target_source','target_boolean'):config_path.write_bytes(config_raw);recipe_path.write_bytes(recipe_raw)


def test_json_exponent_overflow_is_nonfinite():
    with pytest.raises(ValueError,match='finite'):
        artifacts._json(b'{"overflow":1e999}')


def test_shared_target_rejects_boolean_material_ids(full_case,tmp_path):
    spec,out,repo,package,trainer,result=full_case
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as pre
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    config=json.loads(Path(spec['teacher_contract']['training_config']['path']).read_bytes())['preflight']
    data=json.loads(Path(config['resolved_reset_manifest']['path']).read_bytes())
    data['target_material_ids'][1]=True
    path=tmp_path/'boolean-target.json';path.write_text(json.dumps(data))
    config['resolved_reset_manifest']=binding(path)
    resolved=pre.load_reset_manifest(config,project=repo.root)
    with pytest.raises(ValueError,match='target'):
        worker.worker_training_target_contract(BackendProfile.load(),resolved)


@pytest.mark.parametrize('damage',['physical_empty','physical_duplicate','budget_overflow',
    'duplicate_both','delivered_empty','reset_epoch','sdk_gap','rod_tick','solver_clock','nominal_timestamp',
    'control_index','substep','request','material_empty','material_nonfinite',
    'budget_backtrack','budget_bool','budget_internal','history_hash','history_token','settlement_endpoint','context_endpoint'])
def test_public_scheduler_rejects_synchronously_corrupted_physical_evidence(full_case,damage):
    spec,out,repo,package,trainer,result=full_case
    rows=[json.loads(line) for line in (out/'training_events.jsonl').read_bytes().splitlines()]
    physical=[r for r in rows if r['operation']=='physical']
    delivered=[r for r in rows if r['operation']=='delivered_exchange']
    budgets=[r for r in rows if r['operation']=='budget_consumed']
    settlements=[r for r in rows if r['operation']=='settlement']
    if damage=='physical_empty':physical[0]['payload']={}
    elif damage=='physical_duplicate':physical[1]['payload']=deepcopy(physical[0]['payload'])
    elif damage=='budget_overflow':budgets[-1]['payload']=dict(sdk_steps=999,solver_calls=999,internal_solver_substeps=11988)
    elif damage=='duplicate_both':
        physical[1]['payload']=deepcopy(physical[0]['payload']);delivered[1]['payload']=deepcopy(delivered[0]['payload'])
    elif damage=='delivered_empty':delivered[0]['payload']={}
    elif damage.startswith('budget_'):
        if damage=='budget_backtrack':budgets[1]['payload']=dict(sdk_steps=1,solver_calls=1,internal_solver_substeps=12)
        elif damage=='budget_bool':budgets[0]['payload']['sdk_steps']=True
        else:budgets[-1]['payload']['internal_solver_substeps']-=1
    elif damage=='history_hash':settlements[0]['payload']['history']['physical_rows_sha256'][0]='9'*64
    elif damage=='history_token':settlements[0]['payload']['history']['policy']['collection_token']='foreign-token'
    elif damage=='settlement_endpoint':settlements[0]['payload']['history']['endpoint']['physics_step']+=1
    elif damage=='context_endpoint':settlements[0]['payload']['post_context']['metadata']['identities'][0]['reset_id']+=1
    else:
        raw=physical[8]['payload'];exchange=raw['exchange'];trace=exchange['action_readback']
        if damage=='reset_epoch':
            for phase in ('pre','post'):trace[phase]['identity'][0]['reset_id']+=1
            exchange['rod']['reset_id']+=1;raw['rod_boundary']['rod_snapshot']['reset_ids'][0]+=1
        elif damage=='sdk_gap':exchange['physics_step']+=1
        elif damage=='rod_tick':trace['post']['identity'][0]['rod_tick']+=12
        elif damage=='solver_clock':trace['post']['identity'][0]['solver_time_s']=trace['pre']['identity'][0]['solver_time_s']
        elif damage=='nominal_timestamp':trace['post']['identity'][0]['timestamp_s']+=1
        elif damage=='control_index':exchange['policy_step']+=1
        elif damage=='substep':exchange['substep']+=1
        elif damage=='request':trace['request']['action24']['__dh116_array__'][0][0]+=.001
        elif damage=='material_empty':raw['rod_boundary']['rod_snapshot']['node_positions']['__dh116_array__']=[[]]
        elif damage=='material_nonfinite':raw['rod_boundary']['rod_snapshot']['node_positions']['__dh116_array__'][0][0][0]={'__dh116_nonfinite__':'nan'}
        else:raise AssertionError(damage)
        delivered[8]['payload']=deepcopy(exchange)
    write_jsonl(out/'training_events.jsonl',rows)
    write_jsonl(out/'physics.jsonl',[dict(raw_sequence=r['sequence'],payload=r['payload']) for r in physical])
    with pytest.raises(ValueError,match='physical|budget|exchange'):
        scheduler.validate_native_artifacts(spec,out,repository=repo,package=package)


def synchronize_physical_evidence(out,rows):
    """Keep views and original C04 history hashes matched, as QUALITY probes do."""
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,_digest
    physical=[r for r in rows if r['operation']=='physical']
    settlements=[r['payload'] for r in rows if r['operation']=='settlement']
    for control,settled in enumerate(settlements):
        settled['history']['physical_rows_sha256']=[_digest(decode_owned(r['payload']))
            for r in physical[control*8:(control+1)*8]]
    write_jsonl(out/'training_events.jsonl',rows)
    write_jsonl(out/'physics.jsonl',[dict(raw_sequence=r['sequence'],payload=r['payload']) for r in physical])


@pytest.mark.parametrize('damage',['raw_identity_bool','snapshot_identity_bool','settlement_control_bool'])
def test_quality_integer_identity_probes_reject_at_public_api(full_case,damage):
    spec,out,repo,package,trainer,result=full_case
    checkpoint_sha=sha((out/'teacher_clean.pt').read_bytes())
    rows=[json.loads(line) for line in (out/'training_events.jsonl').read_bytes().splitlines()]
    physical=[r['payload'] for r in rows if r['operation']=='physical']
    delivered=[r['payload'] for r in rows if r['operation']=='delivered_exchange']
    if damage=='raw_identity_bool':
        for raw,exchange in zip(physical,delivered):
            raw['exchange']['rod'].update(env_id=False,reset_id=True)
            exchange['rod'].update(env_id=False,reset_id=True)
    elif damage=='snapshot_identity_bool':
        for raw in physical:raw['rod_boundary']['rod_snapshot'].update(env_ids=[False],reset_ids=[True])
    else:next(r['payload'] for r in rows if r['operation']=='settlement')['result']['control_step']=False
    synchronize_physical_evidence(out,rows)
    assert sha((out/'teacher_clean.pt').read_bytes())==checkpoint_sha
    with pytest.raises(ValueError,match='integer|identity|control'):
        scheduler.validate_native_artifacts(spec,out,repository=repo,package=package)


@pytest.mark.parametrize('where,key,value',[
    ('rod','env_id',False),('rod','env_id',0.0),('rod','reset_id',True),('rod','reset_id',1.0),
    ('rod','physics_step',3.0),('rod','rod_tick',12.0),('rod','clock_valid',1),('rod','clock_valid',1.0),
    ('boundary','rod_tick',12.0),('snapshot','physics_step',3.0),('snapshot','rod_tick',12.0),
    ('snapshot','env_ids',[False]),('snapshot','env_ids',[0.0]),
    ('snapshot','reset_ids',[True]),('snapshot','reset_ids',[1.0]),
    ('snapshot','env_ids',{'__dh116_array__':[0],'dtype':'int64'}),
    ('snapshot','reset_ids',{'__dh116_array__':[1],'dtype':'int64'}),
    ('exchange','policy_step',False),('exchange','policy_step',0.0),
    ('exchange','substep',False),('exchange','substep',0.0),('exchange','physics_step',3.0),
    ('result','control_step',0.0),('result','physical_samples',8.0),
    ('decision','policy_version',False),('decision','policy_version',0.0),
])
def test_integer_identities_do_not_accept_aliases_with_synchronized_hashes(full_case,where,key,value):
    spec,out,repo,package,trainer,result=full_case
    rows=[json.loads(line) for line in (out/'training_events.jsonl').read_bytes().splitlines()]
    physical=next(r['payload'] for r in rows if r['operation']=='physical')
    delivered=next(r for r in rows if r['operation']=='delivered_exchange')
    if where=='rod':physical['exchange']['rod'][key]=value
    elif where=='boundary':physical['rod_boundary'][key]=value
    elif where=='snapshot':physical['rod_boundary']['rod_snapshot'][key]=value
    elif where=='exchange':physical['exchange'][key]=value
    elif where=='result':next(r['payload'] for r in rows if r['operation']=='settlement')['result'][key]=value
    else:next(r['payload'] for r in rows if r['operation']=='decision')['policy'][key]=value
    delivered['payload']=deepcopy(physical['exchange'])
    synchronize_physical_evidence(out,rows)
    with pytest.raises(ValueError,match='integer|identity|control|policy|journal|boolean'):
        scheduler.validate_native_artifacts(spec,out,repository=repo,package=package)


def test_exact_integer_checks_preserve_legal_boolean_masks(full_case):
    spec,out,repo,package,trainer,result=full_case
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    rows=[json.loads(line) for line in (out/'training_events.jsonl').read_bytes().splitlines()]
    physical=decode_owned(next(r['payload'] for r in rows if r['operation']=='physical'))
    observation=decode_owned(next(r['payload'] for r in rows if r['operation']=='decision'))['observation']
    assert physical['rod_boundary']['rod_snapshot']['contact_valid'].dtype==np.bool_
    assert observation['common_valid'].dtype==np.bool_ and (~observation['common_valid']).any()
    assert observation['rod_mask'].dtype==np.bool_ and observation['rod_mask'].all()
    evidence=scheduler.validate_native_artifacts(spec,out,repository=repo,package=package)
    assert evidence['checkpoint_audit']['counters']==dict(environment_steps=32,updates=1)
    assert evidence['checkpoint_audit']['checkpoint_sha256']==result['clean_checkpoint']['sha256']
    assert evidence['native_qualification'] is False and evidence['native_ready'] is False
