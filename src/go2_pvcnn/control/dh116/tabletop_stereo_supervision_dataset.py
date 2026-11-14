"""Versioned synchronized samples; schema fixtures never qualify BC success."""
import copy
import json
from pathlib import Path

import torch

from .tabletop_stereo_contract import make_contract,validate_contract,validate_identity,require_aligned,require_sha,file_sha256
from .tabletop_stereo_observation import StereoHistoryBatch
from .tabletop_stereo_teacher_collector import clone_tree,require_teacher

VERSION='dh116-stereo-supervision-v1'


def _sample(tree,index,n,*,keep_batch=False):
    if isinstance(tree,torch.Tensor):
        if tree.ndim and tree.shape[0]==n:return tree[index:index+1].clone() if keep_batch else tree[index].clone()
        return tree.clone()
    if isinstance(tree,dict):return {k:_sample(v,index,n,keep_batch=keep_batch) for k,v in tree.items()}
    if isinstance(tree,list) and len(tree)==n:return copy.deepcopy(tree[index])
    return copy.deepcopy(tree)


def rows_from_records(records):
    rows=[]
    for record in records:
        n=len(record['identities'])
        for i in range(n):
            row=dict(history=_sample(record['sensor_history'],i,n,keep_batch=True),
                identity=copy.deepcopy(record['identities'][i]),privileged_identity=copy.deepcopy(record['privileged_identities'][i]),
                execution=copy.deepcopy(record['execution'][i]),mode=record['mode'],schema_only=record['schema_only'],
                action_owner=record['action_owner'][i],teacher_qualification=copy.deepcopy(record['teacher_qualification']))
            for key in ('privileged','privileged_valid','extra_labels','label_validity','state_sources','actor_raw',
                        'teacher_mean','teacher_sample','teacher_std','student_nominal','issued_actions','applied_targets',
                        'reward','done','info'):
                row[key]=_sample(record.get(key),i,n)
            binding=record.get('force_qualification')
            row['force_qualification']=None if binding is None else {
                key:(_sample(value,i,n) if isinstance(value,torch.Tensor) else copy.deepcopy(value))
                for key,value in binding.items()}
            rows.append(row)
    return clone_tree(rows,cpu=True)


def identity_key(row):
    return tuple(row[k] for k in ('run_uuid','env_id','episode_uid','reset_id','policy_tick'))


def _row_action(value,name,required=True):
    if value is None and not required:return
    if (not isinstance(value,torch.Tensor) or value.shape!=(24,) or not value.is_floating_point()
            or not bool(torch.isfinite(value).all()) or bool((value.abs()>1).any())):
        raise ValueError('finite canonical24 '+name+' required')


def validate_rows(rows,*,contract,schema_only):
    if not isinstance(rows,list) or not rows:raise ValueError('nonempty synchronized dataset rows required')
    seen=set();previous={}
    for row in rows:
        identity=validate_identity(row['identity']);require_aligned(identity,row['privileged_identity'])
        require_sha(identity.get('physical_initial_condition_sha256'),'physical initial condition SHA256')
        if not isinstance(identity.get('source_episode_group'),str) or not identity['source_episode_group']:
            raise ValueError('source episode group required for independent holdout')
        if identity['calibration_sha256']!=contract['calibration_sha256']:raise ValueError('dataset calibration contract mismatch')
        if row.get('schema_only') is not schema_only:raise ValueError('schema/native evidence scope mismatch')
        if not schema_only and row.get('info',{}).get('evidence_scope')!='native_runtime_capture_unqualified':
            raise ValueError('formal dataset cannot admit CPU/non-native executed rows')
        key=identity_key(identity)
        if key in seen:raise ValueError('duplicate sample identity')
        seen.add(key)
        history=StereoHistoryBatch(**row['history']).validate()
        if len(history)!=1 or not bool(history.history_valid[0,-1]):raise ValueError('one current sensor history per row required')
        if bool((history.image_age[:,-1]>1/60).any()) or not bool(history.image_valid[:,-1].all()):raise ValueError('stale stereo sample')
        for name in ('teacher_mean','issued_actions'):_row_action(row.get(name),name)
        for name in ('teacher_sample','student_nominal'):_row_action(row.get(name),name,False)
        privileged,valid=row.get('privileged'),row.get('privileged_valid')
        if (not isinstance(privileged,torch.Tensor) or privileged.shape!=(371,) or not privileged.is_floating_point()
                or not bool(torch.isfinite(privileged).all()) or not isinstance(valid,torch.Tensor)
                or valid.shape!=(371,) or valid.dtype!=torch.bool):raise ValueError('371D labels and explicit per-dimension validity required')
        if row.get('action_owner') not in ('teacher','student'):raise ValueError('whole-action owner required')
        expected=row['teacher_mean'] if row['action_owner']=='teacher' else row.get('student_nominal')
        if expected is None or not torch.equal(row['issued_actions'],expected):raise ValueError('issued action differs from declared complete owner')
        execution=row.get('execution',{})
        if (execution.get('physics_tick_start')!=identity['physics_tick_end']
                or execution.get('physics_tick_end',-1)<=identity['physics_tick_end']
                or execution.get('simulation_timestamp_start')!=identity['simulation_timestamp']
                or execution.get('simulation_timestamp_end',-1)<=identity['simulation_timestamp']):
            raise ValueError('executed physics/time range must follow the preserved pre-action identity')
        env=(identity['run_uuid'],identity['env_id'])
        old=previous.get(env)
        if old is not None:
            old_id=old['identity'];same=identity['episode_uid']==old_id['episode_uid']
            if (identity['simulation_timestamp']<=old_id['simulation_timestamp'] or identity['frame_id']<=old_id['frame_id']
                    or identity['physics_tick_start']<old['execution']['physics_tick_end']):
                raise ValueError('sample time/frame/physics clocks must remain ordered')
            if same:
                if (bool(old['done']) or identity['reset_id']!=old_id['reset_id']
                        or identity['policy_tick']!=old_id['policy_tick']+1):raise ValueError('episode ticks must be ordered without crossing terminal/reset')
                if ('episode_policy_tick' in identity or 'episode_policy_tick' in old_id) and (
                        'episode_policy_tick' not in identity or 'episode_policy_tick' not in old_id
                        or identity['episode_policy_tick']!=old_id['episode_policy_tick']+1):
                    raise ValueError('episode-local policy ticks must increase exactly once')
                if identity.get('trial_episode_index')!=old_id.get('trial_episode_index'):
                    raise ValueError('trial episode index changed within the same episode UID')
                if not torch.equal(history.proprio[0,-1,38:],old['issued_actions']):raise ValueError('previous actually-issued action mismatch')
            elif (not bool(old['done']) or identity['reset_id']<=old_id['reset_id']
                    or int(history.history_valid.sum())!=1 or not bool((history.proprio[0,-1,38:]==0).all())):
                raise ValueError('reset boundary has contaminated sensor history or episode label')
        previous[env]=row
    return rows


def _success_pool(data):
    # A historical qualified Teacher cannot bless unrelated newly collected
    # episodes. Re-verify native proof and match the exact episode identities.
    require_teacher(data.get('qualification'),data['source_sha256'])
    from .tabletop_stereo_promotion import qualified_episode_identities
    qualified=qualified_episode_identities(data['qualification'])
    groups={}
    for row in data['rows']:
        identity=row['identity'];key=tuple(identity[k] for k in ('run_uuid','env_id','episode_uid','reset_id'))
        if key not in qualified:raise ValueError('collected episode is not bound to native success proof')
        groups.setdefault(key,[]).append(row)
    for episode in groups.values():
        if (episode[0]['identity'].get('episode_policy_tick',episode[0]['identity']['policy_tick'])!=0 or not bool(episode[-1]['done'])
                or any(bool(row['done']) for row in episode[:-1])):raise ValueError('complete first episode from reset through terminal required')
        for row in episode:
            info=row['info']
            if (not bool(info.get('first_episode',False))
                    or bool(info.get('recovered',True)) or bool(info.get('restored',True))):
                raise ValueError('successful BC requires first native episode without recovery/restore')
        if not bool(episode[-1]['info'].get('success',False)):raise ValueError('failed/incomplete episode cannot become a success BC pool')


def validate_dataset(data,*,allow_schema_only=False):
    if not isinstance(data,dict) or data.get('version')!=VERSION:raise ValueError('versioned stereo dataset required')
    validate_contract(data.get('contract'))
    if data['contract'].get('kind') not in ('bc','dagger'):raise ValueError('BC/DAgger dataset contract required')
    require_sha(data.get('source_sha256'),'Teacher checkpoint/source SHA256')
    if type(data.get('schema_only')) is not bool:raise ValueError('explicit schema-only flag required')
    if data['schema_only'] and not allow_schema_only:raise ValueError('schema-only CPU fixture cannot qualify native dataset')
    if (data['schema_only'] or data.get('purpose')=='dagger_visited_states') and data.get('successful_demonstrations') is not False:
        raise ValueError('schema fixtures and visited failure states cannot be successful demonstrations')
    if data.get('native_qualified') is not False:raise ValueError('dataset schema validation must not claim native qualification')
    validate_rows(data.get('rows'),contract=data['contract'],schema_only=data['schema_only'])
    if not data['schema_only']:
        if data.get('purpose')=='successful_bc':_success_pool(data)
        elif data.get('purpose')=='dagger_visited_states':require_teacher(data.get('qualification'),data['source_sha256'])
        else:raise ValueError('explicit production dataset purpose required')
    return data


def build_dataset(records,*,source_sha256,contract=None,schema_only=False,qualification=None,purpose='successful_bc'):
    rows=rows_from_records(list(records))
    if not rows:raise ValueError('nonempty executed records required')
    if contract is None:contract=make_contract('dagger' if purpose=='dagger_visited_states' else 'bc',{},
        calibration_sha256=rows[0]['identity']['calibration_sha256'])
    data=dict(version=VERSION,contract=copy.deepcopy(contract),source_sha256=source_sha256,schema_only=schema_only,
        native_qualified=False,validation_scope='schema_only_not_native_success',purpose=purpose,
        successful_demonstrations=False if schema_only or purpose!='successful_bc' else True,
        qualification=str(qualification) if isinstance(qualification,Path) else copy.deepcopy(qualification),rows=rows)
    return validate_dataset(data,allow_schema_only=schema_only)


def save_dataset(data,path):
    validate_dataset(data,allow_schema_only=data.get('schema_only') is True)
    path=Path(path)
    with path.open('xb') as stream:torch.save(clone_tree(data,cpu=True),stream)
    return dict(path=str(path),sha256=file_sha256(path),samples=len(data['rows']),native_qualified=False)


def load_dataset(path,*,allow_schema_only=False,expected_sha256=None,expected_contract=None):
    if expected_sha256 is not None and file_sha256(path)!=require_sha(expected_sha256):raise ValueError('dataset bytes SHA mismatch')
    data=torch.load(path,map_location='cpu',weights_only=True)
    validate_dataset(data,allow_schema_only=allow_schema_only)
    if expected_contract is not None:validate_contract(data['contract'],expected=expected_contract)
    return data


def _stack_trees(rows):
    if not rows:return {}
    keys=set.intersection(*(set(row) for row in rows))
    result={}
    for key in keys:
        values=[row[key] for row in rows]
        if all(isinstance(x,torch.Tensor) for x in values):result[key]=torch.stack(values)
        elif all(isinstance(x,dict) for x in values):result[key]=_stack_trees(values)
    return result


def batch_dataset(data,indices,*,device='cpu'):
    validate_dataset(data,allow_schema_only=data.get('schema_only') is True)
    indices=[int(i) for i in indices]
    if not indices or any(i<0 or i>=len(data['rows']) for i in indices):raise ValueError('nonempty valid dataset indices required')
    rows=[data['rows'][i] for i in indices]
    history=StereoHistoryBatch.cat([StereoHistoryBatch(**row['history']) for row in rows]).to(device)
    targets=_stack_trees([row['extra_labels'] for row in rows]);targets['privileged_state']=torch.stack([row['privileged'] for row in rows])
    validity=_stack_trees([row['label_validity'] for row in rows]);validity['privileged_state']=torch.stack([row['privileged_valid'] for row in rows])
    validity['state_sources']=_stack_trees([row['state_sources'] for row in rows])
    def move(tree):return {k:(v.to(device) if isinstance(v,torch.Tensor) else move(v)) for k,v in tree.items()}
    force=None
    bindings=[row.get('force_qualification') for row in rows]
    if all(isinstance(binding,dict) for binding in bindings):
        mask_keys=('source_valid','contact_valid','attributable')
        metadata=[{k:v for k,v in binding.items() if k not in mask_keys} for binding in bindings]
        if all(item==metadata[0] for item in metadata[1:]):
            force=copy.deepcopy(metadata[0])
            for key in mask_keys:
                if not all(isinstance(binding.get(key),torch.Tensor) for binding in bindings):force=None;break
                force[key]=torch.stack([binding[key] for binding in bindings]).to(device)
    return dict(history=history,teacher_actions=torch.stack([row['teacher_mean'] for row in rows]).detach().to(device),
        action_valid=torch.ones(len(rows),24,dtype=torch.bool,device=device),targets=move(targets),validity=move(validity),
        force_qualification=force,schema_only=data['schema_only'],native_qualified=False,
        _dataset=data,indices=indices)


def split_by_initial_condition(data,*,heldout_initial_condition_sha256=()):
    validate_dataset(data,allow_schema_only=data.get('schema_only') is True)
    heldout=set(heldout_initial_condition_sha256)
    known={row['identity']['physical_initial_condition_sha256'] for row in data['rows']}
    if heldout-known:raise ValueError('unknown physical initial condition holdout SHA')
    # Source groups connect every physical-state copy in an episode family;
    # expand transitively so no shared initial condition OR source leaks.
    test_hashes=set(heldout);test_sources=set()
    changed=True
    while changed:
        old=(len(test_hashes),len(test_sources))
        for row in data['rows']:
            ident=row['identity']
            if ident['physical_initial_condition_sha256'] in test_hashes or ident['source_episode_group'] in test_sources:
                test_hashes.add(ident['physical_initial_condition_sha256']);test_sources.add(ident['source_episode_group'])
        changed=old!=(len(test_hashes),len(test_sources))
    train=[];test=[]
    for i,row in enumerate(data['rows']):
        (test if row['identity']['physical_initial_condition_sha256'] in test_hashes else train).append(i)
    if not train:raise ValueError('holdout consumed every independent initial-condition/source group')
    return dict(train_indices=train,heldout_indices=test,split_unit='physical_initial_condition_and_source_episode_connected_groups')
