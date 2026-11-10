"""Explicit, re-readable training artifact inventory; never native qualification.

Registration runs after producers close their raw files. Completeness is an
explicit request checked against locally recorded dependencies, not discovery
of unrecorded training. Unknown/CPU producers remain incomplete.
"""
from copy import deepcopy
import argparse
import json
import math
from pathlib import Path

from .tabletop_stereo_contract import digest,file_sha256,physical_initial_sha256,require_sha,write_json_new

VERSION='dh116-training-source-manifest-v1'
_FORBIDDEN={'privileged','teacher_state_dict','teacher_weights','teacher_policy','critic','critic_state_dict',
    'optimizer','policy','state_dict','weights','ground_truth','teacher_network'}


def validate_training_provenance(value,*,require_artifacts=False):
    """Only JSON metadata travels with an Actor; unavailable paths grant nothing."""
    if value is None:value={}
    if not isinstance(value,dict):raise ValueError('training provenance metadata dictionary required')
    def visit(item):
        if isinstance(item,dict):
            if any(not isinstance(key,str) or key in _FORBIDDEN for key in item):
                raise ValueError('training provenance must contain metadata, never Teacher/critic state')
            if 'path' in item or 'sha256' in item:
                if not isinstance(item.get('path'),str):raise ValueError('provenance artifact path/SHA metadata required')
                require_sha(item.get('sha256'),'provenance artifact SHA')
                path=Path(item['path'])
                if require_artifacts or path.exists():
                    if not path.is_file() or file_sha256(path)!=item['sha256']:
                        raise ValueError('training provenance artifact SHA mismatch')
            for child in item.values():visit(child)
        elif isinstance(item,list):
            if any(isinstance(child,(int,float,list)) and not isinstance(child,bool) for child in item):
                raise ValueError('provenance numeric arrays/tensors are not metadata')
            for child in item:visit(child)
        elif item is not None and type(item) not in (str,bool,int,float):
            raise ValueError('training provenance must contain JSON metadata, not tensors/networks')
        elif isinstance(item,float) and not math.isfinite(item):raise ValueError('finite provenance metadata required')
    visit(value)
    return deepcopy(value)


def _binding(value,base=None):
    declared=value if isinstance(value,dict) else None
    path=Path(value['path'] if declared is not None else value)
    if not path.is_absolute():path=(Path.cwd() if base is None else base)/path
    path=path.resolve()
    if not path.is_file():raise ValueError('training artifact is missing: '+str(path))
    sha=file_sha256(path)
    if declared is not None and sha!=require_sha(value.get('sha256'),'training artifact SHA'):
        raise ValueError('training artifact SHA mismatch: '+str(path))
    return dict(path=str(path),sha256=sha)


def _artifacts(value,prefix=''):
    result=[]
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:result.append((prefix,value))
        else:
            for key,child in value.items():result.extend(_artifacts(child,prefix+'.'+key if prefix else key))
    elif isinstance(value,list):
        for child in value:result.extend(_artifacts(child,prefix))
    return result


def _raw_inventory(binding):
    path=Path(binding['path']);before=file_sha256(path)
    try:rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except (OSError,ValueError) as error:raise ValueError('raw episode JSONL cannot be read') from error
    if before!=binding['sha256'] or file_sha256(path)!=before:raise ValueError('raw artifact changed during registration; close producer first')
    episodes=[];headers={};counts={};reasons=[]
    for number,row in enumerate(rows,1):
        if not isinstance(row,dict):raise ValueError('raw episode dictionary required')
        if row.get('kind') not in ('episode_start','sample'):raise ValueError('unknown raw episode record kind')
        key=tuple(row.get(name) for name in ('run_uuid','env_id','episode_uid','reset_id'))
        if (not isinstance(key[0],str) or not key[0] or type(key[1]) is not int or key[1]<0
                or not isinstance(key[2],str) or not key[2] or type(key[3]) is not int or key[3]<0):
            raise ValueError('raw episode/reset identity required')
        if row['kind']=='episode_start':
            if key in headers:raise ValueError('duplicate raw episode_start identity')
            actual=physical_initial_sha256(row.get('physical_initial_condition'))
            source=row.get('source_episode_group')
            if not isinstance(source,str) or not source:
                reasons.append('raw_source_episode_group_missing:'+key[2]);source=None
            if row.get('initial_measurement_valid') is not True or row.get('physical_initial_condition_source')!='actual_sdk_readback':
                reasons.append('raw_initial_measurement_not_actual_readback:'+key[2])
            if row.get('source_scope')!='native':reasons.append('raw_source_not_native:'+key[2])
            episode=dict(run_uuid=key[0],env_id=key[1],episode_uid=key[2],reset_id=key[3],
                physical_initial_condition_sha256=actual,source_episode_group=source,
                raw_artifact_sha256=binding['sha256'],header_line=number)
            episodes.append(episode);headers[key]=row;counts[key]=0
        else:
            if key not in headers:raise ValueError('raw sample has no preceding episode_start')
            counts[key]+=1
            if row.get('source_scope')!='native':reasons.append('raw_source_not_native:'+key[2])
    if not episodes:reasons.append('raw_episode_headers_missing:'+str(path))
    for key,count in counts.items():
        if not count:reasons.append('raw_episode_has_no_physical_samples:'+key[2])
    return episodes,reasons


def _model_provenance(binding):
    # A generic file can be inventoried, but cannot prove the ancestry of a
    # currently executable Student model.
    import torch
    try:payload=torch.load(binding['path'],map_location='cpu',weights_only=True)
    except Exception:return {},['model_format_has_no_auditable_actor_training_provenance']
    if not isinstance(payload,dict) or payload.get('version')!='dh116-stereo-rgb-deployment-actor-v1':
        return {},['model_format_has_no_auditable_actor_training_provenance']
    from .tabletop_stereo_checkpoint import load_deployment_actor
    actor=load_deployment_actor(binding['path'])
    provenance=validate_training_provenance(actor.deployment_contract.get('training_provenance'),require_artifacts=True)
    reasons=[] if provenance else ['actor_training_provenance_missing']
    return provenance,reasons


def _dataset_sources(binding):
    """Extract references from actual registered tensors, never submitted lists."""
    import torch
    try:value=torch.load(binding['path'],map_location='cpu',weights_only=True)
    except Exception:return [],[],[]
    if not isinstance(value,dict) or value.get('version')!='dh116-stereo-supervision-v1':return [],[],[]
    paths=[];identities=[];reasons=[]
    if value.get('schema_only') is not False:reasons.append('training_dataset_is_schema_only:'+binding['sha256'])
    for row in value.get('rows',[]):
        path=row.get('info',{}).get('native_evidence_file')
        if not isinstance(path,str) or not path:
            reasons.append('training_dataset_row_has_no_raw_episode_artifact');continue
        paths.append(path)
        identity=row.get('identity',{})
        identities.append((tuple(identity.get(key) for key in ('run_uuid','env_id','episode_uid','reset_id')),
            identity.get('physical_initial_condition_sha256'),identity.get('source_episode_group')))
    return paths,identities,reasons


def _unique(bindings):
    result={}
    for binding in bindings:
        old=result.get(binding['path'])
        if old is not None and old!=binding:raise ValueError('same training artifact path has conflicting SHA')
        result[binding['path']]=binding
    return [result[path] for path in sorted(result)]


def _build(model,raw,parent_bindings,training,requested,*,stack):
    provenance,reasons=_model_provenance(model)
    references=_artifacts(provenance)
    # Declared manifests are dependencies, even if a caller omits --parent.
    for role,reference in references:
        if 'manifest' in role:parent_bindings.append(_binding(reference))
        else:training.append(_binding(reference))
    parents=[];parent_models=set();parent_episodes=[]
    for binding in _unique(parent_bindings):
        parent=_load(Path(binding['path']),stack=stack)
        parents.append(binding);parent_models.add(parent['model_checkpoint_sha256'])
        raw.extend(parent['raw_episode_artifacts']);parent_episodes.extend(parent['episodes'])
        training.extend(parent['training_artifacts'])
        if not parent['complete_source_inventory']:reasons.append('parent_source_inventory_incomplete:'+binding['sha256'])
    for role,reference in references:
        if ('actor_package' in role or 'resume_checkpoint' in role) and 'teacher' not in role:
            if reference['sha256'] not in parent_models:reasons.append('input_model_training_manifest_missing:'+reference['sha256'])
    # Known producer paths may be recorded before files close. Hashing happens
    # here, after the caller finishes the producer, rather than in checkpoints.
    for path in provenance.get('raw_episode_source_paths',[]):raw.append(_binding(path))
    expected=[]
    for binding in _unique(training):
        paths,identities,blocked=_dataset_sources(binding)
        raw.extend(_binding(path) for path in paths);expected.extend(identities);reasons.extend(blocked)
    raw=_unique(raw);episodes=[]
    for binding in raw:
        found,blocked=_raw_inventory(binding);episodes.extend(found);reasons.extend(blocked)
    indexed={}
    for episode in episodes:
        key=tuple(episode[name] for name in ('run_uuid','env_id','episode_uid','reset_id'))
        if key in indexed and indexed[key]!=episode:raise ValueError('episode identity appears in conflicting raw artifacts')
        indexed[key]=episode
    for key,initial,group in expected:
        actual=indexed.get(key)
        if actual is None or actual['physical_initial_condition_sha256']!=initial or actual['source_episode_group']!=group:
            raise ValueError('registered training dataset identity differs from actual raw episode_start')
    if not episodes:reasons.append('raw_training_episode_inventory_empty')
    if not requested:reasons.append('complete_source_inventory_not_explicitly_requested')
    reasons=sorted(set(reasons))
    result=dict(version=VERSION,model_checkpoint_artifact=model,model_checkpoint_sha256=model['sha256'],
        complete_source_inventory_requested=requested,complete_source_inventory=requested and not reasons,
        source_scope='artifact_inventory_only',native_qualified=False,
        completeness_scope='explicitly_declared_recorded_dependency_closure;unrecorded_training_not_discoverable',
        raw_episode_artifacts=raw,training_artifacts=_unique(training),parents=parents,
        episodes=sorted(indexed.values(),key=lambda row:(row['run_uuid'],row['env_id'],row['reset_id'],row['episode_uid'])),
        physical_initial_condition_sha256=sorted({row['physical_initial_condition_sha256'] for row in episodes}),
        source_episode_groups=sorted({row['source_episode_group'] for row in episodes if row['source_episode_group'] is not None}),
        incomplete_reasons=reasons,source_sha256={Path(__file__).name:file_sha256(__file__)})
    result['manifest_sha256']=digest(result)
    return result


def _load(path,*,stack):
    path=path.resolve()
    if path in stack:raise ValueError('training manifest parent cycle')
    try:value=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as error:raise ValueError('training manifest unavailable') from error
    if not isinstance(value,dict) or value.get('version')!=VERSION:raise ValueError('versioned training source manifest required')
    if value.get('manifest_sha256')!=digest({key:item for key,item in value.items() if key!='manifest_sha256'}):
        raise ValueError('training source manifest SHA mismatch')
    if type(value.get('complete_source_inventory_requested')) is not bool:raise ValueError('explicit inventory completeness request required')
    model=_binding(value['model_checkpoint_artifact'],path.parent)
    raw=[_binding(item,path.parent) for item in value['raw_episode_artifacts']]
    parents=[_binding(item,path.parent) for item in value['parents']]
    training=[_binding(item,path.parent) for item in value['training_artifacts']]
    rebuilt=_build(model,raw,parents,training,value['complete_source_inventory_requested'],stack=stack|{path})
    if rebuilt!=value:raise ValueError('training inventory derived from actual raw headers/artifacts differs from manifest')
    return rebuilt


def register_training_lineage(output,model_checkpoint,raw_episode_artifacts,*,parent_manifests=(),
                              complete_source_inventory=False,training_artifacts=()):
    if type(complete_source_inventory) is not bool:raise ValueError('explicit boolean completeness request required')
    path=Path(output)
    if path.exists():raise FileExistsError(path)
    model=_binding(model_checkpoint)
    value=_build(model,[_binding(item) for item in raw_episode_artifacts],
        [_binding(item) for item in parent_manifests],[_binding(item) for item in training_artifacts],
        complete_source_inventory,stack={path.resolve()})
    write_json_new(path,value)
    return dict(path=str(path.resolve()),sha256=file_sha256(path),model_checkpoint_sha256=model['sha256'],
        complete_source_inventory=value['complete_source_inventory'],native_qualified=False)


def load_training_lineage(path,expected_model_checkpoint_sha256=None,require_complete=False):
    result=_load(Path(path),stack=set())
    if expected_model_checkpoint_sha256 is not None and result['model_checkpoint_sha256']!=require_sha(expected_model_checkpoint_sha256):
        raise ValueError('training manifest model checkpoint SHA mismatch')
    if require_complete and not result['complete_source_inventory']:
        raise ValueError('complete training source inventory unavailable: '+';'.join(result['incomplete_reasons']))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-checkpoint',type=Path,required=True)
    parser.add_argument('--raw-episode-artifact',type=Path,action='append',default=[])
    parser.add_argument('--parent-manifest',type=Path,action='append',default=[])
    parser.add_argument('--training-artifact',type=Path,action='append',default=[])
    parser.add_argument('--declare-source-inventory-complete',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=register_training_lineage(args.output,args.model_checkpoint,args.raw_episode_artifact,
        parent_manifests=args.parent_manifest,training_artifacts=args.training_artifact,
        complete_source_inventory=args.declare_source_inventory_complete)
    print(json.dumps(result,sort_keys=True));return 0
