"""Offline merge of explicitly bound native base02 and Tier1 success pools.

No native trial is run or reused as a new trial. Parent first-terminal proofs
remain attached to their own reports. This producer does not impersonate one
Teacher collection, create heldout sources, fit an Actor or change main gates.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from .tabletop_vision_supervision_dataset import (
    validate_dataset, build_dataset, join_sequence_context, _canonical_visual_contract)
from .tabletop_vision_sequence_dataset import SequenceDataset
from .tabletop_vision_stage6_workflow import require_native_shadow_parity, verify_command_windows
from .tabletop_vision_command_shadow import controller_source_sha256, action_contract_sha256


VERSION = 'dh116-native-success-pool-merge-v1'
PINCH = 'pinch_thumb_index'
ROW_TREES = ('samples','controller_start','physics_context','teacher_targets','events')
RAW_ROWS = ('native_backend_targets','command_cache','raw_measured_packets','post_physics_feedback')
STATIC = ('version','controller_source_sha256','action_contract_sha256','controller_contract',
          'event_sources','command_cache_source')
CONTEXT_KEYS = set(ROW_TREES + RAW_ROWS + STATIC + ('dataset_sha256','teacher_shadow_parity'))


def file_identity(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return dict(path=str(path),sha256=digest.hexdigest())


def _checked(identity):
    if (not isinstance(identity,dict) or not isinstance(identity.get('path'),str)
            or not isinstance(identity.get('sha256'),str)):
        raise ValueError('explicit original path/SHA256 required')
    actual=file_identity(identity['path'])
    if actual['sha256'] != identity['sha256']: raise ValueError('immutable input SHA mismatch: '+identity['path'])
    return actual


def _json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2,allow_nan=False)
    return file_identity(path)


def _key(row): return tuple(row[k] for k in ('run_uuid','env_id','episode_uid','tick'))


def _parity(context, report):
    checked=int((context['physics_context']['live'] & context['physics_context']['source_valid']).sum())
    for value in (context.get('teacher_shadow_parity'),report['successful_sequence_context'].get('teacher_shadow_parity')):
        require_native_shadow_parity(value)
        if (value.get('max_float64_error') != 0. or value.get('max_sdk_cast_float32_error') != 0.
                or type(value.get('checked_physics_substeps')) is not int or value['checked_physics_substeps']!=checked or checked<1):
            raise ValueError('exact float64/cast32 parent parity proof required')
        for name in ('backend','command_cache'):
            p=value['native_sdk_readback'][name]
            if p.get('available') is not True or p.get('missing_hand_substeps') != 0 or p.get('max_error') != 0.:
                raise ValueError('actual separate native backend/cache parent parity required')


def _tier1(report,data,identities):
    from .tabletop_vision_teacher_perturbation import VERSION as TIER1, validate_teacher_perturbation_manifest
    proof=report.get('teacher_perturbation')
    if (not isinstance(proof,dict) or proof.get('version') != TIER1 or proof.get('tier') != 1
            or proof.get('native_runtime_verified') is not True or proof.get('first_packet_verified') is not True
            or proof.get('state_query_oracle') is not False):
        raise ValueError('actual Tier1 native reset/first-packet proof required')
    manifest_id=_checked(proof.get('input_manifest')); identities.append(manifest_id)
    evidence=proof.get('evidence',{})
    required=('manifest','reset_readback','warmup_readback','camera_geometry','runtime_camera')
    saved={k:_checked(evidence.get(k)) for k in required}; identities.extend(saved.values())
    if saved['manifest']['sha256'] != manifest_id['sha256']: raise ValueError('Tier1 frozen/input manifest mismatch')
    manifest=validate_teacher_perturbation_manifest(_json(saved['manifest']['path']))
    meta=manifest['metadata'];visual=data['visual_contract']
    if (meta['teacher_reference']['sha256'] != data['teacher_source_sha256']
            or meta['camera_sha256'] != visual['camera_config_sha256']
            or saved['runtime_camera']['sha256'] != visual['camera_config_sha256']
            or [meta['initializations'][s]['sha256'] for s in ('pinch_thumb_index','tripod_thumb_index_middle')]
                != [visual['primary_reset_sha256'],visual['secondary_reset_sha256']]):
        raise ValueError('Tier1 original reference/reset/camera contract mismatch')
    from .tabletop_evaluation_trials import _digest,state_sha256
    from .tabletop_vision_evaluation_contract import evaluation_physics_sha256
    if (meta['physics_sha256'] != evaluation_physics_sha256(visual['privileged_critic_contract'])
            or meta['source_privileged_critic_contract_sha256'] != _digest(visual['privileged_critic_contract'])):
        raise ValueError('Tier1 original physics/asset contract mismatch')
    for identity in (meta['source_checkpoint'],meta['teacher_reference'],*meta['initializations'].values()):
        identities.append(_checked(identity))
    implementations=proof.get('implementation')
    expected={'tabletop_vision_teacher_perturbation','tabletop_vision_evaluation_reset','tabletop_vision_teacher_collection'}
    if not isinstance(implementations,dict) or set(implementations)!=expected:
        raise ValueError('actual Tier1 collection/reset/generator code SHA identities required')
    for identity in implementations.values(): identities.append(_checked(identity))
    before,after,geometry=(_json(saved[k]['path']) for k in ('reset_readback','warmup_readback','camera_geometry'))
    if (before.get('native_runtime_verified') is not True or before.get('readback') != after.get('readback')
            or before.get('clock') != after.get('clock') or geometry.get('passed') is not True):
        raise ValueError('Tier1 original reset/camera warmup proof is not verified')
    states=before.get('readback',{}).get('actual_states',[]); trials={t['trial_id']:t for t in manifest['trials']}
    actual={s['trial_id']:s for s in states}
    if len(actual)!=len(manifest['trials']) or len(states)!=len(actual): raise ValueError('complete unique Tier1 reset readback required')
    cases={}
    for old,ep in data['episodes'].items():
        trial=trials.get(ep.get('teacher_perturbation_trial_id')); measured=actual.get(ep.get('teacher_perturbation_trial_id'))
        if not trial or not measured or trial['grasp_style'] != PINCH: raise ValueError('actual successful pinch Tier1 case identity required')
        expected=dict(teacher_perturbation_tier=1,teacher_perturbation_manifest_sha256=manifest_id['sha256'],
            teacher_reset_authored_state_sha256=trial['state_sha256'],
            teacher_reset_readback_state_sha256=measured['readback_state_sha256'],
            teacher_family_lineage_id=trial['family_lineage_id'],source_group_id=trial['source_group_id'])
        if (measured['authored_state_sha256'] != trial['state_sha256']
                or any(ep.get(k)!=v for k,v in expected.items())):
            raise ValueError('Tier1 successful episode/reset parent provenance mismatch')
        for row in data['sample_metadata']:
            if str(row['episode_id'])==old and any(row.get(k)!=v for k,v in expected.items()):
                raise ValueError('Tier1 sample and episode reset provenance mismatch')
        if trial['state_sha256']==state_sha256(manifest['base_templates'][PINCH]):
            raise ValueError('duplicate unchanged base case cannot be a new Tier1 source')
        cases[old]=(trial['source_group_id'],trial['state_sha256'])
    return cases


def load_native_parent(value):
    """Verify one original report and its exact success dataset/context files.

    These are supplied collector proofs, not an independent re-run of physics.
    Caller-provided SHA mismatches and incomplete/unqualified pools hard fail.
    """
    if not isinstance(value,dict) or value.get('role') not in ('base02','tier1'):
        raise ValueError('explicit base02/tier1 parent role required')
    bound={k:_checked(value.get(k)) for k in ('report','dataset','context')};identities=list(bound.values())
    report=_json(bound['report']['path'])
    if (report.get('status')!='completed' or report.get('mode')!='teacher_reference_mpc'
            or report.get('mpc_profile')!='pressure_geometry_constrained'
            or report.get('native_runtime_verified') is not True or report.get('ppo_updates')!=0
            or report.get('teacher_success_is_student_success') is not False
            or report.get('dagger_state_query_oracle') is not False or report.get('synthetic_cpu_fixture')
            or tuple(report.get(k) for k in ('actor_observation_size','critic_observation_size','action_size'))!=(3844,371,24)):
        raise ValueError('genuine original native Teacher first-success collection proof required')
    for name,key in (('successful_dataset','dataset'),('successful_sequence_context','context')):
        if _checked(report.get(name)) != bound[key]: raise ValueError('explicit parent files differ from native report')
    data=torch.load(bound['dataset']['path'],map_location='cpu',weights_only=False)
    context=torch.load(bound['context']['path'],map_location='cpu',weights_only=False)
    validate_dataset(data)
    _canonical_visual_contract(data['visual_contract'])
    assets=data['visual_contract'].get('privileged_critic_contract',{}).get('source_sha256')
    if (not isinstance(assets,dict) or set(assets)!={'left','right'}
            or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in assets.values())):
        raise ValueError('original left/right asset SHA bindings required')
    if data.get('native_pool_merge') is not None: raise ValueError('merged pool is not a single original native parent')
    if (report.get('source_sha256') != data['teacher_source_sha256']
            or report['successful_dataset'].get('samples')!=len(data['observations'])
            or report.get('successful_episodes')!=len(data['episodes'])):
        raise ValueError('actual parent sample/count/source SHA mismatch')
    for key in ('native_successful_episodes','observed_terminal_successes'):
        if type(report.get(key)) is not int or report[key]<len(data['episodes']):
            raise ValueError('actual parent native terminal counts contradict selected successful episodes')
    run_proof=report.get('sequence_context_collection')
    if (not isinstance(run_proof,dict) or run_proof.get('enabled') is not True
            or run_proof.get('successful_sidecar_available') is not True
            or not isinstance(run_proof.get('run_uuid'),str) or not run_proof['run_uuid']
            or any(row.get('run_uuid')!=run_proof['run_uuid'] for row in data['sample_metadata'])):
        raise ValueError('parent native run UUID/context collection proof mismatch or missing')
    if value['role']=='base02' and ('teacher_perturbation' in report or len(data['episodes'])<4):
        raise ValueError('base02 requires original four-or-more successful pinch first episodes')
    if len(data['episodes'])<1: raise ValueError('at least one actual Tier1 successful pinch episode required')
    for key,ep in data['episodes'].items():
        proof=report.get('episodes',{}).get(key)
        if (proof!=ep or ep.get('first_episode') is not True or ep.get('native_physics') is not True
                or ep.get('success') is not True or ep.get('recovered') is not False or ep.get('grasp_style')!=PINCH):
            raise ValueError('parent pool includes unqualified/non-pinch/failed/unfinished episode')
        if str(Path(ep['proof_path']).resolve())!=bound['report']['path']:
            raise ValueError('episode proof must point to its actual original native report')
        audit=ep.get('native_contact_style_audit')
        if (not isinstance(audit,dict) or audit.get('style')!=PINCH
                or audit!=report.get('contact_style_audit',{}).get(str(ep['env_id']))):
            raise ValueError('independently recorded matching native pinch contact-style audit required')
        terminal=ep.get('terminal')
        if (not isinstance(terminal,dict) or terminal.get('success') is not True
                or terminal.get('failure')!=0 or terminal.get('recovered') is not False):
            raise ValueError('original successful first terminal without recovery required')
    if (context.get('controller_source_sha256') != controller_source_sha256()
            or context.get('action_contract_sha256') != action_contract_sha256()
            or context.get('controller_contract',{}).get('controller_source_sha256') != context['controller_source_sha256']
            or context.get('controller_contract',{}).get('action_contract_sha256') != context['action_contract_sha256']):
        raise ValueError('original current controller/action source SHA must match; no alias or rebinding allowed')
    if report['successful_sequence_context'].get('controller_contract')!=context['controller_contract']:
        raise ValueError('original report and context controller source contracts differ; no SHA alias')
    if any(v.dtype!=torch.float64 for v in context['teacher_targets'].values()):
        raise ValueError('original float64 Teacher target trajectories required; casts are not original evidence')
    if set(context)!=CONTEXT_KEYS: raise ValueError('unsupported context fields; refuse rather than omit original records')
    joined=join_sequence_context(data,context,dataset_sha256=bound['dataset']['sha256'],
        controller_source_sha256=context['controller_source_sha256'],action_contract_sha256=context['action_contract_sha256'])
    if not bool(joined['row_valid'].all()) or len(context['samples']['tick'])!=len(data['observations']):
        raise ValueError('complete exact context/sample identity join required')
    _parity(context,report)
    for name in RAW_ROWS: _raw_rows(dict(data=data,context=context),name,copy_rows=False)
    groups={}
    for old,ep in data['episodes'].items():
        rows=[r for r in data['sample_metadata'] if str(r['episode_id'])==old]
        ids={(r['run_uuid'],r['env_id'],r['episode_uid']) for r in rows}
        if len(ids)!=1: raise ValueError('one original execution UID per complete first episode required')
        group=ep.get('source_group_id',ep['source_trajectory_id'])
        if group != ep['source_trajectory_id'] or any(r.get('source_group_id',group)!=group for r in rows):
            raise ValueError('original parent family grouping cannot become a derived independent source')
        groups[old]=group
    cases=_tier1(report,data,identities) if value['role']=='tier1' else {}
    sequence=SequenceDataset(data,context,dataset_sha256=bound['dataset']['sha256'],sampling_mode='event')
    parity=verify_command_windows(sequence,context['controller_contract'])
    for item in identities: _checked(item)
    return dict(role=value['role'],bound=bound,identities=identities,report=report,data=data,context=context,
        joined=joined,groups=groups,cases=cases,accumulated_window_parity=parity)


def _row_tree(parents,name):
    def combine(values):
        first=values[0]
        if isinstance(first,dict):
            if any(not isinstance(v,dict) or set(v)!=set(first) for v in values): raise ValueError('context row-tree field layout mismatch')
            return {k:combine([v[k] for v in values]) for k in first}
        selected=[]
        for p,value in zip(parents,values):
            indices=p['joined']['context_indices']
            if isinstance(value,torch.Tensor):
                if (not isinstance(first,torch.Tensor) or value.dtype!=first.dtype or value.shape[1:]!=first.shape[1:]
                        or value.shape[0]!=len(p['context']['samples']['tick'])):
                    raise ValueError('context tensor dtype/shape mismatch; casts forbidden')
                selected.append(value.index_select(0,indices))
            elif isinstance(value,(list,tuple)) and len(value)==len(p['context']['samples']['tick']):
                selected.extend(deepcopy(value[int(i)]) for i in indices)
            else: raise ValueError('unsupported original context row leaf')
        return torch.cat(selected) if isinstance(first,torch.Tensor) else (tuple(selected) if isinstance(first,tuple) else selected)
    return combine([p['context'][name] for p in parents])


def _raw_rows(parent,name,*,copy_rows=True):
    records=parent['context'][name]
    if not isinstance(records,list): raise ValueError('actual raw context records must remain identity-keyed lists')
    lookup={}
    for row in records:
        key=tuple(row['sample_key'])
        if key in lookup: raise ValueError('duplicate raw SDK/measured sample identity')
        if not isinstance(row.get('physics_substeps'),list) or len(row['physics_substeps'])!=8:
            raise ValueError('raw native feedback/backend records must preserve all eight substep slots')
        lookup[key]=row
    keys=[_key(row) for row in parent['data']['sample_metadata']]
    if set(lookup)!=set(keys): raise ValueError('missing/extra raw SDK/measured sample identities')
    return [deepcopy(lookup[key]) if copy_rows else lookup[key] for key in keys]


def merge_native_success_pools(base, tier1, output):
    """Create a new standalone pool/merge manifest from explicit file identities.

    ``base`` and every ``tier1`` entry have role, report, dataset and context;
    each file is {path,sha256}. No report path is guessed from an output name.
    """
    if not isinstance(tier1,(list,tuple)) or not tier1: raise ValueError('actual Tier1 successful parent required; merge gate closed')
    if not isinstance(base,dict) or base.get('role')!='base02' or any(v.get('role')!='tier1' for v in tier1):
        raise ValueError('one original base02 plus explicitly selected Tier1 parents required')
    target=Path(output).absolute()
    if target.exists() or target.is_symlink(): raise FileExistsError(target)
    implementation={name:file_identity(Path(__file__).with_name(name+'.py')) for name in (
        'tabletop_vision_sequence_pool_merge','tabletop_vision_supervision_dataset','tabletop_vision_sequence_dataset',
        'tabletop_vision_stage6_workflow','tabletop_vision_command_shadow','tabletop_command_kernels',
        'tabletop_free_wrist_grasp','tabletop_tensor_world','tabletop_vision_teacher_perturbation')}
    parents=[load_native_parent(v) for v in [base,*tier1]]
    reference=parents[0]; source_sha=reference['data']['teacher_source_sha256']
    for p in parents[1:]:
        if (p['data']['teacher_source_sha256']!=source_sha
                or _canonical_visual_contract(p['data']['visual_contract']) != _canonical_visual_contract(reference['data']['visual_contract'])):
            raise ValueError('same original camera/assets/observation/action and Teacher source bytes required')
        for key in STATIC:
            if p['context'][key]!=reference['context'][key]: raise ValueError('same controller/event provenance required: '+key)
    identities,episode_uids,keys,cases=set(),set(),set(),set()
    for item in implementation.values(): _checked(item)
    for p in parents:
        rid=p['bound']['report']['sha256']
        if rid in identities: raise ValueError('duplicate original parent report')
        identities.add(rid)
        for old in p['data']['episodes']:
            row=next(r for r in p['data']['sample_metadata'] if str(r['episode_id'])==old)
            uid=(row['run_uuid'],row['env_id'],row['episode_uid'])
            if uid in episode_uids: raise ValueError('duplicate execution UID across parent pools')
            episode_uids.add(uid)
            if old in p['cases']:
                case=p['cases'][old]
                if case in cases: raise ValueError('duplicate actual Tier1 case in a different run; not a new source')
                cases.add(case)
        for row in p['data']['sample_metadata']:
            key=_key(row)
            if key in keys: raise ValueError('duplicate original sample identity')
            keys.add(key)
        for name in RAW_ROWS: _raw_rows(p,name,copy_rows=False)
    rows,episodes,mapping=[],{},[]
    for parent_index,p in enumerate(parents):
        local={}
        for old in sorted(p['data']['episodes'],key=int):
            new=len(episodes);local[old]=new;ep=deepcopy(p['data']['episodes'][old])
            if 'origin_episode_id' in ep: raise ValueError('native parent cannot already be a derived merge episode')
            ep.update(origin_episode_id=int(old),origin_parent_report=deepcopy(p['bound']['report']),
                origin_parent_dataset=deepcopy(p['bound']['dataset']),source_group_id=p['groups'][old])
            episodes[str(new)]=ep
            mapping.append(dict(parent_index=parent_index,parent_report=p['bound']['report'],origin_episode_id=int(old),episode_id=new))
        for original in p['data']['sample_metadata']:
            row=deepcopy(original);old=str(row['episode_id'])
            if 'origin_episode_id' in row: raise ValueError('native parent cannot contain already merged sample IDs')
            row.update(origin_episode_id=int(old),origin_parent_report_sha256=p['bound']['report']['sha256'],
                origin_parent_dataset_sha256=p['bound']['dataset']['sha256'],episode_id=local[old],source_group_id=p['groups'][old])
            rows.append(row)
    for name in ('observations','actions','reference_actions'):
        if any(p['data'][name].dtype!=reference['data'][name].dtype for p in parents):
            raise ValueError('original dataset dtypes must match; casts forbidden')
    data=build_dataset(*(torch.cat([p['data'][key] for p in parents]) for key in ('observations','actions','reference_actions')),
        rows,episodes,visual_contract=reference['data']['visual_contract'],teacher_source_sha256=source_sha)
    provenance=dict(version=VERSION,parent_inputs=[dict(role=p['role'],**p['bound']) for p in parents],
        episode_id_mapping=mapping,source_group_policy='same original parent family;no invented heldout;seed/run/env not independent')
    data['native_pool_merge']=deepcopy(provenance)
    context={key:deepcopy(reference['context'][key]) for key in STATIC}
    context.update({key:_row_tree(parents,key) for key in ROW_TREES})
    context.update({key:[row for p in parents for row in _raw_rows(p,key)] for key in RAW_ROWS})
    context['merge_provenance']=deepcopy(provenance)
    context['teacher_shadow_parity']=deepcopy(reference['context']['teacher_shadow_parity'])
    context['teacher_shadow_parity'].update(
        checked_physics_substeps=sum(p['context']['teacher_shadow_parity']['checked_physics_substeps'] for p in parents),
        native_proof_scope='aggregate_of_individual_parent_native_readbacks;no_new_native_execution',
        parent_proofs=[dict(report=p['bound']['report'],parity=deepcopy(p['context']['teacher_shadow_parity'])) for p in parents])
    for p in parents:
        for item in p['identities']: _checked(item)
    target.mkdir(parents=True,exist_ok=False)
    try:
        dataset_path=target/'merged_visual_supervision.pt'
        with dataset_path.open('xb') as stream: torch.save(data,stream)
        dataset_id=file_identity(dataset_path); context['dataset_sha256']=dataset_id['sha256']
        sequence=SequenceDataset(data,context,dataset_sha256=dataset_id['sha256'],sampling_mode='event')
        parity=verify_command_windows(sequence,context['controller_contract'])
        context_path=target/'merged_sequence_context.pt'
        with context_path.open('xb') as stream: torch.save(context,stream)
        context_id=file_identity(context_path)
        index_path=target/'merged_sequence_index.pt'
        with index_path.open('xb') as stream:
            torch.save(dict(version='dh116-merged-sequence-index-v1',windows=sequence.windows,
                probabilities=sequence.sampling_probabilities,report=sequence.report),stream)
        for p in parents:
            for item in p['identities']: _checked(item)
        for item in implementation.values(): _checked(item)
        result=dict(version=VERSION,status='merged_offline',dataset=dataset_id,context=context_id,index=file_identity(index_path),
            parent_inputs=provenance['parent_inputs'],episode_id_mapping=mapping,sequence_report=sequence.report,
            accumulated_window_parity=parity,controller_source_sha256=context['controller_source_sha256'],
            action_contract_sha256=context['action_contract_sha256'],teacher_reference_source_sha256=source_sha,
            fit_only=True,heldout_independent_source_count=0,merged_episode_count=len(episodes),merged_samples=len(rows),
            parent_native_counts=[dict(role=p['role'],report=p['bound']['report'],
                reported_native_successful_episodes=p['report'].get('native_successful_episodes'),
                reported_observed_terminal_successes=p['report'].get('observed_terminal_successes'),
                selected_successful_pinch_first_episodes=len(p['data']['episodes'])) for p in parents],
            new_native_trials=0,single_native_collection_report=False,student_success_claimed=False,ppo_updates=0,
            source_group_policy=provenance['source_group_policy'],
            case_identity_policy='base02 replay copies retained as one parent family;Tier1 canonical float32 writer cases must be unique and differ from base',
            implementation=implementation,no_original_input_modified=True)
        result['merge_report']=_write_json(target/'merge_report.json',result)
        return result
    except BaseException as error:
        _write_json(target/'merge_failure.json',dict(version=VERSION,status='refused_offline',error=repr(error),
            partial_new_outputs_are_unqualified=True,parent_inputs=provenance['parent_inputs'],new_native_trials=0))
        raise


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True,help='JSON with explicit role/report/dataset/context path+SHA')
    parser.add_argument('--tier1',type=Path,nargs='+',required=True,help='one JSON identity bundle per actual successful Tier1 collection')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=merge_native_success_pools(_json(args.base),[_json(p) for p in args.tier1],args.output)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
