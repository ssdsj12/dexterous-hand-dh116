"""Synthetic CPU file/provenance fixtures; no test is native success evidence.

The qualification flags exercise the file gates, just as other file-workflow
fixtures do. No fixture, output or count is used as actual Stage6 evidence.
"""
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from test_vision_sequence_dataset import fixture_data, fixture_context
from test_vision_sequence_training import command_dataset
from test_vision_evaluation_reset import templates


@pytest.fixture(autouse=True)
def cpu_threads_and_rng():
    threads=torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(2)
        try:yield
        finally:torch.set_num_threads(threads)


def api():
    name='go2_pvcnn.control.dh116.tabletop_vision_sequence_pool_merge'
    assert importlib.util.find_spec(name), 'independent native-success pool merge entrypoint missing'
    return importlib.import_module(name)


def identity(path):
    import hashlib
    return dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def save_json(path,value):path.write_text(json.dumps(value),encoding='utf-8');return identity(path)


def common_files(root):
    root.mkdir()
    paths={}
    for name in ('checkpoint','teacher','camera','pinch','tripod'):
        p=root/(name+'.fixture');p.write_bytes(('synthetic-'+name).encode());paths[name]=identity(p)
    return paths


def parent(root,common,*,role='base02',run='cpu-base',count=None,case_seed=116603):
    """Native booleans below are synthetic schema inputs, never actual trials."""
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_perturbation as perturb
    from go2_pvcnn.control.dh116.tabletop_evaluation_trials import GRASP_STYLES,_digest
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import evaluation_physics_sha256
    from go2_pvcnn.control.dh116.tabletop_vision_command_shadow import action_contract_sha256
    root.mkdir();count=(4 if role=='base02' else 1) if count is None else count
    _,recorded,contract=command_dataset();data=fixture_data((16,)*count,groups=['1'*64]*count)
    data['observations']=recorded.data['observations'].repeat(count,1)
    data['actions'].zero_();data['reference_actions'].zero_();data['teacher_source_sha256']=common['teacher']['sha256']
    from go2_pvcnn.control.dh116.tabletop_surface_contract import bind_contract,ACTION_NAME,LEGACY_OBSERVATION_VERSION
    critic=bind_contract(dict(action_size=24,action=ACTION_NAME,observation_size=351,
        observation_version=LEGACY_OBSERVATION_VERSION,source_sha256={'left':'b'*64,'right':'c'*64},object_radius_m=.008))
    visual=copy.deepcopy(data['visual_contract']);visual.update(camera_config_sha256=common['camera']['sha256'],
        primary_reset_sha256=common['pinch']['sha256'],secondary_reset_sha256=common['tripod']['sha256'],
        privileged_critic_contract=critic)
    data['visual_contract']=visual;report_path=root/'teacher_collection_report.json'
    for row in data['sample_metadata']:
        row.update(run_uuid=run,episode_uid=f'{run}:{row["env_id"]}:first',previous_action=[0.]*24)
    audits={}
    for key,ep in data['episodes'].items():
        ep['proof_path']=str(report_path);ep['run_uuid']=run;ep['episode_uid']=f'{run}:{ep["env_id"]}:first'
        audit=dict(style='pinch_thumb_index',always_bearing_fingers_per_hand=[['thumb','index']]*2,
            mean_digit_load_n=[[1.,1.,0.,0.,0.]]*2)
        ep['native_contact_style_audit']=audit;audits[key]=audit
    context=fixture_context(data)
    for group in ('controller_start','physics_context','teacher_targets','events'):
        context[group]={key:torch.cat([value.clone() for _ in range(count)]) for key,value in recorded.context[group].items()}
    context.update(controller_contract=contract,controller_source_sha256=contract['controller_source_sha256'],
        action_contract_sha256=action_contract_sha256(),event_sources=dict(native_runtime_verified=True),
        command_cache_source='separate_recorded_isaaclab_user_cache_not_backend')
    parity=dict(passed=True,checked_physics_substeps=16*8*count,max_float64_error=0.,max_sdk_cast_float32_error=0.,
        native_sdk_readback_required=True,native_sdk_readback={key:dict(available=True,missing_hand_substeps=0,max_error=0.,passed=True)
            for key in ('backend','command_cache')})
    context['teacher_shadow_parity']=parity
    for name in ('native_backend_targets','command_cache','raw_measured_packets','post_physics_feedback'):
        context[name]=[dict(sample_key=tuple(row[k] for k in ('run_uuid','env_id','episode_uid','tick')),
            physics_substeps=[dict(owned_fixture_tensor=torch.tensor([float('nan'),-0.],dtype=torch.float64)) for _ in range(8)])
            for row in data['sample_metadata']]
    tier=None
    if role=='tier1':
        prov=dict(source_checkpoint=common['checkpoint'],teacher_reference=common['teacher'],
            initializations=dict(zip(GRASP_STYLES,(common['pinch'],common['tripod']))),
            physics_sha256=evaluation_physics_sha256(visual['privileged_critic_contract']),
            camera_sha256=visual['camera_config_sha256'],source_privileged_critic_contract_sha256=_digest(visual['privileged_critic_contract']),
            parent_source_groups=dict(zip(GRASP_STYLES,('1'*64,'2'*64))))
        manifest=perturb.generate_teacher_perturbation_manifest(templates(),seed=case_seed,provenance=prov,count_per_style=max(1,count))
        mid=save_json(root/'perturbation_manifest.json',manifest)
        actual=[dict(trial_id=t['trial_id'],grasp_style=t['grasp_style'],authored_state_sha256=t['state_sha256'],
            readback_state_sha256=t['state_sha256'],physical_state=t['physical_state']) for t in manifest['trials']]
        readback=dict(actual_states=actual,unique_readback_count=len(actual),native_validation_complete=False)
        evidence=dict(manifest=mid,reset_readback=save_json(root/'reset.json',dict(native_runtime_verified=True,clock={'step':0},readback=readback)),
            warmup_readback=save_json(root/'warmup.json',dict(clock={'step':0},readback=readback)),
            camera_geometry=save_json(root/'geometry.json',dict(passed=True,failed_env_ids=[])),runtime_camera=common['camera'])
        tier=dict(version=perturb.VERSION,tier=1,first_packet_verified=True,native_runtime_verified=True,
            input_manifest=mid,evidence=evidence,state_query_oracle=False,pool_scope='fresh_successful_pinch_first_episodes_only;tripod_outcomes_audit_only')
        tier['implementation']={name:identity(Path(perturb.__file__).with_name(name+'.py')) for name in
            ('tabletop_vision_teacher_perturbation','tabletop_vision_evaluation_reset','tabletop_vision_teacher_collection')}
        for row in data['sample_metadata']:
            t=manifest['trials'][row['env_id']]
            row.update(source_group_id='1'*64,teacher_perturbation_tier=1,teacher_perturbation_manifest_sha256=mid['sha256'],
                teacher_perturbation_trial_id=t['trial_id'],teacher_reset_authored_state_sha256=t['state_sha256'],
                teacher_reset_readback_state_sha256=t['state_sha256'],teacher_family_lineage_id=t['family_lineage_id'])
        for key,ep in data['episodes'].items():
            row=next(row for row in data['sample_metadata'] if str(row['episode_id'])==key)
            ep.update({k:v for k,v in row.items() if k.startswith('teacher_') or k=='source_group_id'})
    torch.save(data,root/'successful.pt');did=identity(root/'successful.pt');context['dataset_sha256']=did['sha256']
    torch.save(context,root/'context.pt');cid=identity(root/'context.pt')
    report=dict(status='completed',mode='teacher_reference_mpc',mpc_profile='pressure_geometry_constrained',
        source_sha256=data['teacher_source_sha256'],native_runtime_verified=True,successful_episodes=count,
        native_successful_episodes=count,observed_terminal_successes=count,ppo_updates=0,
        teacher_success_is_student_success=False,dagger_state_query_oracle=False,
        actor_observation_size=3844,critic_observation_size=371,action_size=24,
        successful_dataset=dict(did,samples=len(data['actions'])),successful_sequence_context=dict(cid,
            teacher_shadow_parity=parity,controller_contract=copy.deepcopy(contract)),
        sequence_context_collection=dict(enabled=True,run_uuid=run,successful_sidecar_available=True),
        contact_style_audit=audits,episodes=copy.deepcopy(data['episodes']))
    if tier is not None:report['teacher_perturbation']=tier
    rid=save_json(report_path,report)
    return dict(role=role,report=rid,dataset=did,context=cid)


def rewrite_report(value,change):
    d=json.loads(Path(value['report']['path']).read_text());change(d);value['report']=save_json(Path(value['report']['path']),d)


def rewrite_pool(value,change):
    data=torch.load(value['dataset']['path'],weights_only=False);context=torch.load(value['context']['path'],weights_only=False)
    change(data,context);torch.save(data,value['dataset']['path']);value['dataset'].update(identity(value['dataset']['path']))
    context['dataset_sha256']=value['dataset']['sha256'];torch.save(context,value['context']['path'])
    value['context'].update(identity(value['context']['path']))
    def update(report):
        report['successful_dataset'].update(value['dataset']);report['successful_sequence_context'].update(value['context'])
        report['successful_sequence_context']['controller_contract']=copy.deepcopy(context['controller_contract'])
        report['episodes']=copy.deepcopy(data['episodes'])
    rewrite_report(value,update)


def test_merge_keeps_tensor_bytes_and_uid_only_remaps_local_episode_ids(tmp_path):
    m=api();common=common_files(tmp_path/'common');base=parent(tmp_path/'base',common)
    extra=parent(tmp_path/'tier1',common,role='tier1',run='cpu-tier1')
    before=[identity(v[k]['path']) for v in (base,extra) for k in ('report','dataset','context')]
    result=m.merge_native_success_pools(base,[extra],tmp_path/'merged')
    data=torch.load(result['dataset']['path'],weights_only=False);context=torch.load(result['context']['path'],weights_only=False)
    originals=[torch.load(v['dataset']['path'],weights_only=False) for v in (base,extra)]
    assert torch.equal(data['observations'],torch.cat([v['observations'] for v in originals]))
    assert torch.equal(data['actions'],torch.cat([v['actions'] for v in originals]))
    assert len(data['episodes'])==5 and set(data['episodes'])==set(map(str,range(5)))
    assert data['sample_metadata'][-1]['origin_episode_id']==0 and data['sample_metadata'][-1]['episode_id']==4
    assert data['sample_metadata'][-1]['episode_uid']=='cpu-tier1:0:first'
    assert all(row['source_group_id']=='1'*64 for row in data['sample_metadata'])
    assert context['dataset_sha256']==identity(result['dataset']['path'])['sha256']
    assert result['sequence_report']['fit_only'] and result['heldout_independent_source_count']==0
    assert result['accumulated_window_parity']['passed'] and result['new_native_trials']==0
    assert result['single_native_collection_report'] is False and len(result['parent_native_counts'])==2
    assert all(identity(item['path'])==item for item in before)
    # Preserve raw NaN payload/signed-zero bytes rather than dropping raw evidence.
    raw=context['raw_measured_packets'][-1]['physics_substeps'][0]['owned_fixture_tensor']
    assert torch.equal(raw.view(torch.uint8),torch.tensor([float('nan'),-0.],dtype=torch.float64).view(torch.uint8))


@pytest.mark.parametrize('mutation', ['not_native','failed','recovered','wrong_style','no_contact_audit',
    'not_first','missing_context','old_controller','synthetic_flag','wrong_camera'])
def test_native_parent_gate_rejects_bad_qualification_or_identity_without_output(tmp_path,mutation):
    m=api();common=common_files(tmp_path/'common');base=parent(tmp_path/'base',common)
    extra=parent(tmp_path/'tier1',common,role='tier1',run='cpu-tier1')
    if mutation=='not_native':rewrite_report(extra,lambda d:d.update(native_runtime_verified=False))
    elif mutation=='failed':rewrite_report(extra,lambda d:d['episodes']['0'].update(success=False))
    elif mutation=='recovered':rewrite_report(extra,lambda d:d['episodes']['0'].update(recovered=True))
    elif mutation=='wrong_style':rewrite_report(extra,lambda d:d['episodes']['0'].update(grasp_style='tripod_thumb_index_middle'))
    elif mutation=='no_contact_audit':rewrite_report(extra,lambda d:d['episodes']['0'].pop('native_contact_style_audit'))
    elif mutation=='not_first':rewrite_report(extra,lambda d:d['episodes']['0'].update(first_episode=False))
    elif mutation=='synthetic_flag':rewrite_report(extra,lambda d:d.update(synthetic_cpu_fixture=True))
    elif mutation=='missing_context':extra['context']['sha256']='0'*64
    else:
        target=extra['context'] if mutation=='old_controller' else extra['dataset']
        value=torch.load(target['path'],weights_only=False)
        if mutation=='old_controller':value['controller_source_sha256']='0'*64
        else:value['visual_contract']['camera_config_sha256']='0'*64
        torch.save(value,target['path']);target.update(identity(target['path']))
    with pytest.raises((ValueError,RuntimeError)):m.merge_native_success_pools(base,[extra],tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


def test_duplicate_uid_and_same_perturb_case_in_another_run_are_refused(tmp_path):
    m=api();common=common_files(tmp_path/'common');base=parent(tmp_path/'base',common)
    extra=parent(tmp_path/'tier1',common,role='tier1',run='cpu-tier1')
    for values in ([extra,extra],[extra,parent(tmp_path/'copy',common,role='tier1',run='another-run')]):
        with pytest.raises(ValueError,match='duplicate|case|identity'):m.merge_native_success_pools(base,values,tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


def test_no_actual_tier1_or_changed_input_never_opens_real_merge_gate(tmp_path):
    m=api();common=common_files(tmp_path/'common');base=parent(tmp_path/'base',common)
    with pytest.raises(ValueError,match='Tier1|tier1'):m.merge_native_success_pools(base,[],tmp_path/'absent')
    extra=parent(tmp_path/'tier1',common,role='tier1',run='cpu-tier1')
    Path(extra['dataset']['path']).write_bytes(b'changed')
    with pytest.raises((ValueError,RuntimeError)):m.merge_native_success_pools(base,[extra],tmp_path/'changed')
    assert not (tmp_path/'changed').exists()


@pytest.mark.parametrize('kind',['missing_run_proof','wrong_run_proof','missing_implementation','inconsistent_native_count'])
def test_parent_run_code_and_native_counts_are_explicit_proofs_not_assumed(tmp_path,kind):
    m=api();common=common_files(tmp_path/'common');base=parent(tmp_path/'base',common)
    extra=parent(tmp_path/'tier1',common,role='tier1',run='cpu-tier1')
    if kind=='missing_run_proof':rewrite_report(extra,lambda d:d.pop('sequence_context_collection'))
    elif kind=='wrong_run_proof':rewrite_report(extra,lambda d:d['sequence_context_collection'].update(run_uuid='another-run'))
    elif kind=='missing_implementation':rewrite_report(extra,lambda d:d['teacher_perturbation'].pop('implementation'))
    else:rewrite_report(extra,lambda d:d.update(native_successful_episodes=0))
    with pytest.raises(ValueError):m.merge_native_success_pools(base,[extra],tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


@pytest.mark.parametrize('kind',['old_source_alias','target_tamper','unknown_field','camera_mismatch','missing_raw_substep'])
def test_even_updated_file_sha_cannot_override_controller_physics_or_full_window_parity(tmp_path,kind):
    m=api();common=common_files(tmp_path/'common');base=parent(tmp_path/'base',common)
    extra=parent(tmp_path/'tier1',common,role='tier1',run='cpu-tier1')
    def change(data,context):
        if kind=='old_source_alias':
            context['controller_source_sha256']='0'*64;context['controller_contract']['controller_source_sha256']='0'*64
        elif kind=='target_tamper':context['teacher_targets']['finger'][0,0,0]+=.001
        elif kind=='unknown_field':context['unhandled_actual_physics']=torch.ones(len(data['actions']),8,3)
        elif kind=='missing_raw_substep':context['native_backend_targets'][0]['physics_substeps'].pop()
        else:data['visual_contract']['camera_config_sha256']='0'*64
    rewrite_pool(base,change)
    with pytest.raises((ValueError,RuntimeError)):m.merge_native_success_pools(base,[extra],tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


@pytest.mark.parametrize('kind',['missing_assets','missing_raw_substep'])
def test_standalone_parent_validation_checks_assets_and_all_raw_slots(tmp_path,kind):
    m=api();common=common_files(tmp_path/'common');base=parent(tmp_path/'base',common)
    def change(data,context):
        if kind=='missing_assets':
            from go2_pvcnn.control.dh116.tabletop_surface_contract import semantic_sha256
            critic=data['visual_contract']['privileged_critic_contract'];critic.pop('source_sha256')
            critic['surface_contract_sha256']=semantic_sha256(critic)
        else:context['post_physics_feedback'][0]['physics_substeps'].pop()
    rewrite_pool(base,change)
    with pytest.raises(ValueError):m.load_native_parent(base)
