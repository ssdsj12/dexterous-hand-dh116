"""Opt-in event identity/cache contracts; synthetic CPU records are not native evidence."""
import copy
import importlib
import json

import pytest

from test_stage6_comparison_cache import fixture as cache_fixture, identity
from test_vision_branching_eval import source_fixture, write_config, runner_fixture


def branch():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_branching_eval')


def event_api():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_stage7_event_reference')


def synthetic_facts(cases):
    return dict(cases=[dict(source={name: case[name] for name in ('run_uuid','env_id','episode_uid')},
        events={**{name:dict(tick=case['event_ticks'][name],available=True) for name in ('CONTACT','LIFT_ONSET','HOLD')},
            'PRE_CONTACT':dict(tick=0,available=True,first_actual_post_contact=dict(policy_step=1))})
        for case in cases])


def test_precontact_cannot_execute_without_actual_event_reference(tmp_path):
    path, cfg = write_config(tmp_path)
    cfg.update(mode='branch', event='PRE_CONTACT', branch_steps=4)
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='event.reference'):
        branch().load_native_config(path)


def test_event_reference_is_normalized_verified_and_bound_to_common_source_inputs(tmp_path,monkeypatch):
    path,cfg=write_config(tmp_path)
    ref=tmp_path/'actual-event-reference.json';ref.write_text('synthetic manifest bytes')
    cfg.update(mode='branch',event='PRE_CONTACT',branch_steps=4,event_reference=dict(identity(ref),path=ref.name))
    calls=[]
    def verify(value,expected_inputs,*,required_event=None):
        calls.append((value,expected_inputs,required_event));return dict(cases=[])
    monkeypatch.setattr(event_api(),'load_event_reference',verify)
    path.write_text(json.dumps(cfg));loaded=branch().load_native_config(path)
    assert calls and calls[0][0]==identity(ref) and calls[0][2]=='PRE_CONTACT'
    assert set(calls[0][1])=={'teacher_dataset','sequence_context','source_initial_state','source_collection_report'}
    assert loaded['checkpoint']==cfg['checkpoint']
    ref.write_text('changed')
    with pytest.raises(ValueError,match='SHA256'):
        branch().verify_inputs(loaded)


def test_attach_precontact_retains_original_events_and_does_not_modify_saved_cases(monkeypatch):
    data,context,proof=source_fixture()
    cases=branch().source_cases(data,context,proof,dataset_sha256='d'*64)
    original=[copy.deepcopy(case['event_ticks']) for case in cases]
    facts=synthetic_facts(cases)
    monkeypatch.setattr(event_api(),'load_event_reference',lambda *a,**k:copy.deepcopy(facts))
    cfg=dict(mode='branch',event='PRE_CONTACT',event_reference=dict(path='/synthetic/reference.json',sha256='a'*64),
        **{name:dict(path='/synthetic/'+name,sha256='b'*64) for name in event_api().COMMON_INPUTS})
    updated=branch().apply_event_reference(cases,cfg)
    assert [case['event_ticks'] for case in cases]==original
    assert all(case['event_ticks']['PRE_CONTACT']==0 for case in updated)
    assert all({name:case['event_ticks'][name] for name in original[0]}==original[i] for i,case in enumerate(updated))
    assert all(case['event_sources']['PRE_CONTACT']['source']=='actual_postphysics_reference_preceding_policy_row' for case in updated)


@pytest.mark.parametrize('wrong',['episode','old_contact','missing_previous','not_strictly_before'])
def test_event_reference_facts_cannot_relabel_another_source_or_contact_boundary(monkeypatch,wrong):
    data,context,proof=source_fixture()
    cases=branch().source_cases(data,context,proof,dataset_sha256='d'*64)
    facts=synthetic_facts(cases)
    if wrong=='episode':facts['cases'][-1]['source']['episode_uid']='another-episode'
    elif wrong=='old_contact':facts['cases'][-1]['events']['CONTACT']['tick']=3
    elif wrong=='missing_previous':facts['cases'][-1]['events']['PRE_CONTACT']['tick']=-1
    else:facts['cases'][-1]['events']['PRE_CONTACT']['first_actual_post_contact']['policy_step']=0
    monkeypatch.setattr(event_api(),'load_event_reference',lambda *a,**k:copy.deepcopy(facts))
    cfg=dict(mode='branch',event='PRE_CONTACT',event_reference=dict(path='/synthetic/ref',sha256='a'*64),
        **{name:dict(path='/synthetic/'+name,sha256='b'*64) for name in event_api().COMMON_INPUTS})
    with pytest.raises(ValueError):branch().apply_event_reference(cases,cfg)
    assert all('PRE_CONTACT' not in case['event_ticks'] for case in cases)


def test_stage7_cache_cannot_reuse_old_no_reference_event_metadata(tmp_path):
    source,_,config,implementation,_=cache_fixture(tmp_path)
    cache=importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_comparison_cache')
    saved=cache.build_comparison_cache(source,tmp_path/'cache')
    ref=tmp_path/'events.json';ref.write_text('synthetic event proof')
    with pytest.raises(ValueError,match='input|reference'):
        cache.load_comparison_cache(saved,dict(config,event_reference=identity(ref)),implementation_sha256=implementation)


def test_stage7_cache_binds_actual_reference_bytes_and_path(tmp_path):
    source,report,config,implementation,_=cache_fixture(tmp_path)
    ref=tmp_path/'events.json';ref.write_text('synthetic event proof')
    config['event_reference']=identity(ref)
    report['inputs']=copy.deepcopy(config);report['same_state_comparisons']['comparison_inputs']=copy.deepcopy(config)
    source.write_text(json.dumps(report))
    cache=importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_comparison_cache')
    saved=cache.build_comparison_cache(source,tmp_path/'cache')
    result=cache.load_comparison_cache(saved,config,implementation_sha256=implementation)
    assert result['comparison_inputs']==config
    moved=tmp_path/'same-events.json';moved.write_bytes(ref.read_bytes())
    with pytest.raises(ValueError):
        cache.load_comparison_cache(saved,dict(config,event_reference=identity(moved)),implementation_sha256=implementation)


def test_native_entry_uses_bound_precontact_and_records_reference_without_claiming_native(tmp_path,monkeypatch):
    args,native,device,task,env,camera,issued=runner_fixture(tmp_path,monkeypatch,
        mode='branch',event='PRE_CONTACT',steps=4,terminal_tick=999)
    path=tmp_path/'native.json';cfg=json.loads(path.read_text())
    ref=tmp_path/'events.json';ref.write_text('synthetic proof; loader is injected for CPU scheduling only')
    cfg['event_reference']=identity(ref);path.write_text(json.dumps(cfg))
    import torch
    data=torch.load(cfg['teacher_dataset']['path'],weights_only=False)
    context=torch.load(cfg['sequence_context']['path'],weights_only=False)
    proof=json.loads(open(cfg['source_collection_report']['path']).read())
    cases=branch().source_cases(data,context,proof,dataset_sha256=cfg['teacher_dataset']['sha256'])
    facts=synthetic_facts(cases)
    for row in facts['cases']:
        row['events']['PRE_CONTACT'].update(tick=1,first_actual_post_contact=dict(policy_step=2))
    monkeypatch.setattr(event_api(),'load_event_reference',lambda *a,**k:copy.deepcopy(facts))
    result=branch().run_stage6_native(native,device,task,env,args,{},camera=camera)
    assert len(issued)==5 and issued[0][0].sum()==0.
    assert all(torch.equal(action[0],torch.full((24,),.73)) for action in issued[1:])
    assert result['inputs']['event_reference']==identity(ref)
    assert result['same_state_comparisons']['comparison_inputs']['event_reference']==identity(ref)
    assert result['native_runtime_verified'] is False and result['pure_student_native_success_count']==0
    assert result['branch_switch_reached_count']==4
    assert all(case['event_entry_ticks']['PRE_CONTACT']==1 for case in result['source_case_definitions'])
