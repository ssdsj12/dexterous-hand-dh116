"""Read-only physical repeat ranges, separate from numerical command errors.

Recorded native executions describe an observed range, never a statistical
confidence interval or an expert query at a new state. At least two actually
available postphysical references are required separately for each field.
Missing post-physics
contact and named palm-link evidence stays unavailable. Nothing here feeds a
controller, changes a threshold, or attributes failure to an Actor difference.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .linear_coupling import ACTIVE


VERSION='dh116-stage6-physical-repeat-divergence-v1'
PHASE='post_physics_before_terminal_reset'
FIELDS=('q_rad','qd_rad_s','backend_target_rad','tracking_error_rad','base_link_position_w_m','palm_link_position_w_m')


def _plain(value):
    if isinstance(value,np.ndarray):return _plain(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def file_identity(path):
    path=Path(path).resolve(strict=True);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return dict(path=str(path),sha256=digest.hexdigest())


def checked_identity(value):
    if not isinstance(value,dict) or not isinstance(value.get('path'),str):raise ValueError('explicit actual path/SHA256 required')
    actual=file_identity(value['path'])
    if actual['sha256']!=value.get('sha256'):raise ValueError('immutable input SHA256 changed: '+value['path'])
    return actual


def active_fingers(values,names,dof_types,units):
    values=np.asarray(values);types=np.asarray(dof_types)
    if (not isinstance(names,list) or len(set(names))!=len(names) or values.shape!=(len(names),)
            or types.shape!=(len(names),) or any(name not in names for name in ACTIVE)):
        raise ValueError('unique native named DOFs with all ACTIVE fingers required')
    indices=[names.index(name) for name in ACTIVE]
    if any(types[i]!=0 or units.get(name)!='rad' for name,i in zip(ACTIVE,indices)):
        raise ValueError('native ACTIVE rotational rad units required;wrist meters cannot be aggregated as radians')
    result=values[indices].astype(np.float64)
    if not np.isfinite(result).all():raise ValueError('finite actual named ACTIVE values required')
    return result


class ReportedClockBridge:
    """One World/SDK offset from real policy anchors, with at most8ULP error.

    Derived substep SDK indices are explicitly not independent getter brackets.
    The original nominal controller/physics dt is neither changed nor inferred.
    """
    def __init__(self,cadence_s,origin_s):
        if not math.isfinite(cadence_s) or cadence_s<=0 or not math.isfinite(origin_s):raise ValueError('qualified finite global reported SDK clock required')
        self.cadence=float(cadence_s);self.origin=float(origin_s);self.offset=None;self.maximum=0.;self.anchors=0;self.posts=0

    def _time(self,sdk_step,when):
        if type(sdk_step) is not int or type(when) not in (int,float) or not math.isfinite(when):raise ValueError('actual finite clock required')
        expected=self.origin+sdk_step*self.cadence;residual=abs(float(when)-expected)
        if residual>8*max(math.ulp(float(when)),math.ulp(expected)):raise ValueError('timestamp does not match qualified global affine reported SDK clock')
        self.maximum=max(self.maximum,residual)

    def add_policy_anchor(self,anchor):
        world,sdk,when=anchor.get('physics_step'),anchor.get('simulation_step_index'),anchor.get('timestamp_s')
        if type(world) is not int or world<0 or type(sdk) is not int or sdk<0:raise ValueError('actual World and SDK policy indices required')
        offset=sdk-world
        if self.offset is not None and offset!=self.offset:raise ValueError('World/SDK index mapping changed across actual policy anchors')
        self._time(sdk,when);self.offset=offset;self.anchors+=1

    def verify_post(self,world_step,when,anchor):
        if self.offset is None:raise ValueError('recorded policy clock anchor required before physical bridge')
        if type(world_step) is not int or not 1<=world_step-anchor['physics_step']<=8:raise ValueError('physical sample is not one of the anchor original8substeps')
        if anchor['simulation_step_index']-anchor['physics_step']!=self.offset:raise ValueError('policy anchor mapping changed')
        sdk=world_step+self.offset;self._time(sdk,when);self.posts+=1
        return dict(substep=world_step-anchor['physics_step'],derived_sdk_physics_step=sdk,
            independent_getter_clock_pair_available=False,derivation='actual_policy_SDK_minus_World_offset;not_an_additional_getter_clock_readback')

    def summary(self):
        return dict(reported_sdk_cadence_s=self.cadence,reported_sdk_origin_s=self.origin,
            sdk_minus_world_physics_step=self.offset,actual_policy_anchors=self.anchors,postphysical_samples=self.posts,
            maximum_timestamp_residual_s=self.maximum,arithmetic_roundoff_allowance='maximum8ULP',
            independent_getter_clock_pair_available=False,controller_or_physics_dt_changed=False)


def _key(row):
    if (type(row.get('env_id')) is not int or row['env_id']<0 or not isinstance(row.get('source_episode_uid'),str)
            or not row['source_episode_uid'] or type(row.get('policy_step')) is not int or row['policy_step']<0
            or type(row.get('substep')) is not int or not 1<=row['substep']<=8):raise ValueError('explicit env/source_episode_uid/tick/original substep required')
    return row['env_id'],row['source_episode_uid'],row['policy_step'],row['substep']


def _index_run(run):
    if run.get('native_runtime_verified') is not True or not isinstance(run.get('execution_run_uuid'),str):raise ValueError('actual native execution provenance required')
    result={};uids={};closed=set();last={};excluded=0
    for row in run['samples']:
        env=row.get('env_id')
        if env in closed:excluded+=1;continue
        key=_key(row)
        if row.get('phase')!=PHASE:raise ValueError('only actual postphysical samples before reset are eligible;no pre-control substitution')
        if not isinstance(row.get('execution_episode_uid'),str):raise ValueError('actual execution episode identity required')
        if env in uids and uids[env]!=row['execution_episode_uid']:raise ValueError('execution episode identity changed before first terminal')
        uids[env]=row['execution_episode_uid']
        if key in result:raise ValueError('duplicate physical sample identity')
        position=key[2:]
        if env in last and position<=last[env]:raise ValueError('physical sample order regressed')
        last[env]=position
        if row.get('q_clock') is not None and row.get('backend_clock') is not None and row['q_clock']!=row['backend_clock']:raise ValueError('backend and postq clocks differ')
        when=row.get('timestamp_s')
        if type(when) not in (int,float) or not math.isfinite(when):raise ValueError('finite actual physical timestamp required')
        result[key]=row
        if row.get('terminal') is True:closed.add(env)
    return result,excluded


def _field(row,name):
    if name=='tracking_error_rad':
        if row.get('tracking_pair_available') is not True:return None
        q,target=_field(row,'q_rad'),_field(row,'backend_target_rad')
        return None if q is None or target is None else np.abs(target-q)
    value=row.get(name)
    if value is None:return None
    value=np.asarray(value,dtype=np.float64)
    expected=(12,) if name in ('q_rad','qd_rad_s','backend_target_rad') else (2,3)
    if value.shape!=expected or not np.isfinite(value).all():return None
    return value


def build_repeat_envelope(references):
    if len(references)<2 or len({r.get('execution_run_uuid') for r in references})!=len(references):raise ValueError('at least two distinct actual native runs required;copies are not repeats')
    indexed=[_index_run(run)[0] for run in references]
    lineage={(key[0],key[1]) for key in indexed[0]}
    if any({(key[0],key[1]) for key in index}!=lineage for index in indexed[1:]):raise ValueError('reference original source episode lineage differs')
    keys=set().union(*(set(index) for index in indexed));ranges={};exact={}
    for name in FIELDS:exact[name]=dict(compared_samples=0,unequal_samples=0,maximum_abs_difference=0.,all_equal=False)
    for key in keys:
        rows=[index.get(key) for index in indexed];fields={}
        for name in FIELDS:
            values=[_field(row,name) for row in rows if row is not None];values=[v for v in values if v is not None]
            entry=dict(reference_samples=len(values),available=len(values)>=2)
            if len(values)>=2:
                stack=np.stack(values);entry.update(minimum=stack.min(0),maximum=stack.max(0))
                difference=float((entry['maximum']-entry['minimum']).max());proof=exact[name]
                proof['compared_samples']+=1;proof['unequal_samples']+=difference!=0.;proof['maximum_abs_difference']=max(proof['maximum_abs_difference'],difference)
            fields[name]=entry
        contacts=[row for row in rows if row is not None and row.get('contact_valid') is True and type(row.get('bilateral_opposed_contact')) is bool]
        fields['repeat_contact']=dict(reference_samples=len(contacts),available=len(contacts)>=2,
            all_valid_opposed=len(contacts)>=2 and all(row['bilateral_opposed_contact'] for row in contacts))
        ranges[key]=fields
    for value in exact.values():value['all_equal']=value['compared_samples']>0 and value['unequal_samples']==0
    availability={}
    for name in (*FIELDS,'repeat_contact'):
        counts=[row[name]['reference_samples'] for row in ranges.values()]
        availability[name]=dict(available=any(n>=2 for n in counts),available_samples=sum(n>=2 for n in counts),
            unavailable_samples=sum(n<2 for n in counts),maximum_reference_samples=max(counts,default=0),required_reference_samples=2)
    return dict(reference_run_count=len(references),reference_execution_run_uuids=[r['execution_run_uuid'] for r in references],
        statistical_confidence_interval=False,range_scope='observed_recorded_run_minmax_only;not_statistical_uncertainty_or_new_expert_sources',
        ranges=ranges,availability=availability,reference_exact_match=exact,
        reference_provenance=[r.get('provenance',{}) for r in references])


def _event(row,**extra):
    owner=row.get('action_owner')
    return dict(env_id=row['env_id'],source_episode_uid=row['source_episode_uid'],execution_episode_uid=row['execution_episode_uid'],
        policy_step=row['policy_step'],substep=row['substep'],world_physics_step=row.get('world_physics_step'),timestamp_s=row['timestamp_s'],
        action_owner=owner,owner_scope='Student_owned_sample;chronology_not_causality' if owner=='student' else 'prefix_event_not_Student_evidence',
        independent_getter_clock_pair_available=row.get('independent_getter_clock_pair_available') is True,
        clock_quality=row.get('clock_quality'),cause_claimed=False,**extra)


def compare_to_repeat(candidate,envelope,trigger_rad):
    trigger=np.asarray(trigger_rad,dtype=np.float64)
    if trigger.shape!=(12,) or not np.isfinite(trigger).all() or (trigger<.01).any():raise ValueError('qualified actual12joint calibration trigger required')
    samples,excluded=_index_run(candidate);per_env={};previous_contact={};unavailable=dict(candidate_tracking_pair=0,repeat_key=0,candidate_contact=0)
    for name in FIELDS:unavailable['candidate_'+name]=0
    for key,row in samples.items():
        env=row['env_id'];result=per_env.setdefault(env,dict(env_id=env,first_numeric_command_departure=None,
            first_numeric_physical_departure=None,first_significant_tracking_departure=None,first_repeat_contact_loss=None,
            first_absolute_task_event=None,first_terminal=None,compared_samples=0))
        result['compared_samples']+=1;bounds=envelope['ranges'].get(key)
        if bounds is None:unavailable['repeat_key']+=1
        for name in FIELDS:
            if _field(row,name) is None:unavailable['candidate_'+name]+=1
        for name in ('backend_target_rad','q_rad','qd_rad_s','base_link_position_w_m'):
            value=_field(row,name);field=None if bounds is None else bounds[name]
            if value is None or field is None or not field['available']:continue
            distance=np.maximum(field['minimum']-value,value-field['maximum'])
            if bool((distance>0).any()):
                target='first_numeric_command_departure' if name=='backend_target_rad' else 'first_numeric_physical_departure'
                if result[target] is None:result[target]=_event(row,field=name,max_distance_outside_observed_range=float(distance.max()),
                    classification='numeric_command_difference;not_physical_causality' if name=='backend_target_rad' else 'physical_numeric_difference;not_a_task_or_significance_criterion')
        tracking=_field(row,'tracking_error_rad');field=None if bounds is None else bounds['tracking_error_rad']
        if tracking is None:unavailable['candidate_tracking_pair']+=1
        elif field is not None and field['available']:
            significant=(tracking>field['maximum'])&(tracking>trigger)
            if significant.any() and result['first_significant_tracking_departure'] is None:
                q,target=_field(row,'q_rad'),_field(row,'backend_target_rad')
                named=[dict(active_dimension=int(dim),hand='left' if dim<6 else 'right',joint=ACTIVE[int(dim)%6],
                    backend_target_rad=float(target[dim]),post_q_rad=float(q[dim]),signed_backend_minus_postq_rad=float(target[dim]-q[dim]),
                    abs_tracking_error_rad=float(tracking[dim]),reference_abs_error_min_rad=float(field['minimum'][dim]),
                    reference_abs_error_max_rad=float(field['maximum'][dim]),trigger_rad=float(trigger[dim])) for dim in np.flatnonzero(significant)]
                result['first_significant_tracking_departure']=_event(row,kind='tracking_error_above_qualified_trigger_and_observed_repeat_max',
                    active_dimensions=np.flatnonzero(significant).tolist(),max_abs_tracking_error_rad=float(tracking.max()),
                    observed_reference_samples=field['reference_samples'],significant_named_joints=named,
                    repeat_min_tracking_error_rad=field['minimum'].tolist(),
                    tracking_error_rad=tracking.tolist(),repeat_max_tracking_error_rad=field['maximum'].tolist(),
                    qualified_trigger_rad=trigger.tolist(),backend_target_rad=_field(row,'backend_target_rad').tolist(),post_q_rad=_field(row,'q_rad').tolist())
        valid=row.get('contact_valid') is True;contact=row.get('bilateral_opposed_contact')
        if not valid or type(contact) is not bool:unavailable['candidate_contact']+=1
        repeat_contact=None if bounds is None else bounds['repeat_contact']
        if (valid and contact is False and repeat_contact is not None and repeat_contact['all_valid_opposed']
                and result['first_repeat_contact_loss'] is None):
            result['first_repeat_contact_loss']=_event(row,kind='postphysical_opposed_contact_missing_vs_two_valid_opposed_references')
        absolute=row.get('original_task_event')
        if absolute is None and row.get('failure_code') in (3,6,7):absolute=dict(kind='original_task_failure_code_'+str(row['failure_code']))
        if absolute is None and valid and previous_contact.get(env) is True and contact is False:absolute=dict(kind='effective_opposed_contact_lost')
        if absolute is None and row.get('slip_valid') is True and row.get('slip_m') is not None and row['slip_m']>.0005:absolute=dict(kind='fixed_anchor_slip_exceeded_original_0.5mm')
        previous_contact[env]=contact if valid and type(contact) is bool else None
        if absolute is not None and result['first_absolute_task_event'] is None:
            result['first_absolute_task_event']=_event(row,**absolute,repeat_divergence_qualified=False,
                classification='absolute_original_task_event;not_sufficient_for_repeat_departure_or_Student_causality')
        if row.get('terminal') is True and result['first_terminal'] is None:result['first_terminal']=_event(row,failure_code=row.get('failure_code'))
    return dict(version=VERSION,per_env=[per_env[k] for k in sorted(per_env)],availability=envelope['availability'],
        reference_exact_match=envelope['reference_exact_match'],reference_run_count=envelope['reference_run_count'],statistical_confidence_interval=False,
        postterminal_samples_excluded=excluded,unavailable_counts=unavailable,causality_claimed=False,control_modified=False,
        new_expert_sources=0,production_evaluation_certificate_issued=False,
        contact_scope='same_postphysical_valid_opposed_references_required;precontrol_contact_never_substituted',
        palm_scope='base_link positions separately named;named palm_link unavailable without actual body-link evidence')


def load_teacher_context(data,context,calibration,*,provenance):
    """Use only actual same-SDK backend/post-q packets, never physics_context q.

    This sidecar version has no postphysical contact/lift/slip packet. Event
    labels and pre-control contact are deliberately not used to fill that gap.
    """
    from . import tabletop_vision_teacher_tracking_calibration as cal
    if calibration.get('qualified') is not True:raise ValueError('qualified actual Teacher tracking provenance required')
    for name in ('controller_source_sha256','action_contract_sha256'):
        if calibration.get(name)!=context.get(name):raise ValueError('Teacher context/calibration source identity differs')
    back=cal._evidence_table(context.get('native_backend_targets'),'native_backend_targets')
    post=cal._evidence_table(context.get('post_physics_feedback'),'post_physics_feedback')
    mapping=calibration['clock_mapping'];clock=ReportedClockBridge(mapping['reported_sdk_cadence_s'],mapping['reported_sdk_time_origin_s'])
    source_samples=context.get('samples',{});identity_fields=('run_uuid','env_id','episode_uid','tick')
    if any(name not in source_samples for name in identity_fields):raise ValueError('explicit original context sample identities required for live masking')
    context_rows={}
    for j in range(len(source_samples['tick'])):
        key=(str(source_samples['run_uuid'][j]),int(source_samples['env_id'][j]),str(source_samples['episode_uid'][j]),int(source_samples['tick'][j]))
        if key in context_rows:raise ValueError('duplicate original context sample identity')
        context_rows[key]=j
    physics=context.get('physics_context',{})
    live=np.asarray(physics.get('live'));valid=np.asarray(physics.get('source_valid'))
    if live.shape!=(len(context_rows),8) or valid.shape!=live.shape or live.dtype!=np.bool_ or valid.dtype!=np.bool_:raise ValueError('explicit original per-substep live/source-valid masks required')
    samples=[];run_ids=set();excluded=0
    for meta in data['sample_metadata']:
        key=tuple(meta[name] for name in ('run_uuid','env_id','episode_uid','tick'));run_ids.add(meta['run_uuid'])
        if key not in back or key not in post or key not in context_rows:raise ValueError('actual Teacher sample_key missing in one independently joined packet table')
        episode=data['episodes'][str(meta['episode_id'])]
        if any(episode.get(k) is not True for k in ('first_episode','native_physics','success')) or episode.get('recovered') is not False:raise ValueError('successful first native Teacher source scope required')
        for substep,(b,p) in enumerate(zip(back[key],post[key]),1):
            j=context_rows[key]
            if not bool(live[j,substep-1] and valid[j,substep-1]):excluded+=1;continue
            q,when=cal._post_feedback(p)
            if cal._clock(b)!=when:raise ValueError('Teacher backend and post-q SDK clocks differ')
            clock._time(when[0],when[1])
            if when[0]!=meta['first_physics_step']+mapping['sdk_minus_metadata_physics_step']+substep:raise ValueError('Teacher original sample/substep SDK mapping changed')
            hands=b.get('hands')
            if not isinstance(hands,list) or len(hands)!=2:raise ValueError('actual two-hand backend packets required')
            target=np.concatenate([cal._backend_fingers(hand,meta['env_id']).numpy() for hand in hands])
            root=p.get('measured_root_pose');base=None
            if root is not None:
                root=np.asarray(root.detach().cpu() if hasattr(root,'detach') else root)
                if root.shape==(1,2,7) and np.isfinite(root).all():base=root[0,:,:3].copy()
            samples.append(dict(env_id=int(meta['env_id']),source_episode_uid=meta['episode_uid'],execution_episode_uid=meta['episode_uid'],
                policy_step=int(meta['tick']),substep=substep,world_physics_step=None,timestamp_s=when[1],sdk_physics_step=when[0],
                phase=PHASE,action_owner='original_Teacher02',q_rad=q.numpy(),
                qd_rad_s=p['measured_qd_rad_s'][0].detach().cpu().double().numpy(),backend_target_rad=target,
                base_link_position_w_m=base,palm_link_position_w_m=None,tracking_pair_available=True,
                q_clock=when,backend_clock=cal._clock(b),independent_getter_clock_pair_available=True,
                clock_quality='exact_recorded_SDK_backend_getter_and_post_feedback_pair',
                contact_valid=None,bilateral_opposed_contact=None,slip_valid=False,slip_m=None,lift_valid=False,lift_m=None,
                failure_code=None,terminal=False,live=True,original_task_event=None))
    if len(run_ids)!=1:raise ValueError('one actual original Teacher run required for this two-run repeat contract')
    return dict(execution_run_uuid=next(iter(run_ids)),native_runtime_verified=True,samples=samples,
        provenance=dict(provenance,kind='original_native_Teacher02_postphysical_sidecar',
            post_contact_available=False,precontrol_contact_used=False,named_palm_link_available=False,
            excluded_nonlive_or_source_invalid_substeps=excluded,
            world_physics_step_available=False,per_physics_first_terminal_available=False,
            original_episode_scope='qualified_successful_first_native_episodes_from_dataset',maximum_timestamp_residual_s=clock.maximum))


def _json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _jsonl(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError('actual JSONL object required')
            yield row


def load_native_run(files,*,dataset_id,context_id,calibration):
    """Join raw ownership/terminal evidence and actual named-DOF NPZ packets.

    The diagnostics UID namespace differs from the live/raw execution UID.
    They are bridged only by actual env/episode/reset/World step/timestamp,
    never by asserting their strings are equal or guessing an SDK getter read.
    """
    identities=[checked_identity(files[name]) for name in ('report','audit_summary','policy_trace','reset_comparison')]
    report=_json(identities[0]['path']);audit=_json(identities[1]['path'])
    job=Path(identities[0]['path']).parent;audit_dir=job/'evaluation_action_diagnostics'
    expected_paths=(job/'stage6_native_report.json',audit_dir/'summary.json',job/'live_policy_60hz.jsonl.gz',job/'source_reset_comparison.json')
    if any(Path(row['path'])!=path for row,path in zip(identities,expected_paths)):raise ValueError('native report/audit/trace/reset must belong to this same actual full job directory')
    if _json(identities[3]['path']).get('matched') is not True:raise ValueError('actual exact source cold-reset comparison did not match')
    if (report.get('native_runtime_verified') is not True or report.get('version')!='dh116-stage6-native-v1'
            or report.get('mode') not in ('source','branch','teacher-replay') or report.get('evaluated_source_count')!=4):raise ValueError('actual native four-source Stage6 report required')
    if (report.get('inputs',{}).get('teacher_dataset')!=dataset_id or report.get('inputs',{}).get('sequence_context')!=context_id):raise ValueError('native execution belongs to another source pool')
    checkpoint=checked_identity(report.get('inputs',{}).get('checkpoint'));identities.append(checkpoint)
    if audit.get('checkpoint_sha256')!=checkpoint['sha256']:raise ValueError('native audit checkpoint differs from actual report model')
    if (audit.get('version')!='dh116-stage6-source-physics-audit-v1' or audit.get('status') not in ('recorded','error')
            or audit.get('capture_timing')!='native_after_consume_before_terminal_reset'
            or audit.get('native_backend_target')!='root_physx_view.get_dof_position_targets;no_cache_fallback'):raise ValueError('actual postphysical native backend/q audit required;no cache fallback')
    run_uuid=report.get('execution_run_uuid')
    if not isinstance(run_uuid,str) or not run_uuid:raise ValueError('actual execution UUID required')
    cases={row['env_id']:row for row in report.get('source_case_definitions',[])}
    outcomes={row['env_id']:row for row in report.get('source_rows',[])}
    if set(cases)!={0,1,2,3} or set(outcomes)!={0,1,2,3}:raise ValueError('four original source case identities required')
    if report['mode']=='teacher-replay' and (report.get('status')!='completed' or report.get('completed_count')!=4):raise ValueError('complete actual Teacher-tape native repeat required')
    mapping=calibration.get('clock_mapping',{});bridge=ReportedClockBridge(mapping['reported_sdk_cadence_s'],mapping['reported_sdk_time_origin_s'])
    policies={}
    for row in _jsonl(identities[2]['path']):
        env=row.get('env_id');tick=row.get('policy_step')
        if env not in cases:continue
        if type(tick) is not int or (env,tick) in policies:raise ValueError('duplicate or invalid live policy identity')
        if (row.get('run_uuid')!=run_uuid or row.get('episode_uid')!=outcomes[env].get('execution_episode_uid')
                or row.get('episode_id')!=outcomes[env].get('execution_episode_id')
                or row.get('recorded_teacher_episode_uid')!=cases[env]['source_episode_uid']):raise ValueError('live policy original/execution episode identity differs')
        bridge.add_policy_anchor(row.get('clock_before_action',{}));policies[env,tick]=row
    if not policies:raise ValueError('actual recorded policy SDK/World anchors required')
    raw_id=checked_identity(report.get('raw_physics'));identities.append(raw_id)
    if Path(raw_id['path'])!=job/'source_physics_480hz.jsonl.gz':raise ValueError('raw physical stream belongs to another actual job')
    raw={};raw_closed=set();expected_terminals={};lift_peaks={};valid_lift_peaks={}
    for row in _jsonl(raw_id['path']):
        env=row.get('env_id')
        if env not in cases:raise ValueError('unevaluated neutral rows cannot become source trials')
        if env in raw_closed:raise ValueError('source raw contains samples after owned first terminal')
        tick=row.get('policy_step');policy=policies.get((env,tick));payload=row.get('payload',{})
        if (policy is None or row.get('episode_uid')!=policy['episode_uid']
                or str(row.get('source_episode_id'))!=str(cases[env]['episode_id']) or row.get('action_owner')!=policy.get('owner')):raise ValueError('raw physics/policy ownership or episode bridge differs')
        world=row.get('physics_step');when=row.get('timestamp_s')
        proof=bridge.verify_post(world,when,policy['clock_before_action'])
        if payload.get('live') is not True:raise ValueError('source raw requires actual owned live first-episode sample')
        key=(env,world,when)
        if key in raw:raise ValueError('duplicate raw physical identity')
        raw[key]=(row,policy,proof)
        lift=payload.get('lift_m')
        if type(lift) in (int,float) and math.isfinite(lift):
            peak=dict(env_id=env,source_episode_uid=cases[env]['source_episode_uid'],execution_episode_uid=policy['episode_uid'],
                policy_step=tick,substep=proof['substep'],world_physics_step=world,timestamp_s=when,net_lift_m=float(lift),
                action_owner=row['action_owner'],source_valid=payload.get('lift_valid') is True and payload.get('measurement_valid') is True,
                scope='actual_same_environment_owned_first_episode_raw_argmax;earliest_sample_on_tie')
            if env not in lift_peaks or lift>lift_peaks[env]['net_lift_m']:lift_peaks[env]=peak
            if peak['source_valid'] and (env not in valid_lift_peaks or lift>valid_lift_peaks[env]['net_lift_m']):valid_lift_peaks[env]=dict(peak)
        if payload.get('done') is True:
            expected=outcomes[env].get('terminal')
            if (not isinstance(expected,dict) or (expected.get('policy_step'),expected.get('physics_step'),expected.get('timestamp_s'),expected.get('failure_code'))
                    !=(tick,world,when,row.get('failure_code'))):raise ValueError('first native terminal report/raw alignment differs')
            raw_closed.add(env);expected_terminals[env]=key
    for env,outcome in outcomes.items():
        if (outcome.get('terminal') is not None)!=(env in raw_closed):raise ValueError('reported first terminal absent from actual raw stream')
        reported_peak=outcome.get('max_net_lift_m');observed=lift_peaks.get(env)
        if reported_peak is not None:
            if observed is None or observed['net_lift_m']!=reported_peak:raise ValueError('actual raw netlift peak differs from original native source-report maximum')
            observed.update(source_report_max_net_lift_m=reported_peak,matches_source_report_max_exactly=True)
        elif observed is not None:observed.update(source_report_max_net_lift_m=None,matches_source_report_max_exactly=None)
    status={side:[] for side in ('left','right')}
    for event in audit.get('source_events',[]):
        if event.get('kind')=='target_source_status' and event.get('hand') in status:
            status[event['hand']].append(event)
    for side in status:status[side].sort(key=lambda row:row['physics_step'])
    paths=[]
    for item in audit.get('files',[]):
        current=checked_identity(item);identities.append(current)
        if Path(current['path']).parent!=audit_dir:raise ValueError('native audit file belongs to another actual full job directory')
        if Path(current['path']).suffix=='.npz':paths.append(current['path'])
    if not paths:raise ValueError('actual immutable named q/qd/backend NPZ files required')
    samples=[];seen=set();uid_bridges={};na_counts={};skipped=0
    for path in sorted(paths):
        with np.load(path,allow_pickle=False) as archive:
            # NPZ __getitem__ decompresses each access. Own this one immutable
            # chunk once rather than re-reading its arrays for every env row.
            arrays={name:archive[name] for name in archive.files}
            shapes=arrays['env_id'].shape
            if len(shapes)!=2:raise ValueError('physical arrays require [batch,environment] identity axes')
            for batch in range(shapes[0]):
                for slot in range(shapes[1]):
                    env=int(arrays['env_id'][batch,slot])
                    if env not in cases:continue
                    select=(batch,slot)
                    if (not bool(arrays['first_trial_eligible'][select]) or bool(arrays['after_first_trial_terminal'][select])
                            or not bool(arrays['live_before_substep'][select])):skipped+=1;continue
                    world=int(arrays['physics_step'][select]);when=float(arrays['timestamp_s'][select]);key=(env,world,when)
                    if key not in raw:raise ValueError('diagnostic physical sample cannot bridge to exact raw World step/timestamp')
                    if key in seen:raise ValueError('duplicate diagnostic first-episode physical identity')
                    seen.add(key);row,policy,clock=raw[key];payload=row['payload']
                    episode=int(arrays['episode_id'][select]);epoch=int(arrays['reset_epoch'][select]);diagnostic_uid=str(arrays['episode_uid'][select])
                    if (episode!=policy['episode_id'] or epoch!=0 or int(arrays['policy_step'][select])!=row['policy_step']
                            or bool(arrays['terminal'][select])!=(payload.get('done') is True)):raise ValueError('diagnostic/raw actual episode/reset/terminal bridge differs')
                    if diagnostic_uid!=f'{job.name}:{epoch}:{env}:{episode}':raise ValueError('diagnostic UID writer namespace belongs to another actual job')
                    uid_bridge=dict(execution_episode_uid=policy['episode_uid'],diagnostic_episode_uid=diagnostic_uid,
                        actual_episode_id=episode,actual_reset_epoch=epoch,source_episode_uid=cases[env]['source_episode_uid'])
                    if env in uid_bridges and uid_bridges[env]!=uid_bridge:raise ValueError('diagnostic/execution episode UID bridge changed')
                    uid_bridges[env]=uid_bridge
                    def available(name):
                        return name in arrays and all(flag in arrays and bool(arrays[flag][select]) for flag in (name+'__available',name+'__finite',name+'__source_valid'))
                    if not available('actor_request24'):raise ValueError('actual issued request required to bind native diagnostic execution')
                    request=np.asarray(arrays['actor_request24'][select],dtype=np.float64);issued=np.asarray(policy.get('issued_action'),dtype=np.float64)
                    if request.shape!=(24,) or issued.shape!=(24,) or not np.isfinite(request).all() or not np.array_equal(request,issued):raise ValueError('native diagnostic issued action differs from same execution live policy tape')
                    fields={}
                    for quantity,out_name in [('q','q_rad'),('qd','qd_rad_s'),('backend_target','backend_target_rad')]:
                        halves=[]
                        for side in ('left','right'):
                            current=[event for event in status[side] if event['physics_step']<=world and event.get('reset_epoch')==epoch]
                            field=side+'_'+quantity
                            if not available(field) or not available(side+'_dof_types') or not current:
                                na_counts[field]=na_counts.get(field,0)+1;halves.append(None);continue
                            evidence=current[-1]['evidence'];backend=evidence.get('backend',{})
                            if quantity=='backend_target' and (backend.get('source')!='physx_backend_get_dof_position_targets' or backend.get('available') is not True or backend.get('error') is not None):raise ValueError('actual native backend unavailable or substituted by cache')
                            halves.append(active_fingers(arrays[field][select],evidence['joint_names'],arrays[side+'_dof_types'][select],evidence.get('units_by_name',{})))
                        fields[out_name]=None if any(value is None for value in halves) else np.concatenate(halves)
                    base=None
                    if available('palm_pose_w'):
                        pose=arrays['palm_pose_w'][select]
                        if pose.shape==(2,7) and np.isfinite(pose).all():base=pose[:,:3].copy()
                    original=None
                    for event_name in ('first_task_related_physical_event','first_prefix_physical_event'):
                        event=outcomes[env].get(event_name)
                        if isinstance(event,dict) and event.get('physics_step')==world:
                            if event.get('timestamp_s')!=when:raise ValueError('original task event clock differs from actual physical sample')
                            original=dict(kind=event['kind'],original_event_record=event)
                    samples.append(dict(env_id=env,source_episode_uid=cases[env]['source_episode_uid'],execution_episode_uid=policy['episode_uid'],
                        diagnostic_episode_uid=diagnostic_uid,policy_step=row['policy_step'],substep=clock['substep'],world_physics_step=world,timestamp_s=when,
                        derived_sdk_physics_step=clock['derived_sdk_physics_step'],phase=PHASE,action_owner=row['action_owner'],**fields,
                        base_link_position_w_m=base,palm_link_position_w_m=None,
                        tracking_pair_available=fields['q_rad'] is not None and fields['backend_target_rad'] is not None,
                        independent_getter_clock_pair_available=False,clock_quality='shared_postphysical_callback_sample_clock;SDK_index_from_policy_anchor;independent_getter_pair_not_recorded',
                        contact_valid=payload.get('contact_valid'),bilateral_opposed_contact=payload.get('bilateral_opposed_contact'),
                        slip_valid=payload.get('slip_valid') is True,slip_m=payload.get('slip_m'),lift_valid=payload.get('lift_valid') is True,lift_m=payload.get('lift_m'),
                        failure_code=row.get('failure_code'),terminal=payload.get('done') is True,live=True,original_task_event=original))
    if seen!=set(raw):raise ValueError('some actual first-episode raw physical samples have no exact named diagnostic bridge')
    for item in identities:checked_identity(item)
    samples.sort(key=lambda row:(row['world_physics_step'],row['env_id']))
    _index_run(dict(execution_run_uuid=run_uuid,native_runtime_verified=True,samples=samples))
    return dict(execution_run_uuid=run_uuid,native_runtime_verified=True,samples=samples,input_identities=identities,
        provenance=dict(report=identities[0],audit_summary=identities[1],policy_trace=identities[2],raw_physics=raw_id,
            reset_comparison=identities[3],
            mode=report['mode'],recorded_status=report.get('status'),recorded_error=report.get('error'),
            clock_bridge=bridge.summary(),UID_bridges=uid_bridges,unavailable_field_counts=na_counts,
            net_lift_peaks=[lift_peaks.get(env,dict(env_id=env,available=False)) for env in sorted(cases)],
            valid_postphysical_net_lift_peaks=[valid_lift_peaks.get(env,dict(env_id=env,available=False)) for env in sorted(cases)],
            peak_report_reduction='finite raw lift_m matches original SourceTelemetryCollector maximum;valid-only peaks separately retained',
            execution_binding='same_actual_full_job_paths+checkpoint_SHA+writer_UID_namespace+exact_live_issued_action;not_just_shared_clock',
            skipped_ineligible_or_postterminal_diagnostics=skipped,independent_getter_clock_pair_available=False,
            native_backend_source='actual_PhysX_getter;not_command_cache',q_source='actual_TensorWorld_refresh_postphysical_before_terminal_reset',
            named_finger_units='ACTIVE names only;rotational DOF type0/rad;full17DOF contains wrist meters',
            palm_scope='palm_pose_w is legacy actual base_link;named palm_link position unavailable'))


def save_physical_divergence(config_path,output_path):
    """Analyze completed recorded files; no simulator import or control writes.

    Input JSON binds dataset, context, qualified calibration, candidate and
    native_teacher_repeats=[run01,run02,...]. Legacy teacher_repeat names one
    native repeat; it remains compatible and cannot qualify contact alone.
    Each native run names report, audit_summary, policy_trace
    and reset_comparison path/SHA identities. All original input bytes are
    checked again before publishing the exclusive analysis JSON.
    """
    import torch
    from . import tabletop_vision_teacher_tracking_calibration as calibration_module
    from . import linear_coupling
    implementation=[file_identity(path) for path in (__file__,calibration_module.__file__,linear_coupling.__file__)]
    config_id=file_identity(config_path);config=_json(config_id['path']);output=Path(output_path).absolute()
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    if config.get('version')!='dh116-stage6-physical-repeat-analysis-inputs-v1':raise ValueError('explicit physical-repeat input schema required')
    identities=[config_id]
    for name in ('dataset','context','calibration'):identities.append(checked_identity(config[name]))
    if str(output.resolve()) in {row['path'] for row in identities}:raise ValueError('analysis cannot replace immutable input')
    dataset_id,context_id,calibration_id=identities[1:4]
    calibration=_json(calibration_id['path'])
    if (calibration.get('version')!='dh116-teacher-tracking-calibration-v1' or calibration.get('qualified') is not True
            or calibration.get('target_source')!='physx_backend_get_dof_position_targets'
            or calibration.get('feedback_source')!='original_advance_returned_refresh_before_TensorEnv_terminal_reset'
            or calibration.get('dataset_sha256')!=dataset_id['sha256'] or calibration.get('context_sha256')!=context_id['sha256']):raise ValueError('actual pool-bound qualified Teacher backend/post-q calibration required')
    p=np.asarray(calibration.get('teacher_error_p999_rad'),dtype=np.float64);trigger=np.asarray(calibration.get('trigger_rad'),dtype=np.float64)
    if p.shape!=(12,) or not np.isfinite(p).all() or (p<0).any() or not np.array_equal(trigger,np.maximum(.01,1.5*p)):raise ValueError('qualified12joint trigger formula differs')
    if 'native_teacher_repeats' in config:
        if 'teacher_repeat' in config:raise ValueError('use explicit native_teacher_repeats or legacy teacher_repeat,not both')
        repeat_files=config['native_teacher_repeats']
        if not isinstance(repeat_files,list) or not repeat_files:raise ValueError('at least one actual native Teacher-tape reference is required')
    else:repeat_files=[config['teacher_repeat']]
    repeat_ids=[checked_identity(files['report']) for files in repeat_files]
    if len({row['path'] for row in repeat_ids})!=len(repeat_ids):raise ValueError('distinct actual native repeat report files required;copied references do not add samples')
    run_files={f'teacher_repeat_{i}':files for i,files in enumerate(repeat_files)};run_files['candidate']=config['candidate']
    native_reports={name:_json(checked_identity(files['report'])['path']) for name,files in run_files.items()}
    for report in native_reports.values():
        for key in ('checkpoint','teacher_dataset','sequence_context','source_initial_state','source_collection_report'):
            identities.append(checked_identity(report.get('inputs',{}).get(key)))
    for key in ('teacher_dataset','sequence_context','source_initial_state','source_collection_report'):
        if any(report['inputs'][key]!=native_reports['teacher_repeat_0']['inputs'][key] for report in native_reports.values()):raise ValueError('candidate/reference source cold-state or collection lineage differs')
    data=torch.load(dataset_id['path'],map_location='cpu',weights_only=False)
    context=torch.load(context_id['path'],map_location='cpu',weights_only=False)
    original=load_teacher_context(data,context,calibration,provenance=dict(dataset=dataset_id,context=context_id,calibration=calibration_id))
    repeated=[load_native_run(files,dataset_id=dataset_id,context_id=context_id,calibration=calibration) for files in repeat_files]
    if any(run['provenance']['mode']!='teacher-replay' or any(row['action_owner']!='teacher_replay' for row in run['samples']) for run in repeated):raise ValueError('reference repeats must actually be native Teacher-tape owned executions;not Student')
    matches=[i for i,files in enumerate(repeat_files) if files==config['candidate']]
    candidate=(repeated[matches[0]] if matches else load_native_run(config['candidate'],dataset_id=dataset_id,context_id=context_id,calibration=calibration))
    for name,report in native_reports.items():
        contract=report.get('execution_controller_contract',{})
        for key in ('controller_source_sha256','action_contract_sha256'):
            if contract.get(key)!=calibration.get(key):raise ValueError('native '+name+' original controller/action differs from calibrated source')
    envelope=build_repeat_envelope([original,*repeated])
    source_pairs={(key[0],key[1]) for key in envelope['ranges']}
    candidate_pairs={(key[0],key[1]) for key in _index_run(candidate)[0]}
    if candidate_pairs!=source_pairs:raise ValueError('candidate original source episode lineage differs')
    analysis=compare_to_repeat(candidate,envelope,trigger)
    reference_analyses=[compare_to_repeat(run,envelope,trigger) for run in repeated]
    for run in repeated:identities.extend(run['input_identities'])
    identities.extend(candidate['input_identities'])
    identities.extend(implementation);unique={}
    for row in identities:
        if row['path'] in unique and row!=unique[row['path']]:raise ValueError('one input path has conflicting expected SHA256 identities')
        unique[row['path']]=row
    for row in unique.values():checked_identity(row)
    result=dict(version=VERSION,status='completed_read_only_recorded_physics_analysis',analysis=analysis,
        reference_provenance=envelope['reference_provenance'],candidate_provenance=candidate['provenance'],
        reference_absolute_task_events=[dict(env_id=row['env_id'],first_absolute_task_event=row['first_absolute_task_event']) for row in reference_analyses[0]['per_env']],
        native_teacher_repeat_absolute_task_events=[dict(execution_run_uuid=run['execution_run_uuid'],events=[dict(env_id=row['env_id'],first_absolute_task_event=row['first_absolute_task_event']) for row in summary['per_env']]) for run,summary in zip(repeated,reference_analyses)],
        input_identities=list(unique.values()),inputs_sha256_unchanged=True,implementation=implementation,
        statistical_confidence_interval=False,reference_run_count=envelope['reference_run_count'],native_teacher_repeat_count=len(repeated),
        minimum_maximum_scope='finite actually observed recorded native runs only;not independent cases or confidence interval',
        significant_tracking_definition='above actual qualified Teacher warning trigger AND outside observed repeat max;not statistical significance or a new safety limit',
        command_physical_task_event_separated=True,Student_failure_causality_claimed=False,
        named_palm_link_available=False,postphysical_reference_contact_available=analysis['availability']['repeat_contact']['available'],
        native_jobs_executed=0,new_expert_sources=0,new_independent_initial_states=0,control_feedback_used=False,controller_or_physics_parameters_changed=False,
        reset_scope='same original measured cold-reset reference;not a PhysX solver/contact-state checkpoint')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(_plain(result),stream,indent=2,allow_nan=False)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    report=save_physical_divergence(args.config,args.output)
    print(json.dumps(dict(report=file_identity(args.output),status=report['status'],native_jobs_executed=0),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
