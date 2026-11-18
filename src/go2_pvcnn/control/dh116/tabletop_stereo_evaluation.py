"""Episode-local statistics from raw records; CPU schemas never qualify physics."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .tabletop_stereo_contract import STYLES, action_sha256, digest, state_contract, physical_initial_sha256

VERSION='dh116-stereo-evaluation-v1'
FORMAL_TRACKER='surface_lift_success_30mm_5s_0.5mm'
THRESHOLDS=dict(net_lift_m=.03,continuous_hold_s=5.,formal_anchor_slip_m=.0005,physics_hz=480,control_hz=60)
OWNERS=('teacher','mpc','bc','dagger','student')


def _finite(value):return type(value) in (int,float) and math.isfinite(value)


def wilson_interval(successes,total,z=1.959963984540054):
    if type(successes) is not int or type(total) is not int or not 0<=successes<=total:
        raise ValueError('valid binomial counts required')
    if total==0:return None,None
    p=successes/total;den=1+z*z/total
    centre=(p+z*z/(2*total))/den
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return max(0.,centre-radius),min(1.,centre+radius)


def _distribution(values,count):
    valid=[float(x) for x in values if _finite(x)]
    if not valid:return dict(p50=None,p90=None,p95=None,max=None,coverage=0.,count=0)
    return dict(zip(('p50','p90','p95','max'),map(float,[*np.quantile(valid,[.5,.9,.95]),max(valid)])),
                coverage=len(valid)/max(count,1),count=len(valid))


def _key(row):
    return tuple(row.get(k) for k in ('run_uuid','env_id','episode_uid','reset_id'))


def replay_formal_tracker(samples):
    """Replay the unchanged real 60Hz tracker from every 480Hz physical sample.

    Pending first-window flags are not completed evidence. Loads/normals are
    dt-averaged by SurfaceLiftSuccess; pose slip remains the per-window RAW max
    relative to the fixed hold anchor. This CPU arithmetic grants no native scope.
    """
    import torch
    from .tabletop_surface_success import SurfaceLiftSuccess
    mapping={'object_pose':'formal_object_pose_w','wrist_pose':'formal_wrist_pose_w',
             'loads':'formal_pad_loads_n','vectors':'formal_pad_vectors_w','depth':'formal_pad_depth_m',
             'ordinary':'formal_rejected_load_n','bottom':'formal_object_bottom_m',
             'object_velocity':'formal_object_velocity'}
    required=set(mapping.values())|{'formal_reference_bottom_m','formal_task_active'}
    if not samples or any(not required<=row.keys() for row in samples):
        return dict(available=False,event=False,windows=[],reasons=['formal_raw_replay_fields_missing'])
    reference=samples[0]['formal_reference_bottom_m']
    if not _finite(reference):return dict(available=False,event=False,windows=[],reasons=['formal_reference_bottom_NA'])
    tracker=SurfaceLiftSuccess(1,'cpu',reference);windows=[];reasons=[];event=False
    try:
        for row in samples:
            if row['formal_reference_bottom_m']!=reference or type(row['formal_task_active']) is not bool:
                raise ValueError('changing formal reference/active source')
            data={name:torch.as_tensor(np.asarray(row[source],dtype=float),dtype=torch.float64)[None]
                  for name,source in mapping.items()}
            result=tracker.update(**data,valid=torch.tensor([row['formal_task_active']]),dt=1/480)
            event=bool(result['event'][0])
            if row.get('formal_window_completed') is not result['updated']:reasons.append('formal_completed_window_source_mismatch')
            if row.get('formal_window_good') is not bool(tracker.window_good[0]):reasons.append('formal_window_good_source_mismatch')
            if row.get('formal_anchor_latched') is not bool(tracker._latched[0]):reasons.append('formal_anchor_source_mismatch')
            if row.get('formal_drift_source')!='last_completed_window_max_over_raw_pose_points':reasons.append('formal_drift_source_unavailable')
            if not _finite(row.get('formal_hold_s')) or not math.isclose(row['formal_hold_s'],float(tracker.duration[0]),rel_tol=1e-8,abs_tol=1e-10):
                reasons.append('formal_duration_differs_from_raw_replay')
            if result['updated']:
                if row.get('formal_tracker_valid') is not bool(tracker._physical_window_valid[0]):
                    reasons.append('formal_window_physical_validity_source_mismatch')
                if not _finite(row.get('formal_drift_m')) or not math.isclose(row['formal_drift_m'],float(tracker.drift_m[0]),rel_tol=1e-7,abs_tol=1e-10):
                    reasons.append('formal_drift_differs_from_raw_pose_max')
                windows.append(dict(good=bool(tracker.window_good[0]),drift_m=float(tracker.drift_m[0]),
                    duration=float(tracker.duration[0]),paired=tracker._paired[0].tolist(),
                    physical_valid=bool(tracker._physical_window_valid[0])))
    except (ValueError,TypeError,RuntimeError,KeyError) as error:
        return dict(available=False,event=False,windows=windows,reasons=['formal_raw_replay_invalid:'+str(error)])
    return dict(available=True,event=event,windows=windows,reasons=sorted(set(reasons)))


def analyze_episode(rows):
    """Arithmetic/continuity only. Public qualification additionally reopens proofs."""
    headers=[r for r in rows if r.get('kind')=='episode_start']
    samples=[r for r in rows if r.get('kind','sample')=='sample' and r.get('live',True)]
    reasons=[];header=headers[0] if len(headers)==1 else {}
    if len(headers)!=1:reasons.append('missing_or_duplicate_episode_start')
    identity=_key(rows[0]) if rows else (None,)*4
    if any(_key(r)!=identity for r in rows):reasons.append('mixed_episode_identity')
    owner=header.get('owner',rows[0].get('owner') if rows else None)
    style=header.get('grasp_style',rows[0].get('grasp_style') if rows else None)
    if owner not in OWNERS:reasons.append('unknown_action_owner')
    if style not in STYLES:reasons.append('unknown_grasp_style')
    if any(r.get('owner')!=owner or r.get('grasp_style')!=style for r in rows):reasons.append('mixed_owner_or_style')
    try:initial_sha=physical_initial_sha256(header.get('physical_initial_condition'))
    except ValueError:initial_sha=None;reasons.append('missing_measured_physical_initial_condition')
    if header.get('first_episode') is not True or header.get('resumed') is not False or header.get('recovered') is not False:
        reasons.append('not_complete_first_unrecovered_episode')
    if any(r.get('recovered',False) or r.get('resumed',False) for r in samples):reasons.append('recovery_or_resume_in_raw')
    start=header.get('initial_physics_step');when=header.get('initial_timestamp')
    if type(start) is not int or not _finite(when):reasons.append('missing_initial_physics_clock')
    else:
        for i,row in enumerate(samples,1):
            if type(row.get('physics_step')) is not int or row['physics_step']!=start+i:reasons.append('480hz_sample_hole_or_duplicate');break
            if not _finite(row.get('timestamp')) or abs(row['timestamp']-(when+i/480))>1e-7:
                reasons.append('nonmonotonic_or_incorrect_480hz_time');break
    terminal=[i for i,r in enumerate(samples) if r.get('terminal') is True]
    complete=bool(samples) and terminal==[len(samples)-1]
    if not complete:reasons.append('truncated_or_after_first_terminal')
    final=samples[-1] if samples else {}
    task_success=complete and final.get('native_success') is True and final.get('failure_code')==0
    if any(r.get('native_success') is True for r in samples[:-1]):reasons.append('success_before_terminal')
    scope='native' if rows and all(r.get('source_scope')=='native' for r in rows) else 'unqualified'
    if scope!='native':reasons.append('non_native_or_unqualified_source_scope')
    lifts=[r.get('object_net_lift_m') if r.get('measurement_valid') is True else None for r in samples]
    completed=[r for r in samples if r.get('formal_window_completed') is True]
    holds=[r.get('formal_hold_s') if r.get('measurement_valid') is True and r.get('formal_tracker_source')==FORMAL_TRACKER else None for r in completed]
    slips=[r.get('formal_drift_m') if r.get('measurement_valid') is True and r.get('formal_tracker_source')==FORMAL_TRACKER
           and r.get('formal_tracker_valid') is True and r.get('formal_anchor_latched') is True
           and r.get('formal_drift_source')=='last_completed_window_max_over_raw_pose_points' else None for r in completed]
    if any(r.get('formal_tracker_source')!=FORMAL_TRACKER for r in samples):reasons.append('formal_30mm_tracker_source_unavailable')
    if not task_success:reasons.append('native_task_not_successful')
    tail=samples[-2400:]
    if len(tail)<2400:reasons.append('less_than_5s_480hz_raw_hold_evidence')
    replay=replay_formal_tracker(samples)
    reasons.extend(replay['reasons'])
    if replay['available'] and task_success:
        if len(replay['windows'])<300 or not replay['event'] or not all(w['good'] for w in replay['windows'][-300:]):
            reasons.append('formal_300_complete_windows_not_supported_by_raw_replay')
    for row in tail:
        raw_height=(_finite(row.get('formal_object_bottom_m')) and _finite(row.get('formal_reference_bottom_m'))
                    and row['formal_object_bottom_m']>=row['formal_reference_bottom_m']+.03)
        if not (raw_height and row.get('measurement_valid') is True and row.get('formal_task_active') is True):
            reasons.append('raw_hold_lift_formal_slip_contact_not_qualified');break
    if not (_finite(final.get('formal_hold_s')) and final['formal_hold_s']>=5.-1e-12):reasons.append('formal_hold_below_5s_or_NA')
    for row in samples:
        if not (row.get('force_source_qualified') is True and row.get('support_source_qualified') is True
                and _finite(row.get('external_support_force_n')) and abs(row['external_support_force_n'])<=1e-6
                and _finite(row.get('external_upward_force_n')) and abs(row['external_upward_force_n'])<=1e-6
                and row.get('attachment_count')==0):
            reasons.append('support_external_force_or_attachment_source_NA');break
    contact_values=[float(r['bilateral_opposed_contact']) for r in samples if r.get('contact_valid') is True and type(r.get('bilateral_opposed_contact')) is bool]
    return dict(run_uuid=identity[0],env_id=identity[1],episode_uid=identity[2],episode_id=identity[2],reset_id=identity[3],
        owner=owner,grasp_style=style,source_scope=scope,physical_initial_condition_sha256=initial_sha,
        source_episode_group=header.get('source_episode_group',identity[2]),complete=complete,
        first_episode=header.get('first_episode') is True,resumed=header.get('resumed'),recovered=header.get('recovered'),
        sample_count=len(samples),native_task_success=task_success,failure_code=final.get('failure_code'),
        lift_m=_distribution(lifts,len(samples)),hold_s=_distribution(holds,len(completed)),formal_slip_m=_distribution(slips,len(completed)),
        formal_window_count=len(replay['windows']) if replay['available'] else len(completed),formal_replay_event=replay['event'],
        raw_measurement_coverage=sum(r.get('measurement_valid') is True for r in samples)/max(len(samples),1),
        contact_retention=(sum(contact_values)/len(contact_values) if contact_values else None),
        contact_validity_coverage=len(contact_values)/max(len(samples),1),
        qualified_success=False,raw_conditions_satisfied=not reasons,reasons=sorted(set(reasons)))


def aggregate_episodes(episodes,*,provenance=None,evidence_artifacts=None,min_independent_initials=100):
    if type(min_independent_initials) is not int or min_independent_initials<1:raise ValueError('positive initial-condition budget required')
    provenance={} if provenance is None else copy.deepcopy(provenance)
    groups={}
    for e in episodes:groups.setdefault((e['owner'],e['grasp_style']),[]).append(e)
    summaries=[];blocked=[]
    for (owner,style),group in sorted(groups.items(),key=lambda x:str(x[0])):
        complete=[e for e in group if e['complete']]
        # Keep the first observed trial for each actual state, never the best
        # clone/retry. Repeated cloned outcomes remain in descriptive raw counts.
        cohort={}
        for e in complete:
            if e['physical_initial_condition_sha256'] is not None and e['first_episode'] and e['resumed'] is False and e['recovered'] is False:
                cohort.setdefault(e['physical_initial_condition_sha256'],e)
        independent_successes=sum(e['native_task_success'] for e in cohort.values())
        unique={e['physical_initial_condition_sha256'] for e in complete if e['physical_initial_condition_sha256'] is not None}
        native_independent={e['physical_initial_condition_sha256'] for e in complete
                            if e['source_scope']=='native' and e['first_episode'] and e['resumed'] is False and e['recovered'] is False}
        native_independent.discard(None)
        successes=sum(e['native_task_success'] for e in complete);qualified=sum(e['qualified_success'] for e in complete)
        failures={}
        for e in complete:
            if not e['native_task_success']:failures[str(e['failure_code'])]=failures.get(str(e['failure_code']),0)+1
        if len(native_independent)<min_independent_initials:blocked.append(f'{owner}/{style}:insufficient_independent_initials')
        if qualified!=len(complete):blocked.append(f'{owner}/{style}:unqualified_episode_evidence')
        summaries.append(dict(owner=owner,grasp_style=style,episodes=len(group),complete_episodes=len(complete),
            independent_initial_count=len(unique),native_independent_initial_count=len(native_independent),
            native_task_successes=successes,qualified_successes=qualified,
            raw_success_rate=successes/len(complete) if complete else None,
            independent_trials=len(cohort),independent_successes=independent_successes,
            success_rate=independent_successes/len(cohort) if cohort else None,wilson95=wilson_interval(independent_successes,len(cohort)),
            qualified_success_rate=qualified/len(complete) if complete else None,failure_breakdown=failures,
            lift_m=_distribution([e['lift_m']['max'] for e in group],len(group)),
            hold_s=_distribution([e['hold_s']['max'] for e in group],len(group)),
            formal_slip_m=dict(_distribution([e['formal_slip_m']['max'] for e in group],len(group)),
                              coverage=sum(e['formal_slip_m']['coverage'] for e in group)/max(len(group),1)),
            contact_retention=_distribution([e['contact_retention'] for e in group],len(group))))
    if not episodes:blocked.append('no_complete_episodes')
    blocked.append('raw_native_proofs_must_be_verified_by_promotion_gate')
    if provenance.get('source_scope')!='native':blocked.append('source_scope_not_native')
    for key in ('checkpoint_sha256','config_sha256','source_sha256'):
        if not provenance.get(key):blocked.append('missing_'+key)
    return dict(version=VERSION,source_scope=provenance.get('source_scope','unqualified'),
        owner=provenance.get('owner'),checkpoint_sha256=provenance.get('checkpoint_sha256'),
        action_contract_sha256=action_sha256(),state_contract_sha256=digest(state_contract()),
        provenance=provenance,validation_split=provenance.get('validation_split'),
        thresholds=THRESHOLDS.copy(),minimum_independent_initials=min_independent_initials,
        episodes=copy.deepcopy(episodes),summaries=summaries,evidence_artifacts=copy.deepcopy(evidence_artifacts or []),
        qualified_for_promotion=not blocked,blocked_reasons=sorted(set(blocked)))


def evaluate_records(records,*,provenance=None,evidence_artifacts=None,min_independent_initials=100):
    if not isinstance(records,list) or any(not isinstance(r,dict) for r in records):raise ValueError('raw record dictionaries required')
    groups={}
    for row in records:groups.setdefault(_key(row),[]).append(row)
    episodes=[analyze_episode(rows) for rows in groups.values()]
    return aggregate_episodes(episodes,provenance=provenance,evidence_artifacts=evidence_artifacts,min_independent_initials=min_independent_initials)


def training_partition(report,*,base=None,require_complete=True):
    """Reopen registered training raw-header lineage for this actual model.

    Registration proves the contents and union of the declared inventory. It
    cannot independently prove an operator never omitted an unregistered source.
    Native qualification still separately verifies actual rollout evidence.
    """
    binding=report.get('provenance',{}).get('training_manifest_artifact')
    if not isinstance(binding,dict):raise ValueError('SHA-bound training source manifest required for native heldout evaluation')
    try:
        path=Path(binding['path'])
        if not path.is_absolute():path=(Path(base) if base is not None else Path.cwd())/path
        raw=path.read_bytes()
    except (KeyError,TypeError,OSError) as error:raise ValueError('training source manifest artifact unavailable') from error
    if hashlib.sha256(raw).hexdigest()!=binding.get('sha256'):raise ValueError('training source manifest SHA mismatch')
    try:from .tabletop_stereo_training_lineage import load_training_lineage
    except ImportError as error:raise ValueError('training source manifest verifier unavailable') from error
    lineage=load_training_lineage(path,expected_model_checkpoint_sha256=report.get('checkpoint_sha256'),require_complete=require_complete)
    initials=set(lineage['physical_initial_condition_sha256']);groups=set(lineage['source_episode_groups'])
    for name,actual in (('training_initial_sha256',initials),('training_episode_groups',groups)):
        declared=report.get('provenance',{}).get(name)
        if declared is not None and set(declared)!=actual:raise ValueError('declared training partition differs from reopened raw-header manifest: '+name)
    return dict(physical_initial_condition_sha256=initials,source_episode_groups=groups,
                manifest_sha256=binding['sha256'],complete_source_inventory=lineage['complete_source_inventory'])


def select_best_checkpoint(reports,*,training_initial_sha256=(),training_episode_groups=()):
    if not reports:raise ValueError('heldout evaluation reports required')
    candidates=[];cohort=None
    for report in reports:
        if report.get('validation_split')!='heldout':raise ValueError('only independently heldout evaluation selects checkpoints')
        train_initials=set(training_initial_sha256);train_groups=set(training_episode_groups)
        if report.get('source_scope')=='native':
            partition=training_partition(report)
            train_initials.update(partition['physical_initial_condition_sha256']);train_groups.update(partition['source_episode_groups'])
        initials={e.get('physical_initial_condition_sha256') for e in report.get('episodes',[])}
        groups={e.get('source_episode_group') for e in report.get('episodes',[])}
        if None in initials or initials&train_initials or groups&train_groups:
            raise ValueError('training/heldout initial or source episode leakage')
        checkpoint=report.get('checkpoint_sha256')
        if not isinstance(checkpoint,str) or len(checkpoint)!=64:raise ValueError('actual model checkpoint SHA required')
        artifacts=[a for a in report.get('evidence_artifacts',[]) if a.get('kind')=='checkpoint']
        if len(artifacts)!=1:raise ValueError('one actual checkpoint artifact required for selection')
        try:raw=Path(artifacts[0]['path']).read_bytes()
        except (OSError,KeyError,TypeError) as error:raise ValueError('actual checkpoint artifact unavailable') from error
        if artifacts[0].get('sha256')!=checkpoint or hashlib.sha256(raw).hexdigest()!=checkpoint:
            raise ValueError('actual checkpoint artifact SHA mismatch')
        if cohort is not None and initials!=cohort:raise ValueError('checkpoint comparison requires the same heldout initial cohort')
        cohort=initials
        episodes=report['episodes']
        if report.get('source_scope')=='native':
            from .tabletop_stereo_promotion import _report_episodes
            _,episodes,_=_report_episodes(report)
            if {e['physical_initial_condition_sha256'] for e in episodes}!=initials:
                raise ValueError('heldout initial set differs from actual native raw proofs')
            if any(e['physical_initial_condition_sha256'] in train_initials or e.get('source_episode_group') in train_groups for e in episodes):
                raise ValueError('actual native heldout raw-header training leakage')
        distinct={}
        for episode in episodes:distinct.setdefault(episode['physical_initial_condition_sha256'],episode)
        success=sum(e['native_task_success'] for e in distinct.values())
        score=success/max(len(distinct),1)
        physical=False;physical_reasons=['source_scope_not_native']
        if report.get('source_scope')=='native':
            from .tabletop_stereo_promotion import require_teacher_qualification,verify_student_qualification
            verifier={'teacher':require_teacher_qualification,'student':verify_student_qualification}.get(report.get('owner'))
            if verifier is None:physical_reasons=['owner_has_no_Teacher_or_pure_Student_physical_qualification']
            else:
                try:
                    qualified=verifier(report,checkpoint)
                    physical=qualified.get('qualified') is True
                    physical_reasons=[] if physical else ['physical_qualification_not_granted']
                except ValueError as error:physical_reasons=[str(error)]
        candidates.append((score,checkpoint,report,physical,physical_reasons))
    score,checkpoint,report,physical,physical_reasons=max(candidates,key=lambda x:(x[0],x[1]))
    return dict(checkpoint_sha256=checkpoint,heldout_success_rate=score,
                physical_qualified=physical,physical_qualification_reasons=physical_reasons,
                selection_source='independent_heldout_episode_outcomes;not_peak_training_reward')
