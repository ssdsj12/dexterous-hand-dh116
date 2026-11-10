"""Bind Stage7 events to an actual completed Teacher-tape native execution.

PRE_CONTACT precedes the policy tick containing the first observed valid
post-physics bilateral contact. CONTACT keeps the original saved pre-action
predicate. This module starts no simulator, queries no expert and changes no
controller, Actor, cache implementation identity or old event timing.
"""
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from .tabletop_vision_supervision_dataset import validate_dataset, join_sequence_context


VERSION = 'dh116-stage7-actual-event-reference-v1'
COMMON_INPUTS = ('teacher_dataset', 'sequence_context', 'source_collection_report', 'source_initial_state')
INPUTS = ('source_checkpoint', *COMMON_INPUTS, 'native_replay_report')
STATE_FIELDS = ('q', 'qd', 'command', 'roots', 'object_position', 'object_quaternion', 'reference_bottom')


def file_identity(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def _checked(value):
    if not isinstance(value, dict) or not isinstance(value.get('path'), str):
        raise ValueError('explicit immutable path/SHA256 identity required')
    actual = file_identity(value['path'])
    if actual['sha256'] != value.get('sha256'):
        raise ValueError('immutable event-reference input SHA256 changed: ' + value['path'])
    return actual


def _json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict): raise ValueError('event-reference JSON object required')
    return value


def _jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict): raise ValueError('actual native JSONL object required')
            yield value


def _clock(value):
    if not isinstance(value, dict): raise ValueError('actual policy World/SDK clock anchor required')
    result = {key:value.get(key) for key in ('physics_step', 'simulation_step_index', 'timestamp_s')}
    if (any(type(result[key]) is not int or result[key] < 0 for key in ('physics_step', 'simulation_step_index'))
            or type(result['timestamp_s']) not in (int, float) or not math.isfinite(result['timestamp_s'])):
        raise ValueError('finite actual policy World/SDK clock anchor required')
    return result


def _equal_action(value, expected):
    try: actual = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError): return False
    return actual.shape == (24,) and torch.equal(actual, expected.detach().cpu().double())


def _legacy_entries(context, indices, ticks):
    """The original saved pre-action event definitions, without sampler halos."""
    n = len(context['samples']['tick']); events = context.get('events', {})
    def vector(name, boolean=False):
        value = events.get(name)
        if value is None: return None
        if (not isinstance(value,torch.Tensor) or value.shape[0] != n
                or boolean and (value.dtype != torch.bool or value.shape != (n,))):
            raise ValueError('original saved native event vector required: '+name)
        return value[indices]
    def first(mask):
        ids = mask.nonzero().flatten()
        return None if not len(ids) else ticks[int(ids[0])]
    contact = vector('opposed_contact',True); hold = vector('native_hold_qualified',True)
    result = dict(CONTACT=None if contact is None else first(contact),
                  HOLD=None if hold is None else first(hold),LIFT_ONSET=None)
    onset = (vector('lift_onset',True) if context.get('lift_onset_source') in
             ('native_executed_event','native_reference_event') else None)
    if onset is not None:
        result['LIFT_ONSET'] = first(onset)
    elif contact is not None:
        velocity = vector('mean_palm_vz_m_s')
        if velocity is None:
            measured = vector('measured_palm_velocity_m_s')
            if measured is not None:
                if measured.shape != (len(ticks),2,3): raise ValueError('actual two-palm native velocity shape required')
                velocity = measured[:,:,2].mean(-1)
        if velocity is not None:
            if (velocity.shape != (len(ticks),) or not velocity.dtype.is_floating_point
                    or not bool(torch.isfinite(velocity).all())):
                raise ValueError('actual finite measured palm velocity required')
            for index in range(len(ticks)-2):
                if bool(contact[index:index+3].all() and (velocity[index:index+3]>.0005).all()):
                    result['LIFT_ONSET'] = ticks[index]; break
    return result


def derive_event_reference(data, context, replay_report, raw_rows, policy_rows, *, dataset_sha256):
    """Derive facts from real saved rows; synthetic CPU fixtures prove schema only.

    Every source row is joined by its original four-part identity. All native
    post samples are joined to actual issued Teacher actions, execution UIDs,
    policy clock anchors and owned first terminals. A missing contact prefix
    cannot certify a first-contact time. No clamping invents a pre-contact row.
    """
    validate_dataset(data)
    joined = join_sequence_context(data, context, dataset_sha256=dataset_sha256)
    if not bool(joined['row_valid'].all()): raise ValueError('complete original sequence identity join required')
    dt = context['physics_context']['dt_s'].double()
    if not bool(torch.isclose(dt, torch.full_like(dt, 1/480), rtol=0., atol=1e-10).all()):
        raise ValueError('original eight-substep480Hz source context required')
    report = replay_report
    if (report.get('version') != 'dh116-stage6-native-v1' or report.get('mode') != 'teacher-replay'
            or report.get('status') != 'completed' or report.get('error') is not None
            or report.get('native_runtime_verified') is not True or report.get('ppo_updates') != 0
            or report.get('evaluated_source_count') != 4 or report.get('completed_count') != 4
            or report.get('pure_student_native_success_count') != 0):
        raise ValueError('completed actual four-first-terminal Teacher-replay execution required')
    run = report.get('execution_run_uuid')
    if not isinstance(run, str) or not run: raise ValueError('actual execution UUID required')
    cases = {r['env_id']:r for r in report.get('source_case_definitions', [])}
    outcomes = {r['env_id']:r for r in report.get('source_rows', [])}
    if set(cases) != set(range(4)) or set(outcomes) != set(range(4)) or len(data['episodes']) != 4:
        raise ValueError('exactly four original evaluated source identities required')
    contact = context.get('events', {}).get('opposed_contact')
    if contact is not None and (not isinstance(contact, torch.Tensor) or contact.dtype != torch.bool
            or contact.shape != (len(context['samples']['tick']),)):
        raise ValueError('original boolean pre-action contact vector required')
    sources = {}; rows_by_tick = {}; legacy_events = {}
    for env in range(4):
        case = cases[env]; selected = [(i,r) for i,r in enumerate(data['sample_metadata']) if r['env_id'] == env]
        if not selected: raise ValueError('actual source episode rows missing')
        first = selected[0][1]; episode = data['episodes'][str(first['episode_id'])]
        if (episode.get('grasp_style') != 'pinch_thumb_index' or str(first['episode_id']) != str(case.get('episode_id'))
                or case.get('source_episode_uid') != first['episode_uid']
                or case.get('source_run_uuid') != first['run_uuid']
                or case.get('recorded_teacher_tape_length') != len(selected)
                or any(r['episode_uid'] != first['episode_uid'] or r['run_uuid'] != first['run_uuid'] for _,r in selected)):
            raise ValueError('original source/native replay episode lineage differs')
        sources[env] = dict(run_uuid=first['run_uuid'], env_id=env, episode_uid=first['episode_uid'],
                            episode_id=first['episode_id'])
        for i,row in selected: rows_by_tick[env,row['tick']] = (i,row)
        indices = joined['context_indices'][[i for i,_ in selected]]
        legacy_events[env] = _legacy_entries(context,indices,[r['tick'] for _,r in selected])
        if any(case.get('event_entry_ticks',{}).get(name) != tick for name,tick in legacy_events[env].items()):
            raise ValueError('saved pre-action event differs from actual replay source-case definition')
        outcome = outcomes[env]
        if (outcome.get('execution_episode_uid') != f"{run}:{env}:{outcome.get('execution_episode_id')}"
                or outcome.get('student_policy_steps_executed') != 0
                or outcome.get('teacher_prefix_policy_steps_executed') != 0
                or type(outcome.get('teacher_replay_policy_steps_executed')) is not int
                or outcome['teacher_replay_policy_steps_executed'] < 1
                or not isinstance(outcome.get('terminal'), dict)):
            raise ValueError('actual Teacher-owned first-episode identity required')
    policies = {}; anchors = {}; offsets = set()
    for row in policy_rows:
        env, tick = row.get('env_id'), row.get('policy_step')
        if env not in sources or type(tick) is not int or (env,tick) in policies or (env,tick) not in rows_by_tick:
            raise ValueError('duplicate/unknown actual policy source identity')
        i,meta = rows_by_tick[env,tick]; outcome = outcomes[env]
        if (row.get('run_uuid') != run or row.get('episode_uid') != outcome['execution_episode_uid']
                or row.get('episode_id') != outcome['execution_episode_id']
                or row.get('recorded_teacher_episode_uid') != meta['episode_uid'] or row.get('owner') != 'teacher_replay'
                or not _equal_action(row.get('issued_action'), data['actions'][i])
                or not _equal_action(row.get('recorded_teacher_action'), data['actions'][i])
                or not _equal_action(row.get('actual_previous_issued_action'), torch.tensor(meta['previous_action'], dtype=torch.float64))):
            raise ValueError('actual issued Teacher tape/previous action or execution UID differs')
        anchor = _clock(row.get('clock_before_action'))
        if tick in anchors and anchors[tick] != anchor: raise ValueError('one native policy tick has inconsistent World/SDK clocks')
        anchors[tick] = anchor; offsets.add(anchor['simulation_step_index'] - anchor['physics_step'])
        policies[env,tick] = row
    if len(offsets) != 1: raise ValueError('actual World/SDK offset changed')
    for env in range(4):
        ticks = sorted(t for e,t in policies if e == env)
        if ticks != list(range(outcomes[env]['teacher_replay_policy_steps_executed'])):
            raise ValueError('missing actual issued Teacher policy prefix')
        for before,after in zip(ticks,ticks[1:]):
            a,b = anchors[before],anchors[after]
            if (b['physics_step']-a['physics_step'] != 8 or b['simulation_step_index']-a['simulation_step_index'] != 8
                    or b['timestamp_s'] <= a['timestamp_s']): raise ValueError('original eight-substep policy clock continuity required')
    first_contact = {}; closed = set(); last_step = {env:anchors[0]['physics_step'] for env in sources}
    last_time = {env:anchors[0]['timestamp_s'] for env in sources}; seen_policies = set(); count = 0
    for row in raw_rows:
        env, tick = row.get('env_id'), row.get('policy_step'); payload = row.get('payload', {})
        if env not in sources or env in closed or (env,tick) not in policies:
            raise ValueError('raw sample outside the actual owned first episode')
        policy = policies[env,tick]; anchor = anchors[tick]; step = row.get('physics_step'); when = row.get('timestamp_s')
        if (row.get('episode_uid') != policy['episode_uid'] or row.get('action_owner') != 'teacher_replay'
                or str(row.get('source_episode_id')) != str(cases[env]['episode_id'])
                or type(step) is not int or step != last_step[env]+1 or not 1 <= step-anchor['physics_step'] <= 8
                or type(when) not in (int,float) or not math.isfinite(when) or when <= last_time[env]
                or when <= anchor['timestamp_s'] or payload.get('live') is not True
                or type(payload.get('done')) is not bool):
            raise ValueError('actual native source/UID/substep clock or live prefix mismatch')
        substep = step-anchor['physics_step']; next_anchor = anchors.get(tick+1)
        if next_anchor is not None and (when > next_anchor['timestamp_s'] or
                substep == 8 and when != next_anchor['timestamp_s']):
            raise ValueError('postphysical time does not meet the actual next policy clock')
        last_step[env] = step; last_time[env] = when; seen_policies.add((env,tick)); count += 1
        if env not in first_contact:
            if payload.get('measurement_valid') is not True or payload.get('contact_valid') is not True or type(payload.get('bilateral_opposed_contact')) is not bool:
                raise ValueError('missing actual contact observation before proposed first contact')
            if payload['bilateral_opposed_contact']:
                first_contact[env] = dict(policy_step=tick, substep=substep, world_physics_step=step,
                    timestamp_s=when, execution_run_uuid=run, execution_episode_uid=row['episode_uid'],
                    source_sample_key=[sources[env]['run_uuid'],env,sources[env]['episode_uid'],tick],
                    policy_clock_anchor=deepcopy(anchor), derived_sdk_physics_step=anchor['simulation_step_index']+substep,
                    sdk_post_index_is_derived=True, independent_getter_clock_pair_available=False,
                    source='actual_live_valid_postphysics_bilateral_opposed_contact;not_prediction')
        if payload['done']:
            terminal = outcomes[env]['terminal']
            if (terminal.get('policy_step'),terminal.get('physics_step'),terminal.get('timestamp_s'),terminal.get('failure_code'),terminal.get('action_owner')) != (
                    tick,step,when,row.get('failure_code'),'teacher_replay'):
                raise ValueError('actual owned first terminal differs from native report')
            if payload.get('native_success') is not outcomes[env].get('native_success'):
                raise ValueError('native first-terminal success differs from report')
            closed.add(env)
    if closed != set(range(4)) or seen_policies != set(policies):
        raise ValueError('complete actual raw first terminals and issued-policy coverage required')
    result = []
    for env in range(4):
        event = first_contact.get(env); initial_contact = legacy_events[env]['CONTACT'] == 0
        tick = None if event is None or event['policy_step'] == 0 or initial_contact else event['policy_step']-1
        previous = None
        if tick is not None:
            if (env,tick) not in rows_by_tick: raise ValueError('actual preceding source policy row missing')
            _,meta = rows_by_tick[env,tick]
            if meta['terminal']: raise ValueError('PRE_CONTACT cannot follow an original source terminal')
            previous = {k:meta[k] for k in ('run_uuid','env_id','episode_uid','tick','policy_timestamp_s','first_physics_step')}
        reason = ('contact_already_established_at_initial_pre_action_boundary' if initial_contact else
                  'no_valid_actual_post_contact_before_first_terminal' if event is None else
                  'first_actual_post_contact_at_tick_zero' if tick is None else None)
        events = {name:dict(tick=entry,available=entry is not None,source=description)
            for name,entry,description in (
                ('CONTACT',legacy_events[env]['CONTACT'],'original_saved_pre_action_opposed_contact;already_established_at_available_policy_boundary'),
                ('LIFT_ONSET',legacy_events[env]['LIFT_ONSET'],'original_saved_native_opposed_and_three_measured_mean_palm_vz_above_0.0005_or_declared_native_onset'),
                ('HOLD',legacy_events[env]['HOLD'],'original_saved_native_hold_qualified'))}
        events['PRE_CONTACT'] = dict(tick=tick,available=tick is not None,unavailable_reason=reason,
            first_actual_post_contact=event,preceding_source_row=previous)
        result.append(dict(source=sources[env],events=events))
    return dict(cases=result, reference_execution_run_uuid=run, actual_raw_samples=count,
        reference_mode='teacher-replay', evaluated_source_count=4, actor_prediction_used=False,
        contact_prediction_used=False, new_independent_initial_states=0, native_success_certificate_issued=False)


def _reset_proof(value, *, warmup=False):
    if (value.get('matched') is not True or value.get('tolerance') != 0.
            or any(value.get('fields', {}).get(name, {}).get(flag) is not True
                for name in STATE_FIELDS for flag in ('matched','dtype_equal','shape_equal','finite'))):
        raise ValueError('complete exact actual source reset/warmup evidence required')
    if warmup and value.get('before_clock') != value.get('after_clock'):
        raise ValueError('actual camera warmup changed the source clock')


def _file_manifest(inputs):
    if not isinstance(inputs, dict) or set(inputs) != set(INPUTS): raise ValueError('six explicit Stage7 source/reference inputs required')
    bound = {name:_checked(inputs[name]) for name in INPUTS}
    replay = _json(bound['native_replay_report']['path']); parent = Path(bound['native_replay_report']['path']).parent
    for reference,runtime in (('source_checkpoint','checkpoint'),*((k,k) for k in COMMON_INPUTS)):
        if _checked(replay.get('inputs', {}).get(runtime)) != bound[reference]:
            raise ValueError('actual Teacher replay belongs to different source/common inputs')
    collection = _json(bound['source_collection_report']['path'])
    for name,key in (('successful_dataset','teacher_dataset'),('successful_sequence_context','sequence_context')):
        if _checked(collection.get(name)) != bound[key]: raise ValueError('original successful source files differ from collection report')
    if collection.get('native_runtime_verified') is not True: raise ValueError('original genuine native Teacher collection required')
    artifacts = dict(raw_physics=_checked(replay.get('raw_physics')),
        policy_trace=file_identity(parent/'live_policy_60hz.jsonl.gz'),
        reset_comparison=file_identity(parent/'source_reset_comparison.json'),
        warmup_comparison=file_identity(parent/'source_camera_warmup_comparison.json'))
    if Path(artifacts['raw_physics']['path']) != parent/'source_physics_480hz.jsonl.gz':
        raise ValueError('native reference raw physics must belong to the same actual job directory')
    _reset_proof(_json(artifacts['reset_comparison']['path']))
    _reset_proof(_json(artifacts['warmup_comparison']['path']),warmup=True)
    data = torch.load(bound['teacher_dataset']['path'], map_location='cpu', weights_only=False)
    context = torch.load(bound['sequence_context']['path'], map_location='cpu', weights_only=False)
    if any(collection.get('episodes', {}).get(key) != episode for key,episode in data['episodes'].items()):
        raise ValueError('original successful episode proof differs from collection report')
    facts = derive_event_reference(data,context,replay,_jsonl(artifacts['raw_physics']['path']),
        _jsonl(artifacts['policy_trace']['path']),dataset_sha256=bound['teacher_dataset']['sha256'])
    result = dict(version=VERSION,inputs=bound,artifacts=artifacts,implementation=file_identity(__file__),**facts)
    for identity in (*bound.values(),*artifacts.values(),result['implementation']): _checked(identity)
    return result


def build_event_reference(inputs, output):
    """Exclusively save a reusable binding manifest; never start native work."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    result = _file_manifest(inputs)
    if str(output.resolve()) in {v['path'] for v in (*result['inputs'].values(),*result['artifacts'].values())}:
        raise ValueError('cannot overwrite immutable native/source input')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    return file_identity(output)


def load_event_reference(identity, expected_inputs, *, required_event=None):
    """Revalidate original bytes and event facts; never alias comparison hashes.

    Runtime callers require the four COMMON_INPUTS. The reference c58 source
    checkpoint may be checked separately; a B0/B1/B2 candidate checkpoint is
    deliberately not one of those four common bindings.
    """
    actual = _checked(identity); value = _json(actual['path'])
    if value.get('version') != VERSION: raise ValueError('explicit Stage7 event-reference version required')
    if (not isinstance(expected_inputs,dict) or not set(COMMON_INPUTS) <= set(expected_inputs)
            or not set(expected_inputs) <= set(INPUTS)):
        raise ValueError('four exact common source inputs required;candidate checkpoint is not a reference checkpoint')
    for key,expected in expected_inputs.items():
        if _checked(expected) != value.get('inputs',{}).get(key): raise ValueError('event reference belongs to different expected source inputs')
    original_implementation = _checked(value.get('implementation'))
    rebuilt = _file_manifest(value.get('inputs'))
    if original_implementation['sha256'] != rebuilt['implementation']['sha256']:
        raise ValueError('event-reference implementation bytes changed;no SHA256 alias allowed')
    # Separate the producer's immutable path provenance from the same bytes
    # executing in a newly frozen per-job capsule. Never rewrite its SHA.
    rebuilt['implementation'] = original_implementation
    if value != rebuilt:
        raise ValueError('event-reference facts or implementation differ from actual immutable original evidence')
    if required_event not in (None,'PRE_CONTACT','CONTACT','LIFT_ONSET','HOLD'): raise ValueError('unknown Stage7 event name')
    if required_event is not None and any(c['events'][required_event]['available'] is not True for c in value['cases']):
        raise ValueError('requested event is unavailable for one or more actual source episodes')
    _checked(actual); _checked(original_implementation)
    return value
