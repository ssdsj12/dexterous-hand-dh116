"""Stage6 source replication and reset-replayed Teacher→Student branches.

The four qualified pinch rows are source cases, never heldout trials. The other
four original mixed-template rows receive neutral actions and are unevaluated.
No restored pose is represented as a PhysX solver checkpoint. All privileged
reads below are diagnostics; VisionInference reads only the real 3844D policy.
"""
from copy import copy
import gzip
import json
import math
from pathlib import Path
import uuid

import torch

from .tabletop_vision_evaluation_contract import file_sha256


VERSION = 'dh116-stage6-native-v1'
INPUT_NAMES = ('checkpoint', 'teacher_dataset', 'sequence_context',
               'source_initial_state', 'source_collection_report')
SOURCE_STATE_FIELDS = ('q', 'qd', 'command', 'roots', 'object_position',
                       'object_quaternion', 'reference_bottom')
ACTION_GROUPS = dict(fingers=list(range(12)), left_thumb=[0, 1], left_index=[2],
    right_thumb=[6, 7], right_index=[8], left_palm_xyz=[12, 13, 14],
    right_palm_xyz=[15, 16, 17], left_wrist_rotation=[18, 19, 20],
    right_wrist_rotation=[21, 22, 23])


def comparison_dependencies():
    """Actual numerical source bytes, independent of the capsule directory."""
    names = ('tabletop_vision_branching_eval', 'tabletop_vision_inference',
        'tabletop_vision_student', 'tabletop_vision_point_activation', 'tabletop_vision_quaternion',
        'tabletop_vision_command_shadow', 'tabletop_command_kernels', 'tabletop_free_wrist_grasp',
        'tabletop_tensor_world', 'tabletop_surface_contract', 'tabletop_velocity_limits',
        'linear_coupling', 'tabletop_vision_action_contract', 'tabletop_vision_stage7_event_reference')
    return {name: file_sha256(Path(__file__).with_name(name + '.py')) for name in names}


def _plain(value):
    if isinstance(value, torch.Tensor): return _plain(value.detach().cpu().tolist())
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def _json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(_plain(value), stream, indent=2, allow_nan=False)


def _write_line(stream, value):
    stream.write(json.dumps(_plain(value), separators=(',', ':'), allow_nan=False) + '\n')


def verify_inputs(config):
    names = INPUT_NAMES + tuple(name for name in ('same_state_comparison_cache','event_reference') if name in config)
    for name in names:
        identity = config[name]
        if file_sha256(identity['path']) != identity['sha256']:
            raise ValueError('immutable Stage6 input SHA256 changed: ' + name)
    return True


def load_native_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('version') != VERSION or value.get('mode') not in ('source', 'branch', 'teacher-replay'):
        raise ValueError('explicit Stage6 source/branch/teacher-replay config required; no PPO')
    if value.get('initial_state_tolerance', 0.) != 0.:
        raise ValueError('source reset uses exact readback; nonzero tolerance requires a separate reviewed contract')
    if value['mode'] == 'branch':
        if value.get('event') not in ('PRE_CONTACT', 'CONTACT', 'LIFT_ONSET', 'HOLD'):
            raise ValueError('recorded CONTACT/LIFT_ONSET/HOLD or bound Stage7 PRE_CONTACT event required')
        if value['event'] == 'PRE_CONTACT' and 'event_reference' not in value:
            raise ValueError('actual postphysics event reference required for PRE_CONTACT')
        if type(value.get('branch_steps')) is not int or value['branch_steps'] not in (1, 4, 16, 60):
            raise ValueError('Stage6 branch_steps must be 1, 4, 16 or 60')
    if 'event_reference' in value and value['mode'] != 'branch':
        raise ValueError('Stage7 event reference belongs only to an explicit diagnostic branch')
    names = INPUT_NAMES + tuple(name for name in ('same_state_comparison_cache','event_reference') if name in value)
    for name in names:
        identity = value.get(name)
        if (not isinstance(identity, dict) or not isinstance(identity.get('path'), str)
                or not identity['path'] or not isinstance(identity.get('sha256'), str)
                or len(identity['sha256']) != 64):
            raise ValueError('explicit path/SHA256 identity required: ' + name)
        actual = Path(identity['path'])
        if not actual.is_absolute(): actual = path.parent / actual
        value[name] = dict(path=str(actual.resolve()), sha256=identity['sha256'].lower())
    verify_inputs(value)
    if 'event_reference' in value:
        _load_event_reference(value, required_event=value['event'])
    value['config_identity'] = dict(path=str(path), sha256=file_sha256(path))
    return value


def _load_event_reference(config, *, required_event=None):
    from .tabletop_vision_stage7_event_reference import load_event_reference, COMMON_INPUTS
    return load_event_reference(config['event_reference'],
        {name: config[name] for name in COMMON_INPUTS}, required_event=required_event)


def apply_event_reference(cases, config):
    """Attach separately proved diagnostic entry points without rewriting data.

    PRE_CONTACT is one policy tick before the first actual postphysics contact,
    whereas CONTACT remains the original observed pre-action boundary. No
    predicate, timestamp or event reaches the Actor, controller or reward.
    """
    if 'event_reference' not in config:
        return cases
    facts = _load_event_reference(config, required_event=config.get('event'))
    lookup = {}
    for row in facts['cases']:
        source = row['source']; key = tuple(source[name] for name in ('run_uuid','env_id','episode_uid'))
        if key in lookup: raise ValueError('duplicate Stage7 event source identity')
        lookup[key] = row
    keys = {tuple(case[name] for name in ('run_uuid','env_id','episode_uid')) for case in cases}
    if set(lookup) != keys: raise ValueError('event reference differs from actual source identities')
    result = []
    for case in cases:
        key = tuple(case[name] for name in ('run_uuid','env_id','episode_uid'))
        events = lookup[key]['events']
        for name in ('CONTACT','LIFT_ONSET','HOLD'):
            if events.get(name, {}).get('tick') != case['event_ticks'][name]:
                raise ValueError('event reference cannot change original saved event: ' + name)
        pre = events['PRE_CONTACT']; tick = pre.get('tick')
        if pre.get('available') is True:
            observed = pre.get('first_actual_post_contact', {}).get('policy_step')
            if (type(tick) is not int or tick < 0 or tick >= len(case['actions'])
                    or type(observed) is not int or observed <= 0 or tick != observed - 1):
                raise ValueError('PRE_CONTACT requires an actual preceding source policy row')
        elif tick is not None:
            raise ValueError('unavailable PRE_CONTACT cannot name an execution tick')
        updated = dict(case, event_ticks=dict(case['event_ticks'], PRE_CONTACT=tick),
            event_sources=dict(case['event_sources'], PRE_CONTACT=dict(
                source='actual_postphysics_reference_preceding_policy_row',
                reference=config['event_reference'], facts=pre)))
        result.append(updated)
    return result


def event_entry_ticks(events, *, trusted_lift_onset=False):
    """Use exact pre-action entry rows, not expanded sampler event windows."""
    result = dict(CONTACT=None, LIFT_ONSET=None, HOLD=None)
    contact = events.get('opposed_contact')
    hold = events.get('native_hold_qualified')
    def first(mask):
        ids = torch.as_tensor(mask, dtype=torch.bool).nonzero().flatten()
        return None if not len(ids) else int(ids[0])
    if contact is not None:
        contact = torch.as_tensor(contact, dtype=torch.bool)
        # A true predicate at the actual cold reset is already its entry.
        result['CONTACT'] = first(contact)
    if hold is not None: result['HOLD'] = first(hold)
    if trusted_lift_onset and events.get('lift_onset') is not None:
        result['LIFT_ONSET'] = first(events['lift_onset'])
    elif contact is not None:
        velocity = events.get('mean_palm_vz_m_s')
        if velocity is None and events.get('measured_palm_velocity_m_s') is not None:
            velocity = events['measured_palm_velocity_m_s'][:, :, 2].mean(-1)
        if velocity is not None:
            velocity = torch.as_tensor(velocity)
            for tick in range(len(contact) - 2):
                if bool(contact[tick:tick+3].all() and (velocity[tick:tick+3] > .0005).all()):
                    result['LIFT_ONSET'] = tick
                    break
    return result


def source_cases(data, context, collection_report, *, dataset_sha256):
    """Require the actual four complete, successful native first episodes."""
    from .tabletop_vision_supervision_dataset import validate_dataset, join_sequence_context
    validate_dataset(data)
    joined = join_sequence_context(data, context, dataset_sha256=dataset_sha256)
    dt = context['physics_context']['dt_s']
    if not bool(torch.isclose(dt.double(), torch.full_like(dt.double(), 1/480), rtol=0., atol=1e-10).all()):
        raise ValueError('source tape requires recorded eight-substep480Hz physics and60Hz policy clock')
    if collection_report.get('native_runtime_verified') is not True:
        raise ValueError('source collection must have qualified native runtime provenance')
    cases = []
    for episode_id, episode in data['episodes'].items():
        if episode.get('grasp_style') != 'pinch_thumb_index':
            raise ValueError('source pool must contain only the four successful pinch cases')
        env_id = episode['env_id']
        proof = collection_report.get('episodes', {}).get(str(episode_id))
        for record in (episode, proof):
            if (not isinstance(record, dict) or record.get('env_id') != env_id
                    or record.get('first_episode') is not True or record.get('success') is not True
                    or record.get('native_physics') is not True or record.get('recovered') is not False
                    or record.get('grasp_style') != 'pinch_thumb_index'):
                raise ValueError('source episode is not qualified in the actual collection report')
        if episode.get('terminal') != proof.get('terminal'):
            raise ValueError('source terminal identity differs from collection report')
        rows = [i for i, row in enumerate(data['sample_metadata']) if str(row['episode_id']) == str(episode_id)]
        for before, after in zip(rows, rows[1:]):
            old, new = data['sample_metadata'][before], data['sample_metadata'][after]
            if (new['first_physics_step'] - old['first_physics_step'] != 8
                    or abs(new['policy_timestamp_s'] - old['policy_timestamp_s'] - 1/60) > 1e-6):
                raise ValueError('continuous original60Hz source action clock required')
        terminal = episode.get('terminal')
        if (not isinstance(terminal, dict) or terminal.get('control_steps') != len(rows)
                or terminal.get('success') is not True or terminal.get('failure') != 0
                or terminal.get('recovered') is not False):
            raise ValueError('complete actually issued source tape through first successful terminal required')
        if not bool(joined['row_valid'][rows].all()):
            raise ValueError('source tape has missing or mismatched native sequence contexts')
        indices = joined['context_indices'][rows]
        events = {key: value[indices] for key, value in context.get('events', {}).items()}
        event_sources = dict(contact='recorded_pre_action_opposed_contact',
            velocity='recorded_pre_action_measured_palm_velocity', hold='recorded_native_hold_qualified')
        if 'opposed_contact' not in events:
            from .tabletop_surface_force_sharing import surface_targets
            raw = {}
            for packet in context.get('raw_measured_packets', []):
                key = tuple(packet['sample_key'])
                if key in raw: raise ValueError('duplicate raw precontrol measurement identity')
                raw[key] = packet
            paired = []
            for index in rows:
                meta = data['sample_metadata'][index]
                key = tuple(meta[name] for name in ('run_uuid', 'env_id', 'episode_uid', 'tick'))
                packet = raw.get(key)
                if packet is None: break
                measured = packet['physics_substeps'][0]
                names = ('effective_pad_loads_n', 'effective_pad_vectors_w', 'effective_pad_valid')
                if not all(name in measured for name in names): break
                loads, normals, valid = (measured[name] for name in names)
                value = surface_targets(loads, normals, valid=valid)['paired'].all(-1)
                if value.shape != (1,): raise ValueError('raw precontrol packet must identify exactly one source environment')
                paired.append(bool(value[0]))
            if len(paired) == len(rows):
                events['opposed_contact'] = torch.tensor(paired, dtype=torch.bool)
                event_sources['contact'] = 'identity_joined_raw_precontrol_effective_pad_measurements'
            else: event_sources['contact'] = 'unavailable;not_inferred_from_phase_or_actions'
        if 'measured_palm_velocity_m_s' not in events and 'mean_palm_vz_m_s' not in events:
            # Roots were read before the actual action at each policy boundary.
            # This backward derivative uses no future action or command target.
            positions = context['physics_context']['measured_root_pose'][indices, 0, :, :3]
            dt = context['physics_context']['dt_s'][indices[:-1]].sum(-1).to(positions)
            velocity = torch.zeros_like(positions)
            velocity[1:] = (positions[1:] - positions[:-1]) / dt[:, None, None]
            events['measured_palm_velocity_m_s'] = velocity
            event_sources['velocity'] = 'backward_difference_recorded_pre_action_measured_root_pose'
            event_sources['velocity_first_row'] = 'no_prior_measured_pose;excluded_from_lift_onset'
        ticks = event_entry_ticks(events, trusted_lift_onset=context.get('lift_onset_source') in
            ('native_executed_event', 'native_reference_event'))
        cases.append(dict(env_id=env_id, episode_id=str(episode_id),
            episode_uid=data['sample_metadata'][rows[0]]['episode_uid'],
            run_uuid=data['sample_metadata'][rows[0]]['run_uuid'], rows=rows,
            context_indices=indices, actions=data['actions'][rows].detach().cpu().clone(),
            event_ticks=ticks, event_sources=event_sources, source_trajectory_id=episode['source_trajectory_id'],
            source_group_id=episode.get('source_group_id', episode['source_trajectory_id'])))
    cases.sort(key=lambda case: case['env_id'])
    if [case['env_id'] for case in cases] != [0, 1, 2, 3]:
        raise ValueError('exactly actual successful source environments 0..3 required')
    return cases


def scheduled_actions(cases, config, tick, student_action, ended):
    """No source exhaustion reuse, Teacher recovery, or post-horizon fallback."""
    if student_action.shape != (8, 24) or not bool(torch.isfinite(student_action).all()):
        raise ValueError('original finite [8,24] SDK action required')
    action = torch.zeros_like(student_action)
    owners = ['unevaluated'] * 8
    for case in cases:
        i = case['env_id']
        if bool(ended[i]): owners[i] = 'first_terminal_closed'; continue
        if config['mode'] == 'source':
            action[i] = student_action[i]; owners[i] = 'student'
        elif config['mode'] == 'teacher-replay':
            if tick < len(case['actions']):
                action[i] = case['actions'][tick].to(action); owners[i] = 'teacher_replay'
            else: owners[i] = 'tape_exhausted'
        else:
            switch = case['event_ticks'][config['event']]
            if switch is None: raise ValueError('source has no recorded pre-action entry for ' + config['event'])
            if tick < switch:
                action[i] = case['actions'][tick].to(action); owners[i] = 'teacher_prefix'
            elif tick < switch + config['branch_steps']:
                action[i] = student_action[i]; owners[i] = 'student'
            else: owners[i] = 'branch_complete'
    return action, owners


def action_group_errors(student, teacher):
    if student.shape != (24,) or teacher.shape != (24,): raise ValueError('SDK24 vectors required')
    delta = (student.detach().cpu().double() - teacher.detach().cpu().double())
    return {name: dict(mse=float(delta[ids].square().mean()), mean_abs=float(delta[ids].abs().mean()),
                       max_abs=float(delta[ids].abs().max())) for name, ids in ACTION_GROUPS.items()}


def compare_source_reset(expected, actual):
    """Keep byte/chart identity separate from quaternion physical equivalence."""
    result = dict(matched=True, tolerance=0., fields={}, quaternion_physical_equivalence={})
    for name in SOURCE_STATE_FIELDS:
        a, b = expected.get(name), actual.get(name)
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError('complete source initial state required: ' + name)
        shape = a.shape == b.shape
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        equal = shape and a.dtype == b.dtype and finite and torch.equal(a.cpu(), b.cpu())
        error = float((a.cpu().double() - b.cpu().double()).abs().max()) if shape and finite else None
        result['fields'][name] = dict(matched=equal, source_dtype=str(a.dtype), actual_dtype=str(b.dtype),
            source_shape=list(a.shape), actual_shape=list(b.shape), dtype_equal=a.dtype == b.dtype,
            shape_equal=shape, finite=finite, max_abs_error=error)
        result['matched'] &= equal
        if shape and finite and name in ('roots', 'object_quaternion'):
            qa, qb = (a[..., 3:7], b[..., 3:7]) if name == 'roots' else (a, b)
            same = (qa.double() - qb.double()).abs().amax(-1)
            opposite = (qa.double() + qb.double()).abs().amax(-1)
            result['quaternion_physical_equivalence'][name] = dict(
                max_sign_equivalent_component_error=float(torch.minimum(same, opposite).max()),
                changes_raw_identity_acceptance=False)
    return result


def _reset_snapshot(device_world, task):
    values = (device_world.q, device_world.qd, task.command, device_world.roots,
              device_world.wp, device_world.wq, task.surface_reference_bottom_m)
    return {key: value.detach().cpu().clone() for key, value in zip(SOURCE_STATE_FIELDS, values)}


def _clock(device_world):
    sim = getattr(getattr(device_world, 'w', None), 'sim', None)
    return dict(physics_step=getattr(device_world, '_physics_steps', None),
        simulation_step_index=getattr(sim, 'current_time_step_index', None),
        timestamp_s=getattr(sim, 'current_time', None))


class SourceTelemetryCollector:
    """Consume each live 480Hz payload before reset; freeze its first terminal."""
    def __init__(self, cases, n, *, native, sink):
        self.cases, self.n, self.native, self.sink = cases, n, native, sink
        self.execution_run_uuid = str(uuid.uuid4())
        self.rows = {case['env_id']: dict(env_id=case['env_id'], episode_id=case['episode_id'],
            observed_physics_samples=0, terminal=None, first_task_related_physical_event=None,
            first_numeric_difference=None,
            first_prefix_physical_event=None, max_net_lift_m=None, max_qualified_hold_s=None,
            student_policy_steps_executed=0, teacher_prefix_policy_steps_executed=0,
            teacher_replay_policy_steps_executed=0,
            observed_native_success_flag=False, native_success=False) for case in cases}
        self.closed = torch.zeros(n, dtype=torch.bool)
        self.previous_contact = {}; self.begun = False
        self._seen_policy_steps = set()
        self.policy_step = -1; self.owners = ['inactive'] * n

    def begin(self, *, reset_clock, reset_episode_ids):
        if self.begun: raise RuntimeError('source first episode collector has already begun')
        self.begun = True; self.reset_clock = dict(reset_clock)
        self.reset_ids = reset_episode_ids.detach().cpu().clone()
        for i, row in self.rows.items():
            row['execution_episode_id'] = int(self.reset_ids[i])
            row['execution_episode_uid'] = f'{self.execution_run_uuid}:{i}:{int(self.reset_ids[i])}'

    def set_policy_step(self, step, owners):
        self.policy_step, self.owners = step, list(owners)

    def record_issued_difference(self, env_id, *, policy_step, owner, issued_action,
                                 recorded_teacher_action, clock):
        """An unexecuted prefix prediction cannot be an actual action divergence."""
        row = self.rows[env_id]
        if (owner != 'student' or row['first_numeric_difference'] is not None
                or recorded_teacher_action is None):
            return
        difference = (issued_action.detach().cpu().double()
                      - recorded_teacher_action.detach().cpu().double()).abs()
        if bool((difference > 0.).any()):
            row['first_numeric_difference'] = dict(policy_step=policy_step,
                env_id=env_id, execution_run_uuid=self.execution_run_uuid,
                episode_uid=row.get('execution_episode_uid'),
                clock_before_action=dict(clock), max_absolute_action_difference=float(difference.max()),
                per_dimension_absolute_difference=difference.tolist(),
                reference='actually_issued_recorded_Teacher_action_at_same_source_tick',
                source_states_equal_claimed=False, caused_failure_claimed=False)

    def observe(self, payload, *, physics_step, timestamp_s, failure, diagnostics=None, episode_ids=None):
        if not self.begun: return
        payload = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                   for key, value in payload.items()}
        failure = failure.detach().cpu()
        for i, row in self.rows.items():
            if self.closed[i] or not bool(payload['live'][i]): continue
            if self.owners[i] not in ('student', 'teacher_prefix', 'teacher_replay'): continue
            if episode_ids is not None and int(episode_ids[i]) != int(self.reset_ids[i]):
                raise RuntimeError('source episode identity advanced without owned first terminal')
            def select(value):
                if isinstance(value, torch.Tensor):
                    return value[i] if value.ndim and len(value) == self.n else value
                if isinstance(value, dict): return {key: select(v) for key, v in value.items()}
                return value
            sample = dict(env_id=i, source_episode_id=row['episode_id'],
                episode_uid=f"{self.execution_run_uuid}:{i}:{int(self.reset_ids[i])}",
                policy_step=self.policy_step, physics_step=physics_step, timestamp_s=timestamp_s,
                action_owner=self.owners[i], payload=select(payload),
                failure_code=int(failure[i]), diagnostics=select(diagnostics or {}))
            row['observed_physics_samples'] += 1
            policy_key = (i, self.policy_step)
            if policy_key not in self._seen_policy_steps:
                self._seen_policy_steps.add(policy_key)
                row[self.owners[i] + '_policy_steps_executed'] += 1
            for field, key in (('lift_m', 'max_net_lift_m'), ('qualified_hold_s', 'max_qualified_hold_s')):
                value = payload.get(field)
                if value is not None and math.isfinite(float(value[i])):
                    row[key] = max(float(value[i]), row[key] if row[key] is not None else -math.inf)
            kind = None
            from .tabletop_tensor_task import FAILURE_NAMES
            if int(failure[i]) in (3, 6, 7): kind = FAILURE_NAMES[int(failure[i])]
            contact = payload.get('bilateral_opposed_contact')
            valid = payload.get('contact_valid')
            if contact is not None and valid is not None and bool(valid[i]):
                if kind is None and self.previous_contact.get(i) is True and not bool(contact[i]):
                    kind = 'effective_opposed_contact_lost'
                self.previous_contact[i] = bool(contact[i])
            if (kind is None and payload.get('slip_valid') is not None and bool(payload['slip_valid'][i])
                    and float(payload['slip_m'][i]) > .0005): kind = 'fixed_anchor_slip_exceeded_original_0.5mm'
            if kind is not None:
                event = dict(kind=kind, policy_step=self.policy_step, physics_step=physics_step,
                    timestamp_s=timestamp_s, failure_code=int(failure[i]),
                    classification='task_related_physical_event;not_proof_of_action_error_causality')
                field = 'first_task_related_physical_event' if self.owners[i] == 'student' else 'first_prefix_physical_event'
                if row[field] is None: row[field] = event
            if bool(payload['done'][i]):
                observed = bool(payload['native_success'][i])
                row['observed_native_success_flag'] = observed
                row['native_success'] = self.native and observed and int(failure[i]) == 0
                row['terminal'] = dict(policy_step=self.policy_step, physics_step=physics_step,
                    timestamp_s=timestamp_s, failure_code=int(failure[i]),
                    failure_name=FAILURE_NAMES.get(int(failure[i]), 'unknown'),
                    observed_native_success_flag=observed, action_owner=self.owners[i],
                    payload=select(payload))
                self.closed[i] = True
            self.sink(sample)

    def summary(self):
        return dict(source_rows=_plain(list(self.rows.values())), evaluated_source_count=len(self.rows),
            execution_run_uuid=self.execution_run_uuid, reset_clock=getattr(self, 'reset_clock', None),
            reset_episode_ids=_plain(getattr(self, 'reset_ids', None)),
            completed_count=sum(row['terminal'] is not None for row in self.rows.values()),
            native_success_count=sum(row['native_success'] for row in self.rows.values()),
            branch_switch_reached_count=sum(row['student_policy_steps_executed'] > 0 for row in self.rows.values()),
            teacher_prefix_terminal_count=sum(row['terminal'] is not None and
                row['terminal']['action_owner'] == 'teacher_prefix' for row in self.rows.values()),
            teacher_prefix_native_success_count=sum(row['native_success'] and
                row['terminal']['action_owner'] == 'teacher_prefix' for row in self.rows.values()),
            student_segment_native_success_count=sum(row['native_success'] and
                row['terminal']['action_owner'] == 'student' for row in self.rows.values()),
            new_independent_initial_states=0,
            tracking_repeat_envelope=dict(available=False, event_not_claimed=True,
                reason='one_recorded_Teacher_trajectory_is_not_a_Teacher_repeat_envelope'))


def _saved_teacher_comparisons(policy, data, context, cases, output):
    """Compare on the EXACT saved Teacher observation, never state-query labels."""
    first = {}; worst = {}; worst_target = {}; count = 0
    path = output / 'same_state_teacher_observation_comparisons.jsonl.gz'
    with gzip.open(path, 'xt', encoding='utf-8', compresslevel=1) as stream:
        for case in cases:
            rows = case['rows']; indices = case['context_indices']
            predictions = []
            for start in range(0, len(rows), 128):
                predictions.append(policy.act(data['observations'][rows[start:start+128]]).detach().cpu())
            student = torch.cat(predictions)
            target_errors = None
            if context.get('controller_contract') is not None:
                from .tabletop_vision_command_shadow import shadow_sequence
                states = {key: value[indices[0]:indices[0]+1] for key, value in context['controller_start'].items()}
                contexts = {key: value[indices][None] for key, value in context['physics_context'].items()}
                predicted = shadow_sequence(states, student[None], contexts, context['controller_contract'])
                target_errors = {key: (value[0] - context['teacher_targets'][key][indices]).abs()
                                 for key, value in predicted.items()}
            for tick, original in enumerate(rows):
                group = action_group_errors(student[tick], case['actions'][tick])
                meta = data['sample_metadata'][original]
                region = 'START'
                for event in ('CONTACT', 'LIFT_ONSET', 'HOLD'):
                    if case['event_ticks'].get(event) is not None and tick >= case['event_ticks'][event]: region = event
                entry = dict(env_id=case['env_id'], episode_uid=case['episode_uid'], tick=tick,
                    source_policy_timestamp_s=meta['policy_timestamp_s'],
                    source_first_physics_step=meta['first_physics_step'],
                    comparison='same_exact_saved_Teacher_3844D_observation', action_groups=group,
                    student_action=student[tick], actually_issued_teacher_action=case['actions'][tick],
                    event_region=region,
                    event_entries=[event for event, step in case['event_ticks'].items() if step == tick],
                    conditional_cumulative_targets=(None if target_errors is None else
                        {key: dict(max_abs=float(value[tick].max()),
                            mean_abs=float(value[tick].mean())) for key, value in target_errors.items()}),
                    conditional_target_scope='recorded_feedback_only;not_contact_or_world_prediction')
                largest = max(item['max_abs'] for item in group.values())
                target_largest = 0. if target_errors is None else max(float(value[tick].max()) for value in target_errors.values())
                if (largest > 0. or target_largest > 0.) and case['env_id'] not in first:
                    first[case['env_id']] = dict(tick=tick, source_policy_timestamp_s=meta['policy_timestamp_s'],
                        event_region=region, max_abs_action_error=largest,
                        max_abs_conditional_target_error=target_largest, caused_failure_claimed=False)
                if case['env_id'] not in worst or largest > worst[case['env_id']]['max_abs_action_error']:
                    worst[case['env_id']] = dict(tick=tick, source_policy_timestamp_s=meta['policy_timestamp_s'],
                        event_region=region, event_entries=entry['event_entries'], max_abs_action_error=largest)
                if target_errors is not None:
                    groups = worst_target.setdefault(case['env_id'], {})
                    for key, value in target_errors.items():
                        maximum = float(value[tick].max())
                        if key not in groups or maximum > groups[key]['max_abs_error']:
                            groups[key] = dict(tick=tick, source_policy_timestamp_s=meta['policy_timestamp_s'],
                                event_region=region, max_abs_error=maximum)
                _write_line(stream, entry); count += 1
    return dict(path=str(path), sha256=file_sha256(path), policy_rows=count,
        comparison_implementation_sha256=file_sha256(__file__),
        comparison_dependencies=comparison_dependencies(),
        first_numeric_difference=first, worst_action_difference=worst,
        worst_conditional_target_difference=worst_target,
        physical_failure_causality_claimed=False)


def _native_execution(world, device_world, env, native_camera):
    if not native_camera: return False
    from .tabletop_vision_initial_state_audit import _require_native
    _require_native(world, device_world, env, native_camera)
    return True


def install_stage6_physics_diagnostics(env, world, manifest, output, *, ownership, **options):
    """Reuse native SDK/q/palm sampling with truthful Teacher/Student labels.

    The original diagnostics writer labels every submitted request as Student.
    Stage6 reuses its complete physical sampling, but the submission and summary
    writers must name the actual mixed per-environment owners before writing.
    """
    from .tabletop_vision_evaluation_diagnostics import EvaluationPhysicsDiagnostics, _plain as audit_plain
    class OwnershipStream:
        def __init__(self, stream): self.stream = stream
        def write(self, text):
            row = json.loads(text)
            row.update(source='actual_issued_SDK24_action', action_owner=list(ownership()))
            return self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        def close(self): self.stream.close()
    class Stage6Diagnostics(EvaluationPhysicsDiagnostics):
        def finish(self, *, error=None):
            if self.closed: return getattr(self, 'summary', None)
            self.closed = True
            self.env.physics_action_diagnostics = self.prior
            self.env._diagnostic_policy_step = self.prior_policy_step
            try: self._flush()
            finally:
                self.policy_stream.close(); self.terminal_stream.close()
            paths = self.paths + [self.policy_path, self.terminal_path]
            self.summary = audit_plain(dict(version='dh116-stage6-source-physics-audit-v1',
                status='error' if error else 'recorded', error=error,
                policy_batches=self.policy_batches, physics_rows=self.rows,
                terminal_count=len(self.terminal_keys), first_trial_terminal_count=int(self.first_closed.sum()),
                checkpoint_sha256=self.checkpoint_sha256, manifest_sha256=self.manifest_sha256,
                trial_identity_count=self.n, evaluated_source_identity_count=min(4, self.n),
                unevaluated_env_ids=list(range(4, self.n)), source_events=self.events, source_coverage=self.coverage,
                actor_source='explicit_per_environment_action_owner;Teacher_prefix_or_Student',
                array_field_semantics=dict(actor_request24='actual_issued_SDK24;join_policy_requests_action_owner'),
                scope='source_template_diagnostics;four_cases_only;remaining_rows_unevaluated',
                native_backend_target='root_physx_view.get_dof_position_targets;no_cache_fallback',
                command_cache='separate_IsaacLab_command_cache;not_backend',
                capture_timing='native_after_consume_before_terminal_reset',
                terminal_policy='owned_first_terminal_per_episode_uid;no_terminal_overwrite',
                native_trials_certified_by_writer=0, new_independent_initial_states=0,
                student_success_claimed=False, physical_control_changed=False, rewards_changed=False,
                files=[dict(path=str(path), sha256=file_sha256(path)) for path in paths]))
            summary = self.output / 'summary.json'
            _json_new(summary, self.summary); self.paths.append(summary)
            return self.summary
    writer = Stage6Diagnostics(env, world, manifest, output, **options)
    # EvaluationPhysicsDiagnostics uses equality with initial_ids solely for
    # first-trial eligibility. Native episode IDs are nonnegative; the negative
    # sentinel excludes neutral rows while preserving their raw actual IDs.
    writer.initial_ids[4:] = -1
    writer.policy_stream = OwnershipStream(writer.policy_stream)
    return writer


@torch.inference_mode()
def run_stage6_native(world, device_world, task, env, args, critic_contract, *, camera=None):
    """One new output per cold-reset source run or per (event,horizon) branch."""
    from .tabletop_vision_training import _contract, _observations
    from .tabletop_vision_student import SensorHistory
    from .tabletop_vision_inference import load_vision_inference
    from .tabletop_vision_evaluation import _verify_native_physical_adapter
    config = load_native_config(args.vision_stage6_native_config)
    if args.num_envs != 8 or device_world.n != 8 or args.seed != 42:
        raise ValueError('source replication requires original mixed eight environments and scene seed42')
    if (getattr(args, 'vision_mirror_preprocessing', 'm0_raw') != 'm0_raw'
            or getattr(args, 'vision_camera_mode', None) != 'fixed'
            or getattr(args, 'vision_action_distribution', None) != 'tanh_gaussian'):
        raise ValueError('source replication requires unchanged fixed-camera Tanh SDK24 M0')
    if any(getattr(args, name, None) for name in ('resume', 'vision_teacher_source', 'surface_mpc',
            'surface_teacher_replay', 'vision_m2_replay', 'bimanual_explicit_sdk_adapter',
            'tripod_recovery_stage', 'bimanual_tripod_teacher_coordination_proof')):
        raise ValueError('Stage6 source/branch excludes PPO, live Teacher and control wrappers')
    native_camera = camera is None
    genuine = _native_execution(world, device_world, env, native_camera)
    if genuine and (device_world.decimation != 8 or device_world.dt != 1/480):
        raise ValueError('native Stage6 requires unchanged original480Hz physics/eight-substep60Hz policy')
    data = torch.load(config['teacher_dataset']['path'], map_location='cpu', weights_only=False)
    context = torch.load(config['sequence_context']['path'], map_location='cpu', weights_only=False)
    execution_controller = None
    if genuine:
        from .tabletop_vision_command_shadow import controller_contract
        execution_controller = controller_contract(task, device_world)
        if execution_controller != context.get('controller_contract'):
            raise ValueError('actual unwrapped M0 controller differs from recorded source controller contract')
    expected = torch.load(config['source_initial_state']['path'], map_location='cpu', weights_only=False)
    collection = json.loads(Path(config['source_collection_report']['path']).read_text(encoding='utf-8'))
    cases = source_cases(data, context, collection, dataset_sha256=config['teacher_dataset']['sha256'])
    cases = apply_event_reference(cases, config)
    if expected.get('seed') != 42 or expected.get('source') != 'actual_native_cold_reset_before_camera':
        raise ValueError('actual original Teacher cold-reset state identity required')
    if config['mode'] == 'branch' and any(case['event_ticks'][config['event']] is None for case in cases):
        raise ValueError('all four sources need the requested recorded event entry')
    contract_args = copy(args)
    if getattr(contract_args, 'rollout_steps', None) is None: contract_args.rollout_steps = 24
    current = _contract(contract_args, critic_contract)
    for name in ('camera_config_sha256', 'primary_reset_sha256', 'secondary_reset_sha256', 'action_contract'):
        if current.get(name) != data['visual_contract'].get(name):
            raise ValueError('actual source camera/reset/action provenance mismatch: ' + name)
    policy = load_vision_inference(config['checkpoint']['path'], current,
        device=args.device, allow_physical_adapter=True)
    if policy.checkpoint_sha256 != config['checkpoint']['sha256']:
        raise ValueError('Stage6 checkpoint changed during load')
    _verify_native_physical_adapter(policy, device_world)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation of every artifact prevents reuse of an old result.
    _json_new(output / 'stage6_native_config.json', config)
    if camera is None:
        from .tabletop_vision_fixed_camera import FixedVisionCamera
        if [args.vision_width, args.vision_height] != [224, 96]:
            raise ValueError('original calibrated fixed camera resolution224x96 required')
        camera = FixedVisionCamera(world, device_world, args.depth_camera_config,
            width=args.vision_width, height=args.vision_height, num_points=128)
    raw_path = output / 'source_physics_480hz.jsonl.gz'
    diagnostics = None; error = None; control_steps = 0
    with gzip.open(raw_path, 'xt', encoding='utf-8', compresslevel=1) as raw:
        collector = SourceTelemetryCollector(cases, 8, native=genuine, sink=lambda row: _write_line(raw, row))
        def telemetry(payload):
            clock = _clock(device_world)
            collector.observe(payload, physics_step=clock['physics_step'], timestamp_s=clock['timestamp_s'],
                failure=task.failure, diagnostics=getattr(device_world, 'last_action_diagnostics', None),
                episode_ids=getattr(env, 'diagnostic_episode_ids', None))
        old_telemetry = getattr(env, 'evaluation_telemetry', None)
        old_callback = getattr(old_telemetry, 'callback', None)
        env.enable_evaluation_telemetry(telemetry)
        try:
            # Camera construction precedes the FINAL cold reset. No manifest
            # injection, pose restoration or solver checkpoint shortcut occurs.
            critic = env.reset()
            before = _reset_snapshot(device_world, task); before_clock = _clock(device_world)
            source_match = compare_source_reset(expected, before)
            with (output / 'source_reset_readback.pt').open('xb') as stream:
                torch.save(dict(source_expected=expected, actual_before_camera=before,
                    clock_before_camera=before_clock, comparison=source_match), stream)
            _json_new(output / 'source_reset_comparison.json', source_match)
            if not source_match['matched']:
                raise RuntimeError('source initial state mismatch; exact per-field readback saved; no tolerance widened')
            packet = camera.sample()
            after = _reset_snapshot(device_world, task); after_clock = _clock(device_world)
            warmup = compare_source_reset(before, after)
            with (output / 'source_reset_after_camera.pt').open('xb') as stream:
                torch.save(dict(actual_after_camera=after, clock_after_camera=after_clock, comparison=warmup), stream)
            _json_new(output / 'source_camera_warmup_comparison.json', dict(warmup,
                before_clock=before_clock, after_clock=after_clock))
            if not warmup['matched'] or before_clock != after_clock:
                raise RuntimeError('camera warmup changed the actual final source reset/clock')
            if not bool(packet['image_valid'].all() & packet['object_valid'][:4].all()):
                raise RuntimeError('source requires actual visible RGB-D; no truth fallback')
            history = SensorHistory(8, device=args.device)
            previous = torch.zeros((8, 24), device=args.device)
            obs = _observations(packet, device_world, previous, history, critic)
            if obs['policy'].shape != (8, 3844): raise ValueError('unchanged3844D Student input required')
            with (output / 'source_initial_actor_input.pt').open('xb') as stream:
                torch.save(dict(policy=obs['policy'].detach().cpu(), history=history.frames.detach().cpu(),
                    previous_action=previous.cpu(), sensor_packet={k: v.detach().cpu() for k, v in packet.items()
                        if isinstance(v, torch.Tensor)}), stream)
            if 'same_state_comparison_cache' in config:
                from .tabletop_vision_comparison_cache import load_comparison_cache
                comparisons = load_comparison_cache(config['same_state_comparison_cache'], config,
                    implementation_sha256=file_sha256(__file__))
            else:
                comparisons = _saved_teacher_comparisons(policy, data, context, cases, output)
                comparisons['comparison_inputs'] = {name: config[name] for name in
                    ('checkpoint', 'teacher_dataset', 'sequence_context')}
                if 'event_reference' in config:
                    comparisons['comparison_inputs']['event_reference'] = config['event_reference']
                comparisons['reused_offline_comparisons'] = False
            collector.begin(reset_clock=before_clock, reset_episode_ids=env.diagnostic_episode_ids)
            if genuine:
                manifest = dict(trials=[dict(trial_id=f'source-template-{i}', grasp_style=
                    'pinch_thumb_index' if i < 4 else 'tripod_thumb_index_middle') for i in range(8)])
                diagnostics = install_stage6_physics_diagnostics(env, device_world, manifest, output,
                    ownership=lambda: collector.owners, checkpoint_sha256=config['checkpoint']['sha256'])
            maximum = 1800 if config['mode'] == 'source' else (max(len(case['actions']) for case in cases)
                if config['mode'] == 'teacher-replay' else
                max(case['event_ticks'][config['event']] + config['branch_steps'] for case in cases))
            with gzip.open(output / 'live_policy_60hz.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as live:
                for tick in range(maximum):
                    student = policy.act(obs)
                    action, owners = scheduled_actions(cases, config, tick, student, collector.closed)
                    active = [i for i in range(4) if owners[i] in ('student', 'teacher_prefix', 'teacher_replay')]
                    if not active: break
                    collector.set_policy_step(tick, owners)
                    for case in cases:
                        i = case['env_id']
                        if i not in active: continue
                        reference = case['actions'][tick] if tick < len(case['actions']) else None
                        collector.record_issued_difference(i, policy_step=tick, owner=owners[i],
                            issued_action=action[i], recorded_teacher_action=reference,
                            clock=_clock(device_world))
                        _write_line(live, dict(env_id=i, policy_step=tick, clock_before_action=_clock(device_world),
                            run_uuid=collector.execution_run_uuid, episode_id=int(collector.reset_ids[i]),
                            episode_uid=collector.rows[i]['execution_episode_uid'],
                            recorded_teacher_episode_uid=case['episode_uid'],
                            owner=owners[i], student_action=student[i], issued_action=action[i],
                            actual_previous_issued_action=previous[i], recorded_teacher_action=reference,
                            recorded_time_aligned_action_error=(None if reference is None else
                                action_group_errors(student[i], reference)),
                            comparison_scope=('live_Teacher_replay_observation_and_recorded_issued_action'
                                if owners[i] in ('teacher_prefix', 'teacher_replay') else
                                'time_aligned_recorded_reference;not_a_state_query_expert'),
                            saved_teacher_observation_max_abs_error=(None if tick >= len(case['rows']) else
                                float((obs['policy'][i].detach().cpu()-data['observations'][case['rows'][tick]]).abs().max()))))
                    if diagnostics is not None: diagnostics.submit(action, policy_step=tick)
                    critic, _, done, extra = env.step(action)
                    control_steps = tick + 1
                    # A real previous action follows what was ACTUALLY issued,
                    # including the Teacher prefix. No future Teacher history.
                    previous = action.clone(); previous[done] = 0.
                    if bool(collector.closed[:4].all()): break
                    if tick + 1 < maximum:
                        obs = _observations(camera.sample(), device_world, previous, history, critic, reset_mask=done)
                    if control_steps % 60 == 0:
                        print('STAGE6_NATIVE', json.dumps(dict(mode=config['mode'], control_steps=control_steps,
                            first_terminals=int(collector.closed[:4].sum()), ppo_updates=0)), flush=True)
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        finally:
            if diagnostics is not None: diagnostics.finish(error=error)
            if old_callback is not None: env.evaluation_telemetry.callback = old_callback
            else: env.evaluation_telemetry = old_telemetry
    verify_inputs(config)
    if file_sha256(config['config_identity']['path']) != config['config_identity']['sha256']:
        raise ValueError('Stage6 native config changed during execution')
    report = collector.summary()
    full_source = config['mode'] == 'source'
    report.update(version=VERSION, mode=config['mode'], status='error' if error else
        ('completed' if report['completed_count'] == 4 else 'bounded_branch_completed'
            if config['mode'] == 'branch' else 'not_completed'), error=error,
        native_runtime_verified=genuine, execution='native_isaac_tensor_world' if genuine else 'injected_fixture_not_native',
        source_task_pure_student=full_source, teacher_prefix_is_pure_student_success=False,
        branch_switch_reached_count=report['branch_switch_reached_count'] if config['mode'] == 'branch' else 0,
        pure_student_native_success_count=report['native_success_count'] if full_source else 0,
        branch_or_teacher_prefix_native_success_count=report['native_success_count'] if not full_source else 0,
        control_steps=control_steps, source_duration_limit_s=30., physics_hz=480, policy_hz=60,
        event=config.get('event'), branch_steps=config.get('branch_steps'),
        source_case_definitions=[dict(env_id=case['env_id'], episode_id=case['episode_id'],
            source_episode_uid=case['episode_uid'], source_run_uuid=case['run_uuid'],
            recorded_teacher_tape_length=len(case['actions']), event_entry_ticks=case['event_ticks'],
            event_sources=case['event_sources'], source_trajectory_id=case['source_trajectory_id']) for case in cases],
        unevaluated_environments=[dict(env_id=i, action='neutral_zero24', evaluated=False, expert=False) for i in range(4, 8)],
        same_state_comparisons=locals().get('comparisons'),
        reset_reference_scope='actual_cold_reset_readback;not_PhysX_solver_checkpoint',
        ppo_updates=0, new_expert_sources=0, heldout_generalization_test=False,
        production_evaluation_certificate_issued=False, inference_provenance=policy.provenance,
        execution_controller_contract=execution_controller,
        inputs={name: config[name] for name in INPUT_NAMES + (('event_reference',) if 'event_reference' in config else ())},
        raw_physics=dict(path=str(raw_path), sha256=file_sha256(raw_path)))
    _json_new(output / 'stage6_native_report.json', report)
    if error: raise RuntimeError(error)
    return report
