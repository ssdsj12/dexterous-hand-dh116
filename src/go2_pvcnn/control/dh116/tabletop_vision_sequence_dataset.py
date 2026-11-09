"""Continuous saved sensor histories and provenance-bound event sampling.

Only ``observations`` reaches the Actor. Native contact/hold metadata is used
for sampling and supervised targets, never to create sensor history or issue
online Teacher actions. CPU schema validation does not certify native success.
"""
import math

import torch

from .tabletop_vision_supervision_dataset import _matrix, join_sequence_context


EVENT_WEIGHTS = dict(START=.10, CONTACT=.35, LIFT_ONSET=.35, HOLD=.20)
RECOVERY_EVENT_WEIGHTS = dict(START=.10, CONTACT=.25, LIFT_ONSET=.25, HOLD=.20, RECOVERY=.20)
VERSION = 'dh116-vision-sequence-index-v1'


def _episode_key(row):
    return (row.get('run_uuid'), row.get('env_id'), row.get('episode_uid', str(row.get('episode_id'))))


def _source_group(row, episode):
    # Explicit shared lineage overrides a derived trajectory's local name.
    source = row.get('source_group_id', episode.get('source_group_id', row.get('source_trajectory_id')))
    if not isinstance(source, str) or not source.strip():
        raise ValueError('each sequence needs an explicit source trajectory group')
    return source


class SequenceDataset:
    """Index 16-step windows; sample with a private CPU generator.

    Long continuous runs contribute every full rolling window, including the
    final window ending at terminal. An episode shorter than 16 contributes a
    masked window; short gap fragments in long episodes are excluded. Missing
    sidecar rows split runs. No row-count-based join or
    SensorHistory reconstruction is performed. ``uniform`` implements B0;
    ``event`` implements 40% source-balanced plus 60% event-stratified B1/B2.
    """
    def __init__(self, data, sequence_context=None, *, dataset_sha256=None,
                 controller_source_sha256=None, action_contract_sha256=None,
                 sequence_length=16, heldout_source_trajectory_ids=(), subset='train',
                 sampling_mode='event', seed=116601):
        if sequence_length != 16 or type(sequence_length) is not int:
            raise ValueError('Stage6 sequence_length must be 16 policy steps')
        if sampling_mode not in ('uniform', 'event') or subset not in ('train', 'heldout'):
            raise ValueError('unknown sequence sampling_mode or subset')
        if type(seed) is not int or seed < 0:
            raise ValueError('sequence seed must be a nonnegative integer')
        if not isinstance(data, dict):
            raise ValueError('original saved supervision pool required')
        metadata, episodes = data.get('sample_metadata'), data.get('episodes')
        if not isinstance(metadata, list) or not metadata or not isinstance(episodes, dict):
            raise ValueError('original sample_metadata and episodes required')
        n = len(metadata)
        _matrix(data.get('observations'), 'observations', n, 3844)
        _matrix(data.get('actions'), 'actions', n, 24)
        self.data, self.context = data, sequence_context
        self.sequence_length, self.sampling_mode = sequence_length, sampling_mode
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.joined = None
        if sequence_context is not None:
            self.joined = join_sequence_context(data, sequence_context, dataset_sha256=dataset_sha256,
                controller_source_sha256=controller_source_sha256,
                action_contract_sha256=action_contract_sha256)
        row_valid = torch.ones(n, dtype=torch.bool) if self.joined is None else self.joined['row_valid'].clone()
        sources, episode_rows = {}, {}
        seen = set()
        for i, row in enumerate(metadata):
            if not isinstance(row, dict) or type(row.get('tick')) is not int or row['tick'] < 0:
                raise ValueError('sample_metadata requires explicit nonnegative tick')
            ep = episodes.get(str(row.get('episode_id')))
            if (not isinstance(ep, dict) or ep.get('success') is not True
                    or ep.get('native_physics') is not True or ep.get('first_episode') is not True
                    or ep.get('recovered') is not False):
                raise ValueError('sequence training requires qualified successful first native episodes')
            if row.get('env_id') != ep.get('env_id'):
                raise ValueError('sequence environment identity disagrees with episode')
            key = _episode_key(row)
            identity = (*key, row['tick'])
            if identity in seen:
                raise ValueError('duplicate sequence dataset identity')
            seen.add(identity)
            sources[i] = _source_group(row, ep)
            episode_rows.setdefault(key, []).append(i)
        groups = set(sources.values())
        heldout = set(heldout_source_trajectory_ids)
        if heldout - groups:
            raise ValueError('unknown heldout source trajectory groups')
        if not groups - heldout:
            raise ValueError('at least one whole source group must remain for training')
        if subset == 'heldout' and not heldout:
            raise ValueError('heldout subset requires an independently selected source group')
        selected = heldout if subset == 'heldout' else groups - heldout
        self.segments = []
        for key, indices in episode_rows.items():
            segment, ended = [], False
            for i in indices:
                row = metadata[i]
                ep = episodes[str(row['episode_id'])]
                if ended:
                    row_valid[i] = False
                    continue
                if not row_valid[i] or sources[i] not in selected:
                    if segment: self.segments.append(segment)
                    segment = []
                else:
                    if segment and not self._adjacent(segment[-1], i, sources):
                        self.segments.append(segment); segment = []
                    segment.append(i)
                terminal = row.get('terminal') is True
                record = ep.get('terminal')
                if isinstance(record, dict) and type(record.get('control_steps')) is int:
                    terminal = terminal or row['tick'] >= record['control_steps'] - 1
                if terminal:
                    if segment: self.segments.append(segment)
                    segment = []; ended = True
            if segment: self.segments.append(segment)
        self.event_labels = self._event_labels(row_valid)
        self.windows = []
        discarded_fragments = 0
        for segment in self.segments:
            episode_length = len(episode_rows[_episode_key(metadata[segment[0]])])
            if len(segment) < sequence_length and episode_length >= sequence_length:
                discarded_fragments += 1
                continue
            starts = range(len(segment) - sequence_length + 1) if len(segment) >= sequence_length else (0,)
            for start in starts:
                rows = segment[start:start + sequence_length]
                first = metadata[rows[0]]
                events = {name for name, label in self.event_labels.items() if bool(label[rows].any())}
                self.windows.append(dict(row_indices=rows, source_group=sources[rows[0]],
                    episode_uid=first.get('episode_uid', str(first['episode_id'])),
                    episode_key=_episode_key(first), run_uuid=first.get('run_uuid'),
                    events=events, valid_steps=len(rows)))
        if not self.windows:
            raise ValueError('no legal continuous sequence windows')
        self.sampling_probabilities, event_weights = self._probabilities()
        counts = [window['valid_steps'] for window in self.windows]
        self.report = dict(version=VERSION, validation_scope='schema_only_not_native_verification',
            fit_only=len(groups) == 1 or not bool(heldout), source_trajectory_count=len(groups),
            heldout_independent_source_count=len(heldout), subset=subset,
            source_groups=sorted(selected), split_unit='source_group;replay_copies_and_overlapping_histories_bound',
            sequence_length=sequence_length, window_count=len(self.windows),
            full_window_count=sum(count == sequence_length for count in counts),
            padded_window_count=sum(count < sequence_length for count in counts),
            discarded_short_fragment_count=discarded_fragments,
            invalid_context_rows=0 if self.joined is None else int((~self.joined['row_valid']).sum()),
            excluded_rows=int((~row_valid).sum()), sampling_mode=sampling_mode,
            uniform_fraction=1. if sampling_mode == 'uniform' or not event_weights else .4,
            event_fraction=0. if sampling_mode == 'uniform' or not event_weights else .6,
            available_event_weights=event_weights, min_valid_steps=min(counts), max_valid_steps=max(counts),
            expected_valid_steps_per_sequence=float(sum(p * count for p, count in zip(self.sampling_probabilities, counts))),
            covered_original_rows=len({i for window in self.windows for i in window['row_indices']}),
            actor_observation_size=3844, actor_uses_privileged_state=False,
            contexts_joined=self.joined is not None, generator='private_cpu_torch_generator')

    def _adjacent(self, before, after, sources):
        old, row = self.data['sample_metadata'][before], self.data['sample_metadata'][after]
        if row['tick'] != old['tick'] + 1 or row.get('reset') is True or row.get('reset_mask') is True:
            return False
        if sources[before] != sources[after] or _episode_key(old) != _episode_key(row):
            return False
        old_time = old.get('policy_timestamp_s', old.get('simulation_time_s'))
        when = row.get('policy_timestamp_s', row.get('simulation_time_s'))
        old_step, step = old.get('first_physics_step', old.get('physics_step')), row.get('first_physics_step', row.get('physics_step'))
        if (type(old_time) not in (int, float) or type(when) not in (int, float)
                or not math.isfinite(old_time) or not math.isfinite(when)
                or type(old_step) is not int or type(step) is not int or step - old_step != 8):
            return False
        duration = 1 / 60
        if self.joined is not None:
            j = int(self.joined['context_indices'][before])
            duration = float(self.context['physics_context']['dt_s'][j].sum())
        return abs(when - old_time - duration) <= 1e-6

    def _event_labels(self, row_valid):
        n = len(row_valid)
        labels = {name: torch.zeros(n, dtype=torch.bool) for name in EVENT_WEIGHTS}
        for segment in self.segments:
            for i in segment:
                if self.data['sample_metadata'][i]['tick'] < 4: labels['START'][i] = True
        if self.context is None:
            return labels
        events = self.context.get('events', {})
        joined = self.joined['context_indices']
        def values(name, segment):
            value = events.get(name)
            return None if value is None else value.detach().cpu()[joined[segment]]
        for segment in self.segments:
            opposed = values('opposed_contact', segment)
            hold = values('native_hold_qualified', segment)
            if hold is not None: labels['HOLD'][segment] = hold
            onset = values('lift_onset', segment)
            trusted = self.context.get('lift_onset_source') in ('native_executed_event', 'native_reference_event')
            positions = onset.nonzero().flatten().tolist() if onset is not None and trusted else []
            if opposed is not None:
                # A segment beginning after a gap lacks the preceding predicate;
                # its first true value alone is not a false->true transition.
                for t in range(1, len(segment)):
                    if bool(opposed[t] and not opposed[t - 1]):
                        labels['CONTACT'][segment[max(0, t - 6):t + 13]] = True
                velocity = values('mean_palm_vz_m_s', segment)
                if velocity is None:
                    measured = values('measured_palm_velocity_m_s', segment)
                    if measured is not None: velocity = measured[:, :, 2].mean(-1)
                if not (onset is not None and trusted) and velocity is not None:
                    for t in range(len(segment) - 2):
                        if bool(opposed[t:t + 3].all() and (velocity[t:t + 3] > .0005).all()):
                            positions = [t]; break
            for t in positions: labels['LIFT_ONSET'][segment[max(0, t - 12):t + 25]] = True
        if self._qualified_recovery():
            labels['RECOVERY'] = torch.zeros(n, dtype=torch.bool)
            for segment in self.segments:
                recovery = values('recovery_qualified', segment)
                if recovery is not None:
                    for t, i in enumerate(segment):
                        row = self.data['sample_metadata'][i]
                        identity = (row.get('run_uuid'), row['env_id'], row.get('episode_uid'), row['tick'])
                        qualified = any(tuple(record['start_identity'])[:3] == identity[:3]
                            and record['start_identity'][3] <= identity[3] <= record['end_identity'][3]
                            for record in self.context['corrective_segments'])
                        labels['RECOVERY'][i] = bool(recovery[t]) and qualified
        return labels

    def _qualified_recovery(self):
        context = self.context
        if (context.get('recovery_native_verified') is not True
                or context.get('recovery_schema_version') != 'dh116-corrective-segment-v1'
                or 'recovery_qualified' not in context.get('events', {})):
            return False
        records = context.get('corrective_segments')
        required = ('parent_student_episode_uid', 'branch_policy_step', 'actual_branch_state_path',
                    'expert_version', 'issued_actions_path', 'native_result_path', 'start_identity', 'end_identity')
        if not isinstance(records, list) or not records:
            return False
        for row in records:
            if not isinstance(row, dict) or row.get('native_success') is not True or any(name not in row for name in required):
                return False
            if type(row['branch_policy_step']) is not int or row['branch_policy_step'] < 0:
                return False
            if (not all(isinstance(row[name], str) and row[name].strip() for name in required[:1] + required[2:6])
                    or len(row['start_identity']) != 4 or len(row['end_identity']) != 4
                    or tuple(row['start_identity'])[:3] != tuple(row['end_identity'])[:3]):
                return False
        return True

    def _probabilities(self):
        sources = {}
        for i, window in enumerate(self.windows):
            sources.setdefault(window['source_group'], {}).setdefault(window['episode_key'], []).append(i)
        uniform = torch.zeros(len(self.windows), dtype=torch.float64)
        for episodes in sources.values():
            for indices in episodes.values(): uniform[indices] = 1 / (len(sources) * len(episodes) * len(indices))
        weights = RECOVERY_EVENT_WEIGHTS if 'RECOVERY' in self.event_labels else EVENT_WEIGHTS
        available = {name: [i for i, window in enumerate(self.windows) if name in window['events']] for name in weights}
        available = {name: indices for name, indices in available.items() if indices}
        total = sum(weights[name] for name in available)
        normalized = {name: weights[name] / total for name in available} if total else {}
        if self.sampling_mode == 'uniform' or not available:
            return uniform, normalized
        event = torch.zeros_like(uniform)
        for name, indices in available.items(): event[indices] += normalized[name] / len(indices)
        probabilities = .4 * uniform + .6 * event
        return probabilities / probabilities.sum(), normalized

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        batch = self.batch([index])
        def squeeze(value):
            if isinstance(value, dict): return {key: squeeze(child) for key, child in value.items()}
            return value[0] if isinstance(value, torch.Tensor) else value
        return {key: squeeze(value) for key, value in batch.items()}

    def sample_batch(self, batch_sequences=4, *, generator=None):
        if type(batch_sequences) is not int or batch_sequences < 1:
            raise ValueError('batch_sequences must be a positive integer')
        generator = self.generator if generator is None else generator
        if not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
            raise ValueError('sequence sampling requires a private CPU generator')
        indices = torch.multinomial(self.sampling_probabilities, batch_sequences, replacement=True, generator=generator)
        return self.batch(indices.tolist())

    def batch(self, indices):
        indices = list(indices)
        if not indices or any(type(i) is not int or i < 0 or i >= len(self.windows) for i in indices):
            raise ValueError('batch indices must select legal sequence windows')
        rows = torch.full((len(indices), self.sequence_length), -1, dtype=torch.long)
        for b, index in enumerate(indices):
            selected = self.windows[index]['row_indices']
            rows[b, :len(selected)] = torch.tensor(selected, dtype=torch.long)
        valid = rows >= 0
        def gather(tensor, selected, mask):
            index = selected.clamp_min(0).to(tensor.device)
            values = tensor[index].detach().cpu().clone()
            values[~mask] = 0
            return values
        result = dict(observations=gather(self.data['observations'], rows, valid),
            actions=gather(self.data['actions'], rows, valid), valid=valid,
            row_indices=rows, window_indices=torch.tensor(indices),
            controller_start=None, physics_context=None, teacher_targets=None, physics_valid=None)
        if self.context is not None:
            selected = self.joined['context_indices'][rows.clamp_min(0)]
            def tree(value):
                if isinstance(value, dict): return {key: tree(child) for key, child in value.items()}
                return gather(value, selected, valid)
            for name in ('controller_start', 'physics_context', 'teacher_targets'):
                result[name] = tree(self.context[name])
            result['physics_valid'] = (result['physics_context']['source_valid']
                & result['physics_context']['live'] & valid[..., None])
        return result
