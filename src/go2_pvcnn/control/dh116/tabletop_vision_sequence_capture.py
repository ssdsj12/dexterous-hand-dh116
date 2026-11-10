"""Opt-in, owned snapshots of the actual native command execution path.

Install only for new collection. No old trace can supply missing precontrol
feedback, and no successful-pool SHA is guessed before the pool is finalized.
"""
import hashlib
import math
from pathlib import Path

import torch

from . import tabletop_vision_command_shadow as shadow
from .tabletop_joint_target_readback import read_joint_target_evidence


def _cpu(tree):
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().clone()
    if isinstance(tree, dict):
        return {key: _cpu(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_cpu(value) for value in tree]
    return tree


def _stack(rows, dim):
    if isinstance(rows[0], dict):
        return {key: _stack([row[key] for row in rows], dim) for key in rows[0]}
    return torch.stack(rows, dim)


def _select(tree, index):
    if isinstance(tree, dict):
        return {key: _select(value, index) for key, value in tree.items()}
    return tree[index:index+1]


class SequenceContextCapture:
    def __init__(self, task, world, sink=None, *, require_native_readback=False):
        self.contract = shadow.controller_contract(task, world)
        if getattr(world, 'decimation', 8) != 8:
            raise ValueError('sequence capture requires eight actual 480 Hz substeps')
        self.task, self.world, self.sink = task, world, sink
        self.require_native_readback = require_native_readback
        self._control, self._advance = task.control, world.advance
        self.records, self.active = [], None
        self.closed = False
        task.control = self._before_control
        world.advance = self._after_advance

    def begin_tick(self, metadata, actions, initial_state=None):
        if self.active is not None:
            raise ValueError('previous sequence context tick was not completed')
        n = self.task.command.shape[0]
        if self.require_native_readback:
            velocity = self.task.m.get('wrist_velocity')
            if (not isinstance(velocity, torch.Tensor) or velocity.shape != (n, 2, 6)
                    or not torch.isfinite(velocity).all()):
                raise ValueError('native event collection requires actual finite wrist_velocity [N,2,6]')
            missing = [key for key in ('effective_pad_loads_n', 'effective_pad_vectors_w',
                'effective_pad_valid') if key not in self.task.m]
            if missing or getattr(self.task, 'surface_lift_success', None) is None:
                raise ValueError('native event collection requires original opposed-contact and qualified-Hold observers')
        ids = [row['env_id'] for row in metadata]
        if len(ids) != len(set(ids)) or any(index < 0 or index >= n for index in ids):
            raise ValueError('unique env sample identities required')
        observer = getattr(self.task, 'surface_lift_success', None)
        # These are the original observer's completed native predicates, never
        # inferred from an immobile palm or an action request.
        events = {}
        if all(key in self.task.m for key in ('effective_pad_loads_n',
                'effective_pad_vectors_w', 'effective_pad_valid')):
            from .tabletop_surface_force_sharing import surface_targets
            native = self.task.m
            events['opposed_contact'] = _cpu(surface_targets(native['effective_pad_loads_n'],
                native['effective_pad_vectors_w'], valid=native['effective_pad_valid'])['paired'].all(1))
        if observer is not None:
            events['native_hold_qualified'] = _cpu(observer.window_good)
        if 'wrist_velocity' in self.task.m:
            events['measured_palm_velocity_m_s'] = _cpu(self.task.m['wrist_velocity'][:, :, :3])
        self.active = dict(metadata=[dict(row) for row in metadata], actions=_cpu(actions),
            controller_start=_cpu(initial_state if initial_state is not None else
                shadow.command_state(self.task, self.world)),
            contexts=[], raw_measured=[], targets=[], backend=[], cache=[], sdk_readback=[],
            post_physics_feedback=[], events=events,
            live_count=torch.zeros(n, dtype=torch.long))

    def _before_control(self, actions, live, dt):
        if self.active is None:
            raise ValueError('native control called outside a sequence capture tick')
        if len(self.active['contexts']) != len(self.active['targets']):
            raise ValueError('missing advance for the preceding captured control')
        context = shadow.measured_context(self.task, self.world, live)
        context['phase'] = self.task.phase.detach().clone()
        self.active['raw_measured'].append(_cpu(self.task.m))
        allowed = self._control(actions, live, dt)
        context['allowed'] = allowed.detach().clone()
        self.active['contexts'].append(_cpu(context))
        self.active['live_count'] += _cpu(live).long()
        return allowed

    def _after_advance(self, task, allowed, live):
        anchors = [getattr(getattr(hand, 'data', None), 'root_pose_w', None) for hand in self.world.hands]
        anchors = [None if value is None else value.detach().clone() for value in anchors]
        result = self._advance(task, allowed, live)
        sim = self.world.w.sim
        sdk_step = getattr(sim, 'current_time_step_index', None)
        sdk_time = getattr(sim, 'current_time', None)
        sdk_step = int(sdk_step) if sdk_step is not None else None
        sdk_time = float(sdk_time) if sdk_time is not None else None
        post = dict(measured_q_rad=_cpu(result.get('q')),
            measured_qd_rad_s=_cpu(result.get('qd')),
            measured_root_pose=_cpu(result.get('wrist_pose')),
            sdk_physics_step=sdk_step, sdk_timestamp_s=sdk_time,
            source='original_advance_returned_refresh_before_TensorEnv_terminal_reset',
            native_runtime_verified=self.require_native_readback)
        # advance has submitted native targets and read physics, but TensorEnv
        # has not yet consumed/reset terminal rows. This is actual command
        # evidence, separately from backend readback and caches below.
        self.active['targets'].append(_cpu(dict(
            finger=self.world.commands.flatten(1), palm=self.world.palms,
            wrist_angle=self.world.wrist_angles)))
        hands = [read_joint_target_evidence(hand) for hand in self.world.hands]
        end_step = getattr(sim, 'current_time_step_index', None)
        end_time = getattr(sim, 'current_time', None)
        post['backend_readback_sdk_physics_step'] = None if end_step is None else int(end_step)
        post['backend_readback_sdk_timestamp_s'] = None if end_time is None else float(end_time)
        post['clock_aligned'] = (sdk_step is not None and sdk_time is not None
            and math.isfinite(sdk_time) and sdk_step == post['backend_readback_sdk_physics_step']
            and sdk_time == post['backend_readback_sdk_timestamp_s'])
        source_valid = torch.full((self.task.command.shape[0],), post['clock_aligned'], dtype=torch.bool)
        for key, tail in (('measured_q_rad', (12,)), ('measured_qd_rad_s', (12,)),
                          ('measured_root_pose', (2, 7))):
            value = post[key]
            if not isinstance(value, torch.Tensor) or value.shape != (len(source_valid), *tail):
                source_valid[:] = False
            else:
                source_valid &= torch.isfinite(value).reshape(len(source_valid), -1).all(-1)
        post['source_valid'] = source_valid
        self.active['post_physics_feedback'].append(post)
        comparisons = []
        for hi, evidence in enumerate(hands):
            expected = self.world.commands[:, hi].float()
            indices = list(self.world.w.ids[hi])
            mounts = getattr(self.world.w, 'mounts', None)
            if mounts is not None and anchors[hi] is not None:
                from .tabletop_palm_mount import local_translation
                mount = mounts[hi]
                wrist = torch.cat((local_translation(self.world.root_targets[:, hi],
                    anchors[hi].to(self.world.root_targets)), self.world.wrist_angles[:, hi]), -1).float()
                expected = torch.cat((expected, wrist), -1)
                indices += list(mount.ids) + list(mount.angular_ids)
            expected = expected.detach().cpu()
            checks = {}
            for source in ('backend', 'command_cache'):
                actual = evidence[source]
                error = None
                if actual['available']:
                    difference = (torch.as_tensor(actual['values'][:, indices])-expected).abs()
                    error = difference.amax(1)
                checks[source] = dict(available=actual['available'], source=actual['source'],
                    max_error_per_env=error, expected_indices=indices,
                    expected_target_float32=expected, cast='native_float64_then_original_float32')
            comparisons.append(checks)
        self.active['sdk_readback'].append(comparisons)
        self.active['backend'].append(dict(sdk_physics_step=post['backend_readback_sdk_physics_step'],
            sdk_timestamp_s=post['backend_readback_sdk_timestamp_s'], clock_aligned=post['clock_aligned'],
            hands=[dict(joint_names=h['joint_names'],
            backend=h['backend'], dof_types=h['dof_types'], units_by_name=h['units_by_name']) for h in hands]))
        self.active['cache'].append(dict(sdk_physics_step=post['backend_readback_sdk_physics_step'],
            sdk_timestamp_s=post['backend_readback_sdk_timestamp_s'], clock_aligned=post['clock_aligned'],
            hands=[dict(joint_names=h['joint_names'],
            command_cache=h['command_cache']) for h in hands]))
        return result

    def finish_tick(self, done):
        row = self.active
        if row is None or len(row['contexts']) != 8 or len(row['targets']) != 8:
            raise ValueError('incomplete native eight-substep context; cannot fabricate missing feedback')
        metadata = row.pop('metadata')
        sample_fields = ('run_uuid', 'env_id', 'episode_uid', 'tick',
            'policy_timestamp_s', 'first_physics_step', 'previous_action')
        # Keep a full environment axis in the native record. metadata identifies
        # exactly which rows were labels at this policy boundary.
        n = row['actions'].shape[0]
        samples = {key: [None]*n for key in sample_fields}
        samples.update(actions=row['actions'], valid=torch.zeros(n, dtype=torch.bool),
            terminal=torch.zeros(n, dtype=torch.bool), event_id=['UNCLASSIFIED']*n)
        for item in metadata:
            index = item['env_id']
            for key in sample_fields:
                samples[key][index] = item[key]
            samples['terminal'][index] = bool(done[index])
            # A first terminal inside the interval excludes that partial
            # interval from command loss. Its actual raw feedback is retained.
            samples['valid'][index] = bool(row['live_count'][index] > 0)
        context = _stack(row['contexts'], 1)
        samples['valid'] &= (~context['live'] | context['source_valid']).all(1)
        # Resets/recovery may never reconnect two command segments into one.
        samples['valid'] &= ~(~context['live'][:, :-1] & context['live'][:, 1:]).any(1)
        record = dict(samples=samples, controller_start=row['controller_start'],
            physics_context=context, teacher_targets=_stack(row['targets'], 1),
            native_backend_targets=row['backend'], command_cache=row['cache'],
            sdk_readback_comparison=row['sdk_readback'], raw_measured_packets=row['raw_measured'],
            post_physics_feedback=row['post_physics_feedback'],
            events=row['events'], event_sources=dict(
                measured_palm_velocity_m_s='task.m.wrist_velocity[...,0:3];free_wrist_grasp.enable_appends_keys;'
                    'TensorTask.consume_preserves_TensorWorld.measure_body_lin_vel_w_at_mount_base_id',
                opposed_contact='original_surface_targets_native_effective_pad_loads_vectors_valid.paired.all_hands',
                native_hold_qualified='original_SurfaceLiftSuccess.window_good',
                native_runtime_verified=self.require_native_readback))
        self.records.append(record)
        self.active = None
        if self.sink is not None:
            self.sink(record)
        return record

    def close(self):
        if not self.closed:
            del self.task.control
            del self.world.advance
            self.closed = True

    def write_unqualified(self, path, *, error=None):
        """Retain actual packets even if qualification or final parity fails."""
        path = Path(path)
        with path.open('xb') as stream:
            torch.save(dict(version='dh116-native-sequence-capture-unqualified-v1',
                qualified=False, controller_contract=self.contract, error=error,
                policy_records=self.records, incomplete_policy_context=self.active,
                dataset_sha256=None, source='actual_precontrol_and_native_submission_snapshots',
                no_successful_dataset_binding_before_finalization=True), stream)
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            qualified=False, captured_policy_ticks=len(self.records))

    def finalize(self, dataset_path, successful_metadata, path):
        """Bind only selected successful sample keys to the exact saved pool."""
        if self.active is not None:
            raise ValueError('cannot finalize an incomplete native sequence tick')
        lookup = {}
        for record in self.records:
            samples = record['samples']
            for index, run in enumerate(samples['run_uuid']):
                if run is None:
                    continue
                key = (run, samples['env_id'][index], samples['episode_uid'][index], samples['tick'][index])
                if key in lookup:
                    raise ValueError('duplicate captured sample identity')
                lookup[key] = (record, index)
        selected = []
        for item in successful_metadata:
            key = tuple(item[k] for k in ('run_uuid', 'env_id', 'episode_uid', 'tick'))
            if key not in lookup:
                raise ValueError('successful sample is missing native sequence context')
            selected.append(lookup[key])
        if not selected:
            raise ValueError('no successful samples to bind')
        groups = {}
        for group in ('controller_start', 'physics_context', 'teacher_targets'):
            groups[group] = _stack([_select(rec[group], index) for rec, index in selected], 0)
            groups[group] = {k: value[:, 0] for k, value in groups[group].items()}
        samples = {}
        for field in selected[0][0]['samples']:
            values = [rec['samples'][field][index] for rec, index in selected]
            samples[field] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
        samples['previous_action'] = torch.tensor(samples['previous_action'], dtype=samples['actions'].dtype)
        events = {key: torch.stack([rec['events'][key][index] for rec, index in selected])
                  for key in selected[0][0]['events']}
        valid = torch.as_tensor(samples['valid'], dtype=torch.bool)
        if not valid.any():
            raise ValueError('no valid native command contexts in successful pool')
        state = {key: value[valid] for key, value in groups['controller_start'].items()}
        contexts = {key: value[valid, None] for key, value in groups['physics_context'].items()}
        predicted = shadow.shadow_sequence(state, samples['actions'][valid, None], contexts, self.contract)
        physics_valid = (contexts['source_valid'] & contexts['live'])[:, 0]
        def maximum_error(key, value, dtype):
            difference = (predicted[key][:, 0].to(dtype)-value[valid].to(dtype)).abs()
            mask = physics_valid.reshape(*physics_valid.shape,
                *([1]*(difference.ndim-2))).expand_as(difference)
            return float(difference[mask].max()) if bool(mask.any()) else 0.
        max64 = max(maximum_error(key, value, torch.float64)
            for key, value in groups['teacher_targets'].items())
        max32 = max(maximum_error(key, value, torch.float32)
            for key, value in groups['teacher_targets'].items())
        checked = int((contexts['source_valid'] & contexts['live']).sum())
        native_errors = {'backend': [], 'command_cache': []}
        unavailable = {'backend': 0, 'command_cache': 0}
        for record, index in selected:
            for substep, comparisons in enumerate(record['sdk_readback_comparison']):
                if not bool(record['physics_context']['live'][index, substep]):
                    continue
                for hand in comparisons:
                    for source, evidence in hand.items():
                        if evidence['available']:
                            native_errors[source].append(float(evidence['max_error_per_env'][index]))
                        else:
                            unavailable[source] += 1
        readback = {source: dict(available=bool(errors), missing_hand_substeps=unavailable[source],
            max_error=None if not errors else max(errors),
            passed=bool(errors) and not unavailable[source] and max(errors) == 0.)
            for source, errors in native_errors.items()}
        if self.require_native_readback and not all(item['passed'] for item in readback.values()):
            raise ValueError('native SDK readback parity failed or missing: '+str(readback))
        parity = dict(passed=checked > 0 and max64 == 0. and max32 == 0.,
            checked_physics_substeps=checked, max_float64_error=max64,
            max_sdk_cast_float32_error=max32,
            sdk_cast_scope='native_float64_command_cast;backend_readback_separate',
            comparison_scope='live_and_source_valid;inactive_native_reset_targets_retained_but_not_compared',
            native_sdk_readback=readback)
        parity['native_sdk_readback_required'] = self.require_native_readback
        if not parity['passed']:
            raise ValueError('Teacher conditional command parity failed: '+str(parity))
        dataset_path, path = Path(dataset_path), Path(path)
        value = dict(version='dh116-sequence-context-v1',
            dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            controller_source_sha256=self.contract['controller_source_sha256'],
            action_contract_sha256=self.contract['action_contract_sha256'],
            controller_contract=self.contract, samples=samples, events=events,
            event_sources=selected[0][0]['event_sources'],
            teacher_shadow_parity=parity, **groups,
            native_backend_targets=[dict(sample_key=tuple(item[k] for k in
                ('run_uuid', 'env_id', 'episode_uid', 'tick')), physics_substeps=[
                dict(sdk_physics_step=evidence['sdk_physics_step'],
                    sdk_timestamp_s=evidence['sdk_timestamp_s'], clock_aligned=evidence['clock_aligned'],
                    hands=[dict(joint_names=h['joint_names'], backend={**h['backend'],
                    'values': None if h['backend']['values'] is None else h['backend']['values'][index:index+1]})
                    for h in evidence['hands']]) for evidence in rec['native_backend_targets']])
                for item, (rec, index) in zip(successful_metadata, selected)],
            command_cache_source='separate_recorded_isaaclab_user_cache_not_backend')
        # Preserve full readback records separately; never relabel reconstructed
        # float32 casts as an actual SDK/backend response.
        value['command_cache'] = [dict(sample_key=tuple(item[k] for k in
            ('run_uuid', 'env_id', 'episode_uid', 'tick')), physics_substeps=rec['command_cache'], env_id=index)
            for item, (rec, index) in zip(successful_metadata, selected)]
        value['raw_measured_packets'] = [dict(sample_key=tuple(item[k] for k in
            ('run_uuid', 'env_id', 'episode_uid', 'tick')), physics_substeps=[
                _select(packet, index) for packet in rec['raw_measured_packets']],
            missingness='raw_NaN_retained;command_source_valid_is_separate')
            for item, (rec, index) in zip(successful_metadata, selected)]
        value['post_physics_feedback'] = [dict(sample_key=tuple(item[k] for k in
            ('run_uuid', 'env_id', 'episode_uid', 'tick')), physics_substeps=[
                {key: (leaf[index:index+1] if isinstance(leaf, torch.Tensor) else leaf)
                 for key, leaf in packet.items()} for packet in rec['post_physics_feedback']])
            for item, (rec, index) in zip(successful_metadata, selected)]
        with path.open('xb') as stream:
            torch.save(value, stream)
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            dataset_sha256=value['dataset_sha256'], controller_contract=self.contract,
            teacher_shadow_parity=parity, successful_sample_keys=[tuple(item[k] for k in
                ('run_uuid', 'env_id', 'episode_uid', 'tick')) for item in successful_metadata])
