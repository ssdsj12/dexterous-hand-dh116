"""Compare two completed native Teacher/MPC collections without starting Isaac.

CPU fixtures exercise reducers only and provide no native validation count.
Success here belongs to Teacher/MPC first episodes and never to a Student.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def compare_initial_states(reference, adapter):
    mismatches = []
    tensor_fields = []
    if reference.keys() != adapter.keys():
        mismatches.append('dictionary_keys')
    for key in reference.keys() & adapter.keys():
        first, second = reference[key], adapter[key]
        if isinstance(first, torch.Tensor):
            equal = (isinstance(second, torch.Tensor) and first.dtype == second.dtype
                     and first.shape == second.shape and tensor_bytes(first) == tensor_bytes(second))
            tensor_fields.append(dict(name=key, shape=list(first.shape), dtype=str(first.dtype), bitwise_equal=equal))
        else:
            equal = type(first) is type(second) and first == second
        if not equal:
            mismatches.append(key)
    return dict(bitwise_equal=not mismatches, mismatches=sorted(mismatches), tensor_fields=tensor_fields,
                seed=reference.get('seed'), source=reference.get('source'))


def action_index(data):
    actions = data['actions']
    metadata = data['sample_metadata']
    if (not isinstance(actions, torch.Tensor) or actions.shape != (len(metadata), 24)
            or not actions.is_floating_point() or not torch.isfinite(actions).all()):
        raise ValueError('finite issued physical 24D action tape required')
    previous = {}
    result = {}
    for index, row in enumerate(metadata):
        identity = row['env_id'], row['episode_id']
        tick = row['tick']
        if tick != previous.get(identity, -1)+1:
            raise ValueError('duplicate, missing or nonzero initial supervision tick')
        previous[identity] = tick
        result[(*identity, tick)] = index
    return result


def compare_action_tapes(reference, adapter):
    first, second = action_index(reference), action_index(adapter)
    identities = {key[:2] for key in first}
    if identities != {key[:2] for key in second}:
        raise ValueError('Teacher first-episode action identities differ')
    common = sorted(first.keys() & second.keys())
    mismatches = []
    maximum = 0.
    for key in common:
        a, b = reference['actions'][first[key]], adapter['actions'][second[key]]
        maximum = max(maximum, float((a.double()-b.double()).abs().max()))
        if a.dtype != b.dtype or tensor_bytes(a) != tensor_bytes(b):
            different = [i for i in range(24) if a.dtype != b.dtype
                         or tensor_bytes(a[i]) != tensor_bytes(b[i])]
            mismatches.append(dict(key=list(key), differing_dimensions=different,
                reference_pre_action_time_s=reference['sample_metadata'][first[key]].get('simulation_time_s'),
                adapter_pre_action_time_s=adapter['sample_metadata'][second[key]].get('simulation_time_s')))
    return dict(bitwise_equal=not mismatches, common_samples=len(common),
        reference_samples=len(first), adapter_samples=len(second),
        mismatch_samples=len(mismatches), first_mismatch=mismatches[0] if mismatches else None,
        max_abs_difference=maximum,
        per_episode=[dict(env_id=env, episode_id=episode,
            common_ticks=sum(key[:2] == (env, episode) for key in common),
            first_mismatch=next((row for row in mismatches if row['key'][:2] == [env, episode]), None))
            for env, episode in sorted(identities)])


def stats(values):
    values = np.array([value for value in values if value is not None], dtype=float)
    values = values[np.isfinite(values)]
    return dict(valid_count=len(values), **{key: float(np.percentile(values, percentile)) if len(values) else None
        for key, percentile in (('p50', 50), ('p90', 90), ('p95', 95), ('maximum', 100))})


def first_episode_metrics(series, columns, identity):
    terminals = np.flatnonzero(series[:, columns['newly_terminal']] == 1)
    end = int(terminals[0])+1 if len(terminals) else len(series)
    first = series[:end]
    if not len(first) or not np.all(first[:, columns['live']] == 1):
        raise ValueError('first physical episode is empty or contains no-op samples')
    for key in ('env_id', 'episode_id'):
        if not np.all(first[:, columns[key]] == identity[key]):
            raise ValueError('physical environment or episode identity changed')
    if (not np.all(np.diff(first[:, columns['physics_step']]) == 1)
            or not np.all(np.diff(first[:, columns['timestamp_s']]) > 0)):
        raise ValueError('physical clock is nonconsecutive or nonmonotonic')
    if len(terminals) > 1 or (len(terminals) and np.any(series[end:, columns['live']] == 1)):
        raise ValueError('first terminal overwritten or closed physical episode reopened')

    def values(name):
        return first[:, columns[name]] if name in columns else None

    def peak(name):
        data = values(name)
        if data is None or not np.isfinite(data).all():
            return None
        index = int(np.argmax(data))
        return dict(**identity, value_mm=float(1000*data[index]),
            physics_step=int(values('physics_step')[index]), timestamp_s=float(values('timestamp_s')[index]))

    lift = values('object_net_lift_m')
    if not np.isfinite(lift).all():
        raise ValueError('nonfinite native object lift')
    hold = values('native_hold_s')
    hold = hold[np.isfinite(hold)] if hold is not None else np.array([])
    drift, drift_valid = values('native_anchor_drift_m'), values('native_anchor_drift_valid')
    slips = drift[(drift_valid == 1) & np.isfinite(drift)] if drift is not None and drift_valid is not None else np.array([])
    fresh = values('fresh_contact_anchor_slip_m')
    fresh_valid = values('fresh_contact_anchor_slip_valid')
    fresh_applicable = values('fresh_contact_anchor_slip_applicable')
    fresh_mask = ((fresh_valid == 1) & (fresh_applicable == 1) & np.isfinite(fresh)
        if fresh is not None and fresh_valid is not None and fresh_applicable is not None
        else np.zeros(len(first), dtype=bool))
    fresh_values = fresh[fresh_mask] if fresh is not None else np.array([])
    error = values('sdk_target_right_finger12_rad')-values('actual_q_right_finger12_rad')
    error = error[np.isfinite(error)]
    native_readback = values('right_sdk_target_native_readback')
    terminal = bool(len(terminals))
    result = dict(identity)
    result.update(physical_samples=end, terminal_recorded=terminal,
        terminal_physics_step=int(values('physics_step')[-1]) if terminal else None,
        terminal_time_s=float(values('timestamp_s')[-1]) if terminal else None,
        native_terminal_success=bool(terminal and values('native_success')[-1] == 1
                                     and values('failure_code')[-1] == 0),
        failure_code=int(values('failure_code')[-1]), lift_peak_mm=float(1000*lift.max()),
        lift_peak=peak('object_net_lift_m'), hold_max_s=float(hold.max()) if len(hold) else None,
        slip_max_mm=float(1000*slips.max()) if len(slips) else None, slip_valid_samples=len(slips),
        slip_source='native_success_observer_fixed_anchor_drift_at_its_recorded_update_cadence',
        fresh_contact_slip_max_mm=float(1000*fresh_values.max()) if len(fresh_values) else None,
        fresh_contact_slip_recorded=fresh is not None,
        fresh_contact_slip_valid_samples=len(fresh_values),
        fresh_contact_slip_applicable_samples=int(np.sum(fresh_applicable == 1)) if fresh_applicable is not None else None,
        fresh_contact_slip_source='fresh_same-substep_contact_anchor_diagnostic;separate_from_native_success_hold_window',
        right_finger12_tracking=dict(valid_samples=len(error),
            max_abs_rad=float(np.abs(error).max()) if len(error) else None,
            mean_abs_rad=float(np.abs(error).mean()) if len(error) else None,
            native_readback_samples=int(np.sum(native_readback == 1)) if native_readback is not None else None))
    pair_masks, pairs = [], []
    for hand in ('left', 'right'):
        valid = values(hand+'_contact_measurement_valid') == 1
        contact = values(hand+'_contact_valid') == 1
        loads = values(hand+'_load_n')
        loads = loads[np.isfinite(loads)] if loads is not None else np.array([])
        result[hand+'_contact'] = dict(valid_samples=int(valid.sum()), paired_samples=int((valid & contact).sum()),
            fraction=float((valid & contact).sum()/valid.sum()) if valid.any() else None)
        result[hand+'_load_mean_n'] = float(loads.mean()) if len(loads) else None
        result[hand+'_load_max_n'] = float(loads.max()) if len(loads) else None
        result[hand+'_palm_peak'] = peak(hand+'_palm_dz_m')
        pair_masks.append(valid)
        pairs.append(contact)
    valid = pair_masks[0] & pair_masks[1]
    result['bilateral_contact'] = dict(valid_samples=int(valid.sum()), paired_samples=int((valid & pairs[0] & pairs[1]).sum()),
        fraction=float((valid & pairs[0] & pairs[1]).sum()/valid.sum()) if valid.any() else None)
    return result


class TeacherArm:
    def __init__(self, folder):
        from go2_pvcnn.control.dh116.tabletop_vision_supervision_dataset import validate_dataset
        self.folder = Path(folder)
        self.report = json.loads((self.folder/'teacher_collection_report.json').read_text())
        if self.report['status'] != 'completed':
            raise ValueError('Teacher collection must be completed before analysis')
        self.dataset = torch.load(self.folder/'all_executed_supervision.pt', map_location='cpu', weights_only=True)
        validate_dataset(self.dataset, require_successful=False)
        initial_paths = [self.folder/name for name in ('initial_native_state.pt', 'init_native_state.pt')
                         if (self.folder/name).exists()]
        if len(initial_paths) != 1:
            raise ValueError('one explicit initial native state artifact required')
        self.initial = torch.load(initial_paths[0], map_location='cpu', weights_only=True)
        self.summary = json.loads((self.folder/'teacher_physics_audit_summary.json').read_text())
        if self.summary['status'] != 'recorded' or self.summary.get('error'):
            raise ValueError('completed physical first-episode audit required')
        with np.load(self.folder/'teacher_physics_audit.npz', allow_pickle=False) as data:
            self.columns = {name: i for i, name in enumerate(data['fields'].tolist())}
            self.samples = data['samples']
            self.terminals = data['terminal_samples']
            self.terminal_valid = data['terminal_valid']
        n = self.summary['environments']
        if (self.samples.shape != (self.summary['physics_batches'], n, len(self.columns))
                or self.terminals.shape != (n, len(self.columns)) or self.terminal_valid.shape != (n,)):
            raise ValueError('physical first-episode audit shape mismatch')
        self.action_keys = action_index(self.dataset)
        expected_ids = {str(row['supervision_episode_id']) for row in self.summary['episodes']}
        if set(self.dataset['episodes']) != expected_ids:
            raise ValueError('physical audit must cover exactly the recorded first supervision episodes')
        if int(self.terminal_valid.sum()) != self.summary['first_terminal_count']:
            raise ValueError('frozen first-terminal count disagrees with audit summary')
        self.episodes = []
        for identity in self.summary['episodes']:
            env = identity['env_id']
            episode = self.dataset['episodes'][str(identity['supervision_episode_id'])]
            record = first_episode_metrics(self.samples[:, env], self.columns, identity)
            record['grasp_style'] = episode['grasp_style']
            record['qualified_supervision_success'] = episode['success']
            record['recovered'] = episode['recovered']
            if bool(self.terminal_valid[env]) != record['terminal_recorded']:
                raise ValueError('frozen terminal coverage disagrees with physical rows')
            if record['terminal_recorded']:
                row = self.samples[record['physical_samples']-1, env]
                if row.tobytes() != self.terminals[env].tobytes():
                    raise ValueError('frozen terminal sample overwritten')
            for row in self.samples[:record['physical_samples'], env]:
                tick = int(row[self.columns['teacher_control_step']])
                key = env, identity['supervision_episode_id'], tick
                source = self.dataset['actions'][self.action_keys[key]].double().numpy()
                measured = row[[self.columns[f'teacher_issued_action_{i}'] for i in range(24)]]
                if source.tobytes() != measured.tobytes():
                    raise ValueError('physical Teacher issued request does not match the same supervision tick')
            self.episodes.append(record)
        self.native = (self.report.get('native_runtime_verified') is True
                       and self.summary.get('native_runtime_verified') is True
                       and self.report.get('teacher_success_is_student_success') is False
                       and all(row['native_physics'] is True and row['first_episode'] is True
                               for row in self.dataset['episodes'].values()))
        self.inputs = {name: digest(self.folder/name) for name in
            ('all_executed_supervision.pt', 'teacher_collection_report.json',
             'teacher_physics_audit.npz', 'teacher_physics_audit_summary.json', 'native_physics_substeps.pt')}
        self.inputs[initial_paths[0].name] = digest(initial_paths[0])

    def by_style(self):
        result = {}
        for style in sorted({row['grasp_style'] for row in self.episodes}):
            rows = [row for row in self.episodes if row['grasp_style'] == style]
            if len(rows) < 4:
                raise ValueError('Teacher regression requires four first episodes per grasp style')
            summary = dict(first_episodes=len(rows), completed_first_episodes=sum(row['terminal_recorded'] for row in rows),
                native_success_count=sum(row['native_terminal_success'] for row in rows) if self.native else 0,
                native_success_rate=sum(row['native_terminal_success'] for row in rows)/len(rows) if self.native else None,
                native_validation_episodes=len(rows) if self.native else 0,
                qualified_supervision_success_count=sum(row['qualified_supervision_success'] for row in rows) if self.native else 0,
                lift_mm=stats(row['lift_peak_mm'] for row in rows), hold_s=stats(row['hold_max_s'] for row in rows),
                slip_mm=stats(row['slip_max_mm'] for row in rows), slip_valid_samples=sum(row['slip_valid_samples'] for row in rows),
                fresh_contact_slip_mm=stats(row['fresh_contact_slip_max_mm'] for row in rows),
                fresh_contact_slip_valid_samples=sum(row['fresh_contact_slip_valid_samples'] for row in rows),
                fresh_contact_slip_recorded_first_episodes=sum(row['fresh_contact_slip_recorded'] for row in rows),
                fresh_contact_slip_applicable_samples=(sum(row['fresh_contact_slip_applicable_samples'] or 0 for row in rows)
                    if any(row['fresh_contact_slip_applicable_samples'] is not None for row in rows) else None),
                right_finger12_max_abs_error_rad=stats(row['right_finger12_tracking']['max_abs_rad'] for row in rows),
                right_finger12_native_readback_samples=sum(row['right_finger12_tracking']['native_readback_samples'] or 0 for row in rows),
                terminal_failure_code_counts=dict(Counter(str(row['failure_code']) for row in rows if row['terminal_recorded'])))
            for hand in ('left', 'right', 'bilateral'):
                contacts = [row[hand+'_contact'] for row in rows]
                valid = sum(row['valid_samples'] for row in contacts)
                summary[hand+'_opposed_contact_fraction'] = sum(row['paired_samples'] for row in contacts)/valid if valid else None
            for hand in ('left', 'right'):
                summary[hand+'_mean_load_n'] = stats(row[hand+'_load_mean_n'] for row in rows)
                summary[hand+'_max_load_n'] = stats(row[hand+'_load_max_n'] for row in rows)
            result[style] = summary
        return result


def compare_physics(reference, adapter):
    if [row['env_id'] for row in reference.episodes] != [row['env_id'] for row in adapter.episodes]:
        raise ValueError('physical episode environment order differs')
    first = torch.load(reference.folder/'native_physics_substeps.pt', map_location='cpu', weights_only=True)
    second = torch.load(adapter.folder/'native_physics_substeps.pt', map_location='cpu', weights_only=True)
    if first.keys() != second.keys():
        raise ValueError('native raw physical trace fields differ')
    results = []
    for a, b in zip(reference.episodes, adapter.episodes):
        env = a['env_id']
        if (a['episode_id'], a['supervision_episode_id'], a['grasp_style']) != (
                b['episode_id'], b['supervision_episode_id'], b['grasp_style']):
            raise ValueError('physical first episode identity or grasp style differs')
        count = min(a['physical_samples'], b['physical_samples'])
        fields = {}
        for key in first:
            av, bv = first[key][:count, env], second[key][:count, env]
            if av.shape != bv.shape or av.shape[0] != count:
                raise ValueError('native raw trace shorter than the identified first episode')
            finite = torch.isfinite(av) & torch.isfinite(bv)
            delta = (av.double()-bv.double()).abs()[finite]
            fields[key] = dict(bitwise_equal=av.dtype == bv.dtype and tensor_bytes(av) == tensor_bytes(bv),
                max_abs_difference=float(delta.max()) if delta.numel() else None,
                finite_pairs=int(finite.sum()))
            if not fields[key]['bitwise_equal']:
                byte_differences = (av.contiguous().reshape(count, -1).view(torch.uint8)
                    != bv.contiguous().reshape(count, -1).view(torch.uint8)).any(1) if av.dtype == bv.dtype else torch.ones(count, dtype=torch.bool)
                index = int(byte_differences.nonzero()[0])
                fields[key]['first_different_sample'] = dict(
                    physics_step=int(reference.samples[index, env, reference.columns['physics_step']]),
                    timestamp_s=float(reference.samples[index, env, reference.columns['timestamp_s']]),
                    teacher_control_step=int(reference.samples[index, env, reference.columns['teacher_control_step']]))
        # This binds the legacy unkeyed PT prefix to the independently keyed NPZ.
        for arm, raw in ((reference, first), (adapter, second)):
            bottom = arm.samples[:count, env, arm.columns['object_bottom_m']]
            if not np.array_equal(raw['bottom'][:count, env].double().numpy(), bottom):
                raise ValueError('legacy raw trace and identified NPZ physical order disagree')
        results.append(dict(env_id=env, supervision_episode_id=a['supervision_episode_id'],
            grasp_style=a['grasp_style'], common_physical_substeps=count, fields=fields))
    return dict(episodes=results, all_common_fields_bitwise_equal=all(
        field['bitwise_equal'] for row in results for field in row['fields'].values()))


def divergence_chronology(actions, physics):
    """Order measured-state and issued-request differences without assigning cause."""
    state_fields = ('bottom', 'wrist_pose_w', 'q_measured', 'qd_measured',
                    'hand_total_loads_n', 'effective_pad_loads_n', 'effective_pad_vectors_w')
    result = []
    for episode in physics['episodes']:
        action = next(row for row in actions['per_episode'] if row['env_id'] == episode['env_id']
                      and row['episode_id'] == episode['supervision_episode_id'])
        candidates = [(key, value['first_different_sample']) for key, value in episode['fields'].items()
                      if key in state_fields and 'first_different_sample' in value]
        first = min(candidates, key=lambda item: item[1]['timestamp_s'], default=None)
        mismatch = action['first_mismatch']
        action_time = mismatch['reference_pre_action_time_s'] if mismatch else None
        order = 'no_issued_difference_on_common_window'
        if mismatch:
            order = 'insufficient_clock_coverage'
            if first is not None and action_time is not None:
                state_time = first[1]['timestamp_s']
                order = ('measured_state_difference_precedes_issued_difference' if state_time < action_time
                         else 'issued_difference_precedes_measured_state_difference' if action_time < state_time
                         else 'same_recorded_time')
        result.append(dict(env_id=episode['env_id'], supervision_episode_id=episode['supervision_episode_id'],
            first_measured_state_difference=dict(field=first[0], **first[1]) if first else None,
            first_issued_difference=mismatch, observed_order=order, causal_attribution=None))
    return result


def validate_teacher_regression_proof(value):
    """Check the T5 gate and every bound native input file before a new run.

    This validates a previously produced audit and its immutable inputs. It
    does not execute an experiment or turn CPU fixtures into native evidence.
    """
    if (value.get('version') != 'dh116-stage5-teacher-regression-v1'
            or value.get('counts_as_student_success') is not False
            or value.get('success_contract') != dict(net_lift_m=.03, hold_s=5., fixed_anchor_slip_m=.0005)):
        raise ValueError('exact T5 Teacher regression and original success contract required')
    pairing = value.get('paired_source_constraints', {})
    if (value.get('initial_state', {}).get('bitwise_equal') is not True
            or any(pairing.get(key) is not True for key in ('same_initial_state', 'same_seed',
                'same_teacher_source', 'same_visual_contract', 'same_controller_profile'))):
        raise ValueError('T5 paired initial/source/seed/contract evidence required')
    native = value.get('native_runtime_attested_by_collector', {})
    for arm in ('reference', 'adapter'):
        pinch = value.get('by_style', {}).get(arm, {}).get('pinch_thumb_index', {})
        if (native.get(arm) is not True or pinch.get('native_success_count') != 4
                or pinch.get('qualified_supervision_success_count') != 4
                or pinch.get('completed_first_episodes') != 4 or pinch.get('first_episodes') != 4):
            raise ValueError('four genuine completed successful pinch first episodes required in each T5 arm')
    internal = value.get('adapter_internal_identity_execution', {})
    semantic = internal.get('contract', {}).get('physical_action_adapter', {})
    if (internal.get('verified_zero_execution_difference') is not True
            or internal.get('max_executed_action_difference') != 0
            or semantic.get('actor_size') != 3844 or semantic.get('critic_size') != 371
            or semantic.get('action', {}).get('size') != 24):
        raise ValueError('T5 Adapter identity execution and original 3844/371/24 contract required')
    if value.get('pinch_teacher_four_of_four_regression_passed') is not True:
        raise ValueError('T5 four-of-four regression result is not passed')
    required = {'all_executed_supervision.pt', 'teacher_collection_report.json',
        'teacher_physics_audit.npz', 'teacher_physics_audit_summary.json', 'native_physics_substeps.pt'}
    checked = []
    for arm in ('reference', 'adapter'):
        folder = Path(value.get('input_directories', {}).get(arm, ''))
        hashes = value.get('input_hashes', {}).get(arm, {})
        expected = required | ({'physical_action_adapter_contract.json'} if arm == 'adapter' else set())
        initial_names = set(hashes) & {'initial_native_state.pt', 'init_native_state.pt'}
        if not folder.is_absolute() or not expected <= hashes.keys() or len(initial_names) != 1:
            raise ValueError('T5 absolute native input directories and complete SHA bindings required')
        for name, expected_sha in hashes.items():
            if Path(name).name != name or not (folder/name).is_file() or digest(folder/name) != expected_sha:
                raise ValueError('T5 native input file SHA mismatch or missing file: '+str(folder/name))
            checked.append(str((folder/name).resolve()))
    return dict(verified=True, checked_input_files=checked,
        teacher_source_sha256=value.get('teacher_source_sha256'),
        adapter_contract_sha256=semantic.get('sha256'), counts_as_student_success=False)


def audit(reference_folder, adapter_folder):
    from go2_pvcnn.control.dh116.tabletop_vision_supervision_dataset import _canonical_visual_contract
    reference, adapter = TeacherArm(reference_folder), TeacherArm(adapter_folder)
    initial = compare_initial_states(reference.initial, adapter.initial)
    if not initial['bitwise_equal']:
        raise ValueError('actual initial native state or seed differs')
    source_sha = reference.dataset['teacher_source_sha256']
    if not (source_sha == adapter.dataset['teacher_source_sha256'] == reference.report['source_sha256']
            == adapter.report['source_sha256']):
        raise ValueError('actual Teacher reference source SHA differs')
    if (reference.report['mode'], reference.report.get('mpc_profile')) != (
            adapter.report['mode'], adapter.report.get('mpc_profile')):
        raise ValueError('Teacher/MPC controller mode or profile differs')
    if _canonical_visual_contract(reference.dataset['visual_contract']) != _canonical_visual_contract(adapter.dataset['visual_contract']):
        raise ValueError('camera, physical or action semantics differ')
    actions = compare_action_tapes(reference.dataset, adapter.dataset)
    physics = compare_physics(reference, adapter)
    adapter_contract_path = adapter.folder/'physical_action_adapter_contract.json'
    adapter_contract = json.loads(adapter_contract_path.read_text())
    semantic = adapter_contract['physical_action_adapter']
    if (semantic['action'] != reference.dataset['visual_contract']['action_contract']
            or semantic['actor_size'] != 3844 or semantic['critic_size'] != 371):
        raise ValueError('Adapter physical semantics or model dimensions differ from the original contract')
    internal_difference = adapter.report.get('physical_sdk_adapter_max_executed_action_difference')
    semantic_roundtrip = adapter.report.get('physical_sdk_adapter_max_semantic_roundtrip_difference')
    adapter_identity_execution = type(internal_difference) in (int, float) and internal_difference == 0
    adapter.inputs[adapter_contract_path.name] = digest(adapter_contract_path)
    groups = dict(reference=reference.by_style(), adapter=adapter.by_style())
    pinch = 'pinch_thumb_index'
    passed = (reference.native and adapter.native and adapter_identity_execution
              and groups['reference'].get(pinch, {}).get('native_success_count') == 4
              and groups['adapter'].get(pinch, {}).get('native_success_count') == 4
              and groups['reference'].get(pinch, {}).get('qualified_supervision_success_count') == 4
              and groups['adapter'].get(pinch, {}).get('qualified_supervision_success_count') == 4
              and all(groups[arm].get(pinch, {}).get('first_episodes') == 4
                      and groups[arm][pinch]['completed_first_episodes'] == 4 for arm in ('reference', 'adapter')))
    return dict(version='dh116-stage5-teacher-regression-v1', source_kind='Teacher_or_MPC;never_Student',
        native_runtime_attested_by_collector=dict(reference=reference.native, adapter=adapter.native),
        initial_state=initial, teacher_source_sha256=source_sha, actions=actions, physics=physics,
        paired_source_constraints=dict(same_initial_state=True, same_seed=True,
            same_teacher_source=True, same_visual_contract=True, same_controller_profile=True),
        audit_source_sha256=digest(__file__),
        divergence_chronology=divergence_chronology(actions, physics),
        source_profiles={key: dict(mode=arm.report['mode'], mpc_profile=arm.report.get('mpc_profile'))
                         for key, arm in (('reference', reference), ('adapter', adapter))},
        adapter_internal_identity_execution=dict(verified_zero_execution_difference=adapter_identity_execution,
            max_executed_action_difference=internal_difference,
            max_semantic_roundtrip_difference=semantic_roundtrip,
            contract=adapter_contract),
        by_style=groups, episodes=dict(reference=reference.episodes, adapter=adapter.episodes),
        input_hashes=dict(reference=reference.inputs, adapter=adapter.inputs),
        input_directories=dict(reference=str(reference.folder.resolve()), adapter=str(adapter.folder.resolve())),
        pinch_teacher_four_of_four_regression_passed=passed, counts_as_student_success=False,
        success_contract=dict(net_lift_m=.03, hold_s=5., fixed_anchor_slip_m=.0005))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.reference, args.adapter)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write('\n')
