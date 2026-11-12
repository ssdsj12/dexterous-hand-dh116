"""Bounded Teacher inference and exact physical-start comparison."""
import argparse
import json
import os
from pathlib import Path

import torch

from .tabletop_evaluation_trials import GRASP_STYLES, state_sha256
from .tabletop_privileged_evaluation import (
    _native_bridge, _policy_digest, _restore_reset_buffers, _save_reset_buffers,
    _staging_reset_state, prepare_cold_trial_batches,
)
from .tabletop_stereo_contract import canonical_json, digest, file_sha256, new_run, require_sha, write_json_new
from .tabletop_vision_evaluation_contract import evaluation_physics_sha256
from .tabletop_vision_evaluation_reset import _install_reset_trials, _readback_reset_trials

TEACHER_MODES = ('teacher_sampled', 'teacher_mean')
BINDING_FIELDS = ('physics_sha256', 'calibration_sha256', 'checkpoint_sha256',
    'controller_sha256', 'pre_action_371_sha256', 'runtime_seed_sha256', 'clock',
    'policy_steps_budget', 'seed')


def diagnostic_plan(manifest, identifiers, *, checkpoint_sha256, physics_sha256,
                    calibration_sha256, seed, steps):
    if type(steps) is not int or steps not in (4, 128):
        raise ValueError('bounded diagnostic requires 4 or 128 control steps')
    plan = prepare_cold_trial_batches(manifest, 2, minimum_per_style=1, schema_only=True)
    meta = plan['manifest']['metadata']
    expected = dict(source_sha256=checkpoint_sha256, physics_sha256=physics_sha256,
        camera_sha256=calibration_sha256, seed=seed)
    if (any(meta[key] != value for key, value in expected.items())
            or any(require_sha(value, key) == '0'*64 for key, value in expected.items() if key != 'seed')):
        raise ValueError('manifest must bind the exact nonzero checkpoint, physics, camera, and seed')
    by_id = {row['trial_id']: row for row in plan['manifest']['trials']}
    selected = [by_id.get(name) for name in identifiers]
    if len(selected) != 2 or tuple(row and row['grasp_style'] for row in selected) != GRASP_STYLES:
        raise ValueError('one declared pinch trial and one declared tripod trial required')
    return dict(trials=selected, manifest_sha256=plan['manifest_sha256'], seed=seed, policy_steps_budget=steps)


@torch.no_grad()
def frozen_teacher_command(model, state, *, mode):
    if mode == 'teacher_sampled':
        sample = model.sample(state)
        return dict(issued_action=sample['actions'], sampled_action=sample['actions'],
            mean_action=sample['latent_mean'].tanh(),
            policy_audit={key: sample[key] for key in ('latent_mean', 'latent_std', 'latent_actions')})
    if mode != 'teacher_mean':
        raise ValueError('only frozen sampled or deterministic Teacher execution supported')
    mean = model.latent_mean(state)
    action = mean.tanh()
    return dict(issued_action=action, sampled_action=None, mean_action=action,
        policy_audit=dict(latent_mean=mean, latent_std=model.log_std.exp().expand_as(mean), latent_actions=None))


def _initials(binding):
    actual = binding['reset_readback']['actual_states']
    identities = [(row['trial_id'], state_sha256(row['physical_state'])) for row in actual]
    if len(identities) != 2 or len({sha for _, sha in identities}) != 2 or any(
            sha != row['readback_state_sha256'] for (_, sha), row in zip(identities, actual)):
        raise ValueError('pairing requires two distinct verified complete reset readback states')
    return identities


def compare_start_bindings(*arms):
    if len(arms) not in (2, 3) or tuple(arm['mode'] for arm in arms) != (
            'teacher_sampled', 'teacher_mean', 'legacy_mpc')[:len(arms)]:
        raise ValueError('pairing requires sampled and mean Teacher first, then compatible legacy MPC')
    uids = [uid for arm in arms for uid in arm['episode_uids']]
    if (any(len(arm['episode_uids']) != 2 for arm in arms)
            or any(not isinstance(uid, str) or not uid for uid in uids)
            or len(set(uids)) != len(uids)):
        raise ValueError('pairing requires unique episode identities for all first episodes')
    first = arms[0]
    initial = _initials(first)
    for arm in arms[1:]:
        for field in BINDING_FIELDS:
            if first[field] != arm[field]:
                raise ValueError(f'paired diagnostic start differs: {field}')
        if initial != _initials(arm):
            raise ValueError('paired diagnostic initial readback state differs')
    return dict(matching_full_physical_readbacks=len(initial), compared_arms=[arm['mode'] for arm in arms],
        native_grasp_qualified=False, causal_root_qualified=False)


def load_completed_sampled_start(path):
    """Require the sampled callback's complete artifact set before pairing."""
    if path.name != 'start_binding.json':
        raise ValueError('sampled start binding filename must be start_binding.json')
    start = json.loads(path.read_text(encoding='utf-8'))
    completion = json.loads(path.with_name('completion.json').read_text(encoding='utf-8'))
    steps = completion.get('policy_steps')
    if (start.get('mode') != 'teacher_sampled'
            or completion.get('status') != 'bounded_frozen_teacher_diagnostic'
            or completion.get('mode') != 'teacher_sampled'
            or completion.get('native_execution_performed') is not True
            or completion.get('ppo_updates') != 0 or type(steps) is not int
            or not 0 < steps <= start['policy_steps_budget']):
        raise ValueError('completed bounded native sampled arm with no PPO required')
    raw = path.parent / 'native/native_physics.jsonl'
    evidence = dict(start_binding_sha256=path, raw_480hz_sha256=raw,
        pre_action_sha256=path.with_name('teacher_pre_action.pt'))
    if any(file_sha256(file) != completion.get(key) for key, file in evidence.items()):
        raise ValueError('sampled start and complete native artifacts must match their recorded SHA256')
    slots = dict(zip(start['episode_uids'], enumerate(_initials(start)), strict=True))
    if len(slots) != 2:
        raise ValueError('two distinct sampled first episode identities required')
    headers, samples = set(), set()
    with raw.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            uid = row.get('episode_uid')
            if uid not in slots or row.get('owner') != 'teacher':
                raise ValueError('sampled raw ledger contains a foreign episode or owner')
            if row['kind'] == 'episode_start':
                slot, (trial, _) = slots[uid]
                if (uid in headers or row.get('env_id') != slot or row.get('trial_id') != trial
                        or row.get('first_episode') is not True or row.get('initial_measurement_valid') is not True
                        or row.get('checkpoint_sha256') != start['checkpoint_sha256']
                        or row.get('calibration_sha256') != start['calibration_sha256']):
                    raise ValueError('sampled raw episode start disagrees with paired initial identity')
                headers.add(uid)
            elif (row['kind'] != 'sample' or uid not in headers or row.get('action_owner') != 'teacher'
                    or row.get('action_audit', {}).get('execution_mode') != 'teacher_sampled'):
                raise ValueError('sampled raw ledger lacks the correct action owner and issued mode')
            else:
                samples.add(uid)
    if samples != set(slots):
        raise ValueError('sampled raw ledger lacks physical evidence from both first episodes')
    return start


def check_frozen_teacher(path, model, ppo, native_contract, task_identity):
    """Inspect the immutable actor on CPU without restoring CUDA RNG or Adam."""
    from .tabletop_privileged_checkpoint import VERSION
    from .tabletop_privileged_cli import training_identity

    expected = training_identity(ppo, native_contract, task_identity)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (not isinstance(payload, dict) or payload.get('version') != VERSION
            or payload.get('identity') != expected
            or payload.get('history_reset') != 'clear'):
        raise ValueError('frozen Teacher checkpoint identity differs from observed native contract')
    saved = payload.get('policy', {})
    current = model.state_dict()
    if saved.keys() != current.keys() or any(
            not isinstance(saved[key], torch.Tensor)
            or saved[key].shape != tensor.shape or saved[key].dtype != tensor.dtype
            or not bool(torch.isfinite(saved[key]).all()) for key, tensor in current.items()):
        raise ValueError('frozen Teacher checkpoint policy tensors are invalid')
    return expected


def check_mpc_reference(path, expected_sha, native_contract):
    from .tabletop_vision_supervision_dataset import load_teacher_reference_dataset

    if file_sha256(path) != expected_sha or expected_sha == '0'*64:
        raise ValueError('exact legacy MPC source SHA256 required')
    return load_teacher_reference_dataset(path, native_contract)


def _tensor_digest(value):
    import hashlib

    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def _run_arm(bridge, model, plan, checkpoint, output, *, mode, expected_start):
    from .tabletop_vision_command_shadow import command_state, controller_contract, measured_context

    if not _native_bridge(bridge) or bridge.n != 2:
        raise RuntimeError('two actual native TensorWorld slots required')
    live = torch.ones(bridge.n, device=bridge.device, dtype=torch.bool)
    world, task = bridge.device_world, bridge.task
    if len(world.hands) != 2 or any(len(hand.joint_names) != 17 for hand in world.hands):
        raise RuntimeError('complete named 17-joint readback per DH116 hand required')
    bridge.bind_teacher_checkpoint(checkpoint)
    staged, stage_task = _staging_reset_state(world, task)
    _install_reset_trials(staged, stage_task, plan['trials'])
    saved = _save_reset_buffers(world, task)
    before_model = _policy_digest(model)
    samples = []
    terminals = {}
    try:
        _install_reset_trials(world, task, plan['trials'])
        bridge.bind_cold_trials(plan['trials'])
        bridge.reset()
        physical = _readback_reset_trials(world, plan['trials'])
        bindings = bridge.teacher_pre_action_identities()
        uids = [row['episode_uid'] for row in bindings]
        start = dict(mode=mode, checkpoint_sha256=bridge.checkpoint_sha256, episode_uids=uids,
            seed=plan['seed'], policy_steps_budget=plan['policy_steps_budget'],
            reset_readback=physical, authored_manifest_sha256=plan['manifest_sha256'],
            physics_sha256=evaluation_physics_sha256(bridge.contract),
            calibration_sha256=bridge.camera.calibration.sha256,
            controller_sha256=digest(controller_contract(task, world)),
            runtime_seed_sha256=digest({
                'controller': {k:_tensor_digest(v) for k,v in command_state(task, world).items()},
                'measured_context': {k:_tensor_digest(v) for k,v in measured_context(task, world, live).items()}}),
            pre_action_371_sha256=_tensor_digest(bridge.critic),
            clock=[int(world._physics_steps), int(bridge.world.sim.current_time_step_index),
                   float(bridge.world.sim.current_time)])
        write_json_new(output / 'start_binding.json', start)
        pairing = compare_start_bindings(expected_start, start) if expected_start is not None else None
        for index in range(plan['policy_steps_budget']):
            if index:
                bindings = bridge.teacher_pre_action_identities()
            observation = bridge.critic.detach().clone()
            command = frozen_teacher_command(model, observation, mode=mode)
            bridge.set_teacher_action_diagnostics(diagnostic_mode=mode, **command)
            _, _, done, _ = bridge.step(command['issued_action'])
            samples.append(dict(identities=bindings, observations=observation.cpu(),
                issued_action=command['issued_action'].detach().cpu(), done=done.cpu().clone()))
            if bool(done.any()):
                terminals = {uid: bridge._terminal_rows[uid] for uid in uids if uid in bridge._terminal_rows}
                break
        if _policy_digest(model) != before_model:
            raise RuntimeError('frozen Teacher policy parameters changed during diagnosis')
    finally:
        _restore_reset_buffers(world, task, saved)
    with (output / 'teacher_pre_action.pt').open('xb') as stream:
        torch.save(dict(samples=samples, start_binding=start, training_expert_pool=False,
            observation_scope='raw inherited 371D; per-field supervision is not qualified by this capture'), stream)
    return dict(status='bounded_frozen_teacher_diagnostic', mode=mode, pairing=pairing,
        policy_steps=len(samples), first_episode_terminals=terminals,
        stopped_at_first_batch_terminal=bool(terminals), ppo_updates=0,
        reset_buffers_restored=True, native_execution_performed=True,
        native_grasp_qualified=False, causal_root_qualified=False)


def verify_native_launch_binding(bridge, observed, inputs):
    """Match the exact on-disk native JSON semantics, including every field."""
    if canonical_json(bridge.contract) != canonical_json(observed['inherited_contract']):
        raise RuntimeError('inherited native contract differs from observed contract')
    if canonical_json(bridge.task_identity) != canonical_json(observed['task_identity']):
        raise RuntimeError('native task identity differs from observed task identity')
    for path, expected in inputs.items():
        if file_sha256(path) != expected:
            raise RuntimeError(f'immutable launch input changed: {path}')


def main(argv=None):
    from .tabletop_privileged_cli import _policy_for_evaluation, load_config
    from .tabletop_privileged_teacher import PrivilegedTeacher
    from .tabletop_stereo_native import launch_native, native_task_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=TEACHER_MODES, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--native-task-config', type=Path, required=True)
    parser.add_argument('--camera-config', type=Path, required=True)
    parser.add_argument('--observed-native-contract', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--trials', nargs=2, required=True, metavar=('PINCH', 'TRIPOD'))
    parser.add_argument('--seed', type=int, default=116)
    parser.add_argument('--steps', type=int, choices=(4,128), required=True)
    parser.add_argument('--expect-start', type=Path)
    parser.add_argument('--check-mpc-reference', type=Path)
    parser.add_argument('--mpc-source-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute-native', action='store_true')
    parser.add_argument('--enable-experiment', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'teacher_sampled' and args.expect_start is not None:
        parser.error('the sampled arm establishes the start binding')
    if (args.check_mpc_reference is None) != (args.mpc_source_sha256 is None):
        parser.error('MPC preflight requires source path and SHA256 together')
    if args.check_mpc_reference is not None and args.execute_native:
        parser.error('MPC compatibility check runs on CPU only')
    observed = json.loads(args.observed_native_contract.read_text(encoding='utf-8'))
    task = native_task_identity(json.loads(args.native_task_config.read_text(encoding='utf-8')))
    if task != observed['task_identity']:
        raise ValueError('native task config differs from the recorded task identity')
    physical = evaluation_physics_sha256(observed['inherited_contract'])
    launcher_rollout_steps = observed['inherited_contract'].get('rollout_steps')
    if type(launcher_rollout_steps) is not int or launcher_rollout_steps < 1:
        raise ValueError('observed native contract must bind the original launcher rollout_steps')
    inputs = {path:file_sha256(path) for path in (
        args.config, args.checkpoint, args.native_task_config, args.camera_config,
        args.observed_native_contract, args.manifest)}
    config, ppo = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    calibration = manifest['metadata']['camera_sha256']
    plan = diagnostic_plan(manifest, args.trials, checkpoint_sha256=inputs[args.checkpoint],
        physics_sha256=physical, calibration_sha256=calibration, seed=args.seed, steps=args.steps)
    check_frozen_teacher(args.checkpoint, PrivilegedTeacher(**config['model']), ppo,
        observed['inherited_contract'], observed['task_identity'])
    if args.check_mpc_reference is not None:
        check_mpc_reference(args.check_mpc_reference, args.mpc_source_sha256, observed['inherited_contract'])
    if not args.execute_native:
        print(json.dumps(dict(mode=args.mode, plan=plan, native_started=False,
            checkpoint_sha256=inputs[args.checkpoint], launcher_rollout_steps=launcher_rollout_steps,
            controller_contract_verified_at_native_start=False,
            mpc_reference_compatible=True if args.check_mpc_reference is not None else None)))
        return 0
    if args.output is None or not args.enable_experiment:
        parser.error('native mode requires explicit --enable-experiment and new --output')
    if args.mode == 'teacher_mean' and args.expect_start is None:
        parser.error('mean native arm requires the sampled native start binding')
    expected = None
    if args.expect_start is not None:
        sampled = args.expect_start
        for path in (sampled, sampled.with_name('completion.json'),
                     sampled.with_name('teacher_pre_action.pt'),
                     sampled.parent / 'native/native_physics.jsonl'):
            inputs[path] = file_sha256(path)
        expected = load_completed_sampled_start(sampled)
    run = new_run(args.output)
    write_json_new(run / 'preflight.json', dict(mode=args.mode, plan=plan,
        immutable_inputs={str(path):sha for path,sha in inputs.items()}, ppo_updates=0,
        launcher_rollout_steps=launcher_rollout_steps))

    def callback(bridge):
        from .tabletop_stereo_camera import StereoCamera

        verify_native_launch_binding(bridge, observed, inputs)
        bridge.bind_camera(StereoCamera.from_native(bridge.world.sim, bridge.device_world.origins,
            args.camera_config, device=bridge.device))
        if bridge.camera.calibration.sha256 != calibration:
            raise RuntimeError('real native camera calibration differs from bound manifest')
        model, _ = _policy_for_evaluation(bridge, config, ppo, args.checkpoint)
        # Model construction and camera setup can touch RNG. Seed the sampling
        # boundary explicitly after both, before the authored first reset.
        torch.manual_seed(args.seed)
        result = _run_arm(bridge, model, plan, args.checkpoint, run,
            mode=args.mode, expected_start=expected)
        if any(file_sha256(path) != sha for path,sha in inputs.items()):
            raise RuntimeError('immutable input changed during bounded diagnosis')
        # Publish complete callback evidence before handing control to the
        # protected launcher's application-close path. Its outer exit status is
        # measured independently by the process supervisor.
        bridge._raw.flush()
        artifacts = dict(start_binding_sha256=run / 'start_binding.json',
            raw_480hz_sha256=run / 'native/native_physics.jsonl',
            pre_action_sha256=run / 'teacher_pre_action.pt')
        for key,path in artifacts.items():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
            result[key] = file_sha256(path)
        result.update(outer_process_verified=False,
            launcher_rollout_steps=launcher_rollout_steps)
        completion = run / 'completion.json'
        write_json_new(completion, result)
        with completion.open('rb') as stream:
            os.fsync(stream.fileno())
        for directory in (run / 'native', run):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return result

    launch_native(callback, native_arguments=args.native_task_config, output=run / 'native',
        num_envs=2, seed=args.seed, rollout_steps=launcher_rollout_steps, enabled=True)
    return 0
