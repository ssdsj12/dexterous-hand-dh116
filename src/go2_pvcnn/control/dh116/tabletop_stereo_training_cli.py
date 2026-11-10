"""Independent stereo BC, Teacher candidate collection and online DAgger CLIs.

Successful demonstration admission and formal online DAgger are deliberately
separate from collecting a diagnostic candidate. No command silently upgrades
candidate/CPU data into a native success pool.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .tabletop_stereo_contract import (
    ROOT, digest, file_sha256, make_contract, new_run, require_opt_in, write_json_new,
)


BC_VERSION = 'dh116-stereo-rgb-bc-artifact-v1'
CANDIDATE_VERSION = 'dh116-stereo-teacher-candidate-records-v1'


def _formal_collection_admission(teacher_proof, teacher_checkpoint, stereo_proof, camera_config):
    from .tabletop_stereo_promotion import require_teacher_qualification, verify_stereo_qualification
    from .tabletop_stereo_calibration import StereoCalibration
    teacher = require_teacher_qualification(teacher_proof, file_sha256(teacher_checkpoint))
    stereo = verify_stereo_qualification(stereo_proof)
    selected = StereoCalibration.from_json(camera_config)
    if stereo['calibration_sha256'] != selected.sha256:
        raise ValueError('formal collection stereo proof calibration differs from the selected camera/resolution')
    from .tabletop_stereo_preprocessing import processing_contract
    measured = deepcopy(stereo.get('depth_estimator_config'))
    if isinstance(measured, dict):
        measured.pop('image_processing_batch_size', None)
    if measured != processing_contract(camera_config)['depth_estimator']:
        raise ValueError('formal collection matcher differs from the verified raw stereo processing')
    return dict(teacher=teacher, stereo=stereo)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _model_config(path):
    from .tabletop_stereo_student_ppo import normalize_network_config
    value = _json(path)
    return normalize_network_config(value.get('network', value))


def dataset_content_sha256(value):
    """Typed, length-delimited tensor/metadata digest independent of zip headers."""
    hasher = hashlib.sha256()
    def part(raw):
        hasher.update(len(raw).to_bytes(8, 'big')); hasher.update(raw)
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            part(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)]).encode())
            part(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            part(b'dict')
            for key in sorted(item):
                part(str(key).encode('utf-8')); visit(item[key])
        elif isinstance(item, (list, tuple)):
            part(('list:'+str(len(item))).encode())
            for child in item: visit(child)
        elif isinstance(item, Path):
            part(json.dumps(['path', str(item)]).encode('utf-8'))
        else:
            part(json.dumps([type(item).__name__, item], sort_keys=True, allow_nan=False).encode('utf-8'))
    visit(value)
    return hasher.hexdigest()


def _bc_provenance(data, config, provenance, schema_only):
    value = {} if provenance is None else deepcopy(provenance)
    if not schema_only:
        if 'dataset_artifact' not in value or 'config_artifact' not in value or 'network_template_artifact' not in value:
            raise ValueError('formal BC requires actual dataset/config/network artifact SHA bindings')
    for key in ('dataset_artifact', 'config_artifact', 'network_template_artifact'):
        if key in value:
            binding = value[key]
            if not isinstance(binding, dict) or file_sha256(binding['path']) != binding.get('sha256'):
                raise ValueError('BC source artifact SHA mismatch: '+key)
    content = dataset_content_sha256(data)
    if 'dataset_artifact' in value:
        from .tabletop_stereo_supervision_dataset import load_dataset
        saved = load_dataset(value['dataset_artifact']['path'], allow_schema_only=schema_only)
        if dataset_content_sha256(saved) != content:
            raise ValueError('BC data differs from the exact bound source dataset')
    value.update(dataset_content_sha256=content, resolved_bc_config_sha256=digest(config),
        scope='cpu_fixture_content_only' if schema_only else 'SHA_bound_native_dataset',
        source_sha256={name: file_sha256(Path(__file__).with_name(name)) for name in (
            'tabletop_stereo_training_cli.py', 'tabletop_stereo_teacher_collector.py',
            'tabletop_stereo_supervision_dataset.py', 'tabletop_stereo_bc.py',
            'tabletop_stereo_losses.py', 'tabletop_stereo_encoder.py',
            'tabletop_stereo_temporal.py', 'tabletop_stereo_state_estimator.py',
            'tabletop_stereo_observation.py')})
    return value


def _dataset_processing(data, *, schema_only):
    """Bind each recorded raw pipeline; calibration alone does not identify it."""
    from .tabletop_stereo_preprocessing import validate_processing_contract
    contracts = [row['identity'].get('observation_preprocessor') for row in data['rows']]
    if all(value is None for value in contracts) and schema_only:
        return None  # CPU history-only fixtures cannot produce a raw RGB deployment.
    if not contracts or any(value is None for value in contracts):
        raise ValueError('formal BC requires the recorded raw stereo processing contract in every sample')
    first = validate_processing_contract(contracts[0])
    for value in contracts:
        validate_processing_contract(value, expected=first)
    if first['calibration_sha256'] != data['contract']['calibration_sha256']:
        raise ValueError('dataset raw processing calibration differs from its observation contract')
    return deepcopy(first)


def _bc_validation(actor, data, indices, device):
    from .tabletop_privileged_dataset import masked_group_normalized_loss
    from .tabletop_stereo_supervision_dataset import batch_dataset
    if not indices:
        return None
    total = 0.
    with torch.no_grad():
        for start in range(0, len(indices), 16):
            batch = batch_dataset(data, indices[start:start+16], device=device)
            loss = masked_group_normalized_loss(actor.deterministic(batch['history']),
                batch['teacher_actions'], batch['action_valid'])
            total += float(loss)*len(batch['indices'])
    return total/len(indices)


def train_bc(data, network_config, bc_config, *, output, steps=100, batch_size=32,
             heldout_initial_sha256=(), seed=116, schema_only=False, device='cpu', provenance=None):
    from .tabletop_stereo_supervision_dataset import validate_dataset, batch_dataset, split_by_initial_condition
    from .tabletop_stereo_student_ppo import make_stereo_actor, normalize_network_config
    from .tabletop_stereo_bc import StereoBCTrainer
    from .tabletop_stereo_checkpoint import save_actor_package
    validate_dataset(data, allow_schema_only=schema_only)
    if type(steps) is not int or type(batch_size) is not int or min(steps, batch_size) < 1:
        raise ValueError('positive bounded BC update/batch budgets required')
    if not schema_only and data['purpose'] != 'successful_bc':
        raise ValueError('initial BC requires the qualified successful demonstration pool')
    if not heldout_initial_sha256 and not schema_only:
        raise ValueError('independent physical-initial-condition holdout required before formal BC')
    preprocessor = _dataset_processing(data, schema_only=schema_only)
    provenance = _bc_provenance(data, bc_config, provenance, schema_only)
    provenance['processing_sha256'] = None if preprocessor is None else preprocessor['processing_sha256']
    split = split_by_initial_condition(data, heldout_initial_condition_sha256=heldout_initial_sha256)
    torch.manual_seed(seed)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    network_config = deepcopy(network_config)
    first_image = data['rows'][0]['history']['left_rgb']
    height, width = first_image.shape[-2:]
    if any(tuple(row['history']['left_rgb'].shape[-2:]) != (height, width) for row in data['rows']):
        raise ValueError('one calibrated resolution is required for each BC dataset')
    # A resource template is not an instruction to resize sensor images.
    # Resolve the actual validated dataset size into a new model contract.
    network_config['image_resolution_wh'] = [width, height]
    network = normalize_network_config(network_config)
    actor = make_stereo_actor(network).to(device)
    trainer = StereoBCTrainer(actor, bc_config, schema_only=schema_only)
    output = Path(output)
    calibration = data['contract']['calibration_sha256']
    contract = make_contract('bc', dict(network=network, bc=deepcopy(bc_config),
        dataset_contract_sha256=data['contract']['contract_sha256'], schema_only=data['schema_only'],
        split=split, provenance=provenance, preprocessor=preprocessor), calibration_sha256=calibration)
    write_json_new(output / 'bc_contract.json', contract)
    write_json_new(output / 'split.json', split)
    baseline = _bc_validation(actor, data, split['heldout_indices'], device)
    best, best_path = float('inf'), None
    losses = []
    with (output / 'bc_updates.jsonl').open('x', encoding='utf-8') as log:
        for step in range(steps):
            pool = split['train_indices']
            selected = torch.randint(len(pool), (min(batch_size, len(pool)),), generator=generator).tolist()
            indices = [pool[i] for i in selected]
            metrics = trainer.step(batch_dataset(data, indices, device=device))
            validation = _bc_validation(actor, data, split['heldout_indices'], device)
            row = dict(update=step+1, validation_action_loss=validation, **metrics)
            log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
            losses.append(row)
            # Keep intermediate models; a heldout score selects weights only.
            # Action loss does not qualify grasping or substitute task success.
            artifact = dict(version=BC_VERSION, contract=contract, network_config=network,
                actor={k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                bc_optimizer=deepcopy(trainer.optimizer.state_dict()), update=step+1,
                teacher_source_sha256=data['source_sha256'], schema_only=data['schema_only'],
                provenance=provenance,
                split=split, validation_action_loss=validation, native_task_qualified=False,
                torch_rng=torch.get_rng_state())
            path = output / f'bc_update_{step+1:06d}.pt'
            with path.open('xb') as stream:
                torch.save(artifact, stream)
            if validation is not None and validation < best:
                best, best_path = validation, path
    selected_path = best_path if best_path is not None else path
    if best_path is not None:
        saved = torch.load(best_path, map_location='cpu', weights_only=False)
        actor.load_state_dict(saved['actor'], strict=True)
    package = save_actor_package(output / 'stereo_bc_actor.pt', actor, contract, network_config=network)
    report = dict(status='independent_stereo_BC_completed', steps=steps, samples=len(data['rows']),
        heldout_samples=len(split['heldout_indices']), baseline_heldout_action_loss=baseline,
        selected_checkpoint=str(selected_path), selected_checkpoint_sha256=file_sha256(selected_path),
        selection='lowest_heldout_action_loss' if best_path else 'last_update_diagnostic_no_holdout',
        actor_package=package, schema_only=data['schema_only'], native_task_qualified=False,
        provenance=provenance,
        validation_scope='offline_BC_action_error_not_native_task_success')
    write_json_new(output / 'bc_report.json', report)
    return report


@torch.no_grad()
def collect_candidate_reference(environment, teacher, steps):
    """Unqualified Teacher-only diagnostic; deliberately no dataset admission."""
    from .tabletop_stereo_teacher_collector import _validate_step, clone_tree
    records = []
    for _ in range(steps):
        before = environment.snapshot().clone().validate()
        mean = teacher.deterministic(before.privileged)
        mu, std = teacher.latent_mean(before.privileged), teacher.log_std.exp().expand_as(mean)
        environment.set_action_owners(['teacher']*len(mean))
        after = environment.step(mean.clone())
        execution = _validate_step(before, after, mean, 1/60)
        record = dict(sensor_history=before.sensor_history.as_dict(), privileged=before.privileged,
            privileged_valid=before.privileged_valid, identities=before.identities,
            privileged_identities=before.identities, extra_labels=before.extra_labels,
            label_validity=before.label_validity, state_sources=before.state_sources,
            force_qualification=None, actor_raw=dict(teacher=mu, student=None), teacher_mean=mean,
            teacher_sample=None, teacher_std=std, student_nominal=None,
            issued_actions=after.issued_actions, applied_targets=after.applied_targets,
            execution=execution, reward=after.reward, done=after.done, info=after.info,
            action_owner=['teacher']*len(mean), mode='teacher_candidate_reference',
            teacher_probability=1., dagger_student_samples=0, teacher_query_count=len(mean),
            candidate_only=True, schema_only=False, native_qualified=False, teacher_qualification=None)
        records.append(clone_tree(record, cpu=True))
    return records


def admit_candidates(candidate_path, qualification, output):
    from .tabletop_stereo_supervision_dataset import build_dataset, save_dataset
    from .tabletop_stereo_promotion import require_teacher_qualification
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=False)
    if candidate.get('version') != CANDIDATE_VERSION:
        raise ValueError('versioned unmodified stereo Teacher candidates required')
    if isinstance(qualification, Path):
        qualification = str(qualification.resolve())
    require_teacher_qualification(qualification, candidate['teacher_checkpoint_sha256'])
    records = deepcopy(candidate['records'])
    # Conversion changes classification only after the common raw proof gate.
    # Every collected identity must also appear in its own complete success proof.
    for record in records:
        record['mode'] = 'teacher_reference'
        record['teacher_qualification'] = qualification
        record['candidate_only'] = False
    dataset = build_dataset(records, source_sha256=candidate['teacher_checkpoint_sha256'],
        qualification=qualification, schema_only=False)
    return save_dataset(dataset, output)


def run_online_dagger(environment, student, teacher, *, config, source_sha256,
                      qualification, output, rounds=1, steps=32, batch_size=32, epochs=1,
                      stage='D4', schema_only=False, actor_package=None, provenance=None):
    from .tabletop_stereo_dagger import run
    from .tabletop_stereo_dagger_dataset import DAggerDataset
    from .tabletop_stereo_dagger_trainer import StereoDAggerTrainer
    from .tabletop_stereo_checkpoint import save_actor_package
    if any(type(v) is not int or v < 1 for v in (rounds, steps, batch_size, epochs)):
        raise ValueError('positive bounded DAgger budgets required')
    if stage == 'D0':
        raise ValueError('D0 is a Teacher reference; use the separate candidate collector')
    output = Path(output)
    probability = config['teacher_probability_by_stage'][stage]
    generator = torch.Generator().manual_seed(config.get('seed', 116))
    if isinstance(qualification, Path):
        qualification = str(qualification.resolve())
    pool = DAggerDataset(source_sha256=source_sha256, schema_only=schema_only, qualification=qualification)
    preprocessor = deepcopy(getattr(environment, 'preprocessing', None))
    if not schema_only:
        from .tabletop_stereo_preprocessing import require_raw_processing
        require_raw_processing(student.deployment_contract, environment.camera_config)
        if actor_package is None:
            raise ValueError('formal DAgger requires the actual input Actor package artifact')
    input_actor = None if actor_package is None else dict(path=str(Path(actor_package).resolve()),
        sha256=file_sha256(actor_package))
    source = {} if provenance is None else deepcopy(provenance)
    source.update(teacher_checkpoint_sha256=source_sha256, input_actor_package_artifact=input_actor,
        resolved_dagger_config_sha256=digest(config),
        source_sha256={name: file_sha256(Path(__file__).with_name(name)) for name in (
            'tabletop_stereo_training_cli.py', 'tabletop_stereo_dagger.py',
            'tabletop_stereo_dagger_dataset.py', 'tabletop_stereo_dagger_trainer.py',
            'tabletop_stereo_bc.py', 'tabletop_stereo_teacher_collector.py')})
    bc = StereoDAggerTrainer(student, schema_only=schema_only)
    bridge = getattr(environment, 'bridge', None)
    if bridge is not None:
        # A mixed, updated DAgger policy is not either frozen model identity.
        bridge.set_execution_context(owner='dagger', checkpoint_sha256=None)
        bridge.set_runtime_descriptor(dict(execution_mode='online_DAgger',
            teacher_queries_enabled=True, teacher_execution_enabled=probability > 0,
            mpc_enabled=False, critic_deployed=False, critic_training=False,
            privileged_actor_inputs=False, actor_input_source='stereo_RGB_and_proprio_only',
            depth_source='computed_from_synchronized_left_right_RGB',
            teacher_checkpoint_sha256=source_sha256, initial_actor_artifact=input_actor,
            model_checkpoint_sha256=None, model_identity_scope='round_frozen_policy;weights_change_between_rounds',
            native_qualified=False))
    reports = []
    for index in range(rounds):
        collected = run(environment, student, teacher, steps, stage=stage,
            teacher_probability=probability, teacher_qualification=qualification,
            source_sha256=source_sha256, schema_only=schema_only, generator=generator)
        pool.append(collected['records'])
        training = bc.train(pool, batch_size=batch_size, epochs=epochs, generator=generator)
        pool_path = output / f'dagger_pool_{index+1:04d}.pt'
        pool.save(pool_path)
        round_source = deepcopy(source)
        round_source.update(input_actor_package_artifact=input_actor,
            dataset_artifact=dict(path=str(pool_path.resolve()), sha256=file_sha256(pool_path)))
        contract = make_contract('dagger', dict(network=student.network_config, stage=stage,
            round=index+1, schema_only=schema_only, preprocessor=preprocessor, provenance=round_source,
            resolved_dagger_config=deepcopy(config)),
            calibration_sha256=environment.snapshot().identities[0]['calibration_sha256'])
        actor_path = output / f'dagger_actor_{index+1:04d}.pt'
        saved_actor = save_actor_package(actor_path, student, contract,
            network_config=student.network_config)
        write_json_new(output / f'dagger_contract_{index+1:04d}.json', contract)
        input_actor = dict(path=str(actor_path.resolve()), sha256=file_sha256(actor_path))
        report = dict(round=index+1, teacher_queries=collected['teacher_query_count'],
            student_owned_samples=collected['student_owned_samples'], stage=stage,
            aggregated_samples=len(pool), training=training, schema_only=schema_only,
            native_task_qualified=False, ppo_storage_used=False, provenance=round_source,
            actor_package=saved_actor, actor_package_artifact=input_actor)
        reports.append(report)
        write_json_new(output / f'dagger_round_{index+1:04d}.json', report)
    return dict(rounds=reports, qualification='visited_state_training_not_native_task_success')


def main(mode='bc', argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--network-config', type=Path, default=ROOT / 'configs/dh116_training/stereo_network_v1.json')
    parser.add_argument('--camera-config', type=Path, default=ROOT / 'configs/dh116_training/stereo_camera_640x480_v2.json')
    parser.add_argument('--native-task-config', type=Path, default=ROOT / 'configs/dh116_training/stereo_native_task_v1.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--teacher-checkpoint', type=Path)
    parser.add_argument('--teacher-config', type=Path, default=ROOT / 'configs/dh116_training/privileged_teacher_ppo_v1.json')
    parser.add_argument('--teacher-proof', type=Path)
    parser.add_argument('--stereo-proof', type=Path)
    parser.add_argument('--actor-package', type=Path)
    parser.add_argument('--heldout-initial-sha', action='append', default=[])
    parser.add_argument('--seed', type=int, default=116)
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--stage', choices=('D1','D2','D3','D4'), default='D4')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--candidate-only', action='store_true')
    parser.add_argument('--schema-only', action='store_true')
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--enable-experiment', action='store_true')
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args(argv)
    config = _json(args.config)
    if mode == 'bc' and config.get('schema_version') != 'dh116-stereo-bc-v1':
        parser.error('versioned stereo BC config required')
    if mode == 'dagger' and config.get('schema_version') != 'dh116-stereo-dagger-v1':
        parser.error('versioned online DAgger config required')
    if any(v < 1 for v in (args.steps, args.batch_size, args.epochs, args.rounds, args.num_envs)):
        parser.error('positive execution budgets required')
    if args.check_config:
        print(json.dumps(dict(mode=mode, validated=True, native_started=False, config=config)))
        return 0
    require_opt_in(args.enable_experiment)
    if args.output is None:
        parser.error('exclusive new --output required')
    if mode == 'admit':
        if not args.candidate or not args.teacher_proof:
            parser.error('candidate and SHA-bound raw success proof required')
        run = new_run(args.output)
        report = admit_candidates(args.candidate, args.teacher_proof, run / 'successful_stereo_dataset.pt')
        write_json_new(run / 'admission.json', report)
        return 0
    if mode == 'bc':
        if args.dataset is None:
            parser.error('--dataset required')
        from .tabletop_stereo_supervision_dataset import load_dataset
        data = load_dataset(args.dataset, allow_schema_only=args.schema_only)
        network = _model_config(args.network_config)
        run = new_run(args.output)
        train_bc(data, network, config, output=run, steps=args.steps, batch_size=args.batch_size,
            heldout_initial_sha256=args.heldout_initial_sha, schema_only=args.schema_only, seed=args.seed, device=args.device,
            provenance={name: {'path': str(path.resolve()), 'sha256': file_sha256(path)}
                for name, path in (('dataset_artifact', args.dataset), ('config_artifact', args.config),
                                   ('network_template_artifact', args.network_config))})
        return 0
    if not args.execute_native or args.schema_only or not args.teacher_checkpoint:
        parser.error('native collection needs explicit native execution and a frozen Teacher; CPU fixtures cannot claim native')
    teacher_sha = file_sha256(args.teacher_checkpoint)
    if mode == 'dagger' or not args.candidate_only:
        if not args.teacher_proof or not args.stereo_proof:
            parser.error('formal Teacher/DAgger collection requires native Teacher and stereo proofs')
        _formal_collection_admission(args.teacher_proof, args.teacher_checkpoint,
            args.stereo_proof, args.camera_config)
    if mode == 'dagger' and args.actor_package is None:
        parser.error('online DAgger requires an independent Student actor package')
    run = new_run(args.output)
    from .tabletop_privileged_cli import load_config, _policy_for_evaluation
    from .tabletop_stereo_native import launch_native
    from .tabletop_stereo_runtime import NativeStereoEnvironment
    teacher_config, ppo = load_config(args.teacher_config)

    def callback(bridge):
        teacher, _ = _policy_for_evaluation(bridge, teacher_config, ppo, args.teacher_checkpoint)
        environment = NativeStereoEnvironment(bridge, args.camera_config,
            owner='dagger' if mode == 'dagger' else 'teacher', checkpoint_sha256=teacher_sha)
        if mode == 'collect':
            records = collect_candidate_reference(environment, teacher, args.steps)
            payload = dict(version=CANDIDATE_VERSION, teacher_checkpoint_sha256=teacher_sha,
                teacher_checkpoint_path=str(args.teacher_checkpoint.resolve()), records=records,
                candidate_only=True, successful_demonstration_pool=False,
                qualification='requires_this_collection_own_complete_raw_success_proofs')
            with (run / 'stereo_teacher_candidates.pt').open('xb') as stream:
                torch.save(payload, stream)
            report = dict(status='teacher_candidate_reference_collected', steps=args.steps,
                samples=sum(len(r['identities']) for r in records), native_qualified=False,
                successful_demonstration_pool=False)
        else:
            from .tabletop_stereo_checkpoint import load_deployment_actor
            student = load_deployment_actor(args.actor_package,
                expected_calibration_sha256=environment.camera.calibration.sha256).to(bridge.device)
            report = run_online_dagger(environment, student, teacher, config=config,
                source_sha256=teacher_sha, qualification=args.teacher_proof, output=run,
                rounds=args.rounds, steps=args.steps, batch_size=args.batch_size, epochs=args.epochs, stage=args.stage,
                actor_package=args.actor_package,
                provenance={name: {'path': str(path.resolve()), 'sha256': file_sha256(path)}
                    for name, path in (('teacher_checkpoint_artifact', args.teacher_checkpoint),
                                       ('config_artifact', args.config))})
        write_json_new(run / 'completion.json', report)
        return report

    launch_native(callback, native_arguments=args.native_task_config, output=run / 'native',
        num_envs=args.num_envs, seed=args.seed, enabled=True)
    return 0
