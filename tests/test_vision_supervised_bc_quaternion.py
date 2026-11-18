"""Opt-in quaternion preprocessing is bound to new BC artifacts, never old weights."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from test_vision_supervised_bc import (
    assert_same, isolate_fixture_rng_and_threads, minimal_data, module, sensor_rows, source_v3,
)


MODE = 'wrist_quaternion_hemisphere_v1'
FIELD = 'actor_wrist_quaternion_preprocessing'
VERSION = 'dh116-vision-supervised-bc-v3-wrist-quaternion-hemisphere-v1'


def sign_pair():
    rows = sensor_rows()
    frames = rows.reshape(-1, 4, 961)
    for offset in (923, 930):
        frames[..., offset:offset+4] = torch.tensor([.5, -.5, .5, .5])
    opposite = rows.clone()
    other = opposite.reshape(-1, 4, 961)
    other[:, ::2, 923:927] *= -1
    other[:, 1::2, 930:934] *= -1
    return rows, opposite


def canonical_artifact(tmp_path):
    m = module()
    source, saved = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'
    torch.save(minimal_data(saved['contract']), dataset)
    artifact = tmp_path/'canonical.pt'
    report = m.fit_supervision(source, dataset, artifact, steps=2, batch_size=4,
        allow_fit_only=True, wrist_quaternion_preprocessing=MODE)
    return source, saved, dataset, artifact, report


def test_canonical_fit_and_visual_audit_are_invariant_to_quaternion_sign():
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    m = module(); torch.set_num_threads(2)
    initial = UnboundedVisionStudentActor().state_dict()
    positive, negative = sign_pair()
    snapshots = [copy.deepcopy(initial), positive.clone(), negative.clone()]
    labels = torch.ones(len(positive), 24)*.4
    before_rng = torch.get_rng_state().clone()
    options = dict(steps=3, batch_size=4, seed=17, wrist_quaternion_preprocessing=MODE)
    left = m.fit_actor(initial, positive, labels, **options)
    right = m.fit_actor(initial, negative, labels, **options)
    assert_same(left, right)
    assert left[FIELD] == MODE
    assert_same(before_rng, torch.get_rng_state())
    assert_same(initial, snapshots[0])
    assert_same(positive, snapshots[1]); assert_same(negative, snapshots[2])
    assert m.audit_actor_visual_branch(initial, positive, wrist_quaternion_preprocessing=MODE) == (
        m.audit_actor_visual_branch(initial, negative, wrist_quaternion_preprocessing=MODE))
    check = UnboundedVisionStudentActor(wrist_quaternion_preprocessing=MODE)
    check.load_state_dict(left['actor'])
    assert len(check.state_dict()) == len(initial) == 12
    assert torch.equal(check(positive), check(negative))


def test_canonical_bc_adam_continuation_remains_exact():
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    m = module(); torch.set_num_threads(2)
    initial = UnboundedVisionStudentActor().state_dict()
    positive, negative = sign_pair(); labels = torch.ones(len(positive), 24)*.4
    options = dict(batch_size=3, seed=17, wrist_quaternion_preprocessing=MODE)
    first = m.fit_actor(initial, positive, labels, steps=2, **options)
    rest = m.fit_actor(first['actor'], negative, labels, steps=3,
        optimizer_state=first['optimizer'], sampling_rng_state=first['sampling_rng'], **options)
    full = m.fit_actor(initial, positive, labels, steps=5, **options)
    for key in ('actor', 'optimizer', 'sampling_rng'): assert_same(rest[key], full[key])


def test_canonical_artifact_derivation_preserves_source_and_complete_ppo_state(tmp_path):
    m = module()
    source, before, dataset, artifact_path, report = canonical_artifact(tmp_path)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    artifact = torch.load(artifact_path, weights_only=False)
    assert artifact['version'] == report['version'] == VERSION
    assert artifact[FIELD] == report[FIELD] == MODE
    assert_same(artifact['source_checkpoint_state'], before)
    assert_same(artifact['visual_contract'], before['contract'])
    expected_contract = copy.deepcopy(before['contract']); expected_contract[FIELD] = MODE
    assert artifact['trained_actor_contract'] == expected_contract
    output = tmp_path/'canonical_warm.pt'
    m.derive_supervised_checkpoint(artifact_path, output)
    warm = torch.load(output, weights_only=False)
    assert warm['version'] == before['version']
    assert warm['contract'] == expected_contract
    for key, value in before.items():
        if key not in ('policy', 'contract'): assert_same(value, warm[key])
    for key, value in before['policy'].items():
        if not key.startswith('actor.'): assert_same(value, warm['policy'][key])
    for key, value in artifact['trained_actor'].items(): assert_same(value, warm['policy']['actor.'+key])
    assert warm['visual_supervision_derivation'][FIELD] == MODE
    assert warm['visual_supervision_derivation']['status'] == 'warm_start_not_exact_resume'
    assert source_sha == hashlib.sha256(source.read_bytes()).hexdigest()
    assert dataset_sha == hashlib.sha256(dataset.read_bytes()).hexdigest()


@pytest.mark.parametrize('mutation', ['artifact_mode', 'report_mode', 'contract_mode',
    'missing_contract', 'unrelated_contract', 'artifact_version', 'report_version'])
def test_derivation_rejects_forged_preprocessing_metadata(tmp_path, mutation):
    m = module(); _, _, _, artifact_path, _ = canonical_artifact(tmp_path)
    artifact = torch.load(artifact_path, weights_only=False)
    if mutation == 'artifact_mode': artifact[FIELD] = 'unknown'
    elif mutation == 'report_mode': artifact['report'][FIELD] = 'raw'
    elif mutation == 'contract_mode': artifact['trained_actor_contract'][FIELD] = 'raw'
    elif mutation == 'missing_contract': artifact.pop('trained_actor_contract')
    elif mutation == 'unrelated_contract': artifact['trained_actor_contract']['num_envs'] += 2
    elif mutation == 'artifact_version': artifact['version'] = m.VERSION
    else: artifact['report']['version'] = m.VERSION
    torch.save(artifact, artifact_path)
    output = tmp_path/'forged.pt'
    with pytest.raises(ValueError): m.derive_supervised_checkpoint(artifact_path, output)
    assert not output.exists() and not output.with_suffix('.json').exists()


@pytest.mark.parametrize('mode', ['canonical', '', None, True])
def test_fit_rejects_unknown_preprocessing_before_writing_outputs(tmp_path, mode):
    m = module()
    with pytest.raises(ValueError, match='preprocessing'):
        m.fit_actor({}, sensor_rows(), torch.zeros(8,24), steps=1,
            wrist_quaternion_preprocessing=mode)
    with pytest.raises(ValueError, match='preprocessing'):
        m.fit_supervision(tmp_path/'missing.pt', tmp_path/'missing_data.pt', tmp_path/'out.pt',
            wrist_quaternion_preprocessing=mode)
    assert not (tmp_path/'out.pt').exists()


def test_historical_raw_artifact_without_adapter_metadata_still_derives_unchanged(tmp_path):
    m = module(); source, before = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(before['contract']), dataset)
    artifact_path = tmp_path/'historical.pt'
    m.fit_supervision(source, dataset, artifact_path, steps=1, allow_fit_only=True)
    artifact = torch.load(artifact_path, weights_only=False)
    artifact.pop(FIELD); artifact.pop('trained_actor_contract'); artifact['report'].pop(FIELD)
    artifact.pop('startup_sampling'); artifact.pop('train_sampling_probabilities')
    for key in ('startup_sampling', 'sampling_algorithm', 'sampling_distribution_sha256'):
        artifact['report'].pop(key)
    torch.save(artifact, artifact_path)
    output = tmp_path/'raw_warm.pt'; m.derive_supervised_checkpoint(artifact_path, output)
    warm = torch.load(output, weights_only=False)
    assert warm['contract'] == before['contract'] and FIELD not in warm['contract']
    assert warm['visual_supervision_derivation']['version'] == m.DERIVATION_VERSION


def test_default_raw_fit_never_silently_removes_source_adapter(tmp_path):
    m = module(); source, before = source_v3(tmp_path)
    before['contract'][FIELD] = MODE
    torch.save(before, source)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(before['contract']), dataset)
    with pytest.raises(ValueError, match='remove.*preprocessing'):
        m.fit_supervision(source, dataset, tmp_path/'bad.pt', steps=1, allow_fit_only=True)
    assert not (tmp_path/'bad.pt').exists()


def test_cli_forwards_only_explicit_preprocessing_option(monkeypatch, capsys):
    m = module(); calls = []
    monkeypatch.setattr(m, 'fit_supervision', lambda *a, **k: calls.append(k) or {})
    path = Path(__file__).parents[1]/'scripts/fit_dh116_vision_supervision.py'
    spec = importlib.util.spec_from_file_location('fit_quaternion_cli', path)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    base = ['--source-checkpoint','source.pt','--dataset','data.pt','--output','out.pt']
    cli.main(base)
    cli.main(base+['--wrist-quaternion-preprocessing', MODE])
    assert [row['wrist_quaternion_preprocessing'] for row in calls] == ['raw', MODE]


def test_default_primitive_retains_randint_and_weighted_rng_continues_exactly(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    m = module(); torch.set_num_threads(2)
    initial = UnboundedVisionStudentActor().state_dict()
    obs = sensor_rows(); labels = torch.ones(8,24)*.4
    weights = torch.tensor([8.,1.,1.,1.,1.,1.,1.,1.], dtype=torch.float64)
    original_weights = weights.clone(); rng = torch.get_rng_state().clone()
    options = dict(seed=17, batch_size=4, wrist_quaternion_preprocessing=MODE)
    full = m.fit_actor(initial, obs, labels, steps=5, sampling_weights=weights, **options)
    first = m.fit_actor(initial, obs, labels, steps=2, sampling_weights=weights, **options)
    rest = m.fit_actor(first['actor'], obs, labels, steps=3, sampling_weights=weights,
        optimizer_state=first['optimizer'], sampling_rng_state=first['sampling_rng'], **options)
    for key in ('actor', 'optimizer', 'sampling_rng'): assert_same(full[key], rest[key])
    assert_same(weights, original_weights); assert_same(rng, torch.get_rng_state())
    monkeypatch.setattr(m.torch, 'multinomial', lambda *a, **k: pytest.fail('default must use randint'))
    default = m.fit_actor(initial, obs, labels, steps=2, **options)
    explicit = m.fit_actor(initial, obs, labels, steps=2, sampling_weights=None, **options)
    assert_same(default, explicit)


@pytest.mark.parametrize('bad', [torch.ones(7), torch.ones(8,1), torch.ones(8,dtype=torch.long),
    torch.tensor([0.]+[1.]*7), torch.tensor([-1.]+[1.]*7),
    torch.tensor([float('nan')]+[1.]*7), torch.tensor([float('inf')]+[1.]*7)])
def test_invalid_sampling_weights_reject_before_actor_construction(bad):
    with pytest.raises(ValueError, match='sampling'):
        module().fit_actor({}, sensor_rows(), torch.zeros(8,24), steps=1, sampling_weights=bad)


@pytest.mark.parametrize('options', [dict(startup_sampling_fraction=-.1),
    dict(startup_sampling_fraction=.51), dict(startup_sampling_fraction=float('nan')),
    dict(startup_sampling_fraction=True), dict(startup_control_steps=0),
    dict(startup_control_steps=True), dict(startup_control_steps=1.2)])
def test_invalid_startup_configuration_rejects_before_file_io(tmp_path, options):
    with pytest.raises(ValueError, match='startup'):
        module().fit_supervision(tmp_path/'missing.pt', tmp_path/'missing_data.pt', tmp_path/'out.pt',
            wrist_quaternion_preprocessing=MODE, **options)


def test_startup_sampling_requires_explicit_canonical_mode(tmp_path):
    with pytest.raises(ValueError, match='canonical|quaternion'):
        module().fit_supervision(tmp_path/'missing.pt', tmp_path/'missing_data.pt', tmp_path/'out.pt',
            startup_sampling_fraction=.25)


def startup_artifact(tmp_path, **kwargs):
    m = module(); source, saved = source_v3(tmp_path)
    dataset = tmp_path/'data.pt'; torch.save(minimal_data(saved['contract']), dataset)
    artifact = tmp_path/'startup.pt'
    report = m.fit_supervision(source, dataset, artifact, steps=2, batch_size=4,
        heldout_trajectories=['b'*64], wrist_quaternion_preprocessing=MODE,
        startup_sampling_fraction=.25, startup_control_steps=1, **kwargs)
    return artifact, report


def test_startup_distribution_uses_only_training_subset_and_binds_identity(tmp_path):
    m = module(); artifact_path, report = startup_artifact(tmp_path)
    artifact = torch.load(artifact_path, weights_only=False)
    expected = torch.tensor([.4375,.1875,.1875,.1875], dtype=torch.float64)
    assert_same(artifact['train_sampling_probabilities'], expected)
    identity = report['startup_sampling']
    assert identity['fraction'] == .25 and identity['control_steps'] == 1
    assert identity['train_sample_count'] == 4 and identity['startup_sample_count'] == 1
    assert identity['distribution_sha256'] == hashlib.sha256(expected.numpy().tobytes()).hexdigest()
    assert identity['algorithm'] == 'torch_multinomial_replacement_private_cpu_generator_v1'
    assert report['heldout_source_trajectory_ids'] == ['b'*64]
    assert artifact['startup_sampling'] == identity
    m.derive_supervised_checkpoint(artifact_path, tmp_path/'warm.pt')


@pytest.mark.parametrize('field', ['fraction', 'distribution_sha256', 'startup_sample_count',
    'probabilities', 'row_order_with_matching_hashes'])
def test_derivation_rejects_forged_startup_distribution(tmp_path, field):
    m = module(); artifact_path, _ = startup_artifact(tmp_path)
    artifact = torch.load(artifact_path, weights_only=False)
    if field == 'row_order_with_matching_hashes':
        artifact['train_sampling_probabilities'] = artifact['train_sampling_probabilities'].roll(1)
        digest = hashlib.sha256(artifact['train_sampling_probabilities'].numpy().tobytes()).hexdigest()
        artifact['report']['sampling_distribution_sha256'] = digest
        artifact['report']['startup_sampling']['distribution_sha256'] = digest
        artifact['startup_sampling']['distribution_sha256'] = digest
    elif field == 'probabilities': artifact['train_sampling_probabilities'][0] += .1
    elif field == 'fraction': artifact['report']['startup_sampling'][field] = 0.
    elif field == 'distribution_sha256': artifact['report']['startup_sampling'][field] = '0'*64
    else: artifact['report']['startup_sampling'][field] += 1
    torch.save(artifact, artifact_path)
    with pytest.raises(ValueError, match='sampling'):
        m.derive_supervised_checkpoint(artifact_path, tmp_path/'bad.pt')
    assert not (tmp_path/'bad.pt').exists()


def test_weighted_fixture_reduces_error_on_rare_startup_labels():
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    m = module(); torch.set_num_threads(2)
    initial = UnboundedVisionStudentActor().state_dict()
    obs = sensor_rows(1).repeat(40,1); labels = torch.zeros(40,24); labels[:2] = .8
    weights = torch.full((40,), .5/40, dtype=torch.float64); weights[:2] += .5/2
    options = dict(steps=35, batch_size=16, learning_rate=.003, seed=42,
        wrist_quaternion_preprocessing=MODE, heldout_observations=obs[:2], heldout_actions=labels[:2])
    uniform = m.fit_actor(initial, obs, labels, **options)
    weighted = m.fit_actor(initial, obs, labels, sampling_weights=weights, **options)
    assert weighted['heldout_metrics_after']['mse'] < uniform['heldout_metrics_after']['mse']*.6


def test_cli_forwards_explicit_startup_configuration(monkeypatch, capsys):
    m = module(); calls = []
    monkeypatch.setattr(m, 'fit_supervision', lambda *a, **k: calls.append(k) or {})
    path = Path(__file__).parents[1]/'scripts/fit_dh116_vision_supervision.py'
    spec = importlib.util.spec_from_file_location('fit_startup_cli', path)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    cli.main(['--source-checkpoint','s.pt','--dataset','d.pt','--output','o.pt',
        '--wrist-quaternion-preprocessing',MODE,'--startup-sampling-fraction','.25',
        '--startup-control-steps','3'])
    assert calls[0]['startup_sampling_fraction'] == .25 and calls[0]['startup_control_steps'] == 3
