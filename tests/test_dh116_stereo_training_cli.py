"""Executed offline BC/CLI fixtures; not successful native demonstrations."""
from pathlib import Path
import json

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_stereo_training_cli as cli
from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import build_dataset
from test_dh116_stereo_collection import FakeEnv, TinyStudent, TinyTeacher, module


ROOT = Path(__file__).resolve().parents[1]


def dataset():
    env = FakeEnv(); student, teacher = TinyStudent(), TinyTeacher()
    records = [module('teacher_collector').collect_step(env, student, teacher, schema_only=True) for _ in range(2)]
    return build_dataset(records, source_sha256='d'*64, schema_only=True)


def test_bc_entry_trains_real_visual_actor_with_heldout_source_and_exports(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import load_deployment_actor
    data = dataset()
    net = dict(embed_dim=32, history=4, num_heads=4, num_layers=1, dropout=0.,
        image_resolution_wh=[32, 32], image_processing_batch_size=2)
    report = cli.train_bc(data, net, {}, output=tmp_path, steps=1, batch_size=1,
        heldout_initial_sha256=['b'*64], schema_only=True)
    assert report['steps'] == 1 and report['heldout_samples'] == 2
    assert report['schema_only'] and not report['native_task_qualified']
    assert report['selection'] == 'lowest_heldout_action_loss'
    assert report['provenance']['dataset_content_sha256']
    assert report['provenance']['source_sha256']['tabletop_stereo_training_cli.py']
    actor = load_deployment_actor(tmp_path / 'stereo_bc_actor.pt', expected_calibration_sha256='c'*64)
    assert actor.deterministic(__import__('go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset',
        fromlist=['batch_dataset']).batch_dataset(data, [0])['history']).shape == (1, 24)
    with pytest.raises(FileExistsError):
        cli.train_bc(data, net, {}, output=tmp_path, steps=1, batch_size=1,
            heldout_initial_sha256=['b'*64], schema_only=True)


def test_formal_bc_refuses_fixture_before_model_or_output(tmp_path):
    with pytest.raises(ValueError, match='schema-only'):
        cli.train_bc(dataset(), {}, {}, output=tmp_path, steps=1)
    assert not list(tmp_path.iterdir())


def test_three_cli_config_modes_do_not_start_native(tmp_path, capsys):
    for mode, config in (('bc','stereo_bc_v1.json'), ('dagger','stereo_dagger_v1.json'),
                         ('collect','privileged_teacher_ppo_v1.json')):
        assert cli.main(mode, ['--config', str(ROOT/'configs/dh116_training'/config), '--check-config']) == 0
        assert not json.loads(capsys.readouterr().out)['native_started']
    assert not list(tmp_path.iterdir())


def test_two_real_visual_dagger_rounds_export_auditable_actor_lineage(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor, normalize_network_config
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_actor_package, load_deployment_actor
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract, file_sha256
    network = normalize_network_config(dict(embed_dim=32, history=4, num_heads=4,
        num_layers=1, image_resolution_wh=[32,32], image_processing_batch_size=2))
    actor = make_stereo_actor(network)
    initial = tmp_path/'initial.pt'
    save_actor_package(initial, actor, make_contract('bc', dict(network=network, schema_only=True),
        calibration_sha256='c'*64), network_config=network)
    student = load_deployment_actor(initial)
    config = json.loads((ROOT/'configs/dh116_training/stereo_dagger_v1.json').read_text())
    result = cli.run_online_dagger(FakeEnv(), student, TinyTeacher(), config=config,
        source_sha256='d'*64, qualification=None, output=tmp_path, rounds=2,
        steps=1, batch_size=2, epochs=1, stage='D4', schema_only=True, actor_package=initial)
    assert [row['student_owned_samples'] for row in result['rounds']] == [2,2]
    latest = load_deployment_actor(tmp_path/'dagger_actor_0002.pt')
    provenance = latest.deployment_contract['training_provenance']
    assert provenance['teacher_checkpoint_sha256'] == 'd'*64
    assert provenance['input_actor_package_artifact']['sha256'] == file_sha256(tmp_path/'dagger_actor_0001.pt')
    assert provenance['dataset_artifact']['sha256'] == file_sha256(tmp_path/'dagger_pool_0002.pt')
    assert not result['rounds'][-1]['native_task_qualified']


def test_teacher_candidates_cannot_be_admitted_from_summary_flag_only(tmp_path):
    path = tmp_path/'candidate.pt'
    torch.save(dict(version=cli.CANDIDATE_VERSION, teacher_checkpoint_sha256='d'*64, records=[]), path)
    with pytest.raises(ValueError):
        cli.admit_candidates(path, {'qualified': True}, tmp_path/'forbidden.pt')
    assert not (tmp_path/'forbidden.pt').exists()


def test_bc_raw_processing_requires_identical_recorded_pipeline():
    from copy import deepcopy
    from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract
    config = json.loads((ROOT/'configs/dh116_training/stereo_camera_v1.json').read_text())
    processing = processing_contract(config)
    data = dict(rows=[{'identity': {'observation_preprocessor': processing}}]*2,
        contract={'calibration_sha256': processing['calibration_sha256']})
    assert cli._dataset_processing(data, schema_only=False) == processing
    bad = deepcopy(data); bad['rows'][0] = {'identity': {}}
    with pytest.raises(ValueError, match='every sample'):
        cli._dataset_processing(bad, schema_only=False)
    config['depth_estimator'] = dict(subpixel=True)
    bad['rows'][0] = {'identity': {'observation_preprocessor': processing_contract(config)}}
    with pytest.raises(ValueError, match='differs'):
        cli._dataset_processing(bad, schema_only=False)


def test_formal_collection_binds_proof_to_the_selected_resolution(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as gate
    from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
    checkpoint = tmp_path/'teacher.pt'; checkpoint.write_bytes(b'CPU metadata fixture only')
    monkeypatch.setattr(gate, 'require_teacher_qualification', lambda *a,**k: {'mock_only':True})
    monkeypatch.setattr(gate, 'verify_stereo_qualification', lambda *a,**k: {'calibration_sha256':'0'*64})
    config = ROOT/'configs/dh116_training/stereo_camera_v1.json'
    with pytest.raises(ValueError,match='calibration differs'):
        cli._formal_collection_admission({},checkpoint,{},config)
    sha = StereoCalibration.from_json(config).sha256
    from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth
    matcher = StereoDepth.from_config(json.loads(config.read_text())).to_config()
    monkeypatch.setattr(gate, 'verify_stereo_qualification', lambda *a,**k:
        {'calibration_sha256':sha, 'depth_estimator_config':matcher})
    result = cli._formal_collection_admission({},checkpoint,{},config)
    assert result['stereo']['calibration_sha256'] == sha  # boolean binding only, not native qualification
    matcher['subpixel'] = not matcher['subpixel']
    with pytest.raises(ValueError, match='matcher differs'):
        cli._formal_collection_admission({},checkpoint,{},config)
