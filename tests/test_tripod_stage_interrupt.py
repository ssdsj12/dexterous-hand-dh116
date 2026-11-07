"""CPU-only interrupted-run reporting; these fixtures are never native trials."""
import importlib
import json
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.parametrize('exception,decision', [
    (KeyboardInterrupt(), 'stage_A_interrupted'),
    (ValueError('fixture reset failure'), 'stage_A_exception'),
    (SystemExit(2), 'stage_A_exception'),
])
def test_interrupted_native_entry_preserves_failure_and_closes_evidence(
        monkeypatch, tmp_path, exception, decision):
    runner = importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_staged_experiment')
    tensor_world = importlib.import_module('go2_pvcnn.control.dh116.tabletop_tensor_world')
    tensor_env = importlib.import_module('go2_pvcnn.control.dh116.tabletop_tensor_env')
    vision = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_training')
    dataset = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_supervision_dataset')
    collection = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_teacher_collection')
    closed = []

    class FixtureTensorWorld:
        decimation = 8

    class FixtureTensorEnv:
        def reset(self):
            raise exception

    class FixtureRuntime:
        def __init__(self, *args):
            pass

        def close(self):
            closed.append('runtime')
            return {'fixture_only': True, 'counts_as_student_success': False}

    class FixtureRaw:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            closed.append('raw')

    # Exact-class checks are replaced only in this isolated CPU fixture. No
    # native controller/proof is created and no physical step is requested.
    monkeypatch.setattr(tensor_world, 'TensorWorld', FixtureTensorWorld)
    monkeypatch.setattr(tensor_env, 'TensorEnv', FixtureTensorEnv)
    monkeypatch.setattr(runner, 'TripodRecoveryRuntime', FixtureRuntime)
    monkeypatch.setattr(runner, '_source_models', lambda: [])
    monkeypatch.setattr(runner, '_verify_original_scene', lambda *args: {'fixture_only': True})
    monkeypatch.setattr(vision, '_contract', lambda *args: {'fixture_only': True})
    monkeypatch.setattr(dataset, 'load_teacher_reference_dataset',
                        lambda *args: {'source_sha256': 'a' * 64})
    monkeypatch.setattr(collection, '_choose_references', lambda *args: [])
    monkeypatch.setattr(collection, 'RawRGBDChunks', FixtureRaw)
    fixture_sim = type('FixtureSim', (), {'__module__': 'isaaclab.cpu_fixture'})()
    world = type('FixtureWorld', (), {'__module__': 'cpu_fixture.tabletop_gpu_isaac'})()
    world.sim = fixture_sim
    task = NS(command=torch.zeros(8, 12), surface_policy_owned_targets=True,
              surface_redistribution=False)
    args = NS(tripod_recovery_stage='A', vision_teacher_source='fixture.pt',
              vision_teacher_mpc=True, vision_teacher_mpc_profile='pressure_reference',
              vision_student=True, vision_action_distribution='tanh_gaussian',
              vision_camera_mode='fixed', num_envs=8, vision_teacher_max_control_steps=560,
              resume=None, vision_evaluation_checkpoint=None, surface_mpc=False,
              vision_lift_reward_experiment='R0', output=tmp_path,
              vision_teacher_raw_chunk_frames=16)
    with pytest.raises(type(exception)):
        runner.run_staged_tripod_experiment(world, FixtureTensorWorld(), task,
                                           FixtureTensorEnv(), args, {})
    report = json.loads((tmp_path / 'tripod_stage_report.json').read_text())
    assert closed == ['runtime', 'raw']
    assert report['decision'] == decision
    assert report['failure'] == repr(exception)
    assert report['completed_controls'] == 0 and report['first_terminal'] == {}
    assert report['student_successes'] == report['ppo_updates'] == 0
    assert report['result']['fixture_only'] is True
