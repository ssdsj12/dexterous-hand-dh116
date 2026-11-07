"""Warm-start changes Adam only in the new process; snapshots are immutable."""
import json

import pytest
import torch

from test_vision_squashed_training import setup, run
from test_vision_resume import assert_same


def source(tmp_path):
    parent = tmp_path/'source'; parent.mkdir()
    dw, task, env, args = setup(parent)
    run(dw, task, env, args)
    path = args.output/'vision_student_ppo.pt'
    saved = torch.load(path, weights_only=False)
    saved['experiment_projection'] = {'version': 'fixture_projection', 'source_sha256': 'frozen'}
    torch.save(saved, path)
    return path, saved


def resumed(tmp_path, path, mode, maximum=2):
    parent = tmp_path/mode; parent.mkdir()
    dw, task, env, args = setup(parent, maximum=maximum)
    args.resume = path
    args.vision_warm_start_optimizer = mode
    args.vision_checkpoint_interval = 10
    return dw, task, env, args


def test_preserve_and_reset_adam_are_explicit_different_warm_starts(tmp_path):
    path, saved = source(tmp_path); original = path.read_bytes()
    models = {}
    for mode, expected_step in [('preserve', 20), ('reset', 10)]:
        dw, task, env, args = resumed(tmp_path, path, mode)
        run(dw, task, env, args)
        result = torch.load(args.output/'vision_student_ppo.pt', weights_only=False)
        start = json.loads((args.output/'training_start.json').read_text())
        assert start['optimizer_warm_start_mode'] == mode
        assert start['initial_iteration'] == 1
        assert start['optimizer_state_entries'] == (0 if mode == 'reset' else 19)
        assert {int(v['step']) for v in result['optimizer']['state'].values()} == {expected_step}
        assert_same(result['experiment_projection'], saved['experiment_projection'])
        assert_same(start['experiment_projection'], saved['experiment_projection'])
        if mode == 'reset':
            reset = result['optimizer_reset_history']
            assert len(reset) == 1 and reset[0]['iteration'] == 1
            assert reset[0]['source_sha256'] == __import__('hashlib').sha256(original).hexdigest()
            assert start['resume_restored_state'] == ['policy', 'learning_rate', 'Torch_RNG', 'counters']
        else:
            assert 'optimizer_reset_history' not in result
        models[mode] = result
    assert path.read_bytes() == original
    assert any(not torch.equal(models['preserve']['policy'][k], models['reset']['policy'][k])
               for k in saved['policy'])


def test_reset_history_persists_and_final_bundle_is_independent(tmp_path):
    path, _ = source(tmp_path)
    dw, task, env, args = resumed(tmp_path, path, 'reset')
    run(dw, task, env, args)
    bundle = args.output/'checkpoints'/'iteration_00000002'
    manifest = json.loads((bundle/'manifest.json').read_text())
    retained = (bundle/'checkpoint.pt').read_bytes()
    path2 = args.output/'vision_student_ppo.pt'
    original = torch.load(path2, weights_only=False)
    dw, task, env, args2 = resumed(tmp_path, path2, 'preserve', maximum=3)
    run(dw, task, env, args2)
    result = torch.load(args2.output/'vision_student_ppo.pt', weights_only=False)
    assert_same(result['optimizer_reset_history'], original['optimizer_reset_history'])
    assert manifest['complete_ppo_update'] is True
    assert (bundle/'checkpoint.pt').read_bytes() == retained


@pytest.mark.parametrize('case', ['no_resume', 'no_update', 'legacy', 'bad_interval'])
def test_invalid_reset_or_retention_options_fail_before_metadata_and_sensors(tmp_path, case):
    path, _ = source(tmp_path)
    dw, task, env, args = resumed(tmp_path, path, 'reset')
    if case == 'no_resume': args.resume = None
    if case == 'no_update': args.max_iterations = 1
    if case == 'legacy': args.vision_action_distribution = 'gaussian_clip'
    if case == 'bad_interval': args.vision_checkpoint_interval = 0
    def forbidden(*a, **kw): raise AssertionError('must fail before sensors/reset')
    env.reset = forbidden
    from test_vision_training import Camera
    camera = Camera(2); camera.sample = forbidden
    with pytest.raises(ValueError): run(dw, task, env, args, camera=camera)
    assert not list(args.output.iterdir())


def test_interval_and_final_retention_do_not_replace_earlier_bundle(tmp_path):
    dw, task, env, args = setup(tmp_path, maximum=3)
    args.vision_checkpoint_interval = 2
    run(dw, task, env, args)
    bundles = sorted((args.output/'checkpoints').glob('iteration_*'))
    assert [p.name for p in bundles] == ['iteration_00000002', 'iteration_00000003']
    saved = [torch.load(p/'checkpoint.pt', weights_only=False) for p in bundles]
    assert [p['iteration'] for p in saved] == [2, 3]


def test_front_launcher_reset_budget_is_checked_before_creating_initialization(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
    path, _ = source(tmp_path)
    _, _, _, args = resumed(tmp_path, path, 'reset')
    del args.max_iterations
    args.updates = 1
    with pytest.raises(ValueError, match='at least one complete'):
        validate_vision_action_options(args)
    assert not list(args.output.iterdir())


def test_runtime_rechecks_reset_budget_of_the_actual_loaded_checkpoint(tmp_path, monkeypatch):
    import shutil
    from go2_pvcnn.control.dh116 import tabletop_vision_training as training
    path, _ = source(tmp_path)
    later = tmp_path/'later'; later.mkdir()
    dw2, task2, env2, args2 = setup(later, maximum=2)
    run(dw2, task2, env2, args2)
    dw, task, env, args = resumed(tmp_path, path, 'reset', maximum=2)
    original = training.validate_vision_action_options
    def change_source_after_preflight(value):
        original(value)
        shutil.copyfile(args2.output/'vision_student_ppo.pt', path)
    monkeypatch.setattr(training, 'validate_vision_action_options', change_source_after_preflight)
    def forbidden(): raise AssertionError('must reject before physical reset')
    env.reset = forbidden
    with pytest.raises(ValueError, match='at least one complete'):
        run(dw, task, env, args)
    assert not list(args.output.iterdir())
