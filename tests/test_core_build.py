import importlib
import json
import hashlib
from pathlib import Path
import pytest

def test_build_rejects_mutated_source_before_creating_outputs(tmp_path):
    mod = importlib.import_module('dh116.physics.solver')
    source = tmp_path / 'vendor/ropesim'
    source.mkdir(parents=True)
    (source / 'CMakeLists.txt').write_text('changed')
    config = tmp_path / 'configs/runtime'
    config.mkdir(parents=True)
    (config / 'ropesim_source_lock.json').write_text(json.dumps({'source_sha256': {'CMakeLists.txt': '0' * 64}}))
    with pytest.raises(ValueError, match='source SHA'):
        mod.verify_source(tmp_path)

def test_build_output_is_new_and_confined_to_project_build(tmp_path):
    mod = importlib.import_module('dh116.physics.solver')
    with pytest.raises(ValueError):
        mod.new_output(tmp_path, tmp_path / 'src/output')
    target = tmp_path / 'build/ropesim'
    assert mod.new_output(tmp_path, target) == target
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match='new'):
        mod.new_output(tmp_path, target)

def test_project_outputs_reject_escape_and_symlink(tmp_path):
    from dh116.env.contracts import ProjectPaths
    paths = ProjectPaths(tmp_path)
    with pytest.raises(ValueError):
        paths.output('../outside')
    outside = tmp_path.parent / 'outside'
    outside.mkdir(exist_ok=True)
    (tmp_path / 'runs').symlink_to(outside)
    with pytest.raises(ValueError):
        paths.output('runs/escape')
    with pytest.raises(ValueError):
        paths.file(outside / 'input')

def test_native_options_reject_invalid_readback_before_creating_app(monkeypatch):
    import dh116.rl.training as teacher_training
    monkeypatch.setenv('DH116_NATIVE_READBACK_BACKEND', 'cached')
    with pytest.raises(ValueError, match='official_cache'):
        teacher_training.validate_native_options()
    monkeypatch.setenv('DH116_NATIVE_READBACK_BACKEND', 'official_cache')
    assert teacher_training.validate_native_options() is None

def test_scene_build_mode_rejected_before_app_launch(monkeypatch):
    import dh116.rl.training as teacher_training
    monkeypatch.setenv('DH116_SCENE_BUILD_MODE', 'typo')
    with pytest.raises(ValueError, match='scene build'):
        teacher_training.validate_native_options()

def test_unqualified_replication_cannot_enter_training():
    import dh116.rl.training as teacher_training
    with pytest.raises(ValueError, match='diagnostic'):
        teacher_training.validate_native_options('replicated', training=True)
