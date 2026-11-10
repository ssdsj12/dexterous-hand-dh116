"""Real runpy/exit control flow with CPU-only launcher and bridge substitutes."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import types

import pytest

from go2_pvcnn.control.dh116 import tabletop_stereo_native as native


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    root = tmp_path / 'cpu_launcher'
    (root / 'scripts').mkdir(parents=True)
    elsewhere = tmp_path / 'changed_by_launcher'
    elsewhere.mkdir()
    launcher = root / 'scripts/train_dh116_tabletop_gpu_ppo.py'
    config = tmp_path / 'native_arguments.json'
    config.write_text(json.dumps(dict(version='dh116-stereo-native-bootstrap-v1',
        arguments=['--cpu-schema-fixture'], grasp_styles=['pinch_thumb_index'])), encoding='utf-8')
    module_name = 'go2_pvcnn.control.dh116.tabletop_vision_training'
    module = types.ModuleType(module_name)
    original = lambda *args, **kwargs: pytest.fail('original training callback must stay isolated')
    module.run_vision_training = original
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(native, 'ROOT', root)
    events = []

    @contextmanager
    def cpu_lock():
        events.append('lock_enter')
        try:
            yield
        finally:
            events.append('lock_exit')

    class CPUBridge:
        def __init__(self, *args, output):
            events.append('bridge_create')
            self.output = Path(output)
            self.close_error = None

        def close(self):
            events.append('bridge_close')
            if self.close_error is not None:
                raise self.close_error

    monkeypatch.setattr(native, 'native_instance_lock', cpu_lock)
    monkeypatch.setattr(native, 'NativeTensorBridge', CPUBridge)
    old_cwd, old_argv = Path.cwd(), sys.argv[:]

    def write(body, *, query=True):
        prefix = ('from types import SimpleNamespace\n'
                  'from go2_pvcnn.control.dh116.tabletop_vision_training import run_vision_training\n')
        if query:
            prefix += 'run_vision_training(None, None, None, None, SimpleNamespace(), {})\n'
        prefix += f'import os, sys\nos.chdir({str(elsewhere)!r})\nsys.argv.append("changed-by-launcher")\n'
        launcher.write_text(prefix + body + '\n', encoding='utf-8')

    def launch(callback):
        return native.launch_native(callback, native_arguments=config, output=tmp_path / 'output', enabled=True)

    def assert_restored(*, queried=True):
        assert module.run_vision_training is original
        assert Path.cwd() == old_cwd
        assert sys.argv == old_argv
        assert events.count('lock_enter') == events.count('lock_exit') == 1
        assert events.count('bridge_create') == events.count('bridge_close') == int(queried)

    return types.SimpleNamespace(write=write, launch=launch, assert_restored=assert_restored,
        events=events, output=tmp_path / 'output')


@pytest.mark.parametrize('exit_code', [None, 0, 'return_normally'])
@pytest.mark.parametrize('empty_result', [False, True])
def test_real_runpy_success_returns_completed_callback_result_and_restores(bootstrap, exit_code, empty_result):
    body = '' if exit_code == 'return_normally' else f'raise SystemExit({exit_code!r})'
    bootstrap.write(body)
    expected = {} if empty_result else {'source_scope': 'cpu_fixture', 'schema_only': True}
    calls = []

    def callback(bridge):
        calls.append(bridge.output)
        return expected

    result = bootstrap.launch(callback)
    assert result == expected and calls == [bootstrap.output]
    # This caller-owned completion is only reachable after successful launch.
    completion = bootstrap.output / 'completion.json'
    completion.write_text(json.dumps(result), encoding='utf-8')
    assert json.loads(completion.read_text(encoding='utf-8')) == expected
    bootstrap.assert_restored()


@pytest.mark.parametrize('exit_code', [1, -2, '0', 0.0, 'native shutdown failed'])
def test_real_runpy_failed_exit_propagates_without_caller_completion(bootstrap, exit_code):
    bootstrap.write(f'raise SystemExit({exit_code!r})')
    with pytest.raises(SystemExit) as caught:
        result = bootstrap.launch(lambda bridge: {'schema_only': True})
        (bootstrap.output / 'completion.json').write_text(json.dumps(result), encoding='utf-8')
    assert type(caught.value.code) is type(exit_code) and caught.value.code == exit_code
    assert not (bootstrap.output / 'completion.json').exists()
    bootstrap.assert_restored()


@pytest.mark.parametrize('exit_code', [None, 0, 'return_normally'])
def test_success_exit_without_callback_is_rejected(bootstrap, exit_code):
    bootstrap.write('' if exit_code == 'return_normally' else f'raise SystemExit({exit_code!r})', query=False)
    with pytest.raises(RuntimeError, match='callback.*not.*reached|callback.*not.*completed'):
        bootstrap.launch(lambda bridge: pytest.fail('stub launcher did not query callback'))
    bootstrap.assert_restored(queried=False)


@pytest.mark.parametrize('error_name', ['RuntimeError', 'KeyboardInterrupt'])
def test_real_runpy_exception_is_not_converted_to_success(bootstrap, error_name):
    bootstrap.write(f'raise {error_name}("launcher failure")')
    exception_type = RuntimeError if error_name == 'RuntimeError' else KeyboardInterrupt
    with pytest.raises(exception_type, match='launcher failure'):
        bootstrap.launch(lambda bridge: {'schema_only': True})
    bootstrap.assert_restored()


def test_callback_failure_still_closes_bridge_and_restores_process(bootstrap):
    bootstrap.write('raise SystemExit(0)')

    def callback(bridge):
        raise ValueError('callback failed before completion')

    with pytest.raises(ValueError, match='callback failed'):
        bootstrap.launch(callback)
    bootstrap.assert_restored()


def test_bridge_cleanup_failure_is_not_success(bootstrap):
    bootstrap.write('raise SystemExit(0)')

    def callback(bridge):
        bridge.close_error = RuntimeError('bridge cleanup failed')
        return {'schema_only': True}

    with pytest.raises(RuntimeError, match='bridge cleanup failed'):
        bootstrap.launch(callback)
    bootstrap.assert_restored()
