import faulthandler
import hashlib
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import calibrate_dh116_stereo_contacts as calibration
from scripts.verify_dh116_stereo_shutdown import verify_trial


def write_json(root, name, value):
    (root / name).write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def evidence(tmp_path):
    rows = [dict(physics_step=i, sdk_step=i + 2, timestamp=(i + 2) / 480,
                 episode_uid='unique-calibration-episode', env_id=0, dt=1 / 480)
            for i in range(1, 241)]
    raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    (tmp_path / 'raw_contact_480hz.jsonl').write_bytes(raw)
    write_json(tmp_path, 'summary.json', dict(sampling_complete=True, raw_fsync_complete=True,
               raw_sha256=hashlib.sha256(raw).hexdigest()))
    write_json(tmp_path, 'callbacks_released.json', dict(full=True, ordinary=True))
    write_json(tmp_path, 'shutdown_started.json', dict(time_unix_s=10))
    write_json(tmp_path, 'simulation_context_cleared.json', dict(
        public_api='SimulationContext.clear_instance', stop_callback_released=True,
        physics_sampling_already_ended=True))
    write_json(tmp_path, 'shutdown_complete.json', dict(close_returned=True, time_unix_s=11,
               pending_simulation_callback_error=None))
    write_json(tmp_path, 'launcher_exit.json', dict(pid=123, exit_code=0, timed_out=False,
               signals_sent=[], started_unix_s=9, exited_unix_s=12))
    write_json(tmp_path, 'sigterm_monitor.json', dict(pid=123, installed_after_app_launch=True))
    write_json(tmp_path, 'shutdown_mode.json', dict(
        pid=123, setting='/app/fastShutdown', effective_fast_shutdown=False))
    return tmp_path


def test_accepts_continuous_durable_sampling_and_child_exit(evidence):
    assert verify_trial(evidence)['physics_steps'] == 240


@pytest.mark.parametrize('name', ['summary.json', 'callbacks_released.json',
                                  'simulation_context_cleared.json', 'shutdown_complete.json',
                                  'launcher_exit.json', 'sigterm_monitor.json', 'shutdown_mode.json'])
def test_missing_evidence_cannot_pass(evidence, name):
    (evidence / name).unlink()
    with pytest.raises(ValueError, match=name):
        verify_trial(evidence)


def test_rejects_historical_sigterm_even_when_exit_is_zero(evidence):
    write_json(evidence, 'shutdown_timeout.json', dict(signal='SIGTERM'))
    with pytest.raises(ValueError, match='shutdown_timeout'):
        verify_trial(evidence)


def test_child_observed_signal_cannot_pass_with_exit_zero(evidence):
    write_json(evidence, 'external_signal.json', dict(signal='SIGTERM', pid=123))
    with pytest.raises(ValueError, match='external_signal'):
        verify_trial(evidence)


def test_late_exit_cannot_override_launcher_timeout(evidence):
    timeout = evidence.with_suffix('.timeout.json')
    try:
        timeout.write_text('{"timed_out":true}', encoding='utf-8')
        with pytest.raises(ValueError, match='timeout'):
            verify_trial(evidence)
    finally:
        timeout.unlink()


@pytest.mark.parametrize('edit', ['truncate', 'missing_newline', 'clock_jump', 'new_episode'])
def test_rejects_incomplete_or_discontinuous_raw_rows(evidence, edit):
    path = evidence / 'raw_contact_480hz.jsonl'
    lines = path.read_bytes().splitlines(keepends=True)
    if edit == 'truncate':
        lines.pop()
    elif edit == 'missing_newline':
        lines[-1] = lines[-1].rstrip()
    else:
        row = json.loads(lines[100])
        if edit == 'clock_jump':
            row['timestamp'] = 0.0
        else:
            row['episode_uid'] = 'other'
        lines[100] = (json.dumps(row) + '\n').encode()
    raw = b''.join(lines)
    path.write_bytes(raw)
    write_json(evidence, 'summary.json', dict(sampling_complete=True, raw_fsync_complete=True,
               raw_sha256=hashlib.sha256(raw).hexdigest()))
    with pytest.raises(ValueError, match='raw_contact'):
        verify_trial(evidence)


def test_native_launch_requires_explicit_approval_flag(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_dh116_stereo_shutdown.py'
    output = tmp_path / 'must-not-be-created'
    result = subprocess.run([sys.executable, str(path), '--output', str(output)],
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['mode'] == 'preflight'
    assert not output.exists()


def test_calibration_uses_non_fast_launcher_and_enables_fatal_diagnostics(monkeypatch):
    app = object()
    events = []

    def launcher(args, **config):
        assert args.headless is True and args.device == 'cuda:0'
        assert config.get('fast_shutdown') is False
        events.append('launch')
        return SimpleNamespace(app=app)

    def enable(*, file, all_threads):
        assert file is sys.__stderr__ and all_threads is True
        events.append('fatal_diagnostics')

    monkeypatch.setattr(faulthandler, 'enable', enable)
    assert calibration._launch_calibration_app(launcher) is app
    assert events == ['launch', 'fatal_diagnostics']


@pytest.mark.parametrize('effective', [False, True, None, 0])
def test_calibration_records_only_literal_false_from_kit_settings(tmp_path, effective):
    def read_setting(name):
        assert name == '/app/fastShutdown'
        return effective

    settings = SimpleNamespace(get=read_setting)
    marker = tmp_path / 'shutdown_mode.json'
    if effective is False:
        calibration._record_shutdown_mode(tmp_path, settings)
        assert json.loads(marker.read_text()) == dict(
            pid=os.getpid(), setting='/app/fastShutdown', effective_fast_shutdown=False)
    else:
        with pytest.raises(RuntimeError, match='/app/fastShutdown'):
            calibration._record_shutdown_mode(tmp_path, settings)
        assert not marker.exists()


@pytest.mark.parametrize('changed', [
    dict(pid=321),
    dict(pid='123'),
    dict(setting='/app/otherSetting'),
    dict(effective_fast_shutdown=True),
    dict(effective_fast_shutdown=0),
])
def test_shutdown_mode_must_be_read_back_false_by_the_same_child(evidence, changed):
    marker = dict(pid=123, setting='/app/fastShutdown', effective_fast_shutdown=False)
    marker.update(changed)
    write_json(evidence, 'shutdown_mode.json', marker)
    with pytest.raises(ValueError, match='shutdown_mode.json'):
        verify_trial(evidence)


@pytest.mark.parametrize('view_mode', ['valid', 'invalid', 'failure', 'absent'])
def test_owned_sampling_handles_release_before_app_close(tmp_path, view_mode):
    handles = calibration._SamplingHandles()
    events = []
    refs = {}
    app, sim = object(), object()

    class Handle:
        pass

    def give(owner, name, handle):
        setattr(owner, name, handle)
        refs[name] = weakref.ref(handle)

    class View:
        is_valid = view_mode != 'invalid'

        def invalidate(self):
            events.append('invalidate')
            assert all(refs[name]() is None for name in (
                'full_subscription', 'ordinary_subscription', 'contact', 'reverse', 'rigid_view'))
            if view_mode == 'failure':
                raise RuntimeError('view invalidation failed')

    for name in ('full_subscription', 'ordinary_subscription', 'contact', 'reverse', 'rigid_view', 'source', 'settings'):
        give(handles, name, Handle())
    give(handles.scene, 'stage', Handle())
    if view_mode != 'absent':
        give(handles, 'view', View())

    def close(actual_app, output, *, simulation):
        assert actual_app is app and output == tmp_path and simulation is sim
        assert all(ref() is None for ref in refs.values())
        assert json.loads((tmp_path / 'callbacks_released.json').read_text()) == dict(full=True, ordinary=True)
        events.append('close')

    if view_mode == 'failure':
        with pytest.raises(RuntimeError, match='view invalidation failed'):
            calibration._close_calibration_trial(handles, app, tmp_path, sim, close)
    else:
        calibration._close_calibration_trial(handles, app, tmp_path, sim, close)
    assert events == (['invalidate'] if view_mode in ('valid', 'failure') else []) + ['close']
