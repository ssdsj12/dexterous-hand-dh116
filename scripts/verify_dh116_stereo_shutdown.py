"""Preflight or run two sequential 240-step Isaac calibration shutdown trials."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'scripts' / 'calibrate_dh116_stereo_contacts.py'
THUMB_MESH = (ROOT / 'resources' / 'manufacturer' / 'DH116V2.0_URDF_MJCF_Files'
              / 'DH116-V2.0-L' / 'meshes' / 'finger14_link.STL')
STEPS = 240
DT = 1 / 480


def read_record(root, name):
    try:
        value = json.loads((root / name).read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('expected object')
        return value
    except (OSError, ValueError) as exc:
        raise ValueError(f'{name}: {exc}') from exc


def write_record(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_trial(root):
    """Verify recorded trial evidence. This function can also check CPU fixtures."""
    root = Path(root)
    if root.with_suffix('.timeout.json').exists():
        raise ValueError('launcher timeout: trial did not exit within the deadline')
    for failure in ('shutdown_timeout.json', 'external_signal.json', 'failed.json'):
        if (root / failure).exists():
            raise ValueError(f'{failure}: failed or externally interrupted trial')

    summary = read_record(root, 'summary.json')
    if summary.get('sampling_complete') is not True or summary.get('raw_fsync_complete') is not True:
        raise ValueError('summary.json: durable sampling completion is missing')
    raw_path = root / 'raw_contact_480hz.jsonl'
    try:
        raw = raw_path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        lines = raw.splitlines()
        if len(lines) != STEPS or not raw.endswith(b'\n') or sha != summary.get('raw_sha256'):
            raise ValueError('expected 240 complete lines and matching raw SHA256')
        sdk_start = start_time = identity = None
        for step, line in enumerate(lines, start=1):
            row = json.loads(line)
            sdk_step = row['sdk_step']
            timestamp = row['timestamp']
            episode_uid = row['episode_uid']
            if step == 1:
                sdk_start, start_time, identity = sdk_step, timestamp, episode_uid
            if (row['physics_step'] != step or type(sdk_step) is not int
                    or sdk_step != sdk_start + step - 1 or not math.isfinite(timestamp)
                    or abs(timestamp - start_time - (step - 1) * DT) > 1e-6
                    or not isinstance(episode_uid, str) or not episode_uid or episode_uid != identity
                    or row['env_id'] != 0 or not math.isclose(row['dt'], DT, abs_tol=1e-9)):
                raise ValueError(f'invalid step, clock or episode identity at line {step}')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'{raw_path.name}: {exc}') from exc

    released = read_record(root, 'callbacks_released.json')
    if released.get('full') is not True or released.get('ordinary') is not True:
        raise ValueError('callbacks_released.json: both owned subscription handles must be released')
    cleared = read_record(root, 'simulation_context_cleared.json')
    if (cleared.get('public_api') != 'SimulationContext.clear_instance'
            or cleared.get('stop_callback_released') is not True
            or cleared.get('physics_sampling_already_ended') is not True):
        raise ValueError('simulation_context_cleared.json: owned context cleanup missing')
    started = read_record(root, 'shutdown_started.json')
    closed = read_record(root, 'shutdown_complete.json')
    child = read_record(root, 'launcher_exit.json')
    monitor = read_record(root, 'sigterm_monitor.json')
    if monitor.get('installed_after_app_launch') is not True or monitor.get('pid') != child.get('pid'):
        raise ValueError('sigterm_monitor.json: child signal monitoring evidence missing')
    if closed.get('close_returned') is not True or closed.get('pending_simulation_callback_error') is not None:
        raise ValueError('shutdown_complete.json: App.close did not return cleanly')
    timestamps = [child.get('started_unix_s'), started.get('time_unix_s'),
                  closed.get('time_unix_s'), child.get('exited_unix_s')]
    if (type(child.get('exit_code')) is not int or child['exit_code'] != 0
            or child.get('timed_out') is not False or child.get('signals_sent') != []
            or type(child.get('pid')) is not int or child['pid'] <= 0
            or not all(isinstance(t, (float, int)) and math.isfinite(t) for t in timestamps)
            or timestamps != sorted(timestamps)):
        raise ValueError('launcher_exit.json: clean outer process exit evidence missing')
    return dict(physics_steps=STEPS, raw_sha256=sha, episode_uid=identity, pid=child['pid'])


def preflight(output):
    versions = {}
    for name in ('isaacsim', 'isaaclab'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    blockers = []
    if sys.platform != 'linux':
        blockers.append('native execution requires the Linux server')
    if versions['isaacsim'] not in ('4.5.0', '4.5.0.0') or versions['isaaclab'] != '0.54.4':
        blockers.append('expected installed Isaac Sim 4.5.0 and Isaac Lab 0.54.4')
    if output.exists() or output.is_symlink():
        blockers.append('output already exists')
    if not output.parent.is_dir():
        blockers.append('output parent must exist')
    if not CALIBRATION.is_file():
        blockers.append('calibration script missing')
    else:
        compile(CALIBRATION.read_bytes(), str(CALIBRATION), 'exec')
    if not THUMB_MESH.is_file():
        blockers.append('manufacturer thumb mesh missing')
    if not (ROOT / 'src' / 'go2_pvcnn' / 'control' / 'dh116'
            / 'tabletop_stereo_contact_calibration.py').is_file():
        blockers.append('calibration source missing')
    return dict(mode='preflight', ready_to_request_launch=not blockers, blockers=blockers,
                interpreter=sys.executable, versions=versions, output=str(output),
                calibration_sha256=hashlib.sha256(CALIBRATION.read_bytes()).hexdigest()
                if CALIBRATION.is_file() else None, physics_steps_per_trial=STEPS, trial_count=2,
                thumb_mesh_sha256=hashlib.sha256(THUMB_MESH.read_bytes()).hexdigest()
                if THUMB_MESH.is_file() else None,
                gpu_index=3, isaac_started=False)


def run_trial(output, timeout):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '3'
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(ROOT / 'src'), env.get('PYTHONPATH'))))
    command = [sys.executable, str(CALIBRATION), '--execute-native', '--side', 'left',
               '--steps', str(STEPS), '--enable-full-report-processing', '--full-contact-callback',
               '--ordinary-contact-callback', '--monitor-sigterm', '--output', str(output)]
    started = time.time()
    with output.with_suffix('.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            write_record(output.with_suffix('.timeout.json'), dict(
                pid=child.pid, timed_out=True, signals_sent=[], started_unix_s=started))
            raise RuntimeError(f'child {child.pid} did not exit naturally before the deadline') from None
    if not output.is_dir():
        write_record(output.with_suffix('.failed.json'), dict(pid=child.pid, exit_code=code,
                     timed_out=False, signals_sent=[]))
        raise RuntimeError(f'child {child.pid} exited without creating a trial directory')
    write_record(output / 'launcher_exit.json', dict(pid=child.pid, exit_code=code, timed_out=False,
                 signals_sent=[], started_unix_s=started, exited_unix_s=time.time()))
    return verify_trial(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-native', action='store_true')
    parser.add_argument('--timeout-s', type=float, default=600)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    status = preflight(output)
    if not args.execute_native:
        print(json.dumps(status, indent=2))
        return
    if status['blockers'] or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error(f'native preflight failed or timeout invalid: {status["blockers"]}')
    output.mkdir(exist_ok=False)
    write_record(output / 'preflight.json', status)
    first = run_trial(output / 'initial', args.timeout_s)
    second = run_trial(output / 'restart', args.timeout_s)
    if first['episode_uid'] == second['episode_uid']:
        raise RuntimeError('restart reused the first episode identity')
    write_record(output / 'shutdown_retest_passed.json', dict(initial=first, restart=second))


if __name__ == '__main__':
    main()
