"""Serial continuation of bounded PPO runs; never retry a physical failure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _hashes(paths):
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def continue_training(checkpoint, output, build_command, *, max_rounds,
                      source_paths=(), poll_interval=1.):
    """Run one child at a time; only a safe unmastered stage can be resumed.

    build_command(checkpoint, new_output_directory, round_index) returns argv.
    Source hashes prohibit silently loading modified control code in a later
    round. Artifact/CLI errors, missing summaries, and physical failures stop.
    """
    checkpoint, output = Path(checkpoint), Path(output)
    if max_rounds < 1 or poll_interval <= 0 or not checkpoint.is_file():
        raise ValueError('checkpoint, positive round limit and poll interval required')
    output.mkdir(parents=True, exist_ok=False)
    pins = _hashes(source_paths)
    state = dict(status='running', grasp_qualified=False, checkpoint=str(checkpoint),
                 max_rounds=max_rounds, rounds=[], source_sha256=pins)
    child = None
    try:
        for index in range(1, max_rounds+1):
            if _hashes(source_paths) != pins:
                state['status'] = 'source_changed'
                break
            directory = output/f'round_{index:04d}'
            record = dict(index=index, output=str(directory), resumed_from=str(checkpoint))
            state['rounds'].append(record)
            state['current_round'] = index
            _write(output/'monitor.json', state)
            with (output/f'round_{index:04d}.log').open('wb') as log:
                child = subprocess.Popen(build_command(checkpoint, directory, index),
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True)
                record['pid'] = child.pid
                _write(output/'monitor.json', state)
                next_report = 0.
                while child.poll() is None:
                    if time.monotonic() >= next_report:
                        progress = directory/'training_progress.json'
                        if progress.exists():
                            state['progress'] = json.loads(progress.read_text())
                        _write(output/'monitor.json', state)
                        print(json.dumps(dict(round=index, status='running',
                                              progress=state.get('progress'))), flush=True)
                        next_report = time.monotonic()+30.
                    time.sleep(poll_interval)
                record['exit_code'] = child.returncode
            child = None
            summary_path = directory/'summary.json'
            if not summary_path.exists():
                state['status'] = 'missing_summary'
                break
            summary = json.loads(summary_path.read_text())
            record.update(training_stopped=summary.get('training_stopped'),
                          ppo_timesteps=summary.get('ppo_timesteps'),
                          ppo_updates=summary.get('ppo_updates'),
                          completed_stages=summary.get('completed_stages', []))
            unsafe = (summary.get('training_stopped') == 'physical_safety_failure'
                or summary.get('error') or summary.get('interrupted')
                or summary.get('training_skipped')
                or any(s.get('safety_stop') or s.get('evaluation', {}).get('physically_safe') is False
                       for s in summary.get('stages', [])))
            if record['exit_code'] != 0 or summary.get('process_exit_code', 0) != 0 or unsafe:
                state['status'] = 'failed'
                break
            checkpoint = directory/'ppo_checkpoint.zip'
            if not checkpoint.is_file() or not checkpoint.with_suffix('.json').is_file():
                state['status'] = 'missing_checkpoint'
                break
            state['checkpoint'] = str(checkpoint)
            if summary.get('passed') is True and summary.get('grasp_qualified') is True:
                state.update(status='qualified', grasp_qualified=True)
                break
            if summary.get('training_stopped') != 'curriculum_stage_not_mastered':
                state['status'] = 'unexpected_stop'
                break
            state.pop('progress', None)
        else:
            state['status'] = 'budget_exhausted'
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=60.)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=10.)
        raise
    except BaseException as exc:
        state.update(status='error', error=f'{type(exc).__name__}: {exc}')
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            child.wait(timeout=60.)
        raise
    finally:
        _write(output/'monitor.json', state)
    return state
