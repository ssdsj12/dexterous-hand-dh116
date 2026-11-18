"""Read-only boundary audit for legacy MPC evidence entering Teacher BC."""
import argparse
import gzip
import io
import json
from pathlib import Path

import torch

from . import tabletop_stereo_promotion as promotion
from .tabletop_privileged_expert_assembly import _episode_source_rows
from .tabletop_privileged_warm_start import _checked_bytes
from .tabletop_stereo_contract import file_sha256, write_json_new


def _json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def _raw_schema(path):
    # An active writer may not have closed the stream; inspect only one
    # available line. Shape observations never certify complete coverage.
    observed = dict(file_present=path.is_file(), schema_observed=False)
    if not observed['file_present']:
        return observed
    opener = gzip.open if path.suffix == '.gz' else open
    try:
        with opener(path, 'rt', encoding='utf-8') as stream:
            line = stream.readline()
        if line:
            row = json.loads(line)
            observed.update(schema_observed=True, first_record_fields=sorted(row))
            for name in ('state', 'producer_evidence'):
                if isinstance(row.get(name), dict):
                    observed[name + '_fields'] = sorted(row[name])
    except (EOFError, OSError, TypeError, ValueError) as error:
        observed['unreadable_preview'] = str(error)
    return observed


def _partition(path, outcome):
    info = dict(file_present=path.is_file(), rows=None, episodes=None, fully_valid_371_rows=None)
    if not info['file_present']:
        return info
    archive = torch.load(path, map_location='cpu', weights_only=True)
    obs, actions, valid, rows = (archive.get(key) for key in
        ('observations', 'actions', 'privileged_valid', 'sample_metadata'))
    if (archive.get('first_episode_outcome') != outcome or archive.get('qualified_for_bc') is not False
            or not isinstance(obs, torch.Tensor) or obs.ndim != 2 or obs.shape[1] != 371
            or not isinstance(actions, torch.Tensor) or actions.shape != (len(obs), 24)
            or not isinstance(valid, torch.Tensor) or valid.shape != obs.shape or valid.dtype != torch.bool
            or not isinstance(rows, list) or len(rows) != len(obs)
            or not all(isinstance(row, dict) and isinstance(row.get('episode_uid'), str) for row in rows)):
        raise ValueError('invalid archived MPC first-episode partition: ' + outcome)
    return dict(info, rows=len(rows), episodes=len({row['episode_uid'] for row in rows}),
                fully_valid_371_rows=int(valid.all(dim=1).sum()),
                valid_fields_max=int(valid.sum(dim=1).max()) if len(rows) else None,
                archive_sha256=file_sha256(path))


def _prove_episode(path, sha256, identity, source):
    outcome = dict(proof_path=str(path), proof_sha256=sha256, ready_for_assembly=False)
    try:
        proof = json.loads(_checked_bytes(path, sha256, 'reviewed native episode proof'))
        result = promotion.verify_native_episode(path, expected_identity=identity, subject='mpc')
        if result.get('native_evidence_verified') is not True or result.get('qualified_success') is not True:
            raise ValueError('independent native MPC success proof has not passed')
        if not source.is_file():
            raise ValueError('same-process successful source ledger is missing')
        raw, _ = promotion._artifact(proof['raw_artifact'], path.parent)
        physical = [row for row in promotion._jsonl(raw)
                    if row.get('kind', 'sample') == 'sample' and row.get('live', True)]
        source_raw, _ = promotion._process_output(proof, path.parent, dict(path=str(source), sha256=file_sha256(source)))
        archive = torch.load(io.BytesIO(source_raw), map_location='cpu', weights_only=True)
        if archive.get('action_owner') != 'legacy_teacher_mpc' or archive.get('first_episode_outcome') != 'successful':
            raise ValueError('exact MPC success partition owner required')
        if file_sha256(path) != sha256 or len(physical) != result['sample_count']:
            raise ValueError('native proof or its physical sample coverage changed')
        observations, _ = _episode_source_rows(archive, result['episode_uid'], physical)
        outcome.update(ready_for_assembly=True, episode_uid=result['episode_uid'], observations=len(observations))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        outcome['error'] = str(error)
    return outcome


def inspect_run(run, *, observer=None, completion=None, episode_proofs=(), expected_identity=None):
    run = Path(run).resolve(strict=True)
    if not run.is_dir():
        raise ValueError('existing MPC run directory required')
    if episode_proofs and not isinstance(expected_identity, dict):
        raise ValueError('exact target Teacher identity required to verify proposed episode proofs')
    monitor = _json(run / 'monitor.json') or {}
    report = _json(run / 'teacher_collection_report.json') or {}
    outcomes = ('successful', 'failed', 'incomplete')
    names = {key: run / f'mpc_{key}_first_episodes_unqualified.pt' for key in outcomes}
    partitions = {key: _partition(names[key], key) for key in outcomes}
    legacy = {path.name: _raw_schema(path) for path in (
        run / 'teacher_native_physics.jsonl', run / 'mpc_state_capture/states_480hz.jsonl.gz',
        run / 'mpc_state_capture/sensor_evidence_480hz.jsonl.gz')}
    advisory = None
    if observer is not None:
        code = Path(observer) / 'exit_code'
        if code.is_file():
            advisory = int(code.read_text().strip())
    completed = _json(Path(completion)) if completion is not None else None
    process = dict(advisory_observed_exit_code=advisory,
        formal_completion_supplied=completed is not None,
        formal_completion_version=completed.get('version') if isinstance(completed, dict) else None,
        formal_completion_returncode=completed.get('returncode') if isinstance(completed, dict) else None)
    proofs = [_prove_episode(Path(path).resolve(strict=True), sha256, expected_identity, names['successful'])
              for path, sha256 in episode_proofs]
    eligible = {row['episode_uid'] for row in proofs if row['ready_for_assembly']}
    blocked = []
    success = partitions['successful']
    if not success['file_present']:
        blocked.append('original_completed_success_partition_missing')
    elif not success['rows']:
        blocked.append('no_successful_native_task_rows')
    elif success['fully_valid_371_rows'] != success['rows']:
        blocked.append('successful_rows_have_unverified_371_state')
    if not eligible:
        blocked.append('no_canonical_480hz_episode_and_force_support_proof')
    if not proofs:
        blocked.append('independent_native_episode_proofs_missing')
    elif any(not row['ready_for_assembly'] for row in proofs):
        blocked.append('proposed_native_episode_proof_rejected')
    return dict(version='dh116-mpc-teacher-proof-preflight-v1', scope='read_only_no_expert_pack_written',
        legacy_native_progress=dict(control_steps=report.get('control_steps', monitor.get('control_steps')),
            task_successes=report.get('successful_episodes', monitor.get('successful_episodes'))),
        legacy_partitions=partitions, legacy_raw_schema=legacy, process_evidence=process,
        episode_proofs=proofs, episodes_ready_for_assembly=len(eligible), blocked_reasons=blocked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--observer', type=Path)
    parser.add_argument('--completion', type=Path)
    parser.add_argument('--episode-proof', nargs=2, action='append', metavar=('PATH', 'SHA256'), default=[])
    parser.add_argument('--model-identity', type=Path)
    parser.add_argument('--model-identity-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    identity = None
    if args.episode_proof:
        if args.model_identity is None or args.model_identity_sha256 is None:
            parser.error('proposed proof verification requires a SHA-bound target model identity')
        identity = json.loads(_checked_bytes(args.model_identity, args.model_identity_sha256, 'target Teacher identity'))
    value = inspect_run(args.run, observer=args.observer, completion=args.completion, episode_proofs=args.episode_proof,
        expected_identity=identity)
    if args.output is not None:
        write_json_new(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0
