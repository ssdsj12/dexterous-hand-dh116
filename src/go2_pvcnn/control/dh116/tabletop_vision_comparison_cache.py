"""Strict byte-bound reuse of saved-observation comparisons, never physics.

Every native job still loads its policy, cold-resets, checks the actual reset,
and collects its own policy/physics/terminal evidence. Only the completed pure
offline comparison gzip may be reused for identical model/data/context files
and identical branch-comparison implementation bytes. Mismatches raise; this
module never silently recomputes, substitutes a cache, or certifies a trial.
"""
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil


VERSION = 'dh116-saved-teacher-comparison-cache-v1'
SCOPE = 'offline_saved_teacher_observation_comparisons_only'
INPUTS = ('checkpoint', 'teacher_dataset', 'sequence_context')


def _digest(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdefABCDEF' for c in value)):
        raise ValueError('explicit SHA256 required')
    return value.lower()


def _file_identity(path):
    try:
        path = Path(path).resolve(strict=True)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
        return dict(path=str(path), sha256=digest.hexdigest())
    except (OSError, TypeError, ValueError) as error:
        raise ValueError('missing/unreadable immutable comparison source: ' + str(path)) from error


def _checked(value, parent):
    if not isinstance(value, dict) or not isinstance(value.get('path'), str) or not value['path']:
        raise ValueError('explicit comparison path/SHA256 identity required')
    path = Path(value['path'])
    if not path.is_absolute(): path = parent / path
    actual = _file_identity(path)
    if actual['sha256'] != _digest(value.get('sha256')):
        raise ValueError('immutable comparison SHA256 changed: ' + str(path))
    return actual


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value): raise ValueError('nonfinite comparison JSON')
    if isinstance(value, dict):
        for child in value.values(): _finite(child)
    if isinstance(value, list):
        for child in value: _finite(child)


def _json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        _finite(value)
        if not isinstance(value, dict): raise ValueError('comparison metadata must be an object')
        return value
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid immutable comparison JSON: ' + str(path)) from error


def _inputs(value, parent):
    if not isinstance(value, dict) or not set(INPUTS) <= value.keys():
        raise ValueError('checkpoint/teacher_dataset/sequence_context identities required')
    return {name: _checked(value[name], parent) for name in INPUTS}


def _dependencies(value):
    from .tabletop_vision_branching_eval import comparison_dependencies
    current = comparison_dependencies()
    if not isinstance(value, dict) or set(value) != set(current):
        raise ValueError('complete numerical comparison dependency manifest required')
    return {name: _digest(sha) for name, sha in value.items()}


def _source_snapshot(path, *, scan_stream):
    identity = _file_identity(path); parent = Path(identity['path']).parent
    report = _json(identity['path'])
    if report.get('native_runtime_verified') is not True:
        raise ValueError('source comparison report requires genuine native runtime provenance')
    summary = report.get('same_state_comparisons')
    if (not isinstance(summary, dict) or summary.get('reused_offline_comparisons') is not False
            or summary.get('physical_failure_causality_claimed') is not False):
        raise ValueError('complete directly computed offline comparison summary required; cache-of-cache is refused')
    implementation = _digest(summary.get('comparison_implementation_sha256'))
    dependencies = _dependencies(summary.get('comparison_dependencies'))
    if dependencies['tabletop_vision_branching_eval'] != implementation:
        raise ValueError('comparison dependency manifest disagrees with branch implementation SHA256')
    expected = _inputs(report.get('inputs'), parent)
    if _inputs(summary.get('comparison_inputs'), parent) != expected:
        raise ValueError('offline comparison input identities differ from native report')
    stream = _checked(summary, parent)
    rows = summary.get('policy_rows')
    if type(rows) is not int or rows < 1:
        raise ValueError('complete positive offline policy_rows count required')
    validation = None
    if scan_stream:
        count = 0
        try:
            with gzip.open(stream['path'], 'rt', encoding='utf-8') as source:
                for line in source:
                    value = json.loads(line); _finite(value)
                    if not isinstance(value, dict): raise ValueError('offline comparison JSONL rows must be objects')
                    count += 1
        except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError('incomplete/corrupt offline comparison gzip') from error
        if count != rows: raise ValueError('offline comparison stream is incomplete: policy_rows mismatch')
        validation = dict(gzip_crc_verified=True, parsed_policy_rows=count, format='finite_JSON_objects_JSONL')
    return dict(identity=identity, report=report, summary=deepcopy(summary), inputs=expected,
        source_stream=stream, implementation=implementation, dependencies=dependencies, stream_validation=validation)


def _check_snapshot(snapshot):
    parent = Path(snapshot['identity']['path']).parent
    for item in (snapshot['identity'], snapshot['source_stream'], *snapshot['inputs'].values()):
        _checked(item, parent)


def build_comparison_cache(source_native_report, output):
    """Copy completed offline bytes into an exclusive cache directory.

    Physical failures/incomplete episodes are retained as source provenance;
    they are neither a cache-eligibility gate nor a new physical certificate.
    A complete gzip/row count and the genuine report flag are still required.
    Return the cache manifest's exact path/SHA identity for native config use.
    """
    output = Path(output).absolute()
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    snapshot = _source_snapshot(source_native_report, scan_stream=True)
    output.mkdir(parents=True, exist_ok=False)
    copied = output / 'comparison_stream.jsonl.gz'
    with Path(snapshot['source_stream']['path']).open('rb') as source, copied.open('xb') as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    cached = _file_identity(copied)
    if cached['sha256'] != snapshot['source_stream']['sha256']:
        raise ValueError('offline stream changed while copying; no usable cache published')
    _check_snapshot(snapshot)
    metadata = dict(version=VERSION, scope=SCOPE, source_native_runtime_verified=True,
        source_native_report=snapshot['identity'], source_status=snapshot['report'].get('status'),
        source_error=snapshot['report'].get('error'), source_summary=snapshot['summary'],
        comparison_inputs=snapshot['inputs'], comparison_implementation_sha256=snapshot['implementation'],
        comparison_dependencies=snapshot['dependencies'],
        source_stream=snapshot['source_stream'], cached_stream=cached,
        stream_validation=snapshot['stream_validation'], native_trials_reused=0)
    manifest = output / 'cache.json'
    with manifest.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return _file_identity(manifest)


def load_comparison_cache(identity, expected_config, *, implementation_sha256):
    """Load only an exact cache match; every discrepancy is an explicit error.

    Model/data/context paths and bytes, the source report, original gzip, copy
    gzip, manifest, and comparison code SHA are checked again. Reset/physics/
    terminal configuration is intentionally never loaded from the cache.
    """
    actual = _checked(identity, Path.cwd()); parent = Path(actual['path']).parent
    metadata = _json(actual['path'])
    if (metadata.get('version') != VERSION or metadata.get('scope') != SCOPE
            or metadata.get('source_native_runtime_verified') is not True
            or metadata.get('native_trials_reused') != 0):
        raise ValueError('unsupported offline-only comparison cache metadata')
    expected_code = _digest(implementation_sha256)
    if metadata.get('comparison_implementation_sha256') != expected_code:
        raise ValueError('comparison implementation SHA256 changed; no silent fallback')
    from .tabletop_vision_branching_eval import comparison_dependencies
    if _dependencies(metadata.get('comparison_dependencies')) != comparison_dependencies():
        raise ValueError('numerical comparison dependency code changed; no silent fallback')
    report_id = _checked(metadata.get('source_native_report'), parent)
    snapshot = _source_snapshot(report_id['path'], scan_stream=False)
    if (snapshot['identity'] != report_id or snapshot['implementation'] != expected_code
            or snapshot['summary'] != metadata.get('source_summary')
            or snapshot['source_stream'] != metadata.get('source_stream')
            or snapshot['inputs'] != metadata.get('comparison_inputs')
            or snapshot['dependencies'] != metadata.get('comparison_dependencies')
            or snapshot['report'].get('status') != metadata.get('source_status')
            or snapshot['report'].get('error') != metadata.get('source_error')):
        raise ValueError('cache metadata differs from immutable native/offline source provenance')
    config_parent = Path(expected_config.get('config_identity', {}).get('path', str(parent))).parent
    if _inputs(expected_config, config_parent) != snapshot['inputs']:
        raise ValueError('comparison cache belongs to different checkpoint/data/context paths or bytes')
    copied = _checked(metadata.get('cached_stream'), parent)
    if (Path(copied['path']) != parent / 'comparison_stream.jsonl.gz'
            or copied['sha256'] != snapshot['source_stream']['sha256']):
        raise ValueError('cache copy stream path/SHA256 differs from original completed offline bytes')
    validation = metadata.get('stream_validation', {})
    if (validation.get('gzip_crc_verified') is not True
            or validation.get('parsed_policy_rows') != snapshot['summary']['policy_rows']
            or validation.get('format') != 'finite_JSON_objects_JSONL'):
        raise ValueError('comparison stream completeness evidence is missing or inconsistent')
    _check_snapshot(snapshot); _checked(actual, parent); _checked(copied, parent)
    result = deepcopy(snapshot['summary'])
    result.update(path=copied['path'], sha256=copied['sha256'], comparison_inputs=snapshot['inputs'],
        reused_offline_comparisons=True, native_trials_reused=0,
        cache_provenance=dict(scope=SCOPE, cache=actual, source_native_report=report_id,
            source_status=metadata['source_status'], source_error=metadata['source_error'],
            original_stream=snapshot['source_stream'], copied_stream=copied,
            native_physics_reused=False, native_reset_reused=False, native_terminal_reused=False))
    return result
