"""Serialize independent stage-five native experiments with frozen source."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _set_option(command, flag, value):
    """Set a single CLI value in a copied command, including --flag=value."""
    result, i = [], 0
    while i < len(command):
        token = command[i]
        if token == flag:
            if i + 1 >= len(command) or command[i+1].startswith('--'):
                raise ValueError(f'{flag} requires a value in evaluation template')
            i += 2
        elif token.startswith(flag + '='):
            i += 1
        else:
            result.append(token)
            i += 1
    return result + [flag, str(value)]


def _verify_bound_evaluation_inputs(command, argument_paths, input_hashes):
    """Check the actual child argv, canonical paths, and the pinned bytes."""
    for flag, expected_path in argument_paths.items():
        values = []
        for i, token in enumerate(command):
            if token == flag:
                if i+1 >= len(command):
                    raise ValueError('bound Student evaluation child argv is missing an input path')
                values.append(command[i+1])
            elif token.startswith(flag + '='):
                values.append(token.split('=', 1)[1])
        if (values != [expected_path]
                or str(Path(values[0]).resolve(strict=True)) != expected_path):
            raise ValueError('bound Student evaluation child argv differs from SHA-verified input paths')
        if hashlib.sha256(Path(values[0]).read_bytes()).hexdigest() != input_hashes[expected_path]:
            raise ValueError('bound Student evaluation input changed while awaiting native lock')


def build_bound_evaluation_command(command, *, checkpoint_path,
                                   expected_checkpoint_sha256, manifest_path,
                                   expected_manifest_sha256=None):
    """Read-only CPU preflight for independent, explicitly bound V4 evaluation.

    Native code/assets are verified again after Kit setup by the existing V4
    inference path, before any action. Authored manifest identities alone do
    not certify native reset/readback, contact, or Student success.
    """
    import torch
    from go2_pvcnn.control.dh116.tabletop_bimanual_checkpoint_contract import (
        VERSION, unwrap_physical_adapter_checkpoint,
    )
    from go2_pvcnn.control.dh116.tabletop_evaluation_trials import validate_manifest
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import evaluation_physics_sha256
    from go2_pvcnn.control.dh116.tabletop_vision_quaternion import validate_wrist_quaternion_preprocessing

    forbidden = ('--paired-bias-arm', '--paired-bias-reference', '--vision-m2-replay',
        '--vision-teacher-source', '--surface-teacher-replay', '--surface-mpc',
        '--bimanual-explicit-sdk-adapter', '--bimanual-joint-tracking-audit',
        '--bimanual-mapping-probe', '--bimanual-tripod-teacher-coordination-proof',
        '--bimanual-teacher-physics-audit')
    flags = {token.split('=', 1)[0] for token in command if token.startswith('--')}
    if flags.intersection(forbidden):
        raise ValueError('bound Student evaluation requires independent plain M0, exclusive of paired/Teacher/replay')
    for i, token in enumerate(command):
        if token == '--vision-mirror-preprocessing':
            if i+1 >= len(command) or command[i+1] != 'm0_raw':
                raise ValueError('bound Student evaluation requires independent plain M0 preprocessing')
        elif token.startswith('--vision-mirror-preprocessing=') and token.split('=', 1)[1] != 'm0_raw':
            raise ValueError('bound Student evaluation requires independent plain M0 preprocessing')
    if (not isinstance(expected_checkpoint_sha256, str) or len(expected_checkpoint_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_checkpoint_sha256)):
        raise ValueError('checkpoint SHA256 must be lowercase hexadecimal')
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    manifest = Path(manifest_path).resolve(strict=True)
    checkpoint_bytes = checkpoint.read_bytes()
    if hashlib.sha256(checkpoint_bytes).hexdigest() != expected_checkpoint_sha256:
        raise ValueError('bound Student checkpoint SHA256 mismatch')
    # torch.load from the exact checked bytes prevents a changing path from
    # selecting different bytes between the hash and structural validation.
    from io import BytesIO
    saved = torch.load(BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or saved.get('version') != VERSION:
        raise ValueError('bound Student evaluation requires independently qualified complete V4 checkpoint')
    core = unwrap_physical_adapter_checkpoint(saved, allow_explicit=True)
    source = core['contract']
    manifest_bytes = manifest.read_bytes()
    if (expected_manifest_sha256 is not None
            and hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256):
        raise ValueError('bound Student manifest changed during CPU preflight')
    value = validate_manifest(json.loads(manifest_bytes), min_trials_per_style=100)
    if len(value['trials']) != 200:
        raise ValueError('bound Student evaluation requires exactly 200 independent authored cases')
    if value['metadata']['camera_sha256'] != source['camera_config_sha256']:
        raise ValueError('evaluation manifest camera provenance mismatch')
    if value['metadata']['physics_sha256'] != evaluation_physics_sha256(source['privileged_critic_contract']):
        raise ValueError('evaluation manifest physics provenance mismatch')
    preprocessing = validate_wrist_quaternion_preprocessing(
        source.get('actor_wrist_quaternion_preprocessing', 'raw'))
    result = list(command)
    for flag, argument in (
        ('--vision-evaluation-checkpoint', checkpoint), ('--vision-evaluation-manifest', manifest),
        ('--num-envs', 200), ('--seed', value['metadata']['seed']),
        ('--vision-action-distribution', 'tanh_gaussian'),
        ('--vision-wrist-quaternion-preprocessing', preprocessing)):
        result = _set_option(result, flag, argument)
    if '--bimanual-checkpoint-adapter' not in result:
        result.append('--bimanual-checkpoint-adapter')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--bimanual-teacher-physics-audit', action='store_true')
    parser.add_argument('--bimanual-tripod-teacher-coordination-proof', type=Path)
    parser.add_argument('--vision-checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--vision-manifest', type=Path)
    parser.add_argument('--watch-env', type=int, action='append', default=[])
    parser.add_argument('--replay-arm', default='recorded', choices=('recorded','right_rotation_after240',
                        'right_translation_after240','tracking_guard','tracking_guard_coupled'))
    parser.add_argument('--mode', choices=('replay', 'probe', 'z0', 'z1', 'z2', 'z3', 'teacher',
                        'teacher_adapter', 'teacher_tripod_coordinated', 'bound_student_eval'), required=True)
    args = parser.parse_args()
    bound = args.mode == 'bound_student_eval'
    bound_options = (args.vision_checkpoint, args.checkpoint_sha256, args.vision_manifest)
    if any(value is not None for value in bound_options) and not bound:
        parser.error('--vision-checkpoint/--checkpoint-sha256/--vision-manifest are exclusive to bound_student_eval')
    if bound and not all(bound_options):
        parser.error('bound_student_eval requires --vision-checkpoint, --checkpoint-sha256 and --vision-manifest')
    if bound and (args.reference or args.watch_env or args.replay_arm != 'recorded'):
        parser.error('bound_student_eval excludes paired/replay options')
    if args.bimanual_teacher_physics_audit and not args.mode.startswith('teacher'):
        parser.error('--bimanual-teacher-physics-audit requires teacher or teacher_adapter mode')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('simple independent run name required')
    root = args.root.resolve()
    candidate = args.mode == 'teacher_tripod_coordinated'
    if bool(args.bimanual_tripod_teacher_coordination_proof) != candidate:
        parser.error('teacher_tripod_coordinated requires its T5 proof exclusively')
    if candidate:
        sys.path.insert(0, str(root/'source/src'))
        from go2_pvcnn.control.dh116.tabletop_bimanual_teacher_coordination import read_tripod_coordination_proof
        read_tripod_coordination_proof(args.bimanual_tripod_teacher_coordination_proof)
    output = root / args.name
    source = root / (args.name + '_source')
    if output.exists() or source.exists():
        raise FileExistsError('experiments are immutable; choose a new run name')
    template = root / ('teacher_template.json' if args.mode.startswith('teacher') else
                       'evaluation_template.json' if bound or args.mode.startswith('z') else 'replay_template.json')
    cfg = json.loads(template.read_text())
    old_source = cfg.get('cwd', str(root/'source'))
    source_command = cfg['command']
    bound_inputs = None
    bound_argument_paths = None
    if bound:
        sys.path.insert(0, str(root/'source/src'))
        bound_checkpoint = args.vision_checkpoint.resolve(strict=True)
        bound_manifest = args.vision_manifest.resolve(strict=True)
        manifest_sha = hashlib.sha256(bound_manifest.read_bytes()).hexdigest()
        source_command = build_bound_evaluation_command(source_command,
            checkpoint_path=bound_checkpoint, expected_checkpoint_sha256=args.checkpoint_sha256,
            manifest_path=bound_manifest, expected_manifest_sha256=manifest_sha)
        bound_inputs = {
            str(bound_checkpoint): args.checkpoint_sha256,
            str(bound_manifest): manifest_sha,
        }
        bound_argument_paths = {
            '--vision-evaluation-checkpoint': str(bound_checkpoint),
            '--vision-evaluation-manifest': str(bound_manifest),
        }
    source.mkdir()
    # This data mount allows content writes, but forbids chmod/utime.
    # Copy only content, then bind every Python file by its hash.
    for current, directories, files in os.walk(root/'source'):
        directories[:] = [name for name in directories if name not in ('.git', '__pycache__', '.pytest_cache')]
        relative = Path(current).relative_to(root/'source')
        destination = source/relative
        destination.mkdir(exist_ok=True)
        for name in files:
            shutil.copyfile(Path(current)/name, destination/name)
    command = [token.replace(old_source, str(source)) for token in source_command]
    command[command.index('--output')+1] = str(output)
    if bound_argument_paths is not None:
        # The legacy all-token source relocation must never relocate explicit
        # data inputs: copied source may contain a different same-named model,
        # and even a substring collision can point outside that copied tree.
        for flag, expected_path in bound_argument_paths.items():
            command = _set_option(command, flag, expected_path)
        _verify_bound_evaluation_inputs(command, bound_argument_paths, bound_inputs)
    if args.mode == 'replay':
        command += ['--bimanual-joint-tracking-audit', '--vision-replay-arm', args.replay_arm]
        for env_id in args.watch_env:
            command += ['--vision-replay-watch-env', str(env_id)]
    elif args.mode == 'probe':
        command[command.index('--num-envs')+1] = '8'
        command += ['--bimanual-mapping-probe']
    elif args.mode.startswith('z'):
        arm = dict(z0='m0', z1='right_z_zero', z2='z2', z3='z3')[args.mode]
        command += ['--paired-bias-arm', arm]
        if args.mode != 'z0':
            command += ['--paired-bias-reference', str(args.reference or root/'t3_z0_complete')]
    elif args.mode == 'teacher_adapter':
        command += ['--bimanual-explicit-sdk-adapter']
    if candidate:
        command += ['--bimanual-tripod-teacher-coordination-proof',
                    str(args.bimanual_tripod_teacher_coordination_proof.resolve())]
    if (args.bimanual_teacher_physics_audit or candidate) and '--bimanual-teacher-physics-audit' not in command:
        command += ['--bimanual-teacher-physics-audit']
    files = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in source.rglob('*') if path.is_file() and path.suffix != '.pyc'}
    meta = dict(command=command, cwd=str(source), source_sha256=files, env=cfg['env'], mode=args.mode)
    if bound_inputs is not None:
        meta['bound_evaluation_input_sha256'] = bound_inputs
        meta['bound_evaluation_argument_paths'] = bound_argument_paths
    with (root/(args.name+'_launch.json')).open('x') as stream:
        json.dump(meta, stream, indent=2)
    with open('/tmp/xk-dh116-stage4-native.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        running = subprocess.run(['pgrep', '-af', 'scripts/train_dh116_tabletop_gpu_ppo.py'],
                                 capture_output=True, text=True)
        if running.stdout.strip():
            raise RuntimeError('another native training/evaluation instance is active')
        if bound_inputs is not None:
            _verify_bound_evaluation_inputs(command, bound_argument_paths, bound_inputs)
        with (root/(args.name+'.log')).open('x') as log:
            status = subprocess.run(command, cwd=source, env=dict(os.environ, **cfg['env']),
                                    stdout=log, stderr=subprocess.STDOUT).returncode
        (root/(args.name+'_exit.json')).write_text(json.dumps(dict(exit_code=status)))
        raise SystemExit(status)


if __name__ == '__main__':
    main()
