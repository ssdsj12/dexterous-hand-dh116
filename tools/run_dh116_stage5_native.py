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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--bimanual-teacher-physics-audit', action='store_true')
    parser.add_argument('--bimanual-tripod-teacher-coordination-proof', type=Path)
    parser.add_argument('--watch-env', type=int, action='append', default=[])
    parser.add_argument('--replay-arm', default='recorded', choices=('recorded','right_rotation_after240',
                        'right_translation_after240','tracking_guard','tracking_guard_coupled'))
    parser.add_argument('--mode', choices=('replay', 'probe', 'z0', 'z1', 'z2', 'z3', 'teacher',
                        'teacher_adapter', 'teacher_tripod_coordinated'), required=True)
    args = parser.parse_args()
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
                       'evaluation_template.json' if args.mode.startswith('z') else 'replay_template.json')
    cfg = json.loads(template.read_text())
    old_source = cfg.get('cwd', str(root/'source'))
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
    command = [token.replace(old_source, str(source)) for token in cfg['command']]
    command[command.index('--output')+1] = str(output)
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
    with (root/(args.name+'_launch.json')).open('x') as stream:
        json.dump(meta, stream, indent=2)
    with open('/tmp/xk-dh116-stage4-native.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        running = subprocess.run(['pgrep', '-af', 'scripts/train_dh116_tabletop_gpu_ppo.py'],
                                 capture_output=True, text=True)
        if running.stdout.strip():
            raise RuntimeError('another native training/evaluation instance is active')
        with (root/(args.name+'.log')).open('x') as log:
            status = subprocess.run(command, cwd=source, env=dict(os.environ, **cfg['env']),
                                    stdout=log, stderr=subprocess.STDOUT).returncode
        (root/(args.name+'_exit.json')).write_text(json.dumps(dict(exit_code=status)))
        raise SystemExit(status)


if __name__ == '__main__':
    main()
