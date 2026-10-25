"""Continue the same PPO curriculum across safe budget exits, serially."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_continuation import continue_training


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _control_arguments(contract, close_reset_snapshot):
    action = contract.get('action')
    if action == '12D_bounded_nominal_residual_v2':
        if close_reset_snapshot is not None:
            raise ValueError('residual checkpoint cannot use a close reset snapshot')
        return []
    if action != '12D_direct_joint_target_v1':
        raise ValueError(f'unsupported checkpoint action contract: {action}')
    if close_reset_snapshot is None:
        raise ValueError('direct checkpoint requires --close-reset-snapshot')
    snapshot = Path(close_reset_snapshot).resolve()
    if not snapshot.is_file() or _sha256(snapshot) != contract.get('close_seed_sha256'):
        raise ValueError('close reset snapshot SHA256 does not match checkpoint contract')
    try:
        radius = float(contract['target_radius_rad'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('direct checkpoint requires a joint target radius') from exc
    if not math.isfinite(radius) or not 0 < radius <= .15:
        raise ValueError('direct checkpoint joint target radius must be in (0,.15]')
    return ['--control-mode', 'joint_target', '--close-reset-snapshot', str(snapshot),
            '--joint-target-radius-rad', str(radius)]


def load_continuation_contract(checkpoint, pose_summary, close_reset_snapshot=None):
    """Reject incompatible reset artifacts before launching any simulator."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError('resume checkpoint does not exist')
    contract = json.loads(checkpoint.with_suffix('.json').read_text())['contract']
    pose = Path(pose_summary)
    if not pose.is_file() or _sha256(pose) != contract.get('pose_sha256'):
        raise ValueError('pose summary SHA256 does not match checkpoint contract')
    _control_arguments(contract, close_reset_snapshot)
    return contract


def build_training_command(root, contract, checkpoint, output, *, pose_summary,
                           timesteps, checkpoint_every, eval_episodes,
                           close_reset_snapshot=None):
    """Keep action/reset/physics settings identical across serial rounds."""
    root = Path(root)
    command = ['bash', str(root/'scripts/python.sh'),
        str(root/'scripts/train_dh116_tabletop_ppo.py'),
        '--pose-summary', str(Path(pose_summary).resolve()), '--output', str(Path(output).resolve()),
        '--resume', str(Path(checkpoint).resolve()), '--curriculum', '--timesteps', str(timesteps),
        '--checkpoint-every', str(checkpoint_every), '--eval-episodes', str(eval_episodes),
        '--residual-scale-rad', str(contract['residual_scale_rad']),
        '--reset-height-jitter-m', str(contract['reset_height_jitter_m']),
        '--solver-type', str(contract['solver_type']),
        '--position-iterations', str(contract['position_iterations']),
        '--physics-hz', str(contract['physics_hz']), '--headless', '--device', 'cpu']
    return command + _control_arguments(contract, close_reset_snapshot)


def continuation_sources(root, pose_summary, close_reset_snapshot=None):
    """Pin runtime code and the reset inputs against changes between rounds."""
    root = Path(root).resolve()
    sources = [root/'scripts/train_dh116_tabletop_ppo.py', Path(__file__).resolve(),
               root/'scripts/python.sh', Path(pose_summary).resolve()]
    sources += [root/'src/go2_pvcnn/control/dh116'/name for name in (
        'tabletop_control.py', 'tabletop_env.py', 'tabletop_task.py', 'tabletop_training.py',
        'tabletop_continuation.py', 'tabletop_direct.py', 'tabletop_policy.py',
        'runtime_mimic.py', 'linear_coupling.py', 'hand_only_scene.py')]
    if close_reset_snapshot is not None:
        sources.append(Path(close_reset_snapshot).resolve())
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--pose-summary', type=Path, required=True)
    parser.add_argument('--close-reset-snapshot', type=Path,
                        help='Required for direct-target checkpoints; SHA256 must match saved contract')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timesteps', type=int, default=65536, help='New steps per stage per round')
    parser.add_argument('--max-rounds', type=int, default=8)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--checkpoint-every', type=int, default=2048)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.max_rounds <= 0 or args.eval_episodes < 2 or args.checkpoint_every <= 0:
        parser.error('positive budgets/interval and >=2 evaluations required')
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    try:
        contract = load_continuation_contract(args.resume, args.pose_summary, args.close_reset_snapshot)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    (root/'runs').mkdir(exist_ok=True)
    lock = (root/'runs/.tabletop_continuation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A manually started trainer does not hold our lock. Do not overlap it.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(Path(arg.decode(errors='replace')).name == 'train_dh116_tabletop_ppo.py'
               for arg in argv if arg):
            raise RuntimeError(f'Existing tabletop trainer PID {proc.name}; refusing a second Isaac instance')
    def command(checkpoint, output, index):
        return build_training_command(root, contract, checkpoint, output,
            pose_summary=args.pose_summary, timesteps=args.timesteps,
            checkpoint_every=args.checkpoint_every, eval_episodes=args.eval_episodes,
            close_reset_snapshot=args.close_reset_snapshot)
    sources = continuation_sources(root, args.pose_summary, args.close_reset_snapshot)
    result = continue_training(args.resume, args.output, command,
                               max_rounds=args.max_rounds, source_paths=sources)
    print(json.dumps(result), flush=True)
    return 0 if result['grasp_qualified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
