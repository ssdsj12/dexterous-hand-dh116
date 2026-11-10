"""Fit only the visual actor to synchronized native Teacher/MPC issued labels."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', type=Path, required=True,
                        help='complete v3 PPO base; its full contract must match the dataset')
    parser.add_argument('--dataset', type=Path, required=True,
                        help='new synchronized successful visual supervision pool')
    parser.add_argument('--output', type=Path, required=True, help='new independent BC artifact')
    parser.add_argument('--warm-start-output', type=Path,
                        help='optionally derive a separate new complete v3 PPO warm start')
    parser.add_argument('--steps', type=int, default=100, help='bounded offline BC updates, never PPO updates')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--actor-initialization', choices=('source', 'reset_pointnet'), default='source',
                        help='source rejects an inactive encoder; explicit reset_pointnet reinitializes only that branch')
    parser.add_argument('--wrist-quaternion-preprocessing',
                        choices=('raw', 'wrist_quaternion_hemisphere_v1'), default='raw',
                        help='explicit new BC actor input convention; raw preserves historical actors')
    parser.add_argument('--startup-sampling-fraction', type=float, default=0.,
                        help='canonical BC only: mix 0..0.5 training probability into startup rows')
    parser.add_argument('--startup-control-steps', type=int, default=4,
                        help='startup rows have tick below this positive control-step count')
    parser.add_argument('--heldout-source-trajectory', action='append', default=[],
                        help='whole independent source trajectory ID, repeatable')
    parser.add_argument('--allow-fit-only', action='store_true',
                        help='allow no independent holdout; reports no generalization evidence')
    args = parser.parse_args(argv)
    from go2_pvcnn.control.dh116.tabletop_vision_supervised_bc import (
        fit_supervision, derive_supervised_checkpoint)
    report = fit_supervision(args.source_checkpoint, args.dataset, args.output,
        steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate,
        seed=args.seed, heldout_trajectories=args.heldout_source_trajectory,
        allow_fit_only=args.allow_fit_only, actor_initialization=args.actor_initialization,
        wrist_quaternion_preprocessing=args.wrist_quaternion_preprocessing,
        startup_sampling_fraction=args.startup_sampling_fraction,
        startup_control_steps=args.startup_control_steps)
    if args.warm_start_output:
        report = dict(fit=report, derivation=derive_supervised_checkpoint(
            args.output, args.warm_start_output))
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return report


if __name__ == '__main__': main()
