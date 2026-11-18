#!/usr/bin/env python3
"""Fit the existing tabletop actor from physically qualified teacher episodes."""
import argparse
import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_behavior_cloning import train_imitation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    args = parser.parse_args()
    result = train_imitation(args.dataset, args.output, device=args.device,
        epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
        learning_rate=args.learning_rate, validation_fraction=args.validation_fraction)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
