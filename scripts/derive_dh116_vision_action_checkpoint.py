"""Derive a new v3 tanh-Gaussian visual checkpoint from a complete v2 source."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='trusted complete v2 visual PPO checkpoint; never modified')
    parser.add_argument('--output', type=Path, required=True,
                        help='new .pt checkpoint; its same-name .json sidecar must also be absent')
    args = parser.parse_args(argv)
    from go2_pvcnn.control.dh116.tabletop_vision_action_derivation import derive_action_checkpoint
    report = derive_action_checkpoint(args.input, args.output)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    main()
