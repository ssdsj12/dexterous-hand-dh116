"""Prepare an explicit immutable surface PPO repair checkpoint, offline."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='original 351D / 24D surface PPO checkpoint')
    parser.add_argument('--output', type=Path, required=True, help='new checkpoint; its .json sidecar must also be absent')
    parser.add_argument('--demonstrations', type=Path, required=True, help='exact saved imitation dataset file')
    args = parser.parse_args(argv)
    from go2_pvcnn.control.dh116.tabletop_surface_repair_resume import prepare_repair_resume
    manifest = prepare_repair_resume(args.input, args.output, args.demonstrations)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False), flush=True)
    return manifest


if __name__ == '__main__':
    main()
