"""Derive an immutable 371D surface PPO checkpoint from its audited 351D source."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='complete original 351D / 24D surface PPO checkpoint')
    parser.add_argument('--output', type=Path, required=True, help='new .pt checkpoint; its .json sidecar must also be absent')
    args = parser.parse_args(argv)
    from go2_pvcnn.control.dh116.tabletop_surface_checkpoint import derive_surface_checkpoint
    manifest = derive_surface_checkpoint(args.input, args.output)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False), flush=True)
    return manifest


if __name__ == '__main__':
    main()
