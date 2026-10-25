#!/usr/bin/env python3
"""Build the portable original DeformX wire without launching Isaac."""
import argparse
import json
from pathlib import Path
from go2_pvcnn.control.dh116.deformx_wire_asset import build_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/xk/coding/DeformX/usd/wire/wire_orange_s20_r0.005_l1_smooth.usdc'))
    parser.add_argument('--output', type=Path, default=Path('assets/deformx_wire_v1'))
    args = parser.parse_args()
    print(json.dumps(build_asset(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
