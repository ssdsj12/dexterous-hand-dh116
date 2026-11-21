#!/usr/bin/env python3
"""Prepare a source-bound Tripod17 Teacher Reset manifest from an exited run.

CPU evidence replay only; no Isaac instance, policy or training is started.
"""
import argparse
import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_tripod_reset_manifest import seal_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--receipt-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    binding = seal_manifest(args.run,receipt=args.receipt,expected_sha256=args.receipt_sha256,
                            destination=args.output)
    print(json.dumps(binding))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
