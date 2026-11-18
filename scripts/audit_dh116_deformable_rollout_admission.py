"""Read-only flexible Teacher rollout audit. Exit 3 means admission denied."""
import argparse
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_deformable_rollout_admission import run_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--request-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    return run_audit(Path(__file__).resolve().parents[1],args.request,args.request_sha256,args.output)


if __name__ == "__main__":
    raise SystemExit(main())
