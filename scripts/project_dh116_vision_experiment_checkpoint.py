"""Offline SHA-pinned v3 snapshot projection to a new small balanced experiment."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--source-iteration',type=int,default=1439,
                        help='independently verified completed source update (default: 1439)')
    parser.add_argument('--num-envs',type=int,choices=(8,16,128),default=16)
    args=parser.parse_args(argv)
    from go2_pvcnn.control.dh116.tabletop_vision_experiment_checkpoint import project_experiment_checkpoint
    report=project_experiment_checkpoint(args.input,args.output,num_envs=args.num_envs,
        expected_source_sha256=args.source_sha256,expected_source_iteration=args.source_iteration)
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return report


if __name__=='__main__':main()
