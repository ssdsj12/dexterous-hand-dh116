#!/usr/bin/env python3
"""One explicit offline arithmetic candidate on the locked historical Pinch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from go2_pvcnn.control.dh116.tabletop_approach_control_composer import evaluate_archive


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--approach-points-per-m",type=float,required=True,
        help="single explicit candidate; no default, original rigid coefficient or training qualification")
    args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("preserve existing output")
    result=evaluate_archive(approach_points_per_m=args.approach_points_per_m,project=ROOT)
    native=[name for name in sys.modules if name.split(".")[0] in
            ("omni","isaacsim","isaaclab","ropesim","warp")]
    if native:raise RuntimeError("offline import boundary crossed: "+str(native))
    result["native_imports"]=native
    raw=(json.dumps(result,indent=2,allow_nan=False)+"\n").encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(args.output),sha256=hashlib.sha256(raw).hexdigest(),
        known_reward_windows=result["known_reward_windows"],
        expected_reward_windows=result["expected_reward_windows"],
        complete_candidate_reward=result["complete_candidate_reward"],
        known_reward_sum=result["known_reward_sum"],
        training_reward=None,native_steps=0,optimizer_updates=0)))
    return 0

if __name__=="__main__":raise SystemExit(main())
