#!/usr/bin/env python3
"""Read a pinned historical journal; no simulation, training or qualification."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from go2_pvcnn.control.dh116.tabletop_readonly_control_cost import evaluate_archive

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp",choices=("pinch","tripod","power"),required=True)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("exclusive output already exists; preserve original evidence")
    result=evaluate_archive(args.grasp,input_path=args.input)
    imported=sorted(n for n in sys.modules if n.split(".")[0] in ("omni","isaaclab","isaacsim","ropesim","warp"))
    if imported:
        raise RuntimeError("native import crossed the offline evaluation boundary")
    result["native_imports"]=imported
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("grasp","status","physical_samples","training_reward","optimizer_updates")}))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
