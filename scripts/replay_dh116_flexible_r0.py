#!/usr/bin/env python3
"""Read-only disk replay of owned physical episode data; no native launch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import (
    decode_owned,encode_owned,replay_diagnostic_bundle,VERSION,
)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.input.resolve()==args.output.resolve():
        parser.error("input and output must be distinct")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; preserve prior replay evidence")
    receipt=dict(version=VERSION,replay_complete=False,report=None,
                 native_qualification=False,teacher_admission=False,
                 new_physics_steps=0,optimizer_updates=0,input_sha256=None)
    try:
        raw=args.input.read_bytes()
        receipt["input_sha256"]=hashlib.sha256(raw).hexdigest()
        def forbidden(value):
            raise ValueError("nonfinite JSON literal must use an explicit owned tag: "+value)
        bundle=decode_owned(json.loads(raw,parse_constant=forbidden))
        receipt["report"]=encode_owned(replay_diagnostic_bundle(bundle))
        receipt["replay_complete"]=True
    except Exception as exc:
        receipt["failure"]=type(exc).__name__+": "+str(exc)
    receipt["native_imports"]=sorted(name for name in sys.modules
        if name in ("omni","isaaclab","isaacsim","warp") or
        name.startswith(("omni.","isaaclab.","isaacsim.","warp.")))
    if receipt["native_imports"]:
        receipt.update(replay_complete=False,report=None,failure="native module import crossed CPU boundary")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="report"},sort_keys=True))
    return 0 if receipt["replay_complete"] else 1


if __name__=="__main__":
    raise SystemExit(main())
