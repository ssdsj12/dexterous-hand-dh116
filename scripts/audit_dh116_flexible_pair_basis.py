#!/usr/bin/env python3
"""CPU-only, non-overwriting replay of explicitly sourced pair basis inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis import analyze_pair_capture

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink() or args.output.resolve()==args.input.resolve():
        parser.error("output must be a new distinct evidence file")
    record=dict(analysis=None,native_qualified=False,teacher_admission=False)
    try:
        raw=args.input.read_bytes()
        record["input_sha256"]=hashlib.sha256(raw).hexdigest()
        def reject(value):raise ValueError("nonfinite JSON input: "+value)
        record["analysis"]=analyze_pair_capture(**json.loads(raw,parse_constant=reject))
        record["replayed"]=True
    except Exception as exc:
        record.update(replayed=False,failure=type(exc).__name__+": "+str(exc))
    record["native_imports"]=sorted(k for k in sys.modules if k.split(".")[0] in ("omni","isaaclab","isaacsim","warp","ropesim"))
    if record["native_imports"]:
        record.update(replayed=False,analysis=None,failure="unexpected native module import")
    files=[Path(__file__),ROOT/"src/go2_pvcnn/control/dh116/tabletop_deformable_pair_source_basis.py",
           ROOT/"src/go2_pvcnn/control/dh116/tabletop_deformx_pair_capture.py",
           ROOT/"src/go2_pvcnn/control/dh116/tabletop_deformable_contact_diagnostics.py"]
    record["implementation_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(record,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in record.items() if k!="analysis"},sort_keys=True))
    return 0 if record["replayed"] else 1

if __name__=="__main__":
    raise SystemExit(main())
