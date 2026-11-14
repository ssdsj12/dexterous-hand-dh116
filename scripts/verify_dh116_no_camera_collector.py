#!/usr/bin/env python3
"""Read-only per-style admission audit; never constructs a policy or physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_no_camera_collector import actual_admission, VERSION


def audit(project):
    project = Path(project).resolve()
    names = ("tabletop_no_camera_collector.py", "tabletop_actual_backend_teacher.py",
             "tabletop_actual_backend_checkpoint.py", "tabletop_deformable_dual_hand_episode.py",
             "tabletop_deformable_control_step.py", "tabletop_deformable_privileged_ppo.py",
             "tabletop_privileged_ppo.py", "tabletop_deformable_rollout_admission.py")
    source = project/"src/go2_pvcnn/control/dh116"
    return dict(version=VERSION, scope="read_only_actual_admission_audit",
        styles={style:actual_admission(project,style) for style in ("pinch","tripod","power")},
        source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names},
        policy_instances_created=0, optimizer_instances_created=0, optimizer_updates=0,
        new_native_steps=0, teacher_admission=False, native_qualification=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    # Existing evidence remains intact even on a repeated CLI call.
    with args.output.open("x",encoding="utf-8") as stream:
        payload=audit(args.project)
        json.dump(payload,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(args.output.resolve()),admitted_styles=[],
                         optimizer_updates=0,new_native_steps=0)))
    return 0  # Successful audit; payload explicitly says all actual entries denied.


if __name__=="__main__":
    raise SystemExit(main())
