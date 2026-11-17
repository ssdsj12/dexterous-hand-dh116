#!/usr/bin/env python3
"""CPU-only dual-hand USD/SDF preparation; no native execution option."""
import argparse
import json
from pathlib import Path
import sys

from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import (
    prepare_scene, author_source_stage, inspect_source_stage,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    prepared = prepare_scene(project)
    stage = author_source_stage(project,prepared)
    result = dict(preparation=prepared,usd_audit=inspect_source_stage(stage,prepared),
        new_sdk_steps=0,new_solver_steps=0,optimizer_updates=0,
        native_execution_performed=False,native_qualified=False,teacher_admitted=False,
        imported_native_simulation_modules=[name for name in ('isaaclab.sim','omni.kit.app',
            'omni.physx','ropesim','warp') if name in sys.modules],
        loaded_usd_schema_only='pxr.PhysxSchema' in sys.modules)
    if result['imported_native_simulation_modules']:
        raise RuntimeError('CPU preparation imported a native simulation module')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output),fingerprint_sha256=prepared['fingerprint_sha256'],
        **result['usd_audit']),sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
