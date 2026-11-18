#!/usr/bin/env python3
"""Independent byte-exact diagnostic codec scope; existing native owner intact."""
from pathlib import Path
import json,os
from scripts import verify_dh116_live_r0_reset as previous
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_deformable_r0_replay as codec
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope,VERSION as CODEC_VERSION
ENTRY="scripts/verify_dh116_fast_r0_reset.py"
WRAPPER="scripts/run_dh116_fast_r0_work_packages.py"
VERSION=previous.VERSION
WORKER_ENV="DH116_FAST_R0_RESET_WORKER_V1"
PARENT_PACKAGE=previous.PARENT_PACKAGE
PARENT_SCOPE=previous.PARENT_SCOPE
PARENT_RECEIPT=previous.PARENT_RECEIPT
PARENT_SHA=previous.PARENT_SHA
PROFILE_SHA=previous.PROFILE_SHA
SCOPE="actual_coupled_reset_live_R0_byte_exact_codec_234_steps_only"
FILES=tuple(dict.fromkeys((*previous.FILES,ENTRY,WRAPPER,
    "src/go2_pvcnn/control/dh116/tabletop_fast_owned_codec.py",
    "tests/test_fast_r0_native_worker.py","tests/test_fast_r0_native_session.py")))
old_preflight=previous.preflight
old_execute=previous.execute

def preflight(*args,**kwargs):
    result=old_preflight(*args,**kwargs)
    result["codec_version"]=CODEC_VERSION
    return result

def execute(args,*rest,**kwargs):
    before=codec.encode_owned
    with codec_scope():
        result=old_execute(args,*rest,**kwargs)
    if codec.encode_owned is not before:raise RuntimeError("owned codec binding not restored")
    marker=dict(version=CODEC_VERSION,binding_restored=True,new_physics_steps=0,ppo_updates=0)
    with (Path(args.output)/"codec_scope.json").open("x") as f:
        json.dump(marker,f,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return result

def main():
    with bindings((previous,"ENTRY",ENTRY),(previous,"WRAPPER",WRAPPER),
        (previous,"WORKER_ENV",WORKER_ENV),(previous,"FILES",FILES),
        (previous,"preflight",preflight),(previous,"execute",execute)):
        return previous.main()
if __name__=="__main__":main()
