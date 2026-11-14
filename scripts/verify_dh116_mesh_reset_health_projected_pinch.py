#!/usr/bin/env python3
"""New once-only mesh Reset/pose/health diagnostic; no reuse of consumed recipes."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib,json
from pathlib import Path
from scripts import verify_dh116_projected_pinch as previous
from scripts import verify_dh116_health_projected_pinch as historical
from scripts import prepare_dh116_mesh_reset_health_projected_pinch as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source

ENTRY="scripts/verify_dh116_mesh_reset_health_projected_pinch.py"
WRAPPER="scripts/run_dh116_mesh_reset_health_work_packages.py"
VERSION=operation.VERSION
WORKER_ENV="DH116_MESH_RESET_HEALTH_PROJECTED_PINCH_WORKER_V1"
PROFILE_SHA=previous.PROFILE_SHA
PARENT_PACKAGE,PARENT_SCOPE=previous.PARENT_PACKAGE,previous.PARENT_SCOPE
PARENT_RECEIPT,PARENT_SHA=previous.PARENT_RECEIPT,previous.PARENT_SHA
PROJECTED_RECEIPT,PROJECTED_SHA=historical.PROJECTED_RECEIPT,historical.PROJECTED_SHA
FILES=tuple(dict.fromkeys((*historical.FILES,ENTRY,WRAPPER,
    "scripts/prepare_dh116_mesh_reset_health_projected_pinch.py",
    "scripts/prepare_dh116_deformx_mesh_observations.py",
    "src/go2_pvcnn/control/dh116/tabletop_mesh_observation_source.py",
    "src/go2_pvcnn/control/dh116/tabletop_mesh_reset_order.py",
    "src/go2_pvcnn/control/dh116/tabletop_mesh_reset_readback.py",
    "src/go2_pvcnn/control/dh116/tabletop_mesh_observation_evidence.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformx_pair_observations_v2.py",
    "src/go2_pvcnn/control/dh116/tabletop_complete_pair_capture.py",
    "tests/test_mesh_reset_health_worker.py","tests/test_mesh_reset_health_owner.py",
    "tests/test_mesh_observation_profile.py","tests/test_deformx_pair_observations_v2.py")))
sha,read,load_parent=previous.sha,previous.read,previous.load_parent
assert_clean_import_boundary=previous.assert_clean_import_boundary
_PREFLIGHT,_EXECUTE,_MAIN=previous.preflight,previous.execute,previous.main
_REQUIRE=previous.require_worker

@contextmanager
def entry_scope():
    with bindings((previous,"ENTRY",ENTRY),(previous,"WRAPPER",WRAPPER),
        (previous,"VERSION",VERSION),(previous,"WORKER_ENV",WORKER_ENV),
        (previous,"FILES",FILES),(previous,"operation",operation)):
        yield

def preflight(project,parent_path=PARENT_RECEIPT,parent_sha=PARENT_SHA):
    project=Path(project)
    # Validate original parent evidence separately; a new scope does not qualify it
    # as a fixed Reset or import its consumed native budget.
    with source.source_scope() as basis,entry_scope():
        contract=_PREFLIGHT(project,parent_path,parent_sha)
    if sha(PROJECTED_RECEIPT)!=PROJECTED_SHA:
        raise ValueError("exact consumed projected source receipt required")
    prior=read(PROJECTED_RECEIPT)
    if (prior["outer_exit_code"]!=0 or prior["debugger_inferior_exit_code"]!=0
        or prior["debugger_inferior_signal"] is not None
        or prior["qualification_scope"]!="pinch_original_safety_projection_2_setup_232_diagnostic_only"):
        raise ValueError("historical bounded lifecycle scope differs")
    sources=deepcopy(contract["sources"])
    for name in FILES:sources["local_sources"][name]=sha(project/name)
    for name,expected in basis["local_sources"].items():
        if sha(project/name)!=expected:raise ValueError("observation profile local source differs")
        sources["local_sources"][name]=expected
    sources["external_sources"][str(PROJECTED_RECEIPT)]=PROJECTED_SHA
    sources["external_sources"].update({k:v for k,v in basis["sources"].items()
        if not k.startswith(str(source.ROOT)+"/")})
    for name in ("physical.jsonl","result.json","sdk_clock.jsonl"):
        path=Path(prior["output"])/name;expected=prior["artifact_sha256"][name]
        if path.is_symlink() or sha(path)!=expected:raise ValueError("historical source changed: "+name)
        sources["external_sources"][str(path)]=expected
    code=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return dict(contract,version=VERSION,sources=sources,code_config_sha256=code,
        qualification_scope=operation.SCOPE,mesh_observation_source=basis,
        projected_source_receipt_sha256=PROJECTED_SHA,native_execution_allowed=True,
        native_execution_performed=False,reset_mesh_qualified=False,
        substep_contact_frame_qualified=False,teacher_admission=False,ppo_updates=0)

def require_worker(args,project):
    with entry_scope():return _REQUIRE(args,project)

def execute(args,project,contract):
    with source.source_scope(),entry_scope():
        return _EXECUTE(args,project,contract)

def main():
    with entry_scope(),bindings((previous,"preflight",preflight),(previous,"execute",execute),
                               (previous,"require_worker",require_worker)):
        return _MAIN()

if __name__=="__main__":main()
