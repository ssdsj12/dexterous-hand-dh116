#!/usr/bin/env python3
"""New exact worker identity; unchanged projected actions and bounded parent chain."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
from scripts import verify_dh116_projected_pinch as previous
from scripts import prepare_dh116_health_projected_pinch as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY="scripts/verify_dh116_health_projected_pinch.py"
WRAPPER="scripts/run_dh116_health_projected_pinch_work_packages.py"
VERSION=operation.VERSION
WORKER_ENV="DH116_HEALTH_PROJECTED_PINCH_WORKER_V1"
NATIVE_BLOCKER="unresolved_deferred_mesh_reset_teleport_overwrite"

PROFILE_SHA=previous.PROFILE_SHA
PARENT_PACKAGE,PARENT_SCOPE=previous.PARENT_PACKAGE,previous.PARENT_SCOPE
PARENT_RECEIPT,PARENT_SHA=previous.PARENT_RECEIPT,previous.PARENT_SHA
PROJECTED_RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f8-original-projected-pinch-native/d2b175b7654b4b30c67028a20a6b0e1a50eda98b3804414554c6d8c9fd3f7d58/"
    "attempt-zy2fao0h/receipt.json")
PROJECTED_SHA="04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3"
FILES=tuple(dict.fromkeys((*previous.FILES,ENTRY,WRAPPER,
    "scripts/prepare_dh116_health_projected_pinch.py","scripts/validate_dh116_health_projected_pinch.py",
    "scripts/replay_dh116_native_parameter_r0.py",
    "src/go2_pvcnn/control/dh116/tabletop_health_getter_evidence.py",
    "src/go2_pvcnn/control/dh116/tabletop_r0_health_separation.py",
    "src/go2_pvcnn/control/dh116/tabletop_r0_hold_necessary_domain.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_r0_applicability.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_control_step.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_task.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_task_budget.py",
    "configs/dh116_tasks/flexible_r0_health_separated_v2.json",
    "configs/dh116_tasks/flexible_r0_hold_necessary_domain_v1.json",
    "configs/dh116_tasks/flexible_r0_applicability_diagnostic_v2.json",
    "configs/dh116_training/deformable_task_budget_examples_v2.json",
    "tests/test_flexible_health_native.py","tests/test_flexible_health_worker.py",
    "tests/test_flexible_health_postexit.py","tests/test_flexible_health_owner.py",
    "tests/test_r0_health_separation.py","tests/test_r0_numerical_health_source.py",
    "tests/test_flexible_parameter_readback.py")))
sha,read,load_parent=previous.sha,previous.read,previous.load_parent
assert_clean_import_boundary=previous.assert_clean_import_boundary
_EXECUTE=previous.execute
_PREFLIGHT=previous.preflight
_MAIN=previous.main

@contextmanager
def entry_scope():
    with bindings((previous,"ENTRY",ENTRY),(previous,"WRAPPER",WRAPPER),(previous,"VERSION",VERSION),
        (previous,"WORKER_ENV",WORKER_ENV),(previous,"FILES",FILES),(previous,"operation",operation)):
        yield

def preflight(project,parent_path=PARENT_RECEIPT,parent_sha=PARENT_SHA):
    with entry_scope():contract=_PREFLIGHT(project,parent_path,parent_sha)
    if sha(PROJECTED_RECEIPT)!=PROJECTED_SHA:raise ValueError("exact completed projected parent required")
    prior=read(PROJECTED_RECEIPT)
    if (prior["native_qualification"] is not True or prior["outer_exit_code"]!=0
        or prior["debugger_inferior_exit_code"]!=0 or prior["debugger_inferior_signal"] is not None
        or prior["qualification_scope"]!="pinch_original_safety_projection_2_setup_232_diagnostic_only"):
        raise ValueError("completed projected source scope differs")
    sources=deepcopy(contract["sources"])
    sources["external_sources"][str(PROJECTED_RECEIPT)]=PROJECTED_SHA
    for name in ("physical.jsonl","result.json","sdk_clock.jsonl"):
        path=Path(prior["output"])/name;expected=prior["artifact_sha256"][name]
        if path.is_symlink() or sha(path)!=expected:raise ValueError("projected source bytes differ")
        sources["external_sources"][str(path)]=expected
    code=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return dict(contract,sources=sources,code_config_sha256=code,projected_source_receipt_sha256=PROJECTED_SHA,
        live_r0_computed=False,health_getter_rows=232,qualification_scope=operation.SCOPE,
        native_execution_allowed=False,blocked_reasons=[NATIVE_BLOCKER],mesh_reset_qualified=False)

def require_worker(args,project):
    with entry_scope():return previous.require_worker(args,project)

def execute(args,project,contract):
    # A separate successor must resolve/reset-source-pin the known timing defect.
    raise RuntimeError(NATIVE_BLOCKER)

def main():
    if "--execute-native" in sys.argv[1:]:raise RuntimeError(NATIVE_BLOCKER)
    with entry_scope(),bindings((previous,"preflight",preflight),(previous,"execute",execute)):
        return _MAIN()

if __name__=="__main__":main()
