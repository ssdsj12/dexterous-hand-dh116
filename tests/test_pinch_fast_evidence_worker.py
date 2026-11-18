"""Check the new bounded worker, preserving the consumed recipe and original guard."""
import json
from pathlib import Path

import pytest
from types import SimpleNamespace
from go2_pvcnn.control.dh116 import tabletop_deformable_dual_evidence as journal

from scripts import verify_dh116_pinch_fast_evidence as entry
from scripts import run_dh116_pinch_fast_evidence_work_packages as wrapper
from scripts import run_dh116_work_packages as scheduler
from test_dh116_pinch_v3 import cpu
from test_dh116_dual_pinch_closure_worker import BLOCKER

ROOT = Path(__file__).resolve().parents[1]
RECIPE = "configs/dh116_execution/deformx_pinch_fast_evidence_v3.json"


def test_recipe_freezes_exact_fast_encoding_and_preserves_consumed_identity():
    repo = scheduler.Repository(ROOT)
    contract = entry.preflight(ROOT)
    assert contract["version"] == entry.VERSION and contract["qualification_scope"] == entry.SCOPE
    assert contract["timed_out_source_receipt_sha256"] == entry.TIMEOUT_SHA
    dependencies = dict(dependencies=[entry.PARENT_PACKAGE],
                        dependency_scopes={entry.PARENT_PACKAGE: dict(scope=entry.PARENT_SCOPE, native=True)})
    old = repo.native_recipe(dict(command_spec="configs/dh116_execution/deformx_mesh_reset_numeric_pinch_v2.json", **dependencies))
    fresh = repo.native_recipe(dict(command_spec=RECIPE, **dependencies))
    assert old["code_config_sha256"] == "c48203afab2a7cb17ca2aaa672a4096ea380c20f85d1146d414a117930d84247"
    assert fresh["code_config_sha256"] != old["code_config_sha256"]
    assert fresh["spec"]["budget"] == old["spec"]["budget"]
    assert entry.NUMERIC_ARTIFACT in fresh["spec"]["validator"]["artifacts"]
    assert fresh["spec"]["validator"]["equals"]["code_config_sha256"] == contract["code_config_sha256"]
    assert contract["sources"]["external_sources"].items() <= fresh["spec"]["inputs"].items()
    assert entry.prior.VERSION != entry.VERSION


@pytest.mark.parametrize("marker", [False, True])
def test_real_fast_cli_requires_own_worker_identity(tmp_path, marker):
    done = cpu(BLOCKER + """
import json, os
from pathlib import Path
from scripts import verify_dh116_pinch_fast_evidence as entry
p=Path(OUTPUT); c=entry.preflight(Path.cwd()); called=[]
entry.execute=lambda *a: called.append(True)
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["DH116_PHYSX_LIFETIME_GDB"]="1"
os.environ.pop(entry.WORKER_ENV,None)
if MARKER:
    os.environ[entry.WORKER_ENV]=json.dumps(dict(version=entry.VERSION,case="explicit",output=str(p.resolve()),
        entry_sha256=entry.prior.prior.sha(Path.cwd()/entry.ENTRY),wrapper_sha256=entry.prior.prior.sha(Path.cwd()/entry.WRAPPER)))
sys.argv=[entry.ENTRY,"--cleanup","explicit","--output",str(p),"--headless","--execute-native","--steps","234",
    "--approved-code-config-sha256",c["code_config_sha256"],"--left-scheduler-receipt",str(entry.PARENT_RECEIPT),
    "--left-receipt-sha256",entry.PARENT_SHA]
try: entry.main()
except (RuntimeError,ValueError,SystemExit):
    assert not MARKER and not called and not p.exists()
else:
    assert MARKER and called==[True]
assert not forbidden.intersection({x.split(".")[0] for x in sys.modules})
print("FAST_EVIDENCE_CLI_CPU")
""".replace("OUTPUT", repr(str(tmp_path / "native"))).replace("if MARKER:", "if " + repr(marker) + ":").replace("not MARKER", "not " + repr(marker)).replace("assert MARKER", "assert " + repr(marker)))
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == "FAST_EVIDENCE_CLI_CPU"


def test_scheduler_reentry_and_all_bindings_use_new_wrapper(monkeypatch):
    old = wrapper.prior.prior.base.__file__
    def base_main():
        base = wrapper.prior.prior.base
        assert Path(base.__file__).resolve() == (ROOT / entry.WRAPPER).resolve()
        assert (base.ENTRY, base.WRAPPER, base.WORKER_ENV) == (entry.ENTRY, entry.WRAPPER, entry.WORKER_ENV)
        assert base.audit_outcome is wrapper.audit_outcome
        return 0
    monkeypatch.setattr(wrapper.prior.prior.base, "main", base_main)
    assert wrapper.main() == 0
    assert wrapper.prior.prior.base.__file__ == old

@pytest.mark.parametrize("throws", [False, True])
def test_actual_scope_routes_only_the_owned_encoder(monkeypatch, throws):
    original = journal.encode_owned
    def execute(args, project, contract):
        assert journal.encode_owned is entry.codec.encode_owned
        assert entry.prior.ENTRY == entry.ENTRY
        if throws:
            raise RuntimeError("owner error")
        return "done"
    monkeypatch.setattr(entry, "_EXECUTE", execute)
    if throws:
        with pytest.raises(RuntimeError, match="owner error"):
            entry.execute(SimpleNamespace(), ROOT, {})
    else:
        assert entry.execute(SimpleNamespace(), ROOT, {}) == "done"
    assert journal.encode_owned is original
    assert entry.prior.ENTRY != entry.ENTRY


def test_failure_and_original_audit_are_retained(monkeypatch, tmp_path):
    inherited = {"lifecycle_qualified": False, "numeric_state_audit": None, "reason": "elapsed"}
    monkeypatch.setattr(wrapper, "_PREVIOUS_AUDIT", lambda *a: inherited)
    result = wrapper.audit_outcome(tmp_path, tmp_path / "outer.json", "explicit")
    assert result == {**inherited, "qualification_scope": entry.SCOPE}
