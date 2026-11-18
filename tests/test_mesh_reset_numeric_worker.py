"""Check the new bounded worker, preserving the consumed recipe and original guard."""
from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from scripts import verify_dh116_mesh_reset_numeric_pinch as entry
from scripts import run_dh116_mesh_reset_numeric_pinch_work_packages as wrapper
from scripts import run_dh116_work_packages as scheduler
from test_dh116_pinch_v3 import cpu
from test_dh116_dual_pinch_closure_worker import BLOCKER

ROOT = Path(__file__).resolve().parents[1]
RECIPE = "configs/dh116_execution/deformx_mesh_reset_numeric_pinch_v2.json"


def test_new_recipe_freezes_numeric_evidence_and_preserves_old_fingerprint():
    repo = scheduler.Repository(ROOT)
    contract = entry.preflight(ROOT)
    assert contract["version"] == entry.VERSION and contract["qualification_scope"] == entry.SCOPE
    assert contract["failed_reset_receipt_sha256"] == entry.FAILED_SHA
    dependencies = dict(dependencies=[entry.PARENT_PACKAGE],
                        dependency_scopes={entry.PARENT_PACKAGE: dict(scope=entry.PARENT_SCOPE, native=True)})
    old = repo.native_recipe(dict(command_spec="configs/dh116_execution/deformx_mesh_reset_health_projected_pinch_v1.json", **dependencies))
    fresh = repo.native_recipe(dict(command_spec=RECIPE, **dependencies))
    assert old["code_config_sha256"] == "4918ab0a9bb51fc20c0bbcc0bd070fafc7c846dac9a4579a170b1af30cd34601"
    assert fresh["code_config_sha256"] != old["code_config_sha256"]
    assert fresh["spec"]["budget"] == old["spec"]["budget"]
    assert entry.NUMERIC_ARTIFACT in fresh["spec"]["validator"]["artifacts"]
    assert fresh["spec"]["validator"]["equals"]["code_config_sha256"] == contract["code_config_sha256"]
    assert contract["sources"]["external_sources"].items() <= fresh["spec"]["inputs"].items()
    assert entry.prior.VERSION != entry.VERSION


@pytest.mark.parametrize("fail", [False, True])
def test_numeric_scope_wraps_actual_reset_and_persists_owned_result(monkeypatch, tmp_path, fail):
    calls = []
    class Reset:
        def run(self, sim, rod, bindings, *, owner_resets):
            assert calls == [("entered", sim, rod)]
            if fail:
                raise ValueError("original reset rejected")
            return {"status": "complete"}

    @contextmanager
    def scope(*, sim, rod, record):
        calls.append(("entered", sim, rod))
        try:
            yield
        finally:
            record({"closed": True})

    monkeypatch.setattr(entry, "_RESET_TYPE", lambda *args: Reset)
    monkeypatch.setattr(entry.numeric, "numeric_reset_state_scope", scope)
    reset_type = entry.make_reset_type(ROOT, {}, tmp_path)
    reset = reset_type()
    sim, rod = object(), object()
    if fail:
        with pytest.raises(ValueError, match="original reset rejected"):
            reset.run(sim, rod, {}, owner_resets=[])
    else:
        assert reset.run(sim, rod, {}, owner_resets=[]) == {"status": "complete"}
    assert json.loads((tmp_path / entry.NUMERIC_ARTIFACT).read_text()) == {"closed": True}


def test_parent_success_requires_independent_numeric_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper, "_PREVIOUS_AUDIT", lambda *a: {"lifecycle_qualified": True})
    with pytest.raises(ValueError, match="regular mesh evidence"):
        wrapper.audit_outcome(tmp_path, tmp_path / "outer.json", "explicit")
    state = {"source": "cpu fixture"}
    (tmp_path / entry.NUMERIC_ARTIFACT).write_text(json.dumps(state))
    (tmp_path / entry.operation.evidence.RESET_FILE).write_text(json.dumps(
        {"readback": {"completed_reset_receipt": {"status": "complete"}, "mesh": 11}}))
    def validate(packet, *, reset_receipt):
        assert packet == state
        assert reset_receipt == {"status": "complete", "mesh_state_readback": {"mesh": 11}}
        return {"cpu_fixture": True}
    monkeypatch.setattr(entry.numeric, "validate_numeric_reset_audit", validate)
    assert wrapper.audit_outcome(tmp_path, tmp_path / "outer.json", "explicit")["numeric_state_audit"] == {"cpu_fixture": True}


def test_parent_failure_preserves_failure_and_does_not_require_numeric_success(monkeypatch,tmp_path):
    monkeypatch.setattr(wrapper, "_PREVIOUS_AUDIT", lambda *a: {"lifecycle_qualified": False})
    checked = wrapper.audit_outcome(tmp_path, tmp_path / "outer.json", "explicit")
    assert checked["lifecycle_qualified"] is False and checked["numeric_state_audit"] is None


@pytest.mark.parametrize("marker", [False, True])
def test_real_numeric_cli_requires_own_worker_identity(tmp_path, marker):
    done = cpu(BLOCKER + """
import json, os
from pathlib import Path
from scripts import verify_dh116_mesh_reset_numeric_pinch as entry
p=Path(OUTPUT); c=entry.preflight(Path.cwd()); called=[]
entry.execute=lambda *a: called.append(True)
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["DH116_PHYSX_LIFETIME_GDB"]="1"
os.environ.pop(entry.WORKER_ENV,None)
if MARKER:
    os.environ[entry.WORKER_ENV]=json.dumps(dict(version=entry.VERSION,case="explicit",output=str(p.resolve()),
        entry_sha256=entry.prior.sha(Path.cwd()/entry.ENTRY),wrapper_sha256=entry.prior.sha(Path.cwd()/entry.WRAPPER)))
sys.argv=[entry.ENTRY,"--cleanup","explicit","--output",str(p),"--headless","--execute-native","--steps","234",
    "--approved-code-config-sha256",c["code_config_sha256"],"--left-scheduler-receipt",str(entry.PARENT_RECEIPT),
    "--left-receipt-sha256",entry.PARENT_SHA]
try: entry.main()
except (RuntimeError,ValueError,SystemExit):
    assert not MARKER and not called and not p.exists()
else:
    assert MARKER and called==[True]
assert not forbidden.intersection({x.split(".")[0] for x in sys.modules})
print("NUMERIC_CLI_CPU")
""".replace("OUTPUT", repr(str(tmp_path / "native"))).replace("if MARKER:", "if " + repr(marker) + ":").replace("not MARKER", "not " + repr(marker)).replace("assert MARKER", "assert " + repr(marker)))
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == "NUMERIC_CLI_CPU"


def test_scheduler_reentry_and_all_bindings_use_new_wrapper(monkeypatch):
    old = wrapper.prior.base.__file__
    def base_main():
        base = wrapper.prior.base
        assert Path(base.__file__).resolve() == (ROOT / entry.WRAPPER).resolve()
        assert (base.ENTRY, base.WRAPPER, base.WORKER_ENV) == (entry.ENTRY, entry.WRAPPER, entry.WORKER_ENV)
        assert base.audit_outcome is wrapper.audit_outcome
        return 0
    monkeypatch.setattr(wrapper.prior.base, "main", base_main)
    assert wrapper.main() == 0
    assert wrapper.prior.base.__file__ == old
