import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_dh116_pinch_v3 import BLOCKER, cpu
from scripts import verify_dh116_dual_pinch_interval as entry
from scripts import run_dh116_dual_pinch_work_packages as wrapper

ROOT = Path(__file__).resolve().parents[1]


def test_real_byte_preflight_has_no_sdk_or_solver_import():
    result = cpu(BLOCKER + """
from pathlib import Path
from scripts import verify_dh116_dual_pinch_interval as entry
contract=entry.preflight(Path.cwd())
assert 'ropesim' not in sys.modules and 'isaaclab' not in sys.modules
assert contract['paired_reset_seed_sha256']==entry.builder.SEED_SHA
assert contract['teacher_admission'] is False
print('BYTE_ONLY_DUAL')
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "BYTE_ONLY_DUAL"


def test_worker_rejects_scope_before_app_or_output(monkeypatch, tmp_path):
    output = tmp_path / "refused"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("DH116_PHYSX_LIFETIME_GDB", "1")
    monkeypatch.delenv(entry.WORKER_ENV, raising=False)
    with pytest.raises(ValueError, match="worker"):
        entry.require_worker(SimpleNamespace(output=output), ROOT)
    assert not output.exists()


def test_exact_worker_marker(monkeypatch, tmp_path):
    output = tmp_path / "owned"
    expected = dict(version=entry.VERSION, case="explicit", output=str(output),
        entry_sha256=entry.prior.sha(ROOT/entry.ENTRY),
        wrapper_sha256=entry.prior.sha(ROOT/entry.WRAPPER))
    monkeypatch.setenv(entry.WORKER_ENV, json.dumps(expected))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("DH116_PHYSX_LIFETIME_GDB", "1")
    entry.require_worker(SimpleNamespace(output=output), ROOT)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError, match="GPU"):
        entry.require_worker(SimpleNamespace(output=output), ROOT)


@pytest.mark.parametrize("owner", ["controller", "filter", "contact", "camera_history"])
def test_reset_owner_absence_has_explicit_scope(owner):
    callback = entry.diagnostic_reset_owners()[owner]
    value = callback(dict(episode_uid="round-test"))
    assert value["status"] == "not_present"
    assert value["owner"] == owner and value["episode_uid"] == "round-test"
    assert value["source"]


def test_wrapper_preserves_exact_zero_update_budget(monkeypatch):
    monkeypatch.setattr(wrapper, "original_case", lambda spec: "explicit")
    sha = "1" * 64
    spec = dict(budget=dict(environments=1, updates=0, physics_steps=10,
        gpu_index=3, timeout_seconds=180), argv=["--approved-code-config-sha256", sha],
        validator=dict(equals=dict(version=entry.VERSION, code_config_sha256=sha)))
    assert wrapper.cleanup_case(spec) == "explicit"
    spec["budget"]["physics_steps"] = 18
    with pytest.raises(ValueError, match="10"):
        wrapper.cleanup_case(spec)
