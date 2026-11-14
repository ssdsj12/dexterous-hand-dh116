"""Verify the distinct wire metadata successor is bound to the retained failure."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from test_dh116_pinch_v3 import BLOCKER, cpu
from scripts import verify_dh116_dual_pinch_interval_v2 as entry


ROOT = Path(__file__).resolve().parents[1]


def test_v2_preflight_keeps_pre_kit_boundary():
    result = cpu(BLOCKER + """
from pathlib import Path
from scripts import verify_dh116_dual_pinch_interval_v2 as entry
c=entry.preflight(Path.cwd())
assert 'ropesim' not in sys.modules and 'isaaclab' not in sys.modules
assert c['v1_failure_receipt_sha256']==entry.FAILED_SHA
assert c['physics_steps']==10 and c['teacher_admission'] is False
print('V2_PROVENANCE')
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "V2_PROVENANCE"


def test_v2_rejects_v1_failure_change_before_acquisition(monkeypatch, tmp_path):
    wrong = tmp_path/"receipt.json"
    wrong.write_text('{}')
    monkeypatch.setattr(entry,"FAILED_RECEIPT",wrong)
    with pytest.raises(ValueError,match="failure receipt"):
        entry.preflight(ROOT)


def test_original_launcher_is_called_with_scoped_layout(monkeypatch,tmp_path):
    calls=[]
    def fake(args,project,contract):
        calls.append((args.output,project,contract["code_config_sha256"]))
        raise ValueError("controlled SDK boundary")
    monkeypatch.setattr(entry.prior,"execute",fake)
    with pytest.raises(ValueError,match="controlled"):
        entry.execute(SimpleNamespace(output=tmp_path), ROOT, {"code_config_sha256":"b"*64})
    assert calls == [(tmp_path,ROOT,"b"*64)]
