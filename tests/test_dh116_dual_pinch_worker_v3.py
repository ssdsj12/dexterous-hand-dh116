"""Cover the original SDK path when the launcher supplies prepared=None."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from test_dh116_pinch_v3 import BLOCKER, cpu
from scripts import verify_dh116_dual_pinch_interval_v3 as entry

from test_dh116_dual_pinch_builder import prepared, geometry

ROOT = Path(__file__).resolve().parents[1]


def test_byte_only_successor_provenance():
    result = cpu(BLOCKER + """
from pathlib import Path
from scripts import verify_dh116_dual_pinch_interval_v3 as entry
r=entry.preflight(Path.cwd())
assert r['v2_failure_receipt_sha256']==entry.FAILED_SHA
assert 'isaaclab' not in sys.modules and 'ropesim' not in sys.modules
print('DUAL_V3_PREFLIGHT')
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "DUAL_V3_PREFLIGHT"


def test_native_default_resolves_same_owned_preparation_once(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene

    marker = dict(fingerprint_sha256="f"*64)
    calls = []
    def prepare(project):
        calls.append(project)
        return marker
    monkeypatch.setattr(scene,"prepare_scene",prepare)
    forwarded = []
    def original(app, **kwargs):
        forwarded.append(kwargs["prepared"])
    monkeypatch.setattr(entry,"original_interval",original)
    entry.run_existing_app(SimpleNamespace(), project=ROOT, prepared=None)
    entry.run_existing_app(SimpleNamespace(), project=ROOT, prepared=marker)
    assert calls == [ROOT]
    assert forwarded[0] is forwarded[1] is marker


def test_integrated_original_owner_with_none_from_launcher(monkeypatch,tmp_path,prepared,geometry):
    from test_dh116_dual_pinch_entry import test_actual_builder_journal_eight_ticks_and_lifecycle_failure_boundary as actual
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from scripts import prepare_dh116_dual_pinch_interval as interval

    # This fixture calls the unchanged builder/reset/control/journal path. Only
    # the launcher argument is changed to None, as in the actual v2 worker.
    real = interval.run_existing_app
    called = []
    def provide(project):
        called.append(project)
        return prepared
    monkeypatch.setattr(scene,"prepare_scene",provide)
    monkeypatch.setattr(entry,"original_interval",real)
    def native_default(*args,**kwargs):
        kwargs["prepared"] = None
        return entry.run_existing_app(*args,**kwargs)
    monkeypatch.setattr(interval,"run_existing_app",native_default)
    actual(monkeypatch,tmp_path,prepared,geometry,None)
    assert called == [ROOT]
