"""Failed native records may be profiled, never promoted by a CPU replay."""
from pathlib import Path
import json
import pytest
from scripts.profile_dh116_live_r0_observer import lock_receipt,compare_phases

def test_refuse_missing_or_wrong_receipt_digest(tmp_path):
    p=tmp_path/"receipt.json";p.write_text("{}")
    with pytest.raises(ValueError,match="receipt SHA"):lock_receipt(p,"0"*64)

def test_no_training_or_cpu_fixture_admission(tmp_path):
    from scripts.profile_dh116_live_r0_observer import sha
    p=tmp_path/"receipt.json";p.write_text(json.dumps(dict(native_execution_performed=False)))
    with pytest.raises(ValueError,match="actual native"):lock_receipt(p,sha(p))

def test_phase_duration_measures_interval_not_root_cause():
    old=[dict(phase="acquire",time_unix_s=1.),dict(phase="dual_reset_complete",time_unix_s=3.),
         dict(phase="owned hand and solver references dropped",time_unix_s=10.),
         dict(phase="before_app_close",time_unix_s=11.),dict(phase="after_app_close",time_unix_s=13.)]
    new=[dict(r,time_unix_s=r["time_unix_s"]*2.) for r in old]
    r=compare_phases(old,new)
    assert r["reset_to_owned_release"]["prior_s"]==7.
    assert r["reset_to_owned_release"]["current_s"]==14.
    assert r["reset_to_owned_release"]["delta_s"]==7.
    assert r["causal_attribution_proven"] is False

def test_repeated_or_missing_phase_is_not_silently_accepted():
    with pytest.raises(ValueError):compare_phases([],[])

def test_failed_native_uses_available_artifact_hashes(tmp_path,monkeypatch):
    from scripts import profile_dh116_live_r0_observer as m
    native=tmp_path/"native";native.mkdir()
    names=("physical.jsonl","director_reset.json","live_r0.jsonl","live_r0_reset.json","ownership.jsonl")
    hashes={}
    for name in names:
        p=native/name;p.write_text("fixture\n");hashes[name]=m.sha(p)
    a=tmp_path/"live-r0-post-exit-audit.json"
    a.write_text(json.dumps(dict(evidence=dict(available_artifact_sha256=hashes))))
    r=dict(native_execution_performed=True,native_qualification=False,
      qualification_scope="actual_coupled_reset_then_live_original_R0_observer_234_steps_only",
      outer_exit_code=0,debugger_inferior_signal=None,lifetime_post_exit_audit_sha256=m.sha(a),
      output=str(native),lifetime_validator_source_sha256={})
    p=tmp_path/"receipt.json";p.write_text(json.dumps(r))
    assert m.lock_receipt(p,m.sha(p))["native_qualification"] is False
    (native/"physical.jsonl").write_text("changed")
    with pytest.raises(ValueError,match="source bytes"):m.lock_receipt(p,m.sha(p))
