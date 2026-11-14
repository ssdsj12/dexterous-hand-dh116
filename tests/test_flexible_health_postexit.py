"""Pure CPU post-exit guard checks; no synthetic result grants native admission."""
import importlib
from copy import deepcopy
import pytest
from test_r0_numerical_health_source import health_case,row_for
from test_flexible_parameter_readback import case
from test_flexible_health_native import api

def audit():return importlib.import_module("scripts.validate_dh116_health_projected_pinch")

@pytest.mark.parametrize("code",[0,5])
def test_recompute_health_reads_saved_native_getter_not_a_boolean(health_case,monkeypatch,code):
    c=health_case;row=row_for(c)
    monkeypatch.setattr(type(c.rope),"last_error",property(lambda self:code))
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    monkeypatch.setattr(director,"director_observation_failure",lambda row:None)
    row["health_getter"]=api().capture_boundary(c.runtime,backend=c.backend,sim=c.sim,exchange=row)
    row["health_prior_guard_failure"]=None
    row["guard_failure"]="numerical_health_failure" if code else None
    result=audit().validate_exchange(row)
    assert result["observation"]["numerical_health"]["value"] is (code==0)
    assert result["capture"]["last_error"]==code
    assert result["native_qualification"] is False

def test_saved_final_terminal_cannot_be_erased(health_case,monkeypatch):
    c=health_case;row=row_for(c)
    monkeypatch.setattr(type(c.rope),"last_error",property(lambda self:9))
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    monkeypatch.setattr(director,"director_observation_failure",lambda row:None)
    row["health_getter"]=api().capture_boundary(c.runtime,backend=c.backend,sim=c.sim,exchange=row)
    row["health_prior_guard_failure"]=None
    with pytest.raises(ValueError,match="terminal"):audit().validate_exchange(row)

def test_parent_failure_cannot_be_overwritten_by_complete_health(monkeypatch,tmp_path):
    m=audit()
    monkeypatch.setattr(m,"inspect_health",lambda *a,**k:dict(source_complete=True,rows=232))
    result=m.audit_outcome(tmp_path,dict(lifecycle_qualified=False,reason="CPU outer failure"))
    assert not result["lifecycle_qualified"] and not result["same_clock_health_getter_qualified"]
    assert result["teacher_admission"] is False
    assert result["mesh_reset_qualified"] is result["substep_contact_frame_qualified"] is False

def test_missing_raw_health_cannot_pass_a_parent_success(monkeypatch,tmp_path):
    m=audit()
    def bad(*a,**k):raise ValueError("missing raw health")
    monkeypatch.setattr(m,"inspect_health",bad)
    result=m.audit_outcome(tmp_path,dict(lifecycle_qualified=True))
    assert result["lifecycle_qualified"] is result["same_clock_health_getter_qualified"] is False
    assert "missing raw health" in result["health_audit_error"]


def test_actual_completed_234_ledger_replays_unknown_getter_without_filling_zero():
    import hashlib,json
    from pathlib import Path
    receipt=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
        "flex.f8-original-projected-pinch-native/d2b175b7654b4b30c67028a20a6b0e1a50eda98b3804414554c6d8c9fd3f7d58/"
        "attempt-zy2fao0h/receipt.json")
    data=receipt.read_bytes()
    assert hashlib.sha256(data).hexdigest()=="04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3"
    prior=json.loads(data);root=Path(prior["output"])
    for name in ("physical.jsonl","director_reset.json"):
        digest=hashlib.sha256()
        with (root/name).open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
        assert digest.hexdigest()==prior["artifact_sha256"][name]
    assert prior["artifact_sha256"]["physical.jsonl"]=="9cec6ed43c06f23cf23b327a22009c7f1b3a98574de27883ba89c60b0bf255d4"
    result=audit().replay_health_r0(root)
    assert result["status"]=="completed_offline" and result["control_windows"]==29
    assert all(r["terms"]["instability"]["raw_sum"] is None for r in result["controls"])
    assert result["training_reward"] is None and result["native_qualification"] is False
