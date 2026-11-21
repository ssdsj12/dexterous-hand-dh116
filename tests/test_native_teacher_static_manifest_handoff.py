"""CPU source replay/handoff; budget I/O is synthetic and grants no lease."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_static_contract_reuses_this_call_verified_manifest_and_rechecks_next_call(monkeypatch,tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as pre
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_session as session
    from go2_pvcnn.control.dh116.tabletop_tripod_reset_manifest import TripodResetManifest
    saved=Path("/mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ref902/tripod-ppo-smoke8-v8-input/training.json")
    training=json.loads(saved.read_bytes())
    value=training["preflight"]
    binding=value["resolved_reset_manifest"]
    original=Path(binding["path"]).read_bytes()
    assert hashlib.sha256(original).hexdigest()==binding["sha256"]
    # Mutate only this new test-owned copy for the stale-source negative case.
    temporary=tmp_path/"manifest.json";temporary.write_bytes(original)
    value["resolved_reset_manifest"]={"path":str(temporary),"sha256":binding["sha256"]}
    limits=dict(max_sdk_steps=66,max_solver_calls=64,max_resets=1,
                max_episode_count=1,wall_timeout_s=180.)
    monkeypatch.setattr(pre,"validate_budget",lambda *a,**k:dict(limits))
    monkeypatch.setattr(session,"validate_budget",lambda *a,**k:dict(limits))
    real_load=pre.load_reset_manifest;loaded=[]
    def read(*a,**k):
        result=real_load(*a,**k);loaded.append(result);return result
    monkeypatch.setattr(pre,"load_reset_manifest",read)
    if hasattr(session,"load_reset_manifest"):
        monkeypatch.setattr(session,"load_reset_manifest",read)
    config=session.NativeSessionConfig(value,training["model_config"],tmp_path/"unused.jsonl",
        training["episode_cap_s"],training["rollout_steps"],ROOT)
    contract=session.verify_static_contract(config)
    assert type(contract.reset_manifest) is TripodResetManifest
    assert contract.reset_manifest is loaded[0]
    assert len(loaded)==1, "one source-verified immutable result must flow through this static transaction"
    assert contract.reset_manifest.data()["teacher_admission"] is False
    assert contract.config.rollout_steps==8 and contract.reset_seed==contract.reset_manifest.data()["reset_seed"]
    # Forwarding this transaction's verified object must not cache a later pass.
    temporary.write_bytes(original+b"\n")
    with pytest.raises(ValueError,match="static contract denied"):
        session.verify_static_contract(config)
    assert len(loaded)==1
    assert not list(tmp_path.glob("*.pt")) and not (tmp_path/"unused.jsonl").exists()
