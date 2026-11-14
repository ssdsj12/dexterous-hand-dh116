"""Synthetic CPU audit protocol; no native contact qualification."""
from copy import deepcopy
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _measurement
from scripts import validate_dh116_complete_pair_r0 as audit
import pytest

class FakeSource:
    def __init__(self,project,*,reset_clock):self.reset=reset_clock
    def capture(self,x):
        return dict(identity=x["clock"],source_sha256="a"*64,
            hands={s:dict(value=None,known_no_effective_contact=False,region_candidate_count=1,raw_pair_count=1)
                   for s in ("left","right")})

def rows():
    reset=dict(physics_step=2,timestamp_s=2/480)
    physical=[dict(operation="reset",payload=dict(reset=dict(after_clock=reset)))]
    r0=[]
    src=FakeSource(None,reset_clock=reset)
    for step in range(3,235):
        x=dict(clock=dict(physics_step=step,timestamp_s=step/480))
        physical.append(dict(operation="physical",payload=dict(exchange=x)))
        r0.append(dict(operation="physical",payload=dict(identity=x["clock"],r0_row=dict(
            pair_diagnostics=dict(complete_effective_pad_source=src.capture(x)),
            measurements={s:_measurement(None,None) for s in ("left","right")},
            r0=dict(training_reward=None)))))
    return physical,r0

def test_all_232_records_recomputed_without_promoting_unknown(monkeypatch):
    monkeypatch.setattr(audit,"LiveEffectivePadSource",FakeSource)
    p,r=rows();out=audit.audit_rows(iter(p),iter(r),None)
    assert out["physical_rows"]==232 and out["known_negative_rows"]==dict(left=0,right=0)
    assert out["region_candidates"]==dict(left=232,right=232)
    assert not out["teacher_admission"] and out["training_reward"] is None

@pytest.mark.parametrize("mutation",["missing","extra","identity","source","contact","reward","duplicate_clock","reordered_clock"])
def test_incomplete_or_forged_scope_rejected(monkeypatch,mutation):
    monkeypatch.setattr(audit,"LiveEffectivePadSource",FakeSource)
    p,r=rows()
    if mutation=="missing":r.pop()
    elif mutation=="extra":r.append(deepcopy(r[-1]))
    elif mutation=="identity":r[4]["payload"]["identity"]={}
    elif mutation=="source":r[4]["payload"]["r0_row"]["pair_diagnostics"]["complete_effective_pad_source"]["source_sha256"]="b"*64
    elif mutation=="contact":r[4]["payload"]["r0_row"]["measurements"]["right"]=_measurement({},None)
    elif mutation=="reward":r[4]["payload"]["r0_row"]["r0"]["training_reward"]=0
    elif mutation=="duplicate_clock":p[5]=deepcopy(p[4]);r[4]=deepcopy(r[3])
    else:p[5],p[6]=p[6],p[5];r[4],r[5]=r[5],r[4]
    with pytest.raises(ValueError):audit.audit_rows(iter(p),iter(r),None)

@pytest.mark.parametrize("bad_header",[False,True])
def test_actual_owned_journal_header_is_validated_before_rows(tmp_path,monkeypatch,bad_header):
    import json
    monkeypatch.setattr(audit,"LiveEffectivePadSource",FakeSource)
    p,r=rows()
    header=dict(version="dh116-dual-physical-evidence-v1",physics_budget_end=234,solver_substeps=12)
    if bad_header:header["physics_budget_end"]=235
    (tmp_path/"physical.jsonl").write_text("".join(json.dumps(row)+"\n" for row in [header,*p]))
    (tmp_path/"live_r0.jsonl").write_text("".join(json.dumps(row)+"\n" for row in r))
    if bad_header:
        with pytest.raises(ValueError):audit.audit(tmp_path)
    else:assert audit.audit(tmp_path)["physical_rows"]==232
