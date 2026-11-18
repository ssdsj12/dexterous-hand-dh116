"""CPU speedup never substitutes for byte equality or native qualification."""
from copy import deepcopy
import pytest
from scripts.compare_dh116_owned_codec import compare_results

def sample():
    return dict(status="complete",counts=dict(reset=1,begin=29,physical=232,end=29),
        encoded_bytes=123,encoded_stream_sha256="a"*64,elapsed_profiled_cpu_s=4.,
        blocked_terms={"instability":232},new_physics_steps=0,ppo_updates=0,teacher_admission=False,
        native_qualification=False,source_receipt_sha256="b"*64,source_budget_passed=False)

def test_equivalent_comparison_keeps_native_failure():
    a=sample();b=deepcopy(a);b["elapsed_profiled_cpu_s"]=2.
    r=compare_results(a,b)
    assert r["profiled_cpu_ratio"]==2
    assert r["byte_equivalent"] and not r["native_qualification"]
    assert not r["source_native_budget_passed"]

@pytest.mark.parametrize("key,value",[("encoded_stream_sha256","c"*64),("encoded_bytes",122),
    ("counts",dict(reset=1,begin=29,physical=231,end=29)),("blocked_terms",{}),
    ("source_receipt_sha256","d"*64)])
def test_mismatched_real_evidence_cannot_claim_optimization(key,value):
    a=sample();b=deepcopy(a);b[key]=value
    with pytest.raises(ValueError):compare_results(a,b)

def test_digest_required_not_just_equal_counts():
    a=sample();a.pop("encoded_stream_sha256")
    with pytest.raises(ValueError):compare_results(a,a)

@pytest.mark.parametrize("key,value",[("status","partial"),("new_physics_steps",1),
    ("ppo_updates",1),("teacher_admission",True),("native_qualification",True)])
def test_no_promotion_or_hidden_native_work(key,value):
    a=sample();b=deepcopy(a);b[key]=value
    with pytest.raises(ValueError):compare_results(a,b)
