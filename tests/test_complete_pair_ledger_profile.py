"""CPU fixture controls; statistics never grant native or Teacher qualification."""
import json
from pathlib import Path
import pytest
from scripts.profile_dh116_complete_pair_r0_ledger import profile

def fixture():
    terms=dict(contact=dict(valid=True,raw=0,contribution=0,source="sample:1"))
    def row(step,sources):
        return dict(operation="physical",payload=dict(identity=dict(physics_step=step),
            r0_row=dict(r0=dict(terms=terms,blocked_terms=["stable"],training_reward=None,
                hold_s=0,phase="approach",success_candidate=False,
                term_stats=dict(contact=dict(source_identities=sources))),
                pair_diagnostics=dict(complete_effective_pad_source=dict(hands={
                    "left":dict(known_no_effective_contact=True,region_candidate_count=0,raw_pair_count=0),
                    "right":dict(known_no_effective_contact=True,region_candidate_count=0,raw_pair_count=1)},
                    pairs=[dict(mesh_name="right.finger14_link",region=dict(first_rejection="wrong_normal",
                        overlap_rejections=["wrong_normal","outside_inner_ellipse"]))])))))
    return [row(3,["sample:1"]),row(4,["sample:1","sample:2"])]

def test_counts_record_duplicates_but_not_as_success():
    r=profile(fixture())
    assert r["physical_rows"]==2
    assert r["source_identity_occurrences"]==3
    assert r["unique_source_identities"]==2
    assert r["first_rejections"]=={"wrong_normal":2}
    assert r["overlap_rejections"]=={"wrong_normal":2,"outside_inner_ellipse":2}
    assert r["valid_terms"]=={"contact":2}
    assert not r["native_qualification"] and not r["teacher_admission"]
    assert r["ppo_updates"]==0 and r["new_physics_steps"]==0

@pytest.mark.parametrize("mutation",["order","reward","missing"])
def test_bad_ledger_not_silently_counted(mutation):
    r=fixture()
    if mutation=="order":r[1]["payload"]["identity"]["physics_step"]=3
    elif mutation=="reward":r[0]["payload"]["r0_row"]["r0"]["training_reward"]=0.
    else:del r[0]["payload"]["r0_row"]["pair_diagnostics"]["complete_effective_pad_source"]
    with pytest.raises((ValueError,KeyError)):profile(r)

def test_failed_receipt_does_not_fabricate_an_artifact_manifest(tmp_path,monkeypatch):
    from scripts import profile_dh116_complete_pair_r0_ledger as m
    ledger=tmp_path/"live_r0.jsonl";ledger.write_text("changed raw ledger\n")
    receipt=tmp_path/"receipt.json"
    receipt.write_text(json.dumps(dict(native_execution_performed=True,native_qualification=False,
        outer_exit_code=0,debugger_inferior_signal=None,duration_seconds=196,output=str(tmp_path))))
    monkeypatch.setattr(m,"RECEIPT",receipt);monkeypatch.setattr(m,"RECEIPT_SHA",m.sha(receipt))
    monkeypatch.setattr(m,"LEDGER_SHA","0"*64)
    with pytest.raises(ValueError,match="raw ledger SHA"):m.run(tmp_path/"profile.json")
    assert not (tmp_path/"profile.json").exists()
