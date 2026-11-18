#!/usr/bin/env python3
"""Read-only accounting of one frozen, budget-failed native ledger."""
from collections import Counter
from pathlib import Path
import argparse,hashlib,json,os
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-complete-pair-r0-native/ccd965a560702192c790b68dd4881ae215cc286cd6383403f54e9016739a961f/attempt-oau20x9r/receipt.json")
RECEIPT_SHA="208b6c6d16f7f7e23c8e72a721343bc74276d242fcc4ef0dd71b6d16182df508"

# Independently pinned after natural exit; failed receipt has no artifact manifest.
LEDGER_SHA="6581685565bdea48ef66f2b35fd670a1c8790856ab889da737ac17d1630f389d"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def size(value):return len(json.dumps(value,separators=(",",":"),allow_nan=False).encode())
def lines(path):
    with Path(path).open("rb") as f:
        for line in f:
            if not line.endswith(b"\n"):raise ValueError("incomplete JSONL")
            yield json.loads(line)

def profile(events):
    n=0;previous=None;valid=Counter();blocked=Counter();first=Counter();overlap=Counter()
    pairs=Counter();candidates=Counter();known=Counter();byterm=Counter()
    occurrences=stats_bytes=r0_bytes=0;unique=set();last=None
    for event in events:
        if event.get("operation")!="physical":continue
        p=event["payload"];step=p["identity"]["physics_step"]
        if type(step) is not int or (previous is not None and step!=previous+1):
            raise ValueError("continuous original physical order required")
        previous=step;n+=1;row=p["r0_row"];last=row["r0"]
        if last["training_reward"] is not None:raise ValueError("diagnostic scope only")
        evidence=row["pair_diagnostics"]["complete_effective_pad_source"]
        for name,term in last["terms"].items():
            (valid if term["valid"] else blocked)[name]+=1
        for name,term in last["term_stats"].items():
            ids=term["source_identities"]
            if any(not isinstance(s,str) for s in ids):raise ValueError("explicit source identities")
            occurrences+=len(ids);unique.update(ids);byterm[name]+=size(ids)
        r0_bytes+=size(last);stats_bytes+=size(last["term_stats"])
        for side,hand in evidence["hands"].items():
            pairs[side]+=hand["raw_pair_count"];candidates[side]+=hand["region_candidate_count"]
            known[side]+=hand["known_no_effective_contact"]
        for p in evidence["pairs"]:
            region=p["region"]
            if region["first_rejection"] is not None:first[region["first_rejection"]]+=1
            overlap.update(region["overlap_rejections"])
    if last is None:raise ValueError("no actual physical rows")
    return dict(physical_rows=n,valid_terms=dict(valid),blocked_terms=dict(blocked),
        first_rejections=dict(first),overlap_rejections=dict(overlap),raw_hand_pairs=dict(pairs),
        region_candidates=dict(candidates),known_negative_rows=dict(known),
        source_identity_occurrences=occurrences,unique_source_identities=len(unique),
        source_identity_compact_bytes_by_term=dict(byterm),
        term_stats_compact_json_bytes=stats_bytes,r0_compact_json_bytes=r0_bytes,
        final_phase=last["phase"],final_hold_s=last["hold_s"],final_success_candidate=last["success_candidate"],
        native_qualification=False,teacher_admission=False,new_physics_steps=0,ppo_updates=0,
        scope="offline_accounting_of_consumed_budget_failed_evidence_only")

def run(output):
    if sha(RECEIPT)!=RECEIPT_SHA:raise ValueError("frozen receipt differs")
    receipt=json.loads(RECEIPT.read_bytes())
    if (receipt["native_execution_performed"] is not True or receipt["native_qualification"] is not False
        or receipt["outer_exit_code"]!=0 or receipt["debugger_inferior_signal"] is not None
        or receipt["duration_seconds"]<=180):
        raise ValueError("exact naturally exited budget-failed source required")
    p=Path(receipt["output"])/"live_r0.jsonl"
    if sha(p)!=LEDGER_SHA:raise ValueError("raw ledger SHA differs")
    out=profile(lines(p))
    if out["physical_rows"]!=232:raise ValueError("complete original232 measurement rows required")
    out.update(source_receipt_sha256=RECEIPT_SHA,source_ledger_sha256=sha(p),
        actual_native_duration_s=receipt["duration_seconds"],ledger_file_bytes=p.stat().st_size,
        implementation_sha256=sha(__file__),
        limitation="Storage duplication is measured; it is not by itself an isolated causal wall-time profile.")
    with Path(output).open("x") as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    print(json.dumps(run(args.output),sort_keys=True))
if __name__=="__main__":main()
