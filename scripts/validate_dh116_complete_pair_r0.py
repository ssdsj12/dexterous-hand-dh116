"""Post-exit retained raw contact/live-R0 reconciliation; no additional physics."""
from collections import Counter
from pathlib import Path
import hashlib,json,math
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,_digest,_measurement
from go2_pvcnn.control.dh116.tabletop_live_effective_pad_source import LiveEffectivePadSource,VERSION
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope

def events(path):
    with Path(path).open("rb") as f:
        for line in f:
            if not line.endswith(b"\n"):raise ValueError("incomplete JSONL")
            yield decode_owned(json.loads(line))

def audit_rows(physical,r0,project):
    source=None;reset_step=None;reset_time=None;physical_count=0;known=Counter();candidates=Counter();pairs=Counter()
    for event in physical:
        if event["operation"]=="reset":
            if source is not None:raise ValueError("exact one actual Reset")
            reset_clock=event["payload"]["reset"]["after_clock"]
            source=LiveEffectivePadSource(project,reset_clock=reset_clock)
            reset_step=reset_clock["physics_step"];reset_time=reset_clock["timestamp_s"]
        if event["operation"]!="physical":continue
        if source is None:raise ValueError("actual completed Reset before contact required")
        saved=next(r0,None)
        if saved is None:raise ValueError("missing same-clock R0 contact record")
        payload=saved["payload"];exchange=event["payload"]["exchange"]
        observed=source.capture(exchange)
        c=observed["identity"];expected=physical_count+1
        if (c["physics_step"]!=reset_step+expected or
            not math.isclose(c["timestamp_s"]-reset_time,expected/480,rel_tol=0,abs_tol=1e-8)):
            raise ValueError("strict complete physical step and timestamp ordering required")
        row=payload["r0_row"]
        if (payload["identity"]!=observed["identity"] or
            _digest(row["pair_diagnostics"]["complete_effective_pad_source"])!=_digest(observed)):
            raise ValueError("same-clock independently recomputed contact evidence differs")
        for side,hand in observed["hands"].items():
            valid_source=None if hand["value"] is None else VERSION+":"+observed["source_sha256"]+":"+side
            if row["measurements"][side]!=_measurement(hand["value"],valid_source):
                raise ValueError("R0 effective contact measurement differs")
            known[side]+=hand["known_no_effective_contact"]
            candidates[side]+=hand["region_candidate_count"];pairs[side]+=hand["raw_pair_count"]
        if row["r0"]["training_reward"] is not None:
            raise ValueError("complete negative evidence cannot grant full R0 training")
        physical_count+=1
    if source is None or physical_count!=232 or next(r0,None) is not None:
        raise ValueError("complete 232 aligned contact/R0 physical records required")
    return dict(physical_rows=physical_count,known_negative_rows=dict(known),
        region_candidates=dict(candidates),raw_hand_pairs=dict(pairs),
        raw_capacity_overflow_retained=True,atomic_aggregate_reconciled=True,
        source_recomputed_from_actual_reset=True,positive_effective_contact_qualified=False,
        teacher_admission=False,training_reward=None,full_R0_ready=False,new_physics_steps=0)

def audit(output):
    output=Path(output)
    rows=(r for r in events(output/"live_r0.jsonl") if r["operation"]=="physical")
    physics=events(output/"physical.jsonl")
    if next(physics,None)!=dict(version="dh116-dual-physical-evidence-v1",physics_budget_end=234,solver_substeps=12):
        raise ValueError("exact original 234-step physical ledger header required")
    with codec_scope():
        result=audit_rows(physics,rows,Path(__file__).resolve().parents[1])
    result["artifact_sha256"]={n:hashlib.sha256((output/n).read_bytes()).hexdigest()
        for n in ("physical.jsonl","live_r0.jsonl")}
    return result
