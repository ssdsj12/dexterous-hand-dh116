"""Post-exit independent origin/provenance reconciliation, no new physics."""
from copy import deepcopy
from pathlib import Path
from scripts.validate_dh116_complete_pair_r0 import events,audit_rows
from scripts.profile_dh116_complete_pair_r0_ledger import sha
from go2_pvcnn.control.dh116.tabletop_r0_measurement_origin import MeasurementOrigin,restore_measurements
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope

def audit(output):
    output=Path(output);adapter=None;origin_ids=set();count=0
    def restored():
        nonlocal adapter,count
        live=(r for r in events(output/"live_r0.jsonl") if r["operation"]=="physical")
        for event in events(output/"physical.jsonl"):
            op=event.get("operation")
            if op=="reset":
                if adapter is not None:raise ValueError("one actual reset")
                adapter=MeasurementOrigin(event["payload"]["reset"]["after_clock"])
            if op!="physical":continue
            if adapter is None:raise ValueError("reset before original sample")
            r=next(live,None)
            if r is None:raise ValueError("missing same-clock R0")
            r=deepcopy(r);row=r["payload"]["r0_row"]
            p=row["pair_diagnostics"].pop("measurement_origin_provenance")
            m=restore_measurements(row["measurements"],p)
            adapted,recomputed=adapter.adapt(event["payload"]["exchange"],m,row["pair_diagnostics"])
            if _digest(recomputed)!=_digest(p) or _digest(adapted)!=_digest(row["measurements"]):
                raise ValueError("origin/sample evidence differs from independently recomputed provenance")
            origin_ids.update(s["origin"] for s in p["samples"].values() if s["origin"] is not None)
            row["measurements"]=m;count+=1
            yield r
        if next(live,None) is not None:raise ValueError("extra R0 physical rows")
    physics=events(output/"physical.jsonl")
    if next(physics,None)!=dict(version="dh116-dual-physical-evidence-v1",physics_budget_end=234,solver_substeps=12):
        raise ValueError("exact original 234-step physical budget required")
    with codec_scope():
        result=audit_rows(physics,restored(),Path(__file__).resolve().parents[1])
    result.update(per_frame_provenance_recomputed=True,origin_count=len(origin_ids),
        artifact_sha256={n:sha(output/n) for n in ("physical.jsonl","live_r0.jsonl")})
    if count!=232:raise ValueError("232 complete matched samples required")
    return result
