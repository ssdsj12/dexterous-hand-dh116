"""Recompute same-instance quiet receipts from full owned source rows."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned, decode_owned
from go2_pvcnn.control.dh116.tabletop_tripod_quiet_gate import TripodQuietGate
from scripts import prepare_dh116_tripod_native_interval as operation

def inspect_quiet(path, *, complete=True):
    gate=None;rows=0;terminal=None;receipt=None
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):raise ValueError("incomplete quiet ledger line")
            entry=json.loads(line)
            if entry.get("operation") not in ("reset","physical"):continue
            payload=decode_owned(entry["payload"])
            if entry["operation"]=="reset":
                if gate is not None:raise ValueError("one actual reset for Tripod quiet gate")
                nodes=np.asarray(payload["reset"]["rod_reset"]["node_positions"])
                if nodes.shape!=(1,41,3):raise ValueError("actual41node reset required")
                gate=TripodQuietGate(reset_nodes_m=nodes[0])
                continue
            if gate is None or terminal is not None:raise ValueError("physical row before reset or after terminal")
            x=payload["exchange"];rows+=1
            if rows>232 or x["physics_step"]!=rows+2:raise ValueError("exact bounded consecutive quiet ledger")
            prior=x["quiet_prior_guard_failure"];error=None
            if prior is not None:
                rebuilt={"status":"not_evaluated","reason":"original_guard"}
            else:
                try:
                    if rows<=8:
                        rebuilt=gate.observe_exchange(dict(x,guard_failure=prior))
                    else:rebuilt=gate.require_ready()
                except Exception as exc:
                    rebuilt=None;error=type(exc).__name__+": "+str(exc)
            if encode_owned(rebuilt)!=encode_owned(x["tripod_quiet"]) or error!=x["quiet_capture_error"]:
                raise ValueError("saved quiet receipt differs from complete source recomputation")
            terminal=operation.observation_failure(x)
            if terminal!=x["guard_failure"]:raise ValueError("quiet terminal differs from saved execution")
            receipt=rebuilt
    if gate is None:raise ValueError("actual reset missing")
    if complete and (rows!=232 or terminal is not None or receipt is None or receipt["status"]!="passed"):
        raise ValueError("complete Tripod sequence and quiet pass required")
    return dict(measured_rows=rows,terminal_reason=terminal,receipt=deepcopy(receipt),
                complete=rows==232 and terminal is None,native_qualification=False,teacher_admission=False)
