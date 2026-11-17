"""Independent post-exit R0 clock/coverage audit, on top of existing physics."""
import json
from pathlib import Path
from scripts import prepare_dh116_live_r0_native as operation
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,_digest
from go2_pvcnn.control.dh116.tabletop_r0_live_observer import LiveR0Observer

def read(path):
    return decode_owned(json.loads(Path(path).read_bytes()))

def audit(output):
    output=Path(output)
    summary=read(output/"live_r0_summary.json")
    if (summary["status"]!="complete" or summary["physical_rows"]!=232
        or summary["control_windows"]!=29 or summary["training_reward"] is not None
        or summary["new_physics_steps_by_observer"]!=0 or summary["ppo_updates"]!=0
        or summary["teacher_admission"] is not False):
        raise ValueError("live observer incomplete or scope exceeded")
    reset=read(output/"live_r0_reset.json")
    if (reset["live_clock_before"]!=reset["live_clock_after"]
        or reset["live_clock_before"]["sdk_physics_step"]!=2
        or reset["temporary_native_references_released"] is not True
        or reset["initialization"]["initializer"]["reset_callbacks_executed"]!=0
        or reset["legacy_six_hook_order_claimed"] is not False):
        raise ValueError("same live post-reset boundary/ownership required")
    def rows(name):
        data=(output/name).read_bytes()
        if not data.endswith(b"\n"):raise ValueError("incomplete JSONL")
        return [decode_owned(json.loads(line)) for line in data.splitlines()]
    physics=rows("physical.jsonl")[1:]
    actual_reset=next(e["payload"] for e in physics if e["operation"]=="reset")
    if _digest(actual_reset["reset"])!=reset["initialization"]["physical_transaction_sha256"]:
        raise ValueError("R0 Reset must bind this actual physical transaction")
    native=rows("live_r0.jsonl")
    if native[0]["operation"]!="reset" or native[0]["payload"]!=reset:
        raise ValueError("one R0 initialization before all actual decisions required")
    r0=[r for r in native[1:]]
    expected=[r for r in physics if r["operation"] in ("begin","physical","end")]
    if len(r0)!=len(expected):raise ValueError("actual physical/R0 event counts differ")
    physical_count=controls=0
    for new,old in zip(r0,expected):
        if new["operation"]!=old["operation"]:raise ValueError("physical/R0 ordering differs")
        a,b=new["payload"],old["payload"]
        if new["operation"]=="begin":
            if a!={"pre_action_sha256":_digest(b["pre_action_state"]),"action_sha256":_digest(b["action"])}:
                raise ValueError("R0 actual pre-state/action binding differs")
        elif new["operation"]=="physical":
            identity=b["exchange"]["action_readback"]["post"]["identity"][0]
            if a["identity"]!=identity or a["actual_event_sha256"]!=_digest(b):
                raise ValueError("R0 physical identity/source differs")
            if a["r0_row"]["r0"]["training_reward"] is not None:raise ValueError("diagnostic cannot become training reward")
            physical_count+=1
        else:
            if _digest(a["control_result"])!=_digest(b):
                raise ValueError("R0 control settlement differs from actual result")
            if a["control_receipt"]["physical_samples"]!=8:raise ValueError("incomplete R0 control")
            controls+=1
    return dict(version=operation.VERSION,qualification_scope=operation.SCOPE,
        live_reset_initialization_verified=True,original_six_hook_order_qualified=False,
        physical_rows=physical_count,control_windows=controls,teacher_admission=False,
        training_reward=None,full_R0_ready=False,new_physics_steps=0)
