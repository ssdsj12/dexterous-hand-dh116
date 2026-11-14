"""Recompute every live R0 row from owned raw evidence after natural exit."""
from pathlib import Path
from functools import partial
from collections import Counter
from scripts.validate_dh116_complete_pair_r0 import events
from scripts.profile_dh116_complete_pair_r0_ledger import sha
from scripts import replay_dh116_native_parameter_r0 as parameters
from scripts import replay_dh116_measurement_origin as previous
from scripts.prepare_dh116_live_measured_approach import MeasuredApproachObserver,VERSION
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope

def equal(a,b,label):
    if _digest(a)!=_digest(b):raise ValueError(label+" differs from full raw recomputation")

def audit(output,project=None):
    output=Path(output)
    project=Path(project or Path(__file__).resolve().parents[1]).resolve()
    if project!=previous.ROOT.resolve():raise ValueError("auditor source project differs")
    files={n:sha(output/n) for n in ("physical.jsonl","live_r0.jsonl","director_reset.json")}
    reset=next(r["payload"] for r in events(output/"physical.jsonl") if r.get("operation")=="reset")
    packet=next(events(output/"director_reset.json"))["payload"]
    radius=parameters.check_radius_packet(packet,reset["reset"]["rod_reset"])
    live=events(output/"live_r0.jsonl");first=next(live,None)
    if first is None or first["operation"]!="reset":raise ValueError("live actual reset missing")
    counts=Counter();blocked=Counter()
    with codec_scope():
        obj=previous.observer(partial(MeasuredApproachObserver,project=project,parameter_packet=packet),reset,radius)
        equal(obj.initialization,first["payload"]["initialization"],"actual reset baseline")
        for row in events(output/"physical.jsonl"):
            op,p=row.get("operation"),row.get("payload")
            if op not in ("begin","physical","end"):continue
            saved=next(live,None)
            if saved is None or saved["operation"]!=op:raise ValueError("live sequence missing or reordered")
            if op=="begin":
                obj.begin(**p)
                equal(saved["payload"],dict(pre_action_sha256=_digest(p["pre_action_state"]),action_sha256=_digest(p["action"])),"actual action")
            elif op=="physical":
                obj.record(p);expected=obj._rows[-1]
                equal(saved["payload"],dict(identity=p["exchange"]["action_readback"]["post"]["identity"][0],
                    actual_event_sha256=_digest(p),r0_row=expected),"same-clock physical R0")
                counts["physical_rows"]+=1
                counts["valid_approach_samples"]+=expected["r0"]["terms"]["approach_progress"]["valid"]
            else:
                result=obj.end(p);equal(saved["payload"],result,"complete control R0")
                counts["control_windows"]+=1
                blocked.update(result["control_receipt"]["blocked_terms"])
        if next(live,None) is not None:raise ValueError("extra live R0 row")
    if counts["physical_rows"]!=232 or counts["control_windows"]!=29:raise ValueError("complete original bounded cohort required")
    if files!={n:sha(output/n) for n in files}:raise ValueError("raw evidence changed")
    return dict(version=VERSION,**counts,blocked_control_terms=dict(blocked),all_live_rows_recomputed=True,
        reset_baseline_available=obj.reset_distance["reset_baseline_available"],
        artifact_sha256=files,new_physics_steps=0,ppo_updates=0,
        training_reward=None,teacher_admission=False,full_R0_ready=False)
