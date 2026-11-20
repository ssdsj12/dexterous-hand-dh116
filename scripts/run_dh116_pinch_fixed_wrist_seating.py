#!/usr/bin/env python3
"""Fixed-wrist Pinch with independent thumb seating and left index relief."""
from pathlib import Path

import numpy as np

from scripts import run_dh116_pinch_asymmetric_preload as prior
from scripts import prepare_dh116_pinch_wrist_first as wrist

preload, closure, base, prelift, pacing = prior.preload, prior.closure, prior.base, prior.prelift, prior.pacing
PREFIX_SDK = prior.LAST_SDK
PREFIX_WINDOWS = (PREFIX_SDK-2)//8
RAMP_WINDOWS = 2
SETTLE_WINDOWS = 8
LAST_SDK = PREFIX_SDK + 8*(RAMP_WINDOWS+SETTLE_WINDOWS)
WALL_S = 520
SEATING_DOF = (0, 2)
SEATING_DELTA_RAD = (-.003, -.0005)
OUTER_SHA = "7142d573032b6b766fb8eaac40fc84704bfffcb38a4388db25479f78e8513b16"
LEDGER_SHA = "7bde45de8bb66ddfc5a5d01c98c299be110f57afbbfbbdf2e77e255cb96811e1"
GATE_KEY = "fixed_wrist_seating"


def compile_seating(previous, states, nominal_targets=None):
    prefix = prelift._finite(previous, (PREFIX_WINDOWS,24))
    commands = {side:prelift._finite(states[side][1], (12,)) for side in ("left","right")}
    targets = {side:value.copy() for side,value in commands.items()}
    targets["left"][list(SEATING_DOF)] += SEATING_DELTA_RAD
    if nominal_targets is not None:
        for side,q in targets.items():
            if not np.allclose(q, prelift._finite(nominal_targets[side],(12,)), atol=2e-9, rtol=0):
                raise ValueError("actual targets do not match the archived seating increments")
    lo,hi = prelift.active_limits()
    if any(np.any(q[:6]<lo) or np.any(q[:6]>hi) for q in targets.values()):
        raise ValueError("seating exceeds original DH116 joint limits")
    delta = prelift.global_channel_order(*(targets[s]-commands[s] for s in ("left","right")))
    if np.any(np.delete(delta, SEATING_DOF)) or np.any(delta[list(SEATING_DOF)]>=0.):
        raise ValueError("only the two retained left finger channels may move")
    ramp = delta/(prelift.WINDOW_SCALE*RAMP_WINDOWS)
    if np.any(np.abs(ramp)>1.):
        raise ValueError("seating exceeds original action rate limits")
    return np.vstack((prefix,np.tile(ramp,(RAMP_WINDOWS,1)),np.zeros((SETTLE_WINDOWS,24)))), targets


def seating_domain(actual, issued, target):
    actual,issued,target = [prelift._finite(x,(12,)) for x in (actual,issued,target)]
    controlled = np.r_[list(SEATING_DOF),6:12]
    other = np.delete(np.arange(12),controlled)
    if np.any(issued[other]!=target[other]) or np.any(issued[6:]!=target[6:]):
        raise ValueError("non-seating command and both wrists must be fixed")
    before,after=(issued-actual)[controlled],(target-actual)[controlled]
    span=np.where(abs(before)>=abs(after),before,after)
    denominator=np.where(span==0.,1.,span)
    fractions=np.stack((before/denominator,after/denominator))
    ref=target.copy()
    ref[controlled]=actual[controlled]+span
    return ref,np.stack((np.minimum(0.,fractions.min(axis=0)),np.maximum(1.,fractions.max(axis=0))))


def certify_seating(row, states, targets):
    nodes=prelift._finite(row["rod"]["node_positions"][0],(41,3))
    result={}
    for side in ("left","right"):
        actual,issued=states[side]
        domain,axes=(seating_domain,SEATING_DOF) if side=="left" else (pacing.tracking_domain,(1,2))
        reference,bounds=domain(actual,issued,targets[side])
        phases,_=wrist.right_geometry(row,actual,issued,reference,
            dict(closure=tuple(bounds),wrist_residuals=(np.zeros(3),np.zeros(3))),
            nodes=nodes,phase_names=("closure",),phase_seconds=40.,side=side,
            independent_wrist=True,active_axes=axes)
        result[side]=phases["closure"]
    return result


def prepare(args, *, screen_geometry=False):
    plan=prior.prepare(args)
    archive,row=pacing.archive_boundary(args.seated_prefix,outer_sha=OUTER_SHA,ledger_sha=LEDGER_SHA,
        sdk_steps=PREFIX_SDK,windows=PREFIX_WINDOWS,contact_guard=closure.guard_selected_contact)
    if plan["action_schedule"]!=archive["action_schedule"] or not preload.boundary_has_opposition(row,sdk_step=PREFIX_SDK):
        raise ValueError("matching fixed-wrist signed contact source required")
    states={side:prelift.joint_and_wrist(row,side) for side in ("left","right")}
    actions,targets=compile_seating(plan["action_schedule"],states)
    if screen_geometry:
        proofs=certify_seating(row,states,targets)
        if not all(proof["proved"] for proof in proofs.values()):
            raise ValueError({side:proof.get("unresolved_shape") for side,proof in proofs.items()})
    for path in (args.seated_prefix/"outer.json",args.seated_prefix/"native/physical.jsonl",
                 args.seated_prefix/"native/frozen_plan.json",Path(__file__).resolve()):
        plan["frozen_files"][str(path)]=base.lifecycle.native.sha(path)
    plan[GATE_KEY]=dict(prefix=preload.late.prefix_measurements(row,expected_sdk=PREFIX_SDK),
                       goals={side:q.tolist() for side,q in targets.items()})
    plan.update(staged_version="dh116-pinch-fixed-wrist-seating-v1",action_schedule=actions.tolist(),
                preserved_scope="left thumb and index redistribution with wrists fixed, no lift or policy update")
    plan["budget"].update(control_steps=len(actions),sdk_steps=LAST_SDK,soft_wall_seconds=WALL_S,
                          solver_steps=(LAST_SDK-2)*base.SUBSTEPS)
    return plan


def acquire(sim, owners, trace, output, *, plan):
    def observe(row):
        if row["physics_step"]==preload.LAST_SDK:
            preload.measure_gate(row,plan,sdk_step=preload.LAST_SDK,prefix_windows=prior.PREFIX_WINDOWS,
                config_key=prior.GATE_KEY,gate_key="asymmetric_preload_gate",compiler=prior.compile_asymmetric)
        elif row["physics_step"]==PREFIX_SDK:
            preload.measure_gate(row,plan,sdk_step=PREFIX_SDK,prefix_windows=PREFIX_WINDOWS,
                config_key=GATE_KEY,gate_key="fixed_wrist_seating_gate",compiler=compile_seating,
                certify=certify_seating)
    try:
        return preload.acquire(sim,owners,trace,output,plan=plan,sdk_steps=LAST_SDK,after=observe)
    finally:
        sim=owners=trace=None


if __name__=="__main__":
    raise SystemExit(closure.main(
        extra_flags=("touched","left-preload","right-preload","preloaded","left-trim","right-trim","seated-prefix"),
        prepare_operation=prepare,acquire_operation=acquire,
    ))
