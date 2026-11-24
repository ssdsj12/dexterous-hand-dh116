"""CPU oracle comparisons against the frozen TrainingReward, never native proof."""
from dataclasses import replace
from pathlib import Path
import math

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_training_reward as scalar
from go2_pvcnn.control.dh116.tabletop_shared_teacher_reward import endpoint as shared_endpoint
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE

ROOT = Path(__file__).resolve().parents[1]
GUARD = "original_guard_joint_force_clearance_geometry"


def api():
    from go2_pvcnn.control.dh116 import tabletop_tensor_training_reward as tensor
    return tensor


def config(grasp="pinch", full=False):
    name = "flexible_train_r1_full" if full else grasp + "_finger_goal_reward_v1"
    return scalar.load_config(ROOT / "configs/dh116_training" / (name + ".json"))


def row(c, lane, tick, *, episode=0, distance=.02, error=.1, failure=False,
        health=True, contact=False, lifted=False, drift=.0001,
        uncertainty=0., prohibited=False, hold=True, anchors_live=None):
    stamp = scalar.SourceStamp(f"ep:{lane}:{episode}", tick, tick / 480,
                              "cpu_fixture", "fixture:endpoint", env_id=lane,
                              reset_uid=f"reset:{lane}:{episode}", rod_uid=f"rod:{lane}")
    def m(raw, unit="dimensionless"):
        return scalar.Measurement(raw, unit, replace(stamp, source="fixture:measure:" + unit))
    fields = dict(distance=m(distance, "m"), numerical_health=m(health),
                  constraint_failure=m(failure), **{GUARD: m(failure)})
    if c.profile == scalar.FULL:
        fields.update(median_z=m(.04 if lifted else 0., "m"),
                      min_z=m(.03 if lifted else 0., "m"),
                      coverage_left=m(float(contact)), coverage_right=m(float(contact)),
                      opposed_left=m(contact), opposed_right=m(contact),
                      bilateral_support=m(contact), off_table=m(lifted),
                      no_external_support=m(not prohibited), prohibited_support=m(prohibited),
                      hold_motion=m(hold),
                      slip=scalar.Measurement(None, "m", replace(stamp, source="slip:history"),
                          applicable=False, history_reason="no_bilateral_contact_since_reset"))
        anchors = tuple(scalar.SlipAnchor(side, kind, f"anchor:{lane}:{episode}:{side}:{kind}",
                          (0., 0., 0.), 0, m(drift, "m"), m(uncertainty, "m"))
                        for side in ("left", "right") for kind in ("material", "tangential")) if (contact if anchors_live is None else anchors_live) else ()
        return scalar.RewardEndpoint(stamp, ("material:0", "material:1"), fields, anchors,
            (("left", ("thumb", "index")), ("right", ("thumb", "index"))))
    fields.update((name, m(error + lane * .001, "rad")) for name in scalar.ERROR_FIELDS)
    return scalar.RewardEndpoint(stamp, tuple(str(i) for i in range(41)), fields)


def batch(rows):
    """Fixture-only packing; the production API has no endpoint conversion."""
    t = api()
    lanes = tuple(t.TensorLaneSchema(
        episode_uid=r.stamp.episode_uid, env_id=r.stamp.env_id, reset_uid=r.stamp.reset_uid,
        rod_uid=r.stamp.rod_uid, scope=r.stamp.scope, source=r.stamp.source,
        target_ids=r.target_ids, required_pad_ids=r.required_pad_ids,
        field_specs=tuple((name,m.unit,m.stamp.source) for name,m in r.fields.items()),
        anchor_specs=tuple((a.reference_uid,a.drift.stamp.source,a.uncertainty.stamp.source) for a in r.anchors)) for r in rows)
    values = {name: torch.tensor([r.fields[name].raw for r in rows], dtype=(
        torch.bool if type(rows[0].fields[name].raw) is bool else torch.float64))
        for name in rows[0].fields if rows[0].fields[name].raw is not None}
    kwargs = {}
    if "median_z" in values:
        present = torch.tensor([bool(r.anchors) for r in rows])
        kwargs = dict(anchor_present=present,
            anchor_drift=torch.tensor([[a.drift.raw for a in r.anchors] if r.anchors else [math.nan]*4 for r in rows], dtype=torch.float64),
            anchor_uncertainty=torch.tensor([[a.uncertainty.raw for a in r.anchors] if r.anchors else [math.nan]*4 for r in rows], dtype=torch.float64),
            anchor_reference=torch.tensor([[a.reference for a in r.anchors] if r.anchors else [[math.nan]*3]*4 for r in rows], dtype=torch.float64),
            anchor_reference_tick=torch.tensor([[a.reference_tick for a in r.anchors] if r.anchors else [-1]*4 for r in rows], dtype=torch.int64))
    return lanes, t.TensorRewardFrame(torch.tensor([r.stamp.physics_tick for r in rows]),
        torch.tensor([r.stamp.solver_time_s for r in rows], dtype=torch.float64), values, **kwargs)


def req(endpoint, token, values):
    return scalar.SubmittedRequest(token, "policy:0", endpoint.stamp, tuple(values),
                                  "normalized_submitted_command24_preprojection")


def start(c, n=3, **kwargs):
    rows = [row(c, i, 0, **kwargs) for i in range(n)]
    engines = [scalar.TrainingReward(c) for _ in rows]
    for i, eng in enumerate(engines):
        eng.reset_from_verified_state(rows[i], previous_request=req(rows[i], "reset", [0.]*24))
    lanes, frame = batch(rows)
    eng = api().TensorTrainingReward(c, num_envs=n, device="cpu")
    eng.reset_selected(tuple(range(n)), lanes, frame,
        previous_request=torch.zeros(n,24,dtype=torch.float64),
        tokens=("reset",)*n, policy_uids=("policy:0",)*n)
    return eng, engines, rows


def run_window(c, eng, engines, previous, start_tick, token, kwargs_at, *, truncated=None, buffer_full=None):
    n=len(engines)
    action=torch.tensor([[(-1)**i * .3]*24 for i in range(n)],dtype=torch.float64)
    # Full schema acquisition belongs to the owner, once per window.
    schema,_=batch([row(c,i,start_tick+1,**kwargs_at(i,1)) for i in range(n)])
    eng.begin(schema, request=action, tokens=(token,)*n, policy_uids=("policy:0",)*n,
              physics_tick=torch.tensor([r.stamp.physics_tick for r in previous]),
              solver_time_s=torch.tensor([r.stamp.solver_time_s for r in previous],dtype=torch.float64),
              remaining_substeps=8)
    for i,e in enumerate(engines):
        e.begin(req(previous[i],token,action[i].tolist()),remaining_substeps=8)
    last=list(previous)
    for step in range(1,9):
        rows=[row(c,i,start_tick+step,**kwargs_at(i,step)) for i in range(n)]
        _,frame=batch(rows)
        eng.observe(frame)
        for i,e in enumerate(engines):
            if not e._terminal:
                e.observe(rows[i]);last[i]=rows[i]
    tr=torch.zeros(n,dtype=torch.bool) if truncated is None else truncated
    bf=torch.zeros(n,dtype=torch.bool) if buffer_full is None else buffer_full
    packet=eng.settle_once((token,)*n,truncated=tr,buffer_full=bf)
    expected=[e.settle_once(token,truncated=bool(tr[i]),buffer_full=bool(bf[i])) for i,e in enumerate(engines)]
    for field in ("reward","duration_s","phi_pre","phi_post_raw","endpoint_phi","integrated_cost","event"):
        torch.testing.assert_close(getattr(packet,field),torch.tensor([getattr(p,field) for p in expected],dtype=torch.float64),rtol=1e-13,atol=1e-13)
    for field in ("terminated","truncated","buffer_full","done"):
        assert getattr(packet,field).tolist()==[getattr(p,field) for p in expected]
    for i,p in enumerate(expected):
        assert packet.event_reasons(i)==p.event_reasons
        assert packet.cost_per_second[i,:len(p.cost_per_second)].tolist()==pytest.approx(p.cost_per_second)
        assert packet.post_context["contact_s"][i].item()==pytest.approx(p.post_context["contact_s"])
        assert packet.post_context["hold_s"][i].item()==pytest.approx(p.post_context["hold_s"])
    return packet,last


@pytest.mark.parametrize("grasp",["pinch","tripod","power"])
def test_actual_shared_smoke_finger_dwell_latch_and_timeout_parity(grasp):
    c=config(grasp)
    eng,engines,last=start(c)
    for k in range(4):
        p,last=run_window(c,eng,engines,last,8*k,f"w{k}",
            lambda i,s:dict(distance=.018-s*.0001,error=.01),
            truncated=torch.tensor([k==3]*3))
        assert p.event.tolist()==([2.]*3 if k==1 else [0.]*3)
    assert p.done.all() and not p.terminated.any()
    assert p.bootstrap(torch.ones(3,dtype=torch.float64)).tolist()==[c.gamma]*3


def test_mixed_terminal_absorption_and_selected_reset():
    c=config();eng,engines,last=start(c)
    p,last=run_window(c,eng,engines,last,0,"fail",
        lambda i,s:dict(distance=.01,error=.01,failure=i==1 and s>=3,health=not(i==2 and s>=5)))
    assert p.substeps.tolist()==[8,3,5]
    assert p.event.tolist()==[0.,-40.,-40.]
    assert p.bootstrap(torch.ones(3,dtype=torch.float64)).tolist()==[c.gamma,0.,0.]
    new=[row(c,i,0,episode=1,error=.2) for i in (1,2)]
    lanes,frame=batch(new)
    eng.reset_selected((1,2),lanes,frame,previous_request=torch.zeros(2,24,dtype=torch.float64),
                       tokens=("reset:1",)*2,policy_uids=("policy:0",)*2)
    assert eng.post_context["physics_tick"].tolist()==[8,0,0]
    assert eng.post_context["distance_ref_m"].tolist()==[.02]*3


def test_original_close_lift_success_after_five_seconds_and_absorption():
    c=config(full=True);eng,engines,last=start(c,2,contact=True)
    for k in range(302):
        p,last=run_window(c,eng,engines,last,8*k,f"full{k}",
            lambda i,s:dict(contact=True,lifted=True,drift=.0001,uncertainty=.0001))
        if bool(p.done.any()):
            break
    assert p.event.tolist()==[100.,100.]
    assert p.substeps.tolist()==[1,1]
    assert p.post_context["hold_s"].tolist()==[5.,5.]


def test_close_lift_conservative_uncertain_slip_and_prohibited_event():
    c=config(full=True);eng,engines,last=start(c,3,contact=True)
    p,last=run_window(c,eng,engines,last,0,"close-lift",
        lambda i,s:dict(contact=True,lifted=True,drift=(.0001,.0005,.001)[i],
                        uncertainty=.0001,prohibited=i==2 and s>=4))
    assert p.post_context["slip_status"].tolist()==[1,-1,0]
    assert p.event.tolist()==[0.,0.,-50.]


def test_schema_clock_numeric_guard_and_token_budget_are_not_weakened():
    c=config();eng,engines,last=start(c)
    lanes,frame=batch(last)
    args=dict(request=torch.zeros(3,24,dtype=torch.float64),tokens=("w",)*3,
              policy_uids=("policy:0",)*3,physics_tick=frame.physics_tick,
              solver_time_s=frame.solver_time_s)
    with pytest.raises(ValueError,match="eight|8"):
        eng.begin(lanes,remaining_substeps=7,**args)
    eng.begin(lanes,remaining_substeps=8,**args)
    with pytest.raises(RuntimeError,match="incomplete"):
        eng.settle_once(("w",)*3)
    _,bad=batch([row(c,i,1,distance=math.nan if i==1 else .01) for i in range(3)])
    with pytest.raises(scalar.SourceFailure,match="numerical|finite"):
        eng.observe(bad)
    with pytest.raises(scalar.SourceFailure,match="quarantin"):
        eng.settle_once(("w",)*3)


def test_shared_actual_q_helper_matches_frozen_endpoint_errors():
    c=config("tripod");t=api()
    q=torch.tensor([c.finger_goal.target_rad]*2,dtype=torch.float64)
    q[0,0,0]+=.07;q[1,1,3]+=.11
    errors=t.measured_finger_errors(q,c.finger_goal)
    for i in range(2):
        hands={side:dict(joint_names=ACTIVE,q=q[i,j].numpy()[None,:]) for j,side in enumerate(("left","right"))}
        clock=dict(episode_uid=f"ep:{i}",physics_step=1,solver_time_s=1/480,env_id=i,reset_id=1,rod_uid=f"rod:{i}")
        endpoint=shared_endpoint("tripod",clock,distance_m=.01,health=0,guard=None,
            hands=hands,finger_goal=c.finger_goal,scope="cpu_fixture")
        assert errors[i].tolist()==pytest.approx([endpoint.fields[n].raw for n in scalar.ERROR_FIELDS])


def test_full_na_before_contact_then_latched_anchors_and_contact_loss():
    c=config(full=True);eng,engines,last=start(c,2)
    p,last=run_window(c,eng,engines,last,0,"na",lambda i,s:dict(contact=False))
    assert torch.isnan(p.post_context["slip_m"]).all()
    assert p.post_context["slip_status"].tolist()==[-1,-1]
    p,last=run_window(c,eng,engines,last,8,"latch",lambda i,s:dict(contact=True,lifted=True))
    p,last=run_window(c,eng,engines,last,16,"lose-contact",
        lambda i,s:dict(contact=s<4,anchors_live=True,lifted=True))
    assert p.post_context["hold_s"].tolist()==[0.,0.]
    assert p.post_context["contact_s"].tolist()==[0.,0.]
    assert p.post_context["slip_m"].tolist()==[.0001,.0001]


def test_packet_owns_state_and_exactly_once_tokens():
    c=config();eng,engines,last=start(c,2)
    p,last=run_window(c,eng,engines,last,0,"once",lambda i,s:dict(error=.01))
    p.post_context["previous_request"].fill_(999.)
    assert eng.post_context["previous_request"].abs().max()<1.
    with pytest.raises(RuntimeError,match="token|settled"):
        eng.settle_once(("once",)*2)
    schema,frame=batch(last)
    with pytest.raises(RuntimeError,match="token|settled"):
        eng.begin(schema,request=torch.zeros(2,24,dtype=torch.float64),tokens=("once",)*2,
            policy_uids=("policy:0",)*2,physics_tick=frame.physics_tick,solver_time_s=frame.solver_time_s,
            remaining_substeps=torch.tensor([8,8]))


@pytest.mark.parametrize("mutation",["source","episode","guard_missing","units","target"])
def test_window_prevalidation_rejects_schema_identity_changes(mutation):
    c=config();eng,engines,last=start(c,2)
    schemas,frame=batch(last)
    bad=schemas[1]
    if mutation=="source": bad=replace(bad,field_specs=tuple((n,u,"changed" if n=="distance" else s) for n,u,s in bad.field_specs))
    elif mutation=="episode": bad=replace(bad,episode_uid="other")
    elif mutation=="guard_missing": bad=replace(bad,field_specs=tuple(x for x in bad.field_specs if x[0]!=GUARD))
    elif mutation=="units": bad=replace(bad,field_specs=tuple((n,"mm" if n=="distance" else u,s) for n,u,s in bad.field_specs))
    else: bad=replace(bad,target_ids=("wrong",))
    with pytest.raises(scalar.SourceFailure,match="schema|identity|units|field"):
        eng.begin((schemas[0],bad),request=torch.zeros(2,24,dtype=torch.float64),tokens=("w",)*2,
            policy_uids=("policy:0",)*2,physics_tick=frame.physics_tick,solver_time_s=frame.solver_time_s,
            remaining_substeps=8)


@pytest.mark.parametrize("corruption",["clock","negative_distance","nan_error","int_boolean","anchor_rebase"])
def test_every_substep_keeps_numerical_and_fixed_anchor_guards(corruption):
    c=config(full=corruption=="anchor_rebase");kwargs=dict(contact=True) if c.profile==scalar.FULL else {}
    eng,engines,last=start(c,2,**kwargs)
    schemas,pre=batch(last)
    eng.begin(schemas,request=torch.zeros(2,24,dtype=torch.float64),tokens=("w",)*2,
            policy_uids=("policy:0",)*2,physics_tick=pre.physics_tick,solver_time_s=pre.solver_time_s,remaining_substeps=8)
    for tick in range(1,5):
        _,frame=batch([row(c,i,tick,**kwargs) for i in range(2)])
        if tick==4:
            if corruption=="clock": frame.solver_time_s[1]+=.001
            elif corruption=="negative_distance": frame.fields["distance"][1]=-.001
            elif corruption=="nan_error": frame.fields[scalar.ERROR_FIELDS[1]][1]=math.nan
            elif corruption=="int_boolean": frame.fields["numerical_health"]=torch.ones(2,dtype=torch.int64)
            else: frame.anchor_reference[1,0,0]=.01
            with pytest.raises(scalar.SourceFailure): eng.observe(frame)
        else: eng.observe(frame)


def test_native_binary32_solver_clock_rounding_matches_frozen_interval_guard():
    c=config();eng,engines,last=start(c,2)
    # Begin with new source-bound reset at a long-running RopeBatch float32 time.
    def native(tick,lane):
        r=row(c,lane,tick,episode=1)
        stamp=replace(r.stamp,solver_time_s=float(np.float32(tick/480)),scope="unqualified_source",
                      source="policy_owned_pinch_endpoint_v1")
        return replace(r,stamp=stamp,fields={n:replace(m,stamp=replace(stamp,source=m.stamp.source)) for n,m in r.fields.items()})
    initial=[native(4000,i) for i in range(2)];schemas,pre=batch(initial)
    eng.reset_selected((0,1),schemas,pre,previous_request=torch.zeros(2,24,dtype=torch.float64),
                       tokens=("native-reset",)*2,policy_uids=("policy:0",)*2)
    originals=[scalar.TrainingReward(c) for i in range(2)]
    for i,o in enumerate(originals):
        o.reset_from_verified_state(initial[i],previous_request=req(initial[i],"native-reset",[0.]*24))
        o.begin(req(initial[i],"native-window",[0.]*24),remaining_substeps=8)
    eng.begin(schemas,request=torch.zeros(2,24,dtype=torch.float64),tokens=("native-window",)*2,
        policy_uids=("policy:0",)*2,physics_tick=pre.physics_tick,solver_time_s=pre.solver_time_s,remaining_substeps=8)
    for tick in range(4001,4009):
        rows=[native(tick,i) for i in range(2)];_,frame=batch(rows);eng.observe(frame)
        for i,o in enumerate(originals): o.observe(rows[i])
    p=eng.settle_once(("native-window",)*2)
    assert p.reward.tolist()==pytest.approx([o.settle_once("native-window").reward for o in originals])


@pytest.mark.parametrize("full",[False,True])
def test_all_potential_event_and_substep_cost_numbers_match_oracle(full):
    c=config(full=full);kwargs=dict(contact=True) if full else {}
    eng,engines,last=start(c,2,**kwargs)
    p,last=run_window(c,eng,engines,last,0,"all-terms",
        lambda i,s:dict(contact=True,lifted=True,drift=.0002,uncertainty=.0001,prohibited=i==1 and s>=4)
            if full else dict(distance=.015,error=.01,failure=i==1 and s>=3))
    for i,e in enumerate(engines):
        for name,term in e._terms.items():
            actual=p.terms[name]
            assert bool(actual["applicable"][i])==term["applicable"]
            assert actual["contribution"][i].item()==pytest.approx(term["contribution"])
            raw=actual["raw"]
            if term["raw"] is not None:
                assert raw[i].tolist()==pytest.approx(term["raw"])
        count=len(e._costs)
        for step,costs in enumerate(e._substep_terms):
            for name,term in costs.items():
                assert p.substep_terms[name]["contribution"][i,step].item()==pytest.approx(term["contribution"])
    assert "trusted_constraint_failure" in p.event_terms


@pytest.mark.parametrize("full",[False,True])
def test_large_finite_ratios_cannot_be_hidden_by_tensor_clamp(full):
    c=config(full=full);kwargs=dict(contact=True) if full else {}
    eng,engines,last=start(c,2,**kwargs)
    schemas,pre=batch(last)
    eng.begin(schemas,request=torch.zeros(2,24,dtype=torch.float64),tokens=("w",)*2,
        policy_uids=("policy:0",)*2,physics_tick=pre.physics_tick,solver_time_s=pre.solver_time_s,remaining_substeps=8)
    rows=[row(c,i,1,**(dict(contact=True,drift=1e307) if full else dict(distance=1e307))) for i in range(2)]
    for i,o in enumerate(engines):
        o.begin(req(last[i],"w",[0.]*24),remaining_substeps=8)
        with pytest.raises(scalar.SourceFailure): o.observe(rows[i])
    _,frame=batch(rows)
    with pytest.raises(scalar.SourceFailure): eng.observe(frame)


def test_failed_window_can_recover_only_through_new_verified_episode_reset():
    c=config();eng,engines,last=start(c,2)
    schemas,pre=batch(last)
    eng.begin(schemas,request=torch.zeros(2,24,dtype=torch.float64),tokens=("bad",)*2,
        policy_uids=("policy:0",)*2,physics_tick=pre.physics_tick,solver_time_s=pre.solver_time_s,remaining_substeps=8)
    _,bad=batch([row(c,i,1,distance=math.nan) for i in range(2)])
    with pytest.raises(scalar.SourceFailure): eng.observe(bad)
    rows=[row(c,i,0,episode=1) for i in range(2)];schemas,pre=batch(rows)
    eng.reset_selected((0,1),schemas,pre,previous_request=torch.zeros(2,24,dtype=torch.float64),
        tokens=("new-reset",)*2,policy_uids=("policy:0",)*2)
    eng.begin(schemas,request=torch.zeros(2,24,dtype=torch.float64),tokens=("good",)*2,
        policy_uids=("policy:0",)*2,physics_tick=pre.physics_tick,solver_time_s=pre.solver_time_s,remaining_substeps=8)
    for tick in range(1,9):
        _,frame=batch([row(c,i,tick,episode=1) for i in range(2)]);eng.observe(frame)
    assert torch.isfinite(eng.settle_once(("good",)*2).reward).all()
