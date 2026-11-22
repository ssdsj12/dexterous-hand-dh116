"""Contact context is bound to owned physical events, never caller counters."""
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util

import numpy as np
import pytest
import torch

from test_native_teacher_source_priming import case
from test_native_teacher_context import backend_profile
from test_dh116_power_palm_contact_evidence import build
from test_action24_history_owner import model_setup, MeasuredCommonTeacher, DeformablePPOTrainer, PPOConfig
from go2_pvcnn.control.dh116.tabletop_native_teacher_context import ContextIdentity, encode_context


def module():
    name = "go2_pvcnn.control.dh116.tabletop_native_teacher_contact_context"
    assert importlib.util.find_spec(name), "missing event-bound contact context"
    return importlib.import_module(name)


def arrange(monkeypatch, tmp_path, profile, *, load=None):
    m, c, ep, history, runtime, journal, rows = case(monkeypatch, tmp_path)
    if load is not None:
        step = ep._exchange._step
        def contacted(action, substep):
            row = step(action, substep)
            # Explicit CPU solver-response double. Complete pair buffers and
            # aggregates still pass the original reducer; not native evidence.
            data = build(proximal=True, load=load(row["physics_step"]) if callable(load) else load)
            rod = row["rod"]
            for key in ("distributed_contact", "named_wrenches", "table_wrench_world",
                        "external_support_wrenches_world", "rod_mesh_contact_pairs"):
                rod[key] = deepcopy(data[key])
            cap = rod["rod_mesh_contact_pairs"]
            cap.update(physics_step=rod["physics_step"], rod_tick=rod["rod_tick"])
            return row
        ep._exchange._step = contacted
    pre = ep.pre_action_state()
    identity = ContextIdentity.from_clock(pre["common"]["identity"])
    state = dict(identity=identity, source_scope="cpu_fixture",
        node_positions=torch.as_tensor(pre["rod_boundary"]["rod_snapshot"]["node_positions"]).float().clone(),
        distance_mean_m=.02, rest_length_m=.5, target_reference_uid="explicit_cpu_fixed_material",
        backend_sha256=profile.binary_sha256,
        external_budget=dict(source="external_budget_clock",identity=identity,cap_s=20.,remaining_s=19.))
    owner = module().PrimedContactContextOwner(backend_profile=profile, scope="cpu_fixture",
        episode_cap_s=1., external_cap_s=20., grasp="tripod")
    owner.reset_pending_from_verified_state(state, history_owner=history)
    return m, c, ep, history, runtime, journal, rows, owner, state


def current_state(state, history):
    identity = ContextIdentity.from_clock(history._sequence.clock)
    return dict(state, identity=identity,
        external_budget=dict(state["external_budget"],identity=identity,remaining_s=18.))


def prime(arranged):
    m, c, ep, history, runtime, journal, rows, owner, state = arranged
    m.prime_supported_sources(ep,history=history,runtime=runtime,journal=journal)
    return owner.commit_primed_state(current_state(state,history),history_owner=history,
                                      priming_event=rows[-1])


def test_pending_reset_is_unavailable_and_prime_retains_s0_with_measured_fields(monkeypatch,tmp_path,backend_profile):
    data=arrange(monkeypatch,tmp_path,backend_profile)
    *_,owner,state=data
    with pytest.raises(ValueError):
        _=owner.current
    packet=prime(data)
    assert packet.metadata.mode=="approach_contact_prelift_v1"
    assert torch.equal(packet.values[0,:4],torch.tensor([0.,1.,0.,0.]))
    assert packet.valid[0,[4,5,6]].all() and not packet.valid[0,[7,47]].any()
    assert torch.isnan(packet.values[0,[7,47]]).all()
    assert not packet.values[0,[4,5,6]].any()
    assert torch.equal(packet.values[0,36:38],torch.tensor([
        float(state["node_positions"][0,:,2].median()),float(state["node_positions"][0,:,2].min())]))
    assert packet.values[0,38] == pytest.approx(.02)
    assert packet.values[0,43] == pytest.approx(1/480)
    assert packet.metadata.identities[0].physics_step==3
    assert owner._reset_identity.physics_step==2
    assert not packet.native_qualified
    assert "contact_event" in dict(packet.metadata.reference_uids)
    with pytest.raises(ValueError):
        prime(data)


def test_real_history_window_uses_all_events_then_bootstraps_before_commit(monkeypatch,tmp_path,backend_profile):
    data=arrange(monkeypatch,tmp_path,backend_profile,load=.125)
    m,c,ep,h,b,j,rows,owner,state=data
    initial=prime(data)
    assert torch.equal(initial.values[0,4:7],torch.tensor([1.,1.,0.]))
    pre=ep.pre_action_state()
    h.capture_pre_action(pre)
    frozen=owner.freeze_pre_action(ContextIdentity.from_clock(pre["common"]["identity"]),history_owner=h)
    config,identity,common,valid,raw,clock,mask=model_setup()
    model=MeasuredCommonTeacher(config,identity)
    obs={k:v[:1].clone() for k,v in model.pack_measured(common,valid,raw,clock,mask).items()}
    trainer=DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),rollout_steps=1,num_envs=1)
    sample=trainer.collect_step(obs,obs)
    h.begin(pre,trainer)
    record=ep._record
    def recorded(row):
        record(row)
        owner.observe_physical_event(row,history_owner=h)
    ep._record=recorded
    result=ep.control(sample["actions"])
    settled=h.end(result)
    packet=owner.advance_physical_endpoint(current_state(state,h),history_owner=h,
                                           collection_token=sample["collection_token"])
    assert packet.metadata.identities[0].physics_step==11
    assert packet.values[0,6] == pytest.approx((8/480)/.25)
    assert frozen.values[0,6] == owner.current.values[0,6] == 0.
    assert torch.equal(packet.values[:,12:36],sample["actions"])
    assert torch.isnan(packet.values[0,[7,8,9,47]]).all()
    assert packet.values[0,48]==0.
    clean=owner.pending_post
    packet.values[0,4]=0.
    assert owner.pending_post.values[0,4]==1.
    owner.commit_post_context(clean)
    assert owner.current.values[0,6]>0.
    assert h.issued_for_pending(trainer,settled).shape==(1,24)


def test_opposed_low_total_load_never_accumulates_persistence(monkeypatch,tmp_path,backend_profile):
    data=arrange(monkeypatch,tmp_path,backend_profile,load=.08)
    m,c,ep,h,b,j,rows,owner,state=data
    packet=prime(data)
    assert packet.values[0,4]==packet.values[0,5]==1.
    assert packet.values[0,6]==0.
    assert not owner.contact_readback["bilateral_support"]


@pytest.mark.parametrize("damage",["raw_event","foreign_clock","caller_counter","foreign_history"])
def test_primed_context_rejects_unowned_or_fabricated_sources(monkeypatch,tmp_path,backend_profile,damage):
    data=arrange(monkeypatch,tmp_path,backend_profile)
    m,c,ep,h,b,j,rows,owner,state=data
    m.prime_supported_sources(ep,history=h,runtime=b,journal=j)
    event=deepcopy(rows[-1]);post=current_state(state,h);history=h
    if damage=="raw_event":
        event["exchange"]["rod"]["node_positions"][0,0,0]+=.01
    elif damage=="foreign_clock":
        post["identity"]=replace(post["identity"],episode_uid="foreign")
    elif damage=="caller_counter":
        post["physical_fields"]={6:{"value":1000.}}
    else:
        history=object()
    with pytest.raises((ValueError,RuntimeError)):
        owner.commit_primed_state(post,history_owner=history,priming_event=event)
    with pytest.raises((ValueError,RuntimeError)):
        _=owner.current


def test_history_event_gap_is_not_interpolated_or_backfilled(monkeypatch,tmp_path,backend_profile):
    data=arrange(monkeypatch,tmp_path,backend_profile)
    m,c,ep,h,b,j,rows,owner,state=data
    prime(data)
    pre=ep.pre_action_state();h.capture_pre_action(pre)
    owner.freeze_pre_action(ContextIdentity.from_clock(pre["common"]["identity"]),history_owner=h)
    config,identity,common,valid,raw,clock,mask=model_setup()
    model=MeasuredCommonTeacher(config,identity)
    obs={k:v[:1].clone() for k,v in model.pack_measured(common,valid,raw,clock,mask).items()}
    trainer=DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),rollout_steps=1,num_envs=1)
    sample=trainer.collect_step(obs,obs);h.begin(pre,trainer)
    ep.control(sample["actions"])
    with pytest.raises((ValueError,RuntimeError)):
        owner.observe_physical_event(rows[-1],history_owner=h)
    with pytest.raises((ValueError,RuntimeError)):
        _=owner.current


def test_power_and_direct_reset_cannot_alias_contact_stage(monkeypatch,tmp_path,backend_profile):
    m=module()
    with pytest.raises(ValueError):
        m.PrimedContactContextOwner(backend_profile=backend_profile,scope="cpu_fixture",
                                   episode_cap_s=1.,external_cap_s=20.,grasp="power")
    data=arrange(monkeypatch,tmp_path,backend_profile)
    *_,owner,state=data
    with pytest.raises(ValueError):
        owner.reset_from_verified_state(state,history_owner=data[3])


def test_contact_fields_reach_actual_actor_with_independent_identity_and_ppo(backend_profile):
    from dataclasses import replace
    from test_native_teacher_model import cpu_model
    from go2_pvcnn.control.dh116.tabletop_native_teacher_model import NativeContextTeacher
    old,ledger,valid,rod,clock,mask,packet,config,base=cpu_model(backend_profile,grasp="tripod")
    # Fresh synthetic CPU model fixture, distinct from the physical source tests.
    packet.values[:,0]=0.;packet.values[:,1]=1.
    packet.values[:,4:7]=torch.tensor([.5,.25,.08])
    packet.valid[:,4:7]=True
    applicable=tuple(bool(packet.valid[0,i]) for i in range(50))
    packet=replace(packet,metadata=replace(packet.metadata,mode="approach_contact_prelift_v1",
        grasp="tripod",applicable=applicable,
        reasons=tuple(None if a else "unanchored or excluded in contact stage" for a in applicable)))
    model=NativeContextTeacher(config,backend_profile=backend_profile,
                               context_mode="approach_contact_prelift_v1")
    assert model.model_identity["checkpoint_namespace"]!=old.model_identity["checkpoint_namespace"]
    assert "tabletop_native_teacher_contact_context.py" in model.model_identity["dependency_source_sha256"]
    with pytest.raises(ValueError):
        old.pack_cpu_fixture(ledger,valid,rod,clock,mask,packet)
    obs=model.pack_cpu_fixture(ledger,valid,rod,clock,mask,packet)
    obs["task_context"].requires_grad_(True)
    model.deterministic(obs).square().sum().backward()
    assert obs["task_context"].grad[:,4:7].abs().sum()>0
    assert obs["task_context"].grad[:,[7,47]].abs().sum()==0
    assert all(p.grad is None for p in model.critic_context.parameters())
    obs={k:v.detach().clone() for k,v in obs.items()}
    trainer=DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),rollout_steps=1,num_envs=2)
    sample=trainer.collect_step(obs,obs)
    trainer.record_step(torch.tensor([.2,.1]),torch.tensor([True,True]),torch.zeros(2),
        executed_actions=sample["actions"],collection_token=sample["collection_token"])
    stats=trainer.update()
    assert stats["minibatch_updates"]==1 and trainer.counters["updates"]==1


def test_mid_window_contact_break_resets_tail_duration_and_readback_is_owned(monkeypatch,tmp_path,backend_profile):
    data=arrange(monkeypatch,tmp_path,backend_profile,load=lambda tick:.08 if tick==7 else .125)
    m,c,ep,h,b,j,rows,owner,state=data
    prime(data)
    readback=owner.contact_readback
    readback["coverage_left"]=0.
    assert owner.contact_readback["coverage_left"]==1.
    pre=ep.pre_action_state();h.capture_pre_action(pre)
    owner.freeze_pre_action(ContextIdentity.from_clock(pre["common"]["identity"]),history_owner=h)
    config,identity,common,valid,raw,clock,mask=model_setup()
    model=MeasuredCommonTeacher(config,identity)
    obs={k:v[:1].clone() for k,v in model.pack_measured(common,valid,raw,clock,mask).items()}
    tr=DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),rollout_steps=1,num_envs=1)
    sample=tr.collect_step(obs,obs);h.begin(pre,tr)
    record=ep._record
    observed=[]
    def recorded(row):
        record(row)
        observed.append(owner.observe_physical_event(row,history_owner=h))
    ep._record=recorded
    result=ep.control(sample["actions"]);h.end(result)
    assert observed[3]["duration_s"]==0.  # SDK7 weak contact
    assert observed[4]["duration_s"]==0.  # first qualified endpoint SDK8
    packet=owner.advance_physical_endpoint(current_state(state,h),history_owner=h,
                                           collection_token=sample["collection_token"])
    assert packet.values[0,6]==pytest.approx((3/480)/.25)


@pytest.mark.parametrize("damage",["budget_increase","success","phase"])
def test_priming_rejects_budget_reset_success_or_phase_injection(monkeypatch,tmp_path,backend_profile,damage):
    data=arrange(monkeypatch,tmp_path,backend_profile)
    m,c,ep,h,b,j,rows,owner,state=data
    m.prime_supported_sources(ep,history=h,runtime=b,journal=j)
    post=current_state(state,h)
    if damage=="budget_increase":
        post["external_budget"]["remaining_s"]=19.5
    elif damage=="phase":
        post["phase"]="Lift"
    else:
        post["event_history"]=dict(source="task_event_history",identity=post["identity"],valid=True,
                                    success_emitted=True,trusted_failure_emitted=False)
    with pytest.raises((ValueError,RuntimeError)):
        owner.commit_primed_state(post,history_owner=h,priming_event=rows[-1])
    with pytest.raises((ValueError,RuntimeError)):
        _=owner.current


def test_closed_contact_owner_cannot_expose_stale_packable_state(monkeypatch,tmp_path,backend_profile):
    data=arrange(monkeypatch,tmp_path,backend_profile)
    prime(data);owner=data[-2]
    owner.close()
    with pytest.raises(RuntimeError):
        _=owner.current
    with pytest.raises(RuntimeError):
        _=owner.contact_readback


def test_base_owner_cannot_bypass_event_bound_contact_implementation(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import TaskContextOwner
    with pytest.raises(ValueError,match="contact"):
        TaskContextOwner(backend_profile=backend_profile,mode="approach_contact_prelift_v1",
                         scope="cpu_fixture",episode_cap_s=1.,external_cap_s=20.,grasp="tripod")


def test_contact_model_identity_rejects_old_smoke_and_unsupported_power(backend_profile):
    from test_native_teacher_model import cpu_model
    from go2_pvcnn.control.dh116.tabletop_native_teacher_model import NativeContextTeacher,verify_native_teacher_identity
    old,*rest=cpu_model(backend_profile,grasp="tripod")
    config=rest[-2]
    contact=NativeContextTeacher(config,backend_profile=backend_profile,context_mode="approach_contact_prelift_v1")
    with pytest.raises(ValueError,match="identity differs"):
        verify_native_teacher_identity(old.model_identity,contact.model_identity)
    power=dict(config,grasp="power")
    with pytest.raises(ValueError):
        NativeContextTeacher(power,backend_profile=backend_profile,context_mode="approach_contact_prelift_v1")


def test_close_before_post_commit_cannot_partially_rewrite_context(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import _packet_hash
    data=arrange(monkeypatch,tmp_path,backend_profile)
    m,c,ep,h,b,j,rows,owner,state=data
    prime(data);pre=ep.pre_action_state();h.capture_pre_action(pre)
    owner.freeze_pre_action(ContextIdentity.from_clock(pre["common"]["identity"]),history_owner=h)
    config,identity,common,valid,raw,clock,mask=model_setup()
    model=MeasuredCommonTeacher(config,identity)
    obs={k:v[:1].clone() for k,v in model.pack_measured(common,valid,raw,clock,mask).items()}
    tr=DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),rollout_steps=1,num_envs=1)
    sample=tr.collect_step(obs,obs);h.begin(pre,tr)
    record=ep._record
    def recorded(row):
        record(row);owner.observe_physical_event(row,history_owner=h)
    ep._record=recorded
    result=ep.control(sample["actions"]);h.end(result)
    post=owner.advance_physical_endpoint(current_state(state,h),history_owner=h,
                                        collection_token=sample["collection_token"])
    before=_packet_hash(owner._current)
    pending=owner._pending
    frozen=owner._pre
    owner.close()
    with pytest.raises(RuntimeError,match="closed"):
        owner.commit_post_context(post)
    assert _packet_hash(owner._current)==before
    assert owner._pending is pending and owner._pre is frozen
