"""C02 CPU-only context contracts; no native/source admission."""
from dataclasses import replace
from pathlib import Path
import copy

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def context_module():
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_context
    return tabletop_native_teacher_context


@pytest.fixture(scope="module")
def backend_profile():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    return BackendProfile.load(ROOT)


def cpu_context(profile, batch=2, clock=None):
    """Synthetic NEW-profile context, never recorded backend data."""
    m = context_module()
    clock = clock or dict(env_ids=list(range(batch)), episode_uids=[f"ep-{i}" for i in range(batch)],
        reset_ids=[0]*batch, rod_uids=["rod0"]*batch, physics_step=24, rod_tick=288, timestamp_s=.05)
    values = torch.zeros(batch, 50, dtype=torch.float32)
    valid = torch.ones(batch, 50, dtype=torch.bool)
    values[:, 0] = 1.
    values[:, 36:43] = torch.tensor([.31, .31, .02, .02, .03, 5., .0005])
    values[:, 44] = 1.
    excluded = (4,5,6,7,8,9,46,47)
    values[:, excluded] = torch.nan
    valid[:, excluded] = False
    identities = tuple(m.ContextIdentity(
        run_uuid="cpu-fixture", env_id=clock["env_ids"][i], episode_uid=clock["episode_uids"][i],
        reset_id=clock["reset_ids"][i], rod_uid=clock["rod_uids"][i],
        physics_step=clock["physics_step"], rod_tick=clock["rod_tick"],
        timestamp_s=clock["timestamp_s"], solver_time_s=clock["rod_tick"]/5760)
        for i in range(batch))
    applicable = tuple(i not in excluded for i in range(50))
    meta = m.ContextMetadata(owner_uid="explicit-cpu-fixture", identities=identities,
        backend_profile_sha256=profile.digest, scope="cpu_fixture",
        mode="approach_wiring_smoke_v1", sources=tuple(f["source"] for f in m.CONTEXT_FIELDS),
        applicable=applicable, reasons=tuple(None if x else "smoke excluded or unanchored/NA" for x in applicable),
        reference_uids=(("target_material", "cpu-fixed-target"),),
        previous_request_token="cpu-explicit-reset-receipt")
    return m.TaskContextPacket(values, valid, meta)


def test_context_schema_has_exact_fifty_slots():
    fields = context_module().CONTEXT_FIELDS
    assert len(fields) == 50
    assert [f["index"] for f in fields] == list(range(50))
    assert [f["name"] for f in fields[12:36]] == [
        "previous_normalized_request24."+str(i) for i in range(24)]
    assert all(set(f) == {"index","name","unit","scale","source","validity","applicability"} for f in fields)
    assert len(context_module().CONTEXT_SCHEMA_SHA256) == 64


def test_generic_mask_encoder_keeps_unknown_raw_and_gradients():
    values = torch.tensor([[1., float("nan")]], requires_grad=True)
    valid = torch.tensor([[True, False]])
    encoded = context_module().encode_context(values, valid)
    assert torch.equal(encoded, torch.tensor([[1.,0.,1.,0.]]))
    encoded.sum().backward()
    assert torch.equal(values.grad, torch.tensor([[1.,0.]]))
    assert torch.isnan(values[0,1])


@pytest.mark.parametrize("damage", ["valid_nan","fake_na","required_missing","smoke_phase","deadline",
    "identity_mutable","identity_bool","bad_source","profile","full_missing","no_reason"])
def test_context_contract_rejects_missing_or_fabricated_sources(backend_profile, damage):
    m = context_module()
    packet = cpu_context(backend_profile)
    if damage == "valid_nan": packet.values[0,0] = torch.nan
    elif damage == "fake_na": packet.values[:,8] = 0
    elif damage == "required_missing":
        packet.values[:,12] = torch.nan; packet.valid[:,12] = False
    elif damage == "smoke_phase": packet.values[:,0] = 0; packet.values[:,1] = 1
    elif damage == "deadline": packet.values[:,45] = 1
    elif damage == "identity_mutable":
        with pytest.raises(ValueError): replace(packet.metadata, identities=list(packet.metadata.identities))
        return
    elif damage == "identity_bool":
        with pytest.raises(ValueError): replace(packet.metadata.identities[0], env_id=True)
        return
    elif damage == "bad_source":
        packet = replace(packet, metadata=replace(packet.metadata, sources=("guessed",)*50))
    elif damage == "profile": packet = replace(packet, metadata=replace(packet.metadata, backend_profile_sha256="0"*64))
    elif damage == "full_missing":
        packet = replace(packet, metadata=replace(packet.metadata, mode="flexible_train_r1_full"))
    else: packet = replace(packet, metadata=replace(packet.metadata, reasons=(None,)*50))
    with pytest.raises(ValueError):
        m.validate_context(packet, backend_profile=backend_profile)


def owner_fixture(monkeypatch, tmp_path, profile):
    from test_action24_history_owner import fixture
    history, ep, trainer, obs, rod, hands, sim, rows, initial = fixture(monkeypatch, tmp_path)
    m = context_module()
    identity = m.ContextIdentity.from_clock(ep.pre_action_state()["common"]["identity"])
    state = dict(identity=identity, source_scope="cpu_fixture",
        node_positions=torch.zeros(1,41,3), distance_mean_m=.02,
        rest_length_m=1., target_reference_uid="cpu-fixed-target",
        backend_sha256=profile.binary_sha256,external_budget=dict(remaining_s=3.,cap_s=3.,
            source="external_budget_clock",identity=identity))
    state["node_positions"][...,2] = .31
    owner = m.TaskContextOwner(backend_profile=profile, mode="approach_wiring_smoke_v1",
        scope="cpu_fixture", episode_cap_s=2., external_cap_s=3.)
    owner.reset_from_verified_state(state, history_owner=history)
    return owner, history, ep, trainer, obs, state


def test_owner_reads_real_history_reset_and_issued_request_copy_on_write(monkeypatch,tmp_path,backend_profile):
    from test_action24_history_owner import begin
    owner, history, ep, tr, obs, state = owner_fixture(monkeypatch,tmp_path,backend_profile)
    pre = ep.pre_action_state()
    history.capture_pre_action(pre)
    frozen = owner.freeze_pre_action(state["identity"], history_owner=history)
    assert torch.equal(frozen.values[:,12:36], torch.zeros(1,24))
    _,_,sample,_ = begin(history,ep,tr,obs)
    result = ep.control(sample["actions"])
    issued = history.end(result)
    endpoint = dict(state, identity=context_module().ContextIdentity.from_clock(issued["endpoint"]))
    endpoint["external_budget"]=dict(state["external_budget"],identity=endpoint["identity"])
    post = owner.advance_physical_endpoint(endpoint, history_owner=history,
        collection_token=sample["collection_token"])
    assert torch.equal(post.values[:,12:36], sample["actions"])
    assert post.values[0,43] > frozen.values[0,43]
    assert torch.equal(owner.current.values[:,12:36], frozen.values[:,12:36])
    post.values[0,36] = -20  # a returned tensor cannot rewrite the pending post context
    clean = owner.pending_post
    assert clean.values[0,36] == frozen.values[0,36]
    owner.commit_post_context(clean)
    assert torch.equal(owner.current.values[:,12:36], sample["actions"])
    assert torch.equal(frozen.values[:,12:36], torch.zeros(1,24))
    with pytest.raises(ValueError): owner.commit_post_context(clean)


@pytest.mark.parametrize("damage", ["time","episode","reanchor","token","foreign_history","mutable_identity"])
def test_owner_rejects_endpoint_time_epoch_reanchor_and_previous_request_token(monkeypatch,tmp_path,backend_profile,damage):
    from test_action24_history_owner import begin
    m = context_module()
    owner, history, ep, tr, obs, state = owner_fixture(monkeypatch,tmp_path,backend_profile)
    history.capture_pre_action(ep.pre_action_state())
    owner.freeze_pre_action(state["identity"], history_owner=history)
    _,_,sample,_ = begin(history,ep,tr,obs)
    result=ep.control(sample["actions"]); issued=history.end(result)
    endpoint=dict(state,identity=m.ContextIdentity.from_clock(issued["endpoint"]))
    endpoint["external_budget"]=dict(state["external_budget"],identity=endpoint["identity"])
    token=sample["collection_token"]
    if damage == "time": endpoint["identity"] = state["identity"]
    elif damage == "episode": endpoint["identity"] = replace(endpoint["identity"], episode_uid="stale")
    elif damage == "reanchor": endpoint["target_reference_uid"] = "new-anchor"
    elif damage == "token": token="foreign"
    elif damage == "foreign_history": history=object()
    else: endpoint["identity"] = vars(type("MutableIdentity",(),{})())
    with pytest.raises((ValueError,TypeError)):
        owner.advance_physical_endpoint(endpoint, history_owner=history, collection_token=token)


def test_owner_requires_real_history_owner_and_explicit_caps(backend_profile):
    m=context_module()
    with pytest.raises(ValueError):
        m.TaskContextOwner(backend_profile=backend_profile,mode="approach_wiring_smoke_v1",
            scope="cpu_fixture",episode_cap_s=None,external_cap_s=3.)
    owner=m.TaskContextOwner(backend_profile=backend_profile,mode="approach_wiring_smoke_v1",
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    with pytest.raises(ValueError):
        owner.reset_from_verified_state({},history_owner=object())

@pytest.mark.parametrize("phase,one_hot",[
    ("approach",[1.,0.,0.,0.]),("grasp",[0.,1.,0.,0.]),
    ("lift",[0.,0.,1.,0.]),("hold",[0.,0.,0.,1.]),
])
def test_full_owner_reads_actual_original_task_phase_receipt(monkeypatch,tmp_path,backend_profile,phase,one_hot):
    from test_action24_history_owner import begin
    m=context_module()
    _,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    source,drive=cpu_phase_owner(state)
    full=m.TaskContextOwner(backend_profile=backend_profile,mode="flexible_train_r1_full",
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    fullstate=dict(state,physical_fields=full_physical_fields(state["identity"]))
    reset=full.reset_from_verified_state(fullstate,history_owner=history,phase_owner=source)
    assert torch.equal(reset.values[0,:4],torch.tensor([1.,0.,0.,0.]))
    assert torch.isnan(reset.values[:,8:10]).all() and not reset.valid[:,8:10].any()
    assert not reset.values[:,10:12].any()
    history.capture_pre_action(ep.pre_action_state())
    full.freeze_pre_action(state["identity"],history_owner=history)
    _,_,sample,_=begin(history,ep,tr,obs)
    result=ep.control(sample["actions"]); settled=history.end(result)
    identity=m.ContextIdentity.from_clock(settled["endpoint"])
    receipt=drive(identity,phase)
    assert receipt["phase"] == phase and source.phase == phase
    # Returned task receipt is caller-owned; mutation cannot change the observer.
    receipt["phase"]="invented"
    endpoint=dict(fullstate,identity=identity,physical_fields=full_physical_fields(identity,source=source),
        external_budget=dict(state["external_budget"],identity=identity))
    try:
        if phase == "hold":
            endpoint["physical_fields"][7]["value"]=.1
            with pytest.raises(ValueError,match="hold"):
                full.advance_physical_endpoint(endpoint,history_owner=history,
                    collection_token=sample["collection_token"],phase_owner=source,anchor_owner=source)
            endpoint["physical_fields"][7]["value"]=source.hold_s
        post=full.advance_physical_endpoint(endpoint,history_owner=history,
            collection_token=sample["collection_token"],phase_owner=source,
            anchor_owner=source if source.material_anchor is not None else None)
        assert torch.equal(post.values[0,:4],torch.tensor(one_hot))
        assert post.values[0,7] == pytest.approx(source.hold_s/5)
        if phase == "hold": assert post.values[0,7] == 0  # first actual good endpoint
        assert post.values[0,6] == pytest.approx(source.stats["contact_persistence"].total/.25)
    finally:
        full.close()


def test_context_metadata_rejects_mutable_uid_boolean_clock_and_unknown_grasp(monkeypatch,tmp_path,backend_profile):
    m=context_module()
    packet=cpu_context(backend_profile)
    with pytest.raises(ValueError):
        replace(packet.metadata,reference_uids=(("target_material",{"mutable":"uid"}),))
    identity=packet.metadata.identities[0]
    with pytest.raises(ValueError):
        replace(identity,physics_step=True)
    with pytest.raises(ValueError):
        m.ContextMetadata(**{**{k:getattr(packet.metadata,k) for k in packet.metadata.__dataclass_fields__},
            "grasp":"invented"})


@pytest.mark.parametrize("damage",["absent","cap","clock","remaining","raw_event_bool"])
def test_owner_requires_actual_external_budget_source_and_event_history(monkeypatch,tmp_path,backend_profile,damage):
    from test_action24_history_owner import begin
    m=context_module()
    owner,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    history.capture_pre_action(ep.pre_action_state())
    owner.freeze_pre_action(state["identity"],history_owner=history)
    _,_,sample,_=begin(history,ep,tr,obs)
    result=ep.control(sample["actions"]);settled=history.end(result)
    identity=m.ContextIdentity.from_clock(settled["endpoint"])
    endpoint=dict(state,identity=identity,external_budget=dict(state["external_budget"],identity=identity))
    if damage == "absent": endpoint.pop("external_budget")
    elif damage == "cap": endpoint["external_budget"]["cap_s"]=4.
    elif damage == "clock": endpoint["external_budget"]["identity"]=state["identity"]
    elif damage == "remaining": endpoint["external_budget"]["remaining_s"]=4.
    else: endpoint["success_event_emitted"]=True
    with pytest.raises(ValueError):
        owner.advance_physical_endpoint(endpoint,history_owner=history,collection_token=sample["collection_token"])

def full_physical_fields(identity,hold=0.,source=None):
    m=context_module()
    values={i:0. for i in m.SMOKE_EXCLUDED}
    values[7]=hold
    if source is not None:
        # These fixtures use a constant pad set throughout the physical prefix,
        # so the original task's accumulated persistence is also continuous.
        values[4]=values[5]=source.stats["required_contact"].mean
        values[6]=source.stats["contact_persistence"].total
        values[7]=source.hold_s
        values[47]=float(source.prohibited_support_latched)
    return {i:dict(value=values[i],valid=True,source=m.CONTEXT_FIELDS[i]["source"],identity=identity)
        for i in m.SMOKE_EXCLUDED}


def cpu_phase_owner(state):
    """Unchanged original reset/step with explicitly synthetic CPU rod/contact inputs."""
    from test_dh116_deformable_grasp_task import CONFIG,ROOT,snapshot,signals,measurement
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import (
        load_flexible_grasp_task,FlexibleGraspEpisode,RESET_HOOKS)
    import numpy as np
    task=load_flexible_grasp_task(CONFIG,ROOT)
    identity=state["identity"]
    owner=FlexibleGraspEpisode(task,"pinch",env_id=identity.env_id)
    def snap(step,dz=0.):
        row=snapshot(task,step,uid=identity.episode_uid,reset_id=identity.reset_id,rod_uid=identity.rod_uid)
        row["rod_tick"]=(step-identity.physics_step)*12
        row["rod_snapshot"]["node_positions"]=np.zeros((1,41,3))
        row["rod_snapshot"]["node_positions"][0,:,1]=np.linspace(0,1,41)
        row["rod_snapshot"]["node_positions"][...,2]=.31+dz
        row["rod_snapshot"]["rest_lengths_m"]=np.full((1,40),.025)
        return row
    owner.reset({k:lambda:None for k in RESET_HOOKS},lambda:(snap(identity.physics_step),measurement(.005)))
    assert owner.phase == "approach" and owner.material_anchor is None
    assert owner.slip_references == {"left":{},"right":{}}
    def drive(endpoint,phase):
        receipt=None
        for step in range(owner.clock[0]+1,endpoint.physics_step+1):
            signal=signals(task,step=step,uid=identity.episode_uid,reset_id=identity.reset_id)
            signal["sample_identity"].update(rod_uid=identity.rod_uid,rod_tick=(step-identity.physics_step)*12)
            for side in ("left","right"):
                contact=signal[side]["value"]
                contact["rod_uid"]=identity.rod_uid
                if phase == "approach": contact.update(pads=[],opposed=False)
                elif phase == "grasp": contact.update(pads=contact["pads"][:1],opposed=False)
            # Actual hold first becomes eligible at the final physical endpoint.
            dz=.035 if phase == "hold" and step == endpoint.physics_step else 0.
            receipt=owner.step(snap(step,dz),signal)
        return receipt
    return owner,drive


def established_cpu_anchor_owner(state, endpoint_identity):
    source,drive=cpu_phase_owner(state)
    drive(endpoint_identity,"lift")
    assert source.material_anchor is not None
    return source


@pytest.mark.parametrize("damage",[None,"fake_owner","wrong_epoch","wrong_clock","guessed_station"])
def test_first_actual_anchor_is_once_established_without_rewriting_pre_context(monkeypatch,tmp_path,backend_profile,damage):
    from test_action24_history_owner import begin
    m=context_module()
    owner,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    history.capture_pre_action(ep.pre_action_state())
    frozen=owner.freeze_pre_action(state["identity"],history_owner=history)
    _,_,sample,_=begin(history,ep,tr,obs)
    result=ep.control(sample["actions"]); settled=history.end(result)
    identity=m.ContextIdentity.from_clock(settled["endpoint"])
    endpoint=dict(state,identity=identity,external_budget=dict(state["external_budget"],identity=identity))
    source=established_cpu_anchor_owner(state,identity)
    if damage == "fake_owner": source=object()
    elif damage == "wrong_epoch": source.identity=("foreign",identity.reset_id,identity.rod_uid)
    elif damage == "wrong_clock": source.clock=(identity.physics_step-1,identity.rod_tick,identity.timestamp_s)
    elif damage == "guessed_station": endpoint["anchor_stations_m"]=(.2,.8)
    if damage is not None:
        with pytest.raises(ValueError):
            owner.advance_physical_endpoint(endpoint,history_owner=history,
                collection_token=sample["collection_token"],anchor_owner=source)
        return
    post=owner.advance_physical_endpoint(endpoint,history_owner=history,
        collection_token=sample["collection_token"],anchor_owner=source)
    assert torch.isnan(frozen.values[:,8:10]).all()
    assert torch.isnan(owner.current.values[:,8:10]).all()
    torch.testing.assert_close(post.values[:,8:10],torch.tensor([[.25,.75]]))
    assert post.valid[:,8:12].all()
    assert any(k == "tangent.0" for k,v in post.metadata.reference_uids)
    owner.commit_post_context(post)
    assert owner.current.valid[:,8:12].all()
    # An explicit attempted reanchor must fail in the next real action window.
    issued=history.issued_for_pending(tr,settled)
    tr.record_step(torch.zeros(1),torch.tensor([False]),torch.zeros(1),
        executed_actions=issued,collection_token=sample["collection_token"])
    nextpre=ep.pre_action_state();history.capture_pre_action(nextpre)
    owner.freeze_pre_action(identity,history_owner=history)
    _,_,sample2,_=begin(history,ep,tr,obs)
    result2=ep.control(sample2["actions"]); settled2=history.end(result2)
    id2=m.ContextIdentity.from_clock(settled2["endpoint"])
    bad=dict(state,identity=id2,external_budget=dict(state["external_budget"],identity=id2),
        anchor_stations_m=(.3,.75),anchor_uids=("replacement","replacement"))
    with pytest.raises(ValueError):
        owner.advance_physical_endpoint(bad,history_owner=history,collection_token=sample2["collection_token"])

@pytest.mark.parametrize("stations,uids",[
    ((.2,.8),("invented-left","invented-right")),
    ((.2,None),("invented-left",None)),
])
def test_fresh_reset_rejects_caller_invented_anchor_state(monkeypatch,tmp_path,backend_profile,stations,uids):
    m=context_module()
    _,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    fresh=m.TaskContextOwner(backend_profile=backend_profile,mode="approach_wiring_smoke_v1",
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    with pytest.raises(ValueError,match="fresh Reset"):
        fresh.reset_from_verified_state(dict(state,anchor_stations_m=stations,anchor_uids=uids),
            history_owner=history)


def test_full_context_cannot_enter_hold_from_phase_source_strings(monkeypatch,tmp_path,backend_profile):
    m=context_module()
    _,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    fresh=m.TaskContextOwner(backend_profile=backend_profile,mode="flexible_train_r1_full",
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    fields={i:dict(value=0.,valid=True,source=m.CONTEXT_FIELDS[i]["source"],identity=state["identity"])
        for i in m.SMOKE_EXCLUDED}
    with pytest.raises(ValueError,match="phase"):
        fresh.reset_from_verified_state(dict(state,phase="Hold",phase_source="physical_phase_state_machine",
            physical_fields=fields),history_owner=history)

@pytest.mark.parametrize("damage",[
    "no_owner","wrong_object","wrong_epoch","wrong_clock","wrong_grasp",
    "replaced_step","caller_hold","mutated_owner_phase","bypass_observer",
])
def test_full_phase_requires_owned_original_step_receipt(monkeypatch,tmp_path,backend_profile,damage):
    from test_action24_history_owner import begin
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import FlexibleGraspEpisode
    m=context_module()
    _,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    source,drive=cpu_phase_owner(state)
    full=m.TaskContextOwner(backend_profile=backend_profile,mode="flexible_train_r1_full",
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    full.reset_from_verified_state(dict(state,physical_fields=full_physical_fields(state["identity"])),
        history_owner=history,phase_owner=source)
    history.capture_pre_action(ep.pre_action_state());full.freeze_pre_action(state["identity"],history_owner=history)
    _,_,sample,_=begin(history,ep,tr,obs)
    result=ep.control(sample["actions"]);settled=history.end(result)
    identity=m.ContextIdentity.from_clock(settled["endpoint"])
    if damage == "bypass_observer":
        observer=source.step
        source.step=observer.original_step
        drive(identity,"approach")
        source.step=observer
    else: drive(identity,"approach")
    endpoint=dict(state,identity=identity,physical_fields=full_physical_fields(identity),
        external_budget=dict(state["external_budget"],identity=identity))
    candidate=source
    if damage == "no_owner": candidate=None
    elif damage == "wrong_object": candidate=cpu_phase_owner(state)[0]
    elif damage == "wrong_epoch": source.identity=("foreign",identity.reset_id,identity.rod_uid)
    elif damage == "wrong_clock": source.clock=(identity.physics_step-1,identity.rod_tick,identity.timestamp_s)
    elif damage == "wrong_grasp": source.grasp="power"
    elif damage == "replaced_step": source.step=lambda *a,**k: {}
    elif damage == "caller_hold": endpoint.update(phase="Hold",phase_source="physical_phase_state_machine")
    elif damage == "mutated_owner_phase": source.phase="hold"
    try:
        with pytest.raises(ValueError,match="phase"):
            full.advance_physical_endpoint(endpoint,history_owner=history,
                collection_token=sample["collection_token"],phase_owner=candidate)
        assert full._pending is None
    finally:
        if damage == "replaced_step": source.step=full._phase_lease
        full.close()


def test_full_phase_observer_lease_close_and_exception_restore_original_step(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import FlexibleGraspEpisode
    m=context_module()
    _,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    source,drive=cpu_phase_owner(state)
    original=source.step
    def fresh():
        return m.TaskContextOwner(backend_profile=backend_profile,mode="flexible_train_r1_full",
            scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    full=fresh()
    fullstate=dict(state,physical_fields=full_physical_fields(state["identity"]))
    full.reset_from_verified_state(fullstate,history_owner=history,phase_owner=source)
    observer=source.step
    with pytest.raises(ValueError,match="phase.*lease"):
        fresh().reset_from_verified_state(fullstate,history_owner=history,phase_owner=source)
    assert source.step is observer
    full.close();full.close()
    assert source.step == original
    assert observer.original_step is None
    # New binding is explicit and never stacks an old wrapper.
    again=fresh()
    again.reset_from_verified_state(fullstate,history_owner=history,phase_owner=source)
    second=source.step
    assert second is not observer and second.original_step == original
    with pytest.raises((ValueError,KeyError)):
        source.step({}, {})  # unchanged original step rejects the malformed physical source
    assert source.step == original
    assert second.original_step is None and again._phase_failed
    assert not any(isinstance(x,BaseException) for x in again.__dict__.values())
    again.close()


def test_smoke_does_not_install_or_require_phase_observer(monkeypatch,tmp_path,backend_profile):
    m=context_module()
    owner,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    source,drive=cpu_phase_owner(state)
    original=source.step
    fresh=m.TaskContextOwner(backend_profile=backend_profile,mode="approach_wiring_smoke_v1",
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    packet=fresh.reset_from_verified_state(dict(state,phase="Hold"),history_owner=history,phase_owner=source)
    assert source.step == original
    assert torch.equal(packet.values[0,:4],torch.tensor([1.,0.,0.,0.]))
    fresh.close()

@pytest.mark.parametrize("fake",[True,{},object()])
def test_full_reset_rejects_untyped_phase_owner_without_installing_observer(monkeypatch,tmp_path,backend_profile,fake):
    m=context_module()
    _,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    full=m.TaskContextOwner(backend_profile=backend_profile,mode="flexible_train_r1_full",
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    with pytest.raises(ValueError,match="phase owner"):
        full.reset_from_verified_state(dict(state,physical_fields=full_physical_fields(state["identity"])),
            history_owner=history,phase_owner=fake)
    assert full._phase_lease is None


@pytest.mark.parametrize("damage",["invented_anchors","caller_hold","not_fresh_owner_phase"])
def test_full_reset_preserves_original_fresh_unanchored_approach(monkeypatch,tmp_path,backend_profile,damage):
    m=context_module()
    _,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    source,drive=cpu_phase_owner(state)
    original=source.step
    full=m.TaskContextOwner(backend_profile=backend_profile,mode="flexible_train_r1_full",
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    fullstate=dict(state,physical_fields=full_physical_fields(state["identity"]))
    if damage == "invented_anchors":
        fullstate.update(anchor_stations_m=(.2,.8),anchor_uids=("invented","invented"))
    elif damage == "caller_hold": fullstate.update(phase="Hold",phase_source="physical_phase_state_machine")
    else: source.phase="hold"
    with pytest.raises(ValueError):
        full.reset_from_verified_state(fullstate,history_owner=history,phase_owner=source)
    assert full._phase_lease is None and source.step == original

def test_explicit_absent_c04_history_module_preserves_original_exact_owner(monkeypatch,tmp_path,backend_profile):
    import sys
    name="go2_pvcnn.control.dh116.tabletop_native_teacher_action_history"
    monkeypatch.setitem(sys.modules,name,None)
    owner,history,*_=owner_fixture(monkeypatch,tmp_path,backend_profile)
    assert owner._history(history) is history


def test_history_owner_does_not_accept_unreviewed_subclasses(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116.tabletop_action24_history_owner import Action24HistoryOwner
    class CallerSubclass(Action24HistoryOwner):
        pass
    m=context_module()
    _,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    fake=CallerSubclass(initial_action24=torch.zeros(1,24),seed_source="explicit CPU fake subclass",
        physics_budget_end=18,solver_substeps=12)
    fresh=m.TaskContextOwner(backend_profile=backend_profile,mode="approach_wiring_smoke_v1",
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    with pytest.raises(ValueError,match="Action24HistoryOwner"):
        fresh.reset_from_verified_state(state,history_owner=fake)


def test_full_phase_and_anchor_share_one_original_physical_owner(monkeypatch,tmp_path,backend_profile):
    from test_action24_history_owner import begin
    m=context_module()
    _,history,ep,tr,obs,state=owner_fixture(monkeypatch,tmp_path,backend_profile)
    source,drive=cpu_phase_owner(state)
    full=m.TaskContextOwner(backend_profile=backend_profile,mode="flexible_train_r1_full",
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    full.reset_from_verified_state(dict(state,physical_fields=full_physical_fields(state["identity"])),
        history_owner=history,phase_owner=source)
    history.capture_pre_action(ep.pre_action_state());full.freeze_pre_action(state["identity"],history_owner=history)
    _,_,sample,_=begin(history,ep,tr,obs)
    result=ep.control(sample["actions"]);settled=history.end(result)
    identity=m.ContextIdentity.from_clock(settled["endpoint"])
    drive(identity,"hold")
    other=established_cpu_anchor_owner(state,identity)
    endpoint=dict(state,identity=identity,physical_fields=full_physical_fields(identity,source=source),
        external_budget=dict(state["external_budget"],identity=identity))
    try:
        with pytest.raises(ValueError,match="phase.*anchor"):
            full.advance_physical_endpoint(endpoint,history_owner=history,
                collection_token=sample["collection_token"],phase_owner=source,anchor_owner=other)
        # Omitting an anchor argument cannot hide a known actual task anchor.
        post=full.advance_physical_endpoint(endpoint,history_owner=history,
            collection_token=sample["collection_token"],phase_owner=source)
        torch.testing.assert_close(post.values[:,8:10],torch.tensor([[.25,.75]]))
        assert post.values[:,10:12].all()
    finally:
        full.close()
