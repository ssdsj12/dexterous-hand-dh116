"""Independent CPU context51; never evidence of fixed S0 or native contact."""
from dataclasses import replace
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module():
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_context_v2
    return tabletop_native_teacher_context_v2


@pytest.fixture(scope="module")
def backend_profile():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    return BackendProfile.load(ROOT)


def fixture(monkeypatch, tmp_path, profile):
    from test_action24_history_owner import fixture as history_fixture
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseOwnerV2
    m = module()
    history, ep, trainer, obs, *rest = history_fixture(monkeypatch, tmp_path)
    identity = m.ContextIdentity.from_clock(ep.pre_action_state()["common"]["identity"])
    phase = PhaseOwnerV2(prelift_stable_s=.25)
    phase.reset(episode_uid=identity.episode_uid, physics_tick=identity.physics_step)
    state = dict(identity=identity, source_scope="cpu_fixture",
        node_positions=torch.zeros(1, 41, 3), distance_mean_m=.02,
        rest_length_m=1., target_reference_uid="cpu-fixed-target",
        fixed_s0_manifest_sha256="1"*64, measurement_snapshot=None,
        phase_source_witness=None, backend_sha256=profile.binary_sha256,
        external_budget=dict(remaining_s=3., cap_s=3., source="external_budget_clock", identity=identity))
    state["node_positions"][..., 2] = .31
    state["physical_fields"] = fields(identity, phase)
    owner = m.TaskContextOwnerV2(backend_profile=profile, mode="fixed_s0_v2",
        scope="cpu_fixture", episode_cap_s=2., external_cap_s=3., prelift_scale_s=.25)
    packet = owner.reset_from_verified_state(state, history_owner=history, phase_owner=phase)
    return owner, history, ep, trainer, obs, state, phase, packet


def fields(identity, phase, source=None):
    m = module()
    values = {4: 0., 5: 0., 6: 0., 7: phase.current_receipt.final_hold_s, 47: 0.}
    if source is not None:
        values.update({4: source.stats["required_contact"].mean,
            5: source.stats["required_contact"].mean,
            6: source.stats["contact_persistence"].total,
            47: float(source.prohibited_support_latched)})
    return {i: dict(value=v, valid=True, source=m.CONTEXT_FIELDS_V2[i]["source"], identity=identity)
        for i, v in values.items()}


def step(owner, history, ep, trainer, obs, state, phase, *, stable=True, bilateral=True,
         final=False, anchor_owner=None, mutate=None):
    from test_action24_history_owner import begin
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseSampleV2
    identity = module().ContextIdentity.from_clock(ep.pre_action_state()["common"]["identity"])
    history.capture_pre_action(ep.pre_action_state())
    frozen = owner.freeze_pre_action(identity, history_owner=history)
    _, _, sampled, _ = begin(history, ep, trainer, obs)
    result = ep.control(sampled["actions"])
    settled = history.end(result)
    identity = module().ContextIdentity.from_clock(settled["endpoint"])
    for tick in range(phase.current_receipt.physics_tick+1, identity.physics_step+1):
        phase.observe(PhaseSampleV2(identity.episode_uid, tick, bilateral, stable, final))
    endpoint = dict(state, identity=identity,
        external_budget=dict(state["external_budget"], identity=identity),
        physical_fields=fields(identity, phase), anchor_owner=anchor_owner)
    if mutate is not None:
        mutate(endpoint)
    post = owner.advance_physical_endpoint(endpoint, history_owner=history,
        collection_token=sampled["collection_token"], phase_owner=phase)
    return frozen, post, sampled, result, settled


def test_context51_schema_preserves_nonphase_fields_and_old_context():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import CONTEXT_FIELDS
    m = module()
    assert len(m.CONTEXT_FIELDS_V2) == 51
    assert [x["name"] for x in m.CONTEXT_FIELDS_V2[:4]] == [
        "execution_phase_one_hot.Close", "execution_phase_one_hot.PreliftHold",
        "execution_phase_one_hot.Lift", "execution_phase_one_hot.LiftHold"]
    assert [dict(x) for x in m.CONTEXT_FIELDS_V2[4:50]] == [dict(x) for x in CONTEXT_FIELDS[4:50]]
    assert m.CONTEXT_FIELDS_V2[50]["name"] == "prelift_stable_duration"
    assert m.CONTEXT_SCHEMA_SHA256_V2 != __import__(
        "go2_pvcnn.control.dh116.tabletop_native_teacher_context", fromlist=["x"]).CONTEXT_SCHEMA_SHA256


def test_owned_reset_reads_history_and_separates_phase_source(monkeypatch, tmp_path, backend_profile):
    m = module()
    owner, history, ep, tr, obs, state, phase, packet = fixture(monkeypatch, tmp_path, backend_profile)
    assert packet.values.shape == packet.valid.shape == (1, 51)
    assert torch.equal(packet.values[0, :4], torch.tensor([1., 0., 0., 0.]))
    assert packet.values[0, 7] == packet.values[0, 50] == 0
    assert torch.equal(packet.values[:, 12:36], torch.zeros(1, 24))
    assert packet.native_qualified is False
    assert packet.metadata.fixed_s0_manifest_sha256 == "1"*64
    assert packet.metadata.phase_contract_sha256 == m.PHASE_CONTRACT_SHA256_V2
    assert packet.metadata.prelift_scale_s == .25
    assert m.encode_context_v2(packet.values, packet.valid).shape == (1, 102)
    assert packet.metadata.scope == "cpu_fixture"
    assert not hasattr(owner, "_phase_lease")


def test_pre_action_and_pending_post_are_copy_on_write(monkeypatch, tmp_path, backend_profile):
    owner, history, ep, tr, obs, state, phase, initial = fixture(monkeypatch, tmp_path, backend_profile)
    frozen, post, sample, result, settled = step(owner, history, ep, tr, obs, state, phase)
    assert post.values[0, 1] == 1
    assert post.values[0, 50] == pytest.approx((7/480)/.25)
    assert post.values[0, 7] == 0
    assert torch.equal(frozen.values[:, 12:36], torch.zeros(1, 24))
    assert torch.equal(post.values[:, 12:36], sample["actions"])
    post.values[0, 36] = -5
    assert owner.pending_post.values[0, 36] == initial.values[0, 36]
    clean = owner.pending_post
    assert clean.metadata is not owner._pending.metadata
    issued = history.issued_for_pending(tr, settled)
    tr.record_step(torch.zeros(1), torch.tensor([False]), torch.zeros(1),
        executed_actions=issued, collection_token=sample["collection_token"])
    assert len(tr.rollout) == 1
    owner.commit_post_context(clean)
    assert torch.equal(owner.current.values[:, 12:36], sample["actions"])
    torch.testing.assert_close(initial.values, frozen.values, equal_nan=True)
    with pytest.raises(ValueError):
        owner.commit_post_context(clean)


@pytest.mark.parametrize("damage", ["scope", "close"])
def test_native_scope_and_closed_commit_cannot_bypass_source_admission(monkeypatch, tmp_path, backend_profile, damage):
    m = module()
    owner, history, ep, tr, obs, state, phase, packet = fixture(monkeypatch, tmp_path, backend_profile)
    if damage == "scope":
        forged = replace(packet, metadata=replace(packet.metadata, scope="owner_source_unqualified"))
        with pytest.raises(ValueError, match="full_v2_phase_source_adapter_missing"):
            m.validate_context_v2(forged, backend_profile=backend_profile)
    else:
        _, post, *_ = step(owner, history, ep, tr, obs, state, phase)
        owner.close()
        with pytest.raises(ValueError, match="closed"):
            owner.commit_post_context(post)


@pytest.mark.parametrize("damage", ["phase_contract", "s0", "scale", "schema_source", "mode",
    "identities", "profile", "owner_token", "applicability", "no_na_reason"])
def test_new_metadata_schema_is_strict_and_does_not_mutate_packet(monkeypatch, tmp_path, backend_profile, damage):
    m = module()
    *_, packet = fixture(monkeypatch, tmp_path, backend_profile)
    meta = packet.metadata
    kwargs = {}
    if damage == "phase_contract": kwargs["phase_contract_sha256"] = "0"*64
    elif damage == "s0": kwargs["fixed_s0_manifest_sha256"] = "not-a-sha"
    elif damage == "scale": kwargs["prelift_scale_s"] = float("nan")
    elif damage == "schema_source": kwargs["sources"] = ("guessed",)*51
    elif damage == "mode": kwargs["mode"] = "approach_wiring_smoke_v1"
    elif damage == "identities": kwargs["identities"] = list(meta.identities)
    elif damage == "profile": kwargs["backend_profile_sha256"] = "0"*64
    elif damage == "owner_token": kwargs["previous_request_token"] = ""
    elif damage == "applicability": kwargs["applicable"] = (True,)*51
    else: kwargs["reasons"] = (None,)*51
    with pytest.raises(ValueError):
        m.validate_context_v2(replace(packet, metadata=replace(meta, **kwargs)), backend_profile=backend_profile)


@pytest.mark.parametrize("damage", ["old_packet", "old_width", "na_zero", "valid_nan", "duration_negative",
    "boolean", "required_missing", "deadline", "wrong_identity"])
def test_v2_packet_validator_rejects_old_and_invalid_values(monkeypatch, tmp_path, backend_profile, damage):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_context as old
    m = module()
    *_, packet = fixture(monkeypatch, tmp_path, backend_profile)
    if damage == "old_packet":
        from test_native_teacher_context import cpu_context
        with pytest.raises(ValueError):
            m.validate_context_v2(cpu_context(backend_profile), backend_profile=backend_profile)
        with pytest.raises(ValueError):
            old.validate_context(packet, backend_profile=backend_profile)
        return
    packet = packet.owned()
    if damage == "old_width": packet = replace(packet, values=packet.values[:, :50], valid=packet.valid[:, :50])
    elif damage == "na_zero": packet.values[:, 46] = 0
    elif damage == "valid_nan": packet.values[:, 50] = torch.nan
    elif damage == "duration_negative": packet.values[:, 50] = -.1
    elif damage == "boolean": packet.values[:, 0] = .5
    elif damage == "required_missing":
        packet.values[:, 50] = torch.nan
        packet.valid[:, 50] = False
    elif damage == "deadline": packet.values[:, 45] = 1
    else:
        with pytest.raises(ValueError):
            m.validate_context_v2(packet, backend_profile=backend_profile,
                identities=[replace(packet.metadata.identities[0], episode_uid="different")])
        return
    with pytest.raises(ValueError):
        m.validate_context_v2(packet, backend_profile=backend_profile)


@pytest.mark.parametrize("kwargs", [
    {"scope": "owner_source_unqualified"}, {"scope": "owner_source_unqualified", "source_adapter": object()},
    {"source_adapter": object()}, {"prelift_scale_s": 0}, {"prelift_scale_s": -1},
    {"prelift_scale_s": True}, {"prelift_scale_s": float("inf")},
    {"episode_cap_s": None}, {"mode": "flexible_train_r1_full"}])
def test_owner_cannot_accept_guessed_native_adapter_or_bad_configuration(backend_profile, kwargs):
    m = module()
    config = dict(backend_profile=backend_profile, mode="fixed_s0_v2", scope="cpu_fixture",
        episode_cap_s=2., external_cap_s=3., prelift_scale_s=.25)
    config.update(kwargs)
    with pytest.raises(ValueError):
        m.TaskContextOwnerV2(**config)


@pytest.mark.parametrize("damage", ["wrong_owner", "wrong_episode", "wrong_tick", "mutated_receipt",
    "caller_receipt", "caller_phase", "fixed_s0", "missing_field", "physical_clock",
    "physical_hold", "request_token", "history_owner", "fixed_d0", "reset_event", "foreign_anchor"])
def test_owned_endpoint_rejects_foreign_source_and_keeps_pending_empty(monkeypatch, tmp_path, backend_profile, damage):
    from test_action24_history_owner import begin
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseSampleV2, PhaseOwnerV2
    m = module()
    owner, history, ep, tr, obs, state, phase, packet = fixture(monkeypatch, tmp_path, backend_profile)
    identity = state["identity"]
    history.capture_pre_action(ep.pre_action_state())
    owner.freeze_pre_action(identity, history_owner=history)
    _, _, sampled, _ = begin(history, ep, tr, obs)
    result = ep.control(sampled["actions"])
    settled = history.end(result)
    identity = m.ContextIdentity.from_clock(settled["endpoint"])
    old_receipt = phase.current_receipt
    for tick in range(phase.current_receipt.physics_tick+1, identity.physics_step+1):
        phase.observe(PhaseSampleV2(identity.episode_uid, tick, True, True, False))
    endpoint = dict(state, identity=identity, physical_fields=fields(identity, phase),
        external_budget=dict(state["external_budget"], identity=identity))
    selected, token, hist = phase, sampled["collection_token"], history
    if damage == "wrong_owner":
        selected = PhaseOwnerV2(prelift_stable_s=.25)
        selected.reset(episode_uid=identity.episode_uid, physics_tick=identity.physics_step)
    elif damage == "wrong_episode": endpoint["identity"] = replace(identity, episode_uid="different")
    elif damage == "wrong_tick": endpoint["identity"] = state["identity"]
    elif damage == "mutated_receipt": object.__setattr__(old_receipt, "phase", "LiftHold")
    elif damage == "caller_receipt": endpoint["phase_receipt"] = phase.current_receipt
    elif damage == "caller_phase": endpoint["phase"] = "LiftHold"
    elif damage == "fixed_s0": endpoint["fixed_s0_manifest_sha256"] = "2"*64
    elif damage == "missing_field": endpoint["physical_fields"].pop(4)
    elif damage == "physical_clock": endpoint["physical_fields"][4]["identity"] = state["identity"]
    elif damage == "physical_hold": endpoint["physical_fields"][7]["value"] = 5.
    elif damage == "request_token": token = "foreign"
    elif damage == "history_owner": hist = object()
    elif damage == "fixed_d0": endpoint["distance_mean_m"] = .5
    elif damage == "reset_event": endpoint["success_event_emitted"] = True
    else: endpoint["anchor_owner"] = object()
    with pytest.raises((ValueError, TypeError)):
        owner.advance_physical_endpoint(endpoint, history_owner=hist, collection_token=token, phase_owner=selected)
    assert owner._pending is None


def test_explicit_cpu_anchor_source_is_fixed_across_contact_loss(monkeypatch, tmp_path, backend_profile):
    from test_native_teacher_context import established_cpu_anchor_owner
    from test_action24_history_owner import begin
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseSampleV2
    m = module()
    owner, history, ep, tr, obs, state, phase, packet = fixture(monkeypatch, tmp_path, backend_profile)
    identity = state["identity"]
    history.capture_pre_action(ep.pre_action_state())
    pre = owner.freeze_pre_action(identity, history_owner=history)
    _, _, sample, _ = begin(history, ep, tr, obs)
    result = ep.control(sample["actions"])
    settled = history.end(result)
    identity = m.ContextIdentity.from_clock(settled["endpoint"])
    source = established_cpu_anchor_owner(state, identity)
    for tick in range(3, identity.physics_step+1):
        phase.observe(PhaseSampleV2(identity.episode_uid, tick, True, True, False))
    endpoint = dict(state, identity=identity, anchor_owner=source,
        physical_fields=fields(identity, phase, source),
        external_budget=dict(state["external_budget"], identity=identity))
    original_step = source.step
    post = owner.advance_physical_endpoint(endpoint, history_owner=history,
        collection_token=sample["collection_token"], phase_owner=phase)
    assert source.step == original_step
    assert torch.isnan(pre.values[:, 8:10]).all()
    torch.testing.assert_close(post.values[:, 8:10], torch.tensor([[.25, .75]]))
    refs = post.metadata.reference_uids
    issued = history.issued_for_pending(tr, settled)
    tr.record_step(torch.zeros(1), torch.tensor([False]), torch.zeros(1),
        executed_actions=issued, collection_token=sample["collection_token"])
    owner.commit_post_context(post)
    _, detached, *_ = step(owner, history, ep, tr, obs, state, phase,
        stable=False, bilateral=False, final=False)
    assert detached.values[0, 0] == 1 and detached.values[0, 50] == detached.values[0, 7] == 0
    assert detached.metadata.reference_uids == refs
    torch.testing.assert_close(detached.values[:, 8:12], post.values[:, 8:12])
    torch.testing.assert_close(detached.values[:, 36:43], packet.values[:, 36:43])
    assert detached.values[0, 43] >= post.values[0, 43]
    owner.close()
    assert owner._phase_source is None and owner._anchor_source is None
    assert owner._pending_anchor_state is None
    assert source.step == original_step  # releasing a borrow does not control its source


def test_context_encoder_has_exact51_raw_fields_and_102_encoded_columns(monkeypatch, tmp_path, backend_profile):
    m = module()
    *_, packet = fixture(monkeypatch, tmp_path, backend_profile)
    values = packet.values.detach().clone().requires_grad_()
    encoded = m.encode_context_v2(values, packet.valid)
    assert encoded.shape == (1, 102)
    encoded.sum().backward()
    assert torch.equal(values.grad[packet.valid], torch.ones_like(values.grad[packet.valid]))
    assert torch.equal(values.grad[~packet.valid], torch.zeros_like(values.grad[~packet.valid]))
    assert torch.isnan(values[~packet.valid]).all()
    for width in (2, 50, 52):
        with pytest.raises(ValueError, match="51"):
            m.encode_context_v2(torch.zeros(1, width), torch.ones(1, width, dtype=torch.bool))


def test_owner_context_holds_distinct_prelift_and_final_clock(monkeypatch, tmp_path, backend_profile):
    from test_action24_history_owner import begin
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseSampleV2
    m = module()
    owner, history, ep, tr, obs, state, phase, packet = fixture(monkeypatch, tmp_path, backend_profile)
    # An explicit CPU-only shortened prelift candidate, never the physical success criterion.
    phase._prelift_required_s = 1/480
    frozen, post, sample, result, settled = step(owner, history, ep, tr, obs, state, phase)
    assert post.values[0, 2] == 1 and post.values[0, 50] > 0 and post.values[0, 7] == 0
    issued = history.issued_for_pending(tr, settled)
    tr.record_step(torch.zeros(1), torch.tensor([False]), torch.zeros(1),
        executed_actions=issued, collection_token=sample["collection_token"])
    owner.commit_post_context(post)
    _, post2, *_ = step(owner, history, ep, tr, obs, state, phase, final=True)
    assert post2.values[0, 3] == 1
    assert post2.values[0, 7] == pytest.approx((7/480)/5)
    assert post2.values[0, 50] == post.values[0, 50]
    assert phase.current_receipt.success_candidate is False
    assert post2.native_qualified is False


def test_reset_requires_fresh_phase_and_does_not_redefine_s0(monkeypatch, tmp_path, backend_profile):
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseOwnerV2, PhaseSampleV2
    m = module()
    *_, state, phase, packet = fixture(monkeypatch, tmp_path, backend_profile)
    assert packet.values[0, 50] == packet.values[0, 7] == 0
    with pytest.raises(ValueError):
        phase.reset(episode_uid=state["identity"].episode_uid, physics_tick=state["identity"].physics_step)
    for key, value in [("prelift_scale_s", True), ("phase_contract_sha256", "0"*64)]:
        with pytest.raises(ValueError):
            replace(packet.metadata, **{key: value})
    # The context51 module borrows neither the old physical task step nor its Reset.
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import FlexibleGraspEpisode
    original_step, original_reset = FlexibleGraspEpisode.step, FlexibleGraspEpisode.reset
    phase.observe(PhaseSampleV2(state["identity"].episode_uid, 3, True, True, False))
    assert FlexibleGraspEpisode.step is original_step and FlexibleGraspEpisode.reset is original_reset


@pytest.mark.parametrize("changed", [
    ("left", "tangential"), ("right", "tangential"),
    ("left", "material"), ("right", "material"), ("both", "material")])
def test_every_established_slip_reference_uid_is_locked_and_rejection_is_atomic(
        monkeypatch, tmp_path, backend_profile, changed):
    from dataclasses import asdict
    from copy import deepcopy
    from test_native_teacher_context import cpu_phase_owner
    from test_action24_history_owner import begin
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseSampleV2
    m = module()
    owner, history, ep, tr, obs, state, phase, packet = fixture(monkeypatch, tmp_path, backend_profile)
    source, drive = cpu_phase_owner(state)

    def execute_next_window():
        pre = ep.pre_action_state()
        identity = m.ContextIdentity.from_clock(pre["common"]["identity"])
        history.capture_pre_action(pre)
        owner.freeze_pre_action(identity, history_owner=history)
        _, _, sample, _ = begin(history, ep, tr, obs)
        result = ep.control(sample["actions"])
        settled = history.end(result)
        identity = m.ContextIdentity.from_clock(settled["endpoint"])
        drive(identity, "lift")  # original task runs every actual CPU physical endpoint
        for tick in range(phase.current_receipt.physics_tick+1, identity.physics_step+1):
            phase.observe(PhaseSampleV2(identity.episode_uid, tick, True, True, False))
        endpoint = dict(state, identity=identity, anchor_owner=source,
            physical_fields=fields(identity, phase, source),
            external_budget=dict(state["external_budget"], identity=identity))
        return endpoint, sample, settled

    def record_and_commit(post, sample, settled):
        issued = history.issued_for_pending(tr, settled)
        tr.record_step(torch.zeros(1), torch.tensor([False]), torch.zeros(1),
            executed_actions=issued, collection_token=sample["collection_token"])
        owner.commit_post_context(post)

    first, sample1, settled1 = execute_next_window()
    first_post = owner.advance_physical_endpoint(first, history_owner=history,
        collection_token=sample1["collection_token"], phase_owner=phase)
    record_and_commit(first_post, sample1, settled1)
    original_references = dict(owner.current.metadata.reference_uids)
    assert {"anchor.0", "anchor.1", "tangent.0", "tangent.1"} <= original_references.keys()

    second, sample2, settled2 = execute_next_window()
    old_refs = deepcopy(source.slip_references)
    side, kind = changed
    for selected in ("left", "right") if side == "both" else (side,):
        uid, provenance = source.slip_references[selected][kind]
        source.slip_references[selected][kind] = ("foreign-"+kind, provenance)

    # Snapshot the context transaction, not the source fault injection.
    before = dict(current=m._packet_hash(owner.current), pending=owner._pending,
        pre=m._packet_hash(owner._pre), tokens=set(owner._used_tokens),
        anchors=owner._anchors, anchor_uids=owner._anchor_uids, refs=owner._references,
        pending_anchor_state=owner._pending_anchor_state, pending_hash=owner._pending_hash,
        phase_source=owner._phase_source, phase_receipt=owner._phase_receipt,
        phase_seen_ref=owner._phase_seen_ref,
        phase_seen_value=deepcopy(owner._phase_seen_value))

    with pytest.raises(ValueError):
        owner.advance_physical_endpoint(second, history_owner=history,
            collection_token=sample2["collection_token"], phase_owner=phase)

    assert owner._pending is before["pending"] is None
    assert m._packet_hash(owner.current) == before["current"]
    assert m._packet_hash(owner._pre) == before["pre"]
    assert owner._used_tokens == before["tokens"]
    assert owner._anchors == before["anchors"]
    assert owner._anchor_uids == before["anchor_uids"]
    assert owner._references == before["refs"]
    assert owner._pending_anchor_state == before["pending_anchor_state"]
    assert owner._pending_hash == before["pending_hash"]
    assert owner._phase_source is before["phase_source"]
    assert owner._phase_receipt is before["phase_receipt"]
    assert owner._phase_seen_ref is before["phase_seen_ref"]
    assert owner._phase_seen_value == before["phase_seen_value"]
    assert asdict(owner._phase_seen_ref) == before["phase_seen_value"]

    # Correct only the corrupt source UID. The same settlement/token remains
    # usable, proving rejection did not consume or partially commit the window.
    source.slip_references = old_refs
    clean = owner.advance_physical_endpoint(second, history_owner=history,
        collection_token=sample2["collection_token"], phase_owner=phase)
    record_and_commit(clean, sample2, settled2)
    assert dict(owner.current.metadata.reference_uids) == original_references
