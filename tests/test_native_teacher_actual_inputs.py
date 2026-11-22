"""Source join regressions over original packed observations, with CPU scope."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest, encode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_rod_boundary import RodBoundaryState
from go2_pvcnn.control.dh116.tabletop_native_teacher_context import ContextIdentity
from go2_pvcnn.control.dh116.tabletop_native_teacher_model import _OwnedClock, pack_native_inputs
from test_native_teacher_context import backend_profile
from test_native_teacher_model import raw_cpu_packets


SCOPE = "cpu_fixture"


def audit(rows, profile, scope=SCOPE):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_inputs import audit_teacher_input_sources

    raw = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode()
    return audit_teacher_input_sources(raw, backend_profile=profile, scope=scope, controls=1)


def packet_at(profile, template, step):
    pre, raw, parameters, context = deepcopy(template)
    from test_dh116_teacher_mounted_wrist_source import capture, clock, hand
    from go2_pvcnn.control.dh116.tabletop_teacher_mounted_wrist_source import adapt_mounted_hands

    current = clock(step)
    elapsed = current["solver_time_s"]
    changes = {k: current[k] for k in ("physics_step", "rod_tick", "timestamp_s", "solver_time_s")}
    pre["common"] = adapt_mounted_hands(
        [capture(hand(side), side, read_identity=lambda: current) for side in ("left", "right")],
        expected_identity=current)
    raw.update(changes)
    if step > 2:
        raw["contact_valid"] = True
        raw["distributed_contact"] = raw["node_positions"].copy() * 0.
    raw["node_positions"][..., 2] += (step - 2) * .0002
    parameters["identity"].update(changes)
    parameters["node_positions_m"] = raw["node_positions"].copy()
    parameters["raw_native_parameters"]["time_s"] = elapsed
    from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot
    parameters["native_parameters"] = validate_snapshot(parameters["raw_native_parameters"],
        expected_num_envs=1, expected_nodes=41, expected_time_s=elapsed)
    boundary = RodBoundaryState(_OwnedClock({k: raw[k] for k in (
        "run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid", "physics_step", "rod_tick",
        "timestamp_s", "solver_time_s", "clock_valid")}), profile.binary_path, profile.binary_sha256)
    pre["rod_boundary"] = boundary.capture(raw)
    identity = ContextIdentity(**{key: raw[key] for key in ContextIdentity.__dataclass_fields__})
    context = replace(context, metadata=replace(context.metadata, identities=(identity,)))
    return pre, raw, parameters, context


@pytest.fixture
def ledger(backend_profile):
    return recorded_ledger(backend_profile)


def recorded_ledger(backend_profile, template=None):
    """Share the owning source-join fixture without duplicating its event writer."""
    template = raw_cpu_packets(backend_profile) if template is None else template
    rows = []

    def emit(operation, value):
        rows.append(dict(sequence=len(rows), operation=operation, scope=SCOPE, payload=encode_owned(value)))
        return len(rows) - 1

    def inputs(step, purpose):
        pre, raw, parameters, context = packet_at(backend_profile, template, step)
        obs = pack_native_inputs(pre, raw, parameters, context, backend_profile=backend_profile)
        buffers = dict(pre_action=pre, raw_rod=raw, parameters=parameters, context=asdict(context), observation=obs)
        seq = emit("teacher_input_source", dict(
            purpose=purpose, clock=asdict(context.metadata.identities[0]),
            context_owner_uid=context.metadata.owner_uid, backend_profile_sha256=backend_profile.digest,
            native_qualification=False, buffers=buffers, sha256={k: _digest(v) for k, v in buffers.items()}))
        return dict(source_pack_sequence=seq, pre_action_sha256=_digest(pre), observation=obs)

    pre, raw, parameters, _ = packet_at(backend_profile, template, 2)
    emit("owner_event", dict(kind="reset", reset=dict(rod_reset=raw), rod_boundary=pre["rod_boundary"]))
    emit("reset_sources", dict(parameters=parameters))
    decision = inputs(2, "decision")
    emit("decision", dict(decision, collection_token="cpu-0"))
    for step in range(3, 11):
        pre, raw, parameters, context = packet_at(backend_profile, template, step)
        emit("physical", dict(kind="physical", exchange=dict(rod=raw, material_director=parameters),
                              rod_boundary=pre["rod_boundary"]))
    bootstrap = inputs(10, "bootstrap")
    endpoint = {name: raw[name] for name in (
        "env_id", "episode_uid", "reset_id", "rod_uid", "physics_step", "rod_tick", "timestamp_s")}
    endpoint["phase"] = "post_action_pre_reset"
    emit("settlement", dict(reward=dict(token="cpu-0", terminated=False, truncated=True),
        post_context=asdict(context), history=dict(endpoint=asdict(context.metadata.identities[0])),
        bootstrap_source=dict(bootstrap, collection_token="cpu-0", identity=endpoint,
                              observation_sha256=_digest(bootstrap["observation"])),
        gae_boundary=dict(done=True, truncated=True, bootstrap_mask=True, gae_carry=False)))
    return rows


def test_truncation_joins_both_model_inputs_to_original_physical_boundaries(ledger, backend_profile):
    result = audit(ledger, backend_profile)
    assert result["decision_packs"] == result["bootstrap_packs"] == 1
    assert result["source_scope"] == SCOPE
    assert result["native_qualification"] is result["teacher_admission"] is False


def test_true_terminal_has_no_bootstrap(ledger, backend_profile):
    ledger.pop(-2)
    ledger[-1]["sequence"] -= 1
    outcome = ledger[-1]["payload"]
    outcome["reward"].update(terminated=True, truncated=False)
    outcome["bootstrap_source"] = None
    outcome["gae_boundary"].update(truncated=False, bootstrap_mask=False)
    result = audit(ledger, backend_profile)
    assert result["bootstrap_packs"] == 0


@pytest.mark.parametrize("damage,reason", [
    ("raw_positions", "physical source"),
    ("parameters", "physical source"),
    ("model_features", "reconstructed"),
    ("common_values", "mounted observation mask/values differ"),
    ("stale_reference", "purpose"),
    ("truncated_missing_bootstrap", "bootstrap"),
    ("wrong_endpoint", "endpoint"),
    ("relabelled_scope", "scope"),
])
def test_rejects_bad_durable_input_binding(ledger, backend_profile, damage, reason):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

    pack = ledger[-2]["payload"]
    row = ledger[-1]["payload"]["bootstrap_source"]
    buffers = pack["buffers"]
    scope = SCOPE
    if damage == "raw_positions":
        buffers["raw_rod"]["node_positions"]["__dh116_array__"][0][0][2] += .02
        pack["sha256"]["raw_rod"] = _digest(decode_owned(buffers["raw_rod"]))
    elif damage == "parameters":
        buffers["parameters"]["raw_native_parameters"]["contact_radius_m"] *= .5
        pack["sha256"]["parameters"] = _digest(decode_owned(buffers["parameters"]))
    elif damage == "model_features":
        buffers["observation"]["rod_nodes"]["__dh116_array__"][0][0][0] += .125
        digest = _digest(decode_owned(buffers["observation"]))
        pack["sha256"]["observation"] = digest
        row.update(observation=deepcopy(buffers["observation"]), observation_sha256=digest)
    elif damage == "common_values":
        common = buffers["pre_action"]["common"]["values"]["__dh116_array__"]
        common[0] += .01
        buffers["observation"]["common371"]["__dh116_array__"][0][0] = common[0]
        pre_digest = _digest(decode_owned(buffers["pre_action"]))
        obs_digest = _digest(decode_owned(buffers["observation"]))
        pack["sha256"].update(pre_action=pre_digest, observation=obs_digest)
        row.update(pre_action_sha256=pre_digest, observation=deepcopy(buffers["observation"]),
                   observation_sha256=obs_digest)
    elif damage == "stale_reference":
        row["source_pack_sequence"] = ledger[3]["payload"]["source_pack_sequence"]
    elif damage == "truncated_missing_bootstrap":
        ledger[-1]["payload"]["bootstrap_source"] = None
    elif damage == "wrong_endpoint":
        row["identity"]["physics_step"] -= 1
    elif damage == "relabelled_scope":
        scope = "owner_source_unqualified"
    with pytest.raises(ValueError, match=reason):
        audit(ledger, backend_profile, scope)
