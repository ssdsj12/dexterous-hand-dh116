"""Check what the actual Teacher used against its original Reset and physics rows."""
from collections import defaultdict
from hashlib import sha256

import numpy as np
import torch

from .tabletop_deformable_r0_replay import _digest, decode_owned
from .tabletop_deformable_rollout_admission import gae_boundary
from .tabletop_native_teacher_context import ContextIdentity, ContextMetadata, TaskContextPacket
from .tabletop_native_teacher_model import _pack
from .tabletop_native_teacher_worker_artifacts import _jsonl
from .tabletop_native_teacher_worker_contract import JSONL_LIMITS


SCOPE = "owner_source_unqualified"
CONTEXT_MODE = "approach_wiring_smoke_v1"
BUFFER_KEYS = {"pre_action", "raw_rod", "parameters", "context", "observation"}
CLOCK_KEYS = tuple(ContextIdentity.__dataclass_fields__)
ENDPOINT_KEYS = ("env_id", "episode_uid", "reset_id", "rod_uid", "physics_step", "rod_tick", "timestamp_s")


def _context(data):
    metadata = dict(data["metadata"])
    metadata["identities"] = tuple(ContextIdentity(**item) for item in metadata["identities"])
    for name in ("sources", "applicable", "reasons"):
        metadata[name] = tuple(metadata[name])
    metadata["reference_uids"] = tuple(tuple(item) for item in metadata["reference_uids"])
    return TaskContextPacket(torch.from_numpy(data["values"]), torch.from_numpy(data["valid"]),
                             ContextMetadata(**metadata))


def _rebuild(buffers, profile, context):
    common = buffers["pre_action"]["common"]
    ledger = torch.as_tensor(np.array(common["values"], copy=True), dtype=torch.float32)[None]
    valid = torch.as_tensor(np.array(common["valid"], copy=True), dtype=torch.bool)[None]
    snapshot = buffers["pre_action"]["rod_boundary"]["rod_snapshot"]
    snapshot = {name: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for name, value in snapshot.items()}
    snapshot = {name: value.float() if isinstance(value, torch.Tensor) and value.is_floating_point()
                else value for name, value in snapshot.items()}
    clock = {name: snapshot[name] for name in ("env_ids", "episode_uids", "reset_ids", "rod_uids",
                                               "physics_step", "rod_tick", "timestamp_s")}
    return _pack(ledger, valid, snapshot, clock, torch.ones((1, 41), dtype=torch.bool),
                 context, profile, CONTEXT_MODE)


def _check_pack(row, reference, anchor, *, purpose, profile, scope):
    recorded = row["payload"]
    if row["operation"] != "teacher_input_source" or recorded.get("purpose") != purpose:
        raise ValueError("Teacher source pack purpose differs from its decision or bootstrap")
    witness = decode_owned(recorded)
    buffers = witness["buffers"]
    digests = witness["sha256"]
    if (set(buffers) != BUFFER_KEYS or set(digests) != BUFFER_KEYS
            or any(_digest(buffers[name]) != digests[name] for name in BUFFER_KEYS)
            or witness["native_qualification"] is not False
            or witness["backend_profile_sha256"] != profile.digest):
        raise ValueError("Teacher pack bytes, profile or qualification flags differ")
    raw, boundary, parameters = anchor
    pre = buffers["pre_action"]
    if (witness["clock"] != {name: raw[name] for name in CLOCK_KEYS}
            or pre["common"]["identity"]["clock_valid"] is not True
            or any(pre["common"]["identity"][name] != raw[name] for name in CLOCK_KEYS)
            or _digest(buffers["raw_rod"]) != _digest(decode_owned(raw))
            or _digest(buffers["parameters"]) != _digest(decode_owned(parameters))
            or _digest(pre["rod_boundary"]) != _digest(decode_owned(boundary))
            or buffers["raw_rod"]["node_positions"].shape != (1, 41, 3)):
        raise ValueError("Teacher pack differs from the original same-clock 41-node physical source")
    if (reference["pre_action_sha256"] != digests["pre_action"]
            or _digest(decode_owned(reference["observation"])) != digests["observation"]):
        raise ValueError("Teacher pack differs from the actual decision or critic observation")
    context = _context(buffers["context"])
    metadata = context.metadata
    if (metadata.scope != scope or metadata.grasp != "pinch"
            or metadata.identities != (ContextIdentity(**witness["clock"]),)
            or metadata.owner_uid != witness["context_owner_uid"]):
        raise ValueError("Teacher context clock, owner, grasp or scope differs")
    if _digest(_rebuild(buffers, profile, context)) != digests["observation"]:
        raise ValueError("Teacher policy input differs from reconstructed original tensors")
    return metadata.owner_uid


def audit_teacher_input_sources(raw, *, backend_profile, scope, controls=32):
    """Independently join immutable source and policy rows, without native admission."""
    if scope not in ("cpu_fixture", SCOPE) or type(controls) is not int or not 1 <= controls <= 32:
        raise ValueError("bounded Teacher source audit and explicit scope required")
    name = "training_events.jsonl"
    rows = _jsonl(raw, name=name, maximum=JSONL_LIMITS[name])
    by_operation = defaultdict(list)
    for index, row in enumerate(rows):
        if (type(row.get("sequence")) is not int or row["sequence"] != index or row.get("scope") != scope
                or type(row.get("operation")) is not str or type(row.get("payload")) is not dict):
            raise ValueError("Teacher input journal row sequence or source scope differs")
        by_operation[row["operation"]].append(row)
    resets = [row for row in by_operation["owner_event"] if row["payload"].get("kind") == "reset"]
    reset_sources = by_operation["reset_sources"]
    decisions, settlements, physics = (by_operation[key] for key in ("decision", "settlement", "physical"))
    if (len(resets) != 1 or len(reset_sources) != 1 or len(decisions) != controls
            or len(settlements) != controls or len(physics) != controls * 8):
        raise ValueError("one Reset and one complete Teacher source ledger required")

    def from_physical(record):
        sample = record["payload"]
        return sample["exchange"]["rod"], sample["rod_boundary"], sample["exchange"]["material_director"]

    used = set()
    context_owners = set()

    def verify(reference, purpose, anchor, lower, upper):
        index = reference.get("source_pack_sequence")
        if type(index) is not int or not lower < index < upper or index in used:
            raise ValueError("Teacher source pack purpose, sequence order or uniqueness differs")
        owner = _check_pack(rows[index], reference, anchor,
                            purpose=purpose, profile=backend_profile, scope=scope)
        used.add(index)
        context_owners.add(owner)

    try:
        reset = resets[0]["payload"]
        start = reset["reset"]["rod_reset"], reset["rod_boundary"], reset_sources[0]["payload"]["parameters"]
        lower = reset_sources[0]["sequence"]
        if resets[0]["sequence"] >= lower:
            raise ValueError("Reset source order differs")
        for index, (decision, settlement) in enumerate(zip(decisions, settlements)):
            first, last = physics[8 * index], physics[8 * index + 7]
            after = from_physical(last)
            ds, ss = decision["sequence"], settlement["sequence"]
            if not ds < first["sequence"] <= last["sequence"] < ss:
                raise ValueError("Teacher pack does not bracket its complete physical control")
            do, so = decision["payload"], settlement["payload"]
            if start[0]["physics_step"] != 2 + 8 * index or after[0]["physics_step"] != 10 + 8 * index:
                raise ValueError("Teacher input physical step markers differ")
            verify(do, "decision", start, lower, ds)
            end = {key: after[0][key] for key in ENDPOINT_KEYS}
            end["phase"] = "post_action_pre_reset"
            reward = so["reward"]
            bootstrap = so["bootstrap_source"]
            if reward["terminated"] is True:
                if bootstrap is not None:
                    raise ValueError("true terminal must omit a Teacher bootstrap")
                next_endpoint = None
            else:
                if not isinstance(bootstrap, dict):
                    raise ValueError("live or truncated Teacher control requires its bootstrap")
                next_endpoint = bootstrap["identity"]
                if (bootstrap["collection_token"] != do["collection_token"]
                        or bootstrap["observation_sha256"] != _digest(decode_owned(bootstrap["observation"]))):
                    raise ValueError("Teacher bootstrap token or observation differs")
                verify(bootstrap, "bootstrap", after, last["sequence"], ss)
                packed_context = rows[bootstrap["source_pack_sequence"]]["payload"]["buffers"]["context"]
                if packed_context != so["post_context"]:
                    raise ValueError("Teacher bootstrap differs from the committed post-action context")
            if (reward["token"] != do["collection_token"]
                    or so["gae_boundary"] != gae_boundary(reward["terminated"], reward["truncated"],
                        endpoint_identity=end, next_value_identity=next_endpoint)
                    or (index + 1 < controls and so["gae_boundary"]["done"])):
                raise ValueError("Teacher control endpoint, token or GAE boundary differs")
            start, lower = after, ss
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("incomplete original Teacher input source evidence") from error

    if len(context_owners) != 1 or used != {row["sequence"] for row in by_operation["teacher_input_source"]}:
        raise ValueError("one context owner and every Teacher input source used exactly once required")
    return dict(source_scope=scope, decision_packs=controls, bootstrap_packs=len(used)-controls,
                source_packs=len(used), journal_prefix_sha256=sha256(raw).hexdigest(),
                native_qualification=False, teacher_admission=False)
