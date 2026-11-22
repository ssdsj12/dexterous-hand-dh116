"""Check what the actual Teacher used against its original Reset and physics rows."""
from collections import defaultdict
from hashlib import sha256

import numpy as np
import torch

from .tabletop_deformable_r0_replay import _digest, decode_owned
from .tabletop_deformable_rollout_admission import gae_boundary
from .tabletop_native_teacher_context import ContextIdentity, ContextMetadata, TaskContextPacket
from .tabletop_native_teacher_model import _pack
from .tabletop_teacher_mounted_wrist_source import validate_mounted_observation
from .tabletop_native_teacher_worker_artifacts import _jsonl
from .tabletop_native_teacher_worker_contract import JSONL_LIMITS


SCOPE = "owner_source_unqualified"
CONTEXT_MODE = "approach_wiring_smoke_v1"
CONTACT_MODE = "approach_contact_prelift_v1"
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


def _rebuild(buffers, profile, context, context_mode=CONTEXT_MODE):
    common = buffers["pre_action"]["common"]
    validate_mounted_observation(common)
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
                 context, profile, context_mode)


def _check_pack(row, reference, anchor, *, purpose, profile, scope, grasp, context_mode=CONTEXT_MODE, contact_expected=None):
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
    if (metadata.scope != scope or metadata.grasp != grasp
            or metadata.identities != (ContextIdentity(**witness["clock"]),)
            or metadata.owner_uid != witness["context_owner_uid"]):
        raise ValueError("Teacher context clock, owner, grasp or scope differs")
    if grasp == "tripod":
        from pathlib import PurePosixPath
        packets={p["side"]:p for p in pre["common"]["source_packets"]}
        for side in ("left","right"):
            root=str(PurePosixPath(buffers["raw_rod"]["mesh_bindings"][side+".base_link"]["prim_path"]).parent)
            if packets[side]["hand_container_path"] != root:
                raise ValueError("Tripod source pack and mounted hand side/root differ")
    if contact_expected is not None:
        measured, duration, event_sha, fixed = contact_expected
        expected = torch.tensor([measured["coverage_left"],measured["coverage_right"],duration/.25])
        if (metadata.mode != CONTACT_MODE or not context.valid[0,[4,5,6]].all()
                or not torch.equal(context.values[0,[4,5,6]],expected)
                or not torch.equal(context.values[0,36:43],fixed)
                or context.valid[0,[7,47]].any() or not torch.isnan(context.values[0,[7,47]]).all()
                or context.values[0,48] != 0.
                or dict(metadata.reference_uids).get("contact_event") != event_sha):
            raise ValueError("contact Context differs from original measured event and Reset S0")
    if _digest(_rebuild(buffers, profile, context, context_mode)) != digests["observation"]:
        raise ValueError("Teacher policy input differs from reconstructed original tensors")
    return metadata.owner_uid


class _ContactTimeline:
    """Independent raw-event replay, not a caller-provided progress counter."""
    def __init__(self, reset, reset_source):
        nodes = np.asarray(decode_owned(reset["reset"]["rod_reset"])["node_positions"],dtype=np.float32)[0]
        distance = reset_source["endpoint"]["fields"]["distance"]["raw"]
        if type(distance) not in (int,float) or not np.isfinite(distance) or distance < 0:
            raise ValueError("real Reset distance required for contact source audit")
        self.fixed = torch.tensor([float(np.median(nodes[:,2])),float(nodes[:,2].min()),
                                   max(distance,.005),distance,.03,5.,.0005])
        self.last = ContextIdentity(**{k:reset["reset"]["rod_reset"][k] for k in CLOCK_KEYS})
        self.start = None
        self.points = {}

    def observe(self, row):
        from .tabletop_native_teacher_contact_measurement import capture_supported_contact
        event = decode_owned(row["payload"])
        rod = event["exchange"]["rod"]
        current = ContextIdentity(**{k:rod[k] for k in CLOCK_KEYS})
        if (current.epoch != self.last.epoch or current.physics_step != self.last.physics_step+1
                or current.rod_tick != self.last.rod_tick+12
                or abs(current.timestamp_s-self.last.timestamp_s-1/480)>2e-7):
            raise ValueError("consecutive raw contact source clocks required")
        contact = capture_supported_contact(rod,grasp="tripod")
        if contact["bilateral_support"]:
            self.start = current.physics_step if self.start is None else self.start
        else:
            self.start = None
        duration = 0. if self.start is None else (current.physics_step-self.start)/480
        if contact["table_contact_failure"] and event["exchange"]["guard_failure"] is None:
            raise ValueError("real table contact requires a persisted physical guard")
        self.points[current.physics_step] = (contact,duration,_digest(event),self.fixed)
        self.last = current

    def at(self, rod):
        return self.points[rod["physics_step"]]


def audit_teacher_input_sources(raw, *, backend_profile, scope, controls=32, grasp="pinch", profile=CONTEXT_MODE):
    """Independently join immutable source and policy rows, without native admission."""
    if (scope not in ("cpu_fixture", SCOPE) or type(controls) is not int or not 1 <= controls <= 32
            or grasp not in ("pinch","tripod") or profile not in (CONTEXT_MODE,CONTACT_MODE)
            or profile == CONTACT_MODE and grasp != "tripod"):
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
            or len(settlements) != controls
            or (len(physics) != controls*8 if profile == CONTEXT_MODE else not controls<=len(physics)<=controls*8)):
        raise ValueError("one Reset and one complete Teacher source ledger required")

    def from_physical(record):
        sample = record["payload"]
        return sample["exchange"]["rod"], sample["rod_boundary"], sample["exchange"]["material_director"]

    used = set()
    contact_timeline = None
    used_physics = set()
    context_owners = set()

    def verify(reference, purpose, anchor, lower, upper):
        index = reference.get("source_pack_sequence")
        if type(index) is not int or not lower < index < upper or index in used:
            raise ValueError("Teacher source pack purpose, sequence order or uniqueness differs")
        owner = _check_pack(rows[index], reference, anchor,
                            purpose=purpose, profile=backend_profile, scope=scope, grasp=grasp,context_mode=profile,
                            contact_expected=None if contact_timeline is None else contact_timeline.at(anchor[0]))
        used.add(index)
        context_owners.add(owner)

    try:
        reset = resets[0]["payload"]
        start = reset["reset"]["rod_reset"], reset["rod_boundary"], reset_sources[0]["payload"]["parameters"]
        if grasp == "tripod":
            from .tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES, _bindings
            baseline=None
            priming_rows = ([item["payload"]["exchange"]["rod"] for item in by_operation["owner_event"]
                             if item["payload"].get("kind") == "priming_physical"]
                            if profile == CONTACT_MODE else [])
            for row in (start[0],*priming_rows,*[item["payload"]["exchange"]["rod"] for item in physics]):
                try:
                    bindings=_bindings(row["mesh_bindings"])
                    if tuple(row["mesh_names"]) != PROXIMAL_POWER_MESH_NAMES or len(bindings) != 14:
                        raise ValueError("complete seventeen-slot layout required")
                    if any(v.get("frame")!="PhysX_link_origin" or v.get("unit")!="m" for v in bindings.values()):
                        raise ValueError("actual contact frame/unit differs")
                    current=_digest(bindings)
                    if baseline is None:baseline=current
                    if current!=baseline:raise ValueError("mid-step contact source bindings changed")
                except (KeyError,ValueError,TypeError) as error:
                    raise ValueError("Tripod physical source layout differs: "+str(error)) from error
        lower = reset_sources[0]["sequence"]
        if resets[0]["sequence"] >= lower:
            raise ValueError("Reset source order differs")
        if profile == CONTACT_MODE:
            primes = [row for row in by_operation["owner_event"]
                      if row["payload"].get("kind") == "priming_physical"]
            completed,primed = by_operation["source_priming_complete"],by_operation["primed_sources"]
            if (len(primes)!=1 or len(completed)!=1 or len(primed)!=1
                    or not lower<primes[0]["sequence"]<completed[0]["sequence"]<primed[0]["sequence"]<decisions[0]["sequence"]):
                raise ValueError("one separately charged priming source before first decision required")
            event = decode_owned(primes[0]["payload"])
            receipt = decode_owned(completed[0]["payload"])["receipt"]
            if (receipt != decode_owned(primed[0]["payload"])["receipt"]
                    or receipt["priming_row_sha256"] != _digest(event) or receipt["ppo_transitions"] != 0
                    or any(receipt["reset_identity"][k] != start[0][k] for k in CLOCK_KEYS)
                    or any(receipt["primed_identity"][k] != event["exchange"]["rod"][k] for k in CLOCK_KEYS)):
                raise ValueError("priming receipt, original Reset or raw source differs")
            contact_timeline = _ContactTimeline(reset,decode_owned(reset_sources[0]["payload"]))
            contact_timeline.observe(primes[0])
            if contact_timeline.at(event["exchange"]["rod"])[0]["table_contact_failure"]:
                raise ValueError("priming table contact cannot start a policy window")
            start,lower = from_physical(primes[0]),primed[0]["sequence"]
        for index, (decision, settlement) in enumerate(zip(decisions, settlements)):
            ds, ss = decision["sequence"], settlement["sequence"]
            window = ([item for item in physics if ds<item["sequence"]<ss] if profile == CONTACT_MODE
                      else physics[8*index:8*index+8])
            if not 1<=len(window)<=8:
                raise ValueError("actual bounded physical source window required")
            first,last = window[0],window[-1]
            used_physics.update(item["sequence"] for item in window)
            if contact_timeline is not None:
                for item in window:
                    contact_timeline.observe(item)
            after = from_physical(last)
            if not ds < first["sequence"] <= last["sequence"] < ss:
                raise ValueError("Teacher pack does not bracket its complete physical control")
            do, so = decision["payload"], settlement["payload"]
            if profile == CONTEXT_MODE:
                if start[0]["physics_step"] != 2+8*index or after[0]["physics_step"] != 10+8*index:
                    raise ValueError("Teacher input physical step markers differ")
            elif after[0]["physics_step"] != start[0]["physics_step"]+len(window):
                raise ValueError("actual primed source interval differs")
            verify(do, "decision", start, lower, ds)
            end = {key: after[0][key] for key in ENDPOINT_KEYS}
            end["phase"] = "post_action_pre_reset"
            reward = so["reward"]
            bootstrap = so["bootstrap_source"]
            if len(window)<8 and reward["terminated"] is not True:
                raise ValueError("partial physical source window requires a real terminal")
            if contact_timeline is not None:
                contact,duration,event_sha,fixed = contact_timeline.at(after[0])
                if contact["table_contact_failure"] and (reward["terminated"] is not True or reward["event"] != -40.):
                    raise ValueError("table contact requires a true failure terminal and -40 event")
                post = _context(decode_owned(so["post_context"]))
                if (not torch.equal(post.values[0,[4,5,6]],torch.tensor(
                        [contact["coverage_left"],contact["coverage_right"],duration/.25]))
                        or not torch.equal(post.values[0,36:43],fixed)
                        or dict(post.metadata.reference_uids).get("contact_event") != event_sha):
                    raise ValueError("post-contact Context differs from raw physical history")
                if reward["post_context"]["contact_s"] != duration or reward["post_context"]["hold_s"] is not None:
                    raise ValueError("contact reward timer differs from raw physical history")
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
    if used_physics != {item["sequence"] for item in physics}:
        raise ValueError("every physical source row must belong to exactly one policy window")
    result = dict(source_scope=scope, decision_packs=controls, bootstrap_packs=len(used)-controls,
                  source_packs=len(used), journal_prefix_sha256=sha256(raw).hexdigest(),
                  native_qualification=False, teacher_admission=False)
    if contact_timeline is not None:
        result.update(profile=profile,priming_packs=1,physical_samples=len(physics))
    return result
