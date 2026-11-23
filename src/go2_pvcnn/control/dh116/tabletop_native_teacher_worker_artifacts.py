"""Read-only C06 artifact auditing, distinct from native receipt qualification.

The old scheduler v1 artifact implementation is untouched. This protocol keeps
all raw journal rows and validates the derived views rather than trusting a
producer's schema/completion marker.
"""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

from .tabletop_native_teacher_worker_contract import (
    validate_teacher_recipe_v2, validate_teacher_training_result, worker_training_source_contract,
    worker_registered_native_package, _sha, _exact_int, SMOKE_PROFILE, CONTACT_PROFILE)

SDK_OPERATIONS = ("delivered_exchange", "physical", "budget_consumed", "policy_pinch_distance")
CONTROL_OPERATIONS = ("decision", "settlement", "transition_recorded")
PRIMARY_OPERATIONS = frozenset((*SDK_OPERATIONS,"policy_tripod_distance","policy_power_distance",*CONTROL_OPERATIONS,
    "collector_origin","policy_epoch","optimizer_update_accepted"))


def _required_native_source_scope():
    """Fixed production source contract; CPU fixtures substitute only this gate."""
    return "owner_source_unqualified"


def _json(raw):
    def invalid(value):
        raise ValueError("nonfinite JSON value: "+value)
    try:
        result=json.loads(raw,parse_constant=invalid)
        def finite(value):
            if isinstance(value,float) and not math.isfinite(value):
                raise ValueError("nonfinite JSON number")
            if isinstance(value,dict):
                for child in value.values():finite(child)
            elif isinstance(value,list):
                for child in value:finite(child)
        finite(result)
        return result
    except (ValueError,UnicodeDecodeError) as error:
        raise ValueError("complete finite JSON artifact required: "+str(error)) from error


def _jsonl(raw, *, name, maximum):
    result=[]
    for line in raw.splitlines(keepends=True):
        if len(result)>=maximum:
            raise ValueError("v2 JSONL exceeds declared row budget: "+name)
        if not line.endswith(b"\n"):
            raise ValueError("v2 JSONL requires complete newline-terminated rows: "+name)
        row=_json(line)
        if not isinstance(row,dict):
            raise ValueError("v2 JSONL rows must be complete objects: "+name)
        result.append(row)
    return result


def _journal(rows, result, *, controls=32, initial_version=0, prior_accepted=0,
             prior_weights=None, grasp="pinch", profile=SMOKE_PROFILE):
    if grasp not in ("pinch","tripod","power"):
        raise ValueError("independent journal grasp required")
    if profile == CONTACT_PROFILE:
        if grasp != "tripod":
            raise ValueError("independent contact journal source required")
        _contact_prime(rows,end_sdk_step=3+8*controls)
    elif profile != SMOKE_PROFILE:
        raise ValueError("known worker journal profile required")
    sdk_operations=(*SDK_OPERATIONS[:-1],"policy_"+grasp+"_distance")
    by_operation={}
    for index,row in enumerate(rows):
        if (set(row)!={"sequence","scope","operation","payload"}
                or not _exact_int(row["sequence"],index)
                or row["scope"]!=_required_native_source_scope()
                or not isinstance(row["operation"],str) or not row["operation"]
                or not isinstance(row["payload"],dict)):
            raise ValueError("raw NativeJournal sequence/scope/operation/object contract differs")
        by_operation.setdefault(row["operation"],[]).append(row)
    if len(by_operation.get("optimizer_update_accepted",[]))!=1:
        raise ValueError("raw journal must contain exactly one accepted optimizer update")
    expected=["collector_origin"]
    for _ in range(controls):
        expected.append("decision")
        for _ in range(8):expected.extend(sdk_operations)
        expected.extend(("settlement","transition_recorded"))
    expected.extend(("policy_epoch","optimizer_update_accepted"))
    actual=[r["operation"] for r in rows if r["operation"] in PRIMARY_OPERATIONS]
    if actual!=expected:
        raise ValueError("raw journal must retain complete original control/SDK operation sequence")
    training=result["training"]
    accepted=by_operation["optimizer_update_accepted"][0]["payload"]
    if any(type(accepted.get(k)) is not type(training[k]) or accepted[k]!=training[k]
            for k in ("updates","accepted_minibatch_updates","parameter_delta_l2")):
        raise ValueError("summary accepted update must equal original journal counters and weight-change metric")
    metrics=accepted.get("metrics")
    new_minibatches=training["accepted_minibatch_updates"]-prior_accepted
    if (new_minibatches<1 or not isinstance(metrics,dict)
            or not _exact_int(metrics.get("minibatch_updates"),new_minibatches)):
        raise ValueError("accepted journal metrics must contain the actual completed minibatch count")
    origin=by_operation["collector_origin"][0]["payload"].get("policy")
    epoch=by_operation["policy_epoch"][0]["payload"]
    if (not isinstance(origin,dict) or not _exact_int(origin.get("policy_version"),initial_version)
            or not _exact_int(epoch.get("policy_version"),initial_version+1)
            or prior_weights is not None and origin.get("weights")!=prior_weights
            or not isinstance(origin.get("session_id"),str) or not origin["session_id"]
            or epoch.get("session_id")!=origin["session_id"] or epoch.get("model")!=origin.get("model")
            or any(not _sha(item.get(key)) for item in (origin,epoch) for key in ("weights","model"))
            or epoch["weights"]==origin["weights"]):
        raise ValueError("journal must bind one real changed policy epoch in the same session/model")
    tokens=[]
    for index,(decision,settlement,transition) in enumerate(zip(by_operation["decision"],
            by_operation["settlement"],by_operation["transition_recorded"])):
        token=decision["payload"].get("collection_token")
        settled=settlement["payload"].get("reward")
        selected_policy=decision["payload"].get("policy")
        if (not isinstance(token,str) or not token or token in tokens
                or not isinstance(selected_policy,dict) or not _exact_int(selected_policy.get("policy_version"),initial_version)
                or selected_policy!=origin
                or not isinstance(settled,dict) or settled.get("token")!=token
                or transition["payload"].get("collection_token")!=token
                or not _exact_int(transition["payload"].get("rollout_length"),index+1)):
            raise ValueError("journal decision/settlement/transition tokens or rollout sequence differ")
        tokens.append(token)
    return deepcopy(epoch)


def _view(name, rows, journal, operation):
    expected=[r for r in journal if (r["operation"]==operation if operation else r["operation"] not in PRIMARY_OPERATIONS)]
    if len(rows)!=len(expected):
        raise ValueError("derived journal view must preserve all scoped raw rows: "+name)
    for row,raw in zip(rows,expected):
        if (set(row)!={"raw_sequence","payload"} or not _exact_int(row["raw_sequence"],raw["sequence"])
                or json.dumps(row["payload"],sort_keys=True,separators=(",",":"),allow_nan=False)
                    !=json.dumps(raw["payload"],sort_keys=True,separators=(",",":"),allow_nan=False)):
            raise ValueError("derived journal view differs from original raw sequence/payload: "+name)


def audit_teacher_artifacts_v2(spec, output):
    """Freeze all artifact bytes once; audit rows before the CPU bytes auditor."""
    from scripts.run_dh116_work_packages import native_artifact
    validate_teacher_recipe_v2(spec)
    raw={name:native_artifact(output,name).read_bytes() for name in spec["validator"]["artifacts"]}
    digests={name:hashlib.sha256(value).hexdigest() for name,value in raw.items()}
    limits=spec["teacher_contract"]["limits"]["jsonl_limits"]
    ledgers={name:_jsonl(raw[name],name=name,maximum=maximum) for name,maximum in limits.items()}
    for name,expected in spec["validator"]["line_counts"].items():
        if len(ledgers[name])!=expected:
            raise ValueError("v2 exact physical/transition row budget differs: "+name)
    result=_json(raw["training_result.json"])
    checkpoint=validate_teacher_training_result(result,
        expected_semantic_sha256=spec["teacher_contract"]["semantic_sha256"],artifact_sha256=digests)
    journal=ledgers["training_events.jsonl"]
    epoch=_journal(journal,result,controls=spec["teacher_contract"]["limits"]["controls"],grasp=spec["teacher_contract"]["style"],profile=spec["teacher_contract"]["profile"])
    _view("physics.jsonl",ledgers["physics.jsonl"],journal,"physical")
    _view("transitions.jsonl",ledgers["transitions.jsonl"],journal,"transition_recorded")
    _view("ownership.jsonl",ledgers["ownership.jsonl"],journal,None)
    close=_json(raw["shutdown_complete.json"])
    if (not isinstance(close,dict) or close.get("close_returned") is not True
            or close.get("process_exit_verified") is not False
            or "pending_simulation_callback_error" not in close or close["pending_simulation_callback_error"] is not None):
        raise ValueError("v2 close marker requires returned App.close without callback error; outer exit remains scheduler-owned")
    return dict(artifact_sha256=digests,ledger_counts={n:len(v) for n,v in ledgers.items()},
        teacher_training=deepcopy(result["training"]),policy_epoch=epoch,
        checkpoint_binding=checkpoint,checkpoint_bytes=raw[checkpoint["path"]],journal_rows=journal)


def _contact_prime(rows, *, end_sdk_step):
    """Replay the original zero exchange and its separately charged receipt."""
    import numpy as np
    from .tabletop_deformable_r0_replay import decode_owned, _digest
    from .tabletop_deformable_dual_evidence import _Sequence, _clock_row
    from .tabletop_native_teacher_source_priming import VERSION, SOURCE
    groups = {}
    for row in rows:
        groups.setdefault(row["operation"], []).append(row)
    def one(operation, kind=None):
        selected=[row for row in groups.get(operation,[]) if kind is None or row["payload"].get("kind")==kind]
        if len(selected)!=1:
            raise ValueError("exactly one original contact priming/Reset record required")
        return selected[0]
    reset=one("owner_event","reset")
    event_row=one("owner_event","priming_physical")
    exchange=one("priming_exchange")
    complete=one("source_priming_complete")
    sources=one("primed_sources")
    decisions=groups.get("decision",[])
    if (not decisions or not reset["sequence"]<exchange["sequence"]<event_row["sequence"]
            <complete["sequence"]<sources["sequence"]<decisions[0]["sequence"]):
        raise ValueError("contact priming must precede every policy decision")
    event=decode_owned(event_row["payload"])
    completed=decode_owned(complete["payload"])
    receipt=completed["receipt"]
    raw=event["exchange"]
    from .tabletop_deformable_bimanual_step import VERSION as ACTION_VERSION
    from .tabletop_surface_contract import ACTION_CONTRACT
    trace=raw["action_readback"]
    reset_snapshot=decode_owned(reset["payload"])["rod_boundary"]["rod_snapshot"]
    if (trace.get("version")!=ACTION_VERSION or trace.get("env_ids")!=[0]
            or type(trace["env_ids"][0]) is not int
            or _digest(trace["request"]["action_contract"])!=_digest(ACTION_CONTRACT)
            or event["rod_boundary"]["rod_snapshot"]["backend_sha256"]!=reset_snapshot["backend_sha256"]):
        raise ValueError("original priming action/environment/backend protocol differs")
    action=np.asarray(raw["action_readback"]["request"]["action24"])
    if (raw["guard_failure"] is not None or action.shape!=(1,24)
            or action.dtype!=np.float64 or not np.array_equal(action,np.zeros((1,24),np.float64))
            or _digest(decode_owned(exchange["payload"]))!=_digest(raw)
            or receipt!=decode_owned(sources["payload"])["receipt"]
            or receipt.get("version")!=VERSION or receipt.get("source")!=SOURCE
            or not isinstance(receipt.get("owner_uid"),str) or not receipt["owner_uid"]
            or receipt.get("priming_row_sha256")!=_digest(event)
            or receipt.get("zero_action_sha256")!=_digest(np.zeros((1,24),np.float32))
            or not _exact_int(receipt.get("ppo_transitions"),0)
            or any(receipt.get(k) is not False for k in ("native_qualification","teacher_admission"))
            or any(not _exact_int(completed.get(k),v) for k,v in
                dict(budget_sdk_steps=3,budget_solver_calls=1,budget_internal_solver_substeps=12).items())):
        raise ValueError("original zero priming request, receipt or charged budget differs")
    sequence=_Sequence(end_sdk_step,12)
    setup=groups.get("setup",[])
    if len(setup)!=2:
        raise ValueError("two original setup boundaries before contact priming required")
    for row in setup:
        sequence.accept("setup",row["payload"])
    sequence.accept("reset",decode_owned(reset["payload"]))
    before=deepcopy(sequence.clock)
    if (not _exact_int(before["env_id"],0)
            or type(before["reset_id"]) is not int or before["reset_id"]<1
            or not _exact_int(before["physics_step"],2) or not _exact_int(before["rod_tick"],0)
            or _clock_row(receipt["reset_identity"])!=before):
        raise ValueError("contact source must be primed from the original SDK2 Reset")
    sequence.action=np.zeros((1,24),np.float64)
    sequence._physical(event)
    if (not _exact_int(sequence.clock["physics_step"],3)
            or not _exact_int(sequence.clock["rod_tick"],12)
            or _clock_row(receipt["primed_identity"])!=sequence.clock):
        raise ValueError("contact prime must end at original SDK3/rod12")
    return dict(clock=deepcopy(sequence.clock),row_sha256=_digest(event),owner_uid=receipt["owner_uid"])


def _physical_journal(rows, *, reward_scope=None, expected_backend_sha256=None, controls=32,
                      initial_policy_version=0, profile=SMOKE_PROFILE):
    """Check existing C04 payloads, never invent Reset/setup or clock values.

    This original action schema records the nominal SDK timestamp contract.
    Its pure validator checks that recorded value; solver accumulation uses the
    original observed-clock validator. No actual-SDK-clock migration is made.
    """
    import numpy as np
    import torch
    from .tabletop_deformable_dual_evidence import _Sequence,_clock_row
    from .tabletop_deformable_bimanual_step import CLOCK_KEYS,VERSION as ACTION_VERSION
    from .tabletop_surface_contract import ACTION_CONTRACT
    from .tabletop_action24_history_owner import _action,VERSION,SOURCE
    from .tabletop_native_teacher_action_history import VERSION as V2,SOURCE as SOURCE2
    from .tabletop_deformable_r0_replay import decode_owned,_digest
    from .tabletop_stereo_rod_state import _owned_tensor,VERSION as ROD_VERSION,UNITS
    groups={op:[r["payload"] for r in rows if r["operation"]==op] for op in PRIMARY_OPERATIONS}
    measured = controls * 8
    priming=1 if profile==CONTACT_PROFILE else 0
    if profile not in (SMOKE_PROFILE,CONTACT_PROFILE):
        raise ValueError("known physical worker profile required")
    prime=_contact_prime(rows,end_sdk_step=2+priming+measured) if priming else None
    sequence=_Sequence(2 + priming + measured,12)
    if prime is not None:
        sequence.clock=deepcopy(prime["clock"])
    checked_reward_scope = _required_native_source_scope() if reward_scope is None else reward_scope
    owner=backend=nodes=previous_action=None
    try:
        def clock(value):
            if not isinstance(value,dict) or set(value)!=set(CLOCK_KEYS):
                raise ValueError("original recorded clock schema required; no SDK clock projection")
            return _clock_row(value)
        def stamp(value,endpoint):
            expected=dict(env_id=endpoint["env_id"],episode_uid=endpoint["episode_uid"],
                reset_uid=str(endpoint["reset_id"]),rod_uid=endpoint["rod_uid"],
                physics_tick=endpoint["physics_step"],solver_time_s=endpoint["solver_time_s"],
                scope=checked_reward_scope,frame="world")
            if any(type(value.get(k)) is not type(v) or value[k]!=v for k,v in expected.items()):
                raise ValueError("reward/physical endpoint clock identity differs")
        for control in range(controls):
            decision=decode_owned(groups["decision"][control]);history=decision["history"]
            action=_action(decision["sampled_action24"]);token=decision["collection_token"]
            policy=history["policy"];origin=decision["policy"]
            from .tabletop_native_teacher_source_priming import VERSION as V3,SOURCE as SOURCE3
            accepted_history=((V3,SOURCE3),) if priming else ((VERSION,SOURCE),(V2,SOURCE2))
            if ((history["version"],history["source"]) not in accepted_history
                    or priming and (history["owner_uid"]!=prime["owner_uid"]
                        or history.get("priming_row_sha256")!=prime["row_sha256"])
                    or not isinstance(history["owner_uid"],str) or not history["owner_uid"]
                    or owner is not None and history["owner_uid"]!=owner
                    or history["pre_action_sha256"]!=decision["pre_action_sha256"]
                    or not _sha(decision["pre_action_sha256"])
                    or policy["collection_token"]!=token or policy["session_id"]!=origin["session_id"]
                    or not _exact_int(policy["policy_version"],initial_policy_version)
                    or policy["model_weights_sha256"]!=origin["weights"]
                    or _digest(policy["model_identity"])!=origin["model"]
                    or not np.array_equal(_action(history["sampled_action24"]),action)
                    or previous_action is not None and history["previous_action_sha256"]!=_digest(previous_action)):
                raise ValueError("decision/history policy token, request or owner chain differs")
            owner=history["owner_uid"];sequence.action=action;sequence.substep=0
            first=None;physical_hashes=[];target_hashes=[];projection_hashes=[]
            for substep in range(8):
                index=control*8+substep;measured=index+1
                raw=groups["physical"][index];delivered=groups["delivered_exchange"][index]
                if (raw.get("kind")!="physical" or json.dumps(raw.get("exchange"),sort_keys=True,separators=(",",":"))
                        !=json.dumps(delivered,sort_keys=True,separators=(",",":"))):
                    raise ValueError("physical callback differs from its original delivered exchange")
                sample=decode_owned(raw);exchange=sample["exchange"]
                if any(packet.get(key) is not False for packet in (sample,exchange)
                        for key in ("native_qualification","teacher_admission")):
                    raise ValueError("original unqualified physical source flags required")
                trace=exchange["action_readback"]
                if (trace["version"]!=ACTION_VERSION or trace["env_ids"]!=[0]
                        or not _exact_int(trace["env_ids"][0],0)
                        or _digest(trace["request"]["action_contract"])!=_digest(ACTION_CONTRACT)):
                    raise ValueError("original action readback version/environment/contract differs")
                pre,post=(clock(exchange["action_readback"][phase]["identity"][0]) for phase in ("pre","post"))
                rod=exchange["rod"];boundary=sample["rod_boundary"];snapshot=boundary["rod_snapshot"]
                # Protected legacy helpers compare values. Reject Python bool/
                # float aliases of integer identities before those comparisons.
                for packet,keys in ((rod,("env_id","reset_id","physics_step","rod_tick")),
                        (boundary,("rod_tick",)),(snapshot,("physics_step","rod_tick"))):
                    if any(not _exact_int(packet.get(key),post[key]) for key in keys):
                        raise ValueError("physical clock identity requires exact integer fields")
                for name,key in (("env_ids","env_id"),("reset_ids","reset_id")):
                    ids=snapshot.get(name)
                    if type(ids) is not list or len(ids)!=1 or not _exact_int(ids[0],post[key]):
                        raise ValueError("physical snapshot identity requires single exact integer ID lists")
                if rod.get("clock_valid") is not True:
                    raise ValueError("physical clock validity requires original boolean true")
                if any(not _exact_int(exchange.get(key),value) for key,value in
                        dict(physics_step=measured+2+priming,policy_step=control,substep=substep).items()):
                    raise ValueError("physical SDK/control/substep requires exact integer counters")
                if sequence.clock is None:
                    if (not _exact_int(pre["env_id"],0) or not _exact_int(pre["physics_step"],2+priming)
                            or not _exact_int(pre["rod_tick"],12*priming) or pre["reset_id"]<1):
                        raise ValueError("initial measured window requires recorded SDK2/rod0 boundary")
                    sequence.clock=pre
                if first is None:first=pre
                sequence._physical(sample)  # Original clock/action/rod-boundary validator, no <=240 cap.
                if (not _exact_int(exchange["physics_step"],measured+2+priming)
                        or not _exact_int(post["rod_tick"],(measured+priming)*12)):
                    raise ValueError("physical endpoints must cover unique contiguous SDK and rod steps")
                if (snapshot["version"]!=ROD_VERSION or snapshot["units"]!=UNITS or snapshot["backend"]!="DeformX"
                        or snapshot["coordinate_frame"]!="world" or not _sha(snapshot["backend_sha256"])
                        or backend is not None and snapshot["backend_sha256"]!=backend
                        or expected_backend_sha256 is not None and snapshot["backend_sha256"]!=expected_backend_sha256):
                    raise ValueError("material readback version/units/backend differs")
                backend=snapshot["backend_sha256"]
                positions=snapshot["node_positions"]
                if (not isinstance(positions,np.ndarray) or positions.ndim!=3 or positions.shape[0]!=1
                        or positions.shape[1]<3 or positions.shape[2]!=3
                        or nodes is not None and nodes!=positions.shape[1]):
                    raise ValueError("complete stable original material-node readback topology required")
                nodes=positions.shape[1]
                shapes=dict(node_positions=(1,nodes,3),velocities=(1,nodes,3),centerline=(1,nodes,3),
                    tangent=(1,nodes,3),curvature=(1,nodes),distributed_contact=(1,nodes,3),
                    contact_valid=(1,nodes),rest_lengths_m=(1,nodes-1),deformation=(1,nodes-1))
                tensors={name:torch.from_numpy(snapshot[name]) for name in shapes}
                _owned_tensor(tensors,"node_positions",shapes["node_positions"])
                for name,shape in shapes.items():
                    _owned_tensor(tensors,name,shape,device=torch.device("cpu"),
                        dtype=torch.bool if name=="contact_valid" else tensors["node_positions"].dtype)
                if (not bool(tensors["contact_valid"].all()) or not bool((tensors["rest_lengths_m"]>0).all())
                        or any(not np.array_equal(snapshot[name],rod[name]) for name in ("node_positions","velocities","distributed_contact"))
                        or not np.array_equal(snapshot["rest_lengths_m"][0],rod["rest_lengths_m"])):
                    raise ValueError("physical raw/boundary material state differs or is unavailable")
                budget=groups["budget_consumed"][index]
                if (set(budget)!={"sdk_steps","solver_calls","internal_solver_substeps"}
                        or any(not _exact_int(budget[k],v) for k,v in dict(sdk_steps=measured,
                            solver_calls=measured,internal_solver_substeps=measured*12).items())):
                    raise ValueError("measured budget must advance one SDK step with 12 internal substeps, excluding setup")
                physical_hashes.append(_digest(sample))
                target_hashes.append(_digest(exchange["action_readback"]["submission"]["expected_backend_targets"]))
                projection_hashes.append(_digest(exchange["wrist_projection"]))
            settlement=decode_owned(groups["settlement"][control]);saved=settlement["history"]
            if (not _exact_int(settlement["result"].get("control_step"),control)
                    or not _exact_int(settlement["result"].get("physical_samples"),8)):
                raise ValueError("settlement control/sample counts require exact integers")
            sequence._end(settlement["result"])
            if (any(_digest(saved[k])!=_digest(v) for k,v in history.items())
                    or not _exact_int(saved["physical_samples"],8)
                    or saved["physical_rows_sha256"]!=physical_hashes
                    or saved["execution_targets_sha256"]!=target_hashes
                    or saved["wrist_projection_sha256"]!=projection_hashes
                    or clock(saved["endpoint"])!=sequence.clock
                    or not np.array_equal(_action(saved["issued_action24"]),action)
                    or not np.array_equal(_action(settlement["issued_actions"]),action)):
                raise ValueError("settlement physical/history/hash/request chain differs")
            metadata=settlement["post_context"]["metadata"]
            endpoint=metadata["identities"][0]
            if (len(metadata["identities"])!=1 or metadata["previous_request_token"]!=token
                    or any(type(endpoint.get(k)) is not type(sequence.clock[k]) or endpoint[k]!=sequence.clock[k]
                        for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step",
                            "rod_tick","timestamp_s","solver_time_s"))):
                raise ValueError("settlement context endpoint/token differs from physical control")
            reward=settlement["reward"]
            stamp(reward["pre_endpoint"]["stamp"],first);stamp(reward["endpoint"]["stamp"],sequence.clock)
            if reward["policy_uid"]!=origin["weights"] or reward["post_context"]["previous_request_token"]!=token:
                raise ValueError("reward policy/token differs from measured control")
            previous_action=action
        if (sequence.physical != measured or sequence.controls != controls
                or sequence.windows != controls or sequence.action is not None):
            raise ValueError("complete control/SDK physical settlement required")
        result = dict(physical_nodes=nodes,measured_sdk_steps=sequence.physical,controls=sequence.controls,
            solver_calls=sequence.physical,internal_solver_substeps=sequence.physical*12,
            first_sdk_endpoint=3+priming,last_sdk_endpoint=sequence.clock["physics_step"],
            clock_schema="original_nominal_timestamp_and_recorded_solver_time")
        if priming:
            result.update(source_priming_sdk_steps=1,source_priming_solver_calls=1,
                          source_priming_row_sha256=prime["row_sha256"])
        return result
    except (KeyError,IndexError,TypeError,AttributeError,ValueError,RuntimeError) as error:
        raise ValueError("physical journal rejected: "+str(error)) from error



def _current_recipe_context(spec, repository, package):
    name,current=worker_registered_native_package(repository,package)
    recipe=repository.native_recipe(current)
    if recipe["spec"]!=spec:
        raise ValueError("v2 artifact spec differs from actual frozen current recipe")
    return dict(package_name=name,package=deepcopy(current),recipe=recipe,
        source_contract=worker_training_source_contract(repository,current,spec))


def _audit_fresh_checkpoint(spec, repository, context, *, checkpoint_bytes, checkpoint_binding,
                            training, epoch, include_rng_digest=False):
    return _audit_role_checkpoint(spec, repository, context, checkpoint_bytes=checkpoint_bytes,
        checkpoint_binding=checkpoint_binding, training=training, epoch=epoch, expected_origin={"kind":"fresh"},
        environment_windows=1, include_rng_digest=include_rng_digest)


def _audit_role_checkpoint(spec, repository, context, *, checkpoint_bytes, checkpoint_binding,
                           training, epoch, expected_origin, environment_windows, include_rng_digest=False):
    with _cpu_reference(spec,repository,context) as (trainer,contract):
        from .tabletop_native_teacher_checkpoint import audit_checkpoint_bytes
        from .tabletop_deformable_r0_replay import _digest
        try:
            metadata=audit_checkpoint_bytes(checkpoint_bytes,trainer,contract,expected_origin=expected_origin,
                include_rng_digest=include_rng_digest)
        except RuntimeError as error:
            raise ValueError("CPU checkpoint audit rejected malformed payload: "+str(error)) from error
        if (metadata["checkpoint_sha256"]!=checkpoint_binding["sha256"]
                or metadata["counters"]!={"environment_steps":environment_windows*spec["teacher_contract"]["limits"]["controls"],
                                          "updates":training["updates"]}
                or metadata["policy_version"]!=epoch["policy_version"]
                or metadata["accepted_minibatch_updates"]!=training["accepted_minibatch_updates"]
                or metadata["policy_weights_sha256"]!=epoch["weights"]
                or _digest(metadata["identity"]["model"])!=epoch["model"]):
            raise ValueError("checkpoint policy weights/model/counters differ from original journal and report")
        return metadata


def validate_teacher_artifacts_v2(spec, output, *, repository=None, package=None):
    audit=audit_teacher_artifacts_v2(spec,output)
    context=_current_recipe_context(spec,repository,package)
    physical=_physical_journal(audit.pop("journal_rows"),controls=spec["teacher_contract"]["limits"]["controls"],
        profile=spec["teacher_contract"]["profile"])
    metadata=_audit_fresh_checkpoint(spec,repository,context,checkpoint_bytes=audit.pop("checkpoint_bytes"),
        checkpoint_binding=audit["checkpoint_binding"], training=audit["teacher_training"],
        epoch=audit["policy_epoch"])
    if metadata["checkpoint_sha256"]!=audit["artifact_sha256"]["teacher_clean.pt"]:
        raise ValueError("checkpoint bytes differ from the frozen artifact inventory")
    audit["checkpoint_audit"]=metadata
    audit.update(physical_evidence=physical,model_nodes=len(context["model_target_contract"]["target_material_ids"]),
        model_target_schema=context["model_target_contract"]["target_schema"],
        physical_to_model_snapshot_binding_verified=False,actual_setup_source_verified=False,
        cap_source_verified=False,native_ready=False,native_qualification=False)
    return audit


@contextmanager
def _cpu_reference(spec,repository,context):
    """Disposable CPU topology/Adam contract; no collector, owner or restore.

    Actual native construction still belongs after live verified Reset. This
    post-exit reference reads immutable source/config identities only.
    """
    import random
    import numpy as np
    import torch
    from .tabletop_native_teacher_profile import BackendProfile,ROOT,checked_file,digest
    from .tabletop_native_teacher_preflight import load_reset_manifest,_binding,validate_config
    from .tabletop_training_reward import load_config
    from . import tabletop_mesh_observation_source as source
    from .tabletop_surface_contract import ACTION_CONTRACT
    from .tabletop_native_teacher_model import NativeContextTeacher
    from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from .tabletop_privileged_ppo import PPOConfig
    from .tabletop_native_teacher_checkpoint import build_training_contract
    from .tabletop_native_teacher_worker_contract import (
        SOURCE_FILES, TRIPOD_SOURCE_FILES, POWER_SOURCE_FILES, CONTACT_SOURCE_FILES, training_semantic_digest, validate_profile_rollout, worker_training_target_contract)
    from scripts.run_dh116_work_packages import native_recipe_sources
    # Tie reference code to the same actual scheduler inventory. A temporary
    # source project may contain identical copies, never caller-equal hashes.
    inventory=native_recipe_sources(repository,context["package"],spec)
    actual=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    if context["source_contract"]!={"worker_source_closure":actual}:
        raise ValueError("current source inventory changed during checkpoint audit")
    for name in SOURCE_FILES:
        checked_file(ROOT/name,inventory[name])
    pin=spec["teacher_contract"];binding=pin["training_config"]
    config=_json(checked_file(binding["path"],binding["sha256"]))
    if not isinstance(config,dict) or not isinstance(config.get("preflight"),dict):
        raise ValueError("complete frozen training configuration required")
    pre=validate_config(config["preflight"])
    extra_sources = {"tripod": TRIPOD_SOURCE_FILES, "power": POWER_SOURCE_FILES}.get(pre["style"], frozenset())
    if pre["profile"]==CONTACT_PROFILE:
        extra_sources = extra_sources | CONTACT_SOURCE_FILES
    for name in extra_sources:
        checked_file(ROOT/name,inventory[name])
    if pre.get("authorization") is not None or pre.get("resource_lock") is not None:
        raise ValueError("frozen training config must not include process lease overlay")
    limits=pin["limits"]
    expected_budget=dict(max_sdk_steps=spec["budget"]["physics_steps"],max_resets=limits["max_resets"],
        max_solver_calls=limits["max_solver_calls"],max_episode_count=limits["max_episode_count"],
        wall_timeout_s=limits["inner_wall_timeout_s"])
    if (pre.get("budget")!=expected_budget
            or any(not _exact_int(pre["budget"][key],value) for key,value in expected_budget.items()
                if key!="wall_timeout_s")
            or type(pre["budget"]["wall_timeout_s"]) not in (int,float)):
        raise ValueError("frozen training process budget differs from finite recipe")
    profile=BackendProfile.load()
    manifest=load_reset_manifest(pre,project=repository.root);reset=manifest.data()
    path,_=_binding(pre["reward_config"],repository.root)
    reward=load_config(path,ppo_gamma=pre["ppo"]["gamma"])
    physical=_json(checked_file(repository.root/source.CONFIG,source.CONFIG_SHA))
    backend=dict(profile_sha256=profile.digest,binary_sha256=profile.binary_sha256,
        inventory_sha256=profile.inventory_sha256,source_tree_sha256=profile.source_tree_sha256,
        patch_sha256=profile.patch_sha256)
    physics=dict(physics_hz=physical["physics_hz"],control_hz=physical["control_hz"],
        solver_substeps=physical["rod"]["solver_substeps"],physics_config_sha256=source.CONFIG_SHA,
        action_contract_sha256=digest(ACTION_CONTRACT),target_schema=reset["target_schema"])
    semantic=training_semantic_digest(config,backend_identity=backend,
        source_identity=context["source_contract"],physics_identity=physics,project=repository.root)
    context["model_target_contract"]=worker_training_target_contract(profile,manifest)
    target=digest(context["model_target_contract"])
    expected=dict(semantic_sha256=semantic,backend_sha256=profile.binary_sha256,target_sha256=target,
        reset_sha256=manifest.sha256,reward_sha256=reward.config_sha256)
    if any(pin[key]!=value for key,value in expected.items()):
        raise ValueError("actual backend/physics/action/target/reset/reward semantic identity differs from recipe")
    mapping={"learning_rate":"lr","gamma":"gamma","gae_lambda":"lam","clip":"clip",
        "value":"value_coef","entropy":"entropy_coef","max_grad_norm":"max_grad_norm",
        "epochs":"epochs","minibatches":"minibatches","target_kl":"target_kl",
        "kl_backtrack_steps":"kl_backtrack_steps"}
    values={mapping[k]:v for k,v in reward.ppo.items() if k in mapping}
    for key,value in pre["ppo"].items():
        target_key=mapping.get(key,key)
        if target_key not in values or type(values[target_key]) is not type(value) or values[target_key]!=value:
            raise ValueError("frozen preflight/PPO/reward numeric configuration differs: "+key)
    if config["rollout_steps"] != limits["controls"]:
        raise ValueError("frozen rollout length differs from approved controls")
    validate_profile_rollout(pre["profile"],pre["style"],config["rollout_steps"], expected_budget, reward.ppo["rollout_steps"])
    ppo=PPOConfig(**values)
    py=random.getstate();numpy=deepcopy(np.random.get_state())
    try:
        with torch.random.fork_rng(devices=[]):
            model=NativeContextTeacher(config["model_config"],backend_profile=profile,context_mode=pre["profile"])
            trainer=DeformablePPOTrainer(model,ppo,rollout_steps=config["rollout_steps"],num_envs=1)
            contract=build_training_contract(trainer,scope=_required_native_source_scope(),
                reward_sha256=reward.config_sha256,target_sha256=target,reset_sha256=manifest.sha256,
                source_sha256=context["source_contract"],config_sha256=semantic)
            yield trainer,contract
    finally:
        random.setstate(py);np.random.set_state(numpy)
