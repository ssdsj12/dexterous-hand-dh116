"""Offline original control-cost arithmetic, never an R0 reward or admission.

Only owned JSON bytes are consumed. No policy, physical owner, optimizer or
native backend is constructed. The first previous action remains unknown:
historical manual episodes did not record TensorTask's previous_actions reset.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .linear_coupling import ACTIVE, JOINTS
from .tabletop_joint_target_readback import WRIST_LINEAR, WRIST_ANGULAR
from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_deformable_bimanual_step import VERSION as ACTION_VERSION
from .tabletop_deformable_dual_evidence import _Sequence, VERSION as LEDGER_VERSION
from .tabletop_deformable_r0_replay import decode_owned, _digest

ROOT = Path(__file__).resolve().parents[4]
VERSION = "dh116-readonly-control-cost-evaluation-v1"
CONTRACT = "configs/dh116_diagnostics/readonly_control_cost_v1.json"
CONTRACT_SHA256 = "5f77d8ad37d1d3edbf98b6e2401b8b95a595647cbea69d16a93713400316bbd7"
TERMS = ("time", "action_rate", "velocity_excess")
VELOCITY_SEMANTICS = "max_abs_full_17DOF_per_hand_numeric_mixed_m_per_s_and_rad_per_s"
DT = 1/480


def _require(value, message):
    if not value:
        raise ValueError(message)


def load_contract(project=ROOT):
    project = Path(project)
    try:
        raw = (project/CONTRACT).read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == CONTRACT_SHA256, "contract SHA differs")
        contract = json.loads(raw)
        for path, sha in contract["source_sha256"].items():
            _require(hashlib.sha256((project/path).read_bytes()).hexdigest() == sha,
                     "frozen formula/parser source SHA differs: "+path)
    except OSError as error:
        raise ValueError("contract/source file unavailable") from error
    return contract


def _array(value, shape, label):
    a = np.asarray(value)
    _require(a.shape == shape and a.dtype.kind == "f" and np.isfinite(a).all(),
             "finite complete "+label+" required")
    return a


def formula_components(action, previous_action, joint_velocity):
    """Formula-only numbers; qd intentionally keeps the original mixed units."""
    action = _array(action, (1,24), "action24")
    _require(np.all(abs(action) <= 1), "normalized action24 required")
    qd = _array(joint_velocity, (2,17), "full joint velocity")
    rate = None
    if previous_action is not None:
        previous_action = _array(previous_action, (1,24), "previous action24")
        _require(np.all(abs(previous_action) <= 1), "normalized previous action24 required")
        rate = -.5*float(np.mean(np.clip((action-previous_action)/2,-1,1)**2))*DT
    speed = float(abs(qd).max())
    normalized = max(0.,speed-3.)/3.
    _require(normalized <= math.sqrt(np.finfo(np.float64).max),
             "velocity square numeric overflow; source cannot be evaluated")
    return dict(time=-.1*DT, action_rate=rate,
                velocity_excess=-.1*min(normalized**2,25.)*DT)


def _hands(snapshot, common):
    _require(set(snapshot["hands"]) == {"left","right"}, "both hand owners required")
    source_packets = common["source_packets"]
    _require(len(source_packets) == 2 and {p["side"] for p in source_packets} == {"left","right"},
             "exactly two unique hand sources required")
    packets = {p["side"]: p for p in source_packets}
    _require(set(packets) == {"left","right"}, "both pre-action sources required")
    owners = []
    for side in ("left","right"):
        hand, packet = snapshot["hands"][side], packets[side]
        names = hand["joint_names"]
        _require(len(names) == len(set(names)) == 17
                 and set(names) == set((*JOINTS,*WRIST_LINEAR,*WRIST_ANGULAR))
                 and names == packet["joint_names"], "original complete named 17DOF required")
        _require(hand["prim_paths"] == [packet["articulation_path"]]
                 and hand["row_indices"] == [0] and hand["mounted"] is True,
                 "same measured mounted hand path/row required")
        _require(hand["active_indices"] == [names.index(n) for n in ACTIVE],
                 "original active indices required")
        _require(np.array_equal(hand["dof_types"],np.asarray(packet["dof_types"])[None]),
                 "pre-action DOF type source differs")
        for name in ("q","qd","backend_targets"):
            _array(hand[name],(1,17),name)
        ids = tuple(hand[k] for k in ("instance_object_id","view_object_id"))
        _require(all(type(v) is int and v > 0 for v in ids), "explicit actual owner IDs required")
        owners.append(ids)
    _require(all(owners[0][i] != owners[1][i] for i in (0,1)),
             "left/right must have distinct actual owners")
    return snapshot["hands"]


def _physical_source(exchange, common, last_hands, first):
    trace = exchange["action_readback"]
    _require(trace["version"] == ACTION_VERSION and trace["env_ids"] == [common["identity"]["env_id"]],
             "actual action readback version/environment differs")
    pre, post = [_hands(trace[p],common) for p in ("pre","post")]
    for side in ("left","right"):
        for key in ("instance_object_id","view_object_id"):
            _require(pre[side][key] == post[side][key], "pre/post owner differs")
    if last_hands is not None:
        _require(_digest(pre) == _digest(last_hands), "pre/post full hand continuity differs")
    if first:
        for packet in common["source_packets"]:
            hand = pre[packet["side"]]
            for name in ("q","qd"):
                _require(np.array_equal(hand[name][0],packet[name]),
                         "first actual pre-state differs from pre-action measurement")
    request = trace["request"]
    _require(request["action_contract"] == ACTION_CONTRACT, "original action contract differs")
    requested = _array(request["requested_finger_targets_rad"],(1,12),"requested finger targets")
    for hi,side in enumerate(("left","right")):
        hand = post[side]; names = hand["joint_names"]
        target = trace["submission"]["expected_backend_targets"][side]
        ids = [names.index(n) for n in (*ACTIVE,*WRIST_LINEAR,*WRIST_ANGULAR)]
        _require(target["joint_indices"] == ids, "full submitted finger/mount indices required")
        values = _array(target["values"],(1,12),"submitted targets")
        _require(np.allclose(hand["backend_targets"][:,ids],values,rtol=0,atol=1e-7),
                 "issued targets differ from actual backend readback")
        _require(np.allclose(requested[:,hi*6:hi*6+6].astype(np.float32),values[:,:6],
                            rtol=0,atol=1e-7), "requested and submitted finger action differ")
    return post


def _summary(samples, term):
    known = [r["components"][term] for r in samples if r["components"][term] is not None]
    observed = math.fsum(known) if known else None
    return dict(valid_samples=len(known), expected_samples=len(samples), observed_sum=observed,
                contribution_sum=observed if len(known)==len(samples) else None)


def _json(raw):
    def invalid(value):
        raise ValueError("nonstandard JSON constant "+value)
    def unique(items):
        result = {}
        for k,v in items:
            _require(k not in result, "duplicate JSON key")
            result[k]=v
        return result
    return json.loads(raw,parse_constant=invalid,object_pairs_hook=unique)


def evaluate_bytes(raw, *, expected_sha256, grasp, project=ROOT):
    """Hash and parse the SAME bytes, rejecting missing or unsettled windows."""
    _require(grasp in ("pinch","tripod","power"), "explicit original grasp required")
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "ledger SHA differs")
    contract = load_contract(project)
    _require(raw and raw.endswith(b"\n"), "complete newline-terminated ledger required")
    lines = iter(raw.splitlines())
    header = _json(next(lines))
    _require(set(header)=={"version","physics_budget_end","solver_substeps"}
             and header["version"]==LEDGER_VERSION, "original journal header required")
    sequence = _Sequence(header["physics_budget_end"],header["solver_substeps"])
    previous_action = None
    common = action = None
    last_hands = None
    samples, controls, window = [], [], []
    controller_reset = None
    backend = None
    try:
        for line in lines:
            entry = decode_owned(_json(line))
            _require(set(entry)=={"operation","payload"}, "exact journal entry required")
            op, payload = entry["operation"],entry["payload"]
            sequence.accept(op,payload)
            if op=="reset":
                controller_reset=deepcopy(payload["reset"]["owner_resets"]["controller"])
                backend=payload["rod_boundary"]["rod_snapshot"]["backend_sha256"]
                _require(isinstance(backend,str) and len(backend)==64, "explicit historical backend SHA required")
            elif op=="begin":
                common=payload["pre_action_state"]["common"]
                action=_array(payload["action"],(1,24),"accepted action24").copy()
                window=[]
                _require(payload["pre_action_state"]["rod_boundary"]["rod_snapshot"]["backend_sha256"]==backend,
                         "pre-action backend identity differs")
            elif op=="physical":
                x=payload["exchange"]
                _require(payload["rod_boundary"]["rod_snapshot"]["backend_sha256"]==backend,
                         "physical backend identity differs")
                hands=_physical_source(x,common,last_hands,not window)
                qd=np.concatenate([hands[s]["qd"] for s in ("left","right")],axis=0)
                row=dict(physics_step=x["physics_step"],rod_tick=x["rod"]["rod_tick"],
                    policy_step=x["policy_step"],substep=x["substep"],
                    actual_max_abs_joint_velocity_numeric=float(abs(qd).max()),
                    components=formula_components(action,previous_action,qd),
                    guard_failure=x["guard_failure"],training_reward=None)
                window.append(row);samples.append(row);last_hands=deepcopy(hands)
            elif op=="end":
                controls.append(dict(control_index=payload["control_step"],physical_samples=len(window),
                    complete_eight_sample_window=len(window)==8,
                    terminal_reason=payload["termination_reason"],action=action.tolist(),
                    previous_action=None if previous_action is None else previous_action.tolist(),
                    previous_action_source="unavailable_at_historical_reset" if previous_action is None
                      else "prior_accepted_begin_and_every_actual_request_in_this_same_episode",
                    rows=window,training_reward=None,**{t:_summary(window,t) for t in TERMS}))
                previous_action=action.copy()
        state=sequence.summary()
        _require(state["reset_recorded"] and controls and not state["unsettled_control_window"],
                 "complete ledger with explicitly settled windows required")
    except (KeyError,IndexError,TypeError) as error:
        raise ValueError("missing or malformed recorded source") from error
    return dict(version=VERSION,scope="offline_original_formula_components_not_training_reward",
        grasp=grasp,status="evaluated",ledger_sha256=expected_sha256,contract_sha256=CONTRACT_SHA256,
        frozen_source_sha256=contract["source_sha256"],
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        historical_backend_sha256=backend,historical_controller_reset=controller_reset,
        journal=state,physical_samples=len(samples),controls=controls,
        totals={t:_summary(samples,t) for t in TERMS},
        velocity_semantics=VELOCITY_SEMANTICS,
        initial_previous_action_source=None,first_action_rate_missing=True,
        terminal_mask_scope="recorded_finite_rows_only_not_reconstruction_of_TensorTask_failure_masks",
        original_failure_penalty_reconstructed=False,
        positive_grasp_terms_included=False,geometry_contact_inferred=False,
        training_reward=None,new_physics_steps=0,optimizer_updates=0,
        native_qualification=False,teacher_admission=False,
        mesh_reset_qualified=False,solver_health_qualified=False,
        frozen_task_mutated=False)


def evaluate_archive(grasp, *, project=ROOT, input_path=None):
    contract=load_contract(project)
    _require(grasp in contract["archives"], "original grasp required")
    archive=contract["archives"][grasp]
    if archive is None:
        _require(input_path is None, "no pinned archive for this grasp")
        return dict(version=VERSION,grasp=grasp,status="unavailable",reason="no_pinned_actual_ledger",
            physical_samples=None,totals=None,training_reward=None,new_physics_steps=0,
            optimizer_updates=0,native_qualification=False,teacher_admission=False)
    path=Path(archive["path"] if input_path is None else input_path)
    result=evaluate_bytes(path.read_bytes(),expected_sha256=archive["sha256"],grasp=grasp,project=project)
    result["ledger_path"]=str(path.resolve())
    result["archive_identity"]="exact_config_pinned_historical_bytes"
    return result
