"""Denied-by-default CPU audit for independent flexible Teacher rollouts.

There is no launch path or native qualification verifier in this module.
Scoped receipts document gaps; their boolean claims cannot grant admission.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .tabletop_deformable_grasp_task import GRASPS
from .tabletop_deformable_privileged_teacher import load_deformable_teacher_config
from .tabletop_deformable_privileged_ppo import verify_base_ppo_source
from .tabletop_deformable_task_budget import plan_flexible_experiment
from .tabletop_teacher_preaction_readback import require_teacher_common, audit_legacy_pinch

REQUEST_VERSION = "dh116-flexible-rollout-admission-request-v1"
VERSION = "dh116-flexible-rollout-admission-audit-v1"
ROLES = {"preaction_inventory", "r0_cpu", "pose_cpu", "pair_calibration"}
REQUIRED = (
    "pre_action_common_371d_incomplete", "reset_source_qualification_missing",
    "issued_action24_source_qualification_missing", "same_clock_rod_source_qualification_missing",
    "same_rod_dual_hand_physics_qualification_missing", "flexible_r0_native_qualification_missing",
    "explicit_native_budget_approval_missing", "deformable_rollout_qualification_verifier_unavailable",
)


def _bound(path, sha):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != sha:
        raise ValueError(f"SHA256 mismatch: {path}")
    return json.loads(raw), dict(path=str(path), sha256=actual, bytes=len(raw))


def _receipt(ref):
    if (not isinstance(ref, dict) or set(ref) != {"role","styles","path","sha256"}
            or ref["role"] not in ROLES or not isinstance(ref["styles"], list)
            or not ref["styles"] or len(set(ref["styles"])) != len(ref["styles"])
            or any(s not in GRASPS for s in ref["styles"])):
        raise ValueError("explicit scoped receipt role/style/path/SHA256 required")
    data, binding = _bound(ref["path"], ref["sha256"])
    role = ref["role"]
    summary = {}
    if role == "preaction_inventory":
        if data.get("version") != "dh116-teacher-legacy-source-inventory-v1" or ref["styles"] != ["pinch"]:
            raise ValueError("original Pinch source inventory required")
        # Replay the original ledger, not its claimed qualification booleans.
        checked = {}
        for name, source in data["inputs"].items():
            raw = Path(source["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise ValueError("inventory source SHA256 differs")
            checked[name] = raw
        raw = checked["ledger"]
        replay = audit_legacy_pinch([json.loads(line) for line in raw.splitlines() if line.strip()])
        if any(data.get(k) != replay[k] for k in ("rows","admitted_371d_channels","unknown_indices")):
            raise ValueError("inventory differs from actual source replay")
        summary = {k: replay[k] for k in ("rows","admitted_371d_channels")}
    elif role == "r0_cpu":
        if data.get("scope") != "cpu_fixtures_only_no_native_or_training":
            raise ValueError("explicit CPU-only R0 scope required")
        observed = {r["grasp"] for r in data.get("receipts", [])}
        if observed and not set(ref["styles"]).issubset(observed):
            raise ValueError("R0 receipt belongs to another grasp")
        summary = dict(fixture_settlements=len(data.get("receipts", [])))
    elif role == "pose_cpu":
        if not all(style in data for style in ref["styles"]):
            raise ValueError("pose receipt does not contain the declared style")
        summary = dict(scope=data.get("scope"), native_recipe_created=data.get("native_recipe_created"))
    else:
        if data.get("scope") != "one_free_rod_one_table_sdf_pair_capture_calibration":
            raise ValueError("only the isolated one-rod/table calibration scope is supported")
        summary = dict(scope=data["scope"], not_a_dual_hand_grasp=True)
    return dict(**binding, role=role, styles=list(ref["styles"]), summary=summary,
                grants_native_qualification=False)


def _common_gate(packet):
    try:
        packet = deepcopy(packet) if isinstance(packet, dict) else {}
        if "valid" in packet:
            if not isinstance(packet["valid"], list) or any(type(v) is not bool for v in packet["valid"]):
                raise ValueError("explicit per-field boolean source validity required")
            packet["valid"] = np.asarray(packet["valid"], dtype=bool)
        if "values" in packet:
            packet["values"] = np.asarray(packet["values"], dtype=float)
        require_teacher_common(packet)
    except (ValueError, TypeError, KeyError) as error:
        return dict(passed=False, reason=str(error))
    return dict(passed=True, reason=None)


def audit_rollout_admission(project, request):
    project = Path(project)
    if (not isinstance(request, dict) or set(request) != {"version","receipts","common_packets"}
            or request["version"] != REQUEST_VERSION or not isinstance(request["receipts"], list)
            or not isinstance(request["common_packets"], dict)
            or any(style not in GRASPS for style in request["common_packets"])):
        raise ValueError("exact admission request schema required; no self-attested approval keys")
    verify_base_ppo_source()
    scoped = [_receipt(ref) for ref in request["receipts"]]
    budget_path = project / "configs/dh116_training/deformable_task_budget_examples_v2.json"
    budget = json.loads(budget_path.read_bytes())["plans"]["teacher_update_smoke"]
    plan = plan_flexible_experiment(budget)
    styles = {}
    for style in GRASPS:
        path = project / f"configs/dh116_training/privileged_deformable_teacher_{style}_v1.json"
        config, identity = load_deformable_teacher_config(path, project)
        gate = _common_gate(request["common_packets"].get(style))
        reasons = list(REQUIRED)
        if gate["passed"]:
            reasons.remove("pre_action_common_371d_incomplete")
        if plan["setup_physics_steps_per_env"] is None:
            reasons.append("setup_physics_budget_unknown")
        if plan["measured_solver_substeps_per_env"] is None:
            reasons.append("solver_substep_budget_unknown")
        styles[style] = dict(
            admitted=False, blocking_reasons=reasons, common_source_gate=gate,
            checkpoint_namespace=config["checkpoint_namespace"], model_identity=identity,
            scoped_receipt_sha256=[r["sha256"] for r in scoped if style in r["styles"]],
            configuration_native_ready=config["native_ready"])
    files = [Path(__file__), Path(__file__).with_name("tabletop_teacher_preaction_readback.py"),
             Path(__file__).with_name("tabletop_deformable_task_budget.py"),
             Path(__file__).with_name("tabletop_privileged_ppo.py"),
             Path(__file__).with_name("tabletop_deformable_privileged_ppo.py"),
             Path(__file__).with_name("tabletop_deformable_privileged_teacher.py"),
             Path(__file__).with_name("tabletop_deformable_grasp_task.py"), budget_path,
             project / "scripts/audit_dh116_deformable_rollout_admission.py"]
    return dict(version=VERSION, scope="read_only_admission_audit_no_native_qualification",
        styles=styles, scoped_receipts=scoped, smoke_budget=plan,
        planned_control_transitions=plan["environments"]*plan["control_steps_per_env"],
        planned_updates_are_not_executed=True,
        implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        gae_contract=dict(done="terminated OR truncated", bootstrap="NOT terminated; same episode endpoint required",
                          carry="NOT done; never crosses reset or truncation"),
        policy_instances_created=0, optimizer_instances_created=0, new_physics_steps=0, optimizer_updates=0)


def run_audit(project, request_path, request_sha256, output):
    request, binding = _bound(request_path, request_sha256)
    result = audit_rollout_admission(project, request)
    result["request"] = binding
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    # A completed audit can correctly deny admission. This is not training failure.
    return 3


def _endpoint(value):
    keys = {"env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s","phase"}
    if not isinstance(value, dict) or set(value) != keys or value["phase"] != "post_action_pre_reset":
        raise ValueError("explicit post-action pre-reset endpoint required")
    if any(type(value[k]) is not int or value[k] < 0 for k in ("env_id","reset_id","physics_step","rod_tick")):
        raise ValueError("endpoint requires integer clock and environment identity")
    if any(not isinstance(value[k], str) or not value[k] for k in ("episode_uid","rod_uid")):
        raise ValueError("endpoint requires episode and rod identity")
    time = value["timestamp_s"]
    if type(time) not in (int,float) or not math.isfinite(time) or not math.isclose(
            time, value["physics_step"]/480, abs_tol=1e-9, rel_tol=0):
        raise ValueError("endpoint timestamp differs from physics clock")
    return deepcopy(value)


def gae_boundary(terminated, truncated, *, endpoint_identity, next_value_identity=None):
    """Return only mask semantics, never a reward, value, or admission decision."""
    if type(terminated) is not bool or type(truncated) is not bool or (terminated and truncated):
        raise ValueError("explicit mutually exclusive terminated/truncated flags required")
    endpoint = _endpoint(endpoint_identity)
    if not terminated or next_value_identity is not None:
        if _endpoint(next_value_identity) != endpoint:
            raise ValueError("next value must use the same pre-reset endpoint")
    done = terminated or truncated
    return dict(done=done, truncated=truncated, bootstrap_mask=not terminated, gae_carry=not done)
