"""Explicit CPU input contract for the actual getter backend and measured66.

This adapter checks owned historical inputs. It does not qualify Reset mesh
synchronization, rewards, a native rollout or either hand's grasp success.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from . import tabletop_native_parameter_endpoints as endpoints
from .tabletop_deformable_privileged_teacher import load_deformable_teacher_config
from .tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher, common_selection
from .tabletop_deformable_rod_boundary import RodBoundaryState
from .tabletop_deformable_r0_replay import decode_owned, _digest
from .tabletop_teacher_mounted_wrist_source import validate_mounted_observation, _clock
from .tabletop_stereo_contract import digest, file_sha256
from scripts.replay_dh116_native_parameter_r0 import check_radius_packet

VERSION = "dh116-actual-backend-measured-teacher-cpu-v1"
CONFIG_VERSION = "dh116-actual-backend-measured-teacher-config-v1"
STYLES = ("pinch", "tripod", "power")
SCOPES = {
    "pinch": ("actual_reset_and_same_clock_measured_approach_R0_234_steps_only",
              0, "pinch_diagnostic_only"),
    "tripod": ("left_tripod_2_setup_8quiet_200approach_24observe_diagnostic_only",
               1, "left_tripod_failure_prefix"),
}
GATES = dict(native_ready=False, training_reward_qualified=False,
             reset_mesh_pose_qualified=False, teacher_admission=False)


def _pinned(path, sha):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular pinned source required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"source SHA256 mismatch: {path}")
    return raw


def _static_parameters(packet):
    return _digest({key: value for key, value in packet["raw_native_parameters"].items()
                    if key != "time_s"})


def _source_basis(source_project):
    """Reuse the complete source lock; only the declared config path relocates."""
    basis = endpoints.source_basis()
    old = str(endpoints.ROOT / endpoints.CONFIG)
    target = str(Path(source_project).resolve(strict=True) / endpoints.CONFIG)
    wanted = basis["sources"].pop(old)
    _pinned(target, wanted)
    basis["sources"][target] = wanted
    return basis


def _first(raw):
    reset = pre = None
    for line in io.BytesIO(raw):
        row = json.loads(line)
        if row.get("operation") == "reset":
            if reset is not None:
                raise ValueError("duplicate reset before first policy boundary")
            event = decode_owned(row["payload"])
            if event["reset"]["status"] != "complete":
                raise ValueError("recorded hand/rod reset transaction is incomplete")
            reset = event["reset"]["rod_reset"]
        elif row.get("operation") == "begin":
            if reset is None:
                raise ValueError("first pre-action lacks a recorded reset")
            pre = decode_owned(row["payload"])["pre_action_state"]
            break
    if pre is None:
        raise ValueError("missing first recorded pre-action boundary")
    return pre, reset


def _dependencies(project):
    names = (
        "tabletop_actual_backend_teacher.py", "tabletop_flexible_measured_common_teacher.py",
        "tabletop_deformable_privileged_teacher.py", "tabletop_deformable_rod_boundary.py",
        "tabletop_native_parameter_endpoints.py", "tabletop_deformx_parameter_snapshot.py",
        "tabletop_teacher_mounted_wrist_source.py", "tabletop_deformable_r0_replay.py",
        "tabletop_stereo_rod_state.py",
    )
    parent = Path(__file__).parent
    return {name: file_sha256(parent / name) for name in names} | {
        "scripts/replay_dh116_native_parameter_r0.py":
            file_sha256(Path(project) / "scripts/replay_dh116_native_parameter_r0.py")}


def load_adapter_config(path, project):
    """Verify a separate per-style source contract before model construction."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    cfg = json.loads(raw)
    keys = {"version", "grasp", "base_config", "base_config_sha256", "checkpoint_namespace",
            "backend_sha256", "source_project", "common_indices", "parameter_static_sha256",
            "evidence", *GATES}
    if (set(cfg) != keys or cfg["version"] != CONFIG_VERSION or cfg["grasp"] not in STYLES
            or any(cfg[key] is not value for key, value in GATES.items())
            or cfg["common_indices"] != list(common_selection())
            or cfg["backend_sha256"] != endpoints.BACKEND_SHA
            or cfg["checkpoint_namespace"] != f"actual_measured_teacher_{cfg['grasp']}_cpu_v1"):
        raise ValueError("exact separate actual-backend measured66 contract required")
    style = cfg["grasp"]
    expected_base = f"configs/dh116_training/privileged_deformable_teacher_{style}_v1.json"
    if cfg["base_config"] != expected_base:
        raise ValueError("base Teacher config belongs to another grasp")
    _pinned(Path(project) / expected_base, cfg["base_config_sha256"])
    base_cfg, base_identity = load_deformable_teacher_config(Path(project) / expected_base, project)
    evidence = cfg["evidence"]
    if style == "power" or evidence is None:
        raise ValueError("no actual Power input evidence; refuse before model/optimizer construction")
    fields = {"grasp", "receipt_path", "receipt_sha256", "physical_path", "physical_sha256",
              "parameter_path", "parameter_sha256", "episode_exit_code", "qualification_scope",
              "style_coverage"}
    if set(evidence) != fields or evidence["grasp"] != style:
        raise ValueError("per-grasp actual evidence required")
    scope, exit_code, coverage = SCOPES[style]
    if (evidence["qualification_scope"], evidence["episode_exit_code"],
            evidence["style_coverage"]) != (scope, exit_code, coverage):
        raise ValueError("actual evidence scope differs from this grasp")
    receipt = json.loads(_pinned(evidence["receipt_path"], evidence["receipt_sha256"]))
    if (receipt["qualification_scope"] != scope or receipt["exit_code"] != exit_code
            or receipt["outer_exit_code"] != exit_code
            or receipt["native_execution_performed"] is not True
            or receipt["debugger_inferior_signal"] is not None
            or receipt["native_qualification"] is not (exit_code == 0)):
        raise ValueError("recorded native receipt status/scope differs")
    output = Path(receipt["output"]).resolve(strict=True)
    if (Path(evidence["physical_path"]).resolve() != output / "physical.jsonl"
            or Path(evidence["parameter_path"]).resolve() != output / "director_reset.json"):
        raise ValueError("actual evidence files do not belong to the receipt output")
    # Failed Tripod receipts have no success artifact map. Its files are pinned
    # separately; the failure is retained rather than promoted to a passed run.
    if exit_code == 0:
        for name, key in (("physical.jsonl", "physical_sha256"),
                          ("director_reset.json", "parameter_sha256")):
            if receipt["artifact_sha256"][name] != evidence[key]:
                raise ValueError("successful receipt artifact SHA differs")
    basis = _source_basis(cfg["source_project"])
    packet = json.loads(_pinned(evidence["parameter_path"], evidence["parameter_sha256"]))
    if packet["capture_error"] is not None:
        raise ValueError("actual parameter capture failed")
    parameter = decode_owned(packet["payload"])
    if parameter["sources"] != basis or parameter["version"] != endpoints.VERSION:
        raise ValueError("actual parameter source differs from locked getter implementation")
    pre, rod = _first(_pinned(evidence["physical_path"], evidence["physical_sha256"]))
    check_radius_packet(parameter, rod)
    if _static_parameters(parameter) != cfg["parameter_static_sha256"]:
        raise ValueError("actual parameter profile differs from config")
    if packet["identity"] != parameter["identity"]:
        raise ValueError("parameter wrapper and captured clock differ")
    binding = dict(version=VERSION, config_path=str(path),
        config_sha256=hashlib.sha256(raw).hexdigest(), config=cfg,
        grasp=style, evidence=deepcopy(evidence), backend_sha256=endpoints.BACKEND_SHA,
        backend_path=str(endpoints.BACKEND), parameter_static_sha256=cfg["parameter_static_sha256"],
        parameter_source_basis=basis, base_config=base_cfg, base_identity=base_identity,
        common_indices=list(common_selection()), implementation=_dependencies(project), **GATES)
    binding["binding_sha256"] = digest(binding)
    # Check the real source reconstruction before admitting even the CPU model.
    _actual_inputs(binding, pre, rod, parameter)
    return binding


def _check_binding(binding):
    if (binding.get("version") != VERSION or binding.get("backend_sha256") != endpoints.BACKEND_SHA
            or binding.get("common_indices") != list(common_selection())
            or any(binding.get(key) is not value for key, value in GATES.items())
            or binding.get("binding_sha256") != digest({
                key:value for key,value in binding.items() if key != "binding_sha256"})):
        raise ValueError("actual backend model binding differs")
    project = Path(__file__).resolve().parents[4]
    config = json.loads(_pinned(binding["config_path"], binding["config_sha256"]))
    if (config != binding["config"] or binding["evidence"] != config["evidence"]
            or binding["grasp"] != config["grasp"]
            or binding["parameter_static_sha256"] != config["parameter_static_sha256"]
            or binding["implementation"] != _dependencies(project)
            or binding["parameter_source_basis"] != _source_basis(config["source_project"])):
        raise ValueError("loaded model source or configuration binding changed")


class _OwnedClock:
    def __init__(self, clock):
        self._clock = deepcopy(clock)

    def clock(self):
        return deepcopy(self._clock)


def _actual_inputs(binding, pre, rod, parameter):
    _check_binding(binding)
    common = pre["common"]
    validate_mounted_observation(common)
    values, valid = common["values"], common["valid"]
    wanted = np.zeros(371, dtype=bool)
    wanted[list(common_selection())] = True
    if not np.array_equal(valid, wanted):
        raise ValueError("exact original measured66 validity required")
    clock = _clock(common["identity"])
    keys = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid", "physics_step",
            "rod_tick", "timestamp_s", "solver_time_s", "clock_valid")
    if any(rod.get(key) != clock[key] for key in keys):
        raise ValueError("pre-action and raw rod boundary clock differ")
    if (parameter["version"] != endpoints.VERSION
            or parameter["sources"] != binding["parameter_source_basis"]
            or parameter["new_sdk_steps"] != 0 or parameter["new_solver_steps"] != 0
            or parameter["task_physics_qualified"] is not False
            or parameter["teacher_admission"] is not False
            or _static_parameters(parameter) != binding["parameter_static_sha256"]):
        raise ValueError("actual parameter source/profile differs")
    check_radius_packet(parameter, rod)
    rebuilt = RodBoundaryState(_OwnedClock({key:rod[key] for key in keys}),
        binding["backend_path"], binding["backend_sha256"]).capture(rod)
    if _digest(pre["rod_boundary"]) != _digest(rebuilt):
        raise ValueError("rod snapshot differs from actual raw boundary reconstruction")
    snapshot = rebuilt["rod_snapshot"]
    dtype = torch.float32
    ledger = torch.as_tensor(values.copy(), dtype=dtype)[None]
    validity = torch.as_tensor(valid.copy(), dtype=torch.bool)[None]
    rod_tensor = deepcopy(snapshot)
    for key, value in rod_tensor.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            value = torch.as_tensor(value).clone()
            rod_tensor[key] = value.to(dtype=dtype) if value.is_floating_point() else value
    identity = {key:deepcopy(snapshot[key]) for key in (
        "env_ids", "episode_uids", "reset_ids", "rod_uids", "physics_step", "rod_tick", "timestamp_s")}
    # All 41 original nodes must be finite, with valid material rest geometry;
    # no missing-node mask can hide a bad source. Existing pack checks features.
    mask = torch.ones((1, 41), dtype=torch.bool)
    return ledger, validity, rod_tensor, identity, mask


def read_first_boundary(binding):
    """Replay only the bound first pre-action; immutable file SHA is checked."""
    _check_binding(binding)
    evidence = binding["evidence"]
    pre, rod = _first(_pinned(evidence["physical_path"], evidence["physical_sha256"]))
    parameter = decode_owned(json.loads(_pinned(evidence["parameter_path"],
                                               evidence["parameter_sha256"]))["payload"])
    return pre, rod, parameter


class ActualBackendMeasuredTeacher(MeasuredCommonTeacher):
    """Same network and algorithms; explicit actual-backend identity only."""
    def __init__(self, binding):
        _check_binding(binding)
        super().__init__(binding["base_config"], binding["base_identity"])
        self._actual_binding = deepcopy(binding)
        parent = self.model_identity
        self.backend_sha256 = binding["backend_sha256"]
        identity = dict(parent, policy_version=VERSION,
            checkpoint_namespace=binding["config"]["checkpoint_namespace"],
            backend_sha256=self.backend_sha256, adapter=deepcopy(binding),
            parent_measured_identity_sha256=parent["model_identity_sha256"], **GATES)
        identity.pop("model_identity_sha256")
        self._model_identity = dict(identity, model_identity_sha256=digest(identity))

    def pack_measured(self, *args, **kwargs):
        raise ValueError("actual backend requires pack_actual with complete owned source packets")

    def pack_actual(self, pre_action, raw_rod, parameters):
        args = _actual_inputs(self._actual_binding, pre_action, raw_rod, parameters)
        return super().pack_measured(*args)
