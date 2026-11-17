"""Actual archived inputs only; no SDK, solver step or optimizer update."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, _digest
from go2_pvcnn.control.dh116.tabletop_deformable_privileged_teacher import load_deformable_teacher_config
from go2_pvcnn.control.dh116.tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher
from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig

ROOT = Path(__file__).resolve().parents[1]
E = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
PINCH_RECEIPT = E / ("native-attempts/flex.f7-approach-r0-native/"
    "96fc30001e0d26e7421b683fc0e18ae9240054e3b6fcffb3e010aa6d6d0d70b3/attempt-wy27jh8g/receipt.json")
PINCH_RECEIPT_SHA = "9b0a8eb4f37ce20e27d5054a4d95043e21fb99febb00128a878fffa739608ee6"


def module():
    from go2_pvcnn.control.dh116 import tabletop_actual_backend_teacher
    return tabletop_actual_backend_teacher


def config_path(style):
    return ROOT / f"configs/dh116_training/actual_measured_teacher_{style}_v1.json"


@pytest.fixture(scope="module")
def actual_first():
    raw = PINCH_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINCH_RECEIPT_SHA
    receipt = json.loads(raw)
    output = Path(receipt["output"])
    reset_raw = (output / "director_reset.json").read_bytes()
    assert hashlib.sha256(reset_raw).hexdigest() == receipt["artifact_sha256"]["director_reset.json"]
    parameter = decode_owned(json.loads(reset_raw)["payload"])
    reset = pre = None
    with (output / "physical.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("operation") == "reset":
                reset = decode_owned(row["payload"])["reset"]["rod_reset"]
            if row.get("operation") == "begin":
                pre = decode_owned(row["payload"])["pre_action_state"]
                break
    assert int(pre["common"]["valid"].sum()) == 66
    assert pre["rod_boundary"]["rod_snapshot"]["backend_sha256"] == (
        "2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec")
    return pre, reset, parameter


def legacy_tensors(pre):
    common = pre["common"]
    rod = deepcopy(pre["rod_boundary"]["rod_snapshot"])
    for name, value in rod.items():
        if isinstance(value, np.ndarray):
            rod[name] = torch.from_numpy(value.copy())
    ledger = torch.as_tensor(common["values"])[None].float()
    valid = torch.as_tensor(common["valid"])[None]
    identity = {k: rod[k] for k in ("env_ids", "episode_uids", "reset_ids", "rod_uids",
                                    "physics_step", "rod_tick", "timestamp_s")}
    mask = torch.ones((1, 41), dtype=torch.bool)
    return ledger, valid, rod, identity, mask


def test_existing_model_rejects_real_backend_without_identity_relabeling(actual_first):
    cfg, identity = load_deformable_teacher_config(
        ROOT / "configs/dh116_training/privileged_deformable_teacher_pinch_v1.json", ROOT)
    model = MeasuredCommonTeacher(cfg, identity)
    assert model.backend_sha256.startswith("c47942")
    with pytest.raises(ValueError, match="backend identity"):
        model.pack_measured(*legacy_tensors(actual_first[0]))


@pytest.fixture(scope="module")
def admitted():
    return module().load_adapter_config(config_path("pinch"), ROOT)


def test_explicit_adapter_packs_real_first_state_preserving_unknowns(actual_first, admitted):
    pre, rod, parameter = actual_first
    before = _digest((pre, rod, parameter))
    model = module().ActualBackendMeasuredTeacher(admitted)
    packed = model.pack_actual(pre, rod, parameter)
    assert packed["common371"].shape == (1, 371)
    assert int(packed["common_valid"].sum()) == 66
    assert torch.isnan(packed["common371"][~packed["common_valid"]]).all()
    assert packed["rod_nodes"].shape == (1, 41, 13)
    assert model.backend_sha256.startswith("2fea8036")
    with torch.no_grad():
        assert model.deterministic(packed).shape == (1, 24)
        assert torch.isfinite(model.value(packed)).all()
    assert _digest((pre, rod, parameter)) == before
    assert model.model_identity["native_ready"] is False
    assert model.model_identity["training_reward_qualified"] is False
    assert model.model_identity["adapter"]["evidence"]["grasp"] == "pinch"


@pytest.mark.parametrize("change", [
    "common_fill", "common_mask", "common_value", "rod_backend", "rod_nodes",
    "rod_velocity", "rod_tangent", "rod_epoch", "parameter_clock", "parameter_radius",
    "parameter_source", "parameter_derived", "parameter_fixed", "foreign_epoch",
])
def test_actual_packet_damage_is_rejected(actual_first, admitted, change):
    pre, rod, parameter = deepcopy(actual_first)
    if change == "common_fill":
        pre["common"]["values"][30] = 0.
    elif change == "common_mask":
        pre["common"]["valid"][30] = True
    elif change == "common_value":
        pre["common"]["values"][0] += .1
    elif change == "rod_backend":
        pre["rod_boundary"]["rod_snapshot"]["backend_sha256"] = "0" * 64
    elif change in ("rod_nodes", "rod_velocity", "rod_tangent"):
        field = {"rod_nodes": "node_positions", "rod_velocity": "velocities",
                 "rod_tangent": "tangent"}[change]
        pre["rod_boundary"]["rod_snapshot"][field].flat[0] += .01
    elif change == "rod_epoch":
        rod["episode_uid"] += "-foreign"
    elif change == "parameter_clock":
        parameter["identity"]["physics_step"] += 1
    elif change == "parameter_radius":
        parameter["raw_native_parameters"]["contact_radius_m"] += .001
    elif change == "parameter_fixed":
        parameter["raw_native_parameters"]["inverse_masses_kg_inv"][0, 0] = 0
    elif change == "parameter_derived":
        parameter["native_parameters"]["fields"]["contact_radius_m"]["value"] += .001
    elif change == "parameter_source":
        parameter["sources"]["source_tree_sha256"] = "0" * 64
    else:
        for packet in (pre["common"]["identity"], rod, parameter["identity"]):
            packet["episode_uid"] += "-not-bound"
    model = module().ActualBackendMeasuredTeacher(admitted)
    with pytest.raises((ValueError, KeyError)):
        model.pack_actual(pre, rod, parameter)


def test_raw_pack_entry_cannot_bypass_actual_source_checks(admitted, actual_first):
    model = module().ActualBackendMeasuredTeacher(admitted)
    with pytest.raises(ValueError, match="pack_actual"):
        model.pack_measured(*legacy_tensors(actual_first[0]))


def test_per_grasp_evidence_does_not_borrow_pinch_qualification(admitted):
    trip = module().load_adapter_config(config_path("tripod"), ROOT)
    model = module().ActualBackendMeasuredTeacher(trip)
    first = module().read_first_boundary(trip)
    packed = model.pack_actual(*first)
    assert int(packed["common_valid"].sum()) == 66
    assert trip["evidence"]["grasp"] == "tripod"
    assert trip["evidence"]["episode_exit_code"] == 1
    assert trip["evidence"]["style_coverage"] == "left_tripod_failure_prefix"
    assert model.model_identity["native_ready"] is False
    assert model.model_identity["model_identity_sha256"] != (
        module().ActualBackendMeasuredTeacher(admitted).model_identity["model_identity_sha256"])
    with pytest.raises(ValueError, match="no actual"):
        module().load_adapter_config(config_path("power"), ROOT)


@pytest.mark.parametrize("change", ["backend", "receipt", "grasp", "parameter", "projection"])
def test_config_changes_are_explicit_and_do_not_relabel_sources(tmp_path, change):
    cfg = json.loads(config_path("pinch").read_bytes())
    if change == "backend":
        cfg["backend_sha256"] = "0" * 64
    elif change == "receipt":
        cfg["evidence"]["receipt_sha256"] = "0" * 64
    elif change == "grasp":
        cfg["grasp"] = "tripod"
    elif change == "parameter":
        cfg["parameter_static_sha256"] = "0" * 64
    else:
        cfg["common_indices"].append(30)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        module().load_adapter_config(path, ROOT)


def trainer(admitted):
    return DeformablePPOTrainer(module().ActualBackendMeasuredTeacher(admitted),
        PPOConfig(epochs=1, minibatches=1), rollout_steps=2, num_envs=1)


def test_new_checkpoint_roundtrip_zero_updates_and_legacy_default_reject(tmp_path, admitted):
    from go2_pvcnn.control.dh116 import tabletop_actual_backend_checkpoint as checkpoints
    from go2_pvcnn.control.dh116 import tabletop_flexible_measured_checkpoint as old
    first, second = trainer(admitted), trainer(admitted)
    path = tmp_path / "new.pt"
    result = checkpoints.save_checkpoint(path, first)
    assert result["version"] != old.VERSION
    checkpoints.load_checkpoint(path, second)
    assert first.counters == second.counters == {"environment_steps": 0, "updates": 0}
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key])
    with pytest.raises(FileExistsError):
        checkpoints.save_checkpoint(path, first)
    with pytest.raises(ValueError):
        old.save_checkpoint(tmp_path / "old.pt", first)
    with pytest.raises(ValueError):
        old.load_checkpoint(path, first)
    assert not (tmp_path / "old.pt").exists()


def test_checkpoint_cross_grasp_and_identity_tamper_cannot_restore(tmp_path, admitted):
    from go2_pvcnn.control.dh116 import tabletop_actual_backend_checkpoint as checkpoints
    path = tmp_path / "pinch.pt"
    a = trainer(admitted)
    checkpoints.save_checkpoint(path, a)
    other = trainer(module().load_adapter_config(config_path("tripod"), ROOT))
    before = {k:v.clone() for k,v in other.model.state_dict().items()}
    with pytest.raises(ValueError, match="identity"):
        checkpoints.load_checkpoint(path, other)
    assert all(torch.equal(v, other.model.state_dict()[k]) for k,v in before.items())
    raw = torch.load(path, weights_only=True)
    raw["identity"]["model"]["adapter"]["parameter_static_sha256"] = "0" * 64
    bad = tmp_path / "bad.pt"
    torch.save(raw, bad)
    with pytest.raises(ValueError, match="identity"):
        checkpoints.load_checkpoint(bad, a)


def test_loaded_binding_cannot_be_resealed_after_source_or_config_change(admitted):
    bad = deepcopy(admitted)
    bad["parameter_source_basis"]["source_tree_sha256"] = "0" * 64
    bad["binding_sha256"] = module().digest({k:v for k,v in bad.items() if k != "binding_sha256"})
    with pytest.raises(ValueError, match="source"):
        module().ActualBackendMeasuredTeacher(bad)


def test_checkpoint_delegation_never_changes_frozen_writer_globals(tmp_path, admitted):
    from go2_pvcnn.control.dh116 import tabletop_actual_backend_checkpoint as checkpoints
    from go2_pvcnn.control.dh116 import tabletop_flexible_measured_checkpoint as old
    original = (old.VERSION, old.checkpoint_identity, dict(old.save_checkpoint.__globals__),
                dict(old.load_checkpoint.__globals__))
    item = trainer(admitted)
    path = tmp_path / "independent.pt"
    checkpoints.save_checkpoint(path, item)
    checkpoints.load_checkpoint(path, item)
    assert original == (old.VERSION, old.checkpoint_identity, old.save_checkpoint.__globals__,
                        old.load_checkpoint.__globals__)
    assert checkpoints._delegate("save_checkpoint").__code__ is old.save_checkpoint.__code__
    assert checkpoints._delegate("load_checkpoint").__code__ is old.load_checkpoint.__code__


def test_checkpoint_rejects_runtime_backend_attribute_drift(tmp_path, admitted):
    from go2_pvcnn.control.dh116 import tabletop_actual_backend_checkpoint as checkpoints
    item = trainer(admitted)
    item.model.backend_sha256 = "0" * 64
    with pytest.raises(ValueError, match="backend"):
        checkpoints.save_checkpoint(tmp_path / "wrong.pt", item)
    assert not (tmp_path / "wrong.pt").exists()


def test_delegation_preserves_closure_defaults_keyword_defaults_and_annotations(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_actual_backend_checkpoint as checkpoints
    from go2_pvcnn.control.dh116 import tabletop_flexible_measured_checkpoint as old
    def factory():
        captured = 7
        def probe(path: str, trainer="default", *, marker: int = 11) -> tuple:
            return path, trainer, marker, captured
        return probe
    probe = factory()
    monkeypatch.setattr(old, "save_checkpoint", probe)
    delegated = checkpoints._delegate("save_checkpoint")
    assert delegated.__code__ is probe.__code__
    assert delegated.__closure__ is probe.__closure__
    assert delegated.__defaults__ == probe.__defaults__
    assert delegated.__kwdefaults__ == probe.__kwdefaults__
    assert delegated.__annotations__ == probe.__annotations__
    assert delegated("p") == probe("p")


def test_old_checkpoint_and_new_checkpoint_are_mutually_rejected(tmp_path, admitted):
    from go2_pvcnn.control.dh116 import tabletop_actual_backend_checkpoint as checkpoints
    from go2_pvcnn.control.dh116 import tabletop_flexible_measured_checkpoint as old
    cfg, identity = load_deformable_teacher_config(
        ROOT / "configs/dh116_training/privileged_deformable_teacher_pinch_v1.json", ROOT)
    legacy = DeformablePPOTrainer(MeasuredCommonTeacher(cfg, identity),
        PPOConfig(epochs=1, minibatches=1), rollout_steps=2, num_envs=1)
    actual = trainer(admitted)
    old_path, new_path = tmp_path / "legacy.pt", tmp_path / "actual.pt"
    old.save_checkpoint(old_path, legacy)
    checkpoints.save_checkpoint(new_path, actual)
    with pytest.raises(ValueError, match="version|format"):
        old.load_checkpoint(new_path, legacy)
    with pytest.raises(ValueError, match="version|format"):
        checkpoints.load_checkpoint(old_path, actual)
    assert legacy.counters["updates"] == actual.counters["updates"] == 0
