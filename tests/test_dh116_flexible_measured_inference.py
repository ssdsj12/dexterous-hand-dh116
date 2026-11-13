"""Replay a real native pre-action packet without Isaac or optimizer execution."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_flexible_measured_inference as inference
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

ROOT = Path(__file__).resolve().parents[1]
LEDGER = Path(
    "/home/xk/.local/share/dh116-autonomous-3c80166/evidence/"
    "approved-dual-pinch-zero-v4-20261113-utbAE0/native-runs/"
    "flex.f7-dual-pinch-zero-interval-v4-native/attempt-fz50kogx/native/physical.jsonl"
)
LEDGER_SHA = "ead0dbf2c15b83864a1c440848ed9cf159b86c88a6ade0e4275208efebcf778c"


@pytest.fixture(scope="module")
def preaction():
    with LEDGER.open("rb") as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == LEDGER_SHA
    with LEDGER.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("operation") == "begin":
                return decode_owned(row["payload"])["pre_action_state"]
    pytest.fail("native begin row missing")


def make(grasp="pinch"):
    return inference.NativeBoundaryDiagnosticTeacher.from_project(ROOT, grasp)


def test_actual_native_reset_boundary_packs_66_and_runs_24d_tanh(preaction):
    model = make()
    state = model.pack_boundary(preaction)
    assert state["rod_nodes"].shape == (1, 41, 13)
    assert state["common_valid"].sum().item() == 66
    assert torch.isnan(state["common371"][~state["common_valid"]]).sum().item() == 305
    with torch.no_grad():
        action = model.deterministic(state)
        value = model.value(state)
    assert action.shape == (1, 24) and (action.abs() <= 1).all()
    assert torch.isfinite(action).all() and value.shape == (1,) and torch.isfinite(value).all()
    identity = model.model_identity
    assert identity["backend_sha256"] == inference.BACKEND_SHA
    assert identity["native_ready"] is identity["training_reward_qualified"] is False
    assert identity["grasp"] == "pinch"


@pytest.mark.parametrize("change", ["time", "reset", "rod_uid", "run", "backend", "legacy_claim", "invented_value"])
def test_boundary_rejects_mixed_sources(preaction, change):
    data = deepcopy(preaction)
    packet = data["rod_boundary"]
    snapshot = packet["rod_snapshot"]
    if change == "time":
        snapshot["timestamp_s"] += .01
    elif change == "reset":
        snapshot["reset_ids"][0] += 1
    elif change == "rod_uid":
        packet["rod_uid"] = "foreign-rod"
    elif change == "run":
        packet["run_uuid"] = "foreign-run"
    elif change == "backend":
        snapshot["backend_sha256"] = "b" * 64
    elif change == "legacy_claim":
        data["teacher_admission"] = True
    else:
        data["common"]["values"][64] = 0
    with pytest.raises(ValueError):
        make().pack_boundary(data)


def test_configs_keep_three_independent_identities_without_mislabeling_pinch():
    models = [make(grasp) for grasp in ("pinch", "tripod", "power")]
    assert len({m.model_identity["model_identity_sha256"] for m in models}) == 3
    assert len({m.model_identity["checkpoint_namespace"] for m in models}) == 3
