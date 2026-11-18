import hashlib

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_rod_boundary import RodBoundaryState
from test_dual_hand_rod_runtime import runtime, reset


def setup(monkeypatch, tmp_path):
    rod, _, _, poses, centers = runtime(monkeypatch, pairs=True)
    reset_row = reset(rod, poses)
    artifact = tmp_path / "fixture.so"
    artifact.write_bytes(b"CPU-only fake solver evidence")
    provider = RodBoundaryState(rod, artifact, hashlib.sha256(artifact.read_bytes()).hexdigest())
    return provider, rod, reset_row, poses, centers, artifact


def test_reset_retains_geometry_but_contact_is_unknown(monkeypatch, tmp_path):
    provider, rod, reset_row, _, _, _ = setup(monkeypatch, tmp_path)
    packet = provider.capture(reset_row)
    state = packet["rod_snapshot"]
    assert packet["rod_uid"] == rod.rod_uid and packet["rod_tick"] == 0
    assert state["env_ids"] == [7] and state["rod_uids"] == [rod.rod_uid]
    assert state["physics_step"] == 2 and not state["contact_valid"].any()
    assert torch.isnan(state["distributed_contact"]).all()
    assert state["node_positions"].shape == (1, 3, 3)
    assert not packet["native_qualified"]


def test_completed_step_keeps_measured_forces_but_rejects_stale_reset(monkeypatch, tmp_path):
    provider, rod, reset_row, poses, centers, _ = setup(monkeypatch, tmp_path)
    step = rod.step(poses, centers, physics_step=3)
    with pytest.raises(ValueError, match="clock"):
        provider.capture(reset_row)
    packet = provider.capture(step)
    assert packet["rod_snapshot"]["distributed_contact"][0, 0, 2] == 3.
    assert packet["rod_snapshot"]["contact_valid"].all()
    packet["rod_snapshot"]["node_positions"][:] = 10
    assert provider.capture(step)["rod_snapshot"]["node_positions"].max() < 1


def test_reset_must_not_advertise_force(monkeypatch, tmp_path):
    provider, _, row, _, _, _ = setup(monkeypatch, tmp_path)
    row["distributed_contact"] = row["node_positions"].copy()
    with pytest.raises(ValueError, match="contact"):
        provider.capture(row)


def test_backend_bytes_rechecked(monkeypatch, tmp_path):
    provider, _, row, _, _, artifact = setup(monkeypatch, tmp_path)
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        provider.capture(row)
