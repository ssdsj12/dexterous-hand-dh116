"""Owned finish-post and original pad-source cache integration contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def original_geometry():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import _canonical_pinch_history
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchGeometry
    with _canonical_pinch_history():
        geometry = PolicyOwnedPinchGeometry(Path(__file__).resolve().parents[1], scope="cpu_fixture")
    geometry._radius = lambda exchange: {"value": .005}
    return geometry


@pytest.fixture
def actual_case(original_geometry):
    from test_dh116_measured_closure_geometry import case
    row, sim, bindings, read, hands, clock = case.__wrapped__(original_geometry)
    row["material_director"] = {"source_scope": "cpu_fixture"}
    return row, sim, bindings, read, hands, clock


def fast_owner(geometry, clock):
    from go2_pvcnn.control.dh116.tabletop_owned_memory_geometry import OwnedGeometryMode
    class Fast(OwnedGeometryMode, type(geometry)): pass
    fast = Fast.__new__(Fast)
    fast.__dict__.update(geometry.__dict__)
    fast.fast_memory = True
    fast._frozen_sources = None
    fast.radius_source = SimpleNamespace(source=SimpleNamespace(_boundary=lambda: deepcopy(clock)))
    return fast


def capture_post(geometry, actual_case, *, post=None, poses=None):
    row, sim, bindings, read, hands, clock = actual_case
    if post is None: post = row["action_readback"]["post"]
    if poses is None: poses = np.stack([h.transforms[0] for h in hands])
    return geometry.capture_from_post(row, sim=sim, bindings=bindings, read_clock=read,
                                      post=post, all_link_poses=poses)


def attach_sdf_cache(fast, actual_case, radius=.005):
    from go2_pvcnn.control.dh116.tabletop_batched_pad_sdf import BatchedPadSDF
    row, sim, bindings, read, hands, clock = actual_case
    pads = [(s, hands[s].body_names.index(name), name, fast._grids[side][name])
            for s, side in enumerate(("left", "right")) for name in fast.REQUIRED_PADS]
    bindings = {(s, name): fast.sources[side]["sdf_binding"][name]
                for s, side in enumerate(("left", "right")) for name in fast.REQUIRED_PADS}
    cache = BatchedPadSDF(pads, bindings=bindings, device=os.environ.get("DH116_PAD_SDF_TEST_DEVICE", "cpu"))
    cache.capture(np.stack([h.transforms[0] for h in hands])[None],
                  row["rod"]["node_positions"], frame=clock, radius=radius)
    fast.sdf_cache = cache
    fast.geometry_lane = 0
    fast.geometry_body_names = tuple(tuple(h.body_names) for h in hands)
    return cache


@pytest.mark.parametrize("readonly_post", [False, True])
def test_finish_post_and_previously_read_poses_preserve_audit_values_without_recapture(original_geometry, actual_case, monkeypatch, readonly_post):
    from go2_pvcnn.control.dh116.tabletop_owned_memory_geometry import owned_equal, freeze_owned
    from go2_pvcnn.control.dh116 import tabletop_measured_closure_geometry as measured
    row, sim, bindings, read, hands, clock = actual_case
    expected = original_geometry.capture(row, sim=sim, bindings=bindings, read_clock=read)
    fast = fast_owner(original_geometry, clock)
    if readonly_post: freeze_owned(row["action_readback"]["post"])
    def forbidden(): raise AssertionError("finish post / full poses must be reused")
    monkeypatch.setattr(measured, "_capture", lambda *a, **k: forbidden())
    for hand in hands:
        for name in ("get_dof_positions", "get_dof_velocities", "get_dof_position_targets", "get_link_transforms"):
            monkeypatch.setattr(hand.root_physx_view, name, forbidden)
    poses = np.stack([h.transforms[0] for h in hands])
    actual = capture_post(fast, actual_case, poses=poses)
    if readonly_post:
        assert actual["raw"]["joint_readback"]["hands"]["left"]["q"] is row["action_readback"]["post"]["hands"]["left"]["q"]
    for key in expected:
        if key != "raw_sha256": assert owned_equal(actual[key], expected[key]), key
    saved = actual["raw"]["hands"]["left"]["link_poses_xyzw"].copy()
    poses += 1
    np.testing.assert_array_equal(actual["raw"]["hands"]["left"]["link_poses_xyzw"], saved)
    assert actual["raw_sha256"] is None
    assert actual["durable_geometry_evidence"] is False


@pytest.mark.parametrize("fault", ["foreign_post", "clock", "instance", "pad_binding", "link_mapping", "joint_names", "mount_indices", "nonfinite_pose", "nonfinite_velocity", "clock_during_velocity"])
def test_reused_post_still_rejects_source_identity_mapping_and_clock_faults(original_geometry, actual_case, monkeypatch, fault):
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
    row, sim, bindings, read, hands, clock = actual_case
    fast = fast_owner(original_geometry, clock)
    post = row["action_readback"]["post"]
    if fault == "foreign_post": post = deepcopy(post)
    if fault == "clock": clock["rod_tick"] += 1
    if fault == "instance": post["hands"]["left"]["instance_object_id"] += 1
    if fault == "pad_binding": row["rod"]["mesh_bindings"]["left.finger14_link"]["sdf_sha256"] = "0"*64
    if fault == "link_mapping": hands[0].root_physx_view.link_paths[0][0] = "/Wrong/base_link"
    if fault == "joint_names": hands[0].joint_names = list(reversed(hands[0].joint_names))
    if fault == "mount_indices": monkeypatch.setattr(bindings[0].mount, "ids", [2, 3, 4])
    if fault == "nonfinite_pose": hands[0].transforms[0, 0, 0] = np.nan
    if fault == "nonfinite_velocity": hands[0].velocities[0, 0, 0] = np.nan
    if fault == "clock_during_velocity":
        def drift():
            clock["rod_tick"] += 1
            return hands[1].velocities
        hands[1].root_physx_view.get_link_velocities = drift
    with pytest.raises(SourceFailure): capture_post(fast, actual_case, post=post)


def test_source_bound_sdf_cache_preserves_original_pad_rows_and_guard_decisions(original_geometry, actual_case):
    from test_dh116_batched_pad_sdf import assert_row
    row, sim, bindings, read, hands, clock = actual_case
    expected = original_geometry.capture(row, sim=sim, bindings=bindings, read_clock=read)
    fast = fast_owner(original_geometry, clock)
    attach_sdf_cache(fast, actual_case)
    def forbidden(*a, **k): raise AssertionError("source-bound pad cache must serve samples")
    fast._sdf = forbidden
    actual = capture_post(fast, actual_case)
    assert actual["batched_sdf_used"] is True
    for key in ("passed", "failure_reasons", "source_errors", "source_complete", "terminal_reason"):
        assert actual[key] == expected[key]
    for side in ("left", "right"):
        for pad in fast.REQUIRED_PADS:
            assert_row(actual["sides"][side]["required_pad_sdf"][pad], expected["sides"][side]["required_pad_sdf"][pad], 1e-7)


@pytest.mark.parametrize("fault", ["frame", "nodes", "poses", "source_binding", "radius", "link_mapping", "body_mapping"])
def test_sdf_cache_refuses_stale_or_foreign_sources(original_geometry, actual_case, fault):
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
    row, sim, bindings, read, hands, clock = actual_case
    fast = fast_owner(original_geometry, clock)
    cache = attach_sdf_cache(fast, actual_case, radius=.0051 if fault == "radius" else .005)
    if fault == "frame": cache.frame["rod_tick"] += 1
    if fault == "nodes": row["rod"]["node_positions"][0, 0, 0] += 1e-6
    if fault == "poses": hands[0].transforms[0, 0, 0] += 1e-6
    if fault == "source_binding": cache.source_bindings[0, fast.REQUIRED_PADS[0]]["sha256"] = "0"*64
    if fault == "link_mapping": cache.links[0, fast.REQUIRED_PADS[0]] += 1
    if fault == "body_mapping": fast.geometry_body_names = tuple(tuple(reversed(h.body_names)) for h in hands)
    with pytest.raises(SourceFailure): capture_post(fast, actual_case)


def test_owned_fixed_radius_configuration_reads_once_while_audit_keeps_live_read(original_geometry, actual_case, tmp_path, monkeypatch):
    config = tmp_path / "configs/dh116_tasks/flexible_pinch_candidate_v1.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"rod": {"radius_m": .005}}))
    monkeypatch.setattr(original_geometry, "project", tmp_path)
    row, sim, bindings, read, hands, clock = actual_case
    fast = fast_owner(original_geometry, clock)
    original_read = Path.read_bytes
    reads = []
    def counted(path):
        if path == config: reads.append(path)
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", counted)
    for _ in range(2): original_geometry.capture(row, sim=sim, bindings=bindings, read_clock=read)
    assert len(reads) == 2
    for _ in range(2): capture_post(fast, actual_case)
    assert len(reads) == 3


def test_readonly_post_arrays_reuse_still_detects_metadata_change_during_velocity_read(original_geometry, actual_case):
    from go2_pvcnn.control.dh116.tabletop_owned_memory_geometry import freeze_owned
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
    row, sim, bindings, read, hands, clock = actual_case
    freeze_owned(row["action_readback"]["post"])
    fast = fast_owner(original_geometry, clock)
    def tamper():
        row["action_readback"]["post"]["hands"]["left"]["backend_source"] = "changed"
        return hands[1].velocities
    hands[1].root_physx_view.get_link_velocities = tamper
    with pytest.raises(SourceFailure, match="post changed during geometry capture"):
        capture_post(fast, actual_case)
