"""Three-pad distance from locked hardware SDFs and CPU SDK-shaped readbacks.

These fixtures are synthesized CPU source-interface data, not new native runs.
The two legacy finger pads alone must not satisfy the Tripod source contract.
"""
from copy import deepcopy
from pathlib import Path
import importlib
import numpy as np
import pytest
import torch

from test_native_teacher_context import backend_profile
from test_native_teacher_policy_pinch_distance import sdk_templates, actual_source_case
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope

ROOT = Path(__file__).resolve().parents[1]
MODULE = "go2_pvcnn.control.dh116.tabletop_native_teacher_policy_tripod_distance"


def tripod_case(monkeypatch, tmp_path, profile, templates, *, visible_middle=True):
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES
    c, old, ep, rod, hands, sim, rows = actual_source_case(
        monkeypatch, tmp_path, profile, templates)
    mesh_bindings = deepcopy(rod.bindings)
    for hand, side, token in zip(hands, ("left", "right"), ("a", "b"), strict=True):
        for name in ("base_link", "finger52_link"):
            key = side + "." + name
            mesh_bindings[key] = dict(mesh_bindings[side + ".finger14_link"],
                body_name=name, body_index=hand.body_names.index(name),
                prim_path=hand.root_physx_view.link_paths[0][hand.body_names.index(name)],
                capture_sha256=token*64, sdf_sha256=token*64, cooked_geometry_sha256=token*64)
    for row in mesh_bindings.values():
        row.update(frame="PhysX_link_origin", unit="m",
                   source_usd_sha256=old.distance.geometry.sources[row["side"]]["source_sha256"]["usd"])
    from scripts.screen_dh116_tripod_supported_reset import scene_request
    import json
    _, task = scene_request()
    config = tmp_path / "tripod_task.json"
    config.write_text(json.dumps(task))
    import hashlib
    module = importlib.import_module(MODULE)
    reader = module.PolicyOwnedTripodApproach(ROOT, scope="cpu_fixture",
        task_config=config, task_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        mesh_bindings=mesh_bindings)
    if visible_middle:
        for hand, side in zip(hands, ("left", "right"), strict=True):
            # Place one fixed material node at an interior lattice cell in the
            # real middle SDF. This is synthetic SDK input, not a feasible grasp.
            i = hand.body_names.index("finger33_link")
            grid = reader.geometry._grids[side]["finger33_link"]
            local = np.asarray(grid["origin"]) + 2*np.asarray(grid["spacing"])
            pose = np.asarray(hand.link_poses[0, i])
            matrix = reader.geometry._pose_matrix(np.r_[pose[:3], pose[6], pose[3:6]])
            position = rod.rope.nodes[0, 20] - matrix[:3, :3] @ local
            hand.link_poses[0, i, :3] = torch.as_tensor(position, dtype=hand.link_poses.dtype)
            hand.data.body_link_pose_w = hand.link_poses[..., [0, 1, 2, 6, 3, 4, 5]].clone()
    pre = ep.pre_action_state()
    c.history.capture_pre_action(pre)
    obs = c.sources.pack(c.trainer.model, pre, c.context.current)
    sample = c.trainer.collect_step(obs, obs)
    c.history.begin(pre, c.trainer)
    row = ep._exchange._step(sample["actions"], 0)
    row["rod"]["mesh_bindings"] = deepcopy(mesh_bindings)
    row["rod"]["mesh_names"] = list(PROXIMAL_POWER_MESH_NAMES)
    # Explicit CPU fixture channels, never retagged as native measurements.
    for key in PROXIMAL_POWER_MESH_NAMES:
        row["rod"]["named_wrenches"].setdefault(key, np.zeros(6))
    row["rod"]["external_support_wrenches_world"] = {
        key: row["rod"]["named_wrenches"][key].copy() for key in ("table", "support_0", "support_1")}
    pairs = row["rod"]["rod_mesh_contact_pairs"]
    pairs["solver_buffer"] = dict(substeps=12, mesh_count=17, capacity=41*17*12,
                                  attempted=len(pairs["pairs"]), overflow=False)
    for hand, side in zip(hands, ("left", "right"), strict=True):
        route = row["rod"]["hand_wrenches_com_world"][side]
        route["body_ids"] = [mesh_bindings[side+"."+name]["body_index"]
            for name in ("finger14_link", "finger23_link", "finger33_link",
                         "finger43_link", "finger53_link", "base_link", "finger52_link")]
        for key in ("forces", "torques"):
            route[key] = np.concatenate((route[key], np.zeros((1, 2, 3))), axis=1)
    with profile_reader_scope(profile):
        row["closure_geometry"] = reader.geometry.capture(
            row, sim=sim, bindings=ep._bindings, read_clock=rod.clock)
    event = dict(kind="physical", exchange=row, rod_boundary=ep._boundary.capture(row["rod"]))
    c.history.record(event)
    return reader, event, c, sim


def test_tripod_uses_three_distinct_pad_sources_and_two_hand_means(
        monkeypatch, tmp_path, backend_profile, sdk_templates):
    reader, event, c, sim = tripod_case(monkeypatch, tmp_path, backend_profile, sdk_templates)
    with profile_reader_scope(backend_profile):
        result = reader.capture(event, history_owner=c.history, grasp="tripod")
    assert result["version"] == "dh116-policy-owned-tripod-distance-v1"
    assert result["source_scope"] == "cpu_fixture"
    for side in ("left", "right"):
        hand = result["per_hand"][side]
        assert hand["required_pads"] == ["finger14_link", "finger23_link", "finger33_link"]
        pads = hand["pad_samples"]
        assert hand["positive_gap_mean_m"] == pytest.approx(
            sum(max(0., pads[name]["minimum_gap_m"]) for name in hand["required_pads"])/3)
    assert result["history_binding"]["physical_samples"] == 1
    assert result["source"]["target_material_ids"] == list(range(41))
    assert result["contact_inferred_from_geometry"] is False
    assert result["teacher_admission"] is False and result["new_physics_steps"] == 0
    assert sim.current_time_step_index == 3


def test_tripod_missing_middle_coverage_is_unknown_not_a_two_pad_mean(
        monkeypatch, tmp_path, backend_profile, sdk_templates):
    reader, event, c, sim = tripod_case(
        monkeypatch, tmp_path, backend_profile, sdk_templates, visible_middle=False)
    geometry = event["exchange"]["closure_geometry"]
    assert not geometry["source_complete"]
    assert any("finger33_link" in reason for reason in geometry["source_errors"])
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
    with profile_reader_scope(backend_profile), pytest.raises(SourceFailure, match="SDF"):
        reader.capture(event, history_owner=c.history, grasp="tripod")


@pytest.mark.parametrize("fault", ["lost_route", "wrong_body_id", "support_missing"])
def test_tripod_rejects_incomplete_or_wrong_actual_wrench_sources(
        monkeypatch, tmp_path, backend_profile, sdk_templates, fault):
    reader, event, c, sim = tripod_case(monkeypatch, tmp_path, backend_profile, sdk_templates)
    row = deepcopy(event["exchange"])
    if fault == "lost_route":
        row["rod"]["hand_wrenches_com_world"]["right"]["forces"] = np.zeros((1, 5, 3))
    elif fault == "wrong_body_id":
        row["rod"]["hand_wrenches_com_world"]["right"]["body_ids"][-1] = 99
    else:
        row["rod"]["external_support_wrenches_world"].pop("support_1")
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
    message = {"lost_route": r"finite measured array.*7, 3", "wrong_body_id": "COM/world/body",
               "support_missing": "both external support"}[fault]
    with profile_reader_scope(backend_profile), pytest.raises((SourceFailure, ValueError), match=message):
        reader.geometry.validate(row["closure_geometry"], row)


def test_tripod_rejects_foreign_history_and_pinch_relabelling(
        monkeypatch, tmp_path, backend_profile, sdk_templates):
    reader, event, c, sim = tripod_case(monkeypatch, tmp_path, backend_profile, sdk_templates)
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
    with profile_reader_scope(backend_profile), pytest.raises(SourceFailure, match="Tripod"):
        reader.capture(event, history_owner=c.history, grasp="pinch")
    bad = deepcopy(event)
    bad["exchange"]["action_readback"]["request"]["action24"][0, 0] += .01
    with profile_reader_scope(backend_profile), pytest.raises(SourceFailure, match="hash"):
        reader.capture(bad, history_owner=c.history, grasp="tripod")
