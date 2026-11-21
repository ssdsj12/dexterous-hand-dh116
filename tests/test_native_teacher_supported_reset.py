"""Reset the real supported rod wrapper with explicit CPU SDK/solver doubles.

The original five-pad reset remains covered separately. These tests retain all
palm/proximal/support slots; they do not grant native or Teacher qualification.
"""
from copy import deepcopy

import numpy as np
import pytest

from test_dual_hand_reset import setup
from test_dual_hand_rod_runtime import Grid
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import DualHandResetTransaction
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PowerSharedRodRuntime


def supported_reset_case(monkeypatch, *, nodes=None):
    events, original, hands, bindings, sim, seed, owners = setup(monkeypatch, nodes=nodes)
    for hand in hands:
        hand.body_names[6] = "finger52_link"
        hand.root_physx_view.link_paths = [[
            f"/World/{hand.side}Hand/{name}" for name in hand.body_names]]
    DualHandResetTransaction(seed, episode_uid="cpu-supported:original").run(
        sim, original, bindings, owner_resets=owners)
    palms, proximal, poses, palm_grids, proximal_grids = {}, {}, {}, {}, {}
    for hand, token in zip(hands, ("a", "b"), strict=True):
        for name, index, rows, grids in (
                ("base_link", 5, palms, palm_grids),
                ("finger52_link", 6, proximal, proximal_grids)):
            key = hand.side + "." + name
            rows[key] = dict(original.bindings[hand.side + ".finger14_link"],
                side=hand.side, body_name=name, body_index=index,
                prim_path=hand.root_physx_view.link_paths[0][index],
                capture_sha256=token*64, source_geometry_sha256=token*64,
                cooked_geometry_sha256=token*64, sdf_sha256=token*64)
            poses[key] = (hand.link_poses[0, index, :3].numpy().copy(),
                          hand.link_poses[0, index, 3:].numpy().copy())
            grids[key] = Grid()
    supports = {name: Grid() for name in ("support_0", "support_1")}
    original.require_pairs = True
    original.rope.enable_contact_pair_capture = lambda count: None
    rod = PowerSharedRodRuntime.from_original_reset(original,
        palm_bindings=palms, palm_grids=palm_grids,
        palm_poses={key: poses[key] for key in palms},
        expected_clock=original.clock(), support_grids=supports,
        support_poses={name: (np.zeros(3), np.array([0., 0., 0., 1.]))
                       for name in supports},
        proximal_bindings=proximal, proximal_grids=proximal_grids,
        proximal_poses={key: poses[key] for key in proximal})
    events.clear()
    return events, rod, hands, bindings, sim, seed, owners


def test_supported_reset_refreshes_all_four_extra_link_poses_without_steps(monkeypatch):
    events, rod, hands, bindings, sim, seed, owners = supported_reset_case(monkeypatch)
    old_rope, old_clock = id(rod.rope), deepcopy(rod.clock())
    before_resets = rod.rope.resets
    for hand in hands:
        hand.link_poses[..., :3] += .1  # stale prior-episode SDK pose cache
    receipt = DualHandResetTransaction(seed, episode_uid="cpu-supported:fresh").run(
        sim, rod, bindings, owner_resets=owners)
    assert id(rod.rope) == old_rope and rod.rope.resets == before_resets + 1
    assert rod.rope.steps == receipt["new_sdk_steps"] == receipt["new_solver_steps"] == 0
    assert rod.clock()["physics_step"] == sim.current_time_step_index == old_clock["physics_step"]
    assert rod.clock()["solver_time_s"] == old_clock["solver_time_s"]
    assert rod.clock()["reset_id"] == old_clock["reset_id"] + 1
    assert rod.clock()["episode_uid"] == "cpu-supported:fresh" and rod.clock()["rod_tick"] == 0
    assert set(receipt["pad_poses_xyzw"]) == set(rod.bindings)
    assert len(receipt["pad_poses_xyzw"]) == 14 and len(rod.mesh_names) == 17
    actual_teleports = {slot: (position, quaternion) for slot, position, quaternion, teleport
                        in rod.rope.poses if teleport}
    for hand in hands:
        for key, bound in rod.bindings.items():
            if bound["side"] != hand.side:
                continue
            expected = hand.link_poses[0, bound["body_index"]].numpy()
            np.testing.assert_array_equal(np.concatenate(receipt["pad_poses_xyzw"][key]), expected)
            np.testing.assert_array_equal(np.concatenate(actual_teleports[rod.mesh_names.index(key)]), expected)
    assert receipt["rod_reset"]["hand_wrenches_com_world"] is None
    assert receipt["rod_reset"]["external_support_wrenches_world"] is None
    assert receipt["teacher_admission"] is receipt["native_qualification"] is False


def test_supported_reset_rejects_wrong_proximal_binding_before_any_write(monkeypatch):
    events, rod, hands, bindings, sim, seed, owners = supported_reset_case(monkeypatch)
    rod.bindings["right.finger52_link"]["body_index"] = 7
    previous = deepcopy(rod.clock())
    with pytest.raises(ValueError, match="binding"):
        DualHandResetTransaction(seed, episode_uid="cpu-supported:bad").run(
            sim, rod, bindings, owner_resets=owners)
    assert not events and rod.clock() == previous


def test_supported_reset_rejects_asymmetric_contact_layout_before_any_write(monkeypatch):
    events, rod, hands, bindings, sim, seed, owners = supported_reset_case(monkeypatch)
    rod.bindings.pop("right.finger52_link")
    previous = deepcopy(rod.clock())
    with pytest.raises(ValueError, match="same original"):
        DualHandResetTransaction(seed, episode_uid="cpu-supported:asymmetric").run(
            sim, rod, bindings, owner_resets=owners)
    assert not events and rod.clock() == previous


def live_supported_reader_case(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_deformable_live_geometry import LiveGeometryReader
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import HAND_NAMES
    events, rod, hands, bindings, sim, seed, owners = supported_reset_case(monkeypatch)
    receipt = DualHandResetTransaction(seed, episode_uid="cpu-supported:geometry").run(
        sim, rod, bindings, owner_resets=owners)
    original_map = {key: deepcopy(row) for key, row in rod.bindings.items() if key in HAND_NAMES}
    shapes = []
    for i, hand in enumerate(hands):
        coms = np.zeros((1, 23, 7), dtype=np.float32)
        coms[..., 0] = .012 + .003*i
        coms[..., 6] = 1.
        hand.root_physx_view.get_coms = lambda coms=coms: coms
        shapes.extend((i, index, np.array([[0., 0., -.01], [0., .01, .01]]))
                      for index in range(23))
    # A non-contact body is the lowest original collision shape. Filtering
    # clearance to the 14 solver-bound bodies must change the result and fail.
    shapes[17][2][0, 2] = -.12
    inspected = dict(identity=[deepcopy(rod.clock())], pad_bindings=original_map,
                     shapes=shapes, table_top_m=.30)
    reader = LiveGeometryReader(sim, bindings, inspected, read_clock=rod.clock,
                               contact_bindings=rod.bindings)
    return reader, rod, hands, bindings, sim, inspected


def test_supported_live_geometry_reads_palm_proximal_com_and_all_original_shapes(monkeypatch):
    reader, rod, hands, bindings, sim, inspected = live_supported_reader_case(monkeypatch)
    packet = reader()
    assert set(packet["pad_poses"]) == set(packet["centers_of_mass"]) == set(rod.bindings)
    assert packet["physics_step"] == sim.current_time_step_index
    for i, hand in enumerate(hands):
        for name in ("base_link", "finger52_link"):
            key = hand.side + "." + name
            pose = hand.link_poses[0, hand.body_names.index(name)].numpy()
            np.testing.assert_array_equal(np.concatenate(packet["pad_poses"][key]), pose)
            np.testing.assert_allclose(packet["centers_of_mass"][key],
                pose[:3] + np.array([.012 + .003*i, 0., 0.]), rtol=0, atol=1e-8)
    # Authored fixture bodies are at Z=.67 with identity rotation. Independent
    # analytic minima are .67-.12-.30 and .67-.01-.30, in metres.
    assert packet["clearance_m"][0].tolist() == pytest.approx([.25, .36], abs=1e-7)
    assert packet["lowest_shape_index"].tolist() == [17]
    assert len(inspected["shapes"]) == 46 and not packet["native_clearance_qualified"]


def test_supported_live_reader_rejects_override_of_inspected_pad_binding(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_deformable_live_geometry import LiveGeometryReader
    reader, rod, hands, bindings, sim, inspected = live_supported_reader_case(monkeypatch)
    wrong = deepcopy(rod.bindings)
    wrong["left.finger14_link"]["body_index"] = hands[0].body_names.index("finger23_link")
    with pytest.raises(ValueError, match="inspected"):
        LiveGeometryReader(sim, bindings, inspected, read_clock=rod.clock, contact_bindings=wrong)
