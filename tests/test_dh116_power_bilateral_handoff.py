"""Power partial handoff checks without starting Isaac."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import verify_dh116_power_bilateral_handoff as handoff


def test_commands_follow_canonical_axis_order_and_exact_target_continuity():
    first = np.zeros((2, 2, 12))
    first[1, 0, 0] = .0001
    first[1, 1, 4] = .002
    second = np.stack([first[-1], first[-1]])
    second[1, 0, 7] = .001
    a, b = np.zeros((1, 24)), np.zeros((1, 24))
    a[0, 12], a[0, 22], b[0, 1] = .6, .24, .2
    np.testing.assert_array_equal(handoff.compile_commands([(first, a), (second, b)]), np.vstack((a, b)))
    shifted = second.copy()
    shifted[:, 0, 2] = .001
    with pytest.raises(ValueError, match="continuous"):
        handoff.compile_commands([(first, a), (shifted, b)])
    with pytest.raises(ValueError, match="target increments"):
        handoff.compile_commands([(first, a), (second, a)])


def test_canonical_group_norm_is_a_hard_limit():
    states = np.zeros((2, 2, 12))
    states[1, 0, :3] = .00015
    action = np.zeros((1, 24))
    action[0, 12:15] = .9
    with pytest.raises(ValueError, match="limits"):
        handoff.compile_commands([(states, action)])


def test_phase_station_ownership_only_opens_contact_when_qualified():
    palms = {"left.base_link": [31], "right.base_link": [9]}
    stations = {"left.base_link": [29], "right.base_link": [9], "left.finger14_link": [37],
                "right.finger14_link": [4], "left.finger23_link": [36], "right.finger43_link": [3]}
    phases = handoff.phase_stations(6504, [284, 202, 88, 82, 307], palms, stations)
    assert [p["end_sdk"] for p in phases] == [8776, 10392, 11096, 11752, 14208, 14216]
    assert phases[0]["stations"] == {**palms, "left.base_link": [29, 30, 31]}
    assert set(phases[1]["stations"]) == set(palms) | {"left.finger14_link", "right.finger14_link"}
    assert all(p["stations"] == stations for p in phases[2:])
    swapped = deepcopy(stations)
    swapped["right.finger43_link"] = [36]
    with pytest.raises(ValueError, match="wire half"):
        handoff.phase_stations(6504, [284, 202, 88, 82, 307], palms, swapped)


def test_completed_live_rod_must_match_geometry_clock_and_bytes():
    nodes = np.arange(123, dtype=float).reshape(1, 41, 3)
    row = {"physics_step": 6505, "rod": {"physics_step": 6505, "node_positions": nodes}}
    handoff.verify_same_rod(row, {"physics_step": 6505, "node_positions": nodes.copy()})
    with pytest.raises(ValueError, match="completed live rod"):
        handoff.verify_same_rod(row, {"physics_step": 6504, "node_positions": nodes})
    with pytest.raises(ValueError, match="completed live rod"):
        handoff.verify_same_rod(row, {"physics_step": 6505, "node_positions": nodes+.001})


def test_anchor_rejects_changed_force_or_wrist_target_even_when_positions_match():
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    from scripts.verify_dh116_power_bilateral_handoff_probe import verify_anchor
    from test_dh116_power_partial_stage_evidence import pair_exchanges

    _, row = pair_exchanges()
    rod = row["rod"]
    rod["velocities"] = np.zeros((1,41,3))
    rod["rest_lengths_m"] = np.zeros(40)
    hand = {field: np.zeros((1, 6)) for field in ("q", "qd", "backend_targets", "native_anchor_pose_w")}
    row.update(action_readback={"post": {"hands": {"left": hand, "right": deepcopy(hand)}}},
               sdk_hand_geometry={"physics_step": 4},
               wrist_projection={"projected_angles": np.zeros((1,2,3))}, guard_failure=None)
    archive = encode_owned(row)
    verify_anchor(row, archive, 4)
    changed = deepcopy(row)
    changed["rod"]["hand_wrenches_com_world"]["right"]["torques"][0,0,0] += .1
    with pytest.raises(ValueError, match="sealed reaction"):
        verify_anchor(changed, archive, 4)
    changed = deepcopy(row)
    changed["wrist_projection"]["projected_angles"][0,0,0] += .1
    with pytest.raises(ValueError, match="projected targets"):
        verify_anchor(changed, archive, 4)


def test_partial_opposition_requires_current_geometry_and_actual_previous_wrench():
    from scripts.verify_dh116_power_bilateral_handoff_probe import SupportedPowerHandoffProbe
    from test_dh116_power_partial_stage_evidence import pair_exchanges, STATIONS, SCENE

    previous, current = pair_exchanges()
    for substep in range(12):
        for mesh, node in (("support_0", 0), ("support_1", 40)):
            current["rod"]["rod_mesh_contact_pairs"]["pairs"].append(
                dict(mesh_name=mesh, node_index=node, substep=substep, normal_force_on_rod_n=.2))
    # The window's last predicate is tested separately with checked solver capture.
    # This row is inside the phase so the current per-SDK guards remain in scope.
    probe = SupportedPowerHandoffProbe.__new__(SupportedPowerHandoffProbe)
    probe.boundary_sdk, probe.anchor_matches, probe.previous = 3, True, previous
    probe.phases = [dict(end_sdk=5, stage="fixed_target_observation", stations=STATIONS)]
    probe.checked_steps = probe.reaction_steps = 0
    probe.scene, probe.terminal, probe._geometry_sdk = SCENE, None, None
    with pytest.raises(ValueError, match="completed live geometry"):
        probe.observe(current)
    probe.check_completed_rod(current, current["rod"], lambda: {"physics_step": 4})
    changed = deepcopy(current)
    changed["issued_composer_wrench"]["left"]["forces"][0,0,0] += .1
    with pytest.raises(ValueError, match="applied world wrench"):
        probe.observe(changed)


def test_publication_evidence_does_not_change_fingerprint_but_runtime_sources_stay_pinned(tmp_path):
    import json
    from pathlib import Path

    root = tmp_path
    registry = str(root/"docs/contracts/dh116_work_package_registry.json")
    manifest = str(root/"docs/contracts/dh116_release_manifest.json")
    readiness = str(root/"docs/contracts/dh116_teacher_power_bilateral_handoff_v1.json")
    physics = str(root/"docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json")
    source = str(root/"scripts/verify_dh116_power_bilateral_handoff.py")
    before = {registry: "1", manifest: "2", source: "3", physics: "4"}
    after = dict(before, **{registry: "5", manifest: "6", readiness: "7"})
    filtered = handoff.execution_file_inventory(before, project=root)
    assert json.dumps(filtered, sort_keys=True) == json.dumps(
        handoff.execution_file_inventory(after, project=root), sort_keys=True)
    assert set(map(Path, filtered)) == {Path(source), Path(physics)}
    assert set(before) == {registry, manifest, source, physics}
