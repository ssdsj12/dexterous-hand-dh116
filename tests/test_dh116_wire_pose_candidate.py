import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import prepare_seed_layout, seed_arrays


@pytest.fixture
def scene():
    nodes = np.column_stack((np.zeros(41), np.linspace(-.2, .2, 41), np.full(41, .3)))
    active = (active_limits()[0] + active_limits()[1]) / 2
    roots = []
    for station, angles in ((30, (.1, -.2, .3)), (10, (-.4, .2, -.1))):
        orientation = Rotation.from_euler("xyz", angles).as_quat()
        roots.append(np.r_[nodes[station] + [.013, .004, .026], np.roll(orientation, 1)])
    seed = dict(nodes=nodes.tolist(), roots=np.asarray(roots).tolist(),
                joints=[expand(active).tolist(), expand(active * .8).tolist()])
    return seed, dict(material_node_ids=list(range(41)), nodes_w_m=nodes.tolist())


def candidate(seed, target, **stations):
    from scripts.prepare_dh116_wire_pose_candidate import retarget_seed

    return retarget_seed(seed, target, stations or {"left": 30, "right": 10})


def test_pose_follows_rigidly_moved_wire_and_enters_existing_seed_layout(scene):
    seed, target = scene
    rotation = Rotation.from_euler("z", .6)
    delta = np.array([.05, -.01, .03])
    target["nodes_w_m"] = (rotation.apply(target["nodes_w_m"]) + delta).tolist()
    result = candidate(seed, target)
    nodes, full, roots, _ = seed_arrays(result)
    np.testing.assert_array_equal(nodes, target["nodes_w_m"])
    np.testing.assert_array_equal(full, seed["joints"])
    start = np.asarray(seed["roots"])
    np.testing.assert_allclose(roots[:, :3], rotation.apply(start[:, :3]) + delta, atol=1e-12)
    start_rot = Rotation.from_quat(np.roll(start[:, 3:], -1, axis=1))
    expected = (rotation * start_rot).as_matrix()
    np.testing.assert_allclose(Rotation.from_quat(np.roll(roots[:, 3:], -1, axis=1)).as_matrix(), expected, atol=1e-12)
    layout = prepare_seed_layout({}, result)
    np.testing.assert_allclose(layout["wire_center"], nodes.mean(axis=0))
    for index, hand in enumerate(("left", "right")):
        np.testing.assert_allclose(layout["hand_roots"][hand] + [0, 0, .04, 0, 0, 0, 0], roots[index])


def test_same_material_ids_give_same_pose_when_target_rows_are_reversed(scene):
    seed, target = scene
    target["nodes_w_m"][30][0] += .005
    descending = {name: data[::-1] for name, data in target.items()}
    assert candidate(seed, target) == candidate(seed, descending)


def test_bent_wire_uses_separate_left_and_right_station_frames(scene):
    seed, target = scene
    nodes = np.asarray(target["nodes_w_m"])
    nodes[:, 0] = np.abs(nodes[:, 1])
    target["nodes_w_m"] = nodes.tolist()
    result = candidate(seed, target)
    for raw_pose, fitted, station, angle in zip(seed["roots"], result["roots"], [30, 10], [-np.pi/4, np.pi/4]):
        yaw = Rotation.from_euler("z", angle)
        np.testing.assert_allclose(fitted[:3], nodes[station] + yaw.apply(np.array(raw_pose[:3]) - seed["nodes"][station]), atol=1e-12)
        start = Rotation.from_quat(np.roll(raw_pose[3:], -1))
        fit = Rotation.from_quat(np.roll(fitted[3:], -1))
        assert (fit.inv() * yaw * start).magnitude() < 1e-12


@pytest.mark.parametrize("change,reason", [
    ("duplicate", "material node IDs"),
    ("bool_id", "material node IDs"),
    ("nonfinite", "finite"),
    ("vertical", "gravity"),
    ("foldback", "folded"),
    ("zero_length", "nonzero segments"),
    ("uncoupled", "passive coupling"),
    ("joint_limit", "coupled limits"),
    ("invalid_root", "unit quaternions"),
])
def test_invalid_wire_identity_frames_and_joint_seeds_are_rejected(scene, change, reason):
    seed, target = scene
    if change == "duplicate":
        target["material_node_ids"][30] = 10
    elif change == "bool_id":
        target["material_node_ids"][1] = True
    elif change == "nonfinite":
        target["nodes_w_m"][30][1] = float("nan")
    elif change == "vertical":
        for row in target["nodes_w_m"]:
            row[1], row[2] = row[1] * 1e-4, row[2] + row[1]
    elif change == "foldback":
        target["nodes_w_m"][31] = target["nodes_w_m"][29][:]
    elif change == "zero_length":
        target["nodes_w_m"][31] = target["nodes_w_m"][30][:]
    elif change == "uncoupled":
        seed["joints"][0][2] += .1
    elif change == "joint_limit":
        seed["joints"][0][0] = 9.
    elif change == "invalid_root":
        seed["roots"][0][3] = 3.
    with pytest.raises(ValueError, match=reason):
        candidate(seed, target)


@pytest.mark.parametrize("stations", [
    {"left": 0, "right": 10}, {"left": 30, "right": 40},
    {"left": 30, "right": 30}, {"left": True, "right": 10},
    {"left": 30}, {"left": 30, "right": 10, "elsewhere": 12},
])
def test_station_requires_two_distinct_interior_material_nodes(scene, stations):
    from scripts.prepare_dh116_wire_pose_candidate import retarget_seed

    with pytest.raises(ValueError, match="stations"):
        retarget_seed(*scene, stations)


def test_candidate_cli_records_hashes_and_keeps_existing_output(scene, tmp_path):
    from hashlib import sha256
    from scripts.prepare_dh116_wire_pose_candidate import main

    seed, target = scene
    reference, wire, output = (tmp_path / name for name in ("reference.json", "wire.json", "candidate.json"))
    reference.write_text(json.dumps(seed))
    wire.write_text(json.dumps(target))
    args = ["--reference", str(reference), "--target", str(wire), "--output", str(output),
            "--grasp", "power", "--left-node", "30", "--right-node", "10"]
    assert main(args) == 0
    data = json.loads(output.read_text())
    assert data["grasp"] == "power" and data["material_node_ids_by_hand"] == {"left": 30, "right": 10}
    assert data["input_sha256"] == {name: sha256(path.read_bytes()).hexdigest()
                                    for name, path in (("reference", reference), ("target", wire))}
    for key in ("collision_qualified", "native_contact_verified", "teacher_admission"):
        assert data[key] is False
    seed_arrays(data["seed"])
    old = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == old


@pytest.mark.parametrize("style,left_node,right_node,expected", [
    ("pinch", 39, 1, {"left": [], "right": []}),
    ("tripod", 36, 8, {"left": [], "right": ["finger32_link", "finger33_link"]}),
    ("power", 33, 8, {"left": ["finger32_link", "finger33_link"],
                       "right": ["finger32_link", "finger33_link"]}),
])
def test_candidate_cli_checks_real_table_geometry_for_each_hand(style, left_node, right_node, expected, tmp_path):
    from pathlib import Path
    from scripts.prepare_dh116_wire_pose_candidate import main

    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "docs/contracts/dh116_wire_pose_candidate_v1.json").read_text())
    inputs = Path(metadata["evidence"]["directory"])
    output = tmp_path / "candidate.json"
    code = main(["--reference", str(inputs / f"{style}-reference.json"),
                 "--target", str(inputs / f"{style}-wire.json"), "--output", str(output),
                 "--grasp", style, "--left-node", str(left_node), "--right-node", str(right_node),
                 "--audit-table-clearance", "--project", str(root)])
    result = json.loads(output.read_bytes())
    clearance = result["static_table_clearance"]
    assert code == (0 if style == "pinch" else 2)
    assert clearance["passed"] is (style == "pinch")
    for hand, links in expected.items():
        arm = clearance["hands"][hand]
        assert list(arm["source_intersections"]) == links
        assert len(arm["source"]) == 17
        assert len(arm["locked_cooked_pads"]) == 5
        assert arm["source_sha256"]["usd"]
    assert not result["collision_qualified"] and not result["native_contact_verified"]
    assert not result["teacher_admission"]


@pytest.mark.parametrize("style,left_id,right_id,expected", [
    ("pinch", 39, 1, (0., 0.)),
    ("tripod", 36, 8, (.196, 4.575)),
    ("power", 33, 8, (6.165, 4.575)),
])
def test_vertical_repair_keeps_rods_and_joints_and_rejects_detached_pads(style, left_id, right_id, expected, tmp_path):
    from pathlib import Path
    from scripts.prepare_dh116_wire_pose_candidate import main

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "docs/contracts/dh116_wire_pose_candidate_v1.json").read_text())
    inputs = Path(config["evidence"]["directory"])
    output = tmp_path / "candidate.json"
    code = main(["--reference", str(inputs / f"{style}-reference.json"),
                 "--target", str(inputs / f"{style}-wire.json"), "--output", str(output),
                 "--grasp", style, "--left-node", str(left_id), "--right-node", str(right_id),
                 "--propose-table-repair", "--project", str(root)])
    record = json.loads(output.read_bytes())
    repair = record["vertical_repair"]
    baseline, trial = record["seed"], repair["trial_seed"]
    assert trial["nodes"] == baseline["nodes"] and trial["joints"] == baseline["joints"]
    for side, old, new, rise in zip(("left", "right"), baseline["roots"], trial["roots"], expected):
        assert new[:2] == old[:2] and new[3:] == old[3:]
        assert (new[2] - old[2])*1000 == pytest.approx(rise, abs=.001)
        assert repair["table_after"]["hands"][side]["passed"]
    if style != "pinch":
        right = repair["hands"]["right"]["required_pad_gaps"]
        assert right["finger23_link"]["comparison"] == "proven_farther"
        assert right["finger33_link"]["after_m"]["upper_m"] < 0
        assert not repair["static_repair_proven"]
        assert code == 2
    else:
        assert repair["status"] == "unchanged" and code == 0
    assert record["native_contact_verified"] is False and record["teacher_admission"] is False
