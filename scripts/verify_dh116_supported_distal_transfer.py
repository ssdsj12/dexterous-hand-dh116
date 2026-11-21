"""Keep a bounded supported distal probe tied to real shapes and solver reactions."""

from collections import Counter

import numpy as np

from go2_pvcnn.control.dh116.tabletop_power_shared_rod import END_SUPPORT_NAMES
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import measure_supported_distal_contact

SIDES = ("left", "right")
DISTAL = ("finger14_link", "finger23_link", "finger33_link")


def target_shape_keys(bindings, clearance_shapes, stations, *, allowed_links=DISTAL):
    counts = Counter((side, body) for side, body, _ in clearance_shapes)
    keys = set()
    for name in stations:
        side, link = name.split(".", 1)
        entry = bindings.get(name)
        if (side not in SIDES or link not in allowed_links or entry is None
                or entry["side"] != side or entry["body_name"] != link
                or entry["prim_path"].rsplit("/", 1)[-1] != link):
            raise ValueError("owned mounted distal pair binding required")
        key = SIDES.index(side), entry["body_index"]
        if counts[key] != 1 or key in keys:
            raise ValueError("exactly one original source envelope per target distal body required")
        keys.add(key)
    return keys


def pairs_outside_station(pairs, stations):
    return sorted({f"{pair['mesh_name']}:{pair['node_index']}" for pair in pairs
                   if pair["mesh_name"] not in END_SUPPORT_NAMES
                   and pair["node_index"] not in stations.get(pair["mesh_name"], ())})


def selected_stations_at(sdk, boundary_sdk, stations, bilateral_sdk=None, extra_stations=None):
    if sdk <= boundary_sdk:
        return {}
    if bilateral_sdk is None or sdk <= bilateral_sdk:
        return stations
    return {**stations, **extra_stations}


def midpoint_reached(row, target):
    from scipy.spatial.transform import Rotation

    hand = row["action_readback"]["post"]["hands"]["left"]
    actual_q = hand["q"][0, hand["active_indices"]]
    xyz, quat = row["sdk_hand_geometry"]["poses"]["left.base_link"]
    target_pose = np.asarray(target["pose_wxyz"])
    orientation_error = (Rotation.from_quat(quat).inv() *
                         Rotation.from_quat(target_pose[[4, 5, 6, 3]])).magnitude()
    return bool(np.linalg.norm(xyz - target_pose[:3]) <= .00020
                and orientation_error <= .008
                and np.max(np.abs(actual_q - target["active_q_rad"])) <= .0025)


def check_reaction_delivery(prior, current):
    expected = prior["rod"]["hand_wrenches_com_world"]
    issued = current["issued_composer_wrench"]
    if (current["physics_step"] != prior["physics_step"] + 1
            or current["reaction_source_physics_step"] != prior["physics_step"]
            or set(issued) != set(expected)):
        raise ValueError("agreed consecutive SDK and wrench source ticks required")
    transferred = False
    for side, route in expected.items():
        if (issued[side]["body_ids"] != route["body_ids"]
                or issued[side]["is_global"] is not True or route["is_global"] is not True
                or any(not np.array_equal(issued[side][key], route[key]) for key in ("forces", "torques"))):
            raise ValueError("applied world wrench differs from previous solver feedback")
        transferred |= bool(np.any(route["forces"]) or np.any(route["torques"]))
    return transferred


def validate_live_pad_gaps(grids, poses, reference_nodes, *, max_penetration_m):
    from scipy.spatial.transform import Rotation
    from scripts.prepare_dh116_pinch_closure_margin import sample_contact

    nodes = np.asarray(reference_nodes, dtype=float)
    if nodes.shape != (1, 41, 3) or not np.isfinite(nodes).all():
        raise ValueError("one intact previous solver node model required")
    for name, grid in grids.items():
        position, quaternion = poses[name]
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        transform[:3, 3] = position
        gap = sample_contact(grid, transform, nodes[0], dtype=np.float64)["minimum_gap_m"]
        if gap is None or gap < -max_penetration_m:
            raise ValueError(f"selected live pad SDF exceeded its penetration bound: {name} gap={gap!r}")


class SupportedDistalProbe:
    """Track measured contact and the following PhysX command, without granting success."""

    def __init__(self, boundary_sdk, stations, bindings, clearance_shapes, *,
                 grids, archive, max_pad_penetration_m, grasp="tripod", bilateral=None):
        if grasp not in ("pinch", "tripod"):
            raise ValueError("only admitted distal diagnostic styles can use this probe")
        self.grasp = grasp
        self.boundary_sdk = boundary_sdk
        self.stations = stations
        self.bilateral = bilateral
        self.additional_stations = {} if bilateral is None else bilateral["stations"]
        if (bilateral is not None and (set(self.additional_stations) & set(stations)
                or bilateral["boundary_sdk"] <= boundary_sdk)):
            raise ValueError("independent later distal contact areas required")
        self.shape_keys = target_shape_keys(bindings, clearance_shapes, stations)
        self.additional_shape_keys = target_shape_keys(bindings, clearance_shapes, self.additional_stations)
        self.grids = {name: grids[name] for name in set(stations) | set(self.additional_stations)}
        self.archive = archive
        self.max_pad_penetration_m = max_pad_penetration_m
        self.previous = None
        self.first_contact_sdk = None
        self.first_transfer_sdk = None
        self.contact_rows = 0
        self.transfer_rows = 0
        self.loaded_rows = {"left": 0, "right": 0}
        self.bilateral_loaded_rows = 0
        self.first_contact_sdk_by_side = {side: None for side in SIDES}

    def active(self, sdk):
        return sdk > self.boundary_sdk

    def stations_at(self, sdk):
        return selected_stations_at(sdk, self.boundary_sdk, self.stations,
            None if self.bilateral is None else self.bilateral["boundary_sdk"], self.additional_stations)

    def contact_shape(self, sdk, key):
        if not self.active(sdk):
            return False
        return key in self.shape_keys or (self.bilateral is not None and
            sdk > self.bilateral["boundary_sdk"] and key in self.additional_shape_keys)

    def check_pad_gaps(self, sdk, poses, reference_nodes):
        grids = {name: self.grids[name] for name in self.stations_at(sdk)}
        validate_live_pad_gaps(grids, poses, reference_nodes, max_penetration_m=self.max_pad_penetration_m)

    def observe(self, row):
        sdk = row["physics_step"]
        if not self.active(sdk):
            if sdk == self.boundary_sdk:
                self.previous = row
            return None, None
        if self.previous is None:
            raise ValueError("expected guarded native contact boundary is absent")
        transferred = check_reaction_delivery(self.previous, row)
        self.previous = row
        if transferred:
            if self.first_transfer_sdk is None:
                self.first_transfer_sdk = sdk
            self.transfer_rows += 1
        rod = row["rod"]
        pairs = rod["rod_mesh_contact_pairs"]["pairs"]
        stations = self.stations_at(sdk)
        violations = pairs_outside_station(pairs, stations)
        evidence = measure_supported_distal_contact(rod, grasp=self.grasp)
        wrong_side = {f"{pair['link']}:{pair['node_index']}" for pair in evidence["wrong_side_contacts"]}
        violations = sorted(set(violations) | wrong_side)
        active_links = [p["mesh_name"] for p in pairs if p["node_index"] in stations.get(p["mesh_name"], ())]
        intended_pairs = len(active_links)
        for side in SIDES:
            if self.first_contact_sdk_by_side[side] is None and any(name.startswith(side + ".") for name in active_links):
                self.first_contact_sdk_by_side[side] = sdk
        if intended_pairs:
            if self.first_contact_sdk is None:
                self.first_contact_sdk = sdk
            self.contact_rows += 1
        for side, hand in evidence["per_hand"].items():
            self.loaded_rows[side] += int(hand["loaded"] and not violations)
        self.bilateral_loaded_rows += int(evidence["bilateral"] and not violations)
        record = dict(sdk_step=sdk, intended_pairs=intended_pairs, violations=violations,
                      reaction_from_previous_solver_row_applied=transferred, force_evidence=evidence)
        return record, "unexpected_contact_outside_distal_station" if violations else None

    def summary(self):
        return dict(first_contact_sdk=self.first_contact_sdk, first_transfer_sdk=self.first_transfer_sdk,
                    contact_rows=self.contact_rows, transfer_rows=self.transfer_rows,
                    all_substeps_loaded_rows_by_hand=self.loaded_rows.copy(),
                    bilateral_loaded_rows=self.bilateral_loaded_rows,
                    first_contact_sdk_by_side=self.first_contact_sdk_by_side.copy(),
                    grasp_qualified=False, teacher_admission=False)
