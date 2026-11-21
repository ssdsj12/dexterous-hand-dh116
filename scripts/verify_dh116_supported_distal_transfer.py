"""Keep a bounded supported distal probe tied to real shapes and solver reactions."""

from collections import Counter

import numpy as np

from go2_pvcnn.control.dh116.tabletop_power_shared_rod import END_SUPPORT_NAMES
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import measure_supported_distal_contact

SIDES = ("left", "right")
DISTAL = ("finger14_link", "finger23_link", "finger33_link")


def target_shape_keys(bindings, clearance_shapes, stations):
    counts = Counter((side, body) for side, body, _ in clearance_shapes)
    keys = set()
    for name in stations:
        side, link = name.split(".", 1)
        entry = bindings.get(name)
        if (side not in SIDES or link not in DISTAL or entry is None
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


class SupportedDistalProbe:
    """Track measured contact and the following PhysX command, without granting success."""

    def __init__(self, boundary_sdk, stations, bindings, clearance_shapes, *,
                 grids, archive, max_pad_penetration_m):
        self.boundary_sdk = boundary_sdk
        self.stations = stations
        self.shape_keys = target_shape_keys(bindings, clearance_shapes, stations)
        self.grids = {name: grids[name] for name in stations}
        self.archive = archive
        self.max_pad_penetration_m = max_pad_penetration_m
        self.previous = None
        self.first_contact_sdk = None
        self.first_transfer_sdk = None
        self.contact_rows = 0
        self.transfer_rows = 0
        self.loaded_rows = {"left": 0, "right": 0}

    def active(self, sdk):
        return sdk > self.boundary_sdk

    def check_pad_gaps(self, poses, reference_nodes):
        from scipy.spatial.transform import Rotation
        from scripts.prepare_dh116_pinch_closure_margin import sample_contact

        nodes = np.asarray(reference_nodes, dtype=float)
        if nodes.shape != (1, 41, 3) or not np.isfinite(nodes).all():
            raise ValueError("one intact previous solver node model required")
        for name, grid in self.grids.items():
            position, quaternion = poses[name]
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
            transform[:3, 3] = position
            result = sample_contact(grid, transform, nodes[0], dtype=np.float64)
            gap = result["minimum_gap_m"]
            if gap is None or gap < -self.max_pad_penetration_m:
                raise ValueError(f"selected live pad SDF exceeded its penetration bound: {name} gap={gap!r}")

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
        violations = pairs_outside_station(pairs, self.stations)
        evidence = measure_supported_distal_contact(rod, grasp="tripod")
        intended_pairs = sum(p["node_index"] in self.stations.get(p["mesh_name"], ()) for p in pairs)
        if intended_pairs:
            if self.first_contact_sdk is None:
                self.first_contact_sdk = sdk
            self.contact_rows += 1
        for side, hand in evidence["per_hand"].items():
            self.loaded_rows[side] += int(hand["loaded"] and not violations)
        record = dict(sdk_step=sdk, intended_pairs=intended_pairs, violations=violations,
                      reaction_from_previous_solver_row_applied=transferred, force_evidence=evidence)
        return record, "unexpected_contact_outside_distal_station" if violations else None

    def summary(self):
        return dict(first_contact_sdk=self.first_contact_sdk, first_transfer_sdk=self.first_transfer_sdk,
                    contact_rows=self.contact_rows, transfer_rows=self.transfer_rows,
                    all_substeps_loaded_rows_by_hand=self.loaded_rows.copy(),
                    grasp_qualified=False, teacher_admission=False)
