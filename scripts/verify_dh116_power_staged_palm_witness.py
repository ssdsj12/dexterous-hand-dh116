"""Qualify a first Power palm contact on its own original physical link."""

import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import measure_power_contact
from scripts import screen_dh116_tripod_measured_precontact as measured
from scripts.verify_dh116_supported_distal_transfer import (
    check_reaction_delivery, pairs_outside_station, target_shape_keys, validate_live_pad_gaps,
)

CONTRACT = measured.ROOT / "docs/contracts/dh116_teacher_power_guarded_wrist_bypass_native_v1.json"
REQUEST = measured.ROOT / "configs/dh116_tasks/flexible_power_staged_palm_witness_v1.json"
SDK = 3584
MAX_PALM_PENETRATION_M = .00025


class SupportedPalmProbe:
    """Observe palm-only contact and its next native reaction, without granting grasp status."""

    def __init__(self, bindings, shapes, *, bake_dirs, boundary_sdk, stations):
        self.shape_keys = target_shape_keys(bindings, shapes, stations, allowed_links=("base_link",))
        self.boundary_sdk = boundary_sdk
        self.stations = stations
        self.bilateral = None
        self.archive = None
        self.previous = None
        self.first_contact = {}
        self.first_reaction = None
        self.contact_rows = 0
        self.reaction_rows = 0
        self.grids = {}
        for name in stations:
            side = name.split(".", 1)[0]
            path = bake_dirs[side] / f"{name}.npz"
            if measured.native.sha(path) != bindings[name]["sdf_sha256"]:
                raise ValueError("Power palm CPU guard must use that physical body's registered SDF")
            with np.load(path, allow_pickle=False) as array:
                self.grids[name] = {field: array[field].copy() for field in array.files}

    def active(self, sdk):
        return sdk > self.boundary_sdk

    def contact_shape(self, sdk, key):
        return self.active(sdk) and key in self.shape_keys

    def check_pad_gaps(self, sdk, poses, nodes):
        if self.active(sdk):
            validate_live_pad_gaps(self.grids, poses, nodes, max_penetration_m=MAX_PALM_PENETRATION_M)

    def observe(self, row):
        sdk = row["physics_step"]
        if not self.active(sdk):
            if sdk == self.boundary_sdk:
                self.previous = row
            return None, None
        if self.previous is None:
            raise ValueError("Power palm witness requires the completed precontact boundary")
        reaction = check_reaction_delivery(self.previous, row)
        self.previous = row
        pairs = row["rod"]["rod_mesh_contact_pairs"]["pairs"]
        violations = pairs_outside_station(pairs, self.stations)
        allowed_pairs = [pair for pair in pairs if pair["node_index"] in self.stations.get(pair["mesh_name"], ())]
        for pair in allowed_pairs:
            self.first_contact.setdefault(pair["mesh_name"], sdk)
        if reaction and self.first_reaction is None:
            self.first_reaction = sdk
        self.contact_rows += int(bool(allowed_pairs))
        self.reaction_rows += int(reaction)
        return dict(sdk_step=sdk, actual_allowed_palm_pairs=len(allowed_pairs), violations=violations,
                    reaction_from_previous_solver_row_applied=reaction, force_evidence=measure_power_contact(row["rod"])), (
                    "unexpected_contact_outside_power_palm_station" if violations else None)

    def summary(self):
        return dict(first_contact_sdk_by_palm=self.first_contact, first_reaction_sdk=self.first_reaction,
                    contact_rows=self.contact_rows, reaction_rows=self.reaction_rows,
                    grasp_qualified=False, teacher_admission=False)


def load_palm_approaches(receipt, config):
    expected = json.loads(CONTRACT.read_bytes())["actual"]
    outer = measured.sources.checked_json(receipt, expected["outer_sha256"])
    native = receipt.parent / "native"
    boundary = native/f"power_control_{SDK}.json"
    layout = native/"power_support_scene.json"
    close = native/"app_close_returned.json"
    if (not expected["native_natural_exit_verified"] or not expected["exact_opening_anchor_replayed"]
            or outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or outer["exceeded_budget"] or outer["forced_exit"]
            or measured.native.sha(boundary) != expected["last_row_sha256"]
            or measured.sources.checked_json(close, outer["artifact_sha256"][close.name])["app_close_returned"] is not True
            or measured.sources.checked_json(layout, outer["artifact_sha256"][layout.name])["supports"]
                != config["support_scene"]["supports"]):
        raise ValueError("Power palm approach requires its exact naturally closed high waypoint")
    raw = measured.sources.checked_json(boundary, expected["last_row_sha256"])
    if (raw["physics_step"] != SDK or raw["sdk_hand_geometry"]["physics_step"] != SDK
            or raw["rod"]["physics_step"] != SDK or raw["rod"]["rod_tick"] != (SDK-2)*12
            or raw["guard_failure"] is not None):
        raise ValueError("Power wrist pose and rod must share the high waypoint clock")
    request = measured.sources.checked_json(REQUEST, "5a99c42dfe5a53d95afcbca4d4d041e84f49a711fe247a924ee49b84d36fe631")
    if request["version"] != "dh116-power-staged-palm-witness-v1" or request["source_sdk"] != SDK:
        raise ValueError("the fixed Power palm witness route is required")
    return raw, request, (CONTRACT, REQUEST, Path(__file__), receipt, boundary, layout, close)
