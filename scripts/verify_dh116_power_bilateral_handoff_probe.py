"""Original supported Power exchange with a live, phase-scoped contact guard."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import IDENTITY, _clock
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import rod_support_lower
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import END_SUPPORT_NAMES
from scripts.audit_dh116_wire_contact_witness import endpoint_gap
from scripts.certify_dh116_power_thumb_route import validate_palm_lattice
from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
from scripts.prepare_dh116_pinch_closure_margin import sample_contact
from scripts.prepare_dh116_wire_pose_candidate import _pad_gap_interval
from scripts.screen_dh116_power_measured_reseat import support_adjacent_nodes
from scripts.screen_dh116_power_partial_opposition import PAD_PENETRATION_M, NON_TARGET_CLEARANCE_M
from scripts.verify_dh116_power_bilateral_handoff import verify_same_rod
from scripts.verify_dh116_power_partial_stage import confirm_local_opposition
from scripts.verify_dh116_power_shared_rod import SIDES, SUBSTEPS, native, verify_waypoint_boundary
from scripts.verify_dh116_power_staged_palm_witness import MAX_PALM_PENETRATION_M
from scripts.verify_dh116_supported_distal_transfer import (
    check_reaction_delivery, pairs_outside_station, target_shape_keys,
)


def verify_anchor(current, archive, sdk):
    """The new process must measure the same physical boundary before continuing."""
    actual = encode_owned(current)
    verify_waypoint_boundary(actual, archive, sdk_step=sdk, exact_geometry=True)
    if (actual["rod"]["rod_mesh_contact_pairs"] != archive["rod"]["rod_mesh_contact_pairs"]
            or actual["rod"]["hand_wrenches_com_world"] != archive["rod"]["hand_wrenches_com_world"]
            or actual["issued_composer_wrench"] != archive["issued_composer_wrench"]
            or actual["wrist_projection"] != archive["wrist_projection"]
            or current["guard_failure"] is not None):
        raise ValueError("live Power anchor must reproduce the sealed reaction and projected targets")


class SupportedPowerHandoffProbe:
    """Keep the prefix probe intact, then use actual phase-scoped source and cooked guards."""

    archive = bilateral = None

    def __init__(self, prefix_probe, meshes, shapes, request, scene, *, project):
        from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, load_geometry

        self.prefix = prefix_probe
        self.boundary_sdk = request["anchor_sdk"]
        self.anchor = request["archive"]
        self.phases = request["phases"]
        self.stations = request["final_stations"]
        if scene["wire_radius_m"] != .005:
            raise ValueError("the sealed Power handoff is specific to the original 10 mm wire")
        self.scene = scene
        self.previous = None
        self.anchor_matches = False
        self._geometry_sdk = None
        self.checked_steps = 0
        self.reaction_steps = 0
        self.terminal = None
        self.boxes = [(x["name"], np.asarray(x["center_m"])-np.asarray(x["size_m"])/2,
                       np.asarray(x["center_m"])+np.asarray(x["size_m"])/2) for x in scene["supports"]]
        self.shape_names = {}
        self.grids = dict(prefix_probe.grids)
        self.cooked = {}
        target_shape_keys(meshes, shapes, self.stations, allowed_links=("base_link", *DISTAL_PAD_LINKS))
        for side in SIDES:
            model = load_geometry(project, side=side)
            for link in ("base_link", *DISTAL_PAD_LINKS):
                name = f"{side}.{link}"
                entry = meshes[name]
                self.shape_names[SIDES.index(side), entry["body_index"]] = name
                if link == "base_link":
                    validate_palm_lattice(self.grids[name], radius=scene["wire_radius_m"])
                    continue
                locked = model["cooked_pads"][link]
                if (entry["capture_sha256"] != locked["capture_sha256"]
                        or entry["source_geometry_sha256"] != locked["geometry_sha256"]):
                    raise ValueError(f"live shape binding does not match locked Power hull: {name}")
                self.cooked[name] = locked["vertices"]
                path = Path(entry["sdf_path"])
                if native.sha(path) != entry["sdf_sha256"]:
                    raise ValueError(f"live Power SDF fingerprint differs: {name}")
                with np.load(path, allow_pickle=False) as values:
                    self.grids[name] = {key: values[key].copy() for key in ("values", "origin", "spacing")}

    def continuing(self, sdk):
        return sdk > self.boundary_sdk

    def active(self, sdk):
        return self.continuing(sdk) or self.prefix.active(sdk)

    def _phase(self, sdk):
        for phase in self.phases:
            if sdk <= phase["end_sdk"]:
                return phase
        raise ValueError("Power contact step exceeds the frozen handoff")

    def contact_shape(self, sdk, key):
        if not self.continuing(sdk):
            return self.prefix.contact_shape(sdk, key)
        return self.shape_names.get(key) in self._phase(sdk)["stations"]

    def check_live_shape(self, sdk, world, nodes, *, hand, body=None, name=None):
        if name is None:
            name = self.shape_names.get((hand,body))
        selected = name is not None and name in self._phase(sdk)["stations"]
        gaps = {key: float(support_clearance_lower(world, low, high)) for key, low, high in self.boxes}
        conditions = ["support_clearance"] if min(gaps.values()) <= CLEARANCE_M else []
        if float((world[:, 1] * (1. if hand == 0 else -1.)).min()) <= 0.:
            conditions.append("hand_side")
        lower = None
        if not (selected and name.endswith(".base_link")):
            minimum = -PAD_PENETRATION_M if selected else NON_TARGET_CLEARANCE_M
            radius = self.scene["wire_radius_m"]
            lower = rod_support_lower(world, nodes[0])
            if lower <= minimum:
                lower = max(lower, _pad_gap_interval(world, nodes[0], radius)["lower_m"])
            if lower <= minimum:
                lower = max(lower, endpoint_gap(world, nodes[0], radius)["lower_m"])
            if not np.isfinite(lower) or lower <= minimum:
                conditions.append("contact_rod_penetration" if selected else "rod_clearance")
        if conditions:
            return dict(conditions=conditions, support_gap_m=gaps, rod_gap_lower_m=lower,
                        selected_distal_shape=bool(selected), selected_mesh=name)
        return None

    def check_pad_gaps(self, sdk, poses, nodes):
        if not self.continuing(sdk):
            self.prefix.check_pad_gaps(sdk, poses, nodes)
            return
        if np.shape(nodes) != (1,41,3) or not np.isfinite(nodes).all():
            raise ValueError("complete finite live Power nodes required")
        for name, vertices in self.cooked.items():
            position, quat = poses[name]
            world = Rotation.from_quat(quat).apply(vertices) + position
            failure = self.check_live_shape(sdk, world, nodes, hand=SIDES.index(name.split(".")[0]), name=name)
            if float(world[:,2].min()) < self.scene["table_top_m"] + CLEARANCE_M or failure is not None:
                raise ValueError(f"live locked cooked Power hull guard: {name} {failure}")
        for name in self._phase(sdk)["stations"]:
            position, quat = poses[name]
            transform = np.eye(4)
            transform[:3,:3] = Rotation.from_quat(quat).as_matrix()
            transform[:3,3] = position
            gap = sample_contact(self.grids[name], transform, nodes[0],
                                 radius=self.scene["wire_radius_m"], dtype=np.float64)["minimum_gap_m"]
            palm = name.endswith(".base_link")
            limit = MAX_PALM_PENETRATION_M if palm else PAD_PENETRATION_M
            if (palm and gap is None) or (gap is not None and (not np.isfinite(gap) or gap < -limit)):
                raise ValueError(f"live registered Power SDF guard: {name} gap={gap!r}")

    def check_completed_rod(self, row, live, remeasure):
        sdk = row["physics_step"]
        if not self.continuing(sdk) or self._geometry_sdk is not None:
            raise ValueError("one live geometry recheck required per continuing SDK exchange")
        verify_same_rod(row, live)
        if remeasure()["physics_step"] != sdk:
            raise ValueError("guard moved the SDK step during the live rod recheck")
        self._geometry_sdk = sdk

    def observe(self, row):
        sdk = row["physics_step"]
        if not self.continuing(sdk):
            evidence, reason = self.prefix.observe(row)
            if sdk == self.boundary_sdk:
                verify_anchor(row, self.anchor, sdk)
                self.previous = row
                self.anchor_matches = True
            return evidence, reason

        if self._geometry_sdk != sdk:
            raise ValueError("completed live geometry required at the current SDK tick")
        self._geometry_sdk = None
        prior = self.previous
        if not self.anchor_matches or prior is None:
            raise ValueError("live palm anchor required before Power contact continuation")
        rod = row["rod"]
        before, now = _clock(lambda: prior["rod"]), _clock(lambda: rod)
        if (any(before[k] != now[k] for k in IDENTITY[:5]) or now["physics_step"] != sdk
                or now["rod_tick"] != before["rod_tick"]+SUBSTEPS):
            raise ValueError("continuing Power exchanges require the same live episode and adjacent solver ticks")
        transferred = check_reaction_delivery(prior, row)
        self.previous = row
        self.checked_steps += 1
        self.reaction_steps += int(transferred)
        phase = self._phase(sdk)
        pairs = rod["rod_mesh_contact_pairs"]["pairs"]
        violations = pairs_outside_station(pairs, phase["stations"])
        supported = {(p["mesh_name"],p["substep"]) for p in pairs if p["normal_force_on_rod_n"] > 0.
                     and p["mesh_name"] in END_SUPPORT_NAMES}
        retains_supports = supported == {(name,tick) for name in END_SUPPORT_NAMES for tick in range(SUBSTEPS)}
        blocked = support_adjacent_nodes(rod, rod["node_positions"][0], self.scene)
        nearby = sorted({f"{p['mesh_name']}:{p['node_index']}" for p in pairs
                         if p["mesh_name"] in phase["stations"] and blocked[p["node_index"]]})
        reason = None
        if violations or nearby:
            reason = "power_contact_outside_owned_unobstructed_material_stations"
        elif not retains_supports:
            reason = "power_partial_contact_lost_calibration_support"
        elif sdk == self.phases[-1]["end_sdk"]:
            self.terminal = confirm_local_opposition(prior, row, self.stations, scene=self.scene)
            if not self.terminal["local_opposition_with_reaction"]:
                reason = "power_terminal_local_opposition_not_sustained"
        return dict(sdk_step=sdk, stage=phase["stage"], completed_live_rod_guard=True,
                    station_violations=violations, support_adjacent_pairs=nearby, supports_retained=retains_supports,
                    previous_reaction_transferred=transferred, terminal_opposition=self.terminal), reason

    def summary(self):
        return dict(scope="power_supported_partial_contact", anchor_matches=self.anchor_matches,
                    prefix=self.prefix.summary(), live_geometry_and_reaction_steps=self.checked_steps,
                    nonzero_reaction_transfer_steps=self.reaction_steps, terminal=self.terminal,
                    local_opposition_with_reaction=bool(self.terminal and self.terminal["local_opposition_with_reaction"]),
                    full_wrap_verified=False, grasp_qualified=False, teacher_admission=False)
