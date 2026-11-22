"""Same-endpoint contact and fixed-Reset height readback for the supported rod."""
from dataclasses import dataclass
import math

import numpy as np

from .tabletop_bimanual_table_contacts import IDENTITY, _clock
from .tabletop_deformable_r0_replay import _digest
from .tabletop_native_teacher_contact_measurement import capture_supported_contact
from .tabletop_power_contact_evidence import (
    JOINT_CONTACT_LINKS, MIN_LINK_LOAD_N, SUBSTEPS, TABLE_LOAD_LIMIT_N,
    measure_power_contact,
)
from .tabletop_power_shared_rod import END_SUPPORT_NAMES, PROXIMAL_POWER_MESH_NAMES


_SIDES = ("left", "right")
_SUPPORTS = ("table", *END_SUPPORT_NAMES)
_EPOCH_KEYS = IDENTITY[:5]


def _points(rod):
    nodes = np.asarray(rod["node_positions"], dtype=np.float64)
    if nodes.shape != (1, 41, 3) or not np.isfinite(nodes).all():
        raise ValueError("complete finite world positions for 41 material nodes required")
    return nodes[0]


@dataclass(frozen=True, slots=True)
class FixedLiftReference:
    """Height origin from the original measured Reset endpoint."""

    epoch: tuple
    physics_step: int
    solver_time_s: float
    median_z_m: float
    minimum_z_m: float
    radius_m: float
    table_top_m: float
    reset_rod_sha256: str

    @classmethod
    def from_rod_reset(cls, rod, *, radius_m, table_top_m):
        clock = _clock(lambda: rod)
        if clock["reset_id"] < 1 or clock["rod_tick"] != 0:
            raise ValueError("original Reset before the first solver advance required")
        if (isinstance(radius_m, (bool, np.bool_)) or isinstance(table_top_m, (bool, np.bool_))
                or not math.isfinite(radius_m) or radius_m <= 0 or not math.isfinite(table_top_m)):
            raise ValueError("finite table height and positive rod radius required")
        z = _points(rod)[:, 2]
        return cls(tuple(clock[key] for key in _EPOCH_KEYS), clock["physics_step"],
                   clock["solver_time_s"], float(np.median(z)), float(np.min(z)),
                   float(radius_m), float(table_top_m), _digest(rod))


@dataclass(frozen=True, slots=True)
class SupportedLiftObservation:
    """Owned physical readback for shaping and diagnostics.

    The three registered supports are measured here. The episode's independent
    force, attachment, slip and stability evidence remains separately required.
    """

    clock: tuple
    grasp: str
    raw_rod_sha256: str
    reset_rod_sha256: str
    node_positions_world_m: tuple
    required_links: tuple
    coverage: tuple
    opposed: tuple
    bilateral: bool
    peak_support_load_n: tuple
    registered_support_clear: bool
    wrong_side: bool
    additional_hand_links: tuple
    median_delta_m: float
    minimum_delta_m: float
    minimum_surface_z_m: float
    off_support_lift_m: float | None


def _power_facts(rod):
    evidence = measure_power_contact(rod)
    coverage, opposed = [], []
    for side in _SIDES:
        rows = evidence["hands"][side]["per_substep"]
        coverage.append(sum(
            all(row["links"][link]["normal_load_n"] > MIN_LINK_LOAD_N for row in rows)
            for link in JOINT_CONTACT_LINKS
        ) / len(JOINT_CONTACT_LINKS))
        opposed.append(all(row["loaded_opposed"] for row in rows))

    wrong_side = bool(evidence["wrong_side_contacts"])
    if wrong_side:
        coverage, opposed = [0., 0.], [False, False]
    required = tuple((side, JOINT_CONTACT_LINKS) for side in _SIDES)
    return (required, tuple(coverage), tuple(opposed), all(opposed),
            evidence["external_support_force_norm_n"], wrong_side)


def _distal_facts(rod, grasp):
    contact = capture_supported_contact(rod, grasp=grasp)
    hands = contact["evidence"]["per_hand"]
    required = tuple((side, tuple(rod["mesh_names"][slot].split(".", 1)[1]
                                  for slot in hands[side]["required_mesh_ids"])) for side in _SIDES)
    return (required, tuple(contact[f"coverage_{side}"] for side in _SIDES),
            tuple(contact[f"opposed_{side}"] for side in _SIDES), contact["bilateral_support"],
            contact["support_substep_load_n"], bool(contact["evidence"]["wrong_side_contacts"]))


def observe_supported_lift(rod, reference, *, grasp):
    """Read contact and height from one complete supported-rod solver packet."""
    if type(reference) is not FixedLiftReference:
        raise ValueError("fixed rod Reset reference required")
    if grasp not in ("pinch", "tripod", "power"):
        raise ValueError("independent pinch, tripod or power grasp required")
    clock = _clock(lambda: rod)
    if tuple(clock[key] for key in _EPOCH_KEYS) != reference.epoch:
        raise ValueError("rod and fixed height Reset identity differ")
    intervals = clock["physics_step"] - reference.physics_step
    if intervals <= 0 or clock["rod_tick"] != intervals * SUBSTEPS:
        raise ValueError("complete solver substeps since fixed Reset required")
    if clock["solver_time_s"] <= reference.solver_time_s:
        raise ValueError("solver clock must advance beyond Reset")
    if tuple(rod["mesh_names"]) != PROXIMAL_POWER_MESH_NAMES:
        raise ValueError("complete named 17-mesh capture with table and two end supports required")

    nodes = _points(rod)
    facts = _power_facts(rod) if grasp == "power" else _distal_facts(rod, grasp)
    required, coverage, opposed, bilateral, loads, wrong_side = facts
    required_names = {f"{side}.{link}" for side, links in required for link in links}
    additional = tuple(sorted({
        pair["mesh_name"] for pair in rod["rod_mesh_contact_pairs"]["pairs"]
        if pair["mesh_name"] not in required_names and pair["mesh_name"] not in _SUPPORTS
    }))
    peak = tuple((name, float(max(loads[name]))) for name in _SUPPORTS)
    support_clear = all(value <= TABLE_LOAD_LIMIT_N for _, value in peak)
    z = nodes[:, 2]
    median_delta = float(np.median(z)) - reference.median_z_m
    minimum_z = float(np.min(z))
    minimum_delta = minimum_z - reference.minimum_z_m
    surface_z = minimum_z - reference.radius_m
    lift = min(median_delta, minimum_delta) if (
        bilateral and support_clear and surface_z > reference.table_top_m
    ) else None
    return SupportedLiftObservation(tuple(clock[key] for key in IDENTITY), grasp, _digest(rod),
        reference.reset_rod_sha256, tuple(tuple(map(float, xyz)) for xyz in nodes), required,
        coverage, opposed, bilateral, peak, support_clear, wrong_side, additional, median_delta,
        minimum_delta, surface_z, lift)
