from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import read_pair_capture
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import measure_power_contact
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import (
    POWER_MESH_NAMES, PADS, PALM_NAMES, END_SUPPORT_NAMES,
)


def build(*, missing_palm=None, palm_aligned=False, table=False, proximal=False):
    def stamp(name):
        return hashlib.sha256(name.encode()).hexdigest()

    names = (*POWER_MESH_NAMES, *END_SUPPORT_NAMES, "left.finger52_link", "right.finger52_link") if proximal else POWER_MESH_NAMES
    static = {"table", *END_SUPPORT_NAMES} if proximal else {"table"}
    bindings = {name: dict(
        side=name.split(".")[0], body_name=name.split(".")[1], body_index=slot,
        prim_path="/World/" + name.replace(".", "/"),
        **{key: stamp(name + key) for key in
           ("source_usd_sha256", "source_geometry_sha256", "cooked_geometry_sha256", "sdf_sha256", "capture_sha256")},
    ) for slot, name in enumerate(names) if name not in static}
    slots = [i for i, name in enumerate(names) if name != missing_palm and (table or name not in static)]
    indices, vectors = [], []
    node_forces, mesh = np.zeros((41, 3)), np.zeros((len(names), 6))
    for step in range(12):
        for slot in slots:
            name = names[slot]
            sign = -1. if name in PALM_NAMES and not palm_aligned else 1.
            if name.endswith("finger14_link"):
                sign = -1.
            # Signed palm and thumb are reversed relative to the remaining fingers.
            direction = np.array([sign, 0., 0.])
            if name in PALM_NAMES and palm_aligned:
                direction = np.array([0., 1., 0.])
            if name == "table":
                direction = np.array([0., 0., 1.])
            force = .1 * direction
            indices.append((step, slot, slot))
            vectors.append([0., 0., 0., *direction, *force, 0., 0., 0.])
            node_forces[slot] += force / 12
            mesh[slot, :3] -= force / 12
    raw = dict(substeps=12, mesh_count=len(names), capacity=41*len(names)*12,
               attempted=len(indices), overflow=False, indices=np.asarray(indices),
               vectors=np.asarray(vectors),
               node_forces_on_rod_world_n=node_forces, mesh_wrenches_on_mesh_world=mesh)
    capture = read_pair_capture(SimpleNamespace(contact_pair_samples=lambda: raw), dt_s=1/480,
                                physics_step=3, rod_tick=12, mesh_names=names)
    capture["solver_buffer"] = {key: raw[key] for key in ("substeps", "mesh_count", "capacity", "attempted", "overflow")}
    rod = dict(run_uuid="native_capture", episode_uid="ep0", reset_id=1, env_id=0, rod_uid="rod0",
               physics_step=3, rod_tick=12, timestamp_s=3/480, clock_valid=True, contact_valid=True,
               source="RopeBatch.positions/velocities/contact_forces/mesh_wrenches",
               pair_capture_status="validated_solver_samples", mesh_names=list(names),
               mesh_bindings=bindings, rod_mesh_contact_pairs=capture,
               distributed_contact=node_forces[None, :, :],
               named_wrenches=dict(zip(names, mesh)), table_wrench_world=mesh[5])
    if proximal:
        rod["external_support_wrenches_world"] = {name: mesh[names.index(name)] for name in static}
    return rod


def test_power_treats_palms_as_hands_and_table_as_support():
    metrics = measure_power_contact(build())
    assert metrics["all_substeps_power_contacts"] and metrics["zero_table_load"]
    assert not metrics["grasp_qualified"] and not metrics["teacher_admission"]
    assert metrics["hands"]["left"]["per_substep"][0]["palm_normal_load_n"] == pytest.approx(.1)
    grounded = measure_power_contact(build(table=True))
    assert grounded["all_substeps_power_contacts"] and not grounded["zero_table_load"]


def test_missing_palm_cannot_be_replaced_by_five_distal_pads():
    metrics = measure_power_contact(build(missing_palm="right.base_link"))
    assert metrics["hands"]["left"]["loaded_opposed_substeps"] == list(range(12))
    assert not metrics["hands"]["right"]["loaded_opposed_substeps"]
    assert not metrics["all_substeps_power_contacts"]


def test_power_palm_needs_an_opposed_normal():
    data = measure_power_contact(build(palm_aligned=True))
    assert not data["all_substeps_power_contacts"]
    assert data["hands"]["left"]["per_substep"][0]["pad_opposition"]["effective"]


@pytest.mark.parametrize("failure", ["slots", "capacity", "overflow", "clock", "aggregate", "palm_source"])
def test_malformed_live_power_sources_are_rejected(failure):
    rod = build()
    capture = rod["rod_mesh_contact_pairs"]
    if failure == "slots":
        rod["mesh_names"][11] = "support_0"
    elif failure == "capacity":
        capture["solver_buffer"]["capacity"] -= 1
    elif failure == "overflow":
        capture["solver_buffer"]["overflow"] = True
    elif failure == "clock":
        capture["rod_tick"] += 1
    elif failure == "aggregate":
        rod["distributed_contact"][0, 0, 2] += 1.
    else:
        rod["mesh_bindings"]["left.base_link"]["capture_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        measure_power_contact(rod)

def test_proximal_load_cannot_replace_a_palm_or_a_distal_pad():
    state = build(proximal=True, missing_palm="right.base_link")
    result = measure_power_contact(state)
    assert result["registered_proximal_links"] == ["finger52_link"]
    assert not result["proximal_sdf_registered"] and not result["all_substeps_power_contacts"]
    assert result["proximal_contacts"]["right"][0]["normal_load_n"] == pytest.approx(.1)
    assert result["hands"]["right"]["per_substep"][0]["palm_normal_load_n"] == 0.
    assert not result["teacher_admission"]
