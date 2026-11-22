"""Independent wire profiles with a native radius, material and mass audit."""
from copy import deepcopy
import math

import numpy as np

from .tabletop_deformx_parameter_snapshot import validate_snapshot
from .tabletop_deformable_grasp_runtime import verify_mass_readback
from .tabletop_native_teacher_profile import digest


SIDES = ("left", "right")
MATERIAL_NODES = 41


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0.:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _scale_material(baseline, radius_m, length_m, *, elastic_modulus_scale=1.):
    """Scale circular geometry and optionally the candidate elastic modulus in SI."""
    modulus = _positive(elastic_modulus_scale, "elastic modulus scale")
    radius = _positive(radius_m, "wire radius")
    length = _positive(length_m, "wire length")
    scale = radius / _positive(baseline["radius_m"], "reference radius")
    rod = deepcopy(baseline)
    rod["radius_m"] = radius
    for field, factor in (("linear_density_kg_m", scale**2), ("axial_stiffness_n", scale**2 * modulus),
                          ("bend_stiffness_n_m2", scale**4 * modulus)):
        rod[field] = _positive(_positive(baseline[field], field) * factor, field)
    return dict(length_m=length, rod=rod)


def _solver_key(*values):
    with np.errstate(over="ignore", under="ignore"):
        key = np.asarray(values, dtype=np.float32)
    if not np.isfinite(key).all() or (key <= 0).any():
        raise ValueError("positive finite physical parameters at solver precision required")
    return tuple(key)


def group_wire_specs(rows, baseline, *, independent_stiffness=False):
    """Group immutable physical candidates and isolate geometry and modulus by split."""
    groups = {"train": [], "validation": [], "test": []}
    physical_specs, names = set(), set()
    geometry_splits, modulus_splits = {}, {}
    fields = {"id", "split", "radius_m", "length_m"}
    if independent_stiffness:
        fields.add("elastic_modulus_scale")
    for row in rows:
        if (set(row) != fields or row["split"] not in groups
                or not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in names):
            raise ValueError("distinct named wire candidates and explicit train/validation/test splits required")
        split = row["split"]
        modulus = row.get("elastic_modulus_scale", 1.)
        spec = _scale_material(baseline, row["radius_m"], row["length_m"],
                               elastic_modulus_scale=modulus)
        rod = spec["rod"]
        geometry = _solver_key(rod["radius_m"], spec["length_m"])
        physical = geometry + _solver_key(rod["linear_density_kg_m"], rod["axial_stiffness_n"],
                                          rod["bend_stiffness_n_m2"])
        if physical in physical_specs:
            raise ValueError("identical physical specification at solver precision assigned more than once")
        if geometry_splits.setdefault(geometry, split) != split:
            raise ValueError("wire geometry assigned across splits")
        if independent_stiffness and modulus_splits.setdefault(_solver_key(modulus), split) != split:
            raise ValueError("wire stiffness assigned across splits")
        physical_specs.add(physical)
        names.add(row["id"])
        case = dict(**spec, id=row["id"], spec_sha256=digest(spec))
        if independent_stiffness:
            case["elastic_modulus_scale"] = modulus
        groups[split].append(case)
    if not groups["train"] or not groups["test"] or (independent_stiffness and not groups["validation"]):
        raise ValueError("training and held-out wire specification groups required")
    return groups


def catalog_wire_specs(catalog, baseline):
    """Load the original size-only catalog or the opt-in size and modulus catalog."""
    mode = catalog["version"], catalog["material_scaling"]
    known = {
        ("dh116-supported-wire-generalization-v1", "same_homogeneous_circular_candidate_material"): False,
        ("dh116-supported-wire-generalization-v2", "circular_candidate_with_independent_elastic_modulus"): True,
    }
    if mode not in known:
        raise ValueError("known opt-in circular wire candidate catalog required")
    return group_wire_specs(catalog["variants"], baseline, independent_stiffness=known[mode])


def _validate_stations(stations):
    if (set(stations) != set(SIDES)
            or any(type(index) is not int or not (20 < index < MATERIAL_NODES if side == "left" else 0 <= index < 20)
                   for side, index in stations.items())):
        raise ValueError("explicit material station in each hand's original half required")


def build_variant(scene, baseline, *, radius_m, length_m, grasp_material_ids, elastic_modulus_scale=1.):
    """Build a separate solver/scene candidate. A native Reset still has to qualify it."""
    spec = _scale_material(baseline, radius_m, length_m, elastic_modulus_scale=elastic_modulus_scale)
    radius, length = spec["rod"]["radius_m"], spec["length_m"]
    geometry = deepcopy(scene)
    support = geometry["support_scene"]
    if support["wire_radius_m"] != baseline["radius_m"]:
        raise ValueError("one matching baseline solver and scene required")
    _validate_stations(grasp_material_ids)
    nodes = np.asarray(scene["rod_nodes_w_m"], dtype=float)
    if nodes.shape != (MATERIAL_NODES, 3) or not np.isfinite(nodes).all():
        raise ValueError("41 finite original reset material positions required")
    intervals = np.diff(nodes, axis=0)
    old_length = _positive(support["wire_length_m"], "reference length")
    axis = (nodes[-1] - nodes[0]) / old_length
    if (abs(axis[2]) > 1e-9 or not math.isclose(np.linalg.norm(axis), 1., rel_tol=0., abs_tol=1e-7)
            or not np.allclose(intervals, axis * old_length / (MATERIAL_NODES-1), rtol=0., atol=1e-7)
            or not np.allclose(nodes[:, 2] - baseline["radius_m"],
                               support["lift_reference_bottom_m"], rtol=0., atol=1e-7)):
        raise ValueError("straight, horizontal, resting reference wire required")
    base = nodes.mean(axis=0)
    shift_z = radius - baseline["radius_m"]
    relative_scale = length / old_length - 1.
    new_nodes = base + (nodes - base) * (1. + relative_scale)
    new_nodes[:, 2] += shift_z
    spans = []
    for box in support["supports"]:
        center = np.asarray(box["center_m"], dtype=float)
        size = np.asarray(box["size_m"], dtype=float)
        if center.shape != (3,) or size.shape != (3,) or not np.isfinite([center, size]).all() or np.any(size <= 0.):
            raise ValueError("finite original support box geometry required")
        moved = center + relative_scale * np.dot(center - base, axis) * axis
        if not np.any((np.abs(new_nodes[:, :2] - moved[:2]) <= size[:2]/2.).all(axis=1)):
            raise ValueError("resized rod material stations no longer rest on both supports")
        box["center_m"] = moved.tolist()
        spans.append((moved, size))
    if len(spans) != 2 or np.all(np.abs(spans[0][0] - spans[1][0]) <= (spans[0][1] + spans[1][1])/2.):
        raise ValueError("new wire geometry requires two disjoint supports")
    for side, index in grasp_material_ids.items():
        pose = geometry["hands"][side]["pose_wxyz"]
        pose[:3] = (np.asarray(pose[:3], dtype=float) + new_nodes[index] - nodes[index]).tolist()
    geometry["rod_nodes_w_m"] = new_nodes.tolist()
    support.update(wire_radius_m=radius, wire_length_m=length)
    return dict(scene=geometry, **spec, spec_sha256=digest(spec), native_reset_verified=False)



def prepare_variant(scene, table, baseline, *, radius_m, length_m, stations, elastic_modulus_scale=1.):
    """Reuse the original hand seed and the independently prepared variant geometry."""
    from .tabletop_dual_hand_reset import prepare_reset_state
    from .tabletop_power_shared_rod import END_SUPPORT_NAMES
    from .tabletop_power_supported_task import TRIPOD_SCOPE, prepare_initial_state

    baseline_seed, active, _, _ = prepare_initial_state(scene, table, expected_scope=TRIPOD_SCOPE)
    variant = build_variant(scene, baseline, radius_m=radius_m, length_m=length_m,
                            grasp_material_ids=stations, elastic_modulus_scale=elastic_modulus_scale)
    config = variant["scene"]
    seed = prepare_reset_state(
        hand_roots={side: config["hands"][side]["pose_wxyz"] for side in SIDES},
        **{key: baseline_seed[key] for key in ("origin", "root_xy", "initial_offset_m",
                                               "joint_positions", "joint_velocities")},
        source_description=variant["spec_sha256"])
    support_poses = {
        name: (np.asarray(box["center_m"]), np.array([0., 0., 0., 1.]))
        for name, box in zip(END_SUPPORT_NAMES, config["support_scene"]["supports"], strict=True)
    }
    return seed, active, np.asarray(config["rod_nodes_w_m"], dtype=np.float32), support_poses, variant


def verify_wire_getter(spec, snapshot, *, solver_time_s):
    """Match the actual solver's raw parameter getter to the declared variant."""
    verified = validate_snapshot(snapshot, expected_num_envs=1, expected_nodes=MATERIAL_NODES,
                                 expected_time_s=solver_time_s)
    rod = spec["rod"]
    rest = np.asarray(snapshot["rest_lengths_m"], dtype=float)[0]
    expected = {
        "contact_radius_m": rod["radius_m"],
        "gravity_world_m_s2": rod["gravity_world_m_s2"],
        "contact_mu_static": rod["mu_s"],
        "contact_mu_kinetic": rod["mu_k"],
        "contact_v_slip_m_s": rod["v_slip_m_s"],
        "contact_kappa": rod["kappa"],
        "stretch_ea_n": rod["axial_stiffness_n"],
        "bend_ei_n_m2": rod["bend_stiffness_n_m2"],
    }
    if (verified["fixed_node_indices"] or not math.isclose(rest.sum(), spec["length_m"], rel_tol=2e-6)
            or any(not np.allclose(snapshot[key], value, rtol=2e-6, atol=1e-9)
                   for key, value in expected.items())):
        raise ValueError("actual free wire getter differs from the declared physical specification")
    density = rod["linear_density_kg_m"]
    masses = np.r_[rest[0]/2, (rest[:-1] + rest[1:])/2, rest[-1]/2] * density
    mass = verify_mass_readback(np.asarray(snapshot["inverse_masses_kg_inv"]), masses)["mass_kg"]
    weight = float(mass * np.linalg.norm(snapshot["gravity_world_m_s2"]))
    return dict(measured_radius_m=float(snapshot["contact_radius_m"]), rest_length_m=float(rest.sum()),
                total_mass_kg=mass, weight_n=weight,
                evidence_scope="native_parameter_getter_readback", hardware_measured=False)


def observe_wire_pose_and_reaction(exchange, spec, *, grasp, grasp_material_ids):
    """Measure the candidate inputs for posture and force refinement, beside the actor."""
    from scipy.spatial.transform import Rotation

    from .tabletop_bimanual_table_contacts import _clock
    from .tabletop_power_contact_evidence import measure_power_contact, measure_supported_distal_contact
    from .tabletop_teacher_lift_progress import _palm_poses

    _validate_stations(grasp_material_ids)
    rod, parameters = exchange["rod"], exchange["material_director"]
    clock = _clock(lambda: rod)
    nodes = np.asarray(rod["node_positions"])
    if (nodes.shape != (1, MATERIAL_NODES, 3) or not np.isfinite(nodes).all()
            or any(parameters["identity"].get(key) != value for key, value in clock.items() if key != "clock_valid")
            or not np.array_equal(parameters["node_positions_m"], nodes)
            or not np.array_equal(parameters["raw_native_parameters"]["rest_lengths_m"],
                                  np.asarray(rod["rest_lengths_m"])[None])):
        raise ValueError("same-physics rod, material and parameter identity required")
    readback = verify_wire_getter(spec, parameters["raw_native_parameters"], solver_time_s=clock["solver_time_s"])
    if readback["weight_n"] <= 0.:
        raise ValueError("measured load normalization requires positive rod weight")
    if grasp == "power":
        contact = measure_power_contact(rod)
        hands, link_field = contact["hands"], "links"
        y = nodes[0, :, 1]
        if y[0] >= y[-1]:
            raise ValueError("supported wire material orientation must run from negative to positive World Y")
        center = (y[0] + y[-1]) / 2.
        wrong_side = []
        for pair in rod["rod_mesh_contact_pairs"]["pairs"]:
            link, index = pair["mesh_name"], pair["node_index"]
            if link in rod["mesh_bindings"]:
                side = rod["mesh_bindings"][link]["side"]
                sign = 1 if side == "left" else -1
                if sign * (index - MATERIAL_NODES // 2) <= 0 or sign * (y[index] - center) <= 0:
                    wrong_side.append(dict(link=link, node_index=index, substep=pair["substep"]))
        unexpected = []
        bilateral = contact["all_substeps_power_contacts"]
    elif grasp in ("pinch", "tripod"):
        contact = measure_supported_distal_contact(rod, grasp=grasp)
        hands, link_field = contact["per_hand"], "per_pad"
        wrong_side = contact["wrong_side_contacts"]
        unexpected = contact["unexpected_hand_contacts"]
        bilateral = contact["bilateral"]
    else:
        raise ValueError("independent pinch, tripod or power measurement required")
    measurements = {}
    for side, (position, xyzw) in zip(SIDES, _palm_poses(exchange, clock["physics_step"]), strict=True):
        if (np.shape(position) != (3,) or np.shape(xyzw) != (4,)
                or not np.isfinite(position).all()
                or not math.isclose(np.linalg.norm(xyzw), 1., abs_tol=1e-5, rel_tol=0.)):
            raise ValueError("same-clock finite palm position and unit XYZW quaternion required")
        inv = Rotation.from_quat(xyzw).inv()
        index = grasp_material_ids[side]
        tangent = nodes[0, min(index+1, MATERIAL_NODES-1)] - nodes[0, max(0, index-1)]
        length = np.linalg.norm(tangent)
        if not length > 0.:
            raise ValueError("current material station has no tangent")
        substeps = hands[side]["per_substep"]
        reaction = {}
        for name in substeps[0][link_field]:
            pad = [row[link_field][name] for row in substeps]
            loads = np.asarray([item["normal_load_n"] for item in pad])
            force = np.asarray([item["force_on_rod_world_n"] for item in pad]).mean(0)
            force_palm = inv.apply(force)
            reaction[name] = dict(minimum_normal_n=float(loads.min()), mean_normal_n=float(loads.mean()),
                                  mean_normal_over_weight=float(loads.mean() / readback["weight_n"]),
                                  mean_force_on_rod_palm_n=force_palm.tolist(),
                                  mean_force_on_hand_palm_n=(-force_palm).tolist())
        measurements[side] = dict(material_id=index, xyz_in_palm_m=inv.apply(nodes[0, index]-position).tolist(),
                                  tangent_in_palm=inv.apply(tangent/length).tolist(), reaction=reaction)
    reaction_valid = not wrong_side and not unexpected
    return dict(spec_sha256=digest(dict(rod=spec["rod"], length_m=spec["length_m"])), clock=clock,
                readback=readback, measured_hands=measurements, bilateral_opposition=bilateral and reaction_valid,
                reaction_valid=reaction_valid, wrong_side_contacts=wrong_side, unexpected_hand_contacts=unexpected,
                observation_scope="recorded_simulation_pose_and_reaction", teacher_admission=False)
