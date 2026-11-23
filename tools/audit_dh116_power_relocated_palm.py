"""Screen relocated palm and large-tilt right Power grasps on the archived rod."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import NUMERIC_GUARD
from scripts import certify_dh116_power_thumb_route as route
from scripts import prepare_dh116_power_preload as preload
from scripts import search_dh116_power_surface_pairs as search
from scripts.audit_dh116_pinch_initial_pose import checked_json, path_displacement_bound
from scripts.audit_dh116_wire_contact_witness import shapes_world
from scripts.prepare_dh116_pinch_closure_margin import fk_poses
from scripts.refit_dh116_power_left_contact import projected_segment_margin
from scripts.screen_dh116_power_measured_reseat import support_adjacent_nodes
from scripts.screen_dh116_power_partial_opposition import DISTAL_PAD_LINKS, endpoint, rod_clearance_required
from tools.audit_dh116_power_coupled_wrist import capsule_planes, material_frame
from tools.audit_dh116_power_remaining_digits import eligible_nodes
from tools.audit_dh116_power_right_index import INDEX, index_hulls


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "docs/contracts/dh116_teacher_power_bilateral_detour_v1.json"
PRIOR_SHA = "121cf3dde2c1762b701c79b4cfc086295c15ae4cd3b224394f49454b1d9da381"
PRIOR_TOOL_SHA = "9831fdb3dd7e97f1792355d49e89cea2fe003913046336f230c12069ff6b68cf"
RING = "finger43_link"
PADS = (preload.THUMB, RING, INDEX)
MIN_TILT_RAD = float(np.deg2rad(31.))
MAX_TILT_RAD = float(np.deg2rad(63.))
POSITION_LIMIT_M = .012
PALM_NODES = (7, 11, 13)
AZIMUTH_HALF_WIDTH_RAD = .40
ROLL_LIMIT_RAD = .42
ORIENTATION_CELLS = 12


def sdf_outer_box(grid, threshold):
    """Enclose every cell that trilinear interpolation can put below threshold."""
    values = np.asarray(grid["values"], dtype=float)
    spacing = float(grid["spacing"])
    origin = np.asarray(grid["origin"], dtype=float)
    if (values.ndim != 3 or min(values.shape) < 2 or not np.isfinite(values).all()
            or not np.isfinite(threshold) or not np.isfinite(spacing) or spacing <= 0.
            or origin.shape != (3,) or not np.isfinite(origin).all()):
        raise ValueError("a finite rectilinear SDF lattice is required")
    hits = np.nonzero(values <= threshold)
    if not len(hits[0]):
        raise ValueError("empty trilinear SDF sublevel")
    low = np.maximum([indices.min()-1 for indices in hits], 0)
    high = np.minimum([indices.max()+1 for indices in hits], np.asarray(values.shape)-1)
    return origin + spacing*np.asarray(list(product(*zip(low, high, strict=True))))


def chord_bounds(palm, indices, reserve):
    """Conservative range of distances between points in the two outer boxes."""
    p_low, p_high = palm.min(axis=0), palm.max(axis=0)
    lower, upper = [], []
    for box in indices:
        axes = np.subtract(box.mean(axis=0), palm.mean(axis=0))
        axes /= max(1e-12, float(np.linalg.norm(axes)))
        projected_gap = float((box @ axes).min() - (palm @ axes).max())
        box_gap = float(np.linalg.norm(np.maximum(
            np.maximum(p_low-box.max(axis=0), box.min(axis=0)-p_high), 0.)))
        lower.append(max(box_gap, projected_gap))
        upper.append(float(np.linalg.norm(box[:, None, :]-palm[None, :, :], axis=2).max()))
    return max(0., min(lower)-reserve-NUMERIC_GUARD), max(upper)+reserve+NUMERIC_GUARD


def family_state(start, seed, old_palm, new_palm, axes, shift, azimuth, tilt, roll):
    """Pivot the old hand onto a different material station with non-axial tilt."""
    transverse_axis = np.cos(azimuth)*axes[:, 0] + np.sin(azimuth)*axes[:, 2]
    rotation = Rotation.from_rotvec(roll*axes[:, 1]) * Rotation.from_rotvec(tilt*transverse_axis)
    new = start.copy()
    new[:3] = new_palm + rotation.apply(start[:3]-old_palm) + axes @ shift
    earlier = seed * Rotation.from_euler("XYZ", start[3:6])
    new[3:6] = (seed.inv() * rotation * earlier).as_euler("XYZ")
    return new, float(rotation.magnitude())


def sector_workspace(start, old_palm, new_palm, axes, workspace_low, workspace_high, azimuth):
    """Bound root position continuously over the search's entire orientation sector."""
    axes_limits = ((azimuth-AZIMUTH_HALF_WIDTH_RAD, azimuth+AZIMUTH_HALF_WIDTH_RAD),
                   (MIN_TILT_RAD, MAX_TILT_RAD), (-ROLL_LIMIT_RAD, ROLL_LIMIT_RAD))
    edges = [np.linspace(low, high, ORIENTATION_CELLS+1) for low, high in axes_limits]
    centers = [(edge[:-1]+edge[1:])/2 for edge in edges]
    az, tilt, roll = np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1).reshape(-1, 3).T
    direction = np.cos(az[:, None])*axes[:, 0] + np.sin(az[:, None])*axes[:, 2]
    rotation = Rotation.from_rotvec(roll[:, None]*axes[:, 1]) * Rotation.from_rotvec(tilt[:, None]*direction)
    lever = start[:3]-old_palm
    roots = new_palm+rotation.apply(lever)
    az_half, tilt_half, roll_half = (float(np.diff(edge).max()/2) for edge in edges)
    # Varying the tilt axis by delta azimuth conjugates the tilt by
    # tangent-axis rotations on both sides. The geodesic change is
    # bounded by twice delta azimuth plus the independent angle changes.
    angular_reserve = 2*az_half + tilt_half + roll_half
    reach = np.abs(axes).sum(axis=1)*POSITION_LIMIT_M + np.linalg.norm(lever)*angular_reserve + NUMERIC_GUARD
    excess = np.maximum(workspace_low-(roots+reach), (roots-reach)-workspace_high).max(axis=1)
    lower = float(excess.min())
    return dict(excluded=lower > 0., minimum_axis_excess_lower_m=lower,
                orientation_cells=len(roots), excluded_cell_count=int(np.count_nonzero(excess > 0.)),
                axial_azimuth_rad=list(axes_limits[0]), tilt_rad=list(axes_limits[1]), roll_rad=list(axes_limits[2]))


def validate_stations(stations, eligible, *, previous_palm):
    if set(stations) != {"base_link", *PADS} or stations["base_link"] == previous_palm:
        raise ValueError("a relocated palm and all three right-pad stations are required")
    if any(type(node) is not int or node < 0 or node >= len(eligible) or not eligible[node]
           for node in stations.values()):
        raise ValueError("all stations must be support-clear on the measured right physical material half")
    thumb, ring, index = (stations[link] for link in PADS)
    if (stations["base_link"] in (thumb, ring, index) or index in (thumb, ring) or abs(thumb-ring) > 1):
        raise ValueError("distinct palm/index stations and adjacent opposing thumb/ring stations required")


def palm_index_chord(hand, start, nodes, radius):
    model = hand["model"]
    angles, _, _ = index_hulls(hand, start)  # Also verifies independence of the index chain.
    pose = route.root_pose(start, hand["seed"])
    q = start[6:].copy()
    palm_pose = fk_poses(model, pose, q)["base_link"]
    palm = sdf_outer_box(hand["grids"]["base_link"], radius)
    palm = palm @ palm_pose[:3, :3].T + palm_pose[:3, 3]
    pad_box = sdf_outer_box(hand["grids"][INDEX], radius)
    align = hand["pad_offsets"][INDEX]
    pad_box = pad_box @ align[:3, :3].T + align[:3, 3]
    proof_model = dict(model, cooked_pads={INDEX: {"vertices": pad_box}})
    slot = DISTAL_PAD_LINKS.index(INDEX)+1
    q[slot] = angles[0]
    delta = np.zeros(6)
    delta[slot] = angles[-1]-angles[0]
    reserve = path_displacement_bound(proof_model, q, delta, cooked=True)[INDEX] / (2*(len(angles)-1))
    boxes = []
    for angle in angles:
        q[slot] = angle
        frame = fk_poses(model, pose, q)[INDEX]
        boxes.append(pad_box @ frame[:3, :3].T + frame[:3, 3])
    low, high = chord_bounds(palm, boxes, reserve)
    return dict(lower_m=low, upper_m=high, index_joint_interval_rad=angles[[0, -1]].tolist(),
                samples=len(angles), unsampled_travel_m=float(reserve))


def station_seed(hand, start, nodes, scene, eligible, chord, palm, azimuth, axes):
    """Rank native-grid contact proposals at one transported and tilted wrist seed."""
    radius = scene["wire_radius_m"]
    old_palm = nodes[hand["preload_targets"]["base_link"]]
    new_palm = nodes[palm]
    shift = np.zeros(3)
    tilt = np.deg2rad(42.)
    field = {link: search.grid_field(hand["grids"][link], track_outside=True) for link in ("base_link", *PADS)}
    model = hand["model"]
    for _ in range(3):
        trial, _ = family_state(start, hand["seed"], old_palm, new_palm, axes, shift, azimuth, tilt, 0.)
        pose = route.root_pose(trial, hand["seed"])
        frame = fk_poses(model, pose, trial[6:])["base_link"]
        local = (new_palm-frame[:3, 3]) @ frame[:3, :3]
        gap = float(field["base_link"](local)[0] - radius)
        normal = search.sliding_world_normal(field["base_link"], local, frame[:3, :3])
        if normal is None or abs(gap) < .00001:
            break
        shift = np.clip(shift + (axes.T @ normal)*(gap + .00002), -POSITION_LIMIT_M, POSITION_LIMIT_M)
        shifted, _ = family_state(start, hand["seed"], old_palm, new_palm, axes, shift, azimuth, tilt, 0.)
        proposed_root = shifted[:3]
        low = hand["origin"] + route.WRIST["palm_lower_m"]
        high = hand["origin"] + route.WRIST["palm_upper_m"]
        correction = np.clip(proposed_root, low, high)-proposed_root
        shift = np.clip(shift + axes.T @ correction, -POSITION_LIMIT_M, POSITION_LIMIT_M)
    trial, _ = family_state(start, hand["seed"], old_palm, new_palm, axes, shift, azimuth, tilt, 0.)
    pose = route.root_pose(trial, hand["seed"])
    origin = hand["origin"]
    wrist_excess = np.linalg.norm(np.maximum(
        np.maximum(origin+route.WRIST["palm_lower_m"]-trial[:3], trial[:3]-origin-route.WRIST["palm_upper_m"]), 0.))
    lower, upper = active_limits()
    options = {}
    for link in PADS:
        slot = DISTAL_PAD_LINKS.index(link)+1
        abductions = np.linspace(lower[0], upper[0], 5) if link == preload.THUMB else [trial[6]]
        score = np.full(len(nodes), np.inf)
        best_angles = np.zeros((len(nodes), 2))
        valid = eligible.copy()
        valid[palm] = False
        if link == INDEX:
            distance = np.linalg.norm(nodes-new_palm, axis=1)
            valid &= (distance >= chord["lower_m"]) & (distance <= chord["upper_m"])
        for abduction, angle in product(abductions, np.linspace(lower[slot], upper[slot], 33)):
            q = trial[6:].copy()
            q[0], q[slot] = abduction, angle
            frame = fk_poses(model, pose, q)[link] @ hand["pad_offsets"][link]
            absolute = np.abs(field[link]((nodes-frame[:3, 3]) @ frame[:3, :3])-radius+.000015)
            improve = valid & (absolute < score)
            score[improve] = absolute[improve]
            best_angles[improve] = [abduction, angle]
        options[link] = score, best_angles
    thumb_scores, ring_scores, index_scores = (options[link][0] for link in PADS)
    choices = []
    for thumb, ring, index in product(np.flatnonzero(np.isfinite(thumb_scores)),
                                      np.flatnonzero(np.isfinite(ring_scores)),
                                      np.flatnonzero(np.isfinite(index_scores))):
        if abs(thumb-ring) <= 1 and index not in (thumb, ring):
            choices.append((thumb_scores[thumb]+ring_scores[ring]+index_scores[index],
                            (int(thumb), int(ring), int(index))))
    if not choices:
        return None
    score, nodes_selected = min(choices)
    joints = trial[6:].copy()
    for link, node in zip(PADS, nodes_selected, strict=True):
        slot = DISTAL_PAD_LINKS.index(link)+1
        abduction, angle = options[link][1][node]
        if link == preload.THUMB:
            joints[0] = abduction
        joints[slot] = angle
    return dict(palm=palm, shift=shift, azimuth=azimuth, tilt=tilt, q=joints, score=float(score+2.*wrist_excess),
                stations={"base_link": palm, **dict(zip(PADS, nodes_selected, strict=True))})


def exact_gate(hand, state, nodes, scene, stations, turn):
    """Use the original locked geometry and real material SDF, independently of the fit."""
    relative = state[:3]-hand["origin"]
    workspace = route.WRIST
    within = bool(np.all(relative >= workspace["palm_lower_m"])
                  and np.all(relative <= workspace["palm_upper_m"])
                  and np.all(np.abs(state[3:6]) <= workspace["rotation_limit_rad"]))
    expand(state[6:])
    evidence, failures = {}, {}
    radius = scene["wire_radius_m"]
    for name, node in stations.items():
        try:
            item = preload.station_contact(hand, state, name, node, nodes, radius=radius)
            evidence[name] = item
            limit = preload.MAX_PALM_PENETRATION_M if name == "base_link" else preload.PAD_PENETRATION_M
            preload.checked_terminal_witness(item, node=node, max_penetration_m=limit)
        except ValueError as error:
            failures[name] = str(error)
    cosine = None
    if preload.THUMB in evidence and RING in evidence:
        normal, partner = (evidence[link]["normal_world"] for link in (preload.THUMB, RING))
        if normal is not None and partner is not None:
            cosine = float(np.dot(normal, partner))
    geometry = endpoint(hand["model"], route.root_pose(state, hand["seed"]), state[6:],
                        nodes, scene, hand["palm_grid"], targets=PADS)
    passed = (within and turn >= MIN_TILT_RAD and not failures and cosine is not None
              and cosine <= search.OPPOSITION_COS_LIMIT and geometry["passed"]
              and geometry["open_hull_guards"]["hand_y_bounds_m"][1] < 0.)
    return dict(passed=bool(passed), workspace_passed=within, relative_tilt_rad=turn,
                stations=evidence, station_failures=failures, thumb_ring_cosine=cosine, geometry=geometry)


def fit_family(hand, start, nodes, scene, axes, seed):
    radius = scene["wire_radius_m"]
    model = hand["model"]
    stations = seed["stations"]
    old_palm = nodes[hand["preload_targets"]["base_link"]]
    new_palm = nodes[seed["palm"]]
    fields = {name: search.grid_field(hand["grids"][name], track_outside=True) for name in stations}
    geometry = {f"{kind}:{name}": (name, shape["vertices"])
                for kind, group in (("source", model["source_colliders"]), ("cooked", model["cooked_pads"]))
                for name, shape in group.items()}
    lower, upper = active_limits()
    initial = np.r_[seed["shift"]/POSITION_LIMIT_M, seed["azimuth"], seed["tilt"], 0., seed["q"]]
    low = np.r_[[-1.]*3, seed["azimuth"]-AZIMUTH_HALF_WIDTH_RAD, MIN_TILT_RAD, -ROLL_LIMIT_RAD, lower]
    high = np.r_[[1.]*3, seed["azimuth"]+AZIMUTH_HALF_WIDTH_RAD, MAX_TILT_RAD, ROLL_LIMIT_RAD, upper]

    def state(x):
        trial, turn = family_state(start, hand["seed"], old_palm, new_palm, axes,
                                   x[:3]*POSITION_LIMIT_M, *x[3:6])
        trial[6:] = x[6:]
        return trial, turn

    initial_state, _ = state(initial)
    hulls = shapes_world(model, route.root_pose(initial_state, hand["seed"]), initial_state[6:])
    planes = capsule_planes(hulls, nodes, radius, PADS)

    def residual(x):
        trial, _ = state(x)
        frames = fk_poses(model, route.root_pose(trial, hand["seed"]), trial[6:])
        world = {label: vertices @ frames[name][:3, :3].T + frames[name][:3, 3]
                 for label, (name, vertices) in geometry.items()}
        margins = [projected_segment_margin(world[label], nodes[i:i+2], normal, radius=radius,
                    required=rod_clearance_required(label, PADS))
                   for (label, i), normal in planes.items()]
        workspace = route.WRIST
        offset = trial[:3]-hand["origin"]
        margins.extend(5.*np.r_[offset-workspace["palm_lower_m"], workspace["palm_upper_m"]-offset,
                                radius*(workspace["rotation_limit_rad"]-np.abs(trial[3:6]))])
        margins.extend(points[:, 2].min()-scene["table_top_m"]-CLEARANCE_M for points in world.values())
        margins.append(-max(points[:, 1].max() for points in world.values()))
        gaps, normals = [], {}
        for name, node in stations.items():
            frame = frames[name]
            if name != "base_link":
                frame = frame @ hand["pad_offsets"][name]
            local = (nodes-frame[:3, 3]) @ frame[:3, :3]
            gap = fields[name](local)-radius
            gaps.append(gap[node]+.000015)
            margins.extend(np.delete(gap, node)-.000002)
            normal = search.sliding_world_normal(fields[name], local[node], frame[:3, :3])
            normals[name] = np.zeros(3) if normal is None else normal
        margins.append(radius*(search.OPPOSITION_COS_LIMIT-normals[preload.THUMB] @ normals[RING]))
        return np.r_[np.asarray(gaps)*2e3, np.minimum(margins, 0.)*4e3, (x-initial)*.004]

    fit = least_squares(residual, initial, bounds=(low, high), max_nfev=110)
    trial, turn = state(fit.x)
    checked = exact_gate(hand, trial, nodes, scene, stations, turn)
    return dict(station_map=stations, seed_rank_cost_m=seed["score"], azimuth_seed_rad=seed["azimuth"],
                fitted_12d_target=trial.tolist(), least_squares_cost=float(fit.cost),
                evaluation_count=int(fit.nfev), capsule_plane_count=len(planes), exact_endpoint=checked)


def audit(args):
    checked_file(ROOT / "tools/audit_dh116_power_right_index.py", PRIOR_TOOL_SHA)
    previous = checked_json(args.right_index_screen, PRIOR_SHA)
    contract = json.loads((ROOT / CONTRACT).read_bytes())["replay"]
    detour = checked_json(args.bilateral_detour, contract["artifact_sha256"])
    base, base_sha, _, nodes, scene, hands, stations, _ = preload.load_inputs(
        args.palm_receipt, args.continuous_audit, args.partial_audit, ROOT)
    if (any(item["archive"] != base["archive"] or item["clock"] != base["clock"] for item in (previous, detour))
            or detour["base_route_sha256"] != base_sha or detour["source_stations"] != stations
            or detour["continuous_target_geometry_proved"] is not True or detour["fixed_rod_only"] is not True
            or previous["detour_sha256"] != contract["artifact_sha256"]
            or previous["certified_start_targets"] != detour["left_target_waypoints"][-1]
            or sum(detour["control_windows_by_stage"]) != 963):
        raise ValueError("the archive and exact certified bilateral endpoint must agree")
    start = np.asarray(previous["certified_start_targets"], dtype=float)
    left_result, _ = search.evaluate(hands["left"], start[0], nodes, scene, side="left")
    right, retained = search.evaluate(hands["right"], start[1], nodes, scene, side="right")
    if left_result != "accepted_static" or right != "accepted_static":
        raise ValueError("the original bilateral baseline failed exact verification")
    hand = hands["right"]
    row, _, _ = route.load_measured_palm_boundary(args.palm_receipt, project=ROOT)
    allowed = eligible_nodes(nodes, support_adjacent_nodes(row["rod"], nodes, scene), "right")
    if not all(allowed[node] and node != hand["preload_targets"]["base_link"] for node in PALM_NODES):
        raise ValueError("new palm stations must be support-clear on the exact measured rod half")
    radius = scene["wire_radius_m"]
    chord = palm_index_chord(hand, start[1], nodes, radius)
    node_pairs = {}
    trials, sectors = [], []
    old_station = hand["preload_targets"]["base_link"]
    low_box = hand["origin"]+route.WRIST["palm_lower_m"]
    high_box = hand["origin"]+route.WRIST["palm_upper_m"]
    for palm in PALM_NODES:
        axes = material_frame(nodes, palm, retained["sdf"]["base_link"]["normal_world"])
        distances = np.linalg.norm(nodes-nodes[palm], axis=1)
        compatible = (distances >= chord["lower_m"]) & (distances <= chord["upper_m"])
        distinct = allowed.copy()
        distinct[palm] = False
        node_pairs[palm] = dict(excluded_index_nodes=np.flatnonzero(distinct & ~compatible).tolist(),
                               compatible_index_nodes=np.flatnonzero(distinct & compatible).tolist())
        seeds = []
        for azimuth in (0., np.pi/2, -np.pi/2, np.pi):
            certificate = sector_workspace(start[1], nodes[old_station], nodes[palm], axes, low_box, high_box, azimuth)
            sectors.append(dict(palm_node=palm, azimuth_seed_rad=azimuth, **certificate))
            if certificate["excluded"]:
                continue
            seed = station_seed(hand, start[1], nodes, scene, allowed, chord, palm, azimuth, axes)
            if seed is not None:
                seeds.append(seed)
        for seed in sorted(seeds, key=lambda item: item["score"])[:2]:
            validate_stations(seed["stations"], allowed, previous_palm=old_station)
            trial = fit_family(hand, start[1], nodes, scene, axes, seed)
            trials.append(trial)
            print(trial["station_map"], trial["least_squares_cost"], trial["exact_endpoint"]["passed"], flush=True)
    count = sum(trial["exact_endpoint"]["passed"] for trial in trials)
    return dict(status="fixed_rod_static_candidate" if count else "blocked_relocated_palm_family",
                archive=base["archive"], clock=base["clock"], prior_result_sha256=PRIOR_SHA,
                detour_sha256=contract["artifact_sha256"], left_held_target=start[0].tolist(),
                old_station_map=hand["preload_targets"], new_palm_nodes=list(PALM_NODES), chord=chord,
                new_palm_index_pairs=node_pairs, orientation_sectors=sectors,
                trial_count=len(trials), trials=trials, exact_endpoint_count=count,
                minimum_non_axial_tilt_rad=MIN_TILT_RAD, native_runs=0, continuous_route_proved=False,
                actual_load_verified=False, full_wrap_verified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("palm-receipt", "continuous-audit", "partial-audit", "bilateral-detour", "right-index-screen", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["exact_endpoint_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
