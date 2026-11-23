"""Refit a left Power partial grasp in the locked right-approach boundary."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_polyline_closest_support import refine
from scripts import prepare_dh116_power_preload as preload
from scripts import search_dh116_power_surface_pairs as search
from scripts.audit_dh116_pinch_initial_pose import checked_json
from scripts.audit_dh116_wire_contact_witness import endpoint_gap, shapes_world
from scripts.certify_dh116_power_right_loaded_target import candidate_for_boundary
from scripts.prepare_dh116_pinch_closure_margin import fk_poses
from scripts.screen_dh116_power_partial_opposition import rod_clearance_required

REFIT_READINESS = "docs/contracts/dh116_teacher_power_free_wrist_refit_v1.json"
RIGHT_READINESS = "docs/contracts/dh116_teacher_power_right_concurrent_route_v1.json"
NEAR_ROD_M = .003
TRUST_SCALE = np.array([.006] * 3 + [.12] * 9)
CONTACT_FIT_M = np.array([-.000015, -.000006, -.000015])  # Palm, thumb, index.


def projected_segment_margin(vertices, segment, normal, *, radius, required):
    """Conservative gap reserve to the entire capsule axis segment."""
    return float(np.min(vertices @ normal) - np.max(segment @ normal) - radius - required)


def owned_station_constraints(gaps, *, owned_node):
    """Shallow negative displacement on the owned node only, in metres."""
    gap = gaps[owned_node]
    return np.r_[np.delete(gaps, owned_node) - .000005, -gap - .000002, gap + .0001]


def bounded_seeds(packet, *, count):
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError("one to three archived Power seed poses required")
    seeds = []
    for item in packet["clearance_refits"]:
        for trial in item["attempts"]:
            limiting = trial.get("geometry", {}).get("limiting_rod")
            if limiting:
                deficit = limiting["required_m"] - limiting["lower_m"]
                seeds.append((deficit, item["ray_indices"], trial))
    if not seeds:
        raise ValueError("left archive has no rejected poses with recorded clearance bounds")
    return [(rays, trial) for _, rays, trial in sorted(seeds, key=lambda row: row[0])[:count]]


def near_rod_planes(hand, state, nodes, scene, targets):
    """Cover all currently near hulls, including adjacent material segments."""
    pose = search.route.root_pose(state, hand["seed"])
    hulls = shapes_world(hand["model"], pose, state[6:])
    radius = scene["wire_radius_m"]
    planes = {}
    for label, vertices in hulls.items():
        if label == "source:base_link":
            continue
        required = rod_clearance_required(label, targets)
        interval = endpoint_gap(vertices, nodes, radius)
        if interval["lower_m"] > required + NEAR_ROD_M:
            continue
        for row in refine(vertices, nodes, max_segments=12)["segments"]:
            if row["lower_m"] > required + NEAR_ROD_M:
                continue
            proposal = row["proposal"]
            updated = proposal["new_normal"] if proposal else None
            planes[label, row["segment"]] = np.asarray(row["normal"] if updated is None else updated)
    return planes


def refit_left(hand, seed, nodes, scene):
    """Use local constraints for fitting, then reapply the full original gate."""
    radius = scene["wire_radius_m"]
    stations = hand["preload_targets"]
    targets = (preload.THUMB, "finger23_link")
    order = ("base_link", *targets)
    if set(stations) != set(order):
        raise ValueError("left palm/thumb/index material station mapping required")
    if seed.shape != (12,) or not np.isfinite(seed).all():
        raise ValueError("finite archived 12D hand target required")

    model = hand["model"]
    shape_lists = {"source": model["source_colliders"], "cooked": model["cooked_pads"]}
    fields = {link: search.grid_field(hand["grids"][link], track_outside=True) for link in order}
    planes = near_rod_planes(hand, seed, nodes, scene, targets)
    low, high = search.active_limits()
    angle_limit = search.route.WRIST["rotation_limit_rad"]
    lo = np.r_[hand["origin"] + search.route.WRIST["palm_lower_m"], [-angle_limit] * 3, low]
    hi = np.r_[hand["origin"] + search.route.WRIST["palm_upper_m"], [angle_limit] * 3, high]
    limits = np.c_[np.maximum((lo - seed) / TRUST_SCALE, -1.), np.minimum((hi - seed) / TRUST_SCALE, 1.)]
    cache = {}

    def evaluate_trial(x):
        key = x.tobytes()
        if key in cache:
            return cache[key]
        state = seed + x * TRUST_SCALE
        frames = fk_poses(model, search.route.root_pose(state, hand["seed"]), state[6:])
        world = {}
        for kind, colliders in shape_lists.items():
            for link, shape in colliders.items():
                t = frames[link]
                world[f"{kind}:{link}"] = shape["vertices"] @ t[:3, :3].T + t[:3, 3]
        margins = [
            projected_segment_margin(world[label], nodes[i:i + 2], normal, radius=radius,
                                     required=rod_clearance_required(label, targets)) - .000002
            for (label, i), normal in planes.items()
        ]
        margins.extend(float(points[:, 2].min()) - scene["table_top_m"] - CLEARANCE_M - .000005
                       for points in world.values())

        depth, normal_vectors, own_gaps = [], [], []
        for link in order:
            frame = frames[link]
            if link != "base_link":
                frame = frame @ hand["pad_offsets"][link]
            field = fields[link]
            local = (nodes - frame[:3, 3]) @ frame[:3, :3]
            owned = stations[link]
            gaps = field(local) - radius
            vector = search.sliding_world_normal(field, local[owned], frame[:3, :3])
            normal_vectors.append(np.zeros(3) if vector is None else vector)
            own_gaps.append(gaps[owned])
            depth.extend(owned_station_constraints(gaps, owned_node=owned))

        cosine = normal_vectors[1] @ normal_vectors[2]
        result = (np.r_[margins, depth, (search.OPPOSITION_COS_LIMIT - cosine) * radius] * 1e3,
                  np.asarray(own_gaps) * 1e3)
        if len(cache) > 60:
            cache.clear()
        cache[key] = result
        return result

    def objective(x):
        gaps = evaluate_trial(x)[1] - CONTACT_FIT_M * 1e3
        return gaps @ gaps + .000001 * (x @ x)

    initial = np.clip(np.zeros(12), limits[:, 0], limits[:, 1])
    fit = minimize(objective, initial, bounds=limits,
                   constraints=[dict(type="ineq", fun=lambda x: evaluate_trial(x)[0])],
                   method="SLSQP", options=dict(maxiter=180, ftol=1e-11))
    state = seed + fit.x * TRUST_SCALE
    status, receipt = search.evaluate(hand, state, nodes, scene, side="left")
    result = dict(state=state.tolist(), result=status, hard_endpoint=receipt,
                  optimizer_converged=bool(fit.success), optimizer_iterations=int(fit.nit),
                  optimizer_message=str(fit.message), min_fit_margin_m=float(evaluate_trial(fit.x)[0].min() * 1e-3),
                  segment_plane_count=len(planes),
                  segments_by_hull={shape: [i for name, i in planes if name == shape]
                                    for shape in dict.fromkeys(name for name, _ in planes)})
    if status == "accepted_static":
        hulls = shapes_world(model, search.route.root_pose(state, hand["seed"]), state[6:])
        result["accepted_pad_envelopes"] = {
            label: endpoint_gap(hulls[label], nodes, radius)
            for name in targets for label in (f"source:{name}", f"cooked:{name}")
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("palm-receipt", "continuous-audit", "partial-audit", "free-wrist-audit", "right-route", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--seeds", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    previous = json.loads((project / REFIT_READINESS).read_bytes())
    right_ready = json.loads((project / RIGHT_READINESS).read_bytes())
    refit_sha = previous["archived_replay"]["artifact_sha256"]
    right_sha = right_ready["archived_proof"]["final_artifact_sha256"]
    old = checked_json(args.free_wrist_audit, refit_sha)
    right_route = checked_json(args.right_route, right_sha)
    certified, route_sha, _, nodes, scene, hands, stations, terminal = preload.load_inputs(
        args.palm_receipt, args.continuous_audit, args.partial_audit, project)
    right = candidate_for_boundary(old, certified, stations)
    route_states = np.asarray(right_route["target_waypoints"])
    if (not right_route["continuous_target_geometry_proved"] or not right_route["fixed_rod_only"]
            or right_route["archive"] != certified["archive"] or right_route["clock"] != certified["clock"]
            or right_route["source_stations"] != stations or right_route["right_refit_sha256"] != refit_sha
            or right_route["previous_route_sha256"] != route_sha
            or not np.array_equal(route_states[0], terminal)
            or not np.array_equal(route_states[-1], np.stack((terminal[0], right)))):
        raise ValueError("right continuation must bind both sides of the original terminal command")
    status, right_witness = search.evaluate(hands["right"], right, nodes, scene, side="right")
    if status != "accepted_static":
        raise ValueError("previous right static endpoint no longer meets the original gate")
    attempts = []
    for rays, seed in bounded_seeds(old["hands"]["left"], count=args.seeds):
        attempt = refit_left(hands["left"], np.asarray(seed["state"]), nodes, scene)
        attempts.append(dict(seed_rays=rays, seed_state=seed["state"], **attempt))
        print("left", rays, attempt["result"], attempt["optimizer_converged"], flush=True)
        if attempt["result"] == "accepted_static":
            break
    left = next((a for a in attempts if a["result"] == "accepted_static"), None)
    report = dict(version="dh116-power-left-contact-refit-v1", archive=old["archive"], clock=old["clock"],
                  source_stations=stations, free_wrist_sha256=refit_sha, right_continuous_sha256=right_sha,
                  left_attempts=attempts, right_state=right.tolist(), right_static_witness=right_witness,
                  bilateral_partial_static_passed=left is not None,
                  endpoint_states_left_then_right=[left["state"], right.tolist()] if left is not None else None,
                  fixed_rod_only=True, left_continuous_route_verified=False, both_forces_verified=False,
                  full_wrap_verified=False, native_execution_ready=False, native_runs=0, grasp_qualified=False,
                  teacher_admission=False, generalization_qualified=False)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, allow_nan=False, indent=2)
        output.write("\n")
    return 0 if left is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
