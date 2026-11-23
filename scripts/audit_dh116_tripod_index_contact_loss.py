"""Separate measured rod and fingertip motion at archived Tripod contact losses."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from scripts.prepare_dh116_pinch_closure_margin import sample_grid


PAD = "left.finger23_link"
SOLVER_SUBSTEPS = 12


def _validated_pose(pose):
    xyz, xyzw = (np.asarray(value, dtype=float) for value in pose)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("finite link origin required")
    if xyzw.shape != (4,) or not np.isclose(np.linalg.norm(xyzw), 1., atol=1e-6, rtol=0.):
        raise ValueError("measured link unit quaternion in XYZW required")
    return xyz, Rotation.from_quat(xyzw)


def _local(points, pose):
    xyz, rotation = _validated_pose(pose)
    return (points - xyz) @ rotation.as_matrix()


def gap_motion(grid, before_pose, after_pose, before_nodes, after_nodes, *, radius):
    """Evaluate the original material nodes under four endpoint combinations."""
    before_nodes = np.asarray(before_nodes, dtype=float)
    after_nodes = np.asarray(after_nodes, dtype=float)
    if (before_nodes.ndim != 2 or before_nodes.shape[-1] != 3 or before_nodes.shape != after_nodes.shape
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError("paired material nodes and positive radius required")
    points = np.concatenate([
        _local(before_nodes, before_pose), _local(after_nodes, before_pose),
        _local(before_nodes, after_pose), _local(after_nodes, after_pose),
    ])
    phi, _, valid = sample_grid(grid, points, dtype=np.float64)
    gaps = phi.reshape(4, -1).T - radius
    g00, g01, g10, g11 = gaps.T
    return dict(gaps_m=gaps, valid=valid.reshape(4, -1).all(axis=0),
                rod_effect_m=((g01 - g00) + (g11 - g10)) / 2.,
                link_effect_m=((g10 - g00) + (g11 - g01)) / 2.,
                interaction_m=g11 - g10 - g01 + g00)


def split_link_motion(grid, palm_poses, link_poses, node_endpoints, *, radius):
    """Separate palm rigid motion from the measured palm-relative pad motion."""
    def carry_link(link_pose, source_palm, destination_palm):
        xyz, link_rotation = _validated_pose(link_pose)
        source_xyz, source_rotation = _validated_pose(source_palm)
        destination_xyz, destination_rotation = _validated_pose(destination_palm)
        rotation = destination_rotation * source_rotation.inv()
        return destination_xyz + rotation.apply(xyz - source_xyz), (rotation * link_rotation).as_quat()

    p0, p1 = palm_poses
    link0, link1 = link_poses
    palm_only = carry_link(link0, p0, p1)
    finger_only = carry_link(link1, p1, p0)
    early_palm = gap_motion(grid, link0, palm_only, *node_endpoints, radius=radius)
    late_palm = gap_motion(grid, finger_only, link1, *node_endpoints, radius=radius)
    early_finger = gap_motion(grid, link0, finger_only, *node_endpoints, radius=radius)
    late_finger = gap_motion(grid, palm_only, link1, *node_endpoints, radius=radius)
    valid = early_palm["valid"] & late_palm["valid"] & early_finger["valid"] & late_finger["valid"]
    return dict(valid=valid,
                palm_effect_m=(early_palm["link_effect_m"] + late_palm["link_effect_m"]) / 2.,
                finger_effect_m=(early_finger["link_effect_m"] + late_finger["link_effect_m"]) / 2.)


def index_loads(pairs, *, node_count):
    """An absent pair in a complete solver packet contributes zero force."""
    loads = np.zeros((SOLVER_SUBSTEPS, node_count))
    for pair in pairs:
        if pair["mesh_name"] != PAD:
            continue
        step, node = pair["substep"], pair["node_index"]
        normal = pair["normal_force_on_rod_n"]
        if (type(step) is not int or type(node) is not int or not 0 <= step < SOLVER_SUBSTEPS
                or not 0 <= node < node_count or not np.isfinite(normal) or normal < 0):
            raise ValueError("invalid index contact sample")
        loads[step, node] += normal
    return loads


def _contact_row(read, step):
    name = f"tripod_control_{step}.json"
    row = read(name)
    rod = row["rod"]
    pairs = rod["rod_mesh_contact_pairs"]
    if (row["physics_step"] != step or rod["physics_step"] != step
            or row["sdk_hand_geometry"]["physics_step"] != step or not rod["clock_valid"]
            or rod["rod_tick"] != SOLVER_SUBSTEPS * (step - 2)
            or rod["pair_capture_status"] != "validated_solver_samples"
            or not rod["contact_valid"] or row["guard_failure"] is not None
            or not np.isclose(pairs["substep_dt_s"], 1. / (480 * SOLVER_SUBSTEPS))):
        raise ValueError(f"invalid same-clock solver packet in {name}")
    return row


def _reaction_delta(before, after, body_id):
    if after["reaction_source_physics_step"] != before["physics_step"]:
        raise ValueError("issued reaction must come from preceding solver packet")
    source = before["rod"]["hand_wrenches_com_world"]["left"]
    issued = after["issued_composer_wrench"]["left"]
    if (not source["is_global"] or not issued["is_global"]
            or source["body_ids"].count(body_id) != 1 or issued["body_ids"].count(body_id) != 1):
        raise ValueError("World-frame index wrench route missing or ambiguous")
    indexes = [source["body_ids"].index(body_id), issued["body_ids"].index(body_id)]
    return {key: float(np.linalg.norm(source[key][0, indexes[0]] - issued[key][0, indexes[1]]))
            for key in ("forces", "torques")}


def _tracking(after):
    readback = after["action_readback"]
    issued = readback["request"]["action24"][0]
    pre, post = (readback[when]["hands"]["left"] for when in ("pre", "post"))
    if pre["joint_names"] != post["joint_names"]:
        raise ValueError("left SDK joint identity changed")
    joint_tracks = {}
    names = ("finger21", "finger22", *(name for name in post["joint_names"] if name.startswith("wrist_")))
    for name in names:
        index = post["joint_names"].index(name)
        commanded = index in post["active_indices"] or (post.get("mounted") and name.startswith("wrist_"))
        unit = post.get("units_by_name", {}).get(name, "rad")
        joint_tracks[name] = dict(
            unit=unit,
            q_before=float(pre["q"][0, index]), q_after=float(post["q"][0, index]),
            qd_after_per_s=float(post["qd"][0, index]),
            backend_target_before=float(pre["backend_targets"][0, index]) if commanded else None,
            backend_target=float(post["backend_targets"][0, index]) if commanded else None,
        )
    return dict(action24=issued.tolist(), joints=joint_tracks)


def review(run, *, receipt_sha256, cohort, cohort_sha256, sdf=None):
    outer = json.loads(checked_file(run / "outer.json", receipt_sha256))
    losses = json.loads(checked_file(cohort, cohort_sha256))
    if (outer["exit_code"] != 0 or outer["forced_exit"] or outer["exceeded_budget"]
            or outer["debugger_status_verified"] is not True
            or outer["debugger_inferior_exit_code"] != 0 or outer["debugger_inferior_signal"] is not None
            or receipt_sha256 != losses["receipt_sha256"]):
        raise ValueError("original natural-exit archive receipt required")
    inventory = outer["artifact_sha256"]

    def read(name):
        return decode_owned(json.loads(checked_file(run / "native" / name, inventory[name])))

    closed = read("app_close_returned.json")
    if closed["app_close_returned"] is not True or closed["close_mode"] != "normal":
        raise ValueError("receipt-bound normal app.close return is required")
    radius = read("tripod_reset_source_2.json")["evidence"]["rod_radius_m"]
    rows = []
    binding = grid = None
    seen = set()
    for sample in losses["samples"]:
        first, lost = sample["previous_loaded_sdk"], sample["first_loss_sdk"]
        if (lost in seen or lost != first + 1
                or sample["artifact_sha256"] != inventory[f"tripod_control_{lost}.json"]):
            raise ValueError("expected unique, adjacent, receipt-bound first loss")
        seen.add(lost)
        before, after = (_contact_row(read, step) for step in (first, lost))
        rod0, rod1 = before["rod"], after["rod"]
        identity = ("run_uuid", "episode_uid", "reset_id", "env_id", "rod_uid", "rest_lengths_m")
        if any(not np.array_equal(rod0[key], rod1[key]) for key in identity):
            raise ValueError("rod material or reset identity changed at loss")
        if binding is None:
            binding = rod0["mesh_bindings"][PAD]
            if binding["frame"] != "PhysX_link_origin" or binding["unit"] != "m":
                raise ValueError("link-local metric SDF required")
            raw = checked_file(sdf or Path(binding["sdf_path"]), binding["sdf_sha256"])
            with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                grid = {key: arrays[key] for key in ("values", "origin", "spacing")}
        if any(rod["mesh_bindings"][PAD] != binding for rod in (rod0, rod1)):
            raise ValueError("locked left index SDF binding changed")
        x0, x1 = (rod["node_positions"][0] for rod in (rod0, rod1))
        load0, load1 = (index_loads(rod["rod_mesh_contact_pairs"]["pairs"], node_count=len(x0))
                        for rod in (rod0, rod1))
        min0, min1 = (float(loads.sum(axis=1).min()) for loads in (load0, load1))
        if not min0 > .05 or not np.isclose(
                min1, sample["hands"]["left"]["minimum_required_pad_load_n"]["index"], atol=1e-12, rtol=0.):
            raise ValueError("indexed pad load disagrees with original threshold or loss record")
        nodes = np.flatnonzero(load0.sum(axis=0) > 0.)
        g0, g1 = (row["sdk_hand_geometry"]["poses"] for row in (before, after))
        motion = gap_motion(grid, g0[PAD], g1[PAD], x0[nodes], x1[nodes], radius=radius)
        split = split_link_motion(grid, (g0["left.base_link"], g1["left.base_link"]),
                                  (g0[PAD], g1[PAD]), (x0[nodes], x1[nodes]), radius=radius)
        effects = ("rod_effect_m", "link_effect_m", "interaction_m")

        def material_result(i, node):
            item = dict(node_id=int(node), samples_loaded_before=int(np.count_nonzero(load0[:, node])),
                        samples_loaded_after=int(np.count_nonzero(load1[:, node])),
                        mean_normal_before_n=float(load0[:, node].mean()),
                        in_all_four_lattices=bool(motion["valid"][i]))
            if motion["valid"][i]:
                item.update(gaps_m=motion["gaps_m"][i].tolist(),
                            **{key: float(motion[key][i]) for key in effects})
            item["palm_finger_split_valid"] = bool(split["valid"][i])
            if split["valid"][i]:
                item.update(palm_effect_m=float(split["palm_effect_m"][i]),
                            finger_effect_m=float(split["finger_effect_m"][i]))
            return item

        rows.append(dict(first_sdk=first, loss_sdk=lost,
                         first_sha256=inventory[f"tripod_control_{first}.json"],
                         loss_sha256=sample["artifact_sha256"], min_index_before_n=min0,
                         min_index_after_n=min1,
                         left_palm_delta_world_m=(g1["left.base_link"][0] - g0["left.base_link"][0]).tolist(),
                         left_index_delta_world_m=(g1[PAD][0] - g0[PAD][0]).tolist(),
                         tracking=_tracking(after),
                         reaction_delta=_reaction_delta(before, after, binding["body_index"]),
                         material_nodes=[material_result(i, node) for i, node in enumerate(nodes)]))
    if not rows:
        raise ValueError("archived first-loss cohort required")
    return dict(version="dh116-tripod-index-contact-loss-v1", receipt_sha256=receipt_sha256,
                cohort_sha256=cohort_sha256, sdf_sha256=binding["sdf_sha256"], radius_m=radius,
                gap_order=["before_pose_before_rod", "before_pose_after_rod",
                           "after_pose_before_rod", "after_pose_after_rod"],
                scope="archived_sdk_endpoint_geometry_and_measured_reactions",
                new_native_steps=0, grasp_qualified=False, generalization_qualified=False, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "cohort", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("receipt-sha256", "cohort-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--sdf", type=Path, help="relocated copy of the original locked SDF")
    args = parser.parse_args()
    audit = review(args.run, receipt_sha256=args.receipt_sha256, cohort=args.cohort,
                   cohort_sha256=args.cohort_sha256, sdf=args.sdf)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(audit, output, allow_nan=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
