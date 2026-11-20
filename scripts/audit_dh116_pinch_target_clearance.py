#!/usr/bin/env python3
"""Replay right finger13 target clearances from a Pinch Teacher journal."""
import argparse
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from scripts.audit_dh116_pinch_initial_pose import load_geometry, pose_matrix, transform
from scripts.audit_dh116_pinch_recentered_tracking import _checked_bytes
from scripts.audit_dh116_teacher_window_wall import _windows
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, relative_pose


LINK = "finger13_link"
SOURCE = "source:" + LINK


def target_clearances(vertices, table_top, link_frame, base, q, target_base, target_q, fk_link):
    """Change only selected targets, retaining the same measured link anchor."""
    measured_fk = fk_link(base, q)
    result = {"measured_m": float(transform(vertices, link_frame)[:, 2].min() - table_top)}
    translated_base = np.r_[target_base[:3], base[3:]]
    for name, pose, joints in (
        ("translation_only_m", translated_base, q),
        ("wrist_only_m", target_base, q),
        ("fingers_only_m", base, target_q),
        ("combined_m", target_base, target_q),
    ):
        frame = relative_pose(link_frame, measured_fk, fk_link(pose, joints))
        result[name] = float(transform(vertices, frame)[:, 2].min() - table_top)
    return result


def replay(project, journal, journal_sha256):
    geometry = load_geometry(project, side="right")
    layout = geometry["reference"]["reference_layout"]
    table_top = layout["table_center"][2] + layout["table_size"][2] / 2
    vertices = geometry["source_colliders"][LINK]["vertices"]

    def fk_link(pose, q):
        return fk_poses(geometry, pose, q)[LINK]

    events = [json.loads(l) for l in _checked_bytes(journal, journal_sha256).splitlines()]
    rows = []
    for window in _windows(events):
        for packet in window["physical"]:
            row = decode_owned(packet)
            source = row["post_step_clearance"]
            if source["floor_m"] != CLEARANCE_M or source["physics_step"] != row["physics_step"]:
                raise ValueError("same-step clearance and original table floor required")
            raw = row["closure_geometry"]["raw"]["hands"]["right"]
            backend = row["action_readback"]["post"]["hands"]["right"]
            idx = backend["active_indices"]
            if tuple(backend["joint_names"][i] for i in idx) != tuple(ACTIVE):
                raise ValueError("canonical right active-joint readback required")
            observed = raw["link_poses_xyzw"][0]
            names = raw["body_names"]
            pose = observed[names.index("base_link")][[0, 1, 2, 6, 3, 4, 5]]
            link_pose = observed[names.index(LINK)][[0, 1, 2, 6, 3, 4, 5]]
            target = row["wrist_projection"]["projected_root_poses"][0, 1]
            q, target_q = backend["q"][0, idx], backend["backend_targets"][0, idx]
            values = target_clearances(vertices, table_top, pose_matrix(link_pose), pose, q,
                                       target, target_q, fk_link)
            measured = row["closure_geometry"]["sides"]["right"]["table_clearance_m"][SOURCE]
            if not np.isclose(values["measured_m"], measured, atol=1e-7, rtol=0.0):
                raise ValueError("same-step measured clearance replay disagrees with the journal")
            rows.append(dict(values, physics_step=row["physics_step"], guard_reason=row["guard_failure"],
                             recorded_m=measured))
    return dict(schema="dh116-pinch-target-clearance-replay-v1",
                scope="measured_anchor_static_targets", journal_sha256=journal_sha256,
                protected_floor_m=CLEARANCE_M, rows=rows, new_physics_steps=0,
                sweeping_path_verified=False, native_qualification=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = replay(args.project, args.journal, args.journal_sha256)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
