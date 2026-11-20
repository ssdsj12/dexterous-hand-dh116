#!/usr/bin/env python3
"""Screen staged right-wrist clearance before releasing the Pinch thumb and index."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from scripts import prepare_dh116_pinch_prelift as prelift

RELEASE_PROGRESS = .55
LINEAR_APPROACH_LIMIT = .65
WRIST_PROGRESS_LIMIT = 1.02


def phase_envelope(measured, issued, goal, *, release_progress):
    actual, command, target = [prelift._finite(x, (12,)) for x in (measured, issued, goal)]
    lead, advance = command - actual, target - actual
    if (not 0. < release_progress <= LINEAR_APPROACH_LIMIT or np.any(advance[1:3] <= 0.)
            or np.any(lead[1:3] < 0.) or np.any(lead[1:3] > advance[1:3])):
        raise ValueError("ordered positive closing lead and wrist release threshold required")
    wrist = (slice(6, 9), slice(9, 12))
    if any(np.linalg.norm(advance[group]) < 1e-10 for group in wrist):
        raise ValueError("nondegenerate translation and intrinsic wrist paths required")
    fractions = [float(lead[g] @ advance[g] / (advance[g] @ advance[g])) for g in wrist]
    residuals = [lead[g] - progress * advance[g] for g, progress in zip(wrist, fractions)]
    return dict(
        angle_first=(np.r_[0., 0., np.minimum(0., fractions)],
                     np.r_[lead[1:3] / advance[1:3], max(0., fractions[0]), max(1., fractions[1])]),
        wrist_first=(np.r_[0., 0., min(0., fractions[0]), release_progress],
                     np.r_[lead[1:3] / advance[1:3], max(LINEAR_APPROACH_LIMIT, fractions[0]), WRIST_PROGRESS_LIMIT]),
        finger_release=(np.array([0., 0., release_progress, release_progress]),
                        np.array([1., 1., WRIST_PROGRESS_LIMIT, WRIST_PROGRESS_LIMIT])),
        wrist_residuals=residuals, original_wrist_progress=fractions,
    )


def right_control_windows(issued, goal):
    command = prelift._finite(issued, (12,))
    target = prelift._finite(goal, (12,))
    rotated = np.r_[command[:9], target[9:]]
    aligned = np.r_[command[:6], target[6:]]
    unchanged_left = np.zeros(12)

    def row(right):
        return prelift.global_channel_order(unchanged_left, right)

    return (prelift.bounded_transition(row(command), row(rotated))[:2],
            prelift.bounded_transition(row(rotated), row(aligned))[:2],
            prelift.bounded_transition(row(aligned), row(target))[:2])


def right_geometry(row, actual, issued, target, envelope, *, nodes,
                   phase_names=("angle_first", "wrist_first", "finger_release"), phase_seconds=75.,
                   side="right", require_pad_clearance=False):
    model, source = prelift._bound_model(prelift.base.ROOT, side)
    raw = row["closure_geometry"]["raw"]["hands"][side]
    frames, xyz = prelift._frames(model, raw["body_names"], raw["link_poses_xyzw"])
    hand = row["action_readback"]["post"]["hands"][side]
    names = hand["joint_names"]
    q = hand["q"][0, [names.index(n) for n in prelift.JOINTS]]
    anchor = Rotation.from_quat(np.roll(hand["native_anchor_pose_w"][0, 3:], -1)).as_matrix()
    reference = model._frames(anchor, xyz, actual[9:], q)
    alignment = {key: frame @ np.linalg.inv(reference[model.paths[key]]) for key, frame in frames.items()}
    radius = row["closure_geometry"]["rod_radius_m"]
    nominal_radius = prelift.CERTIFICATE_RADIUS_M
    if abs(radius - nominal_radius) > 1e-8:
        raise ValueError("sealed 5 mm rod radius required")
    rod = prelift.RodCapsuleSeparation(nodes, radius, node_motion_bound_m=np.full(len(nodes), prelift.NODE_RESERVE_M))
    delta = target - actual
    active = actual[:6]
    base_expanded = prelift.expand(active)
    q_groups = []
    for index in (1, 2):
        command = active.copy()
        command[index] = target[index]
        q_groups.append(prelift.expand(command) - base_expanded)
    others = []
    for known in (issued, target):
        command = known[:6].copy()
        command[1:3] = active[1:3]
        others.append(prelift.expand(command) - base_expanded)
    other_joint_motion = np.maximum(abs(others[0]), abs(others[1]))
    linear_residual, angular_residual = envelope["wrist_residuals"]
    bounds, reserves = {}, {}
    for key, angles in model.joint_radii.items():
        radius_from_wrist = model.wrist_radii[key]
        bounds[key] = np.array([*(angles @ abs(g) for g in q_groups),
                                np.linalg.norm(delta[6:9]), radius_from_wrist * abs(delta[9:]).sum()])
        reserves[key] = float(np.linalg.norm(linear_residual) + radius_from_wrist * abs(angular_residual).sum()
                              + angles @ other_joint_motion)

    def moved(x, key):
        world = model._frames(anchor, xyz + x[2] * delta[6:9], actual[9:] + x[3] * delta[9:],
                              q + q_groups[0] * x[0] + q_groups[1] * x[1])
        return alignment[key] @ world[model.paths[key]]

    def one_shape(key, x):
        fr = moved(x, key)
        verts = model.vertices[key] @ fr[:3, :3].T + fr[:3, 3]
        gap = {"table": float(verts[:, 2].min() - model.table_top - prelift.CLEARANCE_M - reserves[key])}
        if require_pad_clearance or key.split(":", 1)[1] not in prelift.REQUIRED:
            lower = rod.gap_lower_m(verts)
            if lower < .00015:
                precise = prelift.endpoint_gap(verts, nodes, nominal_radius)["lower_m"]
                lower = max(lower, precise + nominal_radius - radius - prelift.NODE_RESERVE_M)
            gap["rod" if require_pad_clearance else "non_target_rod"] = lower - reserves[key]
        return gap

    def certify_phase(lo, hi, *, seconds):
        start = time.monotonic()
        verified = {}
        for key in frames:
            left = seconds - (time.monotonic() - start)
            if left <= 0:
                return dict(proved=False, verified_shapes=verified, unresolved_shape=key, reason="phase time budget")
            labels = one_shape(key, (lo + hi) / 2)
            coefficient = {name: bounds[key] for name in labels}
            query = lambda x: one_shape(key, x)
            proof = prelift.certify_box(query, coefficient, lo, hi, max_samples=768, max_seconds=left)
            if not proof["proved"]:
                return dict(proved=False, verified_shapes=verified, unresolved_shape=key, counterexample=proof,
                            reason="conditional margin or proof budget")
            verification = prelift.verify_box_certificate(proof, query, coefficient, lo, hi)
            verified[key] = dict(summary=verification, certificate=proof, residual_reserve_m=reserves[key])
        return dict(proved=True, verified_shapes=verified)

    # The same measured rod shape underlies each conditional phase proof.
    phases = {key: certify_phase(*envelope[key], seconds=phase_seconds) for key in phase_names}
    if "finger_release" not in phases:
        return phases, None
    grids = {"left": {}, "right": {}}
    prelift._bind_pad_grids(row, side, source, grids, {})
    staged_end = np.array([1., 1., LINEAR_APPROACH_LIMIT, 1.])
    gaps = [prelift._contact(grids[side][key], moved(staged_end, key), nodes, radius)[0] for key in prelift.PADS]
    return phases, gaps


def screen(folder, sdk, *, rod_sdk=None):
    rod_sdk = sdk if rod_sdk is None else rod_sdk
    if not sdk <= rod_sdk <= 74:
        raise ValueError("rod sensitivity snapshot must follow the same sealed phase boundary")
    endpoint, _ = prelift.archived_endpoint(folder)
    nominal = prelift.retained_target(*prelift.joint_and_wrist(endpoint, "right"), "right")
    row = rod_row = None
    with (folder / "native" / "physical.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("operation") == "physical":
                observation = prelift.base.decode_owned(event["payload"])["exchange"]
                if observation["physics_step"] == sdk:
                    row = observation
                if observation["physics_step"] == rod_sdk:
                    rod_row = observation
                    break
    if row is None or row["guard_failure"] is not None:
        raise ValueError("a valid corresponding same-instance native boundary is required")
    identity = prelift._same_clock(row)
    rod_identity = prelift._same_clock(rod_row)
    actual, command = prelift.joint_and_wrist(row, "right")
    envelope = phase_envelope(actual, command, nominal, release_progress=RELEASE_PROGRESS)
    staged_target = nominal.copy()
    staged_target[6:9] = actual[6:9] + LINEAR_APPROACH_LIMIT * (nominal[6:9] - actual[6:9])
    rotate, align, close = right_control_windows(command, staged_target)
    phases, pad_gaps = right_geometry(row, actual, command, nominal, envelope,
                                      nodes=rod_row["rod"]["node_positions"][0])
    source = {key: [a.tolist(), b.tolist()] for key, (a, b) in envelope.items()
              if key in ("angle_first", "wrist_first", "finger_release")}
    return dict(schema="dh116-pinch-right-wrist-first-v1", identity=identity, sdk_step=sdk,
                source_ledger_sha256=prelift.SOURCE_LEDGER_SHA, phase_bounds=source,
                source_right_target=command.tolist(), condition_right_final_target=staged_target.tolist(),
                full_linear_progress_reference=nominal.tolist(),
                original_wrist_progress=envelope["original_wrist_progress"],
                angle_only_requests=rotate.tolist(), translation_only_requests=align.tolist(),
                finger_only_requests=close.tolist(), rod_snapshot_identity=rod_identity,
                rod_snapshot_sdk_step=rod_sdk,
                conditioned_right_pose_proof=phases, nominal_right_pad_gaps_m=pad_gaps,
                node_reserve_m=prelift.NODE_RESERVE_M, native_ready=False, teacher_admission=False,
                scope="right-side conditional geometry with measurable phase gate; no lift or native action")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sdk", type=int, choices=(50, 58), default=58)
    p.add_argument("--rod-sdk", type=int)
    args = p.parse_args(argv)
    result = screen(args.archive.resolve(strict=True), args.sdk, rod_sdk=args.rod_sdk)
    prelift.base.lifecycle.native.write_new(args.output, result)
    phases = result["conditioned_right_pose_proof"]
    print(json.dumps(dict(
        phases={key: {k: data.get(k) for k in ("proved", "unresolved_shape", "reason")} for key, data in phases.items()},
        nominal_right_pad_gaps_m=result["nominal_right_pad_gaps_m"],
        evidence_sha256=prelift.base.lifecycle.native.sha(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
