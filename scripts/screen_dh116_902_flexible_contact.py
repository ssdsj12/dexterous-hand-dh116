"""Offline contact candidates anchored to measured link poses; never grants grasp success."""
import numpy as np


def measured_fk_pose(measured, reference_fk, candidate_fk):
    """Carry the measured local link offset with the new FK frame, not the world frame."""
    matrices = [np.asarray(m, dtype=float) for m in (measured, reference_fk, candidate_fk)]
    for m in matrices:
        if (m.shape != (4, 4) or not np.isfinite(m).all()
                or not np.allclose(m[3], [0., 0., 0., 1.], atol=1e-9, rtol=0.)
                or not np.allclose(m[:3,:3].T @ m[:3,:3], np.eye(3), atol=1e-6, rtol=0.)
                or abs(np.linalg.det(m[:3,:3]) - 1.) > 1e-6):
            raise ValueError("finite rigid link transforms required")
    observed, old, new = matrices
    return new @ np.linalg.solve(old, observed)


def contact_route_passed(rows, side, *, pad_count, max_penetration_m):
    """Accept selected-pad approach to contact without exempting other geometry."""
    from scripts.screen_dh116_902_flexible_precontact import material_side_matches
    if side not in ("left", "right") or pad_count not in (2, 3):
        return False
    if (not np.isfinite(max_penetration_m) or not 0. < max_penetration_m <= .00015
            or len(rows) != 17 or not material_side_matches(rows, side)):
        return False
    for row in rows:
        gaps = np.asarray(row["pad_gaps_m"], dtype=float)
        normals = np.asarray(row["opposition"], dtype=float)
        limits = np.array([row["noncontact_rod_clearance_m"], row["support_clearance_m"],
                           row["table_clearance_m"], row.get("contact_rod_clearance_m", np.nan)], dtype=float)
        if (row["shape_count"] != 22 or not np.isfinite(limits).all()
                or limits[0] <= .00025 or limits[1] <= 0. or limits[2] < .0021
                or limits[3] < -max_penetration_m
                or gaps.shape != (pad_count,) or not np.isfinite(gaps).all()
                or np.any(gaps < -max_penetration_m)
                or normals.shape != (pad_count-1,) or not np.isfinite(normals).all()):
            return False
    return bool(max(rows[-1]["opposition"]) < -.2 and max(rows[-1]["pad_gaps_m"]) < 0.)


def fixed_wrist_commands(row, target_q, *, windows):
    """Integrate from actual issued targets; preserve all 12 wrist action slots."""
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, expand
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    if type(windows) is not int or not 1 <= windows <= 60:
        raise ValueError("bounded control window count required")
    delta = []
    for side in ("left", "right"):
        hand = row["action_readback"]["post"]["hands"][side]
        indices = hand["active_indices"]
        if [hand["joint_names"][i] for i in indices] != list(ACTIVE):
            raise ValueError("original active joint order required")
        target = np.asarray(target_q[side], dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("six finite joint targets required")
        expand(target)
        issued = np.asarray(hand["backend_targets"], dtype=float)[0, indices]
        delta.extend(target - issued)
    command = np.r_[np.asarray(delta) / (windows / 60. * ACTION_CONTRACT["joint_rate_rad_s"]), np.zeros(12)]
    if not np.isfinite(command).all() or np.max(np.abs(command)) > 1.:
        raise ValueError("fixed-wrist closure exceeds original joint action rate")
    return np.repeat(command[None], windows, axis=0)


def held_route_passed(rows):
    return bool(len(rows) == 17 and all(row["shape_count"] == 22
        and np.isfinite([row["rod_clearance_m"], row["support_clearance_m"], row["table_clearance_m"]]).all()
        and row["rod_clearance_m"] > .00025 and row["support_clearance_m"] > 0.
        and row["table_clearance_m"] >= .0021 for row in rows))


def independent_entry(task, contact):
    """Admit only an explicit independently sealed boundary; legacy paths remain strict."""
    from scripts.diagnose_dh116_teacher_initialization import digest
    if contact is None or contact.get("entry_contract") != "dh116-902-distal-contact-v1":
        return False
    if contact.get("envelope_refinement") is not None:
        raise ValueError("legacy 902 v1 cannot enable envelope refinement")
    sdk = contact.get("boundary_sdk")
    row = contact.get("archive", {})
    if (task not in ("pinch", "tripod") or contact.get("grasp") != task
            or type(sdk) is not int or not 248 <= sdk <= 432 or (sdk-240) % 8
            or row.get("physics_step") != sdk or row.get("guard_failure", True) is not None
            or row.get("rod", {}).get("physics_step") != sdk
            or row.get("rod", {}).get("rod_tick") != (sdk-2)*12
            or row.get("sdk_hand_geometry", {}).get("physics_step") != sdk
            or digest(row) != contact.get("archive_digest")):
        raise ValueError("owned same-clock 902 contact boundary required")
    side=contact.get("contact_side")
    stations=contact.get("stations",{})
    cooked=contact.get("continuous_cooked_hulls")
    pads=("finger14_link","finger23_link") + (("finger33_link",) if task=="tripod" else ())
    expected={str(side)+"."+pad for pad in pads}
    if (side not in ("left","right") or set(stations)!=expected or not isinstance(cooked,dict)
            or set(cooked)!=expected or not 0. < contact.get("max_pad_penetration_m",0.) <= .00015
            or any(not isinstance(ids,list) or not ids or any(type(i) is not int
                or not (21<=i<=40 if side=="left" else 0<=i<20) for i in ids)
                for ids in stations.values())):
        raise ValueError("source-bound continuous contact geometry and original-side stations required")
    for vertices in cooked.values():
        v=np.asarray(vertices,dtype=float)
        if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all():
            raise ValueError("finite continuous contact hull vertices required")
    return True


def boundary(receipt, grasp):
    import json
    from pathlib import Path
    from scripts import screen_dh116_tripod_measured_precontact as measured
    root = Path(__file__).resolve().parents[1]
    if grasp == "tripod":
        contract = root / "docs/contracts/dh116_teacher_902_tripod_safe_boundary_v1.json"
        data = json.loads(contract.read_bytes())
        if data.get("native_boundary_verified") is not True:
            raise ValueError("naturally verified Tripod boundary required")
        expected = data["actual"]; sdk = expected["sdk_end"]
    elif grasp == "pinch":
        contract = root / "docs/contracts/dh116_teacher_902_flexible_distal_precontact_v1.json"
        expected = json.loads(contract.read_bytes())["native_actual"]["pinch"]
        sdk = expected["last_complete_sdk"]
    else:
        raise ValueError("owned distal grasp required")
    rows, supports, files = measured.archive(receipt, contract=contract, sdk=sdk, task=grasp, expected=expected)
    return rows, supports, files, sdk


def commanded_goals(row):
    goals = {}
    for index, side in enumerate(("left", "right")):
        hand = row["action_readback"]["post"]["hands"][side]
        pose = np.asarray(row["wrist_projection"]["projected_root_poses"][0,index], dtype=float).copy()
        pose[3:] /= np.linalg.norm(pose[3:])
        q = np.asarray(hand["backend_targets"])[0,hand["active_indices"]]
        goals[side] = dict(pose_wxyz=pose.tolist(),active_q_rad=q.tolist())
    return goals


def stage_actions(config, row, request):
    from scripts.verify_dh116_tripod_supported_reset import waypoint_controls
    side=request["contact_side"];other="right" if side=="left" else "left"
    goals=commanded_goals(row);goals[side]=request["target"]
    actions=waypoint_controls(config,row,goals,windows=request["control_windows"])
    slots=(slice(0,6),slice(12,15),slice(18,21)) if other=="left" else (slice(6,12),slice(15,18),slice(21,24))
    if max(np.max(abs(actions[:,sl])) for sl in slots)>1e-10:
        raise ValueError("held hand unexpectedly received an approach command")
    for sl in slots:actions[:,sl]=0.
    return np.concatenate([actions,np.zeros((request["hold_windows"],24))])



def checked_request(request, *, grasp, sdk, outer_sha):
    if request.get("envelope_refinement") is not None:
        raise ValueError("legacy 902 v1 cannot enable envelope refinement")
    windows, hold = request.get("control_windows"), request.get("hold_windows")
    if (request.get("version") != "dh116-902-distal-contact-v1"
            or request.get("grasp") != grasp or grasp not in ("pinch", "tripod")
            or request.get("contact_side") not in ("left", "right")
            or type(windows) is not int or not 1 <= windows <= 24
            or type(hold) is not int or not 0 <= hold <= 12
            or not 0. < request.get("max_penetration_m", 0.) <= .00015
            or request.get("source_sdk") != sdk
            or request.get("source_outer_sha256") != outer_sha):
        raise ValueError("bounded side-specific source-bound contact request required")


def checked_stations(samples, side, grasp, stations=None):
    from scripts.screen_dh116_902_flexible_precontact import material_side_matches
    pads = ("finger14_link", "finger23_link") + (("finger33_link",) if grasp == "tripod" else ())
    if (side not in ("left", "right") or grasp not in ("pinch", "tripod")
            or len(samples) != 17 or not material_side_matches(samples, side)
            or any(len(row["node_ids"]) != len(pads) or any(type(i) is not int or not 0<=i<=40
                 for i in row["node_ids"]) for row in samples)):
        raise ValueError("complete original-side contact stations required")
    expected = {side+"."+pad: sorted({row["node_ids"][i] for row in samples})
                for i, pad in enumerate(pads)}
    if stations is not None and stations != expected:
        raise ValueError("contact stations differ from screened material nodes")
    return expected


def contact_dependency_files():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return tuple(root / name for name in (
        "scripts/screen_dh116_tripod_measured_precontact.py",
        "scripts/verify_dh116_supported_distal_transfer.py",
        "scripts/audit_dh116_pinch_initial_pose.py",
        "scripts/audit_dh116_certified_segment_distance.py",
        "scripts/prepare_dh116_pinch_closure_margin.py",
        "scripts/screen_dh116_tripod_supported_reset.py",
        "scripts/certify_dh116_tripod_distal_route.py",
        "src/go2_pvcnn/control/dh116/tabletop_power_contact_evidence.py"))


def prepare(receipt, request_path, *, expected_grasp=None):
    import json
    from pathlib import Path
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from scripts import verify_dh116_902_flexible_reset as reset
    from scripts import screen_dh116_tripod_measured_precontact as measured
    from scripts.screen_dh116_tripod_supported_reset import archived_tube
    from scripts.verify_dh116_tripod_supported_reset import waypoint_controls
    request = json.loads(request_path.read_bytes())
    grasp, side = request.get("grasp"), request.get("contact_side")
    windows, hold = request.get("control_windows"), request.get("hold_windows")
    if (request.get("version") != "dh116-902-distal-contact-v1" or side not in ("left", "right")
            or (expected_grasp is not None and grasp != expected_grasp)
            or type(windows) is not int or not 1 <= windows <= 24
            or type(hold) is not int or not 0 <= hold <= 12
            or not 0. < request.get("max_penetration_m", 0.) <= .00015):
        raise ValueError("bounded independent hand contact request required")
    reference, config, configuration_files = reset.load_candidate(grasp)  # Includes both unchanged stereo SHA checks.
    raw, supports, files, sdk = boundary(receipt, grasp)
    checked_request(request, grasp=grasp, sdk=sdk, outer_sha=reset.sha(receipt))
    rows = [decode_owned(row) for row in raw]; row = rows[-1]
    nodes = np.stack([v["rod"]["node_positions"][0] for v in rows])
    tube = archived_tube(nodes)
    boxes = [(np.asarray(b["center_m"])-np.asarray(b["size_m"])/2,
              np.asarray(b["center_m"])+np.asarray(b["size_m"])/2) for b in supports]
    grids = scene.load_sdf_arrays(scene.prepare_scene(reset.ROOT))
    pads = ("finger14_link","finger23_link") if grasp == "pinch" else ("finger14_link","finger23_link","finger33_link")
    target = request["target"]
    goals = commanded_goals(row)
    goals[side] = target
    table = reference["table"]
    table_top = table["center_m"][2] + table["size_m"][2] / 2
    measurements = {hand: measured.measure_side(hand, row, nodes[-1], tube, boxes, grids, goals[hand], table_top,
        pads=pads, max_delta_m=.004, max_delta_rad=.06, contact_pads=pads if hand == side else ())
        for hand in ("left","right")}
    passed = contact_route_passed(measurements[side],side,pad_count=len(pads),
                                 max_penetration_m=request["max_penetration_m"])
    other = "right" if side == "left" else "left"
    passed = bool(passed and held_route_passed(measurements[other]))
    actions = stage_actions(config,row,request)
    stations = checked_stations(measurements[side], side, grasp)
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    model = load_geometry(reset.ROOT, side=side)
    cooked = {side+"."+pad: model["cooked_pads"][pad]["vertices"].tolist() for pad in pads}
    contact = dict(entry_contract=request["version"],grasp=grasp,contact_side=side,boundary_sdk=sdk,
                   archive=raw[-1],archive_digest=measured.native.digest(raw[-1]),stations=stations,
                   max_pad_penetration_m=request["max_penetration_m"],continuous_cooked_hulls=cooked)
    independent_entry(grasp,contact)
    proof = dict(version=request["version"],grasp=grasp,contact_side=side,before_sdk=sdk,
                 request_sha256=reset.sha(request_path),outer_sha256=reset.sha(receipt),
                 samples=measurements,all_passed=passed,control_windows=windows,hold_windows=hold,
                 action_sha256=measured.native.digest(actions.tolist()),stations=stations,continuous_cooked_hulls=cooked,
                 native_contact_verified=False,teacher_admission=False)
    return proof,actions,contact,(*files,*configuration_files,request_path,Path(__file__),*contact_dependency_files())


def load_stage(receipt, request_path, proof_path, grasp, *, before_sdk):
    """Recheck seals and recompile original 24D commands without importing USD."""
    import json
    from pathlib import Path
    from scripts import verify_dh116_902_flexible_reset as reset
    from scripts.diagnose_dh116_teacher_initialization import digest
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    request=json.loads(request_path.read_bytes());proof=json.loads(proof_path.read_bytes())
    raw,_,files,sdk=boundary(receipt,grasp)
    checked_request(request,grasp=grasp,sdk=sdk,outer_sha=reset.sha(receipt))
    if (request["grasp"]!=grasp or proof.get("grasp")!=grasp or sdk!=before_sdk
            or proof.get("version")!=request["version"]
            or proof.get("contact_side")!=request["contact_side"]
            or proof.get("control_windows")!=request["control_windows"]
            or proof.get("hold_windows")!=request["hold_windows"]
            or proof.get("before_sdk")!=sdk or proof.get("request_sha256")!=reset.sha(request_path)
            or proof.get("outer_sha256")!=reset.sha(receipt) or proof.get("all_passed") is not True):
        raise ValueError("exact native boundary and contact proof required")
    side=request["contact_side"];other="right" if side=="left" else "left"
    count=2 if grasp=="pinch" else 3
    if (not contact_route_passed(proof["samples"][side],side,pad_count=count,
                max_penetration_m=request["max_penetration_m"])
            or not held_route_passed(proof["samples"][other])):
        raise ValueError("contact proof does not preserve geometric guards")
    _,config,config_files=reset.load_candidate(grasp)
    actions=stage_actions(config,decode_owned(raw[-1]),request)
    if digest(actions.tolist())!=proof["action_sha256"]:
        raise ValueError("contact actions differ from screened commands")
    checked_stations(proof["samples"][side],side,grasp,proof["stations"])
    cooked=proof.get("continuous_cooked_hulls",{})
    if set(cooked)!=set(proof["stations"]):
        raise ValueError("original cooked target hulls required")
    for vertices in cooked.values():
        v=np.asarray(vertices,dtype=float)
        if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all():
            raise ValueError("finite original cooked hull vertices required")
    contact=dict(entry_contract=request["version"],grasp=grasp,contact_side=side,boundary_sdk=sdk,
        archive=raw[-1],archive_digest=digest(raw[-1]),stations=proof["stations"],
        max_pad_penetration_m=request["max_penetration_m"],continuous_cooked_hulls=cooked)
    independent_entry(grasp,contact)
    return actions,contact,(*files,*config_files,request_path,proof_path,Path(__file__),*contact_dependency_files())


def main(argv=None):
    import argparse
    from pathlib import Path
    from scripts.diagnose_dh116_teacher_initialization import write_new
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    proof,*_=prepare(args.receipt,args.request)
    write_new(args.output,proof)
    return 0 if proof["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
