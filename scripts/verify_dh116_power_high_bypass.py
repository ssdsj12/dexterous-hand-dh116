"""Power wrist transfer in the original 24D supported-rod scene."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts import screen_dh116_tripod_measured_precontact as measured

ROOT = measured.ROOT
CONTRACT = ROOT / "docs/contracts/dh116_teacher_power_retreat_then_open_native_v1.json"
REQUEST = ROOT / "configs/dh116_tasks/flexible_power_high_bypass_v1.json"
SDK = 880
WINDOWS = (66, 272)


def read_source(receipt):
    info = json.loads(CONTRACT.read_bytes())["actual"]
    outer = measured.sources.checked_json(receipt, info["outer_sha256"])
    if (outer["forced_exit"] or outer["exceeded_budget"] or outer["exit_code"] != 0
            or outer["debugger_inferior_exit_code"] != 0):
        raise ValueError("the Power route requires the naturally closed opening capture")
    raw = receipt.parent/"native"
    bound = [CONTRACT, receipt]

    def read(name):
        path = raw/name
        bound.append(path)
        return measured.sources.checked_json(path, outer["artifact_sha256"][name])

    close = read("app_close_returned.json")
    layout = read("power_support_scene.json")
    last = read(f"power_control_{SDK}.json")
    if (close["app_close_returned"] is not True or last["physics_step"] != SDK
            or last["rod"]["physics_step"] != SDK or last["rod"]["rod_tick"] != (SDK - 2)*12
            or last["sdk_hand_geometry"]["physics_step"] != SDK or last["guard_failure"] is not None
            or {p["mesh_name"] for p in last["rod"]["rod_mesh_contact_pairs"]["pairs"]} != {"support_0", "support_1"}):
        raise ValueError("Power repositioning requires the sealed hands-clear SDK 880 state")
    return last, layout, bound


def compile_route(row, request):
    if (request["version"] != "dh116-power-high-bypass-v1" or request["source_boundary_sdk"] != SDK
            or tuple(request["phase_window_counts"]) != WINDOWS
            or request["runtime_interpolation"] != "world_xyz_and_reset_relative_intrinsic_XYZ"
            or set(request["hands"]) != set(measured.SIDES)):
        raise ValueError("expected the frozen high bypass route")
    if np.max(np.abs(row["wrist_projection"]["projected_angles"])) > 1e-9:
        raise ValueError("the high route must start from the unchanged reset rotation basis")
    return compile_wrist_waypoints(row, request["hands"], WINDOWS)


def compile_wrist_waypoints(row, hands, stages_count):
    projection = row["wrist_projection"]
    bases = projection["projected_root_poses"][0]
    seed_xyz = bases[:, :3] - projection["palm_offsets_m"][0]
    if projection["safe_z_clipped"].any():
        raise ValueError("the high route must start from the unchanged reset rotation basis")
    windows = []
    targets = {}
    for lane, side in enumerate(measured.SIDES):
        poses = np.asarray(hands[side]["waypoints"], dtype=float)
        if (poses.shape != (len(stages_count)+1, 7) or not np.isfinite(poses).all()
                or not np.allclose(np.linalg.norm(poses[:,3:], axis=1), 1., atol=5e-7, rtol=0)
                or not np.allclose(poses[0], bases[lane], atol=1e-9, rtol=0)):
            raise ValueError("each Power hand must begin at its actual projected wrist pose")
        offsets = poses[:, :3] - seed_xyz[lane]
        seed = (Rotation.from_quat(poses[0, [4,5,6,3]]) *
                Rotation.from_euler("XYZ", projection["projected_angles"][0,lane]).inv())
        angles = (seed.inv()*Rotation.from_quat(poses[:, [4,5,6,3]])).as_euler("XYZ")
        if ((offsets < WRIST["palm_lower_m"]).any() or (offsets > WRIST["palm_upper_m"]).any()
                or np.any(np.abs(angles) > WRIST["rotation_limit_rad"])):
            raise ValueError("Power route target exceeds original wrist range")
        targets[side] = poses[1:]
        windows.append((np.diff(poses[:,:3], axis=0), np.diff(angles, axis=0)))
    stages = []
    for stage, count in enumerate(stages_count):
        command = np.zeros(24)
        for lane, (translation, rotation) in enumerate(windows):
            for start, delta, rate in ((12, translation, ACTION_CONTRACT["translation_rate_m_s"]),
                                      (18, rotation, ACTION_CONTRACT["rotation_rate_rad_s"])):
                value = delta[stage] / (count * rate / 60.)
                if np.linalg.norm(value) > 1.:
                    raise ValueError("Power high route exceeds original rate")
                command[start+3*lane:start+3*(lane+1)] = value
        stages.append(np.repeat(command[None], count, axis=0))
    return np.vstack(stages), targets
