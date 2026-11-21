"""Read six locked Tripod SDFs from owned post-Reset SDK buffers.

This read-only distance source does not qualify the Reset transaction, native
mesh state or contact. Historical eleven-slot receipts stay eleven-slot data.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

from . import tabletop_dual_hand_reset as reset_owner
from .linear_coupling import JOINTS
from .tabletop_joint_target_readback import WRIST_LINEAR, WRIST_ANGULAR
from .tabletop_measured_closure_geometry import MeasuredClosureGeometry
from .tabletop_native_teacher_policy_tripod_distance import PolicyOwnedTripodApproach
from .tabletop_deformable_r0_replay import IDENTITY, _digest
from .tabletop_power_shared_rod import (
    MESH_NAMES, POWER_MESH_NAMES, SUPPORTED_POWER_MESH_NAMES,
    PROXIMAL_POWER_MESH_NAMES, SIDES, contact_links, _bindings,
)

VERSION = "dh116-measured-tripod-reset-distance-v1"


class MeasuredTripodResetApproach:
    """Share the three-pad metric, never the policy or native qualification."""

    _metric = PolicyOwnedTripodApproach._metric

    def __init__(self, project, *, scope="owner_source_unqualified"):
        if scope not in ("cpu_fixture", "owner_source_unqualified"):
            raise ValueError("explicit unqualified Tripod Reset source scope required")
        self.project = Path(project).resolve()
        self.geometry = MeasuredClosureGeometry(self.project)
        self.scope = scope

    def capture_reset(self, reset, parameter_packet, *, grasp="tripod"):
        if grasp != "tripod":
            raise ValueError("independent Tripod Reset distance cannot be relabelled")
        if (reset.get("version") != reset_owner.VERSION or reset.get("status") != "complete"
                or reset.get("new_sdk_steps") != 0 or reset.get("new_solver_steps") != 0):
            raise ValueError("completed zero-step actual Tripod Reset receipt required")
        rod, identity = reset["rod_reset"], reset["after_clock"]
        actual = reset["after_actual_clocks"]
        if (identity != actual["rod"] or identity.get("rod_tick") != 0
                or identity.get("clock_valid") is not True or rod["contact_valid"] is not False
                or set(reset["hands"]) != set(SIDES) or identity.get("reset_id", 0) < 1):
            raise ValueError("actual fresh Reset clock/rod/hands required")
        if (type(identity["physics_step"]) is not int
                or actual["sdk_physics_step"] != identity["physics_step"]
                or actual["sdk_dt_s"] != 1/480
                or not math.isclose(actual["sdk_timestamp_s"], identity["timestamp_s"], abs_tol=1e-9, rel_tol=0)
                or not math.isclose(identity["timestamp_s"], identity["physics_step"]/480, abs_tol=1e-9, rel_tol=0)):
            raise ValueError("actual Reset SDK clock differs")
        reset_owner._no_step(reset["before_clock"], actual)
        expected_scope = "cpu_fixture" if self.scope == "cpu_fixture" else "native_getter"
        if parameter_packet.get("source_scope") != expected_scope:
            raise ValueError("parameter and Tripod Reset source scopes differ")
        if parameter_packet["reward_radius_sample"]["source"]["identity"] != identity:
            raise ValueError("Reset radius provenance identity differs")
        from .tabletop_mesh_observation_source import PARAMETER_VERSION
        from .tabletop_native_teacher_profile import policy_source_basis
        if (parameter_packet.get("version") != PARAMETER_VERSION
                or parameter_packet.get("sources") != policy_source_basis()):
            raise ValueError("current Tripod Reset parameter source/version differs")
        from scripts.replay_dh116_native_parameter_r0 import check_radius_packet
        radius = check_radius_packet(parameter_packet, rod)
        if radius["source"]["identity"] != identity:
            raise ValueError("actual Reset and native parameter identities differ")
        seed = reset_owner._seed(reset["seed"])
        config_raw = (self.project/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_bytes()
        config_sha = hashlib.sha256(config_raw).hexdigest()
        if any(self.geometry.sources[side]["source_sha256"]["old_candidate"] != config_sha for side in SIDES):
            raise ValueError("original candidate config SHA differs")
        reset_owner._verify_after(seed, reset["before_hands"], reset["hands"],
                                  json.loads(config_raw)["hand"]["active_effort_limit_nm"])
        bindings = _bindings(rod["mesh_bindings"])
        links = contact_links(bindings)
        names = tuple(rod["mesh_names"])
        layouts = {
            10: MESH_NAMES, 12: (POWER_MESH_NAMES, SUPPORTED_POWER_MESH_NAMES),
            14: PROXIMAL_POWER_MESH_NAMES,
        }
        expected = layouts[len(bindings)]
        if len(bindings) == 12:
            correct_layout = names in expected
        else:
            correct_layout = names == expected
        if not correct_layout:
            raise ValueError("actual Reset mesh layout and binding identities differ")
        if set(reset["pad_poses_xyzw"]) != set(bindings):
            raise ValueError("complete actual Reset contact pose copies required")
        hands = {}
        for side in SIDES:
            row = reset["hands"][side]
            bodies, joints = row["body_names"], row["joint_names"]
            if (len(bodies) != 23 or len(set(bodies)) != 23 or len(joints) != 17
                    or set(joints) != set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)):
                raise ValueError("complete actual 23-link/17-DOF Reset hand required")
            types = np.asarray(row["dof_types"])
            if types.shape != (1, 17) or not np.array_equal(
                    types[0], [1 if n in WRIST_LINEAR else 0 for n in joints]):
                raise ValueError("actual Reset DOF types differ")
            for field in ("q", "qd", "backend_targets"):
                reset_owner._array(row[field], (1, 17), "actual "+field)
            poses = reset_owner._array(row["body_poses_xyzw"], (1, 23, 7), "actual body poses")
            if not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1., rtol=0, atol=1e-5):
                raise ValueError("actual Reset body XYZW quaternion must be unit")
            first = bindings[side+"."+reset_owner.PADS[0]]
            container = str(PurePosixPath(first["prim_path"]).parent)
            if (PurePosixPath(container).name != side+"Hand"
                    or row["articulation_paths"] not in ([container], [container+"/wrist_fixed"])
                    or row["link_paths"] != [[container+"/"+n for n in bodies]]):
                raise ValueError("side-specific actual Reset link paths differ")
            for name in links:
                key = side+"."+name
                binding, idx = bindings[key], bodies.index(name)
                if (binding["side"] != side or binding["body_name"] != name
                        or binding["body_index"] != idx or binding["prim_path"] != row["link_paths"][0][idx]
                        or binding.get("frame") != "PhysX_link_origin" or binding.get("unit") != "m"
                        or binding["source_usd_sha256"] != self.geometry.sources[side]["source_sha256"]["usd"]):
                    raise ValueError("Reset contact identity/frame/unit/USD source differs")
                if name in reset_owner.PADS:
                    locked = self.geometry.sources[side]["sdf_binding"][name]
                    if (binding["sdf_sha256"] != locked["sha256"]
                            or binding["capture_sha256"] != locked["capture_sha256"]):
                        raise ValueError("Reset pad locked SDF source differs")
                position, quaternion = reset["pad_poses_xyzw"][key]
                if not np.array_equal(np.r_[position, quaternion], poses[0, idx]):
                    raise ValueError("Reset rod-collider pose copy differs from actual body pose")
            hands[side] = dict(body_names=list(bodies), link_poses_xyzw=poses)
        raw = dict(identity=deepcopy(identity), hands=hands,
                   rod_nodes_m=reset_owner._array(rod["node_positions"], (1, 41, 3), "actual Reset rod")[0])
        packet = dict(raw=raw, raw_sha256=_digest(reset), sources=deepcopy(self.geometry.sources))
        result = self._metric(packet, radius)
        result["version"] = VERSION
        result["source"].update(version=VERSION, boundary="actual_reset_readback",
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            metric_implementation_sha256=hashlib.sha256(
                (Path(__file__).parent/"tabletop_native_teacher_policy_tripod_distance.py").read_bytes()).hexdigest(),
            reset_owner_source_sha256=hashlib.sha256(Path(reset_owner.__file__).read_bytes()).hexdigest(),
            original_config_sha256=config_sha, observed_mesh_count=len(names),
            pose_api="PhysX get_link_transforms", reset_body_pose_is_not_authored_seed=True)
        result["source_sha256"] = _digest(result["source"])
        sample = result["auxiliary"]["samples"]["approach_distance_m"]
        sample["source"] = None if sample["value"] is None else "actual_Tripod_Reset_SDK_material41_SDF:"+result["source_sha256"]
        result["reset_approach"] = dict(
            sample_identity={k: identity[k] for k in IDENTITY if k != "run_uuid"},
            approach_distance_m=deepcopy(sample))
        result["reset_baseline_available"] = sample["value"] is not None
        result["r0_native_reset_complete"] = False
        result["coupling_mesh_reset_qualified"] = False
        result["observed_mesh_count"] = len(names)
        return result
