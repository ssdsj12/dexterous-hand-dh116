"""Measured Pinch approach geometry; never a contact or stability sensor.

Only existing SDK readback/solver-node evidence is consumed. The distance is
the positive minimum over *sampled material nodes* in each locked pad SDF,
averaged over both required pads per hand. It is not an exact continuous
rod/pad distance. Missing lattice coverage is not interpolated or zero-filled.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import numpy as np

from .tabletop_measured_closure_geometry import MeasuredClosureGeometry,REQUIRED,required_sdf_valid
from .tabletop_deformable_r0_replay import IDENTITY,encode_owned
from . import tabletop_dual_hand_reset as reset_owner
from .tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR
from .linear_coupling import JOINTS
from scripts.replay_dh116_native_parameter_r0 import check_radius_packet

VERSION="dh116-r0-measured-pinch-approach-v1"

def positive_pad_mean(rows):
    if len(rows)!=2 or any(not required_sdf_valid(row) for row in rows):
        return None
    return math.fsum(max(0.,row["minimum_gap_m"]) for row in rows)/2.

class MeasuredPinchApproach:
    def __init__(self,project):
        self.project=Path(project).resolve()
        self.geometry=MeasuredClosureGeometry(project)
        self.source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def capture(self,exchange,*,grasp="pinch"):
        if grasp!="pinch":
            raise ValueError("Pinch-only measured recipe; independent Tripod/Power evidence required")
        packet=self.geometry.validate(exchange["closure_geometry"],exchange)
        # Reuse the exact raw/derived/radius/clock checks used by pinned replay.
        radius=check_radius_packet(exchange["material_director"],exchange["rod"])
        identity=packet["raw"]["identity"]
        if any(identity[k]!=radius["source"]["identity"][k] for k in radius["source"]["identity"]):
            raise ValueError("geometry and native radius clock differs")
        return self._metric(packet,radius)

    def _metric(self,packet,radius):
        identity=packet["raw"]["identity"]
        nodes=packet["raw"]["rod_nodes_m"]
        per_hand={}
        for side in ("left","right"):
            raw=packet["raw"]["hands"][side]
            names=raw["body_names"]
            poses=np.asarray(raw["link_poses_xyzw"])[0]
            pads={}
            for name in REQUIRED:
                pose=poses[names.index(name)]
                transform=self.geometry._pose_matrix(np.r_[pose[:3],pose[6],pose[3:6]])
                pads[name]=self.geometry._sdf(self.geometry._grids[side][name],transform,nodes,
                                               radius=radius["value"])
            per_hand[side]=dict(required_pads=list(REQUIRED),pad_samples=pads,
                positive_gap_mean_m=positive_pad_mean(list(pads.values())))
        distances=[per_hand[s]["positive_gap_mean_m"] for s in ("left","right")]
        value=None if any(d is None for d in distances) else distances
        source=dict(version=VERSION,implementation_sha256=self.source_sha256,
            geometry_raw_sha256=packet["raw_sha256"],geometry_sources=deepcopy(packet["sources"]),
            radius_provenance=radius,units="m",coordinate_frame="world_to_actual_pad_local",
            quantity="discrete_solver_node_to_locked_pad_SDF_positive_gap",
            uncertainty_m=None,hardware_measured=False,contact_qualified=False,
            timestamp_s=identity["timestamp_s"])
        digest=hashlib.sha256(json.dumps(encode_owned(source),sort_keys=True,
            separators=(",",":"),allow_nan=False).encode()).hexdigest()
        auxiliary=dict(identity={k:identity[k] for k in IDENTITY},samples=dict(
            approach_distance_m=dict(value=value,source=None if value is None else
                "actual_SDK_pose_solver_nodes_locked_SDF_discrete_approach:"+digest)))
        return dict(version=VERSION,metric="mean_positive_gap_over_required_sampled_nodes",
            auxiliary=auxiliary,per_hand=per_hand,source=source,source_sha256=digest,
            missing_reason="required_pad_lattice_coverage_unavailable" if value is None else None,
            reset_baseline_available=False,contact_observed=None,contact_inferred_from_geometry=False,stability_measured=False,
            new_physics_steps=0,ppo_updates=0,native_qualification=False,teacher_admission=False)


    def capture_reset(self,reset,parameter_packet,*,grasp="pinch"):
        """Derive from actual post-reset getter buffers, never authored FK/step 3.

        Source extraction is read-only. The caller's native record owner, or
        the pinned historical receipt, owns the reset identity. This method
        does not rerun/reset any owner or qualify native R0 reset ordering.
        """
        if grasp!="pinch":
            raise ValueError("Pinch-only measured reset recipe")
        if (reset.get("version")!=reset_owner.VERSION or reset.get("status")!="complete"
                or reset.get("new_sdk_steps")!=0 or reset.get("new_solver_steps")!=0):
            raise ValueError("completed zero-step actual reset receipt required")
        rod=reset["rod_reset"];identity=reset["after_clock"]
        if (identity!=reset["after_actual_clocks"]["rod"] or identity.get("rod_tick")!=0
                or identity.get("clock_valid") is not True or rod["contact_valid"] is not False
                or set(reset["hands"])!={"left","right"}):
            raise ValueError("actual fresh reset clock/rod/hands required")
        actual=reset["after_actual_clocks"]
        if (type(identity["physics_step"]) is not int or actual["sdk_physics_step"]!=identity["physics_step"]
                or actual["sdk_dt_s"]!=1/480 or not math.isclose(actual["sdk_timestamp_s"],
                    identity["timestamp_s"],rel_tol=0,abs_tol=1e-9)):
            raise ValueError("actual reset SDK timestamp/tick differs")
        reset_owner._no_step(reset["before_clock"],actual)
        radius=check_radius_packet(parameter_packet,rod)
        if radius["source"]["identity"]!=identity:
            raise ValueError("actual reset and native parameter identities differ")
        seed=reset_owner._seed(reset["seed"])
        config_path=self.project/"configs/dh116_tasks/flexible_pinch_candidate_v1.json"
        config_raw=config_path.read_bytes()
        config_sha=hashlib.sha256(config_raw).hexdigest()
        if any(self.geometry.sources[side]["source_sha256"]["old_candidate"]!=config_sha for side in ("left","right")):
            raise ValueError("original candidate config hash differs")
        cfg=json.loads(config_raw)
        reset_owner._verify_after(seed,reset["before_hands"],reset["hands"],
                                  cfg["hand"]["active_effort_limit_nm"])
        hands={}
        expected_pad_keys={side+"."+name for side in ("left","right") for name in reset_owner.PADS}
        if set(reset["pad_poses_xyzw"])!=expected_pad_keys:
            raise ValueError("complete actual reset pad copies required")
        for side in ("left","right"):
            row=reset["hands"][side];names=row["body_names"];joints=row["joint_names"]
            if (len(names)!=23 or len(set(names))!=23 or len(joints)!=17 or
                    set(joints)!=set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)):
                raise ValueError("complete actual 23-link/17-DOF reset hand required")
            types=np.asarray(row["dof_types"])
            if (types.shape!=(1,17) or not np.array_equal(types[0],
                    [1 if n in WRIST_LINEAR else 0 for n in joints])):
                raise ValueError("actual reset DOF types differ")
            poses=reset_owner._array(row["body_poses_xyzw"],(1,23,7),"actual body poses")
            if not np.allclose(np.linalg.norm(poses[...,3:],axis=-1),1.,rtol=0,atol=1e-5):
                raise ValueError("actual reset body XYZW quaternion must be unit")
            first=rod["mesh_bindings"][side+"."+REQUIRED[0]]
            container=str(PurePosixPath(first["prim_path"]).parent)
            if (PurePosixPath(container).name!=side+"Hand"
                    or row["articulation_paths"] not in ([container],[container+"/wrist_fixed"])
                    or row["link_paths"]!=[[container+"/"+n for n in names]]):
                raise ValueError("side-specific actual reset link paths differ")
            for name in reset_owner.PADS:
                binding=rod["mesh_bindings"][side+"."+name]
                locked=self.geometry.sources[side]["sdf_binding"][name]
                idx=names.index(name)
                if (binding["side"]!=side or binding["body_name"]!=name
                        or binding["body_index"]!=idx or binding["prim_path"]!=row["link_paths"][0][idx]
                        or binding["frame"]!="PhysX_link_origin" or binding["unit"]!="m"
                        or binding["sdf_sha256"]!=locked["sha256"]
                        or binding["capture_sha256"]!=locked["capture_sha256"]
                        or binding["source_usd_sha256"]!=self.geometry.sources[side]["source_sha256"]["usd"]):
                    raise ValueError("reset pad identity/SDF source differs")
                position,quaternion=reset["pad_poses_xyzw"][side+"."+name]
                if not np.array_equal(np.r_[position,quaternion],poses[0,idx]):
                    raise ValueError("reset rod-collider pose copy differs from actual body pose")
            hands[side]=dict(body_names=list(names),link_poses_xyzw=poses)
        raw=dict(identity=deepcopy(identity),hands=hands,
                 rod_nodes_m=reset_owner._array(rod["node_positions"],(1,41,3),"actual reset rod")[0])
        raw_sha=hashlib.sha256(json.dumps(encode_owned(reset),sort_keys=True,
             separators=(",",":"),allow_nan=False).encode()).hexdigest()
        packet=dict(raw=raw,raw_sha256=raw_sha,sources=deepcopy(self.geometry.sources))
        result=self._metric(packet,radius)
        result["source"].update(boundary="actual_reset_readback",pose_api="PhysX get_link_transforms",
            reset_body_pose_is_not_authored_seed=True,reset_owner_source_sha256=hashlib.sha256(
                Path(reset_owner.__file__).read_bytes()).hexdigest(),original_config_sha256=config_sha)
        source_sha=hashlib.sha256(json.dumps(encode_owned(result["source"]),sort_keys=True,
             separators=(",",":"),allow_nan=False).encode()).hexdigest()
        result["source_sha256"]=source_sha
        sample=result["auxiliary"]["samples"]["approach_distance_m"]
        sample["source"]=None if sample["value"] is None else "actual_reset_SDK_pose_solver_nodes_SDF:"+source_sha
        result["reset_approach"]=dict(
            sample_identity={k:identity[k] for k in IDENTITY if k!="run_uuid"},
            approach_distance_m=deepcopy(sample))
        result["reset_baseline_available"]=sample["value"] is not None
        result["r0_native_reset_complete"]=False
        return result
