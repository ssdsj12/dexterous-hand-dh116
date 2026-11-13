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
from pathlib import Path
import numpy as np

from .tabletop_measured_closure_geometry import MeasuredClosureGeometry,REQUIRED,required_sdf_valid
from .tabletop_deformable_r0_replay import IDENTITY,encode_owned
from scripts.replay_dh116_native_parameter_r0 import check_radius_packet

VERSION="dh116-r0-measured-pinch-approach-v1"

def positive_pad_mean(rows):
    if len(rows)!=2 or any(not required_sdf_valid(row) for row in rows):
        return None
    return math.fsum(max(0.,row["minimum_gap_m"]) for row in rows)/2.

class MeasuredPinchApproach:
    def __init__(self,project):
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
