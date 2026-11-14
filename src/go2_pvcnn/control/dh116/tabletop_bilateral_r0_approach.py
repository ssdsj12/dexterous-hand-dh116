"""Opt-in measured approach for the frozen bilateral original-action scope.

Only actual SDK transforms, locked SDFs and solver node geometry are read.
This creates no contact, stability, slip, reward or Teacher qualification.
"""
import hashlib,json
from pathlib import Path
from .tabletop_r0_measured_approach import MeasuredPinchApproach
from .tabletop_bilateral_owned_geometry import BilateralOwnedGeometry,VERSION as GEOMETRY_VERSION
from .tabletop_deformable_r0_replay import encode_owned

VERSION="dh116-r0-bilateral-measured-pinch-approach-v1"

class BilateralMeasuredPinchApproach(MeasuredPinchApproach):
    def __init__(self,project):
        self.project=Path(project).resolve()
        self.geometry=BilateralOwnedGeometry(self.project)
        self.source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def _metric(self,packet,radius):
        result=super()._metric(packet,radius)
        result["version"]=VERSION
        source=result["source"]
        source.update(version=VERSION,geometry_contract_version=GEOMETRY_VERSION,
            inherited_metric_source_sha256=hashlib.sha256(
                (self.project/"src/go2_pvcnn/control/dh116/tabletop_r0_measured_approach.py").read_bytes()).hexdigest())
        digest=hashlib.sha256(json.dumps(encode_owned(source),sort_keys=True,
            separators=(",",":"),allow_nan=False).encode()).hexdigest()
        result["source_sha256"]=digest
        sample=result["auxiliary"]["samples"]["approach_distance_m"]
        sample["source"]=None if sample["value"] is None else "actual_SDK_pose_solver_nodes_bilateral_segment_SDF_discrete_approach:"+digest
        return result
