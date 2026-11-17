"""Same complete original guard/effective-pad geometry, independent frozen plan."""
import hashlib
from pathlib import Path
from scripts import prepare_dh116_tracking_pinch as plan
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from . import tabletop_effective_right_pinch_geometry as original

def validate_projection(pair,exchange,expected_actions=None):
    with bindings((original,"plan",plan)):
        return original.validate_projection(pair,exchange,expected_actions)

class TrackingPinchGeometry(original.EffectiveRightGeometry):
    def __init__(self,project):
        with bindings((original,"plan",plan)):
            super().__init__(project)
        self.sources["tracking_pinch"]={n:hashlib.sha256((Path(project)/n).read_bytes()).hexdigest()
            for n in (plan.CONFIG,"scripts/prepare_dh116_tracking_pinch.py",
                "src/go2_pvcnn/control/dh116/tabletop_tracking_pinch_geometry.py",
                "scripts/audit_dh116_pinch_tracking_target_path.py")}
        self.sources["tracking_pinch"]["static_and_empirical_proof_sha256"]=plan.PROOF_SHA
    def _evaluate(self,raw,exchange):
        with bindings((original,"plan",plan)):
            return super()._evaluate(raw,exchange)
