"""Opt-in full-3D polyline proof; original measured geometry remains the default."""
import hashlib
from pathlib import Path
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from . import tabletop_measured_closure_geometry as original
from .tabletop_polyline_support_certificate import certify,validate as validate_certificate,VERSION as CERTIFICATE_VERSION
from .tabletop_deformable_r0_replay import encode_owned,decode_owned

VERSION="dh116-measured-segment-closure-geometry-v1"

class SegmentClosureGeometry(original.MeasuredClosureGeometry):
    def __init__(self,project):
        super().__init__(project)
        for name in ("src/go2_pvcnn/control/dh116/tabletop_segment_closure_geometry.py",
                     "src/go2_pvcnn/control/dh116/tabletop_polyline_support_certificate.py"):
            self.sources["implementation"][name]=hashlib.sha256((Path(project)/name).read_bytes()).hexdigest()

    def _evaluate(self,raw,exchange):
        certificates=[];old_bound=original.rod_support_lower
        def bound(vertices,nodes):
            old=old_bound(vertices,nodes)
            if old>=0:return old
            packet=certify(vertices,nodes)
            validate_certificate(packet,vertices,nodes)
            certificates.append(packet)
            return packet["lower_m"]
        with bindings((original,"rod_support_lower",bound)):
            packet=super()._evaluate(raw,exchange)
        packet.update(version=VERSION,support_certificate_version=CERTIFICATE_VERSION,
            conservative_refinement_certificates=certificates,
            non_target_bound_scope="full_original_vertices_and_actual_adjacent_node_capsules")
        return packet

    def validate(self,packet,exchange):
        if packet.get("version")!=VERSION or packet.get("sources")!=self.sources:
            raise ValueError("independent segment contract/source identity required")
        rebuilt=self._evaluate(decode_owned(encode_owned(packet["raw"])),exchange)
        if original._bytes(rebuilt)!=original._bytes(packet):
            raise ValueError("segment proof packet differs from full raw recomputation")
        return rebuilt
