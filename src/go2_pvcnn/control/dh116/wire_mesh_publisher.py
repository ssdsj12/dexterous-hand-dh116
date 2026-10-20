"""Presentation-only USD publication; never owns collision or rod state."""
import numpy as np
from pxr import Gf, Vt
from go2_pvcnn.control.fixed_dual_panda_o6.wire_visual import rod_surface


class WireMeshPublisher:
    """Publish a fixed-topology rod skin without repeated topology notices."""

    def __init__(self, mesh, radius):
        self.radius = radius
        self.points = mesh.CreatePointsAttr()
        self.counts = mesh.CreateFaceVertexCountsAttr()
        self.indices = mesh.CreateFaceVertexIndicesAttr()
        self.extent = mesh.CreateExtentAttr()
        self.vertex_count = None

    def __call__(self, snapshot):
        vertices, counts, indices = rod_surface(
            snapshot['rod_position'], snapshot['rod_director'], self.radius)
        if self.vertex_count is not None and len(vertices) != self.vertex_count:
            raise ValueError('wire visual topology changed')
        if self.vertex_count is None:
            self.counts.Set(counts)
            self.indices.Set(indices)
            self.vertex_count = len(vertices)
        self.points.Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        self.extent.Set([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])
