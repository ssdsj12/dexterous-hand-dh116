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
        self._topology = None

    def publish(self, snapshot):
        vertices, counts, indices = rod_surface(
            snapshot['rod_position'], snapshot['rod_director'], self.radius)
        topology = (tuple(counts), tuple(indices))
        if self._topology is not None and (
            len(vertices) != self.vertex_count or topology != self._topology
        ):
            raise ValueError('wire visual topology changed; create a new publisher')
        if self.vertex_count is None:
            self.counts.Set(counts)
            self.indices.Set(indices)
            self.vertex_count = len(vertices)
            self._topology = topology
        self.points.Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        self.extent.Set([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])

    def topology_arrays(self):
        """Return immutable cached topology for diagnostics and tests."""
        if self._topology is None:
            return None
        return self._topology

    def __call__(self, snapshot):
        self.publish(snapshot)
