"""Mesh rigid body built on Open3D raycasting."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from numba import njit

from elastica.rigidbody.rigid_body import RigidBodyBase


@njit(cache=True)
def _contact_candidate_mask(local, radii, box_min, box_max):
    """Conservative point/sphere-to-AABB test; no fastmath or stale geometry.

    Distance to the containing box is a lower bound on distance to the mesh.
    Interior and boundary points stay candidates. Keep the existing float32
    ray-query padding; exact mesh queries still determine every contact force.
    """
    result = np.empty(len(local), dtype=np.bool_)
    box_scale = 1.0
    for axis in range(3):
        box_scale = max(box_scale, abs(box_min[axis]), abs(box_max[axis]))
    for i in range(len(local)):
        scale = box_scale
        squared_distance = 0.0
        for axis in range(3):
            value = local[i, axis]
            scale = max(scale, abs(value))
            delta = max(box_min[axis] - value, value - box_max[axis], 0.0)
            squared_distance += delta * delta
        padded = radii[i] + 64 * np.finfo(np.float32).eps * scale
        result[i] = squared_distance <= padded * padded
    return result


class MeshRigidBody(RigidBodyBase):
    """
    Rigid body representing a triangle mesh.

    The mesh geometry is stored in the material frame and never mutated; queries
    transform points into the material frame to avoid BVH rebuilds.

    Notes
    -----
    - Geometry is assumed to be COM-centered in the material frame. This class
      does not estimate or recenter COM automatically.
    - If `center_of_mass` is omitted, world COM is initialized at the origin.
    """

    # Contact probe tolerances (mirrors mytest/contact_test.py Option B)
    _EPS_DIST = 1.0e-9
    _EPS_SURFACE = 1.0e-6
    _OCCUPANCY_NSAMPLES = 1

    def __init__(
        self,
        mesh,
        center_of_mass: NDArray[np.float64] | None = None,
        mass_second_moment_of_inertia: NDArray[np.float64] | None = None,
        density: float | None = None,
        volume: float | None = None,
    ) -> None:
        super().__init__()

        # Mesh geometry is COM-centered in the material frame.
        self._mesh_material_frame = mesh.mesh
        self._vertices_material = np.asarray(
            self._mesh_material_frame.vertices, dtype=np.float32
        )
        self._contact_box_min = self._vertices_material.min(axis=0).astype(np.float64)
        self._contact_box_max = self._vertices_material.max(axis=0).astype(np.float64)
        self._triangles_indices = np.asarray(
            self._mesh_material_frame.triangles, dtype=np.uint32
        )
        self._triangle_normals_material = np.asarray(
            self._mesh_material_frame.triangle_normals, dtype=np.float64
        )
        self._is_watertight = bool(getattr(mesh, "is_watertight", False))
        # A single parity ray can misclassify an exterior point when its ray
        # passes through nearly coplanar triangle boundaries. Use halfspaces
        # ONLY when the existing, closed mesh itself is proven convex; do not
        # replace a concave mesh by its hull or change its nearest surfaces.
        self._convex_planes = None
        vertices64=np.asarray(self._mesh_material_frame.vertices,dtype=np.float64)
        normals=self._triangle_normals_material
        tolerance=64*np.finfo(np.float64).eps*max(1.,float(np.abs(vertices64).max()))
        if self._is_watertight and np.allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-10,rtol=0):
            offsets=-np.sum(normals*vertices64[self._triangles_indices[:,0]],axis=1)
            convex=True
            for start in range(0,len(normals),64):
                if np.max(vertices64@normals[start:start+64].T+offsets[start:start+64])>tolerance:
                    convex=False;break
            if convex:self._convex_planes=(normals.copy(),offsets,tolerance)
        self.signed_distance_method=('convex-halfspaces' if self._convex_planes is not None
                                     else 'ray-parity' if self._is_watertight else 'unsigned-open-mesh')

        self._scene = self._build_raycasting_scene(
            self._mesh_material_frame, self._vertices_material, self._triangles_indices
        )

        if density is None:
            density = 1.0
        if volume is None:
            volume = mesh.compute_volume()
        if center_of_mass is None:
            # Strict assumption: material-frame COM is already at origin.
            center_of_mass = np.zeros(3, dtype=np.float64)
        if mass_second_moment_of_inertia is None:
            # Compute inertia about the provided/assumed COM at origin.
            mass_second_moment_of_inertia = mesh.compute_inertia_tensor(
                density=density, com=np.zeros(3, dtype=np.float64)
            )

        self.density = np.float64(density)
        self.volume = np.float64(volume)
        self.mass = np.float64(self.volume * self.density)

        moi = np.asarray(mass_second_moment_of_inertia, dtype=np.float64)
        self.mass_second_moment_of_inertia = moi.reshape(3, 3, 1)
        self.inv_mass_second_moment_of_inertia = np.linalg.inv(moi).reshape(3, 3, 1)

        self.director_collection = np.zeros((3, 3, 1), dtype=np.float64)
        self.director_collection[:, :, 0] = np.eye(3, dtype=np.float64)
        self.position_collection = np.asarray(center_of_mass, dtype=np.float64).reshape(
            3, 1
        )

        self.velocity_collection = np.zeros((3, 1), dtype=np.float64)
        self.omega_collection = np.zeros((3, 1), dtype=np.float64)
        self.acceleration_collection = np.zeros((3, 1), dtype=np.float64)
        self.alpha_collection = np.zeros((3, 1), dtype=np.float64)

        self.external_forces = np.zeros((3, 1), dtype=np.float64)
        self.external_torques = np.zeros((3, 1), dtype=np.float64)

        bbox = self._mesh_material_frame.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent(), dtype=np.float64)
        self.radius = np.float64(0.5 * np.max(extent))
        self.length = np.float64(np.max(extent))

    @staticmethod
    def _build_raycasting_scene(
        legacy_mesh: o3d.geometry.TriangleMesh,
        vertices: NDArray[np.float32],
        triangles: NDArray[np.uint32],
    ) -> o3d.t.geometry.RaycastingScene:
        scene = o3d.t.geometry.RaycastingScene()
        if hasattr(o3d.t.geometry, "TriangleMesh"):
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
            scene.add_triangles(tmesh)
            return scene
        v_tensor = o3d.core.Tensor(vertices.astype(np.float32))
        f_tensor = o3d.core.Tensor(triangles.astype(np.uint32))
        scene.add_triangles(v_tensor, f_tensor)
        return scene

    def _rotation_matrices(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r_mesh_from_world = self.director_collection[:, :, 0]
        r_world_from_mesh = r_mesh_from_world.T
        return r_mesh_from_world, r_world_from_mesh

    def query_closest_points(
        self, query_points_world: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Closest points, signed distances, and normals for query points in world frame.

        Returns
        -------
        closest_points_world : (N, 3) array
        distances : (N,) array, signed (positive outside, negative inside)
        normals_world : (N, 3) array
        """
        query_pts = np.asarray(query_points_world, dtype=np.float64)
        r_mesh_from_world, r_world_from_mesh = self._rotation_matrices()
        query_material = (
            r_mesh_from_world @ (query_pts - self.position_collection[:, 0]).T
        ).T

        q_tensor = o3d.core.Tensor(query_material.astype(np.float32))
        result = self._scene.compute_closest_points(q_tensor)
        closest_material = result["points"].numpy().astype(np.float64)
        triangle_ids = result["primitive_ids"].numpy().astype(np.int64)
        # check for error in triangle_ids
        if np.any(triangle_ids == np.iinfo(np.uint32).max):
            raise ValueError(
                f"Query point error: {q_tensor}"
            )

        direction = query_material - closest_material
        unsigned_distances = np.linalg.norm(direction, axis=1)

        tri_normals = self._triangle_normals_material
        face_normals = np.zeros_like(direction)
        valid_triangles = (triangle_ids >= 0) & (triangle_ids < tri_normals.shape[0])
        face_normals[valid_triangles] = tri_normals[triangle_ids[valid_triangles]]

        normals_material = np.zeros_like(direction)
        stable = unsigned_distances > self._EPS_DIST
        normals_material[stable] = direction[stable] / unsigned_distances[stable][:, None]
        normals_material[~stable] = face_normals[~stable]

        norm_mag = np.linalg.norm(normals_material, axis=1)
        bad_normals = norm_mag < 1.0e-12
        if np.any(bad_normals):
            normals_material[bad_normals] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            norm_mag = np.linalg.norm(normals_material, axis=1)
        normals_material /= np.maximum(norm_mag, 1.0e-12)[:, None]

        if self._is_watertight:
            if self._convex_planes is not None:
                plane_normals,plane_offsets,tolerance=self._convex_planes
                inside=np.zeros(len(query_material),dtype=bool)
                candidates=np.flatnonzero(np.all(
                    (query_material>=self._contact_box_min-self._EPS_SURFACE)&
                    (query_material<=self._contact_box_max+self._EPS_SURFACE),axis=1))
                for start in range(0,len(candidates),256):
                    ids=candidates[start:start+256]
                    inside[ids]=np.all(query_material[ids]@plane_normals.T+plane_offsets<=tolerance,axis=1)
            else:
                try:
                    occupancy = self._scene.compute_occupancy(
                        q_tensor, nsamples=int(self._OCCUPANCY_NSAMPLES)
                    ).numpy()
                except TypeError:
                    occupancy = self._scene.compute_occupancy(q_tensor).numpy()
                inside = occupancy > 0.5
            inside = inside & (unsigned_distances >= self._EPS_SURFACE)
            sign = np.where(inside, -1.0, 1.0)
            signed_distances = unsigned_distances * sign
            normals_material = normals_material * sign[:, None]
        else:
            signed_distances = unsigned_distances

        closest_world = (
            r_world_from_mesh @ closest_material.T
        ).T + self.position_collection[:, 0]
        normals_world = (r_world_from_mesh @ normals_material.T).T
        distances = signed_distances

        return closest_world, distances, normals_world

    def query_contact_points(self, query_points_world, radii):
        """Exact narrow phase where contact is possible, sentinel elsewhere.

        For the force kernel ONLY: rejected samples have distance > radius and
        no usable closest point/normal. Never use these sentinels as geometric
        observations. An expanded material-frame mesh AABB contains all possible
        contacts, including interior queries. Float32 ray-query roundoff is
        padded conservatively; candidate queries retain the original algorithm.
        """
        points=np.asarray(query_points_world,dtype=np.float64)
        radius=np.asarray(radii,dtype=np.float64)
        if (points.ndim!=2 or points.shape[1]!=3 or radius.shape!=(len(points),)
                or not np.isfinite(points).all() or not np.isfinite(radius).all()
                or np.any(radius<=0)):
            raise ValueError('contact points/radii must be finite N x 3 / positive N arrays')
        local=(points-self.position_collection[:,0])@self.director_collection[:,:,0].T
        candidate=_contact_candidate_mask(
            local,radius,self._contact_box_min,self._contact_box_max)
        closest=points.copy();distance=radius+1.;normal=np.zeros_like(points)
        if np.any(candidate):
            closest[candidate],distance[candidate],normal[candidate]=self.query_closest_points(points[candidate])
        return closest,distance,normal
