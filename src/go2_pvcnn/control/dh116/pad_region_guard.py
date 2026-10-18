"""All-triangle non-front clearance, plus local load-patch proximity checks.

Source/DeformX mesh qualification only, not PhysX cooked geometry proof. The
caller supplies actual mesh-COM pose. Non-front triangles are never dropped.
"""
import hashlib
import numpy as np
from .pad_regions import sample_wire
from .front_partition import partition_front_cap
from .pad_front import check_pad_front


class PadRegionGuard:
    def __init__(self,vertices,faces,reference_point,reference_normal):
        import open3d as o3d
        v=np.asarray(vertices,dtype=float).copy();f=np.asarray(faces).copy()
        partition=partition_front_cap(v,f,reference_point,reference_normal)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f))
        if not mesh.is_watertight() or not mesh.is_orientable() or mesh.is_self_intersecting():
            raise ValueError('region guard requires a closed orientable nonintersecting mesh')
        mesh.compute_triangle_normals()
        self.normals=np.asarray(mesh.triangle_normals).copy()
        offsets=-np.sum(self.normals*v[f[:,0]],axis=1)
        tolerance=64*np.finfo(float).eps*max(1.,float(np.abs(v).max()))
        convex=all(np.max(v@self.normals[i:i+64].T+offsets[i:i+64])<=tolerance
                   for i in range(0,len(f),64))
        self._convex_planes=(offsets,tolerance) if convex else None
        self.signed_distance_method='convex-halfspaces' if convex else 'ray-parity'
        self._box_min=v.min(0);self._box_max=v.max(0)
        self.reference_point=np.asarray(reference_point,dtype=float).copy()
        self.reference_normal=np.asarray(reference_normal,dtype=float).copy()
        self.front_face_ids=np.unique(partition['front_source_ids']).tolist();self.total_faces=len(f)
        self.front_cap_method=partition['method']
        self.partially_clipped_face_ids=sorted(set(self.front_face_ids)&set(partition['remainder_source_ids'].tolist()))
        triangles=partition['remainder_triangles']
        self.partition_sha256=hashlib.sha256(triangles.astype('<f8').tobytes()+
            partition['remainder_source_ids'].astype('<i8').tobytes()).hexdigest()
        self.geometry_sha256=hashlib.sha256(v.astype('<f8').tobytes()+f.astype('<i8').tobytes()).hexdigest()
        self.full=o3d.t.geometry.RaycastingScene()
        self.full.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        remainder=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(triangles.reshape(-1,3)),
            o3d.utility.Vector3iVector(np.arange(triangles.size//3).reshape(-1,3)))
        self.other=o3d.t.geometry.RaycastingScene()
        self.other.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(remainder))

    def surface_query(self,local):
        """Original nearest surfaces, robust sign for certified convex meshes."""
        import open3d as o3d
        local=np.asarray(local,dtype=float)
        if local.ndim!=2 or local.shape[1]!=3 or not np.isfinite(local).all():
            raise ValueError('surface query requires finite N x 3 points')
        query=o3d.core.Tensor(local.astype(np.float32))
        hit=self.full.compute_closest_points(query)
        closest=hit['points'].numpy().astype(float);ids=hit['primitive_ids'].numpy()
        vector=local-closest;length=np.linalg.norm(vector,axis=1)
        if self._convex_planes is None:
            distance=self.full.compute_signed_distance(query).numpy().astype(float)
        else:
            offsets,tolerance=self._convex_planes
            inside=np.zeros(len(local),dtype=bool)
            candidates=np.flatnonzero(np.all((local>=self._box_min-tolerance)&
                                             (local<=self._box_max+tolerance),axis=1))
            for start in range(0,len(candidates),256):
                subset=candidates[start:start+256]
                inside[subset]=np.all(local[subset]@self.normals.T+offsets<=tolerance,axis=1)
            distance=length*np.where(inside,-1.,1.)
        normal=self.normals[ids].copy();stable=length>1e-9
        normal[stable]=vector[stable]/length[stable,None]
        normal[distance<0]*=-1
        return closest,distance,normal

    def check(self,wire_nodes,rotation_wb,position_w,radius):
        import open3d as o3d
        R=np.asarray(rotation_wb,dtype=float);p=np.asarray(position_w,dtype=float)
        if (R.shape!=(3,3) or p.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(p).all()
                or not np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0)
                or not np.isclose(np.linalg.det(R),1.,atol=1e-6,rtol=0)
                or not np.isfinite(radius) or radius<=0):
            raise ValueError('invalid region guard pose or wire radius')
        points,allowance=sample_wire(wire_nodes)
        local=(points-p)@R;query=o3d.core.Tensor(local.astype(np.float32))
        closest,distance,normal=self.surface_query(local)
        rounded_radius=float(radius+allowance+1e-7)
        patch=check_pad_front(closest,normal,distance,reference_point=self.reference_point,
            reference_normal=self.reference_normal,wire_radius=rounded_radius,
            patch_radius=.006,normal_cosine=float(np.cos(np.pi/6)))
        distances=self.other.compute_distance(query).numpy().astype(float)
        if not np.isfinite(distances).all():raise ValueError('nonfinite non-front distance')
        bound=float(distances.min()-rounded_radius)
        failures=list(patch['failures'])
        if bound<.001:failures.append('non-front triangle clearance below 1 mm')
        return dict(passed=not failures,failures=failures,nonfront_lower_bound_m=bound,
                    signed_distance_method=self.signed_distance_method,
                    front_cap_method=self.front_cap_method,partition_sha256=self.partition_sha256,
                    local_patch=patch,sample_allowance_m=allowance,
                    numerical_allowance_m=1e-7,wire_radius_m=float(radius),grasp_qualified=False)
