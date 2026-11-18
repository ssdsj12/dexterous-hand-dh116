"""Portable original DeformX wire and world-space Cosserat-to-USD skin driver.

USD uses row vectors. Rod directors are material-frame rows; the original
Blender bone's local Y is the tangent (director row 2).
"""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt


def build_asset(source: Path, output: Path) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != '990c18a99064c700fb4a742070aefff6036a8efd5bbb7c6bf8e56f873398d5a9':
        raise ValueError('Expected the original DeformX 1 m, 5 mm, 20-bone source asset')
    original = Usd.Stage.Open(str(source))
    if not original or not original.GetDefaultPrim():
        raise ValueError('Source must have a default prim')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / source.name)
    dependencies = []
    for prim in original.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                value = attr.Get()
                if not value or not value.path:
                    continue
                relative = Path(value.path)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(f'Nonportable dependency: {relative}')
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source.parent / relative, target)
                dependencies.append(str(relative))
    entry = output / 'wire.usda'
    if entry.exists():
        entry.unlink()
    stage = Usd.Stage.CreateNew(str(entry))
    root = stage.DefinePrim('/Wire', 'Xform')
    root.GetReferences().AddReference('./' + source.name)
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    lights = []
    for prim in list(Usd.PrimRange(root)):
        if 'Light' in prim.GetTypeName():
            lights.append(str(prim.GetPath()))
            prim.SetActive(False)
        for schema in prim.GetAppliedSchemas():
            if 'Physics' in schema or 'Physx' in schema:
                prim.RemoveAppliedSchema(schema)
    stage.GetRootLayer().Save()
    license_path = next((p / 'LICENSE' for p in source.parents if (p / 'LICENSE').is_file()), None)
    if license_path is None:
        raise ValueError('Source license not found')
    shutil.copy2(license_path, output / 'LICENSE')
    manifest = dict(source_project='DeformX', source_asset=source.name,
                    source_sha256=source_hash,
                    license='LICENSE', entry='wire.usda', dependencies=dependencies,
                    length_m=1.0, radius_m=0.005, elements=20, vertices=2652,
                    suppressed_lights=lights,
                    example_material=dict(density_kg_m3=700, young_modulus_pa=200000, shear_modulus_pa=200000/3, damping=0.1, dt_s=1e-5, physics_radius_m=0.005, provenance='DeformX RL demo example; radius intentionally aligned to mesh; not calibrated'),
                    runtime='External Cosserat solver required; no USD rigid-body physics',
                    physics_radius_note='Mesh radius 0.005 m; differs from upstream demo physics radius 0.00635 m')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


class SkeletonWire:
    """One original wire driven by world-space Cosserat nodes and directors.

    ``usd_skel`` leaves deformation to the renderer. ``mesh_points`` publishes
    the original linear-blend skin directly, for renderers that do not consume
    live UsdSkel edits. Only this instance's skin binding is disabled; topology,
    material, rest-data sources, and the animated skeleton are preserved.
    """

    def __init__(self, stage, prim_path, asset_path, *, render_mode='usd_skel'):
        if render_mode not in ('usd_skel', 'mesh_points'):
            raise ValueError("render_mode must be 'usd_skel' or 'mesh_points'")
        self.render_mode = render_mode
        self.stage = stage
        path = Sdf.Path(prim_path)
        if not path.IsAbsolutePath() or path == Sdf.Path.absoluteRootPath or stage.GetPrimAtPath(path):
            raise ValueError('Wire requires a new absolute prim path')
        self.root = stage.DefinePrim(path, 'Xform')
        self.root.GetReferences().AddReference(str(Path(asset_path).resolve()))
        prims = list(Usd.PrimRange(self.root))
        self.skeleton = UsdSkel.Skeleton(next(p for p in prims if p.IsA(UsdSkel.Skeleton)))
        self.skel_root = UsdSkel.Root(next(p for p in prims if p.IsA(UsdSkel.Root)))
        self.mesh = UsdGeom.Mesh(next(p for p in prims if p.IsA(UsdGeom.Mesh)))
        joints = self.skeleton.GetJointsAttr().Get()
        if len(joints) != 20:
            raise ValueError('Expected original 20-bone DeformX wire')
        self.parents = list(UsdSkel.Topology(joints).GetParentIndices())
        self.animation = UsdSkel.Animation.Define(stage, path.AppendChild('RodAnimation'))
        self.animation.CreateJointsAttr(joints)
        self.animation.CreateTranslationsAttr()
        self.animation.CreateRotationsAttr()
        self.animation.CreateScalesAttr()
        UsdSkel.BindingAPI.Apply(self.skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets([self.animation.GetPath()])
        UsdSkel.BindingAPI.Apply(self.skel_root.GetPrim()).CreateAnimationSourceRel().SetTargets([self.animation.GetPath()])
        if self.render_mode == 'mesh_points':
            self._prepare_mesh_skin()

    def _prepare_mesh_skin(self):
        binding = UsdSkel.BindingAPI(self.mesh.GetPrim())
        points = np.asarray(self.mesh.GetPointsAttr().Get(), dtype=float)
        self._rest_points = np.column_stack((points, np.ones(len(points))))
        self._indices = np.asarray(binding.GetJointIndicesPrimvar().ComputeFlattened(), dtype=int).reshape(len(points), -1)
        self._weights = np.asarray(binding.GetJointWeightsPrimvar().ComputeFlattened(), dtype=float).reshape(len(points), -1)
        self._geom_bind = np.asarray(binding.GetGeomBindTransformAttr().Get(), dtype=float)
        self._inverse_binds = np.linalg.inv(np.asarray(self.skeleton.GetBindTransformsAttr().Get(), dtype=float))
        self._rest_normals = np.asarray(self.mesh.GetNormalsAttr().Get(), dtype=float)
        # The original Blender asset authors one normal for each face corner.
        if self.mesh.GetNormalsInterpolation() != UsdGeom.Tokens.faceVarying:
            raise ValueError('Expected original face-varying wire normals')
        self._normal_vertices = np.asarray(self.mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        # A stronger empty relationship and API removal detach this instance
        # from all renderer-side skinning, preventing a second deformation.
        binding.CreateSkeletonRel().SetTargets([])
        self.mesh.GetPrim().RemoveAPI(UsdSkel.BindingAPI)

    def _mesh_pose(self, world_transforms):
        skin = self._geom_bind @ self._inverse_binds @ world_transforms
        blended = np.einsum('vi,vijk->vjk', self._weights, skin[self._indices])
        world_points = np.einsum('vi,vij->vj', self._rest_points, blended)
        mesh_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.mesh.GetPrim()))
        points = (world_points @ np.linalg.inv(mesh_world))[:, :3]
        # Transform each original normal with the same joint influences.
        normal_skin = np.linalg.inv(skin[:, :3, :3]).transpose(0, 2, 1)
        normal_blend = np.einsum('vi,vijk->vjk', self._weights, normal_skin[self._indices])
        normals = np.einsum('vi,vij->vj', self._rest_normals, normal_blend[self._normal_vertices])
        normals = normals @ mesh_world[:3, :3].T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return (Vt.Vec3fArray.FromNumpy(points.astype(np.float32)),
                Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)),
                Vt.Vec3fArray.FromNumpy(np.array([points.min(axis=0), points.max(axis=0)], dtype=np.float32)))

    def update(self, positions, directors):
        positions, directors = np.asarray(positions, dtype=float), np.asarray(directors, dtype=float)
        if positions.shape != (3,21) or directors.shape != (3,3,20):
            raise ValueError('Expected positions (3,21) and directors (3,3,20)')
        if not np.isfinite(positions).all() or not np.isfinite(directors).all():
            raise ValueError('Positions and directors must be finite')
        frames = np.moveaxis(directors, 2, 0)
        if not np.allclose(frames @ frames.transpose(0,2,1), np.eye(3), atol=1e-5, rtol=0) or not np.allclose(np.linalg.det(frames), 1, atol=1e-5, rtol=0):
            raise ValueError('Directors must be orthonormal right-handed frames')
        inverse = UsdGeom.XformCache().GetLocalToWorldTransform(self.skeleton.GetPrim()).GetInverse()
        global_transforms = []
        world_transforms = []
        for j in range(20):
            # Transpose conventional column-vector orientation into USD rows.
            matrix = np.eye(4)
            matrix[:3,:3] = directors[[1,2,0],:,j]
            matrix[3,:3] = positions[:,j]
            world_transforms.append(matrix)
            global_transforms.append(Gf.Matrix4d(*matrix.ravel()) * inverse)
        local = Vt.Matrix4dArray([
            transform if parent < 0 else transform * global_transforms[parent].GetInverse()
            for transform,parent in zip(global_transforms,self.parents)])
        translations, rotations, scales = UsdSkel.DecomposeTransforms(local)
        mesh_pose = self._mesh_pose(np.asarray(world_transforms)) if self.render_mode == 'mesh_points' else None
        # Prepare the complete valid pose before mutating any USD attributes.
        with Sdf.ChangeBlock():
            self.animation.GetTranslationsAttr().Set(translations)
            self.animation.GetRotationsAttr().Set(rotations)
            self.animation.GetScalesAttr().Set(scales)
            if mesh_pose is not None:
                points, normals, extent = mesh_pose
                self.mesh.GetPointsAttr().Set(points)
                self.mesh.GetNormalsAttr().Set(normals)
                self.mesh.GetExtentAttr().Set(extent)
