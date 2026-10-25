"""Generate an open-bore rigid straw without starting Isaac.

Thirty-two separate convex annular prisms preserve the hole. The standard
polygonal approximation has inner/outer vertices on the nominal circles;
its explicitly reported inner inradius is slightly smaller than that circle.
Mass and inertia describe the nominal continuous hollow cylinder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def straw_parameters(*, outer_radius_m=.008, wall_thickness_m=.0005,
                     length_m=.5, density_kg_m3=900., collision_segments=32):
    values = np.asarray([outer_radius_m, wall_thickness_m, length_m, density_kg_m3], dtype=float)
    if (not np.isfinite(values).all() or np.any(values <= 0)
            or wall_thickness_m >= outer_radius_m
            or type(collision_segments) is not int or not 8 <= collision_segments <= 256):
        raise ValueError('finite positive dimensions/density and 8..256 integer segments required')
    outer, wall, length, density = map(float, values)
    inner = outer-wall
    mass = density*np.pi*(outer**2-inner**2)*length
    axial = mass*(outer**2+inner**2)/2
    transverse = mass*(3*(outer**2+inner**2)+length**2)/12
    return dict(outer_radius_m=outer, inner_radius_m=inner, wall_thickness_m=wall,
        length_m=length, density_kg_m3=density, axis='Y', collision_segments=collision_segments,
        mass_kg=float(mass), diagonal_inertia_kg_m2=[float(transverse),float(axial),float(transverse)],
        inner_polygon_inradius_m=float(inner*np.cos(np.pi/collision_segments)),
        outer_polygon_inradius_m=float(outer*np.cos(np.pi/collision_segments)))


def collision_wedges(parameters):
    """Each eight-vertex convex prism lies outside the polygonal bore."""
    p = parameters
    outer, inner, half = p['outer_radius_m'], p['inner_radius_m'], p['length_m']/2
    count = p['collision_segments']
    faces = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[3,7,6,2],[0,4,7,3],[1,2,6,5]]
    result = []
    for i in range(count):
        a,b = np.array([i,i+1])*2*np.pi/count
        def point(radius, angle, y):
            return [radius*np.cos(angle),y,radius*np.sin(angle)]
        points = [point(outer,a,-half),point(outer,b,-half),point(outer,b,half),point(outer,a,half),
                  point(inner,a,-half),point(inner,b,-half),point(inner,b,half),point(inner,a,half)]
        result.append((np.asarray(points), [face.copy() for face in faces]))
    return result


def visual_tube(parameters):
    """An annular wall mesh: outside, inside and two ring-shaped end faces."""
    p = parameters
    count = p['collision_segments']
    angles = np.arange(count)*2*np.pi/count
    points = np.asarray([[radius*np.cos(a), y, radius*np.sin(a)]
        for radius,y in ((p['outer_radius_m'],-p['length_m']/2),
                         (p['outer_radius_m'],p['length_m']/2),
                         (p['inner_radius_m'],-p['length_m']/2),
                         (p['inner_radius_m'],p['length_m']/2)) for a in angles])
    faces = []
    for i in range(count):
        j = (i+1)%count
        faces.extend([[i,count+i,count+j,j],
                      [2*count+i,2*count+j,3*count+j,3*count+i],
                      [i,j,2*count+j,2*count+i],
                      [count+i,3*count+i,3*count+j,count+j]])
    return points,faces


def build_straw(output, **kwargs):
    """Write straw.usda and a pinned manifest into a fresh output directory."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    p = straw_parameters(**kwargs)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('refusing to overwrite an existing straw output')
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage,'/Straw')
    root.AddTranslateOp().Set(Gf.Vec3d(0.))
    root.AddOrientOp().Set(Gf.Quatf(1.))
    root.AddScaleOp().Set(Gf.Vec3f(1.))
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(p['mass_kg'])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*p['diagonal_inertia_kg_m2']))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.))
    root.GetPrim().CreateAttribute('straw:axis',Sdf.ValueTypeNames.Token).Set('Y')
    for name in ('outer_radius_m','inner_radius_m','length_m','wall_thickness_m'):
        root.GetPrim().CreateAttribute('straw:'+name,Sdf.ValueTypeNames.Double).Set(p[name])

    def mesh_at(path,points,faces):
        mesh = UsdGeom.Mesh.Define(stage,path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
        mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0)),Gf.Vec3f(*points.max(axis=0))])
        return mesh

    visual = mesh_at('/Straw/Visual',*visual_tube(p))
    visual.CreateDoubleSidedAttr(True)
    material = UsdShade.Material.Define(stage,'/Straw/Looks/PaleYellow')
    shader = UsdShade.Shader.Define(stage,'/Straw/Looks/PaleYellow/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.97,.92,.65))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.35)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    for i,(points,faces) in enumerate(collision_wedges(p)):
        mesh = mesh_at(f'/Straw/Collisions/segment_{i:02d}',points,faces)
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    # No physics material is authored: keep the runtime's existing friction.
    output.mkdir(parents=True)
    asset = output/'straw.usda'
    stage.GetRootLayer().Export(str(asset))
    project = Path(__file__).resolve().parents[4]
    generators = [Path(__file__).resolve(),project/'scripts/build_dh116_hollow_straw.py']
    report = dict(format='dh116-hollow-straw-v1',**p,rigid=True,physics_verified=False,
        default_prim='/Straw',usd_file='straw.usda',
        collision_description=f"{p['collision_segments']} independent convex annular prisms; open polygonal bore",
        mass_model='analytic nominal continuous hollow cylinder',
        physics_material='unmodified runtime default',
        usd_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        generators=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    for path in generators])
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
