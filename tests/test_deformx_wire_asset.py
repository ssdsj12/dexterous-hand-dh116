from pathlib import Path
import shutil
import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdSkel, UsdPhysics, UsdShade, Vt
from go2_pvcnn.control.dh116.deformx_wire_asset import build_asset, SkeletonWire

SOURCE = Path('/home/xk/coding/DeformX/usd/wire/wire_orange_s20_r0.005_l1_smooth.usdc')

@pytest.fixture
def bundle(tmp_path):
    build_asset(SOURCE, tmp_path / 'original')
    shutil.move(tmp_path / 'original', tmp_path / 'relocated')
    return tmp_path / 'relocated' / 'wire.usda'

def pose(bent=False):
    angles = np.linspace(0, 1.1 if bent else 0, 20)
    tangent = np.array([np.sin(angles), np.zeros(20), np.cos(angles)])
    positions = np.zeros((3,21)); positions[:,1:] = np.cumsum(tangent * .05, axis=1)
    directors = np.empty((3,3,20))
    directors[1] = np.array([np.cos(angles), np.zeros(20), -np.sin(angles)])
    directors[2] = tangent
    directors[0] = np.cross(directors[1].T, directors[2].T).T
    return positions, directors

def skin(wire):
    cache=UsdSkel.Cache(); cache.Populate(wire.skel_root, Usd.PrimDefaultPredicate)
    query=cache.GetSkelQuery(wire.skeleton)
    points=Vt.Vec3fArray(wire.mesh.GetPointsAttr().Get())
    assert cache.GetSkinningQuery(wire.mesh.GetPrim()).ComputeSkinnedPoints(query.ComputeSkinningTransforms(), points)
    world=UsdGeom.XformCache().GetLocalToWorldTransform(wire.mesh.GetPrim())
    return np.array([world.Transform(Gf.Vec3d(*p)) for p in points])

def test_portable_original(bundle):
    stage=Usd.Stage.Open(str(bundle))
    meshes=[UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    skels=[UsdSkel.Skeleton(p) for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
    assert len(meshes)==len(skels)==1
    assert len(meshes[0].GetPointsAttr().Get())==2652
    assert len(skels[0].GetJointsAttr().Get())==20
    assert not any('Light' in p.GetTypeName() for p in stage.Traverse())
    assert not any('Physics' in schema or 'Physx' in schema for p in stage.Traverse() for schema in p.GetAppliedSchemas())
    assert (bundle.parent/'textures/color_0C0C0C.exr').is_file()
    assert (bundle.parent/'LICENSE').is_file()

@pytest.mark.parametrize('bent',[False,True])
@pytest.mark.parametrize('transformed',[False,True])
def test_actual_original_vertex_skinning(bundle,bent,transformed):
    stage=Usd.Stage.CreateInMemory(); parent=UsdGeom.Xform.Define(stage,'/Scene')
    if transformed:
        parent.AddTranslateOp().Set((2,3,4)); parent.AddRotateXYZOp().Set((20,30,40)); parent.AddScaleOp().Set((1.4,1.4,1.4))
    UsdPhysics.RigidBodyAPI.Apply(parent.GetPrim())
    unrelated=UsdGeom.Xform.Define(stage,'/Unrelated')
    unrelated.AddTranslateOp().Set((7,8,9))
    before={str(p.GetPath()): [(str(a.GetName()),str(a.Get())) for a in p.GetAttributes()] for p in (parent.GetPrim(), unrelated.GetPrim())}
    w=SkeletonWire(stage,'/Scene/Wire',bundle)
    pos,d=pose(bent); pos+=np.array([[.3],[.4],[.5]])
    w.update(pos,d)
    actual=skin(w)
    # Independently evaluate linear blend skinning of every original vertex.
    binding=UsdSkel.BindingAPI(w.mesh.GetPrim())
    indices=np.array(binding.GetJointIndicesPrimvar().ComputeFlattened()).reshape(2652,-1)
    weights=np.array(binding.GetJointWeightsPrimvar().ComputeFlattened()).reshape(2652,-1)
    original=np.array(w.mesh.GetPointsAttr().Get()); binds=w.skeleton.GetBindTransformsAttr().Get()
    expected=np.zeros_like(original)
    for vertex in range(len(original)):
        for j,weight in zip(indices[vertex],weights[vertex]):
            local=np.array(binds[int(j)].GetInverse().Transform(Gf.Vec3d(*map(float, original[vertex]))))
            expected[vertex]+=weight*(np.column_stack((d[1,:,j],d[2,:,j],d[0,:,j]))@local+pos[:,j])
    np.testing.assert_allclose(actual,expected,atol=1.2e-4 if transformed else 2e-6)
    assert parent.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    assert before=={str(p.GetPath()): [(str(a.GetName()),str(a.Get())) for a in p.GetAttributes()] for p in (parent.GetPrim(), unrelated.GetPrim())}
    if not bent:
        np.testing.assert_allclose(actual,original+pos[:,0],atol=1.2e-4 if transformed else 2e-6)

def test_independent_and_atomic(bundle):
    s=Usd.Stage.CreateInMemory(); a=SkeletonWire(s,'/A',bundle); b=SkeletonWire(s,'/B',bundle)
    a.update(*pose()); a_before=skin(a).copy(); b.update(*pose(True))
    np.testing.assert_array_equal(skin(a),a_before)
    for positions,directors in [(np.zeros((3,20)),pose()[1]),(pose()[0],np.full((3,3,20),np.nan)),(np.full((3,21),np.inf),pose()[1])]:
        before=s.GetRootLayer().ExportToString()
        with pytest.raises(ValueError): a.update(positions,directors)
        assert before==s.GetRootLayer().ExportToString()

@pytest.mark.parametrize('kind',['zero','reflection','skew'])
def test_invalid_director_frames_reject_atomically(bundle,kind):
    stage=Usd.Stage.CreateInMemory(); wire=SkeletonWire(stage,'/Wire',bundle)
    positions,directors=pose(); wire.update(positions,directors)
    if kind=='zero': directors[:]=0
    if kind=='reflection': directors[0]*=-1
    if kind=='skew': directors[0,0,0]+=.1
    before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError): wire.update(positions,directors)
    assert before==stage.GetRootLayer().ExportToString()

def test_builder_rejects_other_asset_before_writing(tmp_path):
    source=tmp_path/'other.usda'; stage=Usd.Stage.CreateNew(str(source)); stage.SetDefaultPrim(stage.DefinePrim('/Cube','Xform')); stage.GetRootLayer().Save()
    with pytest.raises(ValueError): build_asset(source,tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('transformed', [False, True])
def test_published_original_mesh_matches_skin_without_renderer_skinning(bundle, transformed):
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, '/Scene')
    if transformed:
        parent.AddTranslateOp().Set((2, 3, 4))
        parent.AddRotateXYZOp().Set((20, 30, 40))
        parent.AddScaleOp().Set((1.4, 1.4, 1.4))
    reference = SkeletonWire(stage, '/Native', bundle)
    wire = SkeletonWire(stage, '/Scene/Published', bundle, render_mode='mesh_points')
    original_points = np.array(reference.mesh.GetPointsAttr().Get())
    faces = np.array(reference.mesh.GetFaceVertexIndicesAttr().Get())
    counts = np.array(reference.mesh.GetFaceVertexCountsAttr().Get())
    source_before = bundle.read_bytes()
    material_before = UsdShade.MaterialBindingAPI(wire.mesh).ComputeBoundMaterial()[0].GetPath()
    # A nontrivial rigid rotation puts the original vertical rod along world Y.
    rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    for bent in (False, True, False):
        positions, directors = pose(bent)
        positions = rotation @ positions + np.array([[.3], [.4], [.5]])
        directors = np.einsum('ab,ibj->iaj', rotation, directors)
        reference.update(positions, directors)
        wire.update(positions, directors)
        published = np.array(wire.mesh.GetPointsAttr().Get())
        world = UsdGeom.XformCache().GetLocalToWorldTransform(wire.mesh.GetPrim())
        actual = np.array([world.Transform(Gf.Vec3d(*map(float, p))) for p in published])
        np.testing.assert_allclose(actual, skin(reference), atol=3e-6)
        np.testing.assert_allclose(wire.mesh.GetExtentAttr().Get(),
                                   [published.min(axis=0), published.max(axis=0)], atol=1e-7)
        assert not UsdSkel.BindingAPI(wire.mesh).GetInheritedSkeleton()
        cache = UsdSkel.Cache()
        cache.Populate(wire.skel_root, Usd.PrimDefaultPredicate)
        assert not cache.ComputeSkelBindings(wire.skel_root, Usd.PrimDefaultPredicate)
        assert len(wire.animation.GetTranslationsAttr().Get()) == 20
        np.testing.assert_array_equal(wire.mesh.GetFaceVertexIndicesAttr().Get(), faces)
        np.testing.assert_array_equal(wire.mesh.GetFaceVertexCountsAttr().Get(), counts)
        assert UsdShade.MaterialBindingAPI(wire.mesh).ComputeBoundMaterial()[0].GetPath() == material_before
    # Another instance and the portable source still hold the original rest data.
    np.testing.assert_array_equal(reference.mesh.GetPointsAttr().Get(), original_points)
    assert bundle.read_bytes() == source_before


def test_published_mesh_rotates_original_normals_and_rejects_invalid_pose_atomically(bundle):
    stage = Usd.Stage.CreateInMemory()
    native = SkeletonWire(stage, '/Native', bundle)
    wire = SkeletonWire(stage, '/Published', bundle, render_mode='mesh_points')
    normals = np.array(native.mesh.GetNormalsAttr().Get())
    rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    positions, directors = pose()
    wire.update(rotation @ positions, np.einsum('ab,ibj->iaj', rotation, directors))
    np.testing.assert_allclose(wire.mesh.GetNormalsAttr().Get(), normals @ rotation.T, atol=2e-6)
    assert wire.mesh.GetNormalsInterpolation() == native.mesh.GetNormalsInterpolation()
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        wire.update(positions, np.zeros_like(directors))
    assert stage.GetRootLayer().ExportToString() == before


def test_invalid_render_mode_does_not_mutate_stage(bundle):
    stage = Usd.Stage.CreateInMemory()
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='render_mode'):
        SkeletonWire(stage, '/Wire', bundle, render_mode='unknown')
    assert stage.GetRootLayer().ExportToString() == before


def test_published_world_pose_survives_changed_nonuniform_parent_and_mesh_transform(bundle):
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, '/Scene')
    translation = parent.AddTranslateOp()
    parent.AddRotateXYZOp().Set((20, 30, 40))
    parent.AddScaleOp().Set((1.4, .8, 1.1))
    wire = SkeletonWire(stage, '/Scene/Published', bundle, render_mode='mesh_points')
    UsdGeom.Xformable(wire.mesh).AddTranslateOp().Set((.1, -.2, .3))
    native = SkeletonWire(stage, '/Native', bundle)
    positions, directors = pose(True)
    native.update(positions, directors)
    for offset in ((2, 3, 4), (-1, 2, 3)):
        translation.Set(offset)
        wire.update(positions, directors)
        world = UsdGeom.XformCache().GetLocalToWorldTransform(wire.mesh.GetPrim())
        actual = np.array([world.Transform(Gf.Vec3d(*p)) for p in wire.mesh.GetPointsAttr().Get()])
        np.testing.assert_allclose(actual, skin(native), atol=3e-6)
