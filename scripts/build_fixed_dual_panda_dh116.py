"""Zero-offset coaxial simulation mount, NOT a calibrated hardware adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = '/FixedDualPandaDH116'
PROJECT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = PROJECT/'assets/fixed_dual_panda_dh116_v1/manifest.json'


def build(output):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('refusing to overwrite output')
    sources = {}
    records = {(r['side'],r['kind']):r for r in json.loads(SOURCE_MANIFEST.read_text())['sources']}
    for side in ('left','right'):
        for kind in ('panda','dh116'):
            path = Path(records[side,kind]['path'])
            if not path.resolve().is_relative_to(PROJECT):raise ValueError('source escapes project')
            if hashlib.sha256(path.read_bytes()).hexdigest() != records[side,kind]['sha256']:
                raise ValueError(f'source identity mismatch: {side}/{kind}')
            sources[side,kind] = path
    stage = Usd.Stage.CreateInMemory()
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,ROOT).GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    for side, y in (('left',.25),('right',-.25)):
        arm_path = ROOT+f'/{side}_arm'
        arm = stage.DefinePrim(arm_path,'Xform')
        arm.GetReferences().AddReference(str(sources[side,'panda']))
        UsdGeom.Xformable(arm).MakeMatrixXform().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0,y,0)))
        root_joint = UsdPhysics.Joint(stage.GetPrimAtPath(arm_path+f'/{side}_root_joint'))
        root_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0,y,0))
        root_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0))
        root_joint.GetLocalRot0Attr().Set(Gf.Quatf(1))
        root_joint.GetLocalRot1Attr().Set(Gf.Quatf(1))
        hand_path = arm_path+'/dh116'
        hand = stage.DefinePrim(hand_path,'Xform')
        hand.GetReferences().AddReference(str(sources[side,'dh116']))
        cache = UsdGeom.XformCache()
        wrist = stage.GetPrimAtPath(arm_path+f'/{side}_panda_link8')
        palm = stage.GetPrimAtPath(hand_path+'/base_link')
        wrist_from_arm,_ = cache.ComputeRelativeTransform(wrist,arm)
        palm_from_hand,_ = cache.ComputeRelativeTransform(palm,hand)
        UsdGeom.Xformable(hand).MakeMatrixXform().Set(palm_from_hand.GetInverse()*wrist_from_arm)
        hand_root = stage.GetPrimAtPath(hand_path+'/root_joint')
        if not hand_root.IsValid():
            raise ValueError('missing hand world root')
        hand_root.SetActive(False)
        mount = UsdPhysics.FixedJoint.Define(stage,ROOT+f'/joints/{side}_mount')
        mount.CreateBody0Rel().SetTargets([wrist.GetPath()])
        mount.CreateBody1Rel().SetTargets([palm.GetPath()])
        for attr in (mount.CreateLocalPos0Attr(),mount.CreateLocalPos1Attr()): attr.Set(Gf.Vec3f(0))
        for attr in (mount.CreateLocalRot0Attr(),mount.CreateLocalRot1Attr()): attr.Set(Gf.Quatf(1))
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Scene): prim.SetActive(False)
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joints = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    if len(roots)!=2 or len(joints)!=36:
        raise ValueError('invalid mounted topology')
    # Disable every drive in the static derivative; runtime owns explicit gains.
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint):
            d=UsdPhysics.DriveAPI.Apply(p,'angular')
            d.CreateStiffnessAttr(0.); d.CreateDampingAttr(0.); d.CreateMaxForceAttr(0.)
    output.mkdir(parents=True)
    asset = output/'fixed_dual_panda_dh116.usd'
    stage.GetRootLayer().Export(str(asset))
    report = dict(physical_dof=36,active_dof=26,articulation_roots=roots,joints=joints,
                  mount='identity link8 to base_link; simulation-only',drives_enabled=False,
                  physics_verified=False,hardware_calibrated=False,
                  asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                  sources=[dict(side=s,kind=k,path=str(p),sha256=records[s,k]['sha256']) for (s,k),p in sources.items()])
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output),indent=2))
