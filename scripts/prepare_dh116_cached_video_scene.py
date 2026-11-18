"""CPU-only: freeze an asset-exact, physics-free scene for cached video rendering."""
import json,hashlib,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
    import numpy as np
    from go2_pvcnn.control.dh116.tabletop_teacher_cached_replay import audited_link_poses
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene_api
    rows=[json.loads(s) for s in (args.source/"native_frames.jsonl").read_text().splitlines()]
    assert len(rows)==481 and [x["policy_step"] for x in rows]==list(range(481))
    bindings,poses=audited_link_poses(rows[0])
    for row in rows:audited_link_poses(row,bindings)
    prepared=scene_api.prepare_scene(ROOT)
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,"/World")
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    source_hashes={}
    for b in bindings:
        path=ROOT/prepared["hands"][b["hand"]]["usd_path"]
        source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        root=f"/World/ReplaySources/env_{b['env_id']}/{b['hand']}Hand"
        stage.DefinePrim(root,"Xform").GetReferences().AddReference(str(path))
        b["source_root"]=root
    display=ReadOnlyHandDisplay(stage,bindings)
    audit=display.sync(poses,read_clock=lambda:("no_simulation_context",0))
    assert audit["max_position_error_m"]<=1e-4 and audit["max_rotation_error_rad"]<=1e-3
    reference=json.loads((ROOT/"configs/dh116_training/pose_reference.json").read_text())["reference_layout"]
    support=json.loads((args.source/"supported_scene.json").read_text())
    def box(path,center,size,color):
        p=UsdGeom.Cube.Define(stage,path);p.CreateSizeAttr(1)
        p.AddTranslateOp().Set(Gf.Vec3d(*center));p.AddScaleOp().Set(Gf.Vec3d(*size))
        p.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    for i in range(3):
        origin=np.array([0,i*3,0.])
        box(f"/World/ReplayTable_{i}",np.array(reference["table_center"])+origin,reference["table_size"],(.55,.55,.55))
        for j,s in enumerate(support["supports"]):
            box(f"/World/ReplaySupport_{i}_{j}",np.array(s["center_m"])+origin,s["size_m"],(.3,.32,.35))
        curve=UsdGeom.BasisCurves.Define(stage,f"/World/ReplayWire_{i}")
        curve.CreateTypeAttr("linear");curve.CreateWrapAttr("nonperiodic")
        curve.CreateCurveVertexCountsAttr([41])
        curve.CreateWidthsAttr([support["wire_radius_m"]*2]);curve.SetWidthsInterpolation("constant")
        curve.CreateDisplayColorAttr([Gf.Vec3f(.95,.3,.025)])
    flattened=Usd.Stage.Open(stage.Flatten())
    # Keep source materials available, but strip every physics opinion from this
    # new render-only copy. No original USD layer is authored.
    joints=[]
    for p in flattened.Traverse():
        if "joint" in p.GetTypeName().lower():joints.append(str(p.GetPath()));continue
        p.SetMetadata("apiSchemas",Sdf.TokenListOp.CreateExplicit([str(a) for a in p.GetAppliedSchemas()
            if not str(a).lower().startswith(("physics","physx"))]))
        for prop in list(p.GetProperties()):
            if prop.GetName().lower().startswith(("physics:","physx:")):p.RemoveProperty(prop.GetName())
    for p in sorted(joints,key=len,reverse=True):flattened.RemovePrim(p)
    for p in flattened.Traverse():
        assert not p.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not p.HasAPI(UsdPhysics.CollisionAPI)
        assert not any(str(a).lower().startswith(("physics","physx")) for a in p.GetAppliedSchemas())
        assert "joint" not in p.GetTypeName().lower()
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/"recorded_scene.usda";flattened.GetRootLayer().Export(str(path))
    manifest=dict(source_directory=str(args.source),source_ledger_sha256=hashlib.sha256((args.source/"native_frames.jsonl").read_bytes()).hexdigest(),
        source_asset_sha256=source_hashes,scene_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bindings=bindings,source_frames=481,render_only=True,physics_components=0,
        new_physics_steps=0,new_solver_steps=0,teacher_queries=0,first_pose_audit={k:v for k,v in audit.items() if k!="records"})
    (args.output/"scene_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(output=str(args.output),scene_sha256=manifest["scene_sha256"],first_pose_audit=manifest["first_pose_audit"])))
    display.close()
if __name__=="__main__":main()
