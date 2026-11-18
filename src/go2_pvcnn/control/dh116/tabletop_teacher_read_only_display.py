"""Visual-only DH116 copies driven by measured SDK poses.

No App, articulation, controller or physics step is created here. The published
static visual synchronization helper is used only on these nonphysical copies.
"""
from types import SimpleNamespace
import math
import numpy as np

VERSION="dh116-read-only-hand-display-v1"


class _PoseArray:
    """CPU view of real readback rows for the existing visual-sync interface."""
    def __init__(self,values):self.values=values
    def __getitem__(self,index):return _PoseArray(self.values[index])
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.values


def _array(value):
    if hasattr(value,"detach"):value=value.detach().cpu().numpy()
    return np.asarray(value,dtype=np.float64)


class ReadOnlyHandDisplay:
    def __init__(self,stage,bindings,prefix="/World/TeacherDisplay"):
        from pxr import Usd,Sdf,UsdGeom
        self.stage=stage;self.prefix=prefix;self.links=[];self.bindings=[]
        self._layer=None;self._closed=False
        if not prefix.startswith("/World/") or stage.GetPrimAtPath(prefix):
            raise ValueError("fresh owned /World display namespace required")
        identities=set()
        for original in bindings:
            row=dict(original)
            key=(row["env_id"],row["hand"])
            if key in identities or row["hand"] not in ("left","right"):
                raise ValueError("unique environment/hand display identity required")
            identities.add(key)
            row["display_root"]=f"{prefix}/env_{row['env_id']}/{row['hand']}Hand"
            self.bindings.append(row)
        # Flatten and deinstance a temporary composed stage only. Original USD
        # references, instanceability and files remain untouched.
        temporary=Usd.Stage.Open(stage.Flatten())
        visual_roots=[]
        for row in self.bindings:
            for name in row["body_names"]:
                path=row["source_root"]+"/"+name+"/visuals"
                if stage.GetPrimAtPath(path):visual_roots.append(path)
        if not visual_roots:raise ValueError("original DH116 visual geometry required")
        for path in visual_roots:
            for _ in range(32):
                root=temporary.GetPrimAtPath(path)
                instances=[p for p in Usd.PrimRange(root) if p.IsInstance()]
                if not instances:break
                for prim in instances:prim.SetInstanceable(False)
            else:raise ValueError("visual instance nesting exceeds bound")
        flattened=temporary.Flatten()
        # All preview opinions live in a removable anonymous session sublayer.
        # Saving the asset/root layer cannot persist this display or hide edits.
        self._layer=Sdf.Layer.CreateAnonymous("dh116-read-only-display.usda")
        stage.GetSessionLayer().subLayerPaths.insert(0,self._layer.identifier)
        try:
            with Usd.EditContext(stage,self._layer):
                UsdGeom.Xform.Define(stage,prefix)
                for row in self.bindings:
                    UsdGeom.Xform.Define(stage,row["display_root"])
                    for name in row["body_names"]:
                        source=row["source_root"]+"/"+name+"/visuals"
                        if source not in visual_roots:continue
                        target=row["display_root"]+"/"+name
                        body=UsdGeom.Xform.Define(stage,target)
                        body.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                        body.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                        if not Sdf.CopySpec(flattened,Sdf.Path(source),
                                self._layer,Sdf.Path(target+"/visuals")):
                            raise ValueError("could not copy original visual geometry")
                        self.links.append(dict(env_id=row["env_id"],hand=row["hand"],
                            link=name,source_visuals=source,display_path=target))
                self.audit_no_physics()
                # Visibility opinions affect rendering only; no physics field
                # is authored, and the original asset layer stays byte-identical.
                for link in self.links:
                    path=link["source_visuals"].rsplit("/",1)[0]
                    UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
        except BaseException:
            self._detach_layer()
            raise

    def audit_no_physics(self):
        from pxr import Usd,UsdPhysics
        root=self.stage.GetPrimAtPath(self.prefix)
        count=0
        for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
            count+=1
            schemas=[str(v).lower() for v in prim.GetAppliedSchemas()]
            properties=[str(v.GetName()).lower() for v in prim.GetProperties()]
            if (any(v.startswith(("physics","physx")) for v in schemas)
                    or any(v.startswith(("physics:","physx")) for v in properties)
                    or "joint" in str(prim.GetTypeName()).lower()
                    or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    or prim.HasAPI(UsdPhysics.CollisionAPI)
                    or prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
                raise ValueError("physics component found in display-only subtree: "+str(prim.GetPath()))
        return dict(passed=True,prim_count=count,links=len(self.links),
            rigid_bodies=0,colliders=0,joints=0)

    def sync(self,poses,*,read_clock):
        from pxr import UsdGeom
        from .visual_pose_sync import apply_static_visual_poses
        if self._closed:raise RuntimeError("display owner closed")
        if not callable(read_clock):raise ValueError("actual physics clock reader required")
        before=read_clock()
        arrays={}
        # Validate every input before changing any display pose.
        for row in self.bindings:
            key=(row["env_id"],row["hand"])
            values=_array(poses[key])
            if values.shape!=(len(row["body_names"]),7) or not np.isfinite(values).all():
                raise ValueError("finite world pose rows required")
            norms=np.linalg.norm(values[:,3:],axis=1)
            if np.any(np.abs(norms-1)>1e-4):
                raise ValueError("unit wxyz quaternion required")
            arrays[key]=values
        updated=0
        from pxr import Usd
        with Usd.EditContext(self.stage,self._layer):
            for row in self.bindings:
                key=(row["env_id"],row["hand"])
                proxy=SimpleNamespace(body_names=row["body_names"],
                    data=SimpleNamespace(body_link_pose_w=_PoseArray(arrays[key][None])))
                updated+=apply_static_visual_poses(self.stage,{row["hand"]:proxy},
                    {row["hand"]:row["display_root"]})
        records=[];cache=UsdGeom.XformCache()
        for link in self.links:
            key=(link["env_id"],link["hand"])
            row=next(r for r in self.bindings if (r["env_id"],r["hand"])==key)
            measured=arrays[key][row["body_names"].index(link["link"])]
            matrix=cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(link["display_path"]))
            position=np.asarray(matrix.ExtractTranslation())
            rotation=matrix.ExtractRotationQuat()
            quaternion=np.asarray([rotation.GetReal(),*rotation.GetImaginary()])
            quaternion/=np.linalg.norm(quaternion)
            expected=measured[3:]/np.linalg.norm(measured[3:])
            angle=2*math.acos(float(np.clip(abs(np.dot(quaternion,expected)),0,1)))
            records.append(dict(**link,sdk_position_m=measured[:3].tolist(),
                display_position_m=position.tolist(),sdk_quaternion_wxyz=expected.tolist(),
                display_quaternion_wxyz=quaternion.tolist(),
                position_error_m=float(np.linalg.norm(position-measured[:3])),
                rotation_error_rad=angle))
        after=read_clock()
        if after!=before:raise RuntimeError("display synchronization advanced physics clock")
        return dict(version=VERSION,physics_clock_unchanged=True,
            physics_clock=before,updated_links=updated,records=records,
            max_position_error_m=max(v["position_error_m"] for v in records),
            max_rotation_error_rad=max(v["rotation_error_rad"] for v in records))

    def _detach_layer(self):
        if self._layer is not None:
            identifier=self._layer.identifier
            session=self.stage.GetSessionLayer()
            session.subLayerPaths=[path for path in session.subLayerPaths if path!=identifier]
            self._layer=None

    def close(self):
        if self._closed:return
        self._closed=True
        self._detach_layer()
        self.bindings.clear();self.links.clear();self.stage=None
