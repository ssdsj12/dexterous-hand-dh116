"""Two existing hand-table sources at one actual SDK boundary; no stepping.

Import is safe before AppLauncher. Open only after the original scene reset.
The returned source is a native owner: keep it in NativeOwners, clear it before
STOP/App close. Reporting absence is NA; malformed or saturated data is refused.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

DT=1/480
CAPACITY=4096
SDK=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py")
SDK_SHA="bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b"
LOCAL_PINS={'scripts/dh116_pinch_evidence_v2.py': '11b6dbb785cd5a546500218f91db72100b0801cd95dd6f8a9501537a6c968f5c', 'src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py': 'f1f0f268c552572344ae8d13b8ceaa88471599fd67630b5d5c7a9ab1290b7512'}
SCHEMAS=Path('/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas.py')
SCHEMAS_SHA='ba72769e57cae4729074231a9f5d58cfed1679b713fb0bf1bc3a8b35483084ef'
IDENTITY=("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s","solver_time_s","clock_valid")


def _clock(reader):
    row=deepcopy(reader())
    if not isinstance(row,dict) or any(k not in row for k in IDENTITY):
        raise ValueError("complete actual shared SDK/rod clock required")
    row={k:row[k] for k in IDENTITY}
    if (row["clock_valid"] is not True
        or any(type(row[k]) is not int or row[k]<0 for k in ("env_id","reset_id","physics_step","rod_tick"))
        or any(not isinstance(row[k],str) or not row[k] for k in ("run_uuid","episode_uid","rod_uid"))
        or any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) or row[k]<0
               for k in ("timestamp_s","solver_time_s"))
        or not math.isclose(row["timestamp_s"],row["physics_step"]*DT,rel_tol=0,abs_tol=2e-7)):
        raise ValueError("valid actual 480Hz clock required")
    return row


def _source_pins():
    project=Path(__file__).resolve().parents[4]
    paths={SDK:SDK_SHA,SCHEMAS:SCHEMAS_SHA,**{project/path:sha for path,sha in LOCAL_PINS.items()}}
    for path,sha in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError("contact source implementation changed: "+str(path))
    return {str(path):sha for path,sha in paths.items()}


def _reporting(stage,paths):
    from pxr import PhysxSchema,UsdPhysics
    missing=[]
    for path in paths:
        prim=stage.GetPrimAtPath(path)
        if (not prim or not prim.IsActive() or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or not UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
                or not prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                or PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr().Get()!=0):
            missing.append(path)
    return missing



def _table_active(stage,path):
    from pxr import UsdPhysics
    prim=stage.GetPrimAtPath(path)
    return bool(prim and prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())


def _view_paths(view,expected,table):
    method=getattr(view.get_contact_data,"__func__",None)
    if method is None or Path(method.__code__.co_filename).resolve()!=SDK.resolve():
        raise ValueError("actual installed RigidContactView getter source required")
    sensors=list(view.sensor_paths);filters=deepcopy(view.filter_paths)
    if (view.sensor_count!=17 or view.filter_count!=1
            or view.max_contact_data_count!=CAPACITY
            or len(sensors)!=17 or len(set(sensors))!=17 or set(sensors)!=set(expected)):
        raise ValueError("actual seventeen sensor paths/capacity differ")
    # Installed API delegates filter_paths directly; preserve its raw layout.
    # Single-table broadcast or one resolved single-table row per sensor only.
    flat=[]
    if not isinstance(filters,(list,tuple)):raise ValueError("resolved table filter paths missing")
    for row in filters:
        if isinstance(row,str):flat.append(row)
        elif isinstance(row,(list,tuple)) and len(row)==1 and isinstance(row[0],str):flat.append(row[0])
        else:raise ValueError("unsupported resolved table filter layout")
    if len(flat) not in (1,17) or any(path!=table for path in flat):
        raise ValueError("actual table partner differs")
    return sensors,filters


@dataclass
class BimanualTableContacts:
    """Caller-owned native views and stage; no hidden callback or global cache."""
    stage: object
    views: tuple
    metadata: dict
    last_step: int

    def release(self):
        self.views=()
        self.stage=None


def open_bimanual_table_contacts(stage,hands,prepared,*,read_clock):
    hand=view=asset=None
    views=[]
    try:
        from pxr import UsdPhysics
        from .linear_coupling import JOINTS
        from .tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR
        before=_clock(read_clock);pins=_source_pins()
        if len(hands)!=2 or hands[0] is hands[1] or hands[0].root_physx_view is hands[1].root_physx_view:
            raise ValueError("two distinct original mounted hands required")
        if hands[0]._physics_sim_view is not hands[1]._physics_sim_view:
            raise ValueError("hands must use one actual root simulation view")
        table=prepared["prefix"]+"/Table";prim=stage.GetPrimAtPath(table)
        if not _table_active(stage,table):
            raise ValueError("actual original table collider required")
        paths={}
        for side,hand in zip(("left","right"),hands):
            names=list(hand.body_names);view=hand.root_physx_view;root=prepared["prefix"]+"/"+side+"Hand"
            expected=list(prepared["hands"][side]["source_colliders"])
            if (len(expected)!=17 or len(set(expected))!=17 or len(names)!=23 or len(set(names))!=23
                    or set(hand.joint_names)!=set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)
                    or len(hand.joint_names)!=17 or not set(expected)<=set(names)
                    or list(view.prim_paths)!=[root+"/wrist_fixed"]
                    or view.link_paths!=[[root+"/"+name for name in names]]):
                raise ValueError("actual original 17DOF/17 body source mapping differs")
            paths[side]=[view.link_paths[0][names.index(name)] for name in expected]
        missing=_reporting(stage,paths["left"]+paths["right"])
        static_table=not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        unavailable=("static GPU table filter unsupported; empty buffers cannot prove absence"
                     if static_table else None)
        metadata=dict(table_is_static=static_table,capability_unavailable=unavailable,created_clock=before,source_sha256=pins,paths=paths,table_path=table,
                      missing_reporting=missing,capacity=CAPACITY)
        if not missing and not static_table:
            asset=hands[0]._physics_sim_view
            for side in ("left","right"):
                view=asset.create_rigid_contact_view(paths[side],
                    filter_patterns=[[table] for _ in paths[side]],max_contact_data_count=CAPACITY)
                _view_paths(view,paths[side],table);views.append(view)
            if views[0] is views[1]:raise ValueError("two independent contact views required")
        if _clock(read_clock)!=before:raise ValueError("clock changed while opening contact sources")
        return BimanualTableContacts(stage,tuple(views),metadata,before["physics_step"])
    finally:
        stage=hands=prepared=read_clock=hand=view=asset=prim=None
        views=[]


def read_bimanual_table_contacts(source,*,read_clock):
    view=None
    try:
        before=_clock(read_clock)
        na=dict(available=False,identity=before,dt_s=DT,hands=None,total_contacts=None,
                native_qualified=False,teacher_admission=False,force_sign_frame_qualified=False)
        if source is None:return dict(na,reason="contact source absent")
        meta=deepcopy(source.metadata);created=meta["created_clock"]
        if any(before[k]!=created[k] for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid")):
            raise ValueError("contact source reset/episode identity changed")
        if source.stage is None or len(source.views)!=2:
            return dict(na,reason=meta.get("capability_unavailable") or "contact reporting/views unavailable",source=meta)
        if not _table_active(source.stage,meta["table_path"]):
            return dict(na,reason="actual table collider unavailable",source=meta)
        missing=_reporting(source.stage,meta["paths"]["left"]+meta["paths"]["right"])
        if missing:return dict(na,reason="actual reporting unavailable",missing_reporting=missing,source=meta)
        if before["physics_step"]<=source.last_step:
            raise ValueError("fresh completed SDK step after view creation/read required")
        from scripts.dh116_pinch_evidence_v2 import read_table_contacts
        # The old parser needs its single-hand phase label; this local adapter
        # does not export or synthesize a policy/phase clock.
        legacy=dict(before,phase_step=before["physics_step"]-created["physics_step"])
        hands={}
        for side,view in zip(("left","right"),source.views):
            sensors,filters=_view_paths(view,meta["paths"][side],meta["table_path"])
            row=read_table_contacts(view,legacy);row.pop("phase_step")
            for contact in row["contacts"]:
                contact.update(body_path=sensors[contact["sensor_index"]],table_path=meta["table_path"])
            row.update(sensor_paths=sensors,filter_paths=filters,dt_s=DT)
            if _view_paths(view,meta["paths"][side],meta["table_path"])!=(sensors,filters):
                raise ValueError("contact paths changed during read")
            hands[side]=row
        if _clock(read_clock)!=before:raise ValueError("clock changed during contact capture")
        if not _table_active(source.stage,meta["table_path"]) or _reporting(source.stage,meta["paths"]["left"]+meta["paths"]["right"]):
            raise ValueError("reporting changed during contact capture")
        source.last_step=before["physics_step"]
        return dict(na,available=True,hands=hands,total_contacts=sum(r["total_contacts"] for r in hands.values()),
                    source=meta,source_api="RigidContactView.get_contact_data",
                    upstream_overflow_absence_attested=False)
    finally:
        source=read_clock=view=None
