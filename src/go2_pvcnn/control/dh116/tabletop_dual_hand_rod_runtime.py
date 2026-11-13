"""Isolated two-hand composition of the frozen one-rod runtime.

No scene, app, policy or native launch is created here. This adapter consumes an
exclusive fresh FreeRodRuntime and adds the second hand's five original SDFs.
All source/contact/native/Teacher qualification remains false.
"""
from copy import deepcopy
import math
from pathlib import PurePosixPath
import re
from types import SimpleNamespace

import numpy as np

from .deformx_dh116_colliders import DISTAL_PAD_LINKS
from .tabletop_deformable_grasp_runtime import DT, FreeRodRuntime, named_wrenches
from .tabletop_deformx_pair_capture import read_pair_capture
from .tabletop_material import hand_material_contract
from scripts.verify_dh116_deformx_rigid_bridge import wrench_at_com

SIDES=("left","right")
PADS=tuple(DISTAL_PAD_LINKS)
HAND_NAMES=tuple(f"{side}.{name}" for side in SIDES for name in PADS)
MESH_NAMES=tuple(["left."+name for name in PADS]+["table"]+["right."+name for name in PADS])
VERSION="dh116-dual-hand-free-rod-composition-v1"


def _array(value, shape, label):
    array=np.array(value,dtype=float,copy=True)
    if array.shape!=shape or not np.isfinite(array).all():
        raise ValueError(f"finite {label} with shape {shape} required")
    return array


def _poses(values, names):
    if set(values)!=set(names):
        raise ValueError("exact named original pad poses required")
    result={}
    for name in names:
        if len(values[name])!=2:
            raise ValueError("world position and xyzw quaternion required")
        pos=_array(values[name][0],(3,),"world link origin")
        quat=_array(values[name][1],(4,),"xyzw quaternion")
        if not math.isclose(float(np.linalg.norm(quat)),1.,abs_tol=1e-5,rel_tol=0):
            raise ValueError("unit xyzw quaternion required")
        result[name]=(pos.astype(np.float32),quat.astype(np.float32))
    return result


def _bindings(values):
    if set(values)!=set(HAND_NAMES):
        raise ValueError("two distinct sets of five original pad bindings required")
    result={}
    roots={}
    for side in SIDES:
        body_ids=[]
        for name in PADS:
            key=side+"."+name
            row=deepcopy(values[key])
            path=PurePosixPath(row["prim_path"])
            if (row.get("side")!=side or row.get("body_name")!=name or
                    not path.is_absolute() or path.name!=name or ".." in path.parts or
                    str(path)!=row["prim_path"] or type(row.get("body_index")) is not int or
                    row["body_index"]<0):
                raise ValueError("side/body/path identity differs")
            for field in ("source_usd_sha256","cooked_geometry_sha256","sdf_sha256"):
                value=row.get(field)
                if not isinstance(value,str) or not re.fullmatch("[0-9a-f]{64}",value) or value=="0"*64:
                    raise ValueError("explicit nonzero source/cooked/SDF SHA256 required")
            roots.setdefault(side,str(path.parent))
            if roots[side]!=str(path.parent):
                raise ValueError("each hand must use one original articulation root")
            body_ids.append(row["body_index"])
            result[key]=row
        if len(set(body_ids))!=5:
            raise ValueError("original pad body indices must be distinct within each hand")
    if roots["left"]==roots["right"]:
        raise ValueError("left and right articulation roots must differ")
    return result


def route_world_wrenches(wrenches, bindings, poses, centers_of_mass):
    """RopeBatch reports force ON each mesh, world torque about its link origin."""
    bindings=_bindings(bindings)
    poses=_poses(poses,HAND_NAMES)
    if set(centers_of_mass)!=set(HAND_NAMES) or set(wrenches)!=set(MESH_NAMES):
        raise ValueError("exact hand COM and eleven independent mesh wrenches required")
    centers={name:_array(centers_of_mass[name],(3,),"world COM") for name in HAND_NAMES}
    vectors={name:_array(wrenches[name],(6,),"mesh wrench") for name in MESH_NAMES}
    routes={}
    for side in SIDES:
        keys=[side+"."+name for name in PADS]
        shifted=np.stack([wrench_at_com(vectors[key],poses[key][0],centers[key]) for key in keys])
        routes[side]=dict(body_ids=[bindings[key]["body_index"] for key in keys],
            forces=shifted[None,:,:3].copy(),torques=shifted[None,:,3:].copy(),is_global=True)
    return routes,vectors["table"].copy()


def write_composer_reactions(composers, routes, *, to_tensor):
    """One exclusive WrenchComposer write per hand; caller owns SDK stepping."""
    if set(composers)!=set(SIDES) or set(routes)!=set(SIDES):
        raise ValueError("both distinct hand composers and routes required")
    if composers["left"] is composers["right"]:
        raise ValueError("left and right composers must be distinct")
    prepared={}
    for side in SIDES:
        row=routes[side];ids=row["body_ids"]
        if (not isinstance(ids,list) or len(ids)!=5 or len(set(ids))!=5 or
                any(type(i) is not int or i<0 for i in ids) or row.get("is_global") is not True):
            raise ValueError("five unique body indices and world wrench frame required")
        if not all(callable(getattr(composers[side],name,None))
                   for name in ("reset","set_forces_and_torques")):
            raise ValueError("actual composer interface required")
        prepared[side]=dict(forces=to_tensor(_array(row["forces"],(1,5,3),"world force")),
            torques=to_tensor(_array(row["torques"],(1,5,3),"world COM torque")),
            body_ids=list(ids),is_global=True)
    for side in SIDES:
        composers[side].reset()
        composers[side].set_forces_and_torques(**prepared[side])


class DualHandRodRuntime:
    """One environment and one free rod; SDK clock is supplied by its caller."""

    def __init__(self, base, *, bindings, right_grids, right_poses, run_uuid, env_id,
                 rod_uid="rod0", require_pairs=False):
        checked=_bindings(bindings)
        poses=_poses(right_poses,PADS)
        if set(right_grids)!=set(PADS):
            raise ValueError("five right original SDFs required")
        if (not isinstance(base,FreeRodRuntime) or base.names!=list(PADS)+["table"] or
                base.physics_step is not None or base.rod_tick!=0 or base.reset_id!=0 or
                getattr(base,"_dual_hand_composition_claimed",False)):
            raise ValueError("exclusive fresh original FreeRodRuntime required")
        if (not isinstance(run_uuid,str) or not run_uuid or type(env_id) is not int or env_id<0 or
                not isinstance(rod_uid,str) or not rod_uid or type(require_pairs) is not bool):
            raise ValueError("explicit run/environment/rod identity required")
        if base.rope.num_envs!=1 or base.rope.fixed_root_orientation:
            raise ValueError("one free rod environment required")
        _array(base.rope.mesh_wrenches(),(1,6,6),"original six mesh slots")
        grid_args={name:right_grids[name].as_args() for name in PADS}
        self._base=base
        self.bindings=checked
        self.mesh_names=MESH_NAMES
        self.run_uuid,self.env_id,self.rod_uid=run_uuid,env_id,rod_uid
        self.require_pairs=require_pairs
        self._needs_reset=True
        self._seen_episodes={base.episode_uid}
        # A failed native append consumes the composition too: the caller must
        # dispose the original owned scope rather than try a second append.
        base._dual_hand_composition_claimed=True
        r=base.cfg["rod"]
        material=hand_material_contract(base.cfg["hand"]["material_profile"])["hand_contact_material"]
        for offset,name in enumerate(PADS,6):
            pos,quat=poses[name]
            slot=base.rope.add_mesh(*grid_args[name],pos=pos,quat=quat,
                mu_s=max(r["mu_s"],material["static_friction"]),
                mu_k=max(r["mu_k"],material["dynamic_friction"]))
            if slot!=offset:
                raise RuntimeError("actual right mesh slot identity differs")
        _array(base.rope.mesh_wrenches(),(1,11,6),"dual hand mesh slots")

    @property
    def rope(self):
        return self._base.rope

    def clock(self):
        b=self._base
        return dict(run_uuid=self.run_uuid,env_id=self.env_id,rod_uid=self.rod_uid,
            episode_uid=b.episode_uid,reset_id=b.reset_id,physics_step=b.physics_step,
            rod_tick=b.rod_tick,timestamp_s=None if b.physics_step is None else b.physics_step*DT,
            solver_time_s=float(self.rope.time),clock_valid=not self._needs_reset)

    def _decorate(self,row):
        row.update(version=VERSION,**self.clock(),source_qualification=False,
            native_qualification=False,contact_pair_qualification=False,grasp_qualified=False,
            full_state371_bound=False,teacher_admission=False,
            binding_provenance="caller_supplied_unverified",mesh_names=list(self.mesh_names),
            mesh_bindings=deepcopy(self.bindings),solver_time_origin_s=self._base.time_origin)
        return row

    def reset(self, *, physics_step, episode_uid, reset_id, pad_poses):
        poses=_poses(pad_poses,HAND_NAMES)
        b=self._base
        if (type(physics_step) is not int or physics_step<0 or
                not isinstance(episode_uid,str) or not episode_uid or episode_uid in self._seen_episodes or
                type(reset_id) is not int or reset_id!=b.reset_id+1):
            raise ValueError("fresh episode, next reset ID and explicit SDK clock required")
        self._needs_reset=True
        b.last=None
        self._seen_episodes.add(episode_uid)
        before=float(self.rope.time)
        row=b.reset(physics_step=physics_step,episode_uid=episode_uid,reset_id=reset_id,
            pad_poses={name:poses["left."+name] for name in PADS})
        for slot,name in enumerate(PADS,6):
            self.rope.set_mesh_poses(slot,*poses["right."+name],teleport=True)
        if not math.isfinite(float(self.rope.time)) or float(self.rope.time)!=before:
            raise RuntimeError("reset unexpectedly advanced solver time")
        row.update(rod_mesh_contact_pairs=None,pair_capture_status="invalid_until_measured_step",
                   hand_wrenches_com_world=None,table_wrench_world=None)
        self._needs_reset=False
        return deepcopy(self._decorate(row))

    def step(self, pad_poses, centers_of_mass, *, physics_step):
        if self._needs_reset:
            raise RuntimeError("valid fresh reset required before another step")
        b=self._base
        if type(physics_step) is not int or physics_step!=b.physics_step+1:
            raise ValueError("SDK physics step must advance exactly once")
        poses=_poses(pad_poses,HAND_NAMES)
        if set(centers_of_mass)!=set(HAND_NAMES):
            raise ValueError("both hands' exact COM bindings required")
        centers={name:_array(centers_of_mass[name],(3,),"world COM") for name in HAND_NAMES}
        pair_reader=getattr(self.rope,"contact_pair_samples",None)
        if self.require_pairs and not callable(pair_reader):
            raise ValueError("required genuine contact pair capture unavailable")
        before=float(self.rope.time)
        b.last=None
        try:
            b.physics_step=physics_step  # SDK has already advanced in the caller.
            for slot,name in enumerate(self.mesh_names):
                if name!="table":
                    self.rope.set_mesh_poses(slot,*poses[name],teleport=False)
            previous_tick=b.rod_tick
            b.rod_tick=None  # A throwing solver call has unknown partial progress.
            self.rope.step_free(DT,b.cfg["rod"]["solver_substeps"])
            # Never invent rollback after the real solver has already advanced.
            b.rod_tick=previous_tick+b.cfg["rod"]["solver_substeps"]
            if self.rope.last_error:
                raise RuntimeError(f"solver error: {self.rope.last_error}")
            after=float(self.rope.time)
            if (not math.isfinite(after) or not math.isclose(after-before,DT,rel_tol=0,abs_tol=2e-6)
                    or not math.isclose(after-b.time_origin,
                        b.rod_tick/b.cfg["rod"]["solver_substeps"]*DT,rel_tol=0,abs_tol=2e-6)):
                raise ValueError("solver and SDK step clocks differ")
            row=b.capture(physics_step,contact_valid=False)
            nodes=_array(self.rope.contact_forces(),(1,len(b.nodes),3),"node contact forces")
            meshes=_array(self.rope.mesh_wrenches(),(1,11,6),"named mesh wrenches")
            named=named_wrenches(meshes,self.mesh_names)
            residual=float(np.linalg.norm(nodes.sum((0,1))+meshes[0,:,:3].sum(0)))
            if residual>1e-3:
                raise ValueError("rod and all eleven mesh forces do not balance")
            pairs=None
            if callable(pair_reader):
                packet=pair_reader()
                if packet.get("substeps")!=b.cfg["rod"]["solver_substeps"]:
                    raise ValueError("pair capture solver substep count differs")
                for observed,expected in (
                    (packet["node_forces_on_rod_world_n"],nodes[0]),
                    (packet["mesh_wrenches_on_mesh_world"],meshes[0])):
                    values=_array(observed,expected.shape,"atomic pair totals")
                    if not np.allclose(values,expected,atol=3e-4,rtol=5e-5):
                        raise ValueError("atomic pair capture does not match this step's totals")
                pairs=read_pair_capture(SimpleNamespace(contact_pair_samples=lambda:packet),
                    dt_s=DT,physics_step=physics_step,rod_tick=b.rod_tick,mesh_names=self.mesh_names)
            routes,table=route_world_wrenches(named,self.bindings,poses,centers)
            row.update(contact_valid=True,distributed_contact=nodes,named_wrenches=named,
                rod_mesh_contact_pairs=pairs,pair_capture_status="validated_solver_samples" if pairs else "unavailable",
                hand_wrenches_com_world=routes,table_wrench_world=table,pair_force_residual_n=residual)
            self._decorate(row)
            b.last=row
            return deepcopy(row)
        except BaseException:
            self._needs_reset=True
            b.last=None
            raise

    def read_rod_snapshot(self):
        if self._needs_reset or self._base.last is None:
            raise RuntimeError("valid measured solver step required")
        packet=self._base.read_rod_snapshot()
        packet.update(env_ids=[self.env_id],source_scope="solver_readback_unqualified",
            source_qualification=False,native_qualification=False,contact_pair_qualification=False,
            full_state371_bound=False,grasp_qualified=False,teacher_admission=False)
        return packet
