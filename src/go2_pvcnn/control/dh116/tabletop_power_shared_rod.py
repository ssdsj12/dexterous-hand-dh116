"""Opt-in Power composition with both original palm meshes on one DeformX rod.

The historical eleven-slot runtime remains frozen for existing receipts.
This successor preserves its slot order and appends two independent palms.
"""
from copy import deepcopy
import math
from pathlib import PurePosixPath
import re
from types import SimpleNamespace

import numpy as np

from .deformx_dh116_colliders import DISTAL_PAD_LINKS
from .tabletop_deformable_grasp_runtime import DT, FreeRodRuntime, named_wrenches
from .tabletop_dual_hand_rod_runtime import DualHandRodRuntime as OriginalDualHandRodRuntime
from .tabletop_deformable_solver_clock import solver_clock_matches
from .tabletop_deformx_pair_capture import read_pair_capture
from .tabletop_material import hand_material_contract
from scripts.verify_dh116_deformx_rigid_bridge import wrench_at_com

SIDES=("left","right")
PADS=tuple(DISTAL_PAD_LINKS)
HAND_NAMES=tuple(f"{side}.{name}" for side in SIDES for name in PADS)
MESH_NAMES=tuple(["left."+name for name in PADS]+["table"]+["right."+name for name in PADS])
PALM_NAMES=tuple(f"{side}.base_link" for side in SIDES)
POWER_MESH_NAMES=(*MESH_NAMES,*PALM_NAMES)
END_SUPPORT_NAMES=("support_0","support_1")
SUPPORTED_POWER_MESH_NAMES=(*POWER_MESH_NAMES,*END_SUPPORT_NAMES)
VERSION="dh116-dual-hand-free-rod-composition-v1"


def contact_links(values):
    names=set(values)
    if names==set(HAND_NAMES):
        return PADS
    if names==set((*HAND_NAMES,*PALM_NAMES)):
        return (*PADS,"base_link")
    raise ValueError("both hands must bind the original five pads or five pads and both palms")


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


def _support_mesh_inputs(grids, poses, *, allow_supports):
    if grids is None and poses is None:
        return {}, {}
    if (not allow_supports or grids is None or poses is None
            or set(grids) != set(END_SUPPORT_NAMES) or set(poses) != set(END_SUPPORT_NAMES)):
        raise ValueError("two static end supports require both palms and a complete pair capture")
    return {name: grid.as_args() for name, grid in grids.items()}, _poses(poses, END_SUPPORT_NAMES)


def _bindings(values):
    links=contact_links(values)
    result={}
    roots={}
    for side in SIDES:
        body_ids=[]
        for name in links:
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
        if len(set(body_ids))!=len(links):
            raise ValueError("original pad body indices must be distinct within each hand")
    if roots["left"]==roots["right"]:
        raise ValueError("left and right articulation roots must differ")
    return result


def route_world_wrenches(wrenches, bindings, poses, centers_of_mass):
    """RopeBatch reports force ON each mesh, world torque about its link origin."""
    bindings=_bindings(bindings)
    links=contact_links(bindings)
    meshes=POWER_MESH_NAMES if "base_link" in links else MESH_NAMES
    poses=_poses(poses,bindings)
    if set(centers_of_mass)!=set(bindings) or set(wrenches)!=set(meshes):
        raise ValueError("exact contact link COM and independent mesh wrenches required")
    centers={name:_array(centers_of_mass[name],(3,),"world COM") for name in bindings}
    vectors={name:_array(wrenches[name],(6,),"mesh wrench") for name in meshes}
    routes={}
    for side in SIDES:
        keys=[side+"."+name for name in links]
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
    counts=[]
    for side in SIDES:
        row=routes[side];ids=row["body_ids"]
        if (not isinstance(ids,list) or len(ids) not in (5,6) or len(set(ids))!=len(ids) or
                any(type(i) is not int or i<0 for i in ids) or row.get("is_global") is not True):
            raise ValueError("five pad indices, optional palm index and world wrench frame required")
        counts.append(len(ids))
        if not all(callable(getattr(composers[side],name,None))
                   for name in ("reset","set_forces_and_torques")):
            raise ValueError("actual composer interface required")
        shape=(1,len(ids),3)
        prepared[side]=dict(forces=to_tensor(_array(row["forces"],shape,"world force")),
            torques=to_tensor(_array(row["torques"],shape,"world COM torque")),
            body_ids=list(ids),is_global=True)
    if counts[0]!=counts[1]:
        raise ValueError("both hand routes must use the same contact layout")
    for side in SIDES:
        composers[side].reset()
        composers[side].set_forces_and_torques(**prepared[side])


class PowerSharedRodRuntime(OriginalDualHandRodRuntime):
    """One environment and one free rod; SDK clock is supplied by its caller."""

    def __init__(self, base, *, bindings, right_grids, right_poses, run_uuid, env_id,
                 rod_uid="rod0", require_pairs=False, palm_grids=None, palm_poses=None,
                 support_grids=None, support_poses=None):
        checked=_bindings(bindings)
        extended="base_link" in contact_links(checked)
        expected=set(PALM_NAMES) if extended else set()
        palm_grids={} if palm_grids is None else palm_grids
        palm_poses={} if palm_poses is None else palm_poses
        if set(palm_grids)!=expected or set(palm_poses)!=expected:
            raise ValueError("Power requires grids and poses for both original palm bindings")
        support_grid_args, support_targets = _support_mesh_inputs(
            support_grids, support_poses, allow_supports=extended and require_pairs is True)
        poses={"right."+name:value for name,value in _poses(right_poses,PADS).items()}
        poses.update(_poses(palm_poses,expected))
        poses.update(support_targets)
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
        grid_args={"right."+name:right_grids[name].as_args() for name in PADS}
        grid_args.update({name:grid.as_args() for name,grid in palm_grids.items()})
        grid_args.update(support_grid_args)
        self._base=base
        self.bindings=checked
        self.mesh_names=POWER_MESH_NAMES if extended else MESH_NAMES
        if support_grid_args:
            self.mesh_names=SUPPORTED_POWER_MESH_NAMES
        self.run_uuid,self.env_id,self.rod_uid=run_uuid,env_id,rod_uid
        self.require_pairs=require_pairs
        self._needs_reset=True
        self._last_solver_time=None
        self._seen_episodes={base.episode_uid}
        # A failed native append consumes the composition too: the caller must
        # dispose the original owned scope rather than try a second append.
        base._dual_hand_composition_claimed=True
        self._append_meshes(poses,grid_args,start=6)

    def _append_meshes(self,poses,grid_args,*,start):
        r=self._base.cfg["rod"]
        material=hand_material_contract(self._base.cfg["hand"]["material_profile"])["hand_contact_material"]
        for slot_id,name in enumerate(self.mesh_names[start:],start):
            pos,quat=poses[name]
            if name in END_SUPPORT_NAMES:
                mu_s=self._base.cfg["table"]["mu_s"]
                mu_k=self._base.cfg["table"]["mu_k"]
            else:
                mu_s=max(r["mu_s"],material["static_friction"])
                mu_k=max(r["mu_k"],material["dynamic_friction"])
            slot=self.rope.add_mesh(*grid_args[name],pos=pos,quat=quat,mu_s=mu_s,mu_k=mu_k)
            if slot!=slot_id:
                raise RuntimeError("actual appended hand or support mesh slot identity differs")
        _array(self.rope.mesh_wrenches(),(1,len(self.mesh_names),6),"dual hand mesh slots")

    @classmethod
    def from_original_reset(cls,original,*,palm_bindings,palm_grids,palm_poses,expected_clock,
                            support_grids=None, support_poses=None):
        """Extend a fresh measured Reset; supports use one new, full pair allocation."""
        if (type(original) is not OriginalDualHandRodRuntime or original._needs_reset
                or original.mesh_names!=MESH_NAMES or original.clock()!=expected_clock
                or expected_clock["rod_tick"]!=0 or expected_clock["reset_id"]<1
                or original._base.last is not None):
            raise ValueError("fresh measured original reset and matching SDK clock required")
        if any(set(value)!=set(PALM_NAMES) for value in (palm_bindings,palm_grids,palm_poses)):
            raise ValueError("both original palms are required for Power")
        checked=_bindings({**original.bindings,**palm_bindings})
        poses=_poses(palm_poses,PALM_NAMES)
        grids={name:grid.as_args() for name,grid in palm_grids.items()}
        support_grid_args, support_targets = _support_mesh_inputs(
            support_grids, support_poses, allow_supports=original.require_pairs is True)
        grids.update(support_grid_args)
        poses.update(support_targets)
        _array(original.rope.mesh_wrenches(),(1,len(MESH_NAMES),6),"unchanged original mesh slots")
        obj=cls.__new__(cls)
        obj._base=original._base
        obj.bindings=checked
        obj.mesh_names=SUPPORTED_POWER_MESH_NAMES if support_grid_args else POWER_MESH_NAMES
        obj.run_uuid,obj.env_id,obj.rod_uid=original.run_uuid,original.env_id,original.rod_uid
        obj.require_pairs=True
        obj._needs_reset=True
        obj._last_solver_time=original._last_solver_time
        obj._seen_episodes=set(original._seen_episodes)
        # A partial append invalidates the original retired owner as well.
        original._needs_reset=True
        obj._append_meshes(poses,grids,start=len(MESH_NAMES))
        if support_grid_args:
            obj.rope.enable_contact_pair_capture(len(obj._base.nodes) * len(obj.mesh_names) * obj._base.cfg["rod"]["solver_substeps"])
        obj._needs_reset=False
        if obj.clock()!=expected_clock:
            raise RuntimeError("palms must register without advancing SDK, rod, or Reset")
        return obj

    def reset(self, *, physics_step, episode_uid, reset_id, pad_poses):
        poses=_poses(pad_poses,self.bindings)
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
        for slot,name in enumerate(self.mesh_names[6:],6):
            if name in self.bindings:
                self.rope.set_mesh_poses(slot,*poses[name],teleport=True)
        if not math.isfinite(float(self.rope.time)) or float(self.rope.time)!=before:
            raise RuntimeError("reset unexpectedly advanced solver time")
        row.update(rod_mesh_contact_pairs=None,pair_capture_status="invalid_until_measured_step",
                   hand_wrenches_com_world=None,table_wrench_world=None)
        if self.mesh_names == SUPPORTED_POWER_MESH_NAMES:
            row["external_support_wrenches_world"] = None
        self._last_solver_time=before
        self._needs_reset=False
        return deepcopy(self._decorate(row))

    def step(self, pad_poses, centers_of_mass, *, physics_step):
        if self._needs_reset:
            raise RuntimeError("valid fresh reset required before another step")
        b=self._base
        if type(physics_step) is not int or physics_step!=b.physics_step+1:
            raise ValueError("SDK physics step must advance exactly once")
        poses=_poses(pad_poses,self.bindings)
        if set(centers_of_mass)!=set(self.bindings):
            raise ValueError("both hands' exact COM bindings required")
        centers={name:_array(centers_of_mass[name],(3,),"world COM") for name in self.bindings}
        pair_reader=getattr(self.rope,"contact_pair_samples",None)
        if self.require_pairs and not callable(pair_reader):
            raise ValueError("required genuine contact pair capture unavailable")
        before=float(self.rope.time)
        b.last=None
        try:
            if before!=self._last_solver_time:
                raise ValueError("solver time changed outside the owned exchange")
            b.physics_step=physics_step  # SDK has already advanced in the caller.
            for slot,name in enumerate(self.mesh_names):
                if name in self.bindings:
                    self.rope.set_mesh_poses(slot,*poses[name],teleport=False)
            previous_tick=b.rod_tick
            b.rod_tick=None  # A throwing solver call has unknown partial progress.
            self.rope.step_free(DT,b.cfg["rod"]["solver_substeps"])
            # Never invent rollback after the real solver has already advanced.
            b.rod_tick=previous_tick+b.cfg["rod"]["solver_substeps"]
            if self.rope.last_error:
                raise RuntimeError(f"solver error: {self.rope.last_error}")
            after=float(self.rope.time)
            if not solver_clock_matches(before,after,steps=1):
                raise ValueError("solver and SDK step clocks differ")
            row=b.capture(physics_step,contact_valid=False)
            nodes=_array(self.rope.contact_forces(),(1,len(b.nodes),3),"node contact forces")
            meshes=_array(self.rope.mesh_wrenches(),(1,len(self.mesh_names),6),"named mesh wrenches")
            named=named_wrenches(meshes,self.mesh_names)
            residual=float(np.linalg.norm(nodes.sum((0,1))+meshes[0,:,:3].sum(0)))
            if residual>1e-3:
                raise ValueError("rod and registered mesh forces do not balance")
            pairs=None
            if callable(pair_reader):
                packet=pair_reader()
                if packet.get("substeps")!=b.cfg["rod"]["solver_substeps"]:
                    raise ValueError("pair capture solver substep count differs")
                if self.mesh_names == SUPPORTED_POWER_MESH_NAMES:
                    capacity=len(b.nodes)*len(self.mesh_names)*b.cfg["rod"]["solver_substeps"]
                    if (type(packet.get("capacity")) is not int or packet["capacity"]!=capacity
                            or type(packet.get("mesh_count")) is not int or packet["mesh_count"]!=len(self.mesh_names)
                            or packet.get("overflow") is not False):
                        raise ValueError("complete ordered Power end-support contact capture required")
                for observed,expected in (
                    (packet["node_forces_on_rod_world_n"],nodes[0]),
                    (packet["mesh_wrenches_on_mesh_world"],meshes[0])):
                    values=_array(observed,expected.shape,"atomic pair totals")
                    if not np.allclose(values,expected,atol=3e-4,rtol=5e-5):
                        raise ValueError("atomic pair capture does not match this step's totals")
                pairs=read_pair_capture(SimpleNamespace(contact_pair_samples=lambda:packet),
                    dt_s=DT,physics_step=physics_step,rod_tick=b.rod_tick,mesh_names=self.mesh_names)
                if self.mesh_names in (POWER_MESH_NAMES, SUPPORTED_POWER_MESH_NAMES):
                    pairs["solver_buffer"] = {key: packet[key] for key in (
                        "substeps", "mesh_count", "capacity", "attempted", "overflow"
                    )}
            hand_meshes={name:named[name] for name in (*self.bindings,"table")}
            routes,table=route_world_wrenches(hand_meshes,self.bindings,poses,centers)
            if self.mesh_names == SUPPORTED_POWER_MESH_NAMES:
                row["external_support_wrenches_world"]={key:named[key].copy() for key in ("table",*END_SUPPORT_NAMES)}
            row.update(contact_valid=True,distributed_contact=nodes,named_wrenches=named,
                rod_mesh_contact_pairs=pairs,pair_capture_status="validated_solver_samples" if pairs else "unavailable",
                hand_wrenches_com_world=routes,table_wrench_world=table,pair_force_residual_n=residual)
            self._decorate(row)
            b.last=row
            self._last_solver_time=after
            return deepcopy(row)
        except BaseException:
            self._needs_reset=True
            b.last=None
            raise
