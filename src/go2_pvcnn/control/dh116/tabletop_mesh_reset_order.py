"""Opt-in zero-step mesh rebuild after the original full hand/rod reset.

The compiled add_mesh immediately commits its own slot. set_mesh_poses(teleport)
does not. No old runtime, backend, recipe or default reset is changed here.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

from .tabletop_dual_hand_reset import DualHandResetTransaction, _no_step
from . import tabletop_dual_hand_reset as original_reset
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings as scoped_bindings
from .tabletop_dual_hand_rod_runtime import MESH_NAMES
from .tabletop_deformable_bimanual_step import _owned

VERSION="dh116-zero-step-reset-mesh-rebuild-v1"
PLAN_VERSION="dh116-original-eleven-mesh-reset-plan-v1"
TASK_CONFIG="configs/dh116_tasks/flexible_dual_hand_grasp_v1.json"
TASK_SHA="8b4f7f8d3fb96c12b9794a7c5daaa2b65f9231699348d5635044bd3f5ac75d63"
_ORIGINAL_CLOCK=original_reset._clock

def _clock(sim,rod):
    """Actual timestamp, tolerating measured FP32 SDK accumulation, never projecting it."""
    try:
        step,stamp,dt=sim.current_time_step_index,float(sim.current_time),float(sim.get_physics_dt())
        if (type(step) is not int or not 0<=step<=240 or not math.isfinite(stamp)
                or not math.isclose(dt,1/480,rel_tol=0,abs_tol=1e-12)):
            raise ValueError("actual 480Hz SDK clock outside this bounded 240-step scope")
        tolerance=step*abs(float(np.float32(dt))-dt)+8*np.finfo(float).eps*max(1.,step*dt)
        if tolerance>=min(2e-7,dt/2) or abs(stamp-step*dt)>tolerance:
            raise ValueError("actual SDK clock exceeds float32 callback accumulation envelope")
        row=deepcopy(rod.clock())
        if not math.isfinite(row["solver_time_s"]):raise ValueError("finite actual solver clock required")
        return dict(sdk_physics_step=step,sdk_timestamp_s=stamp,sdk_dt_s=dt,rod=row)
    finally:sim=rod=None


def _encoded(value):
    if isinstance(value,np.ndarray):
        a=np.ascontiguousarray(value)
        return dict(dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
    if isinstance(value,dict):return {k:_encoded(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_encoded(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value

def _digest(value):
    return hashlib.sha256(json.dumps(_encoded(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def seal_plan(value):
    result=deepcopy(value);result["sha256"]=_digest(value)
    return result

def _cpu_sdf(path):
    """Load the already verified, NumPy-only original sampler without importing ropesim."""
    name=__name__+"._original_sdf"
    if name in sys.modules:raise RuntimeError("nested original SDF sampler import")
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    try:
        sys.modules[name]=module;spec.loader.exec_module(module)
        return module
    finally:sys.modules.pop(name,None)

def verify_plan_sources(plan):
    from .tabletop_native_parameter_endpoints import source_basis,BUILD,CONFIG
    from .tabletop_dual_hand_scene_preparation import _verify,load_sdf_arrays,SDF_DIR,SDF_SHA
    from scripts.bake_dh116_deformx_pad_sdfs import checked_json
    from .tabletop_deformable_grasp_runtime import load_candidate,box_sdf
    from .tabletop_material import hand_material_contract
    if plan["source_scope"]!="locked_original_sdf_constructor_inputs" or source_basis()!=plan["source_basis"]:
        raise ValueError("locked actual backend source basis differs")
    for path,expected in plan["source_files"].items():
        p=Path(path)
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError("original reset source bytes differ: "+path)
    prepared=plan["prepared_scene"];project=Path(plan["project"])
    _verify(project,prepared)
    task_path=project/TASK_CONFIG
    if task_path.is_symlink() or hashlib.sha256(task_path.read_bytes()).hexdigest()!=TASK_SHA:
        raise ValueError("original task table source differs")
    scene=json.loads(task_path.read_text())["scene"]
    expected_table=dict(center_m=scene["table_center_m"],size_m=scene["table_size_m"],top_m=scene["table_top_m"])
    if prepared["table"]!=expected_table:
        raise ValueError("embedded table differs from locked task scene")
    summary=checked_json(SDF_DIR/"result.json",SDF_SHA)
    names=set(MESH_NAMES)-{"table"}
    if set(prepared["pads"])!=names or set(plan["bindings"])!=names:
        raise ValueError("original ten pad identities required")
    for name in names:
        side,body=name.split(".")
        row=prepared["pads"][name];source=summary["pads"][name]
        expected=dict(side=side,body_name=body,sdf_path=str(SDF_DIR/(name+".npz")),
            sdf_sha256=source["sdf_file_sha256"],capture_sha256=source["gpu_row_sha256"],
            frame="PhysX_link_origin",unit="m")
        if any(row.get(k)!=v or plan["bindings"][name].get(k)!=v for k,v in expected.items()):
            raise ValueError("pad identity differs from immutable SDF result: "+name)
        if any(plan["bindings"][name].get(k)!=v for k,v in row.items()):
            raise ValueError("embedded pad source differs from runtime plan binding")
    cfg=load_candidate(project/CONFIG,project)
    if cfg!=plan["cfg"]:raise ValueError("original candidate configuration differs")
    grids=load_sdf_arrays(prepared)
    sampler=_cpu_sdf(BUILD/"source/ropesim/sdf.py")
    table=cfg["table"];size=np.asarray(prepared["table"]["size_m"]);half=size/2
    values,origin,spacing=sampler.from_function(lambda p:box_sdf(p,size),
        -half-table["sdf_margin_m"],half+table["sdf_margin_m"],table["sdf_spacing_m"]).as_args()
    grids["table"]=dict(values=values,origin=origin,spacing=spacing)
    hand=hand_material_contract(cfg["hand"]["material_profile"])["hand_contact_material"]
    if tuple(row["name"] for row in plan["entries"])!=MESH_NAMES:
        raise ValueError("original eleven source slots differ")
    for row in plan["entries"]:
        name=row["name"]
        if name=="table":
            friction=table
            pose=(np.asarray(prepared["table"]["center_m"],np.float32),np.array([0,0,0,1],np.float32))
        else:
            friction=dict(mu_s=max(cfg["rod"]["mu_s"],hand["static_friction"]),
                mu_k=max(cfg["rod"]["mu_k"],hand["dynamic_friction"]))
            pose=None
        grid=grids[name]
        expected=dict(name=name,values=np.asarray(grid["values"],np.float32),
            origin=np.asarray(grid["origin"],np.float32),spacing=float(grid["spacing"]),
            mu_s=friction["mu_s"],mu_k=friction["mu_k"],table_pose=pose)
        if _digest(row)!=_digest(expected):raise ValueError("mesh entry differs from original grid/material/pose: "+name)

def verify_runtime_source(rod):
    from .tabletop_native_parameter_endpoints import BACKEND
    backend=sys.modules.get("ropesim._ropesim")
    try:
        package=sys.modules.get("ropesim")
        if (backend is None or getattr(package,"_ropesim",None) is not backend
                or Path(backend.__file__).resolve()!=BACKEND
                or type(rod.rope) is not backend.RopeBatch):
            raise ValueError("already loaded exact getter backend owner required")
    finally:rod=backend=package=None

def prepare_mesh_reset_plan(project,prepared,rod,*,sdf_module):
    """Own the exact ten locked grids plus the original analytic table grid."""
    try:
        from .tabletop_dual_hand_scene_preparation import _verify,load_sdf_arrays
        from .tabletop_native_parameter_endpoints import source_basis,BUILD,CONFIG
        from .tabletop_deformable_grasp_runtime import load_candidate,box_sdf
        from .tabletop_material import hand_material_contract
        project=Path(project).resolve()
        _verify(project,prepared);basis=source_basis();verify_runtime_source(rod)
        cfg=load_candidate(project/CONFIG,project)
        if cfg!=rod._base.cfg:raise ValueError("runtime candidate material/configuration differs")
        sdf_path=BUILD/"build/python/ropesim/sdf.py"
        source_path=BUILD/"source/ropesim/sdf.py"
        sdf_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
        if Path(sdf_module.__file__).resolve()!=sdf_path or hashlib.sha256(sdf_path.read_bytes()).hexdigest()!=sdf_sha:
            raise ValueError("original CPU analytic SDF implementation required")
        arrays=load_sdf_arrays(prepared)
        table=cfg["table"];size=np.asarray(prepared["table"]["size_m"]);half=size/2
        table_grid=sdf_module.from_function(lambda p:box_sdf(p,size),
            -half-table["sdf_margin_m"],half+table["sdf_margin_m"],table["sdf_spacing_m"])
        values,origin,spacing=table_grid.as_args()
        arrays["table"]=dict(values=values,origin=origin,spacing=spacing)
        hand=hand_material_contract(cfg["hand"]["material_profile"])["hand_contact_material"]
        entries=[];files={str(project/CONFIG):hashlib.sha256((project/CONFIG).read_bytes()).hexdigest(),
                        str(sdf_path):sdf_sha}
        for name in MESH_NAMES:
            grid=arrays[name]
            if name=="table":
                friction=table;pose=(np.asarray(prepared["table"]["center_m"],np.float32),
                                    np.array([0,0,0,1],np.float32))
            else:
                source=prepared["pads"][name];bound=rod.bindings[name]
                if any(bound.get(k)!=source[k] for k in source):
                    raise ValueError("original pad source binding differs")
                files[source["sdf_path"]]=source["sdf_sha256"]
                friction=dict(mu_s=max(cfg["rod"]["mu_s"],hand["static_friction"]),
                              mu_k=max(cfg["rod"]["mu_k"],hand["dynamic_friction"]))
                pose=None
            entries.append(dict(name=name,values=np.array(grid["values"],np.float32,copy=True),
                origin=np.array(grid["origin"],np.float32,copy=True),spacing=float(grid["spacing"]),
                mu_s=friction["mu_s"],mu_k=friction["mu_k"],table_pose=pose))
        return seal_plan(dict(version=PLAN_VERSION,entries=entries,bindings=deepcopy(rod.bindings),
            cfg=deepcopy(cfg),source_basis=basis,source_files=files,
            prepared_scene=deepcopy(prepared),project=str(project),
            source_scope="locked_original_sdf_constructor_inputs"))
    finally:rod=sdf_module=table_grid=None

def _plan(plan,rod):
    try:
        p=deepcopy(plan);claimed=p.pop("sha256",None)
        if claimed!=_digest(p) or p["version"]!=PLAN_VERSION:
            raise ValueError("owned mesh plan changed")
        if tuple(x["name"] for x in p["entries"])!=MESH_NAMES or tuple(rod.mesh_names)!=MESH_NAMES:
            raise ValueError("all eleven original mesh slots required")
        if _digest(p["bindings"])!=_digest(rod.bindings) or p["cfg"]!=rod._base.cfg:
            raise ValueError("runtime source identity/material changed")
        for i,row in enumerate(p["entries"]):
            a=row["values"];o=row["origin"]
            if (a.dtype!=np.float32 or a.ndim!=3 or min(a.shape)<2 or not np.isfinite(a).all()
                    or o.dtype!=np.float32 or o.shape!=(3,) or not np.isfinite(o).all()
                    or not math.isfinite(row["spacing"]) or row["spacing"]<=0
                    or any(not math.isfinite(row[k]) or row[k]<0 for k in ("mu_s","mu_k"))
                    or (i==5)!=(row["table_pose"] is not None)):
                raise ValueError("finite unchanged grid/material/pose arguments required")
            if row["table_pose"] is not None:
                pos,quat=map(np.asarray,row["table_pose"])
                if (pos.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(pos).all()
                        or not np.isfinite(quat).all() or abs(float(np.linalg.norm(quat))-1)>1e-6):
                    raise ValueError("finite unit table pose required before any mutation")
        verify_plan_sources(plan);verify_runtime_source(rod)
        return p
    finally:rod=None

def _state(rope):
    """parameter_snapshot synchronizes the exact native solver stream before clear."""
    try:
        params=deepcopy(rope.parameter_snapshot())
        error=rope.last_error
        if type(error) is not int or error!=0:raise ValueError("explicit zero actual solver error required")
        return dict(parameters=params,last_error=error,positions=_owned(rope.positions()),
                    velocities=_owned(rope.velocities()),quaternions=_owned(rope.quaternions()))
    finally:rope=None

class RebuiltMeshReset(DualHandResetTransaction):
    """One full original reset followed by same-slot reconstruction; reusable rod, new transaction."""
    def __init__(self,seed,*,episode_uid,mesh_plan):
        super().__init__(seed,episode_uid=episode_uid)
        self.mesh_plan=deepcopy(mesh_plan)

    def run(self,sim,rod,bindings,*,owner_resets):
        receipt=rope=before_state=after_state=None
        try:
            if self.status!="new":raise RuntimeError("reset transaction cannot be reused")
            plan=_plan(self.mesh_plan,rod)  # Fail before touching either hand.
            if original_reset._clock is not _ORIGINAL_CLOCK:
                raise ValueError("unexpected original reset clock binding")
            with scoped_bindings((original_reset,"_clock",_clock)):
                receipt=super().run(sim,rod,bindings,owner_resets=owner_resets)
            before=_clock(sim,rod)
            if before["rod"]["rod_tick"]!=0:raise ValueError("mesh rebuild is reset-only")
            rope=rod.rope
            before_state=_state(rope)
            _no_step(before,_clock(sim,rod))
            # Pair buffers/capacity survive clear_meshes in the pinned source;
            # any borrowed mesh/grid view must be considered invalid after this point.
            rod._needs_reset=True
            rope.clear_meshes()
            for index,row in enumerate(plan["entries"]):
                pos,quat=(row["table_pose"] if row["name"]=="table"
                          else receipt["pad_poses_xyzw"][row["name"]])
                slot=rope.add_mesh(row["values"],row["origin"],row["spacing"],
                    pos=np.asarray(pos,np.float32),quat=np.asarray(quat,np.float32),
                    mu_s=row["mu_s"],mu_k=row["mu_k"])
                if type(slot) is not int or slot!=index:raise ValueError("reconstructed collider slot identity differs")
            after_state=_state(rope)
            mutation_after=_clock(sim,rod)
            if rod._needs_reset is not True or mutation_after["rod"]["clock_valid"] is not False:
                raise ValueError("mutation region lost its poisoned clock state")
            # Only this expected local validity transition is normalized for comparison.
            # Keep the actual false state in the returned receipt.
            after=deepcopy(mutation_after)
            after["rod"]["clock_valid"]=before["rod"]["clock_valid"]
            _no_step(before,after)
            if after!=before:raise ValueError("full reset identity changed during mesh reconstruction")
            if _digest(before_state)!=_digest(after_state):
                raise ValueError("mesh-only reset changed rod nodes/velocity/directors/parameters")
            if np.asarray(rope.mesh_wrenches()).shape!=(1,11,6):
                raise ValueError("reconstructed actual eleven-mesh layout differs")
            # Scratch wrench values are not valid reset observations. Original reset
            # already zeroed both hand composers and clears exchange reactions.
            if _digest(plan["bindings"])!=_digest(rod.bindings) or plan["cfg"]!=rod._base.cfg:
                raise ValueError("reset source binding changed during reconstruction")
            rod._needs_reset=False
            receipt["mesh_reset"]=dict(version=VERSION,mesh_plan_sha256=self.mesh_plan["sha256"],
                before_clock=before,mutation_after_clock=mutation_after,
                after_clock=_clock(sim,rod),mesh_names=list(MESH_NAMES),
                native_state_before=before_state,native_state_after=after_state,
                source_tree_sha256=plan["source_basis"]["source_tree_sha256"],
                sdk_clock_scope="at_most_240_steps_float32_callback_envelope",
                native_current_pose_readback=None,source_derived_commit=True,
                pair_capacity_readback=None,pair_buffer_policy="preserved_by_locked_clear_add_source",
                contact_readback_valid=False,borrowed_mesh_views_invalidated=True,
                new_sdk_steps=0,new_solver_steps=0,native_qualification=False,teacher_admission=False)
            return receipt
        except Exception:
            if self.status!="new":
                self.status="failed";rod._needs_reset=True;rod._base.last=None
            raise
        finally:sim=rod=bindings=owner_resets=receipt=rope=before_state=after_state=None
