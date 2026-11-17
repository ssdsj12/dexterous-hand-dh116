"""Bounded same-rod two-hand Pinch setup/reset, owned by an external app scope.

Import and paired byte validation are safe before AppLauncher. All SDK/geometry
imports occur only in build_reset_owned, which requires an existing simulation.
No launcher, lock, recipe, measured physics loop or Teacher qualification.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

VERSION="dh116-dual-pinch-builder-v1"
PAIR=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/paired-pinch-reset-BaP0hx/result.json")
PAIR_SHA="996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37"
SEED_SHA="647247fe31219f04337b6f6e29ddefaa2c81bcee27dbed8dd3d6f962e406230d"
BACKEND=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so")
BACKEND_SHA="92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0"
CLEARANCE_M=.0021
DT=1/480


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pair_proposal(path=PAIR):
    """Exact private CPU evidence; no USD, SDK, solver or geometry imports."""
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PAIR_SHA:
        raise ValueError("paired evidence bytes differ")
    packet=json.loads(raw)
    seed=deepcopy(packet["reset_seed"]);digest=seed.pop("sha256")
    actual=hashlib.sha256(json.dumps(seed,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if (digest!=SEED_SHA or actual!=digest
            or packet["wrist_clearance_floor_m"]!=CLEARANCE_M
            or packet["native_qualified"] is not False
            or packet["teacher_admission"] is not False):
        raise ValueError("paired reset identity/scope differs")
    return packet


def _clock(sim):
    try:
        step=sim.current_time_step_index
        time=float(sim.current_time)
        if (type(step) is not int or step<0 or not math.isfinite(time)
                or not math.isclose(time,step*DT,abs_tol=2e-7,rel_tol=0)
                or float(sim.get_physics_dt())!=DT):
            raise ValueError("actual 480Hz SDK clock required")
        return dict(sdk_step=step,sdk_time_s=time)
    finally:
        sim=None


def _paired_sources(project,packet,prepared):
    if _sha(project/"configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json")!=packet["left_pose_sha256"]:
        raise ValueError("left proposal source changed")
    aliases=dict(pose_reference="configs/dh116_training/pose_reference.json",
                 pad_evidence_lock="docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json",
                 old_candidate="configs/dh116_tasks/flexible_pinch_candidate_v1.json")
    for side in ("left","right"):
        sources=(packet["left"]["candidate"]["source_sha256"] if side=="left"
                 else packet["right"]["source_sha256"])
        for name,expected in sources.items():
            path=(project/prepared["hands"][side]["usd_path"] if name=="usd"
                  else project/aliases.get(name,name))
            if _sha(path)!=expected:
                raise ValueError("paired hand source identity changed: "+side+"/"+name)


def _initial_pads(prepared,hands):
    """Bootstrap from actual link buffers; no fabricated pre-rod clock."""
    import numpy as np
    from .tabletop_deformable_bimanual_step import _owned
    view=hand=None
    try:
        if len(hands)!=2 or hands[0] is hands[1] or hands[0].root_physx_view is hands[1].root_physx_view:
            raise ValueError("two distinct actual hand views required")
        bindings,poses,paths={},{},{}
        for side,hand in zip(("left","right"),hands):
            view=hand.root_physx_view
            root=prepared["prefix"]+"/"+side+"Hand"
            names=list(hand.body_names)
            actual=list(view.prim_paths)
            if (len(names)!=23 or len(set(names))!=23
                    or actual not in ([root],[root+"/wrist_fixed"])
                    or view.link_paths!=[[root+"/"+n for n in names]]
                    or not set(prepared["hands"][side]["source_colliders"])<=set(names)):
                raise ValueError("actual original mounted link identity differs")
            values=_owned(view.get_link_transforms())
            if (values.shape!=(1,23,7) or not np.isfinite(values).all()
                    or not np.allclose(np.linalg.norm(values[...,3:],axis=-1),1,rtol=0,atol=1e-5)):
                raise ValueError("finite actual native XYZW link poses required")
            paths[side]=actual[0]
            for key,pad in prepared["pads"].items():
                if not key.startswith(side+"."):continue
                name=key.split(".",1)[1];index=names.index(name)
                bindings[key]=dict(pad,body_index=index,prim_path=view.link_paths[0][index])
                poses[key]=(values[0,index,:3].astype(np.float32),values[0,index,3:].astype(np.float32))
        return bindings,poses,paths
    finally:
        prepared=hands=view=hand=None


class _Composers:
    """Existing run_owned_call resets this group before its CUDA synchronize."""
    def __init__(self,members):
        try:self.members=tuple(members)
        finally:self=members=None

    def reset(self):
        import traceback
        member=None
        failures=[]
        try:
            for member in self.members:
                try:member.reset()
                except BaseException as error:
                    failures.append(traceback.format_exc())
                    traceback.clear_frames(error.__traceback__)
                    error.__traceback__=None
            if failures:raise RuntimeError(chr(10).join(failures)) from None
        finally:
            self=member=None


def build_reset_owned(sim,owners,*,project,prepared,ownership,record,owner_resets,
                      run_uuid,episode_uid,physics_budget_end):
    """Borrowed native handles are registered in caller's NativeOwners only.

    This function performs exactly one counted SDK reset and one same-rod
    episode reset. The caller must use run_owned_call before STOP/App close.
    The stored episode may be used only inside that owner's separately
    authorized control scope; this helper never calls episode.control.
    """
    scene=hands=hand=view=base=rod=episode=bindings=mount=composers=None
    inspect=read_clock=backend=grids=members=value=geometry=None
    try:
        # Reject a missing/nonfresh context before any lazy SDK/geometry import.
        if sim is None or _clock(sim)!={"sdk_step":0,"sdk_time_s":0.}:
            raise ValueError("fresh existing App-owned simulation required")
        if (not callable(record) or not isinstance(run_uuid,str) or not run_uuid
                or not isinstance(episode_uid,str) or not episode_uid
                or episode_uid==run_uuid+":construction"
                or type(physics_budget_end) is not int or physics_budget_end<10):
            raise ValueError("explicit identity, recorder and future interval budget required")
        packet=load_pair_proposal()
        project=Path(project).resolve()
        from . import tabletop_dual_hand_scene_preparation as scene_api
        from .tabletop_deformable_grasp_runtime import FreeRodRuntime,load_candidate
        from .tabletop_dual_hand_rod_runtime import DualHandRodRuntime,PADS
        from .tabletop_deformable_bimanual_step import HandBinding
        from .tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
        from .tabletop_palm_mount import MountView
        from .tabletop_deformable_wrist_projection import CLEARANCE_M as POLICY_FLOOR
        from ropesim import _ropesim as backend,sdf
        import numpy as np
        if Path(backend.__file__).resolve()!=BACKEND.resolve() or _sha(BACKEND)!=BACKEND_SHA:
            raise ValueError("loaded qualified pair backend identity differs")
        if POLICY_FLOOR!=CLEARANCE_M:
            raise ValueError("original policy floor changed")
        prepared=scene_api.prepare_scene(project) if prepared is None else prepared
        scene_api._verify(project,prepared)
        _paired_sources(project,packet,prepared)
        cfg=load_candidate(project/"configs/dh116_tasks/flexible_pinch_candidate_v1.json",project)
        nodes=np.asarray(packet["rod_seed"]["proposed_nodes_m"],dtype=np.float32)
        if (nodes.shape!=(41,3) or packet["rod_seed"]["radius_m"]!=cfg["rod"]["radius_m"]
                or packet["rod_seed"]["gravity_world_m_s2"]!=cfg["rod"]["gravity_world_m_s2"]
                or packet["rod_seed"]["pinned_node_ids"]!=[]
                or packet["rod_seed"]["fix_root_orientation"] is not False):
            raise ValueError("one unchanged free supported rod proposal required")
        scene=owners.keep("scene",scene_api.assemble_prepared_scene(sim,project,prepared))
        hands=scene["hands"]
        if scene["wire"] is not None or scene["sensors"]:
            raise ValueError("external single-rod scene required")
        record(dict(kind="setup",phase="before_reset",**_clock(sim)))
        sim.reset()
        after=_clock(sim)
        record(dict(kind="setup",phase="after_reset",**after))
        if after["sdk_step"]!=2:raise ValueError("SDK setup must advance exactly two observed steps")
        settings=scene_api.validate_simulation_settings(sim,prepared)
        members=[]
        for hand in hands:
            members.append(hand.permanent_wrench_composer)
            members.append(hand.instantaneous_wrench_composer)
        composers=owners.keep("composer",_Composers(members))
        members=None
        for side,hand in zip(("left","right"),hands):
            hand.update(DT)
            for suffix,value in (("hand",hand),("view",hand.root_physx_view),
                                 ("asset_sim_view",hand._physics_sim_view),
                                 ("permanent_composer",hand.permanent_wrench_composer),
                                 ("instantaneous_composer",hand.instantaneous_wrench_composer)):
                ownership.watch(side+"_"+suffix,value,owner="dual builder",
                                release="owned scope then SDK STOP",callbacks=True)
            value=None
        initial,poses,paths=_initial_pads(prepared,hands)
        if _clock(sim)!=after:raise ValueError("SDK clock changed during initial mapping")
        arrays=scene_api.load_sdf_arrays(prepared)
        grids={key:sdf.SdfGrid(**args) for key,args in arrays.items()}
        base=owners.keep("base_runtime",FreeRodRuntime(cfg,nodes,
            {n:grids["left."+n] for n in PADS},{n:poses["left."+n] for n in PADS},
            prepared["table"]["center_m"],prepared["table"]["size_m"],
            episode_uid=run_uuid+":construction"))
        rod=owners.keep("runtime",DualHandRodRuntime(base,bindings=initial,
            right_grids={n:grids["right."+n] for n in PADS},
            right_poses={n:poses["right."+n] for n in PADS},run_uuid=run_uuid,env_id=prepared["env_id"],
            rod_uid="rod0",require_pairs=True))
        rod.rope.enable_contact_pair_capture(41*11*12)
        ownership.watch("runtime",rod,owner="dual builder",release="owned scope")
        ownership.watch("rope",rod.rope,owner="dual runtime",release="owned scope")
        bindings=[]
        for side,hand in zip(("left","right"),hands):
            bindings.append(HandBinding(side,hand,{prepared["env_id"]:paths[side]},MountView(hand)))
        bindings=owners.keep("hand_bindings",tuple(bindings))
        def read_clock():
            return [dict(rod.clock(),policy_step=0,substep=0)]
        def inspect():
            return scene_api.bind_actual_geometry(project,prepared,sim.stage,hands,read_clock=read_clock)
        episode=owners.keep("dual_episode",DualHandPhysicalEpisode(sim,rod,bindings,
            backend_artifact=BACKEND,backend_sha256=BACKEND_SHA,inspect_geometry=inspect,
            physics_budget_end=physics_budget_end,record=record))
        ownership.watch("episode",episode,owner="dual builder",release="owned scope")
        reset=episode.reset(packet["reset_seed"],episode_uid=episode_uid,owner_resets=owner_resets)
        from scripts.audit_dh116_pinch_initial_pose import load_geometry
        from .tabletop_pinch_initial_geometry_v2 import capture_pinch_initial_geometry
        locked_clearance=[]
        for side,hand in zip(("left","right"),hands):
            geometry=load_geometry(project,side=side)
            captured=capture_pinch_initial_geometry(sim.stage,hand,geometry,
                cooking_interface=None,read_clock=lambda:_clock(sim),
                table_center_m=prepared["table"]["center_m"],table_size_m=prepared["table"]["size_m"],
                rod_nodes_w=nodes,rod_radius_m=cfg["rod"]["radius_m"],
                minimum_table_gap_m=CLEARANCE_M,minimum_rod_gap_m=0.)
            record(dict(kind="initial_geometry",side=side,evidence=captured))
            if captured["initial_geometry_passed"] is not True:
                raise ValueError("paired source17/locked5 geometry below unchanged floor or rod gap")
            locked_clearance.append(min(row["table"]["top_clearance_m"] for row in captured["locked_pads"].values()))
        if _clock(sim)!=after or rod.clock()["rod_tick"]!=0:
            raise ValueError("preparation must not advance a measured SDK or solver step")
        ownership.checkpoint("dual_reset_complete")
        return dict(version=VERSION,paired_evidence_sha256=PAIR_SHA,reset_seed_sha256=SEED_SHA,
            prepared_scene_sha256=prepared["fingerprint_sha256"],backend_sha256=BACKEND_SHA,
            setup_steps=2,measured_steps=0,solver_steps=0,ppo_updates=0,clock=rod.clock(),
            clearance_floor_m=CLEARANCE_M,clearance_m=reset["clearance_m"].tolist(),
            mesh_names=list(rod.mesh_names),actual_physics_settings=settings,
            locked_pad_table_clearance_m=locked_clearance,
            source_qualification=False,native_qualification=False,grasp_qualified=False,
            teacher_admission=False,task_unknown_rod_fields=packet["rod_seed"]["unknown_task_rod_fields"],
            native_table_support_verified=False)
    finally:
        sim=owners=scene=hands=hand=view=base=rod=episode=bindings=mount=composers=None
        inspect=read_clock=backend=grids=prepared=ownership=record=owner_resets=members=value=geometry=None


def run_preparation(sim,*,project,ownership,synchronize,record,owner_resets,
                    run_uuid,episode_uid,physics_budget_end,prepared=None):
    """Bounded preparation-only adapter; caller retains STOP/App close."""
    from .tabletop_deformable_grasp_runtime import run_owned_call
    operation=None
    try:
        def operation(owners):
            return build_reset_owned(sim,owners,project=project,prepared=prepared,
                ownership=ownership,record=record,owner_resets=owner_resets,run_uuid=run_uuid,
                episode_uid=episode_uid,physics_budget_end=physics_budget_end)
        return run_owned_call(operation,synchronize=synchronize,ownership=ownership)
    finally:
        sim=ownership=synchronize=record=owner_resets=operation=prepared=None
