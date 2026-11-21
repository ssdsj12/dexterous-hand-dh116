"""Fresh supported Tripod scene inside an existing App and ownership scope.

Build all seventeen original collision slots before the caller's first Reset.
No App/SDK owner, settlement, controller motion or Teacher admission is created.
"""
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

VERSION = "dh116-fresh-supported-tripod-builder-v1"
SOURCE_KEYS = {"rays", "left_bake", "right_bake", "proximal_capture", "proximal_bake"}


@dataclass(frozen=True)
class FreshSupportedEpisode:
    episode: object
    setup_events: tuple
    initial_geometry: object
    reset_seed: dict
    reset_seed_sha256: str
    mesh_plan: dict
    task_config_sha256: str


def _contact_poses(sim, bindings, records):
    """Copy real XYZW link buffers at SDK2, independently of frozen SDF poses."""
    import numpy as np
    from .tabletop_deformable_bimanual_step import _owned
    from .tabletop_dual_pinch_builder import _clock

    binding = view = None
    try:
        before = _clock(sim)
        result = {}
        for side, binding in bindings.items():
            view = binding.hand.root_physx_view
            values = _owned(view.get_link_transforms())
            if (values.shape != (1, 23, 7) or not np.isfinite(values).all()
                    or not np.allclose(np.linalg.norm(values[..., 3:], axis=-1), 1., atol=1e-5, rtol=0)):
                raise ValueError("finite current original23 XYZW link buffers required")
            for name, row in records.items():
                if row["side"] != side:
                    continue
                index = row["body_index"]
                if (binding.hand.body_names[index] != row["body_name"]
                        or view.link_paths[0][index] != row["prim_path"]):
                    raise ValueError("same-side current contact link identity differs")
                result[name] = (values[0,index,:3].astype(np.float32, copy=True),
                                values[0,index,3:].astype(np.float32, copy=True))
        if set(result) != set(records) or before != _clock(sim):
            raise ValueError("all fourteen source links must share one actual SDK boundary")
        return result
    finally:
        sim = bindings = binding = view = None


def _initial_geometry(project, prepared, sim, hands, episode, radius_m, support_scene):
    from .tabletop_dual_pinch_builder import _fresh_initial_geometry
    from .tabletop_power_supported_task import guard_hand_supports

    try:
        rows = list(_fresh_initial_geometry(project, prepared, sim, hands, episode, radius_m))
        hulls = {f"{row['side']}.{group}.{name}": item["world_vertices"]
                 for row in rows for group in ("source_colliders", "locked_pads")
                 for name, item in row["evidence"][group].items()}
        if len(hulls) != 44:
            raise ValueError("all34 original source and10 cooked envelopes required")
        clearance = guard_hand_supports(hulls, support_scene)
        for row in rows:
            yield dict(row, support_clearance=deepcopy(clearance))
    finally:
        sim = hands = episode = None


def build_fresh_supported_owned(sim, owners, *, project, prepared, ownership, record,
                                backend_profile, task_config, task_config_sha256,
                                extra_sources, run_uuid, episode_uid, physics_budget_end):
    """Return the new original episode; the caller exclusively performs Reset.

    Requires the exact current backend profile. All side-specific source USD/SDF
    producers and the authenticated17 Reset plan run unchanged. The old11-slot
    diagnostic path and all of its receipts remain independent.
    """
    scene = hands = hand = base = rod = episode = binding = value = None
    mount_bindings = hand_sources = poses = grids = source_view = members = sdf = backend = None
    try:
        from .tabletop_dual_pinch_builder import _clock, _initial_pads, _Composers, _inspect_geometry
        from .tabletop_native_teacher_profile import BackendProfile, checked_file, profile_reader_scope
        if sim is None or _clock(sim) != dict(sdk_step=0, sdk_time_s=0.):
            raise ValueError("fresh existing App-owned simulation required")
        if (not callable(record) or type(backend_profile) is not BackendProfile
                or not isinstance(run_uuid,str) or not run_uuid
                or not isinstance(episode_uid,str) or not episode_uid or episode_uid == run_uuid+":construction"
                or type(physics_budget_end) is not int or physics_budget_end < 10):
            raise ValueError("actual backend, explicit episode identity, recorder and bounded interval required")
        backend_profile.verify()
        path = Path(task_config)
        raw = checked_file(path, task_config_sha256)
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise ValueError("canonical frozen supported task path required")
        config = json.loads(raw)
        from . import tabletop_dual_hand_scene_preparation as scene_api
        from .tabletop_power_supported_task import prepare_initial_state, TRIPOD_SCOPE, assemble_supported_scene
        from .tabletop_deformable_grasp_runtime import FreeRodRuntime, load_candidate
        from .tabletop_dual_hand_rod_runtime import PADS
        from .tabletop_power_shared_rod import (
            PowerSharedRodRuntime, PROXIMAL_POWER_MESH_NAMES, PALM_NAMES, PROXIMAL_NAMES, END_SUPPORT_NAMES)
        from .tabletop_deformx_end_supported_scene import make_support_sdf_grids
        from .tabletop_deformable_bimanual_step import HandBinding
        from .tabletop_palm_mount import MountView
        from .tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
        from .tabletop_supported_mesh_reset import prepare_supported_mesh_reset_plan, CAPACITY
        from scripts import verify_dh116_power_shared_rod as sources
        from ropesim import _ropesim as backend, sdf
        project = Path(project).resolve()
        if (Path(backend.__file__).resolve() != Path(backend_profile.binary_path)
                or not isinstance(extra_sources,dict) or set(extra_sources) != SOURCE_KEYS):
            raise ValueError("loaded backend identity and complete independent source paths required")
        checked_file(backend_profile.binary_path, backend_profile.binary_sha256)
        prepared = scene_api.prepare_scene(project) if prepared is None else prepared
        scene_api._verify(project,prepared)
        seed, active, nodes, support_poses = prepare_initial_state(
            config, prepared["table"], expected_scope=TRIPOD_SCOPE)
        cfg = load_candidate(project/"configs/dh116_tasks/flexible_pinch_candidate_v1.json",project)
        if (cfg["rod"]["gravity_world_m_s2"] != config["gravity_w_m_s2"]
                or cfg["rod"]["radius_m"] != config["support_scene"]["wire_radius_m"]):
            raise ValueError("unchanged task rod gravity and radius required")
        source_args = SimpleNamespace(**{k:Path(v) for k,v in extra_sources.items()})
        scene, _, _ = assemble_supported_scene(sim,project,prepared,seed,active,config["support_scene"])
        owners.keep("scene",scene)
        hands = scene["hands"]
        if scene["wire"] is not None or scene["sensors"]:
            raise ValueError("one external free rod and no duplicate rigid wire required")
        before = dict(kind="setup",phase="before_reset",**_clock(sim))
        record(deepcopy(before))
        sim.reset()
        after = dict(kind="setup",phase="after_reset",**_clock(sim))
        record(deepcopy(after))
        if after["sdk_step"] != 2:
            raise ValueError("SDK setup must advance exactly two measured steps")
        scene_api.validate_simulation_settings(sim,prepared)
        members = [getattr(hand,key) for hand in hands
                   for key in ("permanent_wrench_composer","instantaneous_wrench_composer")]
        owners.keep("composer",_Composers(members))
        members = None
        for side, hand in zip(("left","right"),hands,strict=True):
            hand.update(1/480)
            for suffix,value in (("hand",hand),("view",hand.root_physx_view),
                    ("asset_sim_view",hand._physics_sim_view),
                    ("permanent_composer",hand.permanent_wrench_composer),
                    ("instantaneous_composer",hand.instantaneous_wrench_composer)):
                ownership.watch(side+"_"+suffix,value,owner=VERSION,release="owned scope then SDK STOP",callbacks=True)
            value = None
        distal, _, paths = _initial_pads(prepared,hands)
        mount_bindings = owners.keep("hand_bindings",tuple(
            HandBinding(side,hand,{prepared["env_id"]:paths[side]},MountView(hand))
            for side,hand in zip(("left","right"),hands,strict=True)))
        hand_sources = {binding.side:binding for binding in mount_bindings}
        # Producers require source bindings, never an already-reset11-slot owner.
        source_view = SimpleNamespace(bindings=distal)
        palms, palm_grids = sources.bind_palm_sources(sim,hand_sources,source_view,args=source_args)
        proximal, proximal_grids = sources.bind_proximal_sources(sim,hand_sources,source_view,args=source_args)
        records = {**distal,**palms,**proximal}
        poses = _contact_poses(sim,hand_sources,records)
        grids = {name:sdf.SdfGrid(**args) for name,args in scene_api.load_sdf_arrays(prepared).items()}
        base = owners.keep("base_runtime",FreeRodRuntime(cfg,nodes,
            {name:grids["left."+name] for name in PADS},{name:poses["left."+name] for name in PADS},
            prepared["table"]["center_m"],prepared["table"]["size_m"],episode_uid=run_uuid+":construction"))
        rod = owners.keep("runtime",PowerSharedRodRuntime(base,bindings=records,
            right_grids={name:grids["right."+name] for name in PADS},
            right_poses={name:poses["right."+name] for name in PADS},run_uuid=run_uuid,
            env_id=prepared["env_id"],require_pairs=True,
            palm_grids=palm_grids,palm_poses={name:poses[name] for name in PALM_NAMES},
            support_grids=dict(zip(END_SUPPORT_NAMES,make_support_sdf_grids(config["support_scene"]),strict=True)),
            support_poses=support_poses,proximal_grids=proximal_grids,
            proximal_poses={name:poses[name] for name in PROXIMAL_NAMES}))
        if tuple(rod.mesh_names) != PROXIMAL_POWER_MESH_NAMES:
            raise ValueError("direct original17 composition required before pair allocation")
        rod.rope.enable_contact_pair_capture(CAPACITY)
        ownership.watch("runtime",rod,owner=VERSION,release="owned scope")
        ownership.watch("rope",rod.rope,owner=VERSION,release="owned scope")
        with profile_reader_scope(backend_profile):
            mesh_plan = prepare_supported_mesh_reset_plan(project,prepared,rod,sdf_module=sdf,
                task_config=path,extra_sources=extra_sources)
        episode = owners.keep("dual_episode",DualHandPhysicalEpisode(sim,rod,mount_bindings,
            backend_artifact=Path(backend_profile.binary_path),backend_sha256=backend_profile.binary_sha256,
            inspect_geometry=partial(_inspect_geometry,project,prepared,sim,hands,rod),
            physics_budget_end=physics_budget_end,record=record))
        ownership.watch("episode",episode,owner=VERSION,release="owned scope")
        if (_clock(sim) != {k:after[k] for k in ("sdk_step","sdk_time_s")}
                or base.physics_step is not None or base.reset_id != 0 or base.rod_tick != 0
                or base.last is not None or episode._state != "new"):
            raise ValueError("fresh construction must not perform episode Reset or settlement")
        return FreshSupportedEpisode(episode,(before,after),
            partial(_initial_geometry,project,prepared,sim,hands,episode,cfg["rod"]["radius_m"],
                    deepcopy(config["support_scene"])),deepcopy(seed),seed["sha256"],mesh_plan,task_config_sha256)
    finally:
        sim = owners = prepared = ownership = record = scene = hands = hand = base = rod = episode = None
        mount_bindings = hand_sources = binding = poses = grids = source_view = members = value = sdf = backend = None
