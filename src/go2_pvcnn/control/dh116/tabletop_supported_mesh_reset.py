"""Authenticated seventeen-slot Reset over the existing free-rod owner.

Share the original zero-step clear/add algorithm. Read every current/target/
solver slot back directly; no stationary first-step substitution or hidden step.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import tabletop_mesh_reset_order as original
from . import tabletop_mesh_reset_readback as readback
from . import tabletop_mesh_observation_source as backend
from .tabletop_deformable_r0_replay import decode_owned
from .tabletop_dual_hand_reset import _no_step, DualHandResetTransaction, VERSION as RESET_TRANSACTION_VERSION
from .tabletop_power_shared_rod import (
    PROXIMAL_POWER_MESH_NAMES, PALM_NAMES, PROXIMAL_NAMES, END_SUPPORT_NAMES,
    HAND_NAMES, SIDES, _bindings, contact_links,
)

PLAN_VERSION = "dh116-supported-seventeen-mesh-reset-plan-v1"
VERSION = "dh116-direct-seventeen-mesh-reset-readback-v1"
REBUILD_VERSION = "dh116-zero-step-seventeen-mesh-rebuild-v1"
CAPACITY = 41*17*12
source_basis = backend.source_basis
verify_runtime_source = backend.verify_runtime


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), "regular source file required: "+str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extra_grids(plan):
    """Recheck the immutable native palm/proximal captures and exact grid bytes."""
    from scripts import verify_dh116_power_shared_rod as shared
    inputs = {k: Path(v) for k, v in plan["extra_sources"].items()}
    _require(set(inputs) == {"rays", "left_bake", "right_bake", "proximal_capture", "proximal_bake"},
             "complete independent palm/proximal source paths required")
    bindings = plan["bindings"]
    rays = shared.sources.checked_json(inputs["rays"], shared.sources.RAYS_SHA)["palms"]
    grids = {}
    for side in SIDES:
        name = side+".base_link"
        bake = inputs[side+"_bake"]
        grid, _, frozen, _ = shared.sources.fixture(side, rays[side], bake)
        archive = shared.sources.checked_json(inputs["rays"].parent/(side+"_base_link.json"),
                                              frozen["gpu_row_sha256"])
        expected = dict(source_geometry_sha256=archive["source"]["geometry_sha256"],
            cooked_geometry_sha256=shared.archive_geometry_digest(archive),
            capture_sha256=frozen["gpu_row_sha256"], sdf_sha256=frozen["sdf_file_sha256"],
            sdf_path=str(bake/(name+".npz")))
        _require(all(bindings[name].get(k) == v for k,v in expected.items()),
                 "palm source binding differs from immutable native/grid sources")
        grids[name] = dict(values=grid[0], origin=grid[1], spacing=grid[2])
    packets, _ = shared.proximal_registration_inputs(inputs["proximal_capture"], inputs["proximal_bake"])
    for name, (archive, grid, frozen) in packets.items():
        expected = dict(source_geometry_sha256=archive["source"]["geometry_sha256"],
            cooked_geometry_sha256=shared.archive_geometry_digest(archive),
            capture_sha256=frozen["native_row_sha256"], sdf_sha256=frozen["grid_sha256"],
            sdf_path=str(inputs["proximal_bake"]/(name+".npz")))
        _require(all(bindings[name].get(k) == v for k,v in expected.items()),
                 "proximal source binding differs from immutable native/grid sources")
        grids[name] = grid
    _require(set(grids) == set(PALM_NAMES+PROXIMAL_NAMES), "all four independent extra SDFs required")
    return grids


def _expected_entries(plan):
    """Reconstruct only constructor arguments, never a native pose/state snapshot."""
    from .tabletop_power_supported_task import prepare_initial_state, POWER_SCOPE, TRIPOD_SCOPE
    from .tabletop_deformable_grasp_runtime import box_sdf
    from .tabletop_material import hand_material_contract
    from .tabletop_native_parameter_endpoints import BUILD

    baseline = plan["original_plan"]
    original.verify_plan_sources(baseline)
    _require(plan["source_basis"] == baseline["source_basis"], "one actual backend source basis required")
    checked = _bindings(plan["bindings"])
    _require(contact_links(checked)[-2:] == ("base_link", "finger52_link"),
             "complete seventeen-slot hand layout required")
    _require(plan["cfg"] == baseline["cfg"], "unchanged original rod/material configuration required")
    for name in HAND_NAMES:
        _require(checked[name] == baseline["bindings"][name], "original distal source binding differs")
    for name in PALM_NAMES+PROXIMAL_NAMES:
        side = checked[name]["side"]
        _require(checked[name].get("frame") == "PhysX_link_origin" and checked[name].get("unit") == "m"
                 and checked[name]["source_usd_sha256"] ==
                     baseline["bindings"][side+".finger14_link"]["source_usd_sha256"],
                 "original side-specific extra source frame/unit/USD required")
    task_path = Path(plan["task_config"]["path"])
    _require(_sha(task_path) == plan["task_config"]["sha256"], "frozen supported task bytes differ")
    task = json.loads(task_path.read_bytes())
    scope = task["scope"]
    _require(scope in (TRIPOD_SCOPE, POWER_SCOPE), "explicit supported Tripod or Power task required")
    _, _, nodes, support_poses = prepare_initial_state(
        task, baseline["prepared_scene"]["table"], expected_scope=scope)
    _require(task["gravity_w_m_s2"] == plan["cfg"]["rod"]["gravity_world_m_s2"]
             and task["support_scene"]["wire_radius_m"] == plan["cfg"]["rod"]["radius_m"],
             "task gravity/radius differs from original physical parameters")
    extra = _extra_grids(plan)
    sampler_path = BUILD/"build/python/ropesim/sdf.py"
    _require(_sha(sampler_path) == baseline["source_files"][str(sampler_path)],
             "original analytic SDF sampler differs")
    sampler = original._cpu_sdf(sampler_path)
    table = plan["cfg"]["table"]
    for name, box in zip(END_SUPPORT_NAMES, task["support_scene"]["supports"], strict=True):
        size = np.asarray(box["size_m"])
        half = size/2
        values, origin, spacing = sampler.from_function(
            lambda p, size=size: box_sdf(p, size),
            -half-table["sdf_margin_m"], half+table["sdf_margin_m"], table["sdf_spacing_m"]).as_args()
        extra[name] = dict(values=values, origin=origin, spacing=spacing)
    hand = hand_material_contract(plan["cfg"]["hand"]["material_profile"])["hand_contact_material"]
    friction = dict(mu_s=max(plan["cfg"]["rod"]["mu_s"], hand["static_friction"]),
                    mu_k=max(plan["cfg"]["rod"]["mu_k"], hand["dynamic_friction"]))
    original_entries = {row["name"]: row for row in baseline["entries"]}
    entries = []
    for name in PROXIMAL_POWER_MESH_NAMES:
        if name in original_entries:
            row = deepcopy(original_entries[name])
            row["static_pose"] = row.pop("table_pose")
        else:
            grid = extra[name]
            mu = table if name in END_SUPPORT_NAMES else friction
            pose = support_poses[name] if name in END_SUPPORT_NAMES else None
            row = dict(name=name, values=np.asarray(grid["values"], np.float32),
                origin=np.asarray(grid["origin"], np.float32), spacing=float(grid["spacing"]),
                mu_s=mu["mu_s"], mu_k=mu["mu_k"],
                static_pose=None if pose is None else tuple(np.asarray(v, np.float32) for v in pose))
        entries.append(row)
    return entries


def _implementation_sources(project):
    project = Path(project)
    files = (Path(__file__), Path(original.__file__), Path(readback.__file__),
             project/"scripts/verify_dh116_power_shared_rod.py",
             project/"scripts/verify_dh116_power_palm_reactions.py",
             project/"scripts/audit_dh116_power_palm_parity.py",
             project/"docs/contracts/dh116_teacher_power_proximal_sdfs_v1.json",
             project/"src/go2_pvcnn/control/dh116/tabletop_deformx_end_supported_scene.py",
             project/"src/go2_pvcnn/control/dh116/tabletop_power_supported_task.py")
    return {str(path): _sha(path) for path in files}


def verify_plan_sources(plan):
    _require(plan["version"] == PLAN_VERSION, "supported constructor plan version differs")
    _require(plan["source_scope"] == "locked_original_seventeen_constructor_inputs",
             "explicit authenticated seventeen-slot constructor sources required")
    _require(plan["source_basis"] == source_basis(), "current supported backend source basis differs")
    expected = _expected_entries(plan)
    _require(original._digest(plan["entries"]) == original._digest(expected),
             "seventeen constructor grids/materials/static poses differ")
    _require(plan["implementation_sources"] == _implementation_sources(plan["original_plan"]["project"]),
             "complete supported constructor implementation sources differ")


def prepare_supported_mesh_reset_plan(project, prepared, rod, *, sdf_module, task_config, extra_sources):
    """Freeze real constructor inputs without Reset, SDK/solver step or new owner."""
    try:
        _require(tuple(rod.mesh_names) == PROXIMAL_POWER_MESH_NAMES,
                 "actual original seventeen-slot runtime required")
        baseline = original.prepare_mesh_reset_plan(project, prepared, rod, sdf_module=sdf_module)
        # This is a constructor-source projection, not a projected native state.
        # The original verifier independently checks the ten distal grids/table.
        baseline.pop("sha256")
        baseline["bindings"] = {name: baseline["bindings"][name] for name in HAND_NAMES}
        baseline = original.seal_plan(baseline)
        path = Path(task_config).resolve(strict=True)
        project = Path(project).resolve()
        plan = dict(version=PLAN_VERSION, source_scope="locked_original_seventeen_constructor_inputs",
            source_basis=deepcopy(baseline["source_basis"]), original_plan=baseline,
            cfg=deepcopy(rod._base.cfg), bindings=deepcopy(rod.bindings),
            task_config=dict(path=str(path), sha256=_sha(path)),
            extra_sources={k: str(Path(v)) for k,v in extra_sources.items()},
            implementation_sources=_implementation_sources(project))
        plan["entries"] = _expected_entries(plan)
        result = original.seal_plan(plan)
        verify_plan_sources(result)
        return result
    finally:
        rod = sdf_module = None


def _plan(plan, rod):
    try:
        p = deepcopy(plan)
        digest = p.pop("sha256", None)
        _require(digest == original._digest(p) and p["version"] == PLAN_VERSION,
                 "sealed seventeen-slot plan differs")
        _require(tuple(row["name"] for row in p["entries"]) ==
                 tuple(rod.mesh_names) == PROXIMAL_POWER_MESH_NAMES,
                 "all seventeen original slots in exact order required")
        _require(original._digest(p["bindings"]) == original._digest(rod.bindings)
                 and p["cfg"] == rod._base.cfg, "actual supported runtime source/material differs")
        for row in p["entries"]:
            a, origin = row["values"], row["origin"]
            _require(a.dtype == np.float32 and a.ndim == 3 and min(a.shape) >= 2
                     and np.isfinite(a).all() and origin.dtype == np.float32
                     and origin.shape == (3,) and np.isfinite(origin).all()
                     and type(row["spacing"]) in (int,float) and np.isfinite(row["spacing"]) and row["spacing"] > 0
                     and all(type(row[k]) in (int,float) and np.isfinite(row[k]) and row[k] >= 0
                             for k in ("mu_s","mu_k")),
                     "finite unchanged supported grid/material arguments required")
            static = row["name"] in ("table", *END_SUPPORT_NAMES)
            _require((row["static_pose"] is not None) == static, "exact three static support/table slots required")
            if static:
                pos, quat = map(np.asarray, row["static_pose"])
                _require(pos.shape == (3,) and quat.shape == (4,) and np.isfinite(pos).all()
                         and np.isfinite(quat).all() and abs(np.linalg.norm(quat)-1) <= 1e-6,
                         "finite original static pose required")
        verify_plan_sources(plan)
        verify_runtime_source(rod)
        return p
    finally:
        rod = None


_array = readback._array
_validate_clock = readback._validate_clock

def validate_supported_capture(packet):
    """Validate one owned same-clock native snapshot, including pre-reset states."""
    _require(set(packet)=={"clock_before","clock_after","source_instance_id","snapshot","capture_error"},
        "exact mesh capture fields required")
    _require(packet["capture_error"] is None,"mesh capture failed: "+str(packet["capture_error"]))
    before,after,raw=packet["clock_before"],packet["clock_after"],packet["snapshot"]
    _validate_clock(before);_validate_clock(after)
    _require(isinstance(before,dict) and before==after,"mesh getter changed actual clock")
    _require(type(packet["source_instance_id"]) is int and packet["source_instance_id"]>0,
        "explicit actual mesh owner identity required")
    expected={"schema","num_envs","mesh_count","current_positions_world_m","current_quaternions_xyzw",
        "target_positions_world_m","target_quaternions_xyzw","solver_positions_world_m",
        "solver_quaternions_xyzw","pending_teleport","device_state_source","pending_set","moving",
        "pair_capacity","pair_ready","flags_source","owner_time_s_before","owner_time_s_after",
        "owner_time_source","last_error"}
    _require(isinstance(raw,dict) and set(raw)==expected,"exact native mesh snapshot fields required")
    _require(raw["schema"]=="dh116.mesh_state_snapshot.v1" and type(raw["num_envs"]) is int
        and raw["num_envs"]==1 and type(raw["mesh_count"]) is int and raw["mesh_count"]==17,
        "actual single-environment seventeen slots required")
    _require(raw["device_state_source"]=="cuda_device_copy"
        and raw["flags_source"]=="native_host_sdf_store"
        and raw["owner_time_source"]=="native_host_owner_metadata_at_readback",
        "explicit device state and host metadata sources required")
    for prefix in ("current","target","solver"):
        _array(raw[prefix+"_positions_world_m"],(1,17,3),prefix+" position")
        q=_array(raw[prefix+"_quaternions_xyzw"],(1,17,4),prefix+" XYZW")
        _require(np.allclose(np.linalg.norm(q,axis=-1),1.,atol=2e-6,rtol=0),
            "unit native mesh XYZW required")
    tele=np.asarray(raw["pending_teleport"])
    _require(tele.shape==(1,17) and tele.dtype==np.dtype(np.uint8) and np.isin(tele,[0,1]).all(),
        "seventeen native teleport flags required")
    _require(all(type(raw[k]) is bool for k in ("pending_set","moving","pair_ready")),
        "typed native host flags required")
    _require(type(raw["pair_capacity"]) is int and raw["pair_capacity"]==CAPACITY,
        "original bounded pair allocation must survive reset")
    _require(type(raw["last_error"]) is int and raw["last_error"]==0,
        "actual mesh getter solver error is not zero")
    time=before["rod"]["solver_time_s"]
    _require(type(time) in (int,float) and math.isfinite(time)
        and type(raw["owner_time_s_before"]) in (int,float)
        and type(raw["owner_time_s_after"]) in (int,float)
        and raw["owner_time_s_before"]==raw["owner_time_s_after"]==time,
        "exact native owner metadata time differs")
    return raw

def capture_supported_mesh_state(sim, rod):
    packet = dict(clock_before=None,clock_after=None,source_instance_id=None,snapshot=None,capture_error=None)
    rope = None
    try:
        try:
            verify_runtime_source(rod)
            rope = rod.rope
            packet["source_instance_id"] = id(rope)
            packet["clock_before"] = original._clock(sim,rod)
            packet["snapshot"] = deepcopy(rope.mesh_state_snapshot())
            packet["clock_after"] = original._clock(sim,rod)
            _require(rod.rope is rope, "actual supported native mesh owner changed")
            validate_supported_capture(packet)
        except Exception as error:
            packet["capture_error"] = type(error).__name__+": "+str(error)
        return packet
    finally:
        sim = rod = rope = None


def validate_supported_readback(packet, *, reset_receipt, mesh_plan):
    packet, reset_receipt, mesh_plan = map(decode_owned, (packet,reset_receipt,mesh_plan))
    verify_plan_sources(mesh_plan)
    p = deepcopy(mesh_plan)
    digest = p.pop("sha256")
    _require(original._digest(p) == digest, "supported plan digest differs")
    _require(set(packet) == {"version","sources","mesh_plan_sha256","mesh_names","before","after",
        "complete","capture_error","new_sdk_steps","new_solver_steps","native_qualification","teacher_admission"},
        "exact supported readback fields required")
    _require(packet["version"] == VERSION and packet["sources"] == source_basis() == mesh_plan["source_basis"]
             and packet["mesh_plan_sha256"] == digest and packet["mesh_names"] == list(PROXIMAL_POWER_MESH_NAMES),
             "supported readback source/slot identities differ")
    _require(packet["complete"] is True and packet["capture_error"] is None
             and all(type(packet[k]) is int and packet[k] == 0 for k in ("new_sdk_steps","new_solver_steps"))
             and packet["native_qualification"] is False and packet["teacher_admission"] is False,
             "complete zero-step readback without native promotion required")
    validate_supported_capture(packet["before"])
    after = validate_supported_capture(packet["after"])
    _require(packet["before"]["source_instance_id"] == packet["after"]["source_instance_id"],
             "same actual supported native owner required")
    before_clock = packet["before"]["clock_before"]
    clock = packet["after"]["clock_after"]
    mesh = reset_receipt["mesh_reset"]
    _require(reset_receipt["version"] == RESET_TRANSACTION_VERSION
             and reset_receipt["status"] == "complete" and mesh["version"] == REBUILD_VERSION
             and mesh["mesh_plan_sha256"] == digest
             and mesh["mesh_names"] == list(PROXIMAL_POWER_MESH_NAMES)
             and mesh["source_tree_sha256"] == mesh_plan["source_basis"]["source_tree_sha256"]
             and all(type(row[k]) is int and row[k] == 0
                     for row in (reset_receipt,mesh) for k in ("new_sdk_steps","new_solver_steps")),
             "actual complete zero-step seventeen-slot Reset metadata differs")
    _require(mesh["before_clock"] == mesh["after_clock"] == clock,
             "mesh rebuild advanced its actual completed Reset clock")
    mutation = deepcopy(mesh["mutation_after_clock"])
    _require(mutation["rod"]["clock_valid"] is False, "poisoned actual mutation clock required")
    mutation["rod"]["clock_valid"] = clock["rod"]["clock_valid"]
    _require(mutation == clock, "mesh mutation clock/identity differs")
    _require(original._digest(mesh["native_state_before"]) == original._digest(mesh["native_state_after"]),
             "mesh reconstruction changed owned rod state/parameters")
    _require(before_clock == reset_receipt["before_clock"] and clock == reset_receipt["after_actual_clocks"]
             == reset_receipt["mesh_reset"]["after_clock"], "supported Reset/readback clocks differ")
    _no_step(before_clock,clock)
    b, a = before_clock["rod"], clock["rod"]
    _require(all(a[k] == b[k] for k in ("run_uuid","env_id","rod_uid"))
             and a["reset_id"] == b["reset_id"]+1 and a["episode_uid"] != b["episode_uid"]
             and a["physics_step"] == clock["sdk_physics_step"] and a["rod_tick"] == 0
             and a["clock_valid"] is True
             and reset_receipt["mesh_reset"]["mutation_after_clock"]["rod"]["clock_valid"] is False,
             "actual fresh supported Reset and poisoned mutation identity required")
    _require(after["pending_set"] is after["moving"] is after["pair_ready"] is False
             and not np.asarray(after["pending_teleport"]).any(),
             "supported mesh Reset left pending targets/motion/contact capture")
    for prefix in ("target","solver"):
        for suffix in ("positions_world_m","quaternions_xyzw"):
            _require(np.array_equal(after["current_"+suffix],after[prefix+"_"+suffix]),
                     "supported current/target/solver state differs")
    _require(tuple(row["name"] for row in mesh_plan["entries"]) == PROXIMAL_POWER_MESH_NAMES,
             "exact seventeen source slots required")
    for slot, entry in enumerate(mesh_plan["entries"]):
        if entry["static_pose"] is not None:
            pos, quat = entry["static_pose"]
        else:
            side, name = entry["name"].split(".")
            hand = reset_receipt["hands"][side]
            idx = hand["body_names"].index(name)
            _require(len(hand["body_names"]) == len(set(hand["body_names"])) == 23
                     and mesh_plan["bindings"][entry["name"]]["body_index"] == idx
                     and hand["link_paths"][0][idx] == mesh_plan["bindings"][entry["name"]]["prim_path"],
                     "same-side actual supported body index/path differs")
            pos, quat = reset_receipt["pad_poses_xyzw"][entry["name"]]
            _require(np.array_equal(np.r_[pos,quat],np.asarray(hand["body_poses_xyzw"])[0,idx]),
                     "supported collider copy differs from actual Reset SDK body")
        pos, quat = np.asarray(pos,np.float32), np.asarray(quat,np.float32)
        quat = quat/np.linalg.norm(quat)
        _require(np.array_equal(after["current_positions_world_m"][0,slot],pos),
                 "native supported current position differs from actual Reset/static source")
        _require(np.allclose(after["current_quaternions_xyzw"][0,slot],quat,
                            atol=readback.QUAT_COMPONENT_ATOL,rtol=0),
                 "native supported current quaternion differs from actual Reset/static source")
    return dict(version=VERSION,seventeen_reset_meshes_observed=True,pair_capacity_preserved=True,
                new_sdk_steps=0,new_solver_steps=0,native_qualification=False,teacher_admission=False)


class MeasuredSupportedMeshReset(DualHandResetTransaction):
    """Same original full Reset, then seventeen-slot clear/add and actual readback."""
    MESH_LAYOUT = PROXIMAL_POWER_MESH_NAMES
    RESET_VERSION = REBUILD_VERSION

    def __init__(self, seed, *, episode_uid, mesh_plan):
        super().__init__(seed, episode_uid=episode_uid)
        self.mesh_plan = deepcopy(mesh_plan)
        self.completed_reset_receipt = None

    def _validate_plan(self, rod):
        return _plan(self.mesh_plan,rod)

    def _rebuild(self, sim, rod, bindings, *, owner_resets):
        """Versioned17-slot execution of the pinned original clear/add algorithm."""
        from . import tabletop_dual_hand_reset as reset_owner
        from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings as scoped_bindings
        receipt = rope = None
        try:
            plan = self._validate_plan(rod)
            _require(reset_owner._clock is original._ORIGINAL_CLOCK,
                     "unexpected original Reset clock binding")
            with scoped_bindings((reset_owner, "_clock", original._clock)):
                receipt = super().run(sim,rod,bindings,owner_resets=owner_resets)
            self.completed_reset_receipt = deepcopy(receipt)
            before = original._clock(sim,rod)
            _require(before["rod"]["rod_tick"] == 0, "supported rebuild is Reset-only")
            rope = rod.rope
            before_state = original._state(rope)
            _no_step(before,original._clock(sim,rod))
            rod._needs_reset = True
            rope.clear_meshes()
            for index, row in enumerate(plan["entries"]):
                pos, quat = self._entry_pose(row,receipt)
                slot = rope.add_mesh(row["values"],row["origin"],row["spacing"],
                    pos=np.asarray(pos,np.float32),quat=np.asarray(quat,np.float32),
                    mu_s=row["mu_s"],mu_k=row["mu_k"])
                _require(type(slot) is int and slot == index, "supported reconstructed slot identity differs")
            after_state = original._state(rope)
            mutation = original._clock(sim,rod)
            _require(rod._needs_reset is True and mutation["rod"]["clock_valid"] is False,
                     "actual supported mesh mutation must poison the clock")
            normalized = deepcopy(mutation)
            normalized["rod"]["clock_valid"] = before["rod"]["clock_valid"]
            _no_step(before,normalized)
            _require(normalized == before, "supported reset identity changed during clear/add")
            _require(original._digest(before_state) == original._digest(after_state),
                     "supported mesh reconstruction changed rod numeric state/parameters")
            _require(np.asarray(rope.mesh_wrenches()).shape == (1,17,6),
                     "actual reconstructed seventeen-slot buffer required")
            _require(original._digest(plan["bindings"]) == original._digest(rod.bindings)
                     and plan["cfg"] == rod._base.cfg, "supported source binding changed during rebuild")
            rod._needs_reset = False
            receipt["mesh_reset"] = dict(version=REBUILD_VERSION,mesh_plan_sha256=self.mesh_plan["sha256"],
                before_clock=before,mutation_after_clock=mutation,after_clock=original._clock(sim,rod),
                mesh_names=list(PROXIMAL_POWER_MESH_NAMES),native_state_before=before_state,
                native_state_after=after_state,source_tree_sha256=plan["source_basis"]["source_tree_sha256"],
                sdk_clock_scope="at_most_240_steps_float32_callback_envelope",
                native_current_pose_readback=None,source_derived_commit=True,pair_capacity_readback=None,
                pair_buffer_policy="preserved_by_locked_clear_add_source",contact_readback_valid=False,
                borrowed_mesh_views_invalidated=True,new_sdk_steps=0,new_solver_steps=0,
                native_qualification=False,teacher_admission=False)
            return receipt
        finally:
            sim = rod = bindings = owner_resets = receipt = rope = None

    @staticmethod
    def _entry_pose(row, receipt):
        return row["static_pose"] if row["static_pose"] is not None else receipt["pad_poses_xyzw"][row["name"]]

    def run(self, sim, rod, bindings, *, owner_resets, record_readback=None):
        receipt = None
        started = emitted = False
        try:
            if self.status != "new":
                raise RuntimeError("supported Reset transaction cannot be reused")
            self.completed_reset_receipt = None
            self.readback_receipt = dict(version=VERSION,sources=source_basis(),
                mesh_plan_sha256=self.mesh_plan["sha256"],mesh_names=list(self.MESH_LAYOUT),
                before=None,after=None,complete=False,capture_error=None,new_sdk_steps=0,new_solver_steps=0,
                native_qualification=False,teacher_admission=False)
            started = True
            self._validate_plan(rod)
            self.readback_receipt["before"] = capture_supported_mesh_state(sim,rod)
            validate_supported_capture(self.readback_receipt["before"])
            receipt = self._rebuild(sim,rod,bindings,owner_resets=owner_resets)
            self.completed_reset_receipt = deepcopy(receipt)
            self.readback_receipt["after"] = capture_supported_mesh_state(sim,rod)
            validate_supported_capture(self.readback_receipt["after"])
            self.readback_receipt["complete"] = True
            validate_supported_readback(self.readback_receipt,reset_receipt=receipt,mesh_plan=self.mesh_plan)
            if record_readback is not None:
                emitted = True
                record_readback(dict(deepcopy(self.readback_receipt),
                                     completed_reset_receipt=deepcopy(receipt)))
            receipt["mesh_state_readback"] = deepcopy(self.readback_receipt)
            return receipt
        except BaseException as error:
            if started:
                self.status = "failed"
                rod._needs_reset = True
                rod._base.last = None
                self.readback_receipt["complete"] = False
                self.readback_receipt["capture_error"] = type(error).__name__+": "+str(error)
                # Unknown if an injected/failing owner stepped; never assert zero.
                self.readback_receipt["new_sdk_steps"] = None
                self.readback_receipt["new_solver_steps"] = None
                if record_readback is not None and not emitted:
                    try:
                        record_readback(dict(deepcopy(self.readback_receipt),
                            completed_reset_receipt=deepcopy(self.completed_reset_receipt)))
                    except BaseException as cleanup:
                        self.readback_receipt["record_error"] = type(cleanup).__name__+": "+str(cleanup)
            raise
        finally:
            sim = rod = bindings = owner_resets = record_readback = receipt = None
