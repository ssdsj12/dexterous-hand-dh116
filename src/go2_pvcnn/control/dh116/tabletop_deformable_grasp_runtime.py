"""Real DeformX binding for a source-locked free-rod Pinch contact candidate.

Importing this module does not import Isaac, CUDA, or the DeformX extension.
Solver writes and real getter readbacks are deliberately reported separately.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .linear_coupling import active_limits
from .deformx_dh116_colliders import DISTAL_PAD_LINKS

VERSION = "dh116-free-pinch-candidate-v1"
REQUIRED = ("finger14_link", "finger23_link")
DT = 1 / 480


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_candidate(path, project):
    raw = Path(path).read_bytes()
    cfg = json.loads(raw)
    if cfg["version"] != VERSION or cfg["side"] != "left":
        raise ValueError("only the explicit left-hand candidate is frozen")
    for name, sha in cfg["sources"].items():
        if digest(Path(project)/name) != sha:
            raise ValueError(f"candidate source changed: {name}")
    if (cfg["physics_hz"], cfg["control_hz"], cfg["setup_steps"], cfg["measured_steps"]) != (480,60,2,238):
        raise ValueError("candidate must count 2 setup and 238 measured SDK steps")
    rod = cfg["rod"]
    if rod["pinned_node_ids"] or rod["gravity_world_m_s2"] != [0.,0.,-9.81]:
        raise ValueError("free ends and explicit downward gravity required")
    for key in ("linear_density_kg_m", "axial_stiffness_n", "bend_stiffness_n_m2", "radius_m"):
        if not math.isfinite(rod[key]) or rod[key] <= 0:
            raise ValueError(f"invalid {key}")
    if cfg["hand"]["fixed_base"] is not True or cfg["hand"]["self_collision"] is not False or rod["self_collision"] is not False:
        raise ValueError("candidate locks fixed base and disabled hand/rod self-collision")
    if rod["solver_substeps"] != 12 or rod["solver_iterations"] != 8:
        raise ValueError("candidate locks 12 rod substeps and 8 iterations")
    return cfg


def lumped_masses(nodes, linear_density):
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 3 or not np.isfinite(nodes).all():
        raise ValueError("finite [V>=3,3] nodes required")
    lengths = np.linalg.norm(np.diff(nodes,axis=0),axis=1)
    if np.any(lengths <= 0) or not math.isfinite(linear_density) or linear_density <= 0:
        raise ValueError("positive segment lengths and kg/m required")
    mass = np.zeros(len(nodes))
    mass[:-1] += lengths * linear_density / 2
    mass[1:] += lengths * linear_density / 2
    return mass


def verify_mass_readback(inv_masses, expected):
    inv = np.asarray(inv_masses, dtype=float).reshape(-1)
    expected = np.asarray(expected, dtype=float)
    if inv.shape != expected.shape or not np.isfinite(inv).all() or np.any(inv <= 0):
        raise ValueError("free rod unexpectedly has pinned/invalid node masses")
    mass = 1 / inv
    if not np.allclose(mass, expected, rtol=2e-5, atol=1e-9):
        raise ValueError("actual node mass differs from linear-density construction")
    return {"source":"RopeBatch.inv_masses", "node_mass_kg":mass.tolist(),
            "mass_kg":float(mass.sum()), "pinned_node_ids":[]}


def material_mapping(cfg):
    r = cfg["rod"]
    return {"density_unit":"kg/m", "constructor":{"density":r["linear_density_kg_m"],
            "k_stretch":r["axial_stiffness_n"], "k_bend":r["bend_stiffness_n_m2"],
            "max_h":DT/r["solver_substeps"], "num_itr":r["solver_iterations"]},
            "young_modulus":{"status":"not_identifiable_from_independent_EA_EI"},
            "shear_modulus":{"status":"unsupported_independent_parameter"},
            "native_getters":{"mass":True, "damping":True, "drag":True, "num_itr":True,
                              "gravity":False, "contact":False, "EA":False, "EI":False},
            "material_status":cfg["material_status"]}


def integrate_pinch_action(target, measured, action, *, side, dt):
    if side not in ("left","right") or not math.isfinite(dt) or dt <= 0:
        raise ValueError("valid side and positive dt required")
    a = np.asarray(action,dtype=float)
    q, measured = np.asarray(target,dtype=float), np.asarray(measured,dtype=float)
    allowed = [1,2] if side == "left" else [7,8]
    if a.shape != (24,) or not np.isfinite(a).all() or np.any(abs(a)>1):
        raise ValueError("finite normalized action24 required")
    if np.any(np.delete(a,allowed) != 0):
        raise ValueError("candidate action contains inactive hand/wrist/digit channels")
    if q.shape != (6,) or measured.shape != (6,) or not np.isfinite([q,measured]).all():
        raise ValueError("six finite named targets and measurements required")
    lo, hi = active_limits()
    offset = 0 if side == "left" else 6
    commanded = q + a[offset:offset+6] * .3 * dt
    return np.clip(np.clip(commanded, measured-.1, measured+.1),lo,hi)


def candidate_action(cfg, step):
    if not 1 <= step <= cfg["measured_steps"]:
        raise ValueError("phase step outside bounded candidate")
    a = np.zeros(24)
    spec = cfg["action"]
    if spec["zero_steps"] < step <= spec["zero_steps"]+spec["move_steps"]:
        a[1], a[2] = spec["thumb_velocity_normalized"], spec["index_velocity_normalized"]
    return a


def named_wrenches(wrenches, names):
    names = list(names)
    if len(names) != len(set(names)):
        raise ValueError("mesh identities must be distinct")
    w = np.asarray(wrenches, dtype=float)
    if w.shape != (1,len(names),6) or not np.isfinite(w).all():
        raise ValueError("actual [1,M,6] mesh wrench buffer required")
    return {name:w[0,i].copy() for i,name in enumerate(names)}


def contact_metrics(wrenches, threshold=1e-5):
    required = [np.asarray(wrenches[name],dtype=float)[:3] for name in REQUIRED]
    magnitudes = np.linalg.norm(required,axis=1)
    active = magnitudes > threshold
    opposed = bool(active.all() and np.dot(*required) < 0)
    return {"required_contact_fraction":float(active.mean()),
            "wrench_opposition_candidate":opposed,
            "table_contact":bool(np.linalg.norm(wrenches["table"][:3])>threshold),
            "source":"RopeBatch.mesh_wrenches:distinct_slots",
            "material_anchor_slip":{"status":"unavailable_no_per_mesh_node_contact_API"},
            "effective_opposed_contact":{"status":"unavailable_no_per_mesh_contact_normals"},
            "grasp_qualified":False}


def box_sdf(points, size):
    q = np.abs(np.asarray(points,dtype=float)) - np.asarray(size,dtype=float)/2
    return np.linalg.norm(np.maximum(q,0),axis=-1) + np.minimum(np.max(q,axis=-1),0)


class FreeRodRuntime:
    """Own a genuine RopeBatch; all bodies keep their original link identities."""
    def __init__(self, cfg, nodes, grids, pad_poses, table_center, table_size, *, episode_uid):
        from ropesim import RopeBatch, sdf
        self.cfg, self.episode_uid, self.reset_id = cfg, episode_uid, 0
        self.nodes = np.asarray(nodes,dtype=np.float32)
        self.rest = np.linalg.norm(np.diff(self.nodes,axis=0),axis=1)
        r = cfg["rod"]
        self.rope = RopeBatch.from_positions(
            self.nodes[None], **material_mapping(cfg)["constructor"],
            pinned=np.zeros(len(nodes),dtype=bool), fix_root_orientation=False,
            straight_rest=True, deterministic=True)
        self.rope.set_gravity(np.asarray(r["gravity_world_m_s2"],dtype=np.float32))
        self.rope.set_contact(radius=r["radius_m"],mu_s=r["mu_s"],mu_k=r["mu_k"],
                              v_slip=r["v_slip_m_s"],kappa=r["kappa"])
        self.rope.set_rope_contact(r["self_collision"])
        for key in ("damping","drag","drag_quad","drag_quad_t"):
            setattr(self.rope,key,r[key])
        self.names = list(DISTAL_PAD_LINKS)
        if set(grids) != set(self.names) or set(pad_poses) != set(self.names):
            raise ValueError("every original distal pad must have its own SDF and pose")
        # DeformX takes effective pair friction, not a PhysX material handle.
        from .tabletop_material import hand_material_contract
        hand = hand_material_contract(cfg["hand"]["material_profile"])["hand_contact_material"]
        for i,name in enumerate(self.names):
            p,q = pad_poses[name]
            slot = self.rope.add_mesh(*grids[name].as_args(),pos=p,quat=q,
                mu_s=max(r["mu_s"],hand["static_friction"]),
                mu_k=max(r["mu_k"],hand["dynamic_friction"]))
            if slot != i:
                raise RuntimeError("DeformX pad slot identity changed")
        table = cfg["table"]
        half = np.asarray(table_size)/2
        grid = sdf.from_function(lambda p:box_sdf(p,table_size),
             -half-table["sdf_margin_m"],half+table["sdf_margin_m"],table["sdf_spacing_m"])
        slot = self.rope.add_mesh(*grid.as_args(),pos=np.asarray(table_center,dtype=np.float32),
                                  mu_s=table["mu_s"],mu_k=table["mu_k"])
        if slot != len(self.names):
            raise RuntimeError("DeformX table slot identity changed")
        self.names.append("table")
        self.rope.enable_contact_forces(True)
        self.mass_readback = verify_mass_readback(self.rope.inv_masses(),lumped_masses(self.nodes,r["linear_density_kg_m"]))
        self.parameter_readback = {key:float(getattr(self.rope,key)) for key in
                                   ("damping","drag","drag_quad","drag_quad_t","num_itr")}
        for key,val in self.parameter_readback.items():
            expected = r["solver_iterations"] if key == "num_itr" else r[key]
            if not np.isclose(val,expected,rtol=1e-6,atol=1e-10):
                raise RuntimeError(f"native {key} readback mismatch")
        if self.rope.num_envs != 1 or self.rope.fixed_root_orientation:
            raise RuntimeError("free single rod construction failed")
        self.rod_tick = 0
        self.physics_step = None
        self.last = None
        self.time_origin = float(self.rope.time)

    def reset(self, *, physics_step, episode_uid, reset_id, pad_poses):
        if reset_id <= self.reset_id or episode_uid == self.episode_uid:
            raise ValueError("reset needs fresh episode UID and increasing reset ID")
        self.rope.reset_envs([0])
        for slot,name in enumerate(self.names[:-1]):
            p,q=pad_poses[name]
            self.rope.set_mesh_poses(slot,p,q,teleport=True)
        self.episode_uid,self.reset_id = episode_uid,reset_id
        self.physics_step,self.rod_tick,self.last = physics_step,0,None
        self.time_origin = float(self.rope.time)
        # reset_envs preserves global time and parameters, and may preserve force
        # scratch. No contact readback is marked valid until the next step_free.
        return self.capture(physics_step, contact_valid=False)

    def step(self, pad_poses, *, physics_step):
        if self.physics_step is None or physics_step != self.physics_step+1:
            raise ValueError("SDK clock skipped/repeated or reset was not bound")
        for slot,name in enumerate(self.names[:-1]):
            p,q=pad_poses[name]
            self.rope.set_mesh_poses(slot,p,q,teleport=False)
        self.rope.step_free(DT,self.cfg["rod"]["solver_substeps"])
        self.rod_tick += self.cfg["rod"]["solver_substeps"]
        self.physics_step = physics_step
        if self.rope.last_error:
            raise RuntimeError(f"DeformX solver error {self.rope.last_error}")
        self.last = self.capture(physics_step, contact_valid=True)
        return self.last

    def capture(self, physics_step, *, contact_valid):
        import torch
        # Clone GPU scratch before the next solver step; the solver stays owned.
        p = torch.as_tensor(self.rope.positions()).clone()
        v = torch.as_tensor(self.rope.velocities()).clone()
        if not torch.isfinite(p).all() or not torch.isfinite(v).all():
            raise RuntimeError("nonfinite actual rod state")
        nodes,vel = p.cpu().numpy(),v.cpu().numpy()
        forces = np.asarray(self.rope.contact_forces()).copy() if contact_valid else None
        w = named_wrenches(self.rope.mesh_wrenches(),self.names) if contact_valid else None
        return {"version":VERSION, "episode_uid":self.episode_uid,"reset_id":self.reset_id,
                "env_id":0,"rod_uid":"rod0","physics_step":physics_step,"rod_tick":self.rod_tick,
                "timestamp_s":physics_step*DT,"solver_time_s":float(self.rope.time),
                "node_positions":nodes,"velocities":vel,"rest_lengths_m":self.rest.copy(),
                "distributed_contact":forces,"named_wrenches":w,"contact_valid":contact_valid,
                "source":"RopeBatch.positions/velocities/contact_forces/mesh_wrenches"}

    def read_rod_snapshot(self):
        """Atomic adapter to the existing DeformableRodStateProvider tensor schema."""
        import torch
        from .tabletop_stereo_rod_state import VERSION as STATE_VERSION, UNITS
        if self.last is None:
            raise RuntimeError("read_rod_snapshot requires a measured solver step")
        row=self.last
        p=torch.as_tensor(row["node_positions"]).clone()
        v=torch.as_tensor(row["velocities"]).clone()
        edges=torch.diff(p,dim=1)
        lengths=torch.linalg.vector_norm(edges,dim=-1)
        directions=edges/lengths[...,None]
        tangent=torch.cat((directions[:,:1],directions[:,:-1]+directions[:,1:],directions[:,-1:]),dim=1)
        tangent=tangent/torch.linalg.vector_norm(tangent,dim=-1,keepdim=True)
        curvature=torch.zeros(p.shape[:2],dtype=p.dtype)
        curvature[:,1:-1]=torch.linalg.vector_norm(torch.diff(directions,dim=1),dim=-1)/(0.5*(lengths[:,:-1]+lengths[:,1:]))
        rest=torch.as_tensor(self.rest[None]).clone()
        return {"version":STATE_VERSION,"coordinate_frame":"world","units":dict(UNITS),
                "env_ids":[0],"episode_uids":[self.episode_uid],"reset_ids":[self.reset_id],
                "physics_step":row["physics_step"],"timestamp_s":row["timestamp_s"],
                "node_positions":p,"velocities":v,"centerline":p.clone(),"tangent":tangent,
                "curvature":curvature,"distributed_contact":torch.as_tensor(row["distributed_contact"]).clone(),
                "contact_valid":torch.ones(p.shape[:2],dtype=torch.bool),
                "rest_lengths_m":rest,"deformation":lengths/rest-1}


def validate_pinch_ledger(rows):
    """Check real multi-pad rows independently of producer success flags."""
    if len(rows) != 238:
        raise ValueError("candidate requires exactly 238 measured SDK rows")
    names=list(DISTAL_PAD_LINKS)
    expected_paths={name:f"/World/DeformXPinch/leftHand/{name}" for name in names}
    episode=None
    previous=np.zeros((5,6))
    contact_rows=0
    for phase,row in enumerate(rows,1):
        if (row.get("phase_step"),row.get("sdk_step"),row.get("rod_substep"),
            row.get("side"),row.get("rod_uid"),row.get("reset_id")) != (phase,phase+2,phase*12,"left","rod0",1):
            raise ValueError("candidate clock/rod/reset identity differs")
        uid=row.get("episode_uid")
        if not isinstance(uid,str) or not uid or (episode is not None and uid != episode):
            raise ValueError("candidate episode identity differs")
        episode=uid
        if (row.get("env_id"),row.get("physics_step"),row.get("policy_step"),row.get("substep"),
            row.get("rod_tick")) != (0,phase+2,(phase-1)//8,(phase-1)%8,phase*12):
            raise ValueError("candidate canonical identity clock differs")
        if not math.isclose(row["timestamp_s"],(phase+2)*DT,rel_tol=0,abs_tol=1e-9):
            raise ValueError("candidate canonical timestamp differs")
        if not math.isclose(row["solver_time_s"]-row["solver_time_origin_s"],phase*DT,rel_tol=0,abs_tol=2e-6):
            raise ValueError("candidate solver and SDK clocks differ")
        if not np.isclose(row["sdk_time_s"],row["sdk_step"]*DT,atol=2e-7,rtol=0):
            raise ValueError("candidate SDK timestamp differs")
        bindings=row["actual_pad_bindings"]
        if set(bindings)!=set(names):
            raise ValueError("candidate actual pad set differs")
        bodies=[]
        for slot,name in enumerate(names):
            b=bindings[name]
            if b["mesh_index"]!=slot or b["body_path"]!=expected_paths[name] or type(b["body_index"]) is not int:
                raise ValueError("candidate actual pad identity differs")
            bodies.append(b["body_index"])
        if len(set(bodies))!=5:
            raise ValueError("candidate actual pad bodies are not distinct")
        w=row["rod_mesh_wrenches_w"]
        if set(w)!=set(names+["table"]):
            raise ValueError("candidate requires separate table and five pad wrenches")
        values=np.asarray([w[n] for n in names+["table"]])
        forces=np.asarray(row["rod_node_contact_forces_w"])
        if values.shape!=(6,6) or forces.shape!=(41,3) or not np.isfinite(values).all() or not np.isfinite(forces).all():
            raise ValueError("candidate actual wrench/node force shape differs")
        ids=np.flatnonzero(np.linalg.norm(forces,axis=-1)>1e-5).tolist()
        if row["contact_node_ids"]!=ids:
            raise ValueError("candidate contact IDs differ from actual force")
        residual=np.linalg.norm(forces.sum(0)+values[:,:3].sum(0))
        if residual>1e-3 or not np.isclose(row["pair_force_residual_n"],residual,atol=1e-9,rtol=1e-6):
            raise ValueError("candidate hand/table/rod force accounting differs")
        if (row["applied_source_phase_step"]!=phase-1 or row["wrench_write_calls"]!=1
                or not np.allclose(row["applied_wrench_com_w"],previous,atol=2e-5,rtol=2e-5)):
            raise ValueError("candidate delayed physical reaction identity differs")
        previous=np.asarray(row["rod_on_hand_com_w"])
        if previous.shape!=(5,6) or not np.isfinite(previous).all():
            raise ValueError("candidate COM reaction shape differs")
        metrics=contact_metrics(w)
        if row["r0_raw_measurements"] != metrics:
            raise ValueError("candidate R0 does not match named contact readback")
        if (row["failure"] is not None or row["external_authored_force_on_rod"] is not False
                or row["attachments"]!=[] or row["full_state371_bound"] is not False):
            raise ValueError("failed or externally supported candidate")
        contact_rows += int(metrics["wrench_opposition_candidate"])
    return {"ledger_validation_passed":True,"wrench_opposition_candidate_rows":contact_rows,
            "grasp_qualified":False}


class NativeOwners:
    def __init__(self):
        self.resources = {}

    def keep(self, name, value):
        if name in self.resources:
            raise ValueError("duplicate native owner")
        self.resources[name] = value
        return value


def run_owned_call(operation, *, synchronize, ownership):
    """Release solver/view/composer owners even when an SDK frame raises.

    The completed failing frame is explicitly cleared before STOP/App close;
    propagating its original traceback would keep closure/SDK owners alive.
    """
    import gc
    import traceback
    owners = NativeOwners()
    result = None
    failure = None
    failed_tb = None
    failed_exception = None
    cleanup_errors = []
    try:
        result = operation(owners)
    except BaseException as error:
        failure = traceback.format_exc()
        failed_tb, failed_exception = error.__traceback__, error
    finally:
        try:
            if "composer" in owners.resources:
                owners.resources["composer"].reset()
        except BaseException:
            cleanup_errors.append(traceback.format_exc())
        try:
            synchronize()
        except BaseException:
            cleanup_errors.append(traceback.format_exc())
        if failed_tb is not None:
            traceback.clear_frames(failed_tb)
            failed_exception.__traceback__ = None
            failed_tb = failed_exception = None
        # Provider owns a bound runtime reader; runtime owns RopeBatch.
        for name in ("provider", "runtime", "view", "composer", "hand"):
            owners.resources.pop(name, None)
        owners.resources.clear()
        gc.collect()
        ownership.checkpoint("owned hand and solver references dropped")
    if failure or cleanup_errors:
        raise RuntimeError("\n".join(([failure] if failure else []) + cleanup_errors)) from None
    return result
