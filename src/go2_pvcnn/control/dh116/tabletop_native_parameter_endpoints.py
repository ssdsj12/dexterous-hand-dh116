"""Opt-in completed-boundary parameters for the isolated getter backend.

Reuse the existing reset/endpoint sequence guard. All native handles are borrowed;
the reader retains only a weak rod identity. No default backend or reward switch.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import weakref
import numpy as np

from . import tabletop_material_director_readback as prior
from .tabletop_deformx_parameter_snapshot import validate_snapshot
from .tabletop_bimanual_table_contacts import _clock
from .tabletop_deformable_grasp_runtime import lumped_masses,verify_mass_readback
from scripts.prepare_dh116_deformx_parameter_getters import GETTER, POST_PAIR_BINDINGS_SHA

VERSION="dh116-native-parameter-endpoints-v1"
ROOT=Path(__file__).resolve().parents[4]
BUILD=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/parameter-getters-build-e96wtu")
BACKEND=BUILD/"build/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so"
BACKEND_SHA="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"
CONFIG="configs/dh116_tasks/flexible_pinch_candidate_v1.json"
CONFIG_SHA="9f751680de805c5075282c16bc5c5e5e30b8b7c261a3dd36f3b7b1cbab8f35c8"
PROBE_RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-native-parameter-getters-native/ca509b7593c281b093e51ce078f17dbb25b58d1795b0fbd2c0640dbaddb0f63c/attempt-4gj_nj48/receipt.json")
PROBE_SHA="85f0b6e89c25eaf9f8e8d3d003eab1bc91cd7b2376459f969b0b66b79ee63ad5"
BINDING_SHA="1be4aad4bfc6de1c4a2c44e891d57ce7e163b78c24a8bf99619d1cb1fe0ce836"

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _locked_source_tree():
    lock=json.loads((ROOT/"docs/contracts/deformx_parameter_getter_source_lock_v1.json").read_text())
    inventory=lock["files"]
    digest=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if (digest!="3cc86d9864d0ffd05da64a2097378a5c8c2b779b2bf7003659ec1fa8e4306472"
            or len(inventory)!=72 or lock["source_commit"]!="96b8514c7c55c724f1abba1fbf6ed6642b7db72b"):
        raise ValueError("whole compiled source inventory differs")
    root=BUILD/"source"
    for name,expected in inventory.items():
        p=root/name
        if (p.is_symlink() or not p.resolve(strict=True).is_relative_to(root)
                or _sha(p)!=expected):
            raise ValueError("compiled source payload differs: "+name)
        if name!="csrc/bindings.cpp" and _sha(prior.BUILD/"source"/name)!=expected:
            raise ValueError("non-getter source differs from the original backend: "+name)
    return inventory,digest

def source_basis():
    """Validate the exact new binary and unchanged dynamics/director API sources."""
    inventory,tree_sha=_locked_source_tree()
    old=prior.source_basis()
    sources={str(BACKEND):BACKEND_SHA,str(ROOT/CONFIG):CONFIG_SHA,
             str(PROBE_RECEIPT):PROBE_SHA,str(BUILD/"source/csrc/bindings.cpp"):BINDING_SHA}
    for name,expected in prior.SOURCE_SHA.items():sources[str(BUILD/"source"/name)]=expected
    for name,details in old["sources"].items():
        if name!="csrc/bindings.cpp":sources[str(BUILD/"source"/name)]=details["sha256"]
    sources[str(BUILD/"build/python/ropesim/__init__.py")]=prior.PACKAGE_SHA
    for path,expected in sources.items():
        if _sha(path)!=expected:raise ValueError("locked parameter endpoint source differs: "+path)
    binding=(BUILD/"source/csrc/bindings.cpp").read_text()
    if binding.count(GETTER)!=1 or hashlib.sha256(binding.replace(GETTER,"").encode()).hexdigest()!=POST_PAIR_BINDINGS_SHA:
        raise ValueError("new bindings contain more than the reviewed getter delta")
    receipt=json.loads(PROBE_RECEIPT.read_text())
    if (receipt["qualification_scope"]!="zero_step_parameter_getter_only"
            or receipt["native_qualification"] is not True or receipt["outer_exit_code"]!=0):
        raise ValueError("actual zero-step probe scope not qualified")
    for name,expected in receipt["artifact_sha256"].items():
        path=Path(receipt["output"])/name
        if _sha(path)!=expected:raise ValueError("probe artifact bytes differ")
        sources[str(path)]=expected
    return dict(version=VERSION,backend_path=str(BACKEND),backend_sha256=BACKEND_SHA,
        sources=sources,source_tree_sha256=tree_sha,source_file_count=len(inventory),
        only_binding_getter_added=True,standalone_probe_qualified=True,
        task_physics_qualified=False,hardware_measured=False)

class NativeParameterEndpoints(prior.MaterialDirectorReadback):
    """Inherited episode sequencing, new explicit backend and native parameter source."""
    def __init__(self,constructor_nodes_m,*,device):
        self.sources=source_basis()
        self.config=json.loads((ROOT/CONFIG).read_text())
        self._nodes=prior._array(np.asarray(constructor_nodes_m,dtype=np.float32),(41,3),"constructor nodes")
        self._rest=np.linalg.norm(np.diff(self._nodes,axis=0),axis=1)
        if np.any(self._rest<=0):raise ValueError("positive constructor segments required")
        if not isinstance(device,str) or not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("explicit CUDA device required")
        self.device=device
        self._last=self._reset_identity=self._material_uid=self._rope_ref=None
        self._seen_episodes=set();self._seen_material=set();self._failed=True

    def _read(self,row,*,rope,backend,sim,read_clock):
        boundary=None
        try:
            identity=_clock(lambda:row)
            if self._rope_ref is None:self._rope_ref=weakref.ref(rope)
            elif self._rope_ref() is not rope:raise ValueError("actual rod instance changed")
            if (Path(backend.__file__).resolve()!=BACKEND or _sha(BACKEND)!=BACKEND_SHA
                    or type(rope) is not backend.RopeBatch or rope.num_envs!=1
                    or rope.verts_per_rope!=41 or rope.fixed_root_orientation is not False):
                raise ValueError("exact isolated free-root 41-node backend required")
            def boundary():
                actual=_clock(read_clock)
                if (actual!=identity or type(sim.current_time_step_index) is not int
                        or sim.current_time_step_index!=identity["physics_step"]
                        or not math.isclose(float(sim.current_time),identity["timestamp_s"],abs_tol=2e-7,rel_tol=0)
                        or float(rope.time)!=identity["solver_time_s"] or rope.last_error!=0):
                    raise ValueError("actual SDK/solver clock or error differs")
                return actual,float(sim.current_time),float(rope.time)
            before=boundary()
            nodes,directors=prior._copy_endpoint(rope,self.device)
            raw=rope.parameter_snapshot()
            checked=validate_snapshot(raw,expected_num_envs=1,expected_nodes=41,expected_time_s=before[2])
            if before!=boundary():raise ValueError("clock changed during native parameter capture")
            nodes=prior._array(nodes,(1,41,3),"actual endpoint nodes")
            directors=prior._array(directors,(1,41,4),"actual endpoint directors")
            if (not np.array_equal(nodes,prior._array(row["node_positions"],(1,41,3),"rod row"))
                    or not np.array_equal(self._rest,prior._array(row["rest_lengths_m"],(40,),"rest row"))
                    or not np.allclose(np.linalg.norm(directors[:,:40],axis=-1),1,atol=1e-4,rtol=0)):
                raise ValueError("actual endpoint geometry/constructor rest differs")
            r=self.config["rod"]
            expected={"contact_radius_m":r["radius_m"],"gravity_world_m_s2":r["gravity_world_m_s2"],
                "contact_mu_static":r["mu_s"],"contact_mu_kinetic":r["mu_k"],
                "contact_v_slip_m_s":r["v_slip_m_s"],"contact_kappa":r["kappa"],
                "stretch_ea_n":np.full((1,40),r["axial_stiffness_n"]),
                "bend_ei_n_m2":np.full((1,40),r["bend_stiffness_n_m2"]),
                "rest_lengths_m":self._rest[None]}
            for key,value in expected.items():
                if not np.allclose(raw[key],value,rtol=2e-6,atol=1e-9):
                    raise ValueError("native parameter differs from unchanged task: "+key)
            mass=verify_mass_readback(np.asarray(raw["inverse_masses_kg_inv"]),
                                      lumped_masses(self._nodes,r["linear_density_kg_m"]))
            if checked["fixed_node_indices"]:raise ValueError("free rod has fixed nodes")
            owned={k:np.array(v,copy=True) if isinstance(v,np.ndarray) else deepcopy(v) for k,v in raw.items()}
            radius_source=dict(api="RopeBatch.parameter_snapshot.contact_radius_m",
                provenance="native_host_metadata",backend_sha256=BACKEND_SHA,identity=deepcopy(identity),
                uncertainty=None,hardware_measured=False,task_physics_qualified=False)
            return dict(version=VERSION,identity=deepcopy(identity),sources=deepcopy(self.sources),
                node_positions_m=nodes,segment_quaternions_xyzw=directors[:,:40].copy(),
                native_parameters=checked,raw_native_parameters=owned,mass_readback=mass,
                reward_radius_sample=dict(value=float(raw["contact_radius_m"]),source=radius_source),
                material_reference_uid=self._material_uid,source_instance_id=id(rope),
                boundary="completed_physics_endpoint",new_sdk_steps=0,new_solver_steps=0,
                task_physics_qualified=False,teacher_admission=False,
                substep_director_history_available=False,continuous_contact_s_m=None,
                r0_tangential_slip_m=None)
        finally:
            self=row=rope=backend=sim=read_clock=boundary=None
