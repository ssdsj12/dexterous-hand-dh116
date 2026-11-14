"""Opt-in, source-locked solver parameter readback; no physics or reward writes.

Configuration, constructor-derived values and native getters are distinct
provenance classes. No material is claimed hardware-calibrated. Invoke only in
the existing physical owner's exclusive completed-boundary read phase.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from .tabletop_bimanual_table_contacts import _clock
from .tabletop_deformable_grasp_runtime import FreeRodRuntime,lumped_masses,verify_mass_readback
from .tabletop_dual_hand_rod_runtime import DualHandRodRuntime,PADS
from .tabletop_material import hand_material_contract
from .tabletop_material_director_readback import source_basis as director_source_basis
from .tabletop_dual_pinch_builder import BACKEND,BACKEND_SHA

VERSION="dh116-flexible-parameter-provenance-readback-v1"
ROOT=Path(__file__).resolve().parents[4]
CONFIG="configs/dh116_tasks/flexible_pinch_candidate_v1.json"
GETTERS=("damping","drag","drag_quad","drag_quad_t","num_itr")
LOCAL_PINS={
    CONFIG:"9f751680de805c5075282c16bc5c5e5e30b8b7c261a3dd36f3b7b1cbab8f35c8",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_runtime.py":
        "4e0f7762589b761c2129d17013e021703f67dde485275c484c5c9ceed95565c7",
    "src/go2_pvcnn/control/dh116/tabletop_dual_hand_rod_runtime.py":
        "e568a5b1951d43c278bff9949f3773514b1cc8f13e5d948ee0bd982a8c894688",
    "src/go2_pvcnn/control/dh116/tabletop_dual_pinch_builder.py":
        "5b190d1896647773c2f8895e73224f387d1dd17a565a44c4e2212db8f477fe17",
    "src/go2_pvcnn/control/dh116/tabletop_material.py":
        "4874308850de6391df2816a175fa543d7cee255eeb6cccb36c0012fb0c8e1e88"}
MISSING_GETTERS=["radius","gravity","EA","EI","friction","rest_lengths"]


def _require(condition,message):
    if not condition:raise ValueError(message)


def source_basis():
    basis=director_source_basis()
    files={}
    for name,wanted in LOCAL_PINS.items():
        path=ROOT/name
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual==wanted,"parameter source SHA differs: "+name)
        files[name]=actual
    return dict(version=VERSION,backend_path=str(BACKEND),backend_sha256=BACKEND_SHA,
        local_source_sha256=files,backend_source_basis=basis,
        reader_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        getters=list(GETTERS),native_getters_unavailable=list(MISSING_GETTERS),
        material_identified=False,native_qualification=False)


def _field(value,unit,frame,provenance,source,*,native=False):
    return dict(value=deepcopy(value),unit=unit,coordinate_frame=frame,
        valid=value is not None,provenance=provenance,source=source,
        native_measured=native,hardware_measured=False,uncertainty=None,
        uncertainty_calibrated=False)


def capture_parameters(runtime,*,sim,backend,expected_clock,device):
    """Return owned data only, including on error; never retain native handles."""
    base=rope=boundary=None
    try:
        import torch
        basis=source_basis()
        _require(type(runtime) is DualHandRodRuntime and type(runtime._base) is FreeRodRuntime,
                 "actual locked dual/free runtime classes required")
        base=runtime._base;rope=runtime.rope
        _require(type(rope) is backend.RopeBatch and Path(backend.__file__).resolve()==BACKEND.resolve(),
                 "actual locked backend type and path required")
        _require(type(rope.num_envs) is int and rope.num_envs==1
            and type(rope.verts_per_rope) is int and rope.verts_per_rope==41 and rope.fixed_root_orientation is False,
                 "one free-root 41-node rod required")
        config=json.loads((ROOT/CONFIG).read_text())
        _require(base.cfg==config,"runtime configuration differs from frozen constructor source")
        cfg=deepcopy(base.cfg);rod=cfg["rod"]
        nodes=np.array(base.nodes,copy=True)
        _require(nodes.shape==(41,3) and nodes.dtype==np.float32 and np.isfinite(nodes).all(),
                 "actual float32 constructor nodes required")
        rest=np.linalg.norm(np.diff(nodes,axis=0),axis=1)
        _require(np.all(rest>0) and np.array_equal(rest,base.rest),
                 "constructor material rest differs")
        expected=_clock(lambda:expected_clock)
        _require(expected==expected_clock and expected["env_id"]==0 and expected["reset_id"]>=1,
                 "exact actual completed single-environment reset identity required")
        dev=torch.device(device)
        _require(dev.type=="cuda" and dev.index is not None and dev==torch.device(sim.device),
                 "explicit actual simulation CUDA device required")
        _require(torch.cuda.current_stream(dev)==torch.cuda.default_stream(dev),
                 "default CUDA stream required")
        def boundary():
            clock=_clock(runtime.clock)
            _require(clock==expected and type(sim.current_time_step_index) is int
                and sim.current_time_step_index==expected["physics_step"]
                and math.isclose(float(sim.current_time),expected["timestamp_s"],rel_tol=0,abs_tol=2e-7)
                and float(rope.time)==expected["solver_time_s"] and rope.last_error==0,
                "actual SDK/solver clock or error differs")
            return clock,float(sim.current_time),float(rope.time)
        before=boundary()
        torch.cuda.synchronize(dev)
        inv=np.array(rope.inv_masses(),copy=True)
        _require(inv.shape==(1,41) and inv.dtype.kind=="f","native [1,41] inverse mass required")
        masses=verify_mass_readback(inv,lumped_masses(nodes,rod["linear_density_kg_m"]))
        parameters={}
        for name in GETTERS:
            value=getattr(rope,name)
            _require(isinstance(value,(int,float,np.integer,np.floating))
                and not isinstance(value,(bool,np.bool_)) and math.isfinite(float(value)),
                "finite native parameter required: "+name)
            requested=rod["solver_iterations" if name=="num_itr" else name]
            _require((type(value) is int and value==requested) if name=="num_itr"
                else np.isclose(float(value),requested,rtol=1e-6,atol=1e-10),
                "native parameter differs from frozen configuration: "+name)
            parameters[name]=int(value) if name=="num_itr" else float(value)
        torch.cuda.synchronize(dev)
        _require(boundary()==before,"clock advanced during parameter readback")
        _require(torch.cuda.current_stream(dev)==torch.cuda.default_stream(dev),
                 "CUDA stream changed during parameter readback")
        _require(base.cfg==cfg and np.array_equal(base.nodes,nodes) and np.array_equal(base.rest,rest),
                 "constructor/config source changed during readback")
        # These cached values were acquired by the original constructor. Compare
        # them to fresh getters; do not substitute the cache for a current read.
        _require(base.mass_readback==masses and all(
            np.isclose(base.parameter_readback[k],v,rtol=1e-6,atol=1e-10)
            for k,v in parameters.items()),"constructor getter cache differs from current native read")
        return _owned_packet(expected,basis,cfg,nodes,rest,inv,masses,parameters,int(rope.verts_per_rope))
    finally:
        runtime=sim=backend=base=rope=boundary=None


def _owned_packet(expected,basis,cfg,nodes,rest,inv,masses,parameters,node_count):
    rod=cfg['rod']
    fields={
        "mass_kg":_field(masses["mass_kg"],"kg","scalar","native_getter_derived","sum(1/RopeBatch.inv_masses)",native=True),
        "node_count":_field(node_count,"count","scalar","native_getter","RopeBatch.verts_per_rope",native=True),
        "pinned_node_ids":_field([],"node_index","material","native_getter_derived","strictly positive finite RopeBatch.inv_masses",native=True),
        "fixed_root_orientation":_field(False,"boolean","material","native_getter","RopeBatch.fixed_root_orientation",native=True),
        "length_m":_field(float(rest.astype(float).sum()),"m","material_rest","constructor_derived_not_native_readback",
                          "sum(norm(diff(float32 constructor nodes)))"),
    }
    configured=(
        ("radius_m","radius_m","m","scalar"),("gravity_world_m_s2","gravity_world_m_s2","m/s^2","world"),
        ("linear_density_kg_m","linear_density_kg_m","kg/m","material"),
        ("axial_stiffness_n","axial_stiffness_n","N","material"),
        ("bend_stiffness_n_m2","bend_stiffness_n_m2","N*m^2","material"),
        ("global_friction_static","mu_s","dimensionless","solver_global_contact"),
        ("global_friction_dynamic","mu_k","dimensionless","solver_global_contact"),
        ("solver_substeps_per_physics_step","solver_substeps","count","clock"),
    )
    for name,key,unit,frame in configured:
        fields[name]=_field(rod[key],unit,frame,"source_locked_configuration_not_native_readback",
                           CONFIG+":rod."+key)
    for name,value in parameters.items():
        fields[name]=_field(value,"count" if name=="num_itr" else "solver_coefficient_units_unqualified",
            "scalar","native_getter","RopeBatch."+name,native=True)
    for name,unit,reason in (
        ("density_kg_m3","kg/m^3","no volume-density getter; independent linear-density simulation coefficient"),
        ("young_modulus_pa","Pa","independent EA/EI do not identify an isotropic material"),
        ("shear_modulus_pa","Pa","no independently exposed shear modulus")):
        fields[name]=_field(None,unit,"material","unavailable",reason)
    hand=hand_material_contract(cfg["hand"]["material_profile"])["hand_contact_material"]
    pair_config={side+"."+pad:dict(mu_s=max(rod["mu_s"],hand["static_friction"]),
        mu_k=max(rod["mu_k"],hand["dynamic_friction"]),native_readback=False,
        provenance="source_locked_add_mesh_configuration_not_native_readback",
        source="FreeRodRuntime/DualHandRodRuntime add_mesh arguments")
        for side in ("left","right") for pad in PADS}
    pair_config["table"]=dict(mu_s=cfg["table"]["mu_s"],mu_k=cfg["table"]["mu_k"],
        native_readback=False,provenance="source_locked_add_mesh_configuration_not_native_readback",
        source="FreeRodRuntime table add_mesh arguments")
    return dict(version=VERSION,identity=deepcopy(expected),sources=basis,fields=fields,
        configured=cfg,configured_contact_pairs=pair_config,
        constructor_nodes_m=nodes.tolist(),constructor_rest_lengths_m=rest.tolist(),
        node_inverse_masses_kg_inv=inv.tolist(),mass_readback=masses,current_native_parameters=parameters,
        material_status=cfg["material_status"],material_identified=False,
        synchronization="default CUDA stream synchronized before and after owned host getter copies",
        exclusive_owner_read_phase_required=True,configuration_proves_native_getter_value=False,
        native_qualification=False,teacher_admission=False,reward_radius_sample=None,
        new_physics_steps=0,new_solver_steps=0,ppo_updates=0)

def validate_packet(packet,*,identity,constructor_nodes):
    """Disk-only reconstruction from source-locked inputs, never a GPU read."""
    _require(isinstance(packet,dict),"owned parameter packet required")
    expected=_clock(lambda:identity)
    _require(expected==identity and packet["identity"]==expected,
             "parameter identity differs from actual endpoint")
    basis=source_basis()
    _require(packet["sources"]==basis,"parameter source basis differs")
    cfg=json.loads((ROOT/CONFIG).read_text())
    _require(packet["configured"]==cfg,"parameter constructor configuration differs")
    nodes=np.asarray(constructor_nodes)
    _require(nodes.shape==(41,3) and nodes.dtype==np.float32 and np.isfinite(nodes).all(),
             "actual finite float32 reset constructor nodes required")
    rest=np.linalg.norm(np.diff(nodes,axis=0),axis=1)
    _require(np.all(rest>0),"positive constructor segments required")
    inv=np.asarray(packet["node_inverse_masses_kg_inv"])
    _require(inv.shape==(1,41) and inv.dtype.kind=="f","owned floating inverse masses required")
    masses=verify_mass_readback(inv,lumped_masses(nodes,cfg["rod"]["linear_density_kg_m"]))
    parameters=packet["current_native_parameters"]
    _require(set(parameters)==set(GETTERS),"exact native getter set required")
    for name,value in parameters.items():
        wanted=cfg["rod"]["solver_iterations" if name=="num_itr" else name]
        _require(type(value) in (int,float) and math.isfinite(value)
            and (type(value) is int and value==wanted if name=="num_itr"
                 else np.isclose(value,wanted,rtol=1e-6,atol=1e-10)),
            "finite matching raw native getter required: "+name)
    canonical=_owned_packet(expected,basis,cfg,nodes,rest,inv,masses,parameters,41)
    _require(json.dumps(packet,sort_keys=True,separators=(",",":"),allow_nan=False)
        ==json.dumps(canonical,sort_keys=True,separators=(",",":"),allow_nan=False),
        "parameter packet fields, units, unknowns or provenance differ from raw source reconstruction")
    return dict(version=VERSION,parameter_readback_valid=True,field_count=len(canonical["fields"]),
        mass_kg=masses["mass_kg"],configured_radius_m=cfg["rod"]["radius_m"],
        material_identified=False,teacher_admission=False,native_qualification=False)
