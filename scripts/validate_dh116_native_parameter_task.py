"""Disk-only raw parameter reconciliation plus every existing physical/lifetime gate."""
from pathlib import Path
import json
import math
import numpy as np
from scripts import validate_dh116_dual_pinch_director_interval as previous
from scripts import validate_dh116_dual_pinch_interval as original
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as reader
from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import _clock
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import lumped_masses,verify_mass_readback
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned

AUDIT_NAME="native-parameter-task-post-exit-audit.json"
SCOPE="exact_2_setup_plus_16_native_parameter_task_endpoints_only"
ERRORS=previous.ERRORS
require=original.require

def packet(p,rod,sdk_time,rest,sources,*,reset,prior=None):
    identity=_clock(lambda:rod)
    expected_keys={"version","identity","sources","node_positions_m","segment_quaternions_xyzw",
        "native_parameters","raw_native_parameters","mass_readback","reward_radius_sample",
        "material_reference_uid","source_instance_id","boundary","new_sdk_steps","new_solver_steps",
        "task_physics_qualified","teacher_admission","substep_director_history_available",
        "continuous_contact_s_m","r0_tangential_slip_m"}
    require(set(p)==expected_keys,"native endpoint fields differ")
    original.exact(p,dict(version=reader.VERSION,
        boundary="completed_reset_endpoint" if reset else "completed_physics_endpoint",
        new_sdk_steps=0,new_solver_steps=0,task_physics_qualified=False,teacher_admission=False,
        substep_director_history_available=False,continuous_contact_s_m=None,r0_tangential_slip_m=None),
        "parameter endpoint scope differs")
    require(p["identity"]==identity and p["sources"]==sources,"parameter identity/source differs")
    require(type(p["source_instance_id"]) is int and p["source_instance_id"]>0
        and isinstance(p["material_reference_uid"],str) and bool(p["material_reference_uid"].strip()),
        "actual source instance/material identity required")
    require(math.isfinite(sdk_time) and math.isclose(sdk_time,identity["timestamp_s"],rel_tol=0,abs_tol=2e-7),
        "actual SDK timestamp differs")
    nodes=previous._array(p["node_positions_m"],(1,41,3),"nodes")
    require(np.array_equal(nodes,previous._array(rod["node_positions"],(1,41,3),"raw rod nodes")),
        "parameter endpoint node positions differ")
    require(np.array_equal(rest,previous._array(rod["rest_lengths_m"],(40,),"raw rest")),
        "constructor rest differs")
    q=previous._array(p["segment_quaternions_xyzw"],(1,40,4),"directors")
    require(np.allclose(np.linalg.norm(q,axis=-1),1.,rtol=0,atol=1e-4),"director is not unit length")
    raw=p["raw_native_parameters"]
    checked=reader.validate_snapshot(raw,expected_num_envs=1,expected_nodes=41,
        expected_time_s=identity["solver_time_s"])
    require(not checked["fixed_node_indices"],"free task rod cannot have fixed nodes")
    require(encode_owned(checked)==encode_owned(p["native_parameters"]),"derived native fields differ from raw")
    r=json.loads((reader.ROOT/reader.CONFIG).read_text())["rod"]
    expected={"contact_radius_m":r["radius_m"],"gravity_world_m_s2":r["gravity_world_m_s2"],
        "contact_mu_static":r["mu_s"],"contact_mu_kinetic":r["mu_k"],"contact_v_slip_m_s":r["v_slip_m_s"],
        "contact_kappa":r["kappa"],"stretch_ea_n":np.full((1,40),r["axial_stiffness_n"]),
        "bend_ei_n_m2":np.full((1,40),r["bend_stiffness_n_m2"]),"rest_lengths_m":rest[None]}
    for key,value in expected.items():
        require(np.allclose(raw[key],value,rtol=2e-6,atol=1e-9),"raw native/config parameter differs: "+key)
    # Original constructor mass uses segment lengths, not evolving node lengths.
    expected_mass=r["linear_density_kg_m"]*np.concatenate(([rest[0]/2],(rest[:-1]+rest[1:])/2,[rest[-1]/2]))
    mass=verify_mass_readback(raw["inverse_masses_kg_inv"],expected_mass)
    require(encode_owned(mass)==encode_owned(p["mass_readback"]),"raw native mass reconciliation differs")
    radius=dict(value=float(raw["contact_radius_m"]),source=dict(api="RopeBatch.parameter_snapshot.contact_radius_m",
        provenance="native_host_metadata",backend_sha256=reader.BACKEND_SHA,identity=identity,
        uncertainty=None,hardware_measured=False,task_physics_qualified=False))
    require(p["reward_radius_sample"]==radius,"native radius value/provenance differs")
    if prior is not None:
        require(all(p[k]==prior[k] for k in ("source_instance_id","material_reference_uid")),
            "actual rod/material instance changed")

def inspect_parameters(path,*,complete=True):
    # Reuse strict reset/physical ordering, clocks, terminal handling and 17-packet
    # count. Only the explicit versioned packet contract is different.
    with bindings((previous,"reader",reader),(previous,"_packet",packet)):
        return previous.inspect_directors(path,complete=complete)

def audit_outcome(output,outer_exit,cleanup,*,profile_sha256=None):
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    from scripts import prepare_dh116_native_parameter_task as operation
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    with bindings((builder,"BACKEND",reader.BACKEND),(builder,"BACKEND_SHA",reader.BACKEND_SHA),
            (director,"VERSION",operation.VERSION),(director,"inspect_interval",operation.inspect_interval),
            (previous,"reader",reader),(previous,"_packet",packet)):
        result=previous.audit_outcome(output,outer_exit,cleanup,profile_sha256=profile_sha256)
    result.update(qualification_scope=SCOPE,native_parameter_task_readback_qualified=result["lifecycle_qualified"],
        backend_sha256=reader.BACKEND_SHA,material_identified=False,
        teacher_admission=False,grasp_qualified=False)
    result["available_artifact_sha256"]={p.name:original.sha(p) for p in sorted(Path(output).glob("*"))
        if p.is_file() and not p.is_symlink() and p.name!=AUDIT_NAME}
    return result
