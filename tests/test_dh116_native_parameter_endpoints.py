"""CPU same-clock protocol doubles; not native task qualification."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_dh116_material_director_readback import case
from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as native

@pytest.fixture
def endpoint(case):
    c=case
    c.owner=native.NativeParameterEndpoints(c.nodes,device="cuda:0")
    c.backend.__file__=str(native.BACKEND)
    c.rod.fixed_root_orientation=False
    cfg=c.owner.config["rod"]
    rest=np.linalg.norm(np.diff(c.nodes,axis=0),axis=1)[None]
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import lumped_masses
    c.raw=dict(schema="dh116.native_parameter_snapshot.v1",num_envs=1,nodes_per_env=41,time_s=0.,
        metadata_source="native_host_metadata",segment_source="cuda_device_copy",hardware_measured=False,
        contact_radius_m=cfg["radius_m"],gravity_world_m_s2=np.array(cfg["gravity_world_m_s2"],np.float32),
        contact_mu_static=cfg["mu_s"],contact_mu_kinetic=cfg["mu_k"],
        contact_v_slip_m_s=cfg["v_slip_m_s"],contact_kappa=cfg["kappa"],
        rest_lengths_m=rest,stored_k_stretch_n_m=rest*cfg["axial_stiffness_n"],
        rest_quaternion_norm_n_m=4*cfg["bend_stiffness_n_m2"]/rest,
        stretch_ea_n=np.full((1,40),cfg["axial_stiffness_n"],np.float32),
        bend_ei_n_m2=np.full((1,40),cfg["bend_stiffness_n_m2"],np.float32),
        inverse_masses_kg_inv=(1/lumped_masses(c.nodes,cfg["linear_density_kg_m"]))[None])
    def capture():
        c.events.append("parameters")
        return dict(deepcopy(c.raw),time_s=c.rod.time)
    c.rod.parameter_snapshot=capture
    return c

def test_actual_new_source_only_adds_owned_getter():
    proof=native.source_basis()
    assert proof["only_binding_getter_added"] is True
    assert proof["standalone_probe_qualified"] is True
    assert proof["task_physics_qualified"] is False
    assert proof["backend_sha256"]=="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"

def test_real_clock_bound_native_radius_and_rest(endpoint):
    c=endpoint
    p=c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args)
    assert p["identity"]==c.clock
    assert p["reward_radius_sample"]["value"]==pytest.approx(.005)
    assert p["reward_radius_sample"]["source"]["provenance"]=="native_host_metadata"
    assert p["reward_radius_sample"]["source"]["identity"]==c.clock
    assert p["native_parameters"]["fields"]["rest_lengths_m"]["source"]=="cuda_device_copy"
    assert p["teacher_admission"] is False and p["task_physics_qualified"] is False
    assert p["native_parameters"]["fixed_node_indices"]==[]
    c.advance()
    p2=c.owner.capture(c.row(),**c.args)
    assert p2["identity"]==c.clock
    assert p2["reward_radius_sample"]["source"]["identity"]!=p["identity"]

@pytest.mark.parametrize("fault",["radius","gravity","ea","ei","rest","mass","fixed","backend","clock","solver","fixed_orientation"])
def test_bad_parameter_or_owner_refused_and_poisons_sequence(endpoint,fault):
    c=endpoint
    c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args)
    c.advance()
    if fault=="radius":c.raw["contact_radius_m"]*=2
    if fault=="gravity":c.raw["gravity_world_m_s2"][2]=0
    if fault=="ea":c.raw["stretch_ea_n"]*=2;c.raw["stored_k_stretch_n_m"]*=2
    if fault=="ei":c.raw["bend_ei_n_m2"]*=2;c.raw["rest_quaternion_norm_n_m"]*=2
    if fault=="rest":c.raw["rest_lengths_m"]*=2
    if fault=="mass":c.raw["inverse_masses_kg_inv"]*=2
    if fault=="fixed":c.raw["inverse_masses_kg_inv"][0,0]=0
    if fault=="backend":c.backend.__file__="/wrong.so"
    if fault=="clock":c.sim.current_time_step_index+=1
    if fault=="solver":c.rod.last_error=1
    if fault=="fixed_orientation":c.rod.fixed_root_orientation=True
    with pytest.raises(ValueError):
        c.owner.capture(c.row(),**c.args)
    with pytest.raises(RuntimeError,match="fresh reset"):
        c.owner.capture(c.row(),**c.args)

def test_mid_getter_clock_drift_refused(endpoint):
    c=endpoint
    c.owner.capture_reset(c.row(),material_reference_uid="material-1",**c.args);c.advance()
    getter=c.rod.parameter_snapshot
    def drift():
        result=getter();c.advance();return result
    c.rod.parameter_snapshot=drift
    with pytest.raises(ValueError,match="clock"):
        c.owner.capture(c.row(),**c.args)

def test_fresh_reset_only_restarts_endpoint_sequence(endpoint):
    c=endpoint
    c.owner.capture_reset(c.row(),material_reference_uid="mat1",**c.args)
    c.advance();c.owner.capture(c.row(),**c.args)
    c.clock.update(reset_id=2,episode_uid="ep2",rod_tick=0)
    p=c.owner.capture_reset(c.row(),material_reference_uid="mat2",**c.args)
    assert p["identity"]["solver_time_s"]>0 and p["identity"]["rod_tick"]==0
    with pytest.raises(ValueError):
        c.owner.capture_reset(c.row(),material_reference_uid="mat2",**c.args)

def test_locked_source_inventory_covers_every_compiled_source():
    proof=native.source_basis()
    assert proof["source_tree_sha256"]=="3cc86d9864d0ffd05da64a2097378a5c8c2b779b2bf7003659ec1fa8e4306472"
    assert proof["source_file_count"]==72

def test_nondefault_stream_and_hidden_owner_retention_are_not_allowed(endpoint):
    import gc,weakref
    c=endpoint
    c.stream["current"]=7
    with pytest.raises(ValueError,match="default"):
        c.owner.capture_reset(c.row(),material_reference_uid="bad-stream",**c.args)
    c.stream["current"]=0
    c.clock.update(reset_id=2,episode_uid="ep2")
    c.owner.capture_reset(c.row(),material_reference_uid="good-stream",**c.args)
    ref=weakref.ref(c.rod)
    # Remove the test's own closures and argument references, retaining the reader.
    c.args.clear();c.rod.parameter_snapshot=None;c.rod=None;c.row=None;c.advance=None
    gc.collect()
    assert ref() is None
