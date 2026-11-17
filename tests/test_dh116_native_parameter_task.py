"""CPU endpoint and owner composition; these are not native evidence."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from tests.test_dh116_material_director_readback import case
from tests.test_dh116_native_parameter_endpoints import endpoint
from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as reader

def validate(packet,row,nodes,reset=True,prior=None):
    from scripts import validate_dh116_native_parameter_task as audit
    rest=np.linalg.norm(np.diff(nodes,axis=0),axis=1)
    audit.packet(packet,row,row["timestamp_s"],rest,reader.source_basis(),reset=reset,prior=prior)

def test_disk_packet_independently_reconciles_raw_parameters(endpoint):
    c=endpoint;p=c.owner.capture_reset(c.row(),material_reference_uid="mat1",**c.args)
    validate(p,c.row(),c.nodes)
    c.advance();q=c.owner.capture(c.row(),**c.args)
    validate(q,c.row(),c.nodes,reset=False,prior=p)

@pytest.mark.parametrize("damage",["clock","raw","native","radius","source","mass","director","nodes","admission","uid","extra"])
def test_packet_forgery_rejected(endpoint,damage):
    c=endpoint;p=c.owner.capture_reset(c.row(),material_reference_uid="mat1",**c.args)
    if damage=="clock":p["identity"]["physics_step"]+=1
    if damage=="raw":p["raw_native_parameters"]["contact_radius_m"]*=2
    if damage=="native":p["native_parameters"]["fixed_node_indices"]=[0]
    if damage=="radius":p["reward_radius_sample"]["value"]*=2
    if damage=="source":p["reward_radius_sample"]["source"]["hardware_measured"]=True
    if damage=="mass":p["mass_readback"]["verified"]=False
    if damage=="director":p["segment_quaternions_xyzw"][0,0]=0
    if damage=="nodes":p["node_positions_m"][0,0,2]+=1
    if damage=="admission":p["teacher_admission"]=True
    if damage=="uid":p["source_instance_id"]=0
    if damage=="extra":p["unknown"]=1
    with pytest.raises((ValueError,KeyError,TypeError)):
        validate(p,c.row(),c.nodes)

def test_only_new_operation_selects_new_backend(monkeypatch):
    from scripts import prepare_dh116_native_parameter_task as operation
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    before=builder.BACKEND,builder.BACKEND_SHA
    calls=[]
    def run(*a,**kw):
        calls.append((builder.BACKEND,builder.BACKEND_SHA,operation.director.make_reader))
        return "owner"
    monkeypatch.setattr(operation,"run_director",run)
    assert operation.run_existing_app()== "owner"
    assert calls==[(reader.BACKEND,reader.BACKEND_SHA,operation.make_reader)]
    assert (builder.BACKEND,builder.BACKEND_SHA)==before

def test_old_backend_restored_after_exception(monkeypatch):
    from scripts import prepare_dh116_native_parameter_task as operation
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    before=builder.BACKEND,builder.BACKEND_SHA
    def fail(*a,**kw):raise RuntimeError("injected")
    monkeypatch.setattr(operation,"run_director",fail)
    with pytest.raises(RuntimeError,match="injected"):operation.run_existing_app()
    assert (builder.BACKEND,builder.BACKEND_SHA)==before

import sys
import test_dh116_dual_pinch_director_interval as director_tests
import test_dh116_dual_pinch_closure_interval as prior_test
from test_dh116_dual_pinch_builder import prepared,geometry
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder

def run_owner(monkeypatch,tmp_path,prepared,geometry,failure=None,old_guard=False):
    from scripts import prepare_dh116_native_parameter_task as operation
    calls=director_tests.install_cpu_directors(monkeypatch)
    old_install=prior_test.install_sdk_execution
    parameter_calls=[]
    def install(sim,rod,bindings,events):
        old_install(sim,rod,bindings,events)
        rod.rope.num_envs=1;rod.rope.fixed_root_orientation=False
        cfg=json.loads((reader.ROOT/reader.CONFIG).read_text())["rod"]
        rest=np.linalg.norm(np.diff(rod._base.nodes,axis=0),axis=1)[None]
        def capture():
            parameter_calls.append(sim.current_time_step_index)
            if sim.current_time_step_index==failure:raise ValueError("CPU injected parameter failure")
            return dict(schema="dh116.native_parameter_snapshot.v1",num_envs=1,nodes_per_env=41,time_s=rod.rope.time,
                metadata_source="native_host_metadata",segment_source="cuda_device_copy",hardware_measured=False,
                contact_radius_m=cfg["radius_m"],gravity_world_m_s2=np.array(cfg["gravity_world_m_s2"],np.float32),
                contact_mu_static=cfg["mu_s"],contact_mu_kinetic=cfg["mu_k"],contact_v_slip_m_s=cfg["v_slip_m_s"],
                contact_kappa=cfg["kappa"],rest_lengths_m=rest.copy(),stored_k_stretch_n_m=rest*cfg["axial_stiffness_n"],
                rest_quaternion_norm_n_m=4*cfg["bend_stiffness_n_m2"]/rest,
                stretch_ea_n=np.full((1,40),cfg["axial_stiffness_n"],np.float32),
                bend_ei_n_m2=np.full((1,40),cfg["bend_stiffness_n_m2"],np.float32),
                inverse_masses_kg_inv=rod.rope.inv_masses()[None])
        rod.rope.parameter_snapshot=capture
    monkeypatch.setattr(prior_test,"install_sdk_execution",install)
    monkeypatch.setattr(prior_test,"closure",operation)
    # The CPU fixture must instantiate the same explicit backend identity as
    # this independent experiment, before original builder setup.
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    with bindings((builder,"BACKEND",reader.BACKEND),(builder,"BACKEND_SHA",reader.BACKEND_SHA)):
        if failure is not None and not old_guard:
            with pytest.raises(RuntimeError,match="material_director"):
                prior_test.run_actual_episode(monkeypatch,tmp_path,prepared,geometry,False)
        else:prior_test.run_actual_episode(monkeypatch,tmp_path,prepared,geometry,old_guard)
    return tmp_path/"closure",calls,parameter_calls

def test_real_owner_reads_seventeen_complete_boundaries(monkeypatch,tmp_path,prepared,geometry):
    from scripts import prepare_dh116_native_parameter_task as operation
    out,d,p=run_owner(monkeypatch,tmp_path,prepared,geometry)
    assert d==p==list(range(2,19))
    checked=operation.inspect_interval(out/"physical.jsonl")
    assert checked["material_director_audit"]["valid_packets"]==17
    assert checked["material_director_audit"]["teacher_admission"] is False
    assert json.loads((out/"director_reset.json").read_text())["version"]==reader.VERSION
    assert json.loads((out/"after_stop.refs.json").read_text())["resources"]["material_director_reader"]["alive"] is False

@pytest.mark.parametrize("old_guard",[False,True])
def test_endpoint_failure_keeps_actual_terminal_and_old_guard(monkeypatch,tmp_path,prepared,geometry,old_guard):
    from scripts import prepare_dh116_native_parameter_task as operation
    out,d,p=run_owner(monkeypatch,tmp_path,prepared,geometry,failure=11,old_guard=old_guard)
    r=operation.inspect_interval(out/"physical.jsonl",complete=False)
    assert r["material_director_audit"]["valid_packets"]==9 and r["last_sdk_step"]==11
    assert d==p==list(range(2,12))
    assert r["terminal_reason"]==("post_step_source_clearance:right" if old_guard else "material_director_source_invalid")
    assert (out/"app_close_returned.json").exists()

def test_reset_read_failure_cannot_issue_first_action(monkeypatch,tmp_path,prepared,geometry):
    from scripts import prepare_dh116_native_parameter_task as operation
    out,d,p=run_owner(monkeypatch,tmp_path,prepared,geometry,failure=2)
    r=operation.inspect_interval(out/"physical.jsonl",complete=False)
    assert r["material_director_audit"]["valid_packets"]==0
    assert r["measured_physics_rows"]==0 and r["last_sdk_step"]==2
    assert d==p==[2]
    assert (out/"app_close_returned.json").exists()
