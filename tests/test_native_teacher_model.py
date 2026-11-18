"""Explicit synthetic context/rod CPU fixtures; no historical snapshot retagging."""
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from test_native_teacher_context import backend_profile, cpu_context, context_module
from test_dh116_flexible_measured_common_teacher import setup
from test_dh116_deformable_privileged_teacher import inputs

ROOT=Path(__file__).resolve().parents[1]


def model_module():
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_model
    return tabletop_native_teacher_model


def cpu_model(profile,grasp="pinch"):
    config,base,ledger,valid,old_rod,old_clock,old_mask=setup(grasp)
    # Generate a NEW synthetic 41-node rod with the actual profile identity.
    # Never relabel the historical (old_rod) fixture or recorded native data.
    _,rod,clock,mask=inputs(dict(backend_sha256=profile.binary_sha256),nodes=41)
    packet=cpu_context(profile,clock=clock)
    model=model_module().NativeContextTeacher(config,backend_profile=profile)
    return model,ledger,valid,rod,clock,mask,packet,config,base


def packed(profile):
    model,ledger,valid,rod,clock,mask,packet,*_=cpu_model(profile)
    return model,model.pack_cpu_fixture(ledger,valid,rod,clock,mask,packet)


def test_new_contract_exact_six_owned_inputs_and_independent_topology(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_teacher import DeformablePrivilegedTeacher
    from go2_pvcnn.control.dh116.tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher
    model,ledger,valid,rod,clock,mask,packet,config,base=cpu_model(backend_profile)
    obs=model.pack_cpu_fixture(ledger,valid,rod,clock,mask,packet)
    assert isinstance(model,DeformablePrivilegedTeacher)
    assert set(obs) == {"common371","common_valid","rod_nodes","rod_mask","task_context","task_context_valid"}
    assert len(model._parts(obs)) == 3
    assert obs["rod_nodes"].shape == (2,41,13)
    assert obs["common_valid"].sum() == 132
    assert torch.isnan(obs["common371"][~obs["common_valid"]]).all()
    assert torch.isfinite(model.deterministic(obs)).all()
    assert model.actor[0].in_features == 66 and model.actor_head.in_features == 448
    assert model.actor_context[0].in_features == 100 and model.actor_context[2].out_features == 64
    assert model.actor[0].weight.data_ptr() != model.critic[0].weight.data_ptr()
    assert model.actor_context[0].weight.data_ptr() != model.critic_context[0].weight.data_ptr()
    assert model.log_std.shape == (24,)
    before={k:v.clone() for k,v in obs.items()}
    ledger.zero_(); valid.zero_(); rod["node_positions"].zero_(); mask.zero_()
    packet.values.zero_(); packet.valid.zero_()
    for k,v in obs.items():
        torch.testing.assert_close(v,before[k],equal_nan=True)
    old=MeasuredCommonTeacher(config,base)
    with pytest.raises(ValueError): old._parts(obs)
    identity=model.model_identity
    assert identity["backend_sha256"] == backend_profile.binary_sha256
    assert identity["backend_profile_sha256"] == backend_profile.digest
    assert identity["input_keys"] == list(obs)
    assert identity["context_schema_sha256"] == context_module().CONTEXT_SCHEMA_SHA256
    assert identity["native_ready"] is False
    assert identity["raw_common_unknown"] == "NaN_false_305"
    assert identity["checkpoint_namespace"] != base["checkpoint_namespace"]


@pytest.mark.parametrize("damage", ["fabricated_na","missing_required","extra_valid","bad_mask",
    "missing_node","hidden_nan","bad_node","old_backend","context_nan","context_missing",
    "context_clock","context_episode","context_profile","extra_key","five_keys"])
def test_exact_raw_inputs_reject_bad_provenance_or_hidden_unavailable_nodes(backend_profile,damage):
    model,ledger,valid,rod,clock,mask,packet,*_=cpu_model(backend_profile)
    if damage == "fabricated_na": ledger[:,64]=0
    elif damage == "missing_required": ledger[:,0]=torch.nan;valid[:,0]=False
    elif damage == "extra_valid": valid[:,64]=True;ledger[:,64]=1
    elif damage == "bad_mask": mask[:,0]=False
    elif damage == "missing_node":
        rod["node_positions"]=rod["node_positions"][:,:40]
    elif damage == "hidden_nan":
        mask[:,0]=False;rod["node_positions"][:,0]=torch.nan
    elif damage == "bad_node": rod["velocities"][:,0]=torch.nan
    elif damage == "old_backend": rod["backend_sha256"]="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"
    elif damage == "context_nan": packet.values[:,12]=torch.nan
    elif damage == "context_missing": packet.valid[:,12]=False;packet.values[:,12]=torch.nan
    elif damage in ("context_clock","context_episode"):
        field={"physics_step":25} if damage == "context_clock" else {"episode_uid":"foreign"}
        ids=(replace(packet.metadata.identities[0],**field),packet.metadata.identities[1])
        packet=replace(packet,metadata=replace(packet.metadata,identities=ids))
    elif damage == "context_profile":
        packet=replace(packet,metadata=replace(packet.metadata,backend_profile_sha256="0"*64))
    else:
        obs=model.pack_cpu_fixture(ledger,valid,rod,clock,mask,packet)
        if damage == "extra_key": obs["metadata"]=torch.zeros(1)
        else: obs.pop("task_context_valid")
        with pytest.raises(ValueError): model.sample(obs)
        return
    with pytest.raises(ValueError):
        model.pack_cpu_fixture(ledger,valid,rod,clock,mask,packet)


def test_actor_gradients_reach_context_and_rod_and_tanh_distribution_is_original(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_squashed_gaussian import squashed_log_prob
    model,obs=packed(backend_profile)
    model.deterministic(obs).square().sum().backward()
    for branch in (model.actor_context,model.actor_rod):
        assert sum(float(p.grad.abs().sum()) for p in branch.parameters()) > 0
    sample=model.sample(obs)
    evaluated=model.evaluate_latents(obs,obs,sample["latent_actions"])
    torch.testing.assert_close(sample["log_prob"],squashed_log_prob(
        sample["latent_actions"],sample["latent_mean"],sample["latent_std"]))
    torch.testing.assert_close(evaluated["log_prob"],sample["log_prob"])
    assert torch.equal(sample["actions"],sample["latent_actions"].tanh())


def test_original_ppo_collect_record_frozen_audit_update_changes_finite_parameters(backend_profile,tmp_path):
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_checkpoint import save_checkpoint
    model,obs=packed(backend_profile)
    initial=copy.deepcopy(model.state_dict())
    trainer=DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),rollout_steps=2,num_envs=2)
    for _ in range(2):
        sample=trainer.collect_step(obs,obs)
        trainer.record_step(torch.tensor([1.,2.]),torch.tensor([True,True]),torch.zeros(2),
            executed_actions=sample["actions"],collection_token=sample["collection_token"])
    batch=trainer.rollout.batch()
    assert torch.isnan(batch["actor_obs"]["common371"][:,64]).all()
    trainer._check_frozen_policy(batch)
    metrics=trainer.update()
    assert metrics["samples"] == 4 and metrics["minibatch_updates"] == 1
    for branch in ("actor_rod","critic_rod","actor_context","critic_context"):
        assert any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items() if k.startswith(branch))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with pytest.raises(ValueError,match="identity"): save_checkpoint(tmp_path/"old-format.pt",trainer)
    assert not (tmp_path/"old-format.pt").exists()


def test_old_weights_checkpoint_identity_and_cross_grasp_rejected_before_loading(backend_profile,tmp_path):
    from go2_pvcnn.control.dh116.tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_teacher import verify_deformable_teacher_identity
    model,ledger,valid,rod,clock,mask,packet,config,base=cpu_model(backend_profile)
    old=MeasuredCommonTeacher(config,base)
    with pytest.raises((RuntimeError,ValueError)): model.load_state_dict(old.state_dict())
    with pytest.raises((RuntimeError,ValueError)): old.load_state_dict(model.state_dict())
    with pytest.raises(ValueError,match="identity"):
        verify_deformable_teacher_identity(model.model_identity,base)
    other,*_=cpu_model(backend_profile,"tripod")
    before=copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError,match="identity"): model.load_state_dict(other.state_dict())
    for k,v in model.state_dict().items():
        if isinstance(v,torch.Tensor): assert torch.equal(v,before[k])
    model_module().verify_native_teacher_identity(model.model_identity,model.model_identity)
    with pytest.raises(ValueError,match="identity"):
        model_module().verify_native_teacher_identity(base,model.model_identity)


def test_historical_model_deterministic_outputs_and_source_remain_unchanged(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher
    config,base,ledger,valid,rod,clock,mask=setup()
    torch.manual_seed(913)
    old=MeasuredCommonTeacher(config,base)
    obs=old.pack_measured(ledger,valid,rod,clock,mask)
    before=old.deterministic(obs).detach().clone()
    model,newobs=packed(backend_profile)
    model.sample(newobs)
    assert torch.equal(before,old.deterministic(obs))
    for key,sha in {
        "tabletop_deformable_privileged_teacher.py":"f5ee314badf045c27b7efc44cc32e03e0b82f8199b35726f0a81e5e63722c872",
        "tabletop_flexible_measured_common_teacher.py":"724650f30d550838c23e3b7c84c423df93fcbf5a746004f2708afec24e2b28e9",
        "tabletop_privileged_ppo.py":"f0a96f4f8c9548b5153d004013bef9d77b97250aa0bc43272626cffb15416854"}.items():
        import hashlib
        assert hashlib.sha256((ROOT/"src/go2_pvcnn/control/dh116"/key).read_bytes()).hexdigest() == sha


def test_native_pack_entry_rejects_historical_source_without_historical_adapter(backend_profile,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_actual_backend_teacher as historical
    def forbidden(*args,**kwargs): raise AssertionError("historical actual-input adapter must never be called")
    monkeypatch.setattr(historical,"_actual_inputs",forbidden)
    packet=cpu_context(backend_profile,batch=1)
    with pytest.raises(ValueError):
        model_module().pack_native_inputs({}, {}, {}, packet, backend_profile=backend_profile)

def raw_cpu_packets(profile):
    """NEW ABI synthetic packets. Every data packet stays explicitly CPU scoped."""
    import numpy as np
    from test_dh116_teacher_mounted_wrist_source import clock,capture,hand
    from go2_pvcnn.control.dh116.tabletop_teacher_mounted_wrist_source import adapt_mounted_hands
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import VERSION as runtime_version
    from go2_pvcnn.control.dh116.tabletop_deformable_rod_boundary import RodBoundaryState
    from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    c=clock(2)
    common=adapt_mounted_hands([capture(hand(s),s,read_identity=lambda:c) for s in ("left","right")],
        expected_identity=c)
    keys=("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick",
        "timestamp_s","solver_time_s","clock_valid")
    raw=dict(version=runtime_version,source_scope="cpu_fixture",**{k:c[k] for k in keys},
        node_positions=np.zeros((1,41,3)),velocities=np.zeros((1,41,3)),
        rest_lengths_m=np.full(40,.025),contact_valid=False,distributed_contact=None,
        rod_mesh_contact_pairs=None,hand_wrenches_com_world=None,table_wrench_world=None)
    raw["node_positions"][0,:,1]=np.linspace(0,1,41)
    raw["node_positions"][...,2]=.31
    class CPUClock:
        def clock(self): return {k:raw[k] for k in keys}
    boundary=RodBoundaryState(CPUClock(),profile.binary_path,profile.binary_sha256).capture(raw)
    import json
    cfg=json.loads((ROOT/source.CONFIG).read_text())["rod"]
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import lumped_masses
    rest=np.full((1,40),.025)
    raw_parameters=dict(schema="dh116.native_parameter_snapshot.v1",num_envs=1,nodes_per_env=41,
        time_s=c["solver_time_s"],metadata_source="native_host_metadata",segment_source="cuda_device_copy",
        hardware_measured=False,contact_radius_m=cfg["radius_m"],
        gravity_world_m_s2=np.array(cfg["gravity_world_m_s2"]),
        contact_mu_static=cfg["mu_s"],contact_mu_kinetic=cfg["mu_k"],
        contact_v_slip_m_s=cfg["v_slip_m_s"],contact_kappa=cfg["kappa"],
        rest_lengths_m=rest,stored_k_stretch_n_m=rest*cfg["axial_stiffness_n"],
        rest_quaternion_norm_n_m=4*cfg["bend_stiffness_n_m2"]/rest,
        stretch_ea_n=np.full((1,40),cfg["axial_stiffness_n"]),
        bend_ei_n_m2=np.full((1,40),cfg["bend_stiffness_n_m2"]),
        inverse_masses_kg_inv=(1/lumped_masses(raw["node_positions"][0],cfg["linear_density_kg_m"]))[None])
    native=validate_snapshot(raw_parameters,expected_num_envs=1,expected_nodes=41,
        expected_time_s=c["solver_time_s"])
    parameters=dict(version=source.PARAMETER_VERSION,source_scope="cpu_fixture",sources=source.source_basis(),
        identity=c,raw_native_parameters=raw_parameters,native_parameters=native,new_sdk_steps=0,
        new_solver_steps=0,teacher_admission=False,task_physics_qualified=False,
        node_positions_m=raw["node_positions"].copy(),reward_radius_sample=dict(value=cfg["radius_m"],source=dict(
        api="RopeBatch.parameter_snapshot.contact_radius_m",provenance="native_host_metadata",
        backend_sha256=profile.binary_sha256,identity=c,uncertainty=None,
        hardware_measured=False,task_physics_qualified=False)))
    sample_clock=dict(env_ids=[c["env_id"]],episode_uids=[c["episode_uid"]],reset_ids=[c["reset_id"]],
        rod_uids=[c["rod_uid"]],physics_step=c["physics_step"],rod_tick=c["rod_tick"],timestamp_s=c["timestamp_s"])
    packet=cpu_context(profile,batch=1,clock=sample_clock)
    packet=replace(packet,metadata=replace(packet.metadata,
        identities=(context_module().ContextIdentity.from_clock(c),)))
    return dict(common=common,rod_boundary=boundary),raw,parameters,packet


def test_current_abi_raw_pack_reconstructs_cpu_data_without_claiming_native(backend_profile,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_actual_backend_teacher as historical
    monkeypatch.setattr(historical,"_actual_inputs",lambda *a,**k:pytest.fail("historical adapter"))
    pre,raw,parameters,packet=raw_cpu_packets(backend_profile)
    obs=model_module().pack_native_inputs(pre,raw,parameters,packet,backend_profile=backend_profile)
    assert obs["rod_nodes"].shape == (1,41,13)
    assert packet.metadata.scope == raw["source_scope"] == parameters["source_scope"] == "cpu_fixture"
    assert packet.native_qualified is False
    assert torch.isnan(obs["common371"][~obs["common_valid"]]).sum() == 305
    assert not torch.isnan(obs["rod_nodes"]).any()
    raw["node_positions"][...,2]=99
    assert (obs["rod_nodes"][...,2] < 1).all()


@pytest.mark.parametrize("damage",["old_parameter_source","wrong_clock","bad_radius","raw_bad_node","old_boundary"])
def test_raw_pack_refuses_historical_source_clock_and_reconstruction_changes(backend_profile,damage):
    pre,raw,parameters,packet=raw_cpu_packets(backend_profile)
    if damage == "old_parameter_source": parameters["sources"]["backend_sha256"]="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"
    elif damage == "wrong_clock": raw["physics_step"] += 1
    elif damage == "bad_radius": parameters["reward_radius_sample"]["source"]["backend_sha256"]="0"*64
    elif damage == "raw_bad_node": raw["node_positions"][0,0,0]=float("nan")
    else: pre["rod_boundary"]["rod_snapshot"]["backend_sha256"]="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"
    with pytest.raises(ValueError):
        model_module().pack_native_inputs(pre,raw,parameters,packet,backend_profile=backend_profile)

def test_deepcopy_and_weights_only_state_roundtrip_preserve_new_model_topology(backend_profile,tmp_path):
    model,obs=packed(backend_profile)
    candidate=copy.deepcopy(model)
    assert candidate.model_identity == model.model_identity
    assert candidate._backend_profile == model._backend_profile
    assert candidate.actor_context[0].weight.data_ptr() != model.actor_context[0].weight.data_ptr()
    path=tmp_path/"cpu-only-state.pt"
    torch.save(model.state_dict(),path)
    state=torch.load(path,map_location="cpu",weights_only=True)
    candidate.load_state_dict(state)
    torch.testing.assert_close(candidate.deterministic(obs),model.deterministic(obs))
    torch.testing.assert_close(candidate.value(obs),model.value(obs))
    wrong=copy.deepcopy(state)
    wrong["_extra_state"]["backend_sha256"]="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"
    before=candidate.actor[0].weight.detach().clone()
    with pytest.raises(ValueError,match="identity"): candidate.load_state_dict(wrong)
    assert torch.equal(candidate.actor[0].weight,before)

@pytest.mark.parametrize("damage",["changed_gravity","changed_constitutive","pinned_node"])
def test_raw_pack_rejects_self_consistent_parameters_outside_locked_physics(backend_profile,damage):
    from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot
    pre,raw,parameters,packet=raw_cpu_packets(backend_profile)
    p=parameters["raw_native_parameters"]
    if damage == "changed_gravity": p["gravity_world_m_s2"]*=.5
    elif damage == "changed_constitutive":
        p["stored_k_stretch_n_m"]*=2;p["stretch_ea_n"]*=2
    else: p["inverse_masses_kg_inv"][0,0]=0
    parameters["native_parameters"]=validate_snapshot(p,expected_num_envs=1,expected_nodes=41,
        expected_time_s=p["time_s"])
    with pytest.raises(ValueError):
        model_module().pack_native_inputs(pre,raw,parameters,packet,backend_profile=backend_profile)
