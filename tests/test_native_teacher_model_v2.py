"""Independent v2 CPU fixtures; no native producer or checkpoint adapter."""
import copy
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_native_teacher_context_v2 import backend_profile
from test_dh116_flexible_measured_common_teacher import setup
from test_dh116_deformable_privileged_teacher import inputs

ROOT = Path(__file__).resolve().parents[1]


def module():
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_model_v2
    return tabletop_native_teacher_model_v2


def context_module():
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_context_v2
    return tabletop_native_teacher_context_v2


def config():
    return json.loads((ROOT/"configs/dh116_training/native_fixed_s0_pinch_v2.json").read_text())


def cpu_context_v2(profile, batch=2, clock=None):
    """Create a context51 fixture directly; never convert a context50 packet."""
    m = context_module()
    clock = clock or dict(env_ids=list(range(batch)), episode_uids=[f"ep-{i}" for i in range(batch)],
        reset_ids=[0]*batch, rod_uids=["rod0"]*batch, physics_step=24, rod_tick=288, timestamp_s=.05)
    values = torch.zeros(batch, 51, dtype=torch.float32)
    valid = torch.ones(batch, 51, dtype=torch.bool)
    values[:, 0] = 1.
    values[:, 36:43] = torch.tensor([.31,.31,.02,.02,.03,5.,.0005])
    values[:, 44] = 1.
    excluded = (8,9,46)
    values[:, excluded] = torch.nan
    valid[:, excluded] = False
    identities = tuple(m.ContextIdentity(
        run_uuid="cpu-fixture", env_id=clock["env_ids"][i], episode_uid=clock["episode_uids"][i],
        reset_id=clock["reset_ids"][i], rod_uid=clock["rod_uids"][i],
        physics_step=clock["physics_step"], rod_tick=clock["rod_tick"],
        timestamp_s=clock["timestamp_s"], solver_time_s=clock["rod_tick"]/5760)
        for i in range(batch))
    applicable = tuple(i not in excluded for i in range(51))
    meta = m.ContextMetadataV2(owner_uid="v2-cpu-fixture", identities=identities,
        backend_profile_sha256=profile.digest, scope="cpu_fixture", mode="fixed_s0_v2",
        sources=tuple(f["source"] for f in m.CONTEXT_FIELDS_V2), applicable=applicable,
        reasons=tuple(None if x else "explicit unanchored or disabled deadline" for x in applicable),
        reference_uids=(("target_material","v2-cpu-fixed-target"),),
        previous_request_token="v2-cpu-reset-receipt", phase_contract_sha256=m.PHASE_CONTRACT_SHA256_V2,
        fixed_s0_manifest_sha256="1"*64, prelift_scale_s=.25)
    return m.TaskContextPacketV2(values, valid, meta)


def cpu_model(profile):
    _,_,ledger,valid,*_ = setup()
    _,rod,clock,mask = inputs(dict(backend_sha256=profile.binary_sha256), nodes=41)
    packet = cpu_context_v2(profile, clock=clock)
    model = module().NativeContextTeacherV2(config(), backend_profile=profile)
    return model, ledger, valid, rod, clock, mask, packet


def packed(profile):
    model,*args = cpu_model(profile)
    return model, model.pack_cpu_fixture_v2(*args)


def test_v2_keeps_exact_six_input_keys():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import INPUT_KEYS
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile_v2 import INPUT_KEYS_V2,CONTEXT_WIDTH_V2
    assert tuple(INPUT_KEYS_V2) == tuple(INPUT_KEYS)
    assert CONTEXT_WIDTH_V2 == 51


def test_contract_owns_new_versions_sha_and_keeps_real_source_profile(backend_profile):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_profile as old
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_profile_v2 as new
    model,obs = packed(backend_profile)
    identity = model.model_identity
    assert model._backend_profile is backend_profile
    assert backend_profile.model_schema == old.MODEL_SCHEMA_VERSION != new.MODEL_SCHEMA_VERSION_V2
    for name in ("model","input","context","phase"):
        assert identity[name+"_schema"] == getattr(new,name.upper()+"_SCHEMA_VERSION_V2")
        assert identity[name+"_schema_sha256"] == getattr(new,name.upper()+"_SCHEMA_SHA256_V2")
        assert len(identity[name+"_schema_sha256"]) == 64
    assert identity["fixed_s0_manifest_sha256"] is None
    assert identity["prelift_stable_s"] == identity["prelift_scale_s"] == .25
    assert identity["backend_profile_sha256"] == backend_profile.digest
    assert identity["context_size"] == 51 and identity["context_encoded_size"] == 102
    assert identity["architecture"]["context"] == [102,64,64]
    assert identity["architecture"]["fusion"] == 448
    assert identity["phase_source"] == "execution_phase_owner_v2"
    assert identity["native_ready"] is identity["checkpoint_adapter_ready"] is False
    assert identity["context_owner_native_connected"] is False
    assert identity["raw_common_unknown"] == "NaN_false_305"
    assert set(obs) == set(new.INPUT_KEYS_V2)
    assert obs["rod_nodes"].shape == (2,41,13)
    assert obs["common_valid"].sum() == 132
    assert torch.isnan(obs["common371"][~obs["common_valid"]]).sum() == 610
    assert model.actor[0].in_features == 66
    assert model.actor_context[0].in_features == model.critic_context[0].in_features == 102
    assert model.actor_head.in_features == model.critic_head.in_features == 448
    for a,b in ((model.actor,model.critic),(model.actor_context,model.critic_context),
                (model.actor_rod,model.critic_rod)):
        assert next(a.parameters()).data_ptr() != next(b.parameters()).data_ptr()
    assert model.log_std.shape == (24,)
    assert all(identity["dependency_source_sha256"].values())


def test_packed_inputs_and_packet_owned_snapshots(backend_profile):
    model,ledger,valid,rod,clock,mask,packet = cpu_model(backend_profile)
    owned = packet.owned()
    obs = model.pack_cpu_fixture_v2(ledger,valid,rod,clock,mask,packet)
    before = {k:v.clone() for k,v in obs.items()}
    ledger.zero_();valid.zero_();rod["node_positions"].zero_();mask.zero_()
    packet.values.zero_();packet.valid.zero_()
    for k,v in obs.items():
        torch.testing.assert_close(v,before[k],equal_nan=True)
    torch.testing.assert_close(owned.values,before["task_context"],equal_nan=True)
    assert torch.equal(owned.valid,before["task_context_valid"])
    assert owned.metadata is not packet.metadata


def test_actor_and_critic_actually_sense_index50_values_and_gradients(backend_profile):
    torch.manual_seed(925)
    model,obs = packed(backend_profile)
    before_actor = model.deterministic(obs).detach()
    before_critic = model.value(obs).detach()
    changed = {k:v.clone() for k,v in obs.items()}
    changed["task_context"][:,50] = .5
    assert not torch.allclose(model.deterministic(changed),before_actor,rtol=0,atol=1e-7)
    assert not torch.allclose(model.value(changed),before_critic,rtol=0,atol=1e-7)
    for branch in ("actor","critic"):
        context = changed["task_context"].detach().clone().requires_grad_()
        candidate = dict(changed,task_context=context)
        output = model.deterministic(candidate) if branch == "actor" else model.value(candidate)
        grad, = torch.autograd.grad(output.square().sum(),context)
        assert torch.isfinite(grad[:,50]).all() and bool((grad[:,50].abs()>0).all())


@pytest.mark.parametrize("damage",["old50","controller_state","five_keys","context_nan",
    "hidden_unknown","common_filled","common_extra_valid","missing_node","hidden_bad_node",
    "wrong_frame","old_backend","clock","episode","profile","phase_source","old_packet","scale"])
def test_exact_v2_inputs_and_fixture_provenance_rejections(backend_profile,damage):
    model,ledger,valid,rod,clock,mask,packet = cpu_model(backend_profile)
    if damage in ("old50","controller_state","five_keys"):
        obs = model.pack_cpu_fixture_v2(ledger,valid,rod,clock,mask,packet)
        if damage == "old50":
            obs["task_context"] = obs["task_context"][:,:50]
            obs["task_context_valid"] = obs["task_context_valid"][:,:50]
        elif damage == "controller_state": obs["controller_state"] = torch.zeros(2,50)
        else: obs.pop("task_context_valid")
        with pytest.raises(ValueError): model.sample(obs)
        return
    if damage == "context_nan": packet.values[:,50] = torch.nan
    elif damage == "hidden_unknown": packet.values[:,8] = 0
    elif damage == "common_filled": ledger[:,64] = 0
    elif damage == "common_extra_valid": valid[:,64] = True;ledger[:,64] = 1
    elif damage == "missing_node": rod["node_positions"] = rod["node_positions"][:,:40]
    elif damage == "hidden_bad_node": mask[:,0] = False;rod["velocities"][:,0] = torch.nan
    elif damage == "wrong_frame": rod["coordinate_frame"] = "camera"
    elif damage == "old_backend": rod["backend_sha256"] = "0"*64
    elif damage in ("clock","episode"):
        field = {"physics_step":25} if damage == "clock" else {"episode_uid":"foreign"}
        packet = replace(packet,metadata=replace(packet.metadata,
            identities=(replace(packet.metadata.identities[0],**field),packet.metadata.identities[1])))
    elif damage == "profile": packet = replace(packet,metadata=replace(packet.metadata,backend_profile_sha256="0"*64))
    elif damage == "phase_source":
        sources = list(packet.metadata.sources);sources[0] = "old.task.step"
        packet = replace(packet,metadata=replace(packet.metadata,sources=tuple(sources)))
    elif damage == "scale": packet = replace(packet,metadata=replace(packet.metadata,prelift_scale_s=.5))
    else:
        from test_native_teacher_context import cpu_context
        packet = cpu_context(backend_profile)
    with pytest.raises(ValueError):
        model.pack_cpu_fixture_v2(ledger,valid,rod,clock,mask,packet)


def test_old_and_new_extra_state_rejected_both_directions_before_weight_mutation(backend_profile):
    from test_native_teacher_model import cpu_model as old_cpu_model
    old,*_ = old_cpu_model(backend_profile)
    new,*_ = cpu_model(backend_profile)
    for destination,source in ((new,old),(old,new)):
        before = copy.deepcopy(destination.state_dict())
        with pytest.raises(ValueError,match="identity"):
            destination.load_state_dict(source.state_dict())
        with pytest.raises(ValueError,match="identity"):
            destination.set_extra_state(source.get_extra_state())
        for key,value in destination.state_dict().items():
            if isinstance(value,torch.Tensor): assert torch.equal(value,before[key])


@pytest.mark.parametrize("field",["model_schema","input_schema","context_schema","phase_schema",
    "phase_contract_sha256","fixed_s0_manifest_sha256","backend_profile_sha256","prelift_scale_s"])
def test_identity_tampering_rejected_before_loading(backend_profile,field):
    model,_ = packed(backend_profile)
    state = copy.deepcopy(model.state_dict())
    state["_extra_state"][field] = "forged"
    before = model.actor[0].weight.detach().clone()
    with pytest.raises(ValueError,match="identity"): model.load_state_dict(state)
    assert torch.equal(model.actor[0].weight,before)


def test_weights_only_roundtrip_and_c05_remains_exact_type_closed(backend_profile,tmp_path):
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    from go2_pvcnn.control.dh116.tabletop_native_teacher_checkpoint import checkpoint_identity, save_checkpoint
    model,obs = packed(backend_profile)
    path = tmp_path/"fixture-weights.pt"
    torch.save(model.state_dict(),path)
    candidate = copy.deepcopy(model)
    candidate.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    torch.testing.assert_close(candidate.deterministic(obs),model.deterministic(obs))
    torch.testing.assert_close(candidate.value(obs),model.value(obs))
    trainer = DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),rollout_steps=2,num_envs=2)
    with pytest.raises(ValueError): checkpoint_identity(trainer,{})
    with pytest.raises(ValueError): save_checkpoint(tmp_path/"native-checkpoint.pt",trainer,{})
    assert not (tmp_path/"native-checkpoint.pt").exists()


def test_original_tanh_and_cpu_ppo_update_have_real_finite_losses(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    from go2_pvcnn.control.dh116.tabletop_squashed_gaussian import squashed_log_prob
    model,obs = packed(backend_profile)
    sample = model.sample(obs)
    assert torch.equal(sample["actions"],sample["latent_actions"].tanh())
    evaluated = model.evaluate_latents(obs,obs,sample["latent_actions"])
    torch.testing.assert_close(sample["log_prob"],squashed_log_prob(
        sample["latent_actions"],sample["latent_mean"],sample["latent_std"]))
    torch.testing.assert_close(evaluated["log_prob"],sample["log_prob"])
    before = copy.deepcopy(model.state_dict())
    trainer = DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),rollout_steps=2,num_envs=2)
    for _ in range(2):
        sample = trainer.collect_step(obs,obs)
        trainer.record_step(torch.tensor([1.,2.]),torch.tensor([True,True]),torch.zeros(2),
            executed_actions=sample["actions"],collection_token=sample["collection_token"])
    batch = trainer.rollout.batch()
    assert torch.isnan(batch["actor_obs"]["common371"][:,64]).all()
    trainer._check_frozen_policy(batch)
    metrics = trainer.update()
    assert metrics["samples"] == 4 and metrics["minibatch_updates"] == 1
    assert all(math.isfinite(metrics[k]) for k in ("policy_loss","value_loss","entropy","kl"))
    for branch in ("actor_context","critic_context","actor_rod","critic_rod"):
        assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith(branch))
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize("damage",["none_adapter","bool_adapter","duck_adapter","missing_snapshot",
    "forged_witness","wrong_clock_witness","typed_receipt_only"])
def test_production_pack_cannot_admit_caller_flags_or_fixture_receipts(backend_profile,damage):
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseOwnerV2
    packet = cpu_context_v2(backend_profile,batch=1)
    phase = PhaseOwnerV2(prelift_stable_s=.25)
    receipt = phase.reset(episode_uid=packet.metadata.identities[0].episode_uid,physics_tick=24)
    adapter = None if damage == "none_adapter" else True if damage == "bool_adapter" else SimpleNamespace(
        source_scope="native",native_ready=True,
        validate_phase_source_witness_v2=lambda **k:pytest.fail("caller validator must not be trusted"))
    snapshot = None if damage == "missing_snapshot" else SimpleNamespace(native_ready=True)
    if damage == "wrong_clock_witness": receipt = replace(receipt,physics_tick=25)
    attempted_context = packet
    if damage in ("forged_witness","wrong_clock_witness"):
        witness = SimpleNamespace(source_adapter=adapter,measurement_snapshot=snapshot,
            phase_receipt=receipt,source_sha256="1"*64,native_ready=True)
        attempted_context = SimpleNamespace(values=packet.values,valid=packet.valid,
            metadata=SimpleNamespace(scope="native",phase_source_witness=witness))
    expected = "full_v2_phase_source_adapter_missing" if adapter is None else "full_v2_phase_source_witness_missing"
    with pytest.raises(ValueError,match=expected):
        module().pack_native_inputs_v2({}, {}, {}, attempted_context, backend_profile=backend_profile,
            source_adapter=adapter,measurement_snapshot=snapshot,phase_receipt=receipt)
    assert packet.native_qualified is False


@pytest.mark.parametrize("field,value",[("profile","old"),("native_ready",True),
    ("checkpoint_adapter_ready",True),("prelift_stable_s",True),("prelift_scale_s",0),
    ("resolved_reset_manifest",{"sha256":"1"*64})])
def test_config_requires_explicit_candidate_without_fabricated_s0(backend_profile,field,value):
    candidate = config();candidate[field] = value
    with pytest.raises(ValueError): module().NativeContextTeacherV2(candidate,backend_profile=backend_profile)

def raw_cpu_packets_v2(profile):
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
    packet=cpu_context_v2(profile,batch=1,clock=sample_clock)
    packet=replace(packet,metadata=replace(packet.metadata,
        identities=(context_module().ContextIdentity.from_clock(c),)))
    return dict(common=common,rod_boundary=boundary),raw,parameters,packet



def test_private_raw_fixture_path_validates_full_current_abi_and_owned_geometry(backend_profile):
    pre,raw,parameters,packet = raw_cpu_packets_v2(backend_profile)
    obs = module()._pack_raw_inputs_v2(pre,raw,parameters,packet,backend_profile=backend_profile)
    assert obs["rod_nodes"].shape == (1,41,13)
    assert packet.values.shape == (1,51) and packet.native_qualified is False
    assert packet.metadata.scope == raw["source_scope"] == parameters["source_scope"] == "cpu_fixture"
    assert torch.isnan(obs["common371"][~obs["common_valid"]]).sum() == 305
    raw["node_positions"][...,2] = 99
    assert (obs["rod_nodes"][...,2]<1).all()


@pytest.mark.parametrize("damage",["parameter_source","raw_clock","parameter_clock","radius",
    "bad_node","boundary_frame","gravity","constitutive","pinned","phase_source","context_clock",
    "native_retag"])
def test_private_raw_fixture_validation_independent_negative_sources(backend_profile,damage):
    from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot
    pre,raw,parameters,packet = raw_cpu_packets_v2(backend_profile)
    if damage == "parameter_source": parameters["sources"]["backend_sha256"] = "0"*64
    elif damage == "raw_clock": raw["physics_step"] += 1
    elif damage == "parameter_clock": parameters["identity"] = dict(parameters["identity"],rod_tick=999)
    elif damage == "radius": parameters["reward_radius_sample"]["source"]["backend_sha256"] = "0"*64
    elif damage == "bad_node": raw["node_positions"][0,0,0] = float("nan")
    elif damage == "boundary_frame": pre["rod_boundary"]["rod_snapshot"]["coordinate_frame"] = "camera"
    elif damage in ("gravity","constitutive","pinned"):
        p = parameters["raw_native_parameters"]
        if damage == "gravity": p["gravity_world_m_s2"] *= .5
        elif damage == "constitutive": p["stored_k_stretch_n_m"] *= 2;p["stretch_ea_n"] *= 2
        else: p["inverse_masses_kg_inv"][0,0] = 0
        parameters["native_parameters"] = validate_snapshot(p,expected_num_envs=1,expected_nodes=41,
            expected_time_s=p["time_s"])
    elif damage == "phase_source":
        sources = list(packet.metadata.sources);sources[0] = "old.task.step"
        packet = replace(packet,metadata=replace(packet.metadata,sources=tuple(sources)))
    elif damage == "context_clock":
        packet = replace(packet,metadata=replace(packet.metadata,
            identities=(replace(packet.metadata.identities[0],physics_step=25),)))
    else: raw["source_scope"] = parameters["source_scope"] = "native"
    with pytest.raises(ValueError):
        module()._pack_raw_inputs_v2(pre,raw,parameters,packet,backend_profile=backend_profile)


def fixture_source_digest(ledger,valid,rod,clock,mask,packet):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    return _digest(dict(ledger=ledger,valid=valid,rod=rod,clock=clock,mask=mask,
        packet=dict(values=packet.values,valid=packet.valid,metadata=asdict(packet.metadata))))


def resize_fixture_context(packet,batch):
    """The exact legal N1/N3 contexts from the independent QUALITY counterexamples."""
    assert len(packet.values) == 2
    if batch == 1:
        return replace(packet,values=packet.values[:1].clone(),valid=packet.valid[:1].clone(),
            metadata=replace(packet.metadata,identities=packet.metadata.identities[:1]))
    assert batch == 3
    third = replace(packet.metadata.identities[0],env_id=99,episode_uid="extra")
    return replace(packet,values=torch.cat([packet.values,packet.values[:1]],0),
        valid=torch.cat([packet.valid,packet.valid[:1]],0),
        metadata=replace(packet.metadata,identities=(*packet.metadata.identities,third)))


def fixture_pack(entrypoint,model,args,profile):
    if entrypoint == "model":
        return model.pack_cpu_fixture_v2(*args)
    return module().pack_cpu_fixture_v2(*args,backend_profile=profile)


@pytest.mark.parametrize("entrypoint,context_batch",[("model",1),("model",3),("module",1),("module",3)],
    ids=["model-n1","model-n3","module-n1","module-n3"])
def test_context_batch_mismatch_rejected_before_clock_or_export(backend_profile,entrypoint,context_batch):
    model,*args = cpu_model(backend_profile)
    args[-1] = resize_fixture_context(args[-1],context_batch)
    # The context is internally valid; the common/rod batch is independently N2.
    context_module().validate_context_v2(args[-1],backend_profile=backend_profile)
    before = fixture_source_digest(*args)
    try:
        with pytest.raises(ValueError,match="context.*batch"):
            fixture_pack(entrypoint,model,args,backend_profile)
    finally:
        assert fixture_source_digest(*args) == before


@pytest.mark.parametrize("entrypoint",["model","module"])
@pytest.mark.parametrize("damage",["valid_n1","valid_n3","identities_n1","identities_n3"])
def test_context_validity_and_identity_batch_mismatches_do_not_export(
        backend_profile,entrypoint,damage):
    model,*args = cpu_model(backend_profile)
    packet = args[-1]
    resized = resize_fixture_context(packet,1 if damage.endswith("n1") else 3)
    if damage.startswith("valid"):
        args[-1] = replace(packet,valid=resized.valid)
    else:
        args[-1] = replace(packet,metadata=replace(packet.metadata,
            identities=resized.metadata.identities))
    before = fixture_source_digest(*args)
    try:
        with pytest.raises(ValueError):
            fixture_pack(entrypoint,model,args,backend_profile)
    finally:
        assert fixture_source_digest(*args) == before


@pytest.mark.parametrize("entrypoint",["model","module"])
def test_matching_batch_two_exports_owned_inputs_and_finite_actor_critic(backend_profile,entrypoint):
    model,*args = cpu_model(backend_profile)
    before = fixture_source_digest(*args)
    obs = fixture_pack(entrypoint,model,args,backend_profile)
    assert fixture_source_digest(*args) == before
    expected = dict(common371=(2,371),common_valid=(2,371),rod_nodes=(2,41,13),
        rod_mask=(2,41),task_context=(2,51),task_context_valid=(2,51))
    assert {k:tuple(v.shape) for k,v in obs.items()} == expected
    assert model.deterministic(obs).shape == (2,24) and torch.isfinite(model.deterministic(obs)).all()
    assert model.value(obs).shape == (2,) and torch.isfinite(model.value(obs)).all()
    for value in obs.values():
        value.zero_()
    assert fixture_source_digest(*args) == before


@pytest.mark.parametrize("context_batch",[2,3])
def test_raw_cpu_path_rejects_context_batch_mismatch_before_export(backend_profile,context_batch):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    pre,raw,parameters,base = raw_cpu_packets_v2(backend_profile)
    # Keep the matching N1 clock as row zero; only extra valid rows cause rejection.
    identities = (base.metadata.identities[0],) + tuple(
        replace(base.metadata.identities[0],env_id=99+i,episode_uid="extra-"+str(i))
        for i in range(1,context_batch))
    packet = replace(base,values=base.values.repeat(context_batch,1),
        valid=base.valid.repeat(context_batch,1),
        metadata=replace(base.metadata,identities=identities))
    context_module().validate_context_v2(packet,backend_profile=backend_profile)
    def stamp():
        return _digest(dict(pre=pre,raw=raw,parameters=parameters,packet=dict(
            values=packet.values,valid=packet.valid,metadata=asdict(packet.metadata))))
    before = stamp()
    try:
        with pytest.raises(ValueError,match="context.*identity"):
            module()._pack_raw_inputs_v2(pre,raw,parameters,packet,backend_profile=backend_profile)
    finally:
        assert stamp() == before
