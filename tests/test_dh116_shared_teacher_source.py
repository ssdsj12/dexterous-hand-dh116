"""Shared parent/row provenance and translation; fixtures never grant native scope."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_native_teacher_context import backend_profile, cpu_context
from test_native_teacher_model import raw_cpu_packets
from go2_pvcnn.control.dh116.tabletop_shared_rod_owner import SharedRodOwner


def fixture(old_profile, profile, offsets=((0., 0., 0.), (3., -6., 0.))):
    from go2_pvcnn.control.dh116.tabletop_teacher_mounted_wrist_source import adapt_mounted_hands
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import ContextIdentity
    common, contexts, nodes, raw = [], [], [], []
    for i, offset in enumerate(offsets):
        pre, rod, parameter, _ = raw_cpu_packets(old_profile, env_id=i)
        packets = deepcopy(pre["common"]["source_packets"])
        old_origin = np.asarray(packets[0]["reset_seed"]["environment_origin_w"])
        delta = np.asarray(offset) - old_origin
        for packet in packets:
            packet["body_link_pose_w"][:3] = (np.asarray(packet["body_link_pose_w"][:3]) + delta).tolist()
            seed = packet["reset_seed"]
            seed["environment_origin_w"] = list(offset)
            authored = np.array(seed["authored_hand_roots_w"])
            authored[:, :3] += offset
            authored[:, :2] += seed["root_xy"]
            seed["root_seed_w"] = authored.tolist()
        c = adapt_mounted_hands(packets, expected_identity=pre["common"]["identity"])
        common.append(c)
        context = cpu_context(profile, batch=1)
        context = replace(context, metadata=replace(context.metadata,
            identities=(ContextIdentity.from_clock(c["identity"]),), grasp="pinch"))
        contexts.append(context)
        # Binary-exact relative positions separate origin math from unavoidable
        # float32 World storage quantization; the latter is tested independently.
        line = np.zeros((41,3))
        line[:,1] = np.arange(41)/64
        nodes.append(line + offset)
        raw.append(parameter["raw_native_parameters"])
    parameter = deepcopy(raw[0])
    parameter["num_envs"] = len(raw)
    for name in ("rest_lengths_m", "stored_k_stretch_n_m", "rest_quaternion_norm_n_m",
                 "stretch_ea_n", "bend_ei_n_m2", "inverse_masses_kg_inv"):
        parameter[name] = np.concatenate([item[name] for item in raw])
    class Solver:
        num_envs = len(raw)
        verts_per_rope = 41
        time = 4.
        last_error = 0
        calls = 0
        def positions(self):
            return torch.tensor(np.array(nodes), dtype=torch.float32)
        def velocities(self):
            return torch.zeros((len(raw), 41, 3))
        def parameter_snapshot(self):
            self.calls += 1
            return deepcopy(parameter)
    solver = Solver()
    owner = SharedRodOwner(solver, run_uuid="fixture", physics_step=2,
        episodes=tuple("ep"+str(i) for i in range(len(raw))),
        mesh_names=("pad", "table"), solver_substeps=12, device="cpu")
    return owner, SimpleNamespace(current_time_step_index=2, current_time=2/480), common, contexts


def reader(old_profile):
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_source import SharedBackendProfile, SharedTeacherSource
    profile = SharedBackendProfile.load()
    runtime, sim, common, context = fixture(old_profile, profile)
    source = SharedTeacherSource(runtime, sim, backend_profile=profile,
        env_origins=np.array([[0.,0.,0.],[3.,-6.,0.]]), scope="cpu_fixture", device="cpu")
    return source, runtime, common, context


def test_shared_parent_once_owned_fields_and_origin_invariance(backend_profile):
    source, runtime, common, context = reader(backend_profile)
    packet = source.capture(common, context)
    assert runtime.solver.calls == 1
    assert packet.parent_parameters["num_envs"] == 2
    assert packet.source.parent_solver_id == id(runtime.solver)
    assert packet.source.env_rows == (0, 1)
    obs = packet.inputs
    assert obs["rod_nodes"].shape == (2,41,13)
    assert obs["common_valid"].sum() == 132
    assert torch.isnan(obs["common371"][~obs["common_valid"]]).all()
    for value in obs.values():
        torch.testing.assert_close(value[:1], value[1:], atol=1e-6, rtol=0, equal_nan=True)
    before = obs["rod_nodes"].clone()
    common[0]["values"].fill(0)
    assert torch.equal(before, packet.inputs["rod_nodes"])
    source.validate_fresh(packet)


@pytest.mark.parametrize("damage", ["reset", "tick", "solver_time", "sdk", "native_error"])
def test_same_sdk_old_epoch_or_changed_native_state_rejected(backend_profile, damage):
    source, runtime, common, context = reader(backend_profile)
    packet = source.capture(common, context)
    if damage == "reset":
        runtime.resets[0] += 1
        runtime.episodes = ("new-ep0", runtime.episodes[1])
    elif damage == "tick":
        runtime.ticks[0] += 12
    elif damage == "solver_time":
        runtime.solver.time += .001
    elif damage == "sdk":
        source.sim.current_time_step_index += 1
    else:
        runtime.solver.last_error = 1
    with pytest.raises(ValueError):
        source.validate_fresh(packet)


def test_wrong_origin_duplicate_lane_and_context_epoch_fail(backend_profile):
    source, _, common, context = reader(backend_profile)
    with pytest.raises(ValueError):
        source.capture(common[::-1], context)
    wrong = deepcopy(common)
    wrong[1]["source_packets"][0]["reset_seed"]["environment_origin_w"][0] += 1
    with pytest.raises(ValueError):
        source.capture(wrong, context)
    bad = [replace(context[0], metadata=replace(context[0].metadata,
            identities=(replace(context[0].metadata.identities[0], episode_uid="old"),))), context[1]]
    with pytest.raises(ValueError):
        source.capture(common, bad)


def test_native_source_rejects_cpu_solver_double(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_source import SharedTeacherSource
    source, runtime, _, _ = reader(backend_profile)
    with pytest.raises(ValueError, match="loaded.*native"):
        SharedTeacherSource(runtime, source.sim, backend_profile=source.profile,
            env_origins=source.origins, scope="owner_source_unqualified", device="cpu")

def test_world_float32_quantization_is_retained_not_rounded_to_claim_parity(backend_profile):
    source, runtime, common, context = reader(backend_profile)
    old = runtime.solver.positions
    def nonbinary():
        p = old()
        p[:, :, 1] = torch.linspace(0.,1.,41)[None] + torch.tensor([0.,-6.])[:,None]
        return p
    runtime.solver.positions = nonbinary
    packet = source.capture(common, context)
    p = nonbinary()
    actual_strain = p.diff(dim=1).norm(dim=-1)/torch.as_tensor(packet.parent_parameters["rest_lengths_m"], dtype=p.dtype)-1
    # Endpoint's adjacent segment strain follows raw CUDA-position precision,
    # rather than a snapped/rounded "ideal" line or changed material rest length.
    torch.testing.assert_close(packet.inputs["rod_nodes"][:,0,11], actual_strain[:,0])
    assert not torch.equal(packet.inputs["rod_nodes"][0], packet.inputs["rod_nodes"][1])

def test_shared_model_isolated_identity_original_ppo_and_origin_actions(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_shared_native_teacher import SharedNativeTeacher
    from go2_pvcnn.control.dh116.tabletop_training_reward import load_config
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig, PPOTrainer
    from test_native_teacher_model import cpu_model
    source, _, common, context = reader(backend_profile)
    packet = source.capture(common, context)
    old, *_, config, _ = cpu_model(backend_profile)
    reward = load_config(ROOT/"configs/dh116_training/pinch_finger_goal_reward_v1.json", ppo_gamma=.995)
    model = SharedNativeTeacher(config, backend_profile=source.profile, num_envs=2, reward_config=reward)
    obs = model.pack_shared(packet, source)
    torch.testing.assert_close(model.deterministic(obs)[0], model.deterministic(obs)[1], atol=1e-6, rtol=0)
    identity = model.model_identity
    assert identity["source_family"] == "shared_batched_solver_v1"
    assert identity["parent_num_envs"] == 2
    assert identity["rod_position_frame"] == "world_minus_environment_origin"
    with pytest.raises(ValueError):
        old.load_state_dict(model.state_dict())
    with pytest.raises(ValueError):
        model.load_state_dict(old.state_dict())
    trainer = DeformablePPOTrainer(model, PPOConfig(lr=1e-4, gamma=.995, target_kl=.02,
        kl_backtrack_steps=3, epochs=1, minibatches=1), rollout_steps=2, num_envs=2, device="cpu")
    for _ in range(2):
        sample = trainer.collect_step(obs,obs)
        trainer.record_step(torch.tensor([1.,2.]), torch.tensor([True,True]), torch.tensor([99.,3.]),
            truncated=torch.tensor([False,True]), bootstrap_mask=torch.tensor([False,True]),
            executed_actions=sample["actions"], collection_token=sample["collection_token"])
    returns, _ = trainer.rollout.returns_and_advantages(trainer.config)
    torch.testing.assert_close(returns[:2], torch.tensor([1.,2.+.995*3.]))
    assert trainer.update()["minibatch_updates"] == 1
    assert trainer.update.__func__ is PPOTrainer.update


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
def test_shared_teacher_cuda_policy_and_original_update(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_shared_native_teacher import SharedNativeTeacher
    from go2_pvcnn.control.dh116.tabletop_training_reward import load_config
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    from test_native_teacher_model import cpu_model
    source, _, common, context = reader(backend_profile)
    _, *_, config, _ = cpu_model(backend_profile)
    reward = load_config(ROOT/"configs/dh116_training/pinch_finger_goal_reward_v1.json", ppo_gamma=.995)
    model = SharedNativeTeacher(config, backend_profile=source.profile, num_envs=2, reward_config=reward)
    obs = model.pack_shared(source.capture(common,context),source)
    cpu_mean = model.latent_mean(obs).detach()
    trainer = DeformablePPOTrainer(model, PPOConfig(epochs=1,minibatches=1,lr=1e-5),
        rollout_steps=2,num_envs=2,device="cuda:0")
    cuda_obs = {k:v.cuda() for k,v in obs.items()}
    torch.testing.assert_close(model.latent_mean(cuda_obs).cpu(), cpu_mean, atol=1e-6, rtol=0)
    for _ in range(2):
        sample = trainer.collect_step(obs,obs)
        assert sample["actions"].is_cuda
        trainer.record_step(torch.tensor([1.,2.]),torch.ones(2,dtype=torch.bool),torch.zeros(2),
            executed_actions=sample["actions"],collection_token=sample["collection_token"])
    result = trainer.update()
    assert result["minibatch_updates"] == 1 and next(model.parameters()).is_cuda


def test_common_rod_uid_is_bound_to_parent_row_not_caller_text(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_teacher_mounted_wrist_source import adapt_mounted_hands
    source, _, common, context = reader(backend_profile)
    damaged = deepcopy(common)
    packets = damaged[0]["source_packets"]
    clock = dict(damaged[0]["identity"], rod_uid="unrelated-rod")
    for packet in packets:
        packet["identity"] = clock
        packet["reset_seed"]["identity"] = clock
    damaged[0] = adapt_mounted_hands(packets, expected_identity=clock)
    altered = [replace(context[0], metadata=replace(context[0].metadata,
               identities=(replace(context[0].metadata.identities[0], rod_uid="unrelated-rod"),))),
               context[1]]
    with pytest.raises(ValueError):
        source.capture(damaged, altered)
