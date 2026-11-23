"""CPU checks for feedback-aware control with the original PPO."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_native_teacher_model import backend_profile, cpu_model

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest


def _fixture(profile, grasp="pinch"):
    from go2_pvcnn.control.dh116.tabletop_wire_feedback_teacher import WireFeedbackTeacher

    base, ledger, valid, rod, clock, mask, context, *_ = cpu_model(profile, grasp)
    context = replace(context, metadata=replace(context.metadata, grasp=grasp))
    obs = base.pack_cpu_fixture(ledger, valid, rod, clock, mask, context)
    return base, obs, WireFeedbackTeacher(base)


def _packet(env, load, *, available=True):
    clock = dict(run_uuid="cpu-run", episode_uid=f"episode-{env}", reset_id=1, env_id=env,
                 rod_uid="rod", physics_step=10, rod_tick=96, timestamp_s=10/480,
                 solver_time_s=8/480, clock_valid=True, policy_step=1)
    force = [[[0., 0., 0., 0., 0.] for _ in range(6)] for _ in range(2)]
    reaction_valid = [[False]*6 for _ in range(2)]
    if available:
        force[0][0] = [0., load, -load/2., 0., 0.]
        reaction_valid[0][0] = True
    observation = dict(position_in_palm_m=[[.01, .02, .03],[-.01, .02, .03]],
                       tangent_in_palm=[[0., 1., 0.],[0., 1., 0.]],
                       reaction_force_n=force, reaction_valid=reaction_valid)
    return dict(clock=clock, actor_used=False, teacher_admission=False,
                status="measured" if available else "reaction_unavailable", purpose="bootstrap",
                collection_token=f"token-{env}",
                source_window=dict(first_sdk_step=3, end_sdk_step=10, end_rod_tick=96, control_index=0),
                pre_action_sha256=_digest({"pre": env}), observation_sha256=_digest(observation),
                observation=observation)


def _pack(model, obs, rows, *, expected=None):
    trusted = rows if expected is None else expected
    return model.pack_recorded_feedback(obs, rows, base_observation_sha256=_digest(obs), clocks=[r["clock"] for r in trusted],
        pre_action_sha256s=[r["pre_action_sha256"] for r in trusted])


@pytest.mark.parametrize("grasp", ["pinch", "tripod", "power"])
def test_candidate_keeps_original_default_motion_and_grasp_identity(backend_profile, grasp):
    base, obs, model = _fixture(backend_profile, grasp)
    no_feedback = model.missing_feedback(obs)
    measured = _pack(model, obs, [_packet(0,.4),_packet(1,.2)])
    for packet in (no_feedback, measured):
        assert torch.equal(model.deterministic(packet),base.deterministic(obs))
        assert torch.equal(model.value(packet),base.value(obs))
    assert model.model_identity["grasp"] == grasp
    assert model.model_identity["input_schema_sha256"] != base.model_identity["input_schema_sha256"]
    assert model.model_identity["checkpoint_namespace"] != base.model_identity["checkpoint_namespace"]
    assert all(p.data_ptr() != q.data_ptr() for p,q in zip(base.parameters(),model.base.parameters(),strict=True))


def test_exact_zero_has_validity_and_changes_action_after_original_ppo_update(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig

    torch.manual_seed(32)
    _, obs, model = _fixture(backend_profile, "tripod")
    zero = _pack(model,obs,[_packet(0,0.),_packet(1,0.)])
    loaded = _pack(model,obs,[_packet(0,.4),_packet(1,.2)])
    missing_force = _pack(model,obs,[_packet(0,0.,available=False),_packet(1,0.,available=False)])
    assert zero["wire_reaction_valid"][0,0,0] and not missing_force["wire_reaction_valid"].any()
    assert not zero["wire_reaction"][0,0,0].any()
    assert torch.isnan(missing_force["wire_reaction"]).all()
    trainer=DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1,lr=.001),
                                  rollout_steps=2,num_envs=2)
    for packet, rewards in ((loaded,[2.,1.]),(zero,[.4,-.8])):
        sample=trainer.collect_step(packet,packet)
        assert sample["actions"].shape == (2,24)
        trainer.record_step(torch.tensor(rewards),torch.ones(2,dtype=torch.bool),torch.zeros(2),
            executed_actions=sample["actions"],collection_token=sample["collection_token"])
    trainer._check_frozen_policy(trainer.rollout.batch())
    metrics=trainer.update()
    assert metrics["minibatch_updates"] == 1 and metrics["samples"] == 4
    assert (model.deterministic(loaded)-model.deterministic(zero)).abs().max() > 0
    assert (model.deterministic(zero)-model.deterministic(missing_force)).abs().max() > 0
    assert (model.value(loaded)-model.value(zero)).abs().max() > 0
    assert torch.equal(model.deterministic(model.missing_feedback(obs)),model.base.deterministic(obs))
    model.zero_grad()
    loaded["wire_reaction"].requires_grad_(True)
    model.latent_mean(loaded).sum().backward()
    assert loaded["wire_reaction"].grad[loaded["wire_reaction_valid"]].abs().sum() > 0
    assert not loaded["wire_reaction"].grad[~loaded["wire_reaction_valid"]].any()
    assert all(p.grad is None for p in model.critic_feedback.parameters())
    model.zero_grad()
    loaded["wire_reaction"].grad = None
    model.value(loaded).sum().backward()
    assert loaded["wire_reaction"].grad[loaded["wire_reaction_valid"]].abs().sum() > 0
    assert all(p.grad is None for p in model.actor_feedback.parameters())


def test_feedback_clock_provenance_and_weights_roundtrip(backend_profile,tmp_path):
    base,obs,model = _fixture(backend_profile)
    rows=[_packet(0,.7),_packet(1,.6)]
    packet=_pack(model,obs,rows)
    changed = {key: value.clone() for key,value in obs.items()}
    changed["rod_nodes"][0,0,0] += .1
    with pytest.raises(ValueError,match="base observation"):
        model.pack_recorded_feedback(changed, rows, base_observation_sha256=_digest(obs),
            clocks=[r["clock"] for r in rows], pre_action_sha256s=[r["pre_action_sha256"] for r in rows])
    for field in ("clock","pre_action_sha256","observation","source_window"):
        damaged=deepcopy(rows)
        if field=="clock":
            damaged[0][field]["policy_step"]+=1
        elif field=="pre_action_sha256":
            damaged[0][field]="1"*64
        elif field=="source_window":
            damaged[0][field]["end_rod_tick"]-=12
        else:
            damaged[0][field]["reaction_force_n"][0][0][1]+=.01
        with pytest.raises(ValueError):
            _pack(model,obs,damaged,expected=rows)
    file=tmp_path/"feedback-policy.pt"
    torch.save(model.state_dict(),file)
    same=type(model)(base)
    same.load_state_dict(torch.load(file,weights_only=True))
    assert torch.equal(same.deterministic(packet),model.deterministic(packet))
    with pytest.raises(ValueError,match="identity"):
        same.load_state_dict(base.state_dict())
    with pytest.raises(ValueError,match="identity"):
        base.load_state_dict(same.state_dict())

    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_native_teacher_checkpoint import checkpoint_identity

    other,*_=cpu_model(backend_profile,"power")
    with pytest.raises(ValueError,match="identity"):
        same.load_state_dict(type(model)(other).state_dict())
    with pytest.raises(ValueError,match="native context CPU Teacher"):
        checkpoint_identity(DeformablePPOTrainer(same),{})
