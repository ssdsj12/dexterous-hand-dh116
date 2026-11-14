"""CPU doubles only: no native scene, checkpoint training, or optimizer update."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_deformable_r0_replay import setup as physical_setup
from test_dh116_flexible_measured_common_teacher import setup as model_setup
from go2_pvcnn.control.dh116.tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher
from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

ROOT = Path(__file__).resolve().parents[1]

def module():
    from go2_pvcnn.control.dh116 import tabletop_no_camera_collector
    return tabletop_no_camera_collector

def fixture(monkeypatch, tmp_path, *, max_controls=2, mutate=None, reward=1.):
    episode, rod, hands, sim, rows, r0 = physical_setup(monkeypatch, tmp_path, mode="empty")
    config, identity, common, valid, raw, clock, mask = model_setup()
    model = MeasuredCommonTeacher(config, identity)
    obs = model.pack_measured(common, valid, raw, clock, mask)
    obs = {k:v[:1].clone() for k,v in obs.items()}
    trainer = DeformablePPOTrainer(model, PPOConfig(epochs=1, minibatches=1),
                                  rollout_steps=2, num_envs=1)
    observations = []
    def pack(pre):
        observations.append(deepcopy(pre["common"]["identity"]))
        return {k:v.clone() for k,v in obs.items()}
    rewards = []
    def scalar(receipt):
        rewards.append(deepcopy(receipt))
        return reward
    if mutate:
        mutate(episode, rod, hands, sim, rows, r0)
    collector = module().CPUFixtureCollector(
        episode, trainer, r0, pack_observation=pack, fixture_reward=scalar,
        journal_path=tmp_path/"collector.jsonl", max_controls=max_controls,
        fixture_label="owned SDK/rod doubles and synthetic scalar; not native R0")
    return collector, episode, trainer, r0, sim, rod, hands, observations, rewards

def events(path):
    return [decode_owned(json.loads(line)) for line in Path(path).read_text().splitlines()]

@pytest.mark.parametrize("style", ["pinch","tripod","power"])
def test_real_gate_denies_before_any_factory_and_does_not_accept_switches(style):
    gate = module().actual_admission(ROOT, style)
    assert not gate["admitted"]
    assert gate["reset_mesh_qualified"] is False
    assert gate["training_reward_qualified"] is False
    assert {"mesh_reset_source_unqualified", "formal_R0_training_source_missing"} <= set(gate["blocked_reasons"])
    assert gate["projection_valid_fields"] == 66
    def forbidden(*a, **kw):
        pytest.fail("denied input constructed policy, optimizer or native episode")
    with pytest.raises(RuntimeError, match="admission"):
        module().create_actual_collector(ROOT, style,
            policy_factory=forbidden, trainer_factory=forbidden, episode_factory=forbidden)
    assert all(gate[key] is False for key in ("native_qualification","teacher_admission"))

def test_two_windows_reuse_original_ppo_and_same_pre_action(monkeypatch,tmp_path):
    c, ep, tr, r0, sim, rod, hands, observations, rewards = fixture(monkeypatch,tmp_path)
    before = {k:v.clone() for k,v in tr.model.state_dict().items()}
    first = c.collect()
    second = c.collect()
    assert first["physical_samples"] == second["physical_samples"] == 8
    assert not first["done"] and second["truncated"] and second["bootstrap_mask"]
    assert sim.current_time_step_index == 18 and rod.rope.steps == 16
    assert len(r0._controls) == 2 and len(rewards) == 2
    assert all(r["control_receipt"]["training_reward"] is None for r in rewards)
    assert all(r["control_receipt"]["diagnostic_reward"] is None for r in rewards)
    assert tr.counters["updates"] == 0
    assert all(torch.equal(v,tr.model.state_dict()[k]) for k,v in before.items())
    batch=tr.rollout.batch()
    assert batch["actions"].shape == (2,24)
    assert torch.isnan(batch["actor_obs"]["common371"][:,64]).all()
    log=events(c.journal_path)
    physical=[r for r in log if r["operation"]=="physical"]
    assert len(physical)==16
    for i,row in enumerate(physical):
        expected=batch["actions"][i//8].double().numpy()
        assert np.array_equal(row["payload"]["exchange"]["action_readback"]["request"]["action24"][0],expected)
    decisions=[r for r in log if r["operation"]=="decision"]
    assert decisions[0]["payload"]["pre_action_identity"]["physics_step"]==2
    assert decisions[1]["payload"]["pre_action_identity"]["physics_step"]==10
    assert second["endpoint_identity"]["physics_step"]==18
    assert observations[-1]["physics_step"]==18
    assert all(r["scope"]=="cpu_fixture_only" for r in log)
    assert len([r for r in log if r["operation"]=="settlement"])==2
    with pytest.raises(RuntimeError,match="closed"):
        c.collect()
    c.close()

def test_terminal_is_persisted_before_stop_and_noop_cannot_replace_it(monkeypatch,tmp_path):
    def failure(ep,rod,hands,sim,rows,r0):
        step=sim.step
        def advance(*,render):
            step(render=render)
            hands[1].measured[0,hands[1].joint_names.index("finger12")]=-.05
        sim.step=advance
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path,mutate=failure)
    result=c.collect()
    assert result["terminated"] and not result["truncated"] and not result["bootstrap_mask"]
    assert result["termination_reason"]=="right:joint_limit"
    assert result["physical_samples"]==1 and rod.rope.steps==1
    assert len(r0._controls)==1 and len(rewards)==1
    log=events(c.journal_path)
    assert [r["operation"] for r in log][-3:]==["physical","settlement","transition_recorded"]
    before=c.journal_path.read_bytes()
    with pytest.raises(RuntimeError,match="closed"):c.collect()
    with pytest.raises(RuntimeError,match="active"):ep.control(torch.zeros(1,24))
    assert c.journal_path.read_bytes()==before
    assert tr.rollout._rows[0]["next_value"].item()==0
    c.close()

def test_truncation_bootstraps_same_pre_reset_and_original_gae_does_not_cross_reset(monkeypatch,tmp_path):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path,max_controls=1,reward=2.)
    result=c.collect()
    row=tr.rollout._rows[0]
    assert row["done"].item() and row["bootstrap_mask"].item()
    assert result["endpoint_identity"]["phase"]=="post_action_pre_reset"
    # Add a second independently sampled transition to the original PPO rollout;
    # the first truncated boundary must stop GAE carry from the new episode.
    pending=tr.collect_step(row["actor_obs"],row["critic_obs"])
    tr.record_step(torch.tensor([100.]),torch.tensor([True]),torch.tensor([0.]),
        executed_actions=pending["actions"],collection_token=pending["collection_token"])
    returns,adv=tr.rollout.returns_and_advantages(tr.config)
    expected=row["rewards"]+tr.config.gamma*row["next_value"]-row["values"]
    assert torch.allclose(adv[:1],expected)
    assert tr.counters["updates"]==0
    c.close()

@pytest.mark.parametrize("damage",["reset","step","rod","epoch"])
def test_foreign_bootstrap_rejected_after_durable_rows(monkeypatch,tmp_path,damage):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path,max_controls=1)
    original=ep.pre_action_state
    calls=0
    def altered():
        nonlocal calls
        value=original();calls+=1
        if calls==3:
            key={"reset":"reset_id","step":"physics_step","rod":"rod_uid","epoch":"episode_uid"}[damage]
            value["common"]["identity"][key] = "foreign" if damage in ("rod","epoch") else 999
        return value
    ep.pre_action_state=altered
    with pytest.raises(ValueError,match="endpoint"):c.collect()
    assert len([r for r in events(c.journal_path) if r["operation"]=="physical"])==8
    assert not len(tr.rollout) and tr._pending is not None
    assert events(c.journal_path)[-1]["operation"]=="failure"
    with pytest.raises(RuntimeError,match="closed"):c.collect()
    c.close()

@pytest.mark.parametrize("damage",["request","issued","skip","duplicate"])
def test_action_readback_and_exactly_once_damage_closes_without_next_sample(monkeypatch,tmp_path,damage):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path)
    step=ep._exchange._inner._step
    def altered(action,substep):
        row=step(action,substep)
        if damage=="request":row["action_readback"]["request"]["action24"][0,0]+=.01
        elif damage=="issued":row["action_readback"]["post"]["hands"]["left"]["backend_targets"][0,row["action_readback"]["post"]["hands"]["left"]["active_indices"][0]]+=.01
        elif damage=="skip":row["substep"]+=1
        return row
    ep._exchange._inner._step=altered
    if damage=="duplicate":
        record=ep._record
        def twice(row):
            record(row);record(row)
        ep._record=twice
    with pytest.raises((ValueError,RuntimeError)):c.collect()
    assert rod.rope.steps==1 and not len(tr.rollout)
    assert events(c.journal_path)[-1]["operation"]=="failure"
    with pytest.raises(RuntimeError,match="closed"):c.collect()
    c.close()

@pytest.mark.parametrize("damage",["policy_epoch","session","weights","origin"])
def test_policy_or_checkpoint_origin_change_rejected_before_sdk(monkeypatch,tmp_path,damage):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path)
    if damage=="policy_epoch":tr.policy_version+=1
    elif damage=="session":tr.session_id="foreign"
    elif damage=="weights":
        with torch.no_grad():next(tr.model.parameters()).add_(.01)
    else:c.origin["model_identity_sha256"]="f"*64
    with pytest.raises(ValueError,match="policy|origin"):c.collect()
    assert rod.rope.steps==0 and sim.current_time_step_index==2
    c.close()

def test_exclusive_journal_refuses_overwrite_and_reward_unknown_is_not_zero(monkeypatch,tmp_path):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path,reward=None)
    before=c.journal_path.read_bytes()
    with pytest.raises(FileExistsError):
        module().ExclusiveJournal(c.journal_path)
    assert c.journal_path.read_bytes()==before
    with pytest.raises(ValueError,match="synthetic reward"):c.collect()
    assert not len(tr.rollout) and rod.rope.steps==8
    assert events(c.journal_path)[-1]["operation"]=="failure"
    c.close()

def test_r0_terminal_on_first_row_stops_original_episode_without_ignored_callback_return(monkeypatch,tmp_path):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path)
    original=r0.record
    def record(event,**kwargs):
        if event["kind"]=="physical":
            clock=event["exchange"]["rod"]
            from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import IDENTITY
            kwargs["auxiliary"]=dict(identity={k:clock[k] for k in IDENTITY},
                samples={"stable":{"value":False,"source":"CPU explicit instability fixture"}})
        return original(event,**kwargs)
    r0.record=record
    result=c.collect()
    assert result["termination_reason"]=="r0:instability"
    assert result["terminated"] and result["physical_samples"]==1
    assert rod.rope.steps==1 and len(rewards)==1
    c.close()


def test_packing_cannot_advance_pre_action_boundary(monkeypatch,tmp_path):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path)
    pack=c._pack
    def moved(pre):
        result=pack(pre)
        ep._control_index+=1
        return result
    c._pack=moved
    with pytest.raises(ValueError,match="pre-action"):c.collect()
    assert rod.rope.steps==0 and tr._pending is None
    c.close()

def test_previous_physical_row_is_on_disk_before_next_sdk_advance(monkeypatch,tmp_path):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path,max_controls=1)
    step=sim.step
    def advance(*,render):
        saved=[r for r in events(c.journal_path) if r["operation"]=="physical"]
        assert len(saved)==rod.rope.steps
        if saved:
            assert saved[-1]["payload"]["exchange"]["physics_step"]==sim.current_time_step_index
        step(render=render)
    sim.step=advance
    c.collect()
    assert rod.rope.steps==8
    c.close()

def test_existing_recorder_failure_still_leaves_the_delivered_sample(monkeypatch,tmp_path):
    def failing_recorder(ep,rod,hands,sim,rows,r0):
        def record(event):
            raise OSError("old evidence sink failed")
        ep._record=record
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path,mutate=failing_recorder)
    with pytest.raises(OSError,match="old evidence"):c.collect()
    log=events(c.journal_path)
    assert [r["operation"] for r in log][-2:]==["physical","failure"]
    assert log[-2]["payload"]["exchange"]["physics_step"]==3
    assert rod.rope.steps==1 and not len(tr.rollout)
    with pytest.raises(RuntimeError,match="closed"):c.collect()
    c.close()

def test_non_r0_guard_partial_stays_saved_but_has_no_invented_reward(monkeypatch,tmp_path):
    def failure(ep,rod,hands,sim,rows,r0):
        step=sim.step
        def advance(*,render):
            step(render=render)
            hand=hands[1]
            hand.measured[0,hand.joint_names.index("finger13")]=.3
        sim.step=advance
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path,mutate=failure)
    with pytest.raises(ValueError,match="R0 cannot settle"):c.collect()
    log=events(c.journal_path)
    assert log[-2]["payload"]["exchange"]["guard_failure"]=="right:mimic_error"
    assert rod.rope.steps==1 and not rewards and not len(tr.rollout)
    c.close()


def test_truncated_budget_closes_underlying_entry_before_noop(monkeypatch,tmp_path):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path,max_controls=1)
    c.collect()
    before=c.journal_path.read_bytes()
    with pytest.raises(RuntimeError,match="closed"):ep.control(torch.zeros(1,24))
    assert rod.rope.steps==8 and sim.current_time_step_index==10
    assert c.journal_path.read_bytes()==before
    c.close()

def test_settlement_and_ppo_commit_have_distinct_durable_records(monkeypatch,tmp_path):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path,max_controls=1)
    receipt=c.collect()
    log=events(c.journal_path)
    assert log[-2]["operation"]=="settlement"
    assert log[-2]["payload"]["ppo_record_status"]=="prepared_before_trainer_record"
    assert log[-1]["operation"]=="transition_recorded"
    assert log[-1]["payload"]["collection_token"]==receipt["collection_token"]
    assert log[-1]["payload"]["rollout_rows"]==1
    c.close()


@pytest.mark.parametrize("damage",["solver_time_s","policy_step","substep"])
def test_bootstrap_requires_full_completed_clock_not_only_gae_fields(monkeypatch,tmp_path,damage):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path,max_controls=1)
    original=ep.pre_action_state
    calls=0
    def altered():
        nonlocal calls
        result=original();calls+=1
        if calls>=3:result["common"]["identity"][damage]+=1
        return result
    ep.pre_action_state=altered
    with pytest.raises(ValueError,match="endpoint"):c.collect()
    assert rod.rope.steps==8 and len(tr.rollout)==0
    c.close()
