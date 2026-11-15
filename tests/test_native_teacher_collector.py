"""Original CPU physical episode + actual C02 network/PPO/reward; no native.

The explicit synthetic source adapter cannot admit production, and no collector
collect/control/model/trainer implementation is mocked in these tests.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch

from test_native_teacher_context import backend_profile


def fixture(monkeypatch,tmp_path,profile,*,rollout_steps=2,cap_controls=99):
    import test_deformable_r0_replay as replay
    from go2_pvcnn.control.dh116.tabletop_action24_history_owner import Action24HistoryOwner
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import TaskContextOwner
    from go2_pvcnn.control.dh116.tabletop_native_teacher_model import NativeContextTeacher
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal,NativeTeacherCollector
    from go2_pvcnn.control.dh116.tabletop_training_reward import TrainingReward,load_config
    from test_dh116_flexible_measured_common_teacher import setup as model_setup
    original=replay.episode_setup
    holder={}
    def setup(*a,**k):
        ep,rod,hands,sim,seed,owners,rows=original(*a,**k)
        ep._end=66
        history=Action24HistoryOwner(initial_action24=torch.zeros(1,24),
            seed_source="explicit new CPU controller zero fixture",physics_budget_end=66,solver_substeps=12)
        owners["controller"]=history.wrap_controller_reset(owners["controller"])
        for phase,step in (("before_reset",0),("after_reset",2)):
            history.record(dict(kind="setup",phase=phase,sdk_step=step,sdk_time_s=step/480))
        ep._record=history.wrap_recorder(rows.append)
        holder["history"]=history;holder["seed"]=deepcopy(seed);holder["owners"]=dict(owners)
        return ep,rod,hands,sim,seed,owners,rows
    monkeypatch.setattr(replay,"episode_setup",setup)
    ep,rod,hands,sim,rows,diagnostic=replay.setup(monkeypatch,tmp_path,mode="empty")
    history=holder["history"]
    for side in ("left","right"):
        history.record(dict(kind="initial_geometry",side=side,evidence=dict(side=side,
            identity=dict(sdk_step=2,sdk_time_s=2/480))))
    config,*_=model_setup("pinch")
    model=NativeContextTeacher(config,backend_profile=profile)
    reward_config=load_config(Path(__file__).resolve().parents[1]/"configs/dh116_training/approach_wiring_smoke_v1.json")
    trainer=DeformablePPOTrainer(model,PPOConfig(gamma=reward_config.gamma,epochs=1,minibatches=1,
        lr=.0001),rollout_steps=rollout_steps,num_envs=1)
    journal=NativeJournal(tmp_path/"journal.jsonl",scope="cpu_fixture")
    context=TaskContextOwner(backend_profile=profile,mode=reward_config.profile,scope="cpu_fixture",
        episode_cap_s=2.,external_cap_s=3.)
    owner=SimpleNamespace(episode=ep,history=history,context=context,ledger=journal,
        pre_action_state=ep.pre_action_state,control=ep.control,_restores=[])
    sources=FixtureSources(owner,profile,rod,cap_controls)
    owner.sources=sources;owner.reset_seed=holder["seed"];owner.original_owner_resets=holder["owners"]
    record=ep._record
    def durable(row):
        journal.emit("physical",row)
        record(row)
    ep._record=durable
    context.reset_from_verified_state(sources.context_state(None),history_owner=history)
    collector=NativeTeacherCollector(owner,trainer,TrainingReward(reward_config),SimpleNamespace())
    return collector,ep,rod,hands,sim,rows


class FixtureSources:
    scope="cpu_fixture"
    solver_substeps=12
    def __init__(self,owner,profile,rod,cap_controls):
        self.owner,self.profile,self.rod,self.cap=owner,profile,rod,cap_controls
        self.damage=None;self.observed=0
        self._last=self.endpoint(rod.clock(),False)
    def endpoint(self,clock,guard):
        from go2_pvcnn.control.dh116.tabletop_training_reward import SourceStamp,Measurement,RewardEndpoint
        stamp=SourceStamp(clock["episode_uid"],clock["physics_step"],clock["solver_time_s"],
            "cpu_fixture","synthetic_cpu_endpoint",env_id=0,reset_uid=str(clock["reset_id"]),rod_uid=clock["rod_uid"])
        fields=dict(distance=Measurement(.02,"m",stamp),numerical_health=Measurement(True,"dimensionless",stamp),
            constraint_failure=Measurement(guard,"dimensionless",stamp),original_guard_joint=Measurement(guard,"dimensionless",stamp))
        return RewardEndpoint(stamp,tuple(str(i) for i in range(41)),fields)
    def reset_reward(self,reward):
        from go2_pvcnn.control.dh116.tabletop_training_reward import SubmittedRequest
        reward.reset_from_verified_state(self._last,previous_request=SubmittedRequest("reset-zero","cpu-reset",
            self._last.stamp,(0.,)*24,"normalized_submitted_command24_preprojection"))
    def submitted_request(self,sample,decision):
        from go2_pvcnn.control.dh116.tabletop_training_reward import SubmittedRequest
        return SubmittedRequest(sample["collection_token"],decision["policy"]["model_weights_sha256"],self._last.stamp,
            tuple(float(v) for v in sample["actions"][0]),"normalized_submitted_command24_preprojection")
    def observe(self,row):
        from go2_pvcnn.control.dh116.tabletop_training_reward import Measurement
        self.observed+=1
        self._last=self.endpoint(row["exchange"]["rod"],row["exchange"]["guard_failure"] is not None)
        if self.observed==3:
            if self.damage=="nan":
                fields=dict(self._last.fields);fields["distance"]=Measurement(float("nan"),"m",self._last.stamp)
                self._last=replace(self._last,fields=fields)
            elif self.damage=="identity": self._last=replace(self._last,stamp=replace(self._last.stamp,episode_uid="foreign"))
        return self._last
    def remaining_substeps(self): return self.owner.episode._end-self.rod.clock()["physics_step"]
    def external_truncated(self): return self.observed>=8*self.cap
    def context_state(self,packet):
        from go2_pvcnn.control.dh116.tabletop_native_teacher_context import ContextIdentity
        identity=ContextIdentity.from_clock(dict(self.rod.clock(),policy_step=self.owner.episode._control_index,substep=0))
        return dict(identity=identity,source_scope="cpu_fixture",backend_sha256=self.profile.binary_sha256,
            node_positions=torch.full((1,41,3),.31),distance_mean_m=.02,rest_length_m=1.,
            target_reference_uid="explicit_cpu_fixed41",external_budget=dict(source="external_budget_clock",
                identity=identity,remaining_s=3.-self.observed/480,cap_s=3.),event_history=dict(
                source="task_event_history",identity=identity,valid=True,success_emitted=False,
                trusted_failure_emitted=bool(packet and packet.event<0)))
    def pack(self,model,pre,context):
        from test_dh116_deformable_privileged_teacher import inputs
        clock=pre["common"]["identity"]
        _,rod,ids,mask=inputs(dict(backend_sha256=self.profile.binary_sha256),batch=1,nodes=41)
        # Brand new synthetic source buffers. Original old-backend rod evidence
        # remains untouched in the physical journal; it is never retagged.
        ids.update(env_ids=[clock["env_id"]],episode_uids=[clock["episode_uid"]],reset_ids=[clock["reset_id"]],
            rod_uids=[clock["rod_uid"]],physics_step=clock["physics_step"],rod_tick=clock["rod_tick"],timestamp_s=clock["timestamp_s"])
        rod.update(ids);rod["velocities"].zero_()
        return model.pack_cpu_fixture(torch.as_tensor(pre["common"]["values"],dtype=torch.float32)[None].clone(),
            torch.as_tensor(pre["common"]["valid"])[None].clone(),rod,ids,mask,context)


def operations(c): return [json.loads(line) for line in c.journal.path.read_text().splitlines()]


def test_normal_buffer_boundary_owned_context_and_update_continue_episode(monkeypatch,tmp_path,backend_profile):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile)
    first=c.collect();frozen=c.trainer.rollout.batch()["actor_obs"]["task_context"].clone()
    second=c.collect()
    assert first.done is False and second.done is False and second.buffer_full is True
    assert sim.current_time_step_index==18 and rod.rope.steps==16
    batch=c.trainer.rollout.batch()
    assert not batch["done"].any() and batch["bootstrap_mask"].all()
    torch.testing.assert_close(batch["actor_obs"]["task_context"][:1],frozen,equal_nan=True)
    assert not torch.equal(batch["actor_obs"]["task_context"][0,12:36],batch["actor_obs"]["task_context"][1,12:36])
    episode=rod.clock()["episode_uid"];reset=rod.clock()["reset_id"]
    metrics=c.trainer.update();c.accept_updated_policy();c.collect()
    assert metrics["samples"]==2 and c.trainer.counters["updates"]==1
    assert rod.clock()["episode_uid"]==episode and rod.clock()["reset_id"]==reset
    assert sim.current_time_step_index==26 and rod.rope.steps==24


def test_third_trusted_guard_durable_exact_three_terminal_rows(monkeypatch,tmp_path,backend_profile):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile)
    original=sim.step
    def step(*,render):
        original(render=render)
        if rod.rope.steps==2: hands[1].measured[0,hands[1].joint_names.index("finger12")]=-.05
    sim.step=step
    packet=c.collect()
    assert packet.terminated and not packet.truncated and packet.event==-40 and packet.endpoint_phi==0
    assert packet.duration_s==pytest.approx(3/480)
    assert len(rows)==4 and rod.rope.steps==3 and sim.current_time_step_index==5
    assert c.trainer.rollout.batch()["done"].all()
    assert not c.trainer.rollout.batch()["bootstrap_mask"].any()
    assert operations(c)[-1]["operation"]=="transition_recorded"


@pytest.mark.parametrize("damage",["nan","identity","request","targets","weights"])
def test_third_bad_source_preserves_raw_pending_and_quarantines(monkeypatch,tmp_path,backend_profile,damage):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile)
    c.sources.damage=damage
    if damage in ("request","targets","weights"):
        step=ep._exchange.inner._step
        def broken(action,substep):
            row=step(action,substep)
            if substep==2:
                if damage=="request": row["action_readback"]["request"]["action24"][0,0]+=.1
                elif damage=="targets": row["action_readback"]["post"]["hands"]["left"]["backend_targets"][0,0]+=.1
                else:
                    with torch.no_grad(): next(c.trainer.model.parameters()).add_(.01)
            return row
        ep._exchange.inner._step=broken
    with pytest.raises((ValueError,RuntimeError)): c.collect()
    assert c.quarantined and c.trainer._pending is not None
    assert c.trainer.counters["updates"]==0 and len(c.trainer.rollout)==0
    assert rod.rope.steps==(8 if damage=="weights" else 3)
    records=operations(c)
    assert records[-1]["operation"]=="rollout_quarantined"
    assert sum(row["operation"]=="physical" for row in records)==rod.rope.steps
    with pytest.raises(RuntimeError): c.collect()
    assert not list(tmp_path.glob("*.pt"))


def test_external_truncation_value_comes_from_same_pre_reset_endpoint(monkeypatch,tmp_path,backend_profile):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile,cap_controls=1)
    packet=c.collect();batch=c.trainer.rollout.batch()
    assert packet.truncated and packet.done and not packet.terminated
    assert batch["bootstrap_mask"].all()
    settled=[r for r in operations(c) if r["operation"]=="settlement"][0]
    assert settled["payload"]["gae_boundary"]["gae_carry"]==0
    assert c.context.current.metadata.identities[0].physics_step==10
    assert ep._state=="active" and sim.current_time_step_index==10
    with pytest.raises(RuntimeError): c.collect()


def test_insufficient_budget_does_not_sample_or_advance(monkeypatch,tmp_path,backend_profile):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile)
    c.sources.remaining_substeps=lambda:7
    assert c.collect() is None
    assert c.trainer._pending is None and rod.rope.steps==0


def test_repeat_reward_token_cannot_reexecute_or_settle(monkeypatch,tmp_path,backend_profile):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile)
    packet=c.collect()
    with pytest.raises(RuntimeError):c.reward.settle_once(packet.token)
    assert rod.rope.steps==8 and len(c.trainer.rollout)==1


def test_raw_exchange_survives_original_boundary_conversion_failure(monkeypatch,tmp_path,backend_profile):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile)
    capture=ep._boundary.capture
    original=ValueError("physical boundary invalid")
    def broken(row):
        if rod.rope.steps==3: raise original
        return capture(row)
    monkeypatch.setattr(ep._boundary,"capture",broken)
    with pytest.raises(ValueError) as exc:c.collect()
    assert exc.value is original
    assert c.consumed_sdk_steps==3 and rod.rope.steps==3
    assert sum(r["operation"]=="delivered_exchange" for r in operations(c))==3
    assert c.quarantined and c.trainer._pending is not None
    with pytest.raises(RuntimeError):c.update()


def test_pre_reset_bootstrap_value_is_explicit_and_owned(monkeypatch,tmp_path,backend_profile):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile,cap_controls=1)
    calls=[]
    def value(obs):
        # A deliberately different endpoint value from the following reset.
        calls.append(obs["task_context"].clone())
        return torch.tensor([37. if rod.clock()["reset_id"]==1 else -999.])
    monkeypatch.setattr(c.trainer.model,"value",value)
    packet=c.collect()
    assert packet.truncated and c.trainer.rollout.batch()["next_value"].item()==37.
    assert calls[-1][0,43]>0 and c.context.current.values[0,43]>0


def test_policy_token_replay_fails_before_any_second_window(monkeypatch,tmp_path,backend_profile):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile)
    first=c.collect();collect=c.trainer.collect_step
    def replay(*args):
        sample=collect(*args)
        sample["collection_token"]=first.token
        return sample
    monkeypatch.setattr(c.trainer,"collect_step",replay)
    with pytest.raises(ValueError,match="replayed"):c.collect()
    assert rod.rope.steps==8 and c.quarantined and c.trainer._pending is not None


def test_full_adapter_cannot_borrow_smoke_sources_or_relabel_grasp():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import NativePinchSources
    for style,profile,match in (("tripod","approach_wiring_smoke_v1","style-specific"),
            ("pinch","flexible_train_r1_full","full_shared")):
        contract=SimpleNamespace(config=SimpleNamespace(preflight=dict(style=style,profile=profile)))
        with pytest.raises(ValueError,match=match):NativePinchSources(object(),contract)


def test_old_history_explicitly_rejects_real_reset_at_sdk_n_greater_than_two(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116.tabletop_action24_history_owner import Action24HistoryOwner
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile,cap_controls=1)
    c.collect();assert sim.current_time_step_index==10
    old=Action24HistoryOwner(initial_action24=torch.zeros(1,24),seed_source="new CPU reset",
        physics_budget_end=66,solver_substeps=12)
    clock=dict(rod.clock(),episode_uid="new-episode",reset_id=2,rod_tick=0,source_seed_sha256="0"*64)
    original=lambda context:dict(owner="controller",episode_uid=context["episode_uid"],status="cleared",source="CPU original callback")
    with pytest.raises(ValueError,match="fresh original reset boundary"):
        old.wrap_controller_reset(original)(clock)


def test_new_reset_bound_history_accepts_actual_absolute_reset_not_fake_setup(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_action_history import NativeAction24HistoryOwner
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile,cap_controls=1)
    c.collect();prior=rod.clock();assert prior["physics_step"]==10
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
    history=NativeAction24HistoryOwner(initial_action24=torch.zeros(1,24),seed_source="new CPU reset actual clock",
        physics_budget_end=66,solver_substeps=12,previous_reset=prior)
    owners=dict(c.owner.original_owner_resets)
    # Original v1 callback was installed by initial fixture. Use its original
    # explicit no-memory owner callback, rather than wrapping the old memory.
    owners["controller"]=history.wrap_controller_reset(lambda context:dict(owner="controller",
        episode_uid=context["episode_uid"],status="cleared",source="CPU explicit original owner"))
    fresh=DualHandPhysicalEpisode(sim,rod,ep._bindings,backend_artifact=ep._boundary._artifact,
        backend_sha256=ep._boundary._sha256,inspect_geometry=ep._inspect_geometry,physics_budget_end=66,
        record=history.wrap_recorder(rows.append))
    reset=fresh.reset(c.owner.reset_seed,episode_uid="run-cpu:episode-two",owner_resets=owners)
    assert reset["reset"]["after_clock"]["physics_step"]==10
    assert rod.rope.steps==8 and sim.current_time_step_index==10
    assert history._reset["identity"]["reset_id"]==prior["reset_id"]+1
    assert not np.any(history._previous) and history._sequence.setup==0


def fresh_cpu_episode(previous,*,uid,cap_controls):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_action_history import NativeAction24HistoryOwner
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import TaskContextOwner
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeTeacherCollector
    from go2_pvcnn.control.dh116.tabletop_training_reward import TrainingReward
    old=previous.owner;ep=old.episode;rod=old.sources.rod;prior=rod.clock();sim=ep._sim
    history=NativeAction24HistoryOwner(initial_action24=torch.zeros(1,24),seed_source="explicit next CPU episode zero",
        physics_budget_end=66,solver_substeps=12,previous_reset=prior)
    owners=dict(old.original_owner_resets)
    owners["controller"]=history.wrap_controller_reset(lambda context:dict(owner="controller",
        episode_uid=context["episode_uid"],status="cleared",source="explicit original CPU controller"))
    rows=[]
    def durable(row):old.ledger.emit("physical",row);rows.append(deepcopy(row));history.record(row)
    fresh=DualHandPhysicalEpisode(sim,rod,ep._bindings,backend_artifact=ep._boundary._artifact,
        backend_sha256=ep._boundary._sha256,inspect_geometry=ep._inspect_geometry,physics_budget_end=66,record=durable)
    fresh.reset(old.reset_seed,episode_uid=uid,owner_resets=owners)
    for side in ("left","right"):
        history.record(dict(kind="initial_geometry",side=side,evidence=dict(side=side,
            identity=dict(sdk_step=sim.current_time_step_index,sdk_time_s=sim.current_time))))
    context=TaskContextOwner(backend_profile=old.sources.profile,mode=previous.reward.config.profile,
        scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
    owner=SimpleNamespace(episode=fresh,history=history,context=context,ledger=old.ledger,
        pre_action_state=fresh.pre_action_state,control=fresh.control,_restores=[],reset_seed=old.reset_seed,
        original_owner_resets=old.original_owner_resets)
    sources=FixtureSources(owner,old.sources.profile,rod,cap_controls);owner.sources=sources
    context.reset_from_verified_state(sources.context_state(None),history_owner=history)
    return NativeTeacherCollector(owner,previous.trainer,TrainingReward(previous.reward.config),SimpleNamespace(),previous=previous),rows


def test_three_actual_cpu_episodes_reset_memory_events_without_clock_or_rollout_reset(monkeypatch,tmp_path,backend_profile):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile,rollout_steps=4,cap_controls=1)
    first=c.collect();old_context=c.context.current;prior_clock=rod.clock()
    assert first.truncated and len(c.trainer.rollout)==1
    second,rows2=fresh_cpu_episode(c,uid="run-cpu:second",cap_controls=99)
    assert sim.current_time_step_index==10 and rod.rope.steps==8
    assert second.context.current.values[0,49]==0 and not second.history._previous.any()
    assert second.context.current.values[0,43]==0
    step=sim.step
    def fail(*,render):
        step(render=render)
        if rod.rope.steps==10:hands[1].measured[0,hands[1].joint_names.index("finger12")]=-.05
    sim.step=fail
    packet=second.collect();assert packet.terminated and packet.event==-40
    assert sim.current_time_step_index==13 and rod.rope.steps==11 and len(second.trainer.rollout)==2
    assert second.context.current.values[0,49]==1 and old_context.values[0,49]==0
    sim.step=step
    third,rows3=fresh_cpu_episode(second,uid="run-cpu:third",cap_controls=99)
    assert sim.current_time_step_index==13 and rod.rope.steps==11
    assert third.context.current.values[0,49]==0 and not third.history._previous.any()
    assert third.context.current.values[0,43]==0 and third.sources.rod.clock()["reset_id"]==3
    packet3=third.collect();assert not packet3.done
    assert third.trainer is c.trainer and len(third.trainer.rollout)==3
    assert rod.rope.steps==19 and sim.current_time_step_index==21
    assert rod.clock()["solver_time_s"]>prior_clock["solver_time_s"]
    assert third.consumed_sdk_steps==19 and third.context.current.metadata.identities[0].episode_uid=="run-cpu:third"


def test_source_primary_exception_survives_quarantine_log_failure(monkeypatch,tmp_path,backend_profile):
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,backend_profile)
    original=ValueError("original source exception");observe=c.sources.observe;emit=c.journal.emit
    def source(row):
        if rod.rope.steps==3:raise original
        return observe(row)
    def journal(operation,payload):
        if operation=="rollout_quarantined":raise OSError("quarantine log failure")
        return emit(operation,payload)
    monkeypatch.setattr(c.sources,"observe",source);monkeypatch.setattr(c.journal,"emit",journal)
    with pytest.raises(ValueError) as exc:c.collect()
    assert exc.value is original and c.quarantined and rod.rope.steps==3
    assert c.trainer._pending is not None and c.consumed_sdk_steps==3
