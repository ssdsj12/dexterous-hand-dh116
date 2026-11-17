"""Real Python controller/rollout methods with explicit CPU SDK/solver doubles."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from test_dh116_dual_hand_episode import setup as episode_setup
from test_dh116_flexible_measured_common_teacher import setup as model_setup
from go2_pvcnn.control.dh116.tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher
from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig


def module():
    from go2_pvcnn.control.dh116 import tabletop_action24_history_owner
    return tabletop_action24_history_owner


def fixture(monkeypatch, tmp_path):
    ep, rod, hands, sim, seed, owners, rows = episode_setup(monkeypatch, tmp_path)
    rod.env_id = 0
    for binding in ep._bindings:
        path = next(iter(binding.env_paths.values()))
        binding.env_paths.clear(); binding.env_paths[0] = path
    for hand in hands:
        hand.data.body_link_pose_w = hand.link_poses[..., [0,1,2,6,3,4,5]]
        hand.data.body_lin_vel_w = torch.zeros((1,23,3))
        hand.data.body_ang_vel_w = torch.zeros((1,23,3))
        hand.root_physx_view.get_link_velocities = lambda: torch.zeros((1,23,6))
        previous = hand.update
        def update(dt, hand=hand, previous=previous):
            previous(dt)
            hand.data.body_link_pose_w = hand.link_poses[..., [0,1,2,6,3,4,5]]
        hand.update = update
    initial = torch.zeros((1,24), dtype=torch.float32)
    owner = module().Action24HistoryOwner(initial_action24=initial,
        seed_source="explicit CPU fixture controller reset zeros; no historical backfill",
        physics_budget_end=18, solver_substeps=12)
    owners["controller"] = owner.wrap_controller_reset(owners["controller"])
    ep._record = owner.wrap_recorder(rows.append)
    # These setup/geometry events are explicitly CPU fixtures. They grant no geometry qualification.
    for phase, step in (("before_reset",0),("after_reset",2)):
        owner.record(dict(kind="setup",phase=phase,sdk_step=step,sdk_time_s=step/480))
    ep.reset(seed,episode_uid="run-cpu:action-history",owner_resets=owners)
    for side in ("left","right"):
        owner.record(dict(kind="initial_geometry",side=side,
            evidence=dict(side=side,identity=dict(sdk_step=2,sdk_time_s=2/480))))
    config, identity, common, valid, raw, clock, mask = model_setup()
    model = MeasuredCommonTeacher(config,identity)
    obs = {k:v[:1].clone() for k,v in model.pack_measured(common,valid,raw,clock,mask).items()}
    trainer = DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),rollout_steps=2,num_envs=1)
    return owner,ep,trainer,obs,rod,hands,sim,rows,initial


def begin(owner,ep,tr,obs):
    pre = ep.pre_action_state()
    history = owner.capture_pre_action(pre)
    sample = tr.collect_step(obs,obs)
    decision = owner.begin(pre,tr)
    return pre,history,sample,decision


def settle(owner,ep,tr,sample):
    result = ep.control(sample["actions"])
    packet = owner.end(result)
    issued = owner.issued_for_pending(tr,packet)
    tr.record_step(torch.tensor([0.]),torch.tensor([result["termination_reason"] is not None]),
        torch.tensor([0.]),executed_actions=issued,collection_token=sample["collection_token"])
    return result,packet,issued


def test_two_actual_windows_bind_previous_memory_and_original_ppo_storage(monkeypatch,tmp_path):
    owner,ep,tr,obs,rod,hands,sim,rows,initial = fixture(monkeypatch,tmp_path)
    weights = {k:v.clone() for k,v in tr.model.state_dict().items()}
    initial.fill_(.9)  # caller seed storage is not retained
    previous = np.zeros((1,24),np.float32)
    for i in range(2):
        pre,history,sample,decision = begin(owner,ep,tr,obs)
        assert np.array_equal(history["previous_action24"],previous)
        assert history["identity"]["physics_step"] == 2+8*i
        assert history["source"] == module().SOURCE
        assert history["native_qualification"] is history["teacher_admission"] is False
        assert pre["common"]["valid"].sum() == 66
        result,packet,issued = settle(owner,ep,tr,sample)
        actual = rows[1+8*i]["exchange"]["action_readback"]["request"]["action24"]
        assert torch.equal(issued,torch.as_tensor(actual,dtype=torch.float32))
        assert torch.equal(issued,sample["actions"])
        assert packet["physical_samples"] == 8 and packet["training_reward"] is None
        assert packet["policy"]["collection_token"] == sample["collection_token"]
        assert packet["previous_action_sha256"] == decision["previous_action_sha256"]
        assert len(packet["execution_targets_sha256"]) == 8
        previous = issued.numpy().copy()
    assert rod.rope.steps == 16 and sim.current_time_step_index == 18
    assert len(tr.rollout) == 2 and tr.counters["updates"] == 0
    assert all(torch.equal(v,tr.model.state_dict()[k]) for k,v in weights.items())
    assert rows[0]["reset"]["owner_resets"]["controller"]["source"] == "CPU explicit owner"


@pytest.mark.parametrize("damage",["dtype","shape","nonfinite","range","nonzero","source"])
def test_reset_seed_requires_explicit_original_zero_float32_source(damage):
    seed = torch.zeros((1,24)); source = "explicit new reset"
    if damage == "dtype": seed = seed.double()
    elif damage == "shape": seed = seed[0]
    elif damage == "nonfinite": seed[0,0] = float("nan")
    elif damage == "range": seed[0,0] = 2
    elif damage == "nonzero": seed[0,0] = .1
    else: source = ""
    with pytest.raises(ValueError):
        module().Action24HistoryOwner(initial_action24=seed,seed_source=source,
            physics_budget_end=18,solver_substeps=12)


@pytest.mark.parametrize("damage",["seed_storage","previous_storage","history_packet","pre_clock","pre_values"])
def test_immutability_and_same_pre_action_boundary_fail_closed(monkeypatch,tmp_path,damage):
    owner,ep,tr,obs,*_ = fixture(monkeypatch,tmp_path)
    pre = ep.pre_action_state()
    history = owner.capture_pre_action(pre)
    if damage == "seed_storage": owner._seed = np.ones((1,24),np.float32)
    elif damage == "previous_storage": owner._previous[0,0] = .1
    elif damage == "history_packet":
        history["previous_action24"][0,0] = .4
        assert owner.capture_pre_action(pre)["previous_action24"][0,0] == 0
        return
    elif damage == "pre_clock": pre["common"]["identity"]["physics_step"] += 1
    else: pre["common"]["values"][0] = 1
    tr.collect_step(obs,obs)
    with pytest.raises((ValueError,RuntimeError)): owner.begin(pre,tr)


@pytest.mark.parametrize("damage",["token","epoch","session","model","pending_action"])
def test_policy_pending_origin_cannot_change_after_sampling(monkeypatch,tmp_path,damage):
    owner,ep,tr,obs,*_ = fixture(monkeypatch,tmp_path)
    _,_,sample,_ = begin(owner,ep,tr,obs)
    result = ep.control(sample["actions"]); packet = owner.end(result)
    if damage == "token": tr._pending["collection_token"] = "foreign"
    elif damage == "epoch": tr.policy_version += 1
    elif damage == "session": tr.session_id = "foreign"
    elif damage == "model": tr.model._model_identity["checkpoint_namespace"] += "_foreign"
    else: tr._pending["actions"][0,0] += .01
    with pytest.raises((ValueError,RuntimeError)): owner.issued_for_pending(tr,packet)


@pytest.mark.parametrize("damage",["action","issued","clock","duplicate"])
def test_actual_readback_is_required_and_failed_row_is_recorded_first(monkeypatch,tmp_path,damage):
    owner,ep,tr,obs,rod,hands,sim,rows,_ = fixture(monkeypatch,tmp_path)
    _,_,sample,_ = begin(owner,ep,tr,obs)
    original = ep._exchange._step
    def broken(action,substep):
        row = original(action,substep)
        if damage == "action": row["action_readback"]["request"]["action24"][0,0] += .01
        elif damage == "issued":
            hand = row["action_readback"]["post"]["hands"]["left"]
            hand["backend_targets"][0,hand["active_indices"][0]] += .01
        elif damage == "clock": row["substep"] += 1
        return row
    ep._exchange._step = broken
    if damage == "duplicate":
        record = ep._record
        def twice(row):
            record(row);record(row)
        ep._record = twice
    with pytest.raises((ValueError,RuntimeError)): ep.control(sample["actions"])
    assert rows[-1]["kind"] == "physical" and rod.rope.steps == 1
    with pytest.raises(RuntimeError): owner.end(dict())
    assert len(tr.rollout) == 0 and tr.counters["updates"] == 0


def test_partial_terminal_keeps_one_issued_sample_and_cannot_be_overwritten(monkeypatch,tmp_path):
    owner,ep,tr,obs,rod,hands,sim,rows,_ = fixture(monkeypatch,tmp_path)
    _,_,sample,_ = begin(owner,ep,tr,obs)
    step = sim.step
    def stop(*,render):
        step(render=render)
        hands[1].measured[0,hands[1].joint_names.index("finger12")] = -.05
    sim.step = stop
    result,packet,issued = settle(owner,ep,tr,sample)
    assert result["termination_reason"] == "right:joint_limit"
    assert packet["physical_samples"] == 1 and rod.rope.steps == 1
    assert rows[-1]["exchange"]["guard_failure"] == "right:joint_limit"
    with pytest.raises((ValueError,RuntimeError)): owner.end(result)
    with pytest.raises((ValueError,RuntimeError)): owner.issued_for_pending(tr,packet)


def test_reset_receipt_cannot_be_replaced_by_historical_or_foreign_owner(monkeypatch,tmp_path):
    owner,ep,tr,obs,rod,hands,sim,rows,_ = fixture(monkeypatch,tmp_path)
    stale = deepcopy(rows[0])
    stale["reset"]["owner_resets"]["controller"].pop("action24_history")
    fresh = module().Action24HistoryOwner(initial_action24=torch.zeros(1,24),
        seed_source="new zero seed",physics_budget_end=18,solver_substeps=12)
    for phase,step in (("before_reset",0),("after_reset",2)):
        fresh.record(dict(kind="setup",phase=phase,sdk_step=step,sdk_time_s=step/480))
    with pytest.raises((ValueError,RuntimeError)): fresh.record(stale)


def test_projection_targets_are_not_reencoded_as_policy_actions(monkeypatch,tmp_path):
    owner,ep,tr,obs,rod,hands,sim,rows,_ = fixture(monkeypatch,tmp_path)
    _,_,sample,_ = begin(owner,ep,tr,obs)
    result,packet,issued = settle(owner,ep,tr,sample)
    row = rows[1]["exchange"]
    assert "wrist_projection" in row
    actual_targets = row["action_readback"]["submission"]["expected_backend_targets"]
    assert packet["execution_targets_sha256"][0] == module()._digest(actual_targets)
    assert torch.equal(issued,sample["actions"])
    # Actual physical targets are dimensional 12+12 values, not normalized action24.
    targets = np.concatenate([actual_targets[s]["values"] for s in ("left","right")],axis=1)
    assert not np.array_equal(targets,issued.numpy())


def test_real_projection_can_clip_wrist_without_changing_issued_policy_action(monkeypatch,tmp_path):
    owner,ep,tr,obs,rod,hands,sim,rows,_ = fixture(monkeypatch,tmp_path)
    # CPU policy double's mean is negative; original Gaussian sampler and storage run.
    tr._actor_mean = lambda observations,n: torch.full((n,24),-1.)
    with torch.no_grad():
        tr.model.log_std.fill_(-20.)
    geometry = ep._exchange._geometry
    def at_floor(*args,**kwargs):
        value = geometry(*args,**kwargs)
        value["clearance_m"].fill_(.0021)
        return value
    ep._exchange._geometry = at_floor
    _,_,sample,_ = begin(owner,ep,tr,obs)
    result,packet,issued = settle(owner,ep,tr,sample)
    row = rows[1]["exchange"]
    request = row["action_readback"]["request"]
    projection = row["wrist_projection"]
    assert request["requested_palm_offsets_m"][0,0,2] < 0
    assert projection["palm_offsets_m"][0,0,2] >= 0
    assert projection["palm_offsets_m"][0,0,2] != request["requested_palm_offsets_m"][0,0,2]
    assert torch.equal(issued,sample["actions"]) and issued[0,14] < 0
    assert packet["wrist_projection_sha256"][0] == module()._digest(projection)


@pytest.mark.parametrize("damage",["window_count","terminal","settlement_action","settlement_target"])
def test_incomplete_or_tampered_settlement_never_enters_ppo(monkeypatch,tmp_path,damage):
    owner,ep,tr,obs,*_ = fixture(monkeypatch,tmp_path)
    _,_,sample,_ = begin(owner,ep,tr,obs)
    result = ep.control(sample["actions"])
    if damage in ("window_count","terminal"):
        if damage == "window_count": result["physical_samples"] = 7
        else: result["termination_reason"] = "invented"
        with pytest.raises(ValueError): owner.end(result)
    else:
        packet = owner.end(result)
        if damage == "settlement_action": packet["issued_action24"][0,0] += .1
        else: packet["execution_targets_sha256"][0] = "0"*64
        with pytest.raises(ValueError): owner.issued_for_pending(tr,packet)
    assert len(tr.rollout) == 0 and tr.counters["updates"] == 0


def test_reset_callback_is_once_and_preserves_unavailable_controller_status():
    owner = module().Action24HistoryOwner(initial_action24=torch.zeros(1,24),
        seed_source="explicit zero controller initialization",physics_budget_end=18,solver_substeps=12)
    context = dict(run_uuid="cpu",env_id=0,episode_uid="cpu:a",reset_id=1,rod_uid="rod",
        physics_step=2,rod_tick=0,policy_step=0,substep=0,timestamp_s=2/480,
        solver_time_s=0.,clock_valid=True,source_seed_sha256="a"*64)
    original = lambda ctx: dict(owner="controller",episode_uid=ctx["episode_uid"],
        status="unavailable",source="other controller state is unknown")
    callback = owner.wrap_controller_reset(original)
    packet = callback(context)
    assert packet["status"] == "unavailable"
    assert packet["action24_history"]["previous_action24"] == [[0.]*24]
    assert packet["action24_history"]["teacher_admission"] is False
    with pytest.raises(ValueError): callback(dict(context,episode_uid="cpu:b",reset_id=2))
