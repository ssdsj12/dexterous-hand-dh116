"""Real CPU history fixtures exercise contact reward; no native qualification."""
from dataclasses import replace
from pathlib import Path
import gc
import weakref

import pytest

from test_native_teacher_contact_context import arrange, prime, current_state
from test_native_teacher_context import backend_profile
from test_action24_history_owner import model_setup, MeasuredCommonTeacher, DeformablePPOTrainer, PPOConfig
from test_dh116_power_palm_contact_evidence import build
from go2_pvcnn.control.dh116 import tabletop_training_reward as rw
from go2_pvcnn.control.dh116.tabletop_native_teacher_context import ContextIdentity

PROFILE = "approach_contact_prelift_v1"
ROOT = Path(__file__).resolve().parents[1]


def config():
    return rw.load_config(ROOT / "configs/dh116_training" / (PROFILE + ".json"))


def endpoint(owner, *, distance=.01):
    contact = owner.contact_readback
    identity = owner._contact_identity
    stamp = rw.SourceStamp(identity.episode_uid, identity.physics_step, identity.solver_time_s,
        "cpu_fixture", "cpu_fixture_contact_endpoint", env_id=identity.env_id,
        reset_uid=str(identity.reset_id), rod_uid=identity.rod_uid)
    def measured(value, unit="dimensionless", source="cpu_fixture_distance_health"):
        return rw.Measurement(value, unit, replace(stamp, source=source))
    fields = dict(distance=measured(distance,"m"), numerical_health=measured(True),
                  constraint_failure=measured(False))
    for name in ("coverage_left","coverage_right","opposed_left","opposed_right",
                 "bilateral_support","table_contact_failure"):
        fields[name] = measured(contact[name], source=contact["source"])
    pads = tuple((side,tuple(contact["evidence"]["per_hand"][side]["required_pads"]))
                 for side in ("left","right"))
    return rw.RewardEndpoint(stamp,tuple(str(i) for i in range(41)),fields,required_pad_ids=pads)


def request(row, token, values=None, policy="cpu_policy"):
    return rw.SubmittedRequest(token,policy,row.stamp,tuple([0.]*24 if values is None else values),
                              "normalized_submitted_command24_preprojection")


def ready(monkeypatch,tmp_path,profile, *, load=.125):
    data = arrange(monkeypatch,tmp_path,profile,load=load)
    prime(data)
    owner = data[-2]
    reward = rw.TrainingReward(config(),contact_owner=owner)
    initial = endpoint(owner)
    reward.reset_from_verified_state(initial,previous_request=request(
        initial,owner.contact_readback["event_sha256"],policy="source_priming_no_policy"))
    return data,reward


def run_window(data,reward, *, damage=None):
    m,c,ep,h,b,j,rows,owner,state = data
    pre = ep.pre_action_state()
    h.capture_pre_action(pre)
    owner.freeze_pre_action(ContextIdentity.from_clock(pre["common"]["identity"]),history_owner=h)
    conf,identity,common,valid,raw,clock,mask = model_setup()
    model = MeasuredCommonTeacher(conf,identity)
    obs = {k:v[:1].clone() for k,v in model.pack_measured(common,valid,raw,clock,mask).items()}
    trainer = DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),rollout_steps=1,num_envs=1)
    sample = trainer.collect_step(obs,obs)
    h.begin(pre,trainer)
    reward.begin(request(endpoint(owner),sample["collection_token"],sample["actions"][0].tolist()),
                 remaining_substeps=8)
    record = ep._record
    def recorded(row):
        record(row)
        owner.observe_physical_event(row,history_owner=h)
        value = endpoint(owner)
        if damage is not None:
            value = damage(value,owner)
        reward.observe(value)
    ep._record = recorded
    result = ep.control(sample["actions"])
    h.end(result)
    return reward.settle_once(sample["collection_token"],buffer_full=True)


def test_loaded_contact_profile_does_not_admit_caller_labels_without_owner():
    c = config()
    with pytest.raises((ValueError,rw.SourceFailure),match="contact"):
        rw.TrainingReward(c)
    with pytest.raises((ValueError,rw.SourceFailure),match="contact"):
        rw.TrainingReward(c,contact_owner=object())


def test_contact_reward_uses_original_reset_scale_and_excludes_grasp_success(monkeypatch,tmp_path,backend_profile):
    data,reward = ready(monkeypatch,tmp_path,backend_profile)
    packet = run_window(data,reward)
    assert packet.phi_pre == pytest.approx(4.)  # 1 distance + 2 coverage + 1 opposition
    assert packet.phi_post_raw == pytest.approx(4. + .5*(8/480)/.25)
    assert packet.post_context["distance_ref_m"] == .02  # Reset2, not primed .01
    assert packet.post_context["contact_s"] == data[-2].contact_readback["duration_s"]
    assert packet.post_context["hold_s"] is None
    assert packet.post_context["contact_event_sha256"] == data[-2].contact_readback["event_sha256"]
    assert packet.event == 0. and not packet.terminated
    for name in ("lift","hold","slip","velocity","prohibited_support"):
        assert packet.terms[name]["raw"] is None and not packet.terms[name]["applicable"]
    assert not packet.event_terms["success"]["applicable"]
    assert not packet.native_ready and not packet.teacher_admission
    with pytest.raises(RuntimeError,match="settled|absent"):
        reward.settle_once(packet.token)


def test_contact_break_uses_same_owner_tail_duration_not_decaying_accumulator(monkeypatch,tmp_path,backend_profile):
    data,reward = ready(monkeypatch,tmp_path,backend_profile,load=lambda tick:.08 if tick==7 else .125)
    packet = run_window(data,reward)
    assert packet.post_context["contact_s"] == pytest.approx(3/480)
    assert packet.phi_post_raw == pytest.approx(4. + .5*(3/480)/.25)


@pytest.mark.parametrize("damage",["coverage","pair_identity","foreign_clock"])
def test_reward_rejects_contact_values_or_identity_not_owned_by_recorded_event(monkeypatch,tmp_path,backend_profile,damage):
    data,reward = ready(monkeypatch,tmp_path,backend_profile)
    def corrupt(row,owner):
        if damage=="pair_identity":
            return replace(row,required_pad_ids=(("left",("thumb",)),("right",("thumb",))))
        if damage=="foreign_clock":
            return replace(row,stamp=replace(row.stamp,reset_uid="foreign_reset"))
        fields=dict(row.fields)
        fields["coverage_left"]=replace(fields["coverage_left"],raw=0.)
        return replace(row,fields=fields)
    with pytest.raises(rw.SourceFailure):
        run_window(data,reward,damage=corrupt)
    assert reward.quarantined
    with pytest.raises(rw.SourceFailure):
        reward.settle_once("quarantined")


def test_weak_opposed_total_load_never_grows_contact_persistence(monkeypatch,tmp_path,backend_profile):
    data,reward = ready(monkeypatch,tmp_path,backend_profile,load=.08)
    packet = run_window(data,reward)
    assert packet.terms["contact"]["raw"] == 1.
    assert packet.terms["geometry"]["raw"] == 1.
    assert packet.terms["persistence"]["raw"] == 0.
    assert packet.post_context["contact_s"] == 0.


def test_closing_contact_owner_invalidates_reward_before_next_action(monkeypatch,tmp_path,backend_profile):
    data,reward = ready(monkeypatch,tmp_path,backend_profile)
    row=endpoint(data[-2]);data[-2].close()
    with pytest.raises(rw.SourceFailure):
        reward.begin(request(row,"after_close"),remaining_substeps=8)
    assert reward.quarantined


def test_reward_lease_does_not_keep_context_and_native_holder_chain_alive(monkeypatch,tmp_path,backend_profile):
    data,reward = ready(monkeypatch,tmp_path,backend_profile)
    row=endpoint(data[-2]);ref=weakref.ref(data[-2])
    del data
    gc.collect()
    assert ref() is None
    with pytest.raises(rw.SourceFailure):
        reward.begin(request(row,"orphan"),remaining_substeps=8)


def test_reset_cannot_claim_priming_as_a_policy_transition(monkeypatch,tmp_path,backend_profile):
    data=arrange(monkeypatch,tmp_path,backend_profile,load=.125);prime(data)
    owner=data[-2];reward=rw.TrainingReward(config(),contact_owner=owner)
    row=endpoint(owner)
    with pytest.raises(rw.SourceFailure,match="prim"):
        reward.reset_from_verified_state(row,previous_request=request(row,"fake_policy_token"))
    assert reward.quarantined
    with pytest.raises(rw.SourceFailure):
        reward.reset_from_verified_state(row,previous_request=request(
            row,owner.contact_readback["event_sha256"],policy="source_priming_no_policy"))
    assert reward.quarantined


def test_source_contact_owner_cannot_be_attached_to_old_smoke_profile(monkeypatch,tmp_path,backend_profile):
    data=arrange(monkeypatch,tmp_path,backend_profile);prime(data)
    old=rw.load_config(ROOT/"configs/dh116_training/approach_wiring_smoke_v1.json")
    with pytest.raises((ValueError,rw.SourceFailure),match="contact"):
        rw.TrainingReward(old,contact_owner=data[-2])

def test_end_stands_are_allowed_then_real_table_contact_terminates_without_success(monkeypatch,tmp_path,backend_profile):
    from copy import deepcopy
    import numpy as np

    data=arrange(monkeypatch,tmp_path,backend_profile)
    ep=data[2];owner=data[-2]
    step=ep._exchange._step
    def supported(action,substep):
        row=step(action,substep)
        # CPU solver-response double: both end stands carry load from SDK3.
        # Table becomes loaded only on the final SDK boundary, never hidden.
        state=build(proximal=True,table=True,load=.125)
        rod=row["rod"]
        if rod["physics_step"]<11:
            cap=state["rod_mesh_contact_pairs"]
            removed=[p for p in cap["pairs"] if p["mesh_name"]=="table"]
            cap["pairs"]=[p for p in cap["pairs"] if p["mesh_name"]!="table"]
            cap["solver_buffer"]["attempted"]=len(cap["pairs"])
            for pair in removed:
                state["distributed_contact"][0,pair["node_index"]]-=np.asarray(pair["force_on_rod_world_n"])/12
            state["named_wrenches"]["table"][:]=0.
            state["table_wrench_world"][:]=0.
            state["external_support_wrenches_world"]["table"][:]=0.
        for key in ("distributed_contact","named_wrenches","table_wrench_world",
                    "external_support_wrenches_world","rod_mesh_contact_pairs"):
            rod[key]=deepcopy(state[key])
        rod["rod_mesh_contact_pairs"].update(physics_step=rod["physics_step"],rod_tick=rod["rod_tick"])
        return row
    ep._exchange._step=supported
    prime(data)
    assert owner.contact_readback["support_load_n"]["support_0"]>.1
    assert owner.contact_readback["support_load_n"]["support_1"]>.1
    assert not owner.contact_readback["no_external_support"]
    reward=rw.TrainingReward(config(),contact_owner=owner)
    row=endpoint(owner)
    reward.reset_from_verified_state(row,previous_request=request(
        row,owner.contact_readback["event_sha256"],policy="source_priming_no_policy"))
    packet=run_window(data,reward)
    assert packet.terminated and packet.event==-40.
    assert packet.event_reasons==("table_contact_failure",)
    assert packet.endpoint_phi==0. and packet.bootstrap(100.)==0.
    assert packet.event_terms["success"]["raw"] is None
    assert packet.duration_s==pytest.approx(8/480)
