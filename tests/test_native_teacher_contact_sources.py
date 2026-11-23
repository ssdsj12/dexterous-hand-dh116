"""Contact source/collector on explicit CPU SDK, geometry and getter substitutes.

The original episode, priming history, Context, Reward, Collector, Actor and PPO
run unchanged. Synthetic geometry/ABI data grant no native qualification.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import sys

import numpy as np
import pytest
import torch

from test_native_teacher_context import backend_profile
from test_native_teacher_source_priming import case
from test_dh116_power_palm_contact_evidence import build
from test_native_teacher_model import raw_cpu_packets
from test_dh116_flexible_measured_common_teacher import setup as model_setup

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "approach_contact_prelift_v1"


def source_module():
    m=importlib.import_module("go2_pvcnn.control.dh116.tabletop_native_teacher_sources")
    assert hasattr(m,"NativeContactTripodSources"), "actual primed Contact Tripod source missing"
    return m


def test_priming_event_readback_is_owned_and_detects_internal_evidence_mutation(monkeypatch,tmp_path):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    m,c,ep,h,runtime,journal,rows=case(monkeypatch,tmp_path)
    def original_owner_record(row):
        # Actual NativeEpisodeOwner receives raw tuple IDs from the controller,
        # unlike history.wrap_recorder's canonical JSON copy.
        journal.emit("owner_event",row)
        rows.append(deepcopy(row));h.record(row)
    ep._record=original_owner_record
    receipt=m.prime_supported_sources(ep,history=h,runtime=runtime,journal=journal)
    assert hasattr(type(h),"priming_event"), "owned priming event readback missing"
    event=h.priming_event
    assert _digest(event)==receipt["priming_row_sha256"]
    event["exchange"]["action_readback"]["request"]["action24"][0,0]=1.
    assert _digest(h.priming_event)==receipt["priming_row_sha256"]
    h._priming_event["exchange"]["rod"]["timestamp_s"]+=1.
    with pytest.raises((ValueError,RuntimeError),match="changed|prim"):
        _=h.priming_event


def arrange(monkeypatch,tmp_path,profile,*,table_at=None,bad_middle=False,bad_prime_parameter=False,controls=1,lift_diagnostic=False,
            wire_trace_geometry=False,wire_pose_error_m=0.):
    m=source_module()
    from go2_pvcnn.control.dh116.tabletop_deformable_rod_boundary import RodBoundaryState
    from go2_pvcnn.control.dh116.tabletop_native_teacher_model import NativeContextTeacher
    from go2_pvcnn.control.dh116.tabletop_training_reward import TrainingReward,load_config
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeTeacherCollector
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope,policy_source_basis
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_tripod_distance import PolicyOwnedTripodApproach
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_pinch_distance import _radius
    from go2_pvcnn.control.dh116.tabletop_r0_health_separation import make_solver_status
    from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as parameters
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_source_priming as pm
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
    journal=NativeJournal(tmp_path/"contact.jsonl",scope="cpu_fixture")
    from test_fresh_supported_episode import episode_case
    end=18 if controls==1 else 3+8*controls
    second=tmp_path/"fresh";second.mkdir()
    if controls == 1:
        c,_,plan,ep,rows,readback=episode_case(monkeypatch,second)
    else:
        # Construct a new worker-shaped CPU env0 before any Reset; never relabel
        # an already captured env7 event, manifest or old native receipt.
        import test_native_teacher_supported_reset as reset_fixture
        from test_dual_hand_reset import setup as fresh_reset_setup
        with monkeypatch.context() as env_scope:
            env_scope.setattr(reset_fixture,"setup",lambda m,**kw:fresh_reset_setup(m,env_id=0,**kw))
            c,_,plan,ep,rows,readback=episode_case(monkeypatch,second)
    for binding in c.rod.bindings.values():
        binding.update(frame="PhysX_link_origin",unit="m")
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_order import seal_plan
    plan=deepcopy(plan);plan.pop("sha256")
    plan["bindings"]=deepcopy(c.rod.bindings)
    plan=seal_plan(plan)  # New CPU constructor input before Reset, never an old native receipt.
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
    ep=DualHandPhysicalEpisode(c.sim,c.rod,c.bindings,backend_artifact=profile.binary_path,
        backend_sha256=profile.binary_sha256,inspect_geometry=ep._inspect_geometry,
        physics_budget_end=end,record=rows.append)
    history=pm.PrimedNativeActionHistory(initial_action24=torch.zeros(1,24),
        seed_source="new explicit CPU zero memory",physics_budget_end=end,solver_substeps=12)
    owners=dict(c.owners);owners["controller"]=history.wrap_controller_reset(owners["controller"])
    def record(event):
        journal.emit("physical" if event["kind"]=="physical" else "owner_event",event)
        rows.append(deepcopy(event));history.record(event)
    ep._record=record
    if controls != 1:
        # Same explicit CPU setup boundaries already charged below, recorded
        # before Reset just as the actual NativeEpisodeOwner does.
        for step in (0,2):
            journal.emit("setup",dict(kind="setup",sdk_step=step,sdk_time_s=step/480,
                phase="before_reset" if step==0 else "after_reset"))
    from go2_pvcnn.control.dh116 import tabletop_deformable_dual_hand_episode as ep_module
    from go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard import ClearanceGuardedExchange
    with monkeypatch.context() as reset_scope:
        reset_scope.setattr(ep_module,"FlexiblePhysicalExchange",ClearanceGuardedExchange)
        reset=ep.reset(c.seed,episode_uid="cpu17:contact_collector",owner_resets=owners,
            supported_mesh_plan=plan,record_reset_readback=readback.append)
    for side in ("left","right"):
        history.record(dict(kind="initial_geometry",side=side,
            evidence=dict(side=side,identity=dict(sdk_step=2,sdk_time_s=2/480))))
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import SessionBudget
    limits=dict(max_sdk_steps=end,max_solver_calls=end-2,max_resets=1,max_episode_count=1,wall_timeout_s=1000.)
    runtime=SessionBudget(limits)
    owner=SimpleNamespace(episode=ep,sim=c.sim,rod=c.rod,bindings=ep._bindings,history=history,
        ledger=journal,episode_uid=c.rod.clock()["episode_uid"],_restores=[],
        pre_action_state=ep.pre_action_state,control=ep.control)
    runtime.record_setup(owner,[dict(sdk_step=0,sdk_time_s=0.),dict(sdk_step=2,sdk_time_s=2/480)])
    runtime.reserve_reset(owner)
    reward_config=load_config(ROOT/"configs/dh116_training/approach_contact_prelift_v1.json")
    contract=SimpleNamespace(config=SimpleNamespace(project=ROOT,episode_cap_s=10.,
        preflight=dict(style="tripod",profile=PROFILE)),backend_profile=profile,
        reward_config=reward_config,budget=limits,runtime=runtime,
        reset_manifest=SimpleNamespace(data=lambda:dict(rod_state=dict(positions=c.rope.nodes.tolist()))))
    owner.contract=contract
    owner.backend_profile=profile

    def metric(packet,radius):
        clock=packet["raw"]["identity"]
        value=None if bad_middle and clock["physics_step"]>3 else [.02,.02]
        from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
        provenance=dict(version="cpu_geometry_metric",target_material_ids=list(range(41)))
        return dict(auxiliary=dict(identity=deepcopy(clock),samples=dict(
            approach_distance_m=dict(value=value,source="explicit_CPU_three_pad_SDF:"+_digest(provenance)))),
            source=provenance,source_sha256=_digest(provenance),
            source_scope="cpu_fixture",contact_inferred_from_geometry=False)

    class Geometry:
        def capture(self,row,**kwargs):
            raw = dict(identity={k:row["rod"][k] for k in ("run_uuid","env_id","episode_uid",
                "reset_id","rod_uid","physics_step","rod_tick","timestamp_s","solver_time_s")})
            if lift_diagnostic:
                raw["rod_nodes_m"] = np.asarray(row["rod"]["node_positions"])[0].copy()
                raw["hands"] = {
                    side: dict(body_names=["base_link"],
                        link_poses_xyzw=np.array([[[.43, y, .40, 0., 0., 0., 1.]]]))
                    for side, y in (("left", .18), ("right", -.18))}
            if wire_trace_geometry:
                raw["rod_nodes_m"] = np.asarray(row["rod"]["node_positions"])[0].copy()
                raw["hands"] = {}
                for binding in ep._bindings:
                    body = binding.mount.base_id
                    pose = np.asarray(binding.hand.root_physx_view.get_link_transforms())[0, body].copy()
                    pose[0] += wire_pose_error_m
                    raw["hands"][binding.side] = dict(body_names=["base_link"],
                                                      link_poses_xyzw=pose.reshape(1,1,7))
            return dict(raw=raw,
                source_complete=True,failure_reasons=[],source_errors=[],terminal_reason=None)
        def validate(self,packet,row):
            assert _digest(packet) == _digest(self.capture(row))
            return deepcopy(packet)

    def configure(self,own,con,scope):
        self.reset_distance=SimpleNamespace(capture_reset=lambda receipt,p,grasp:
            dict(auxiliary=dict(identity=deepcopy(own.rod.clock()),samples=dict(
                approach_distance_m=dict(value=[.02,.02])))))
        reader=PolicyOwnedTripodApproach.__new__(PolicyOwnedTripodApproach)
        reader.geometry=Geometry();reader.scope="cpu_fixture";reader._metric=metric
        self.distance=reader

    monkeypatch.setattr(m.NativeContactTripodSources,"_configure_distances",configure)
    _,_,template,_=raw_cpu_packets(profile)
    rod_config=json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_text())["rod"]
    def parameter_packet(row):
        packet=deepcopy(template);clock={k:row[k] for k in template["identity"] if k in row}
        packet.update(identity=clock,source_instance_id=id(c.rope),source_scope="cpu_fixture",
            sources=policy_source_basis(profile),node_positions_m=np.array(row["node_positions"],copy=True))
        raw=packet["raw_native_parameters"]
        rest=np.asarray(row["rest_lengths_m"])
        masses=np.r_[rest[:1]/2,(rest[:-1]+rest[1:])/2,rest[-1:]/2]*rod_config["linear_density_kg_m"]
        raw.update(rest_lengths_m=rest[None].copy(),inverse_masses_kg_inv=(1/masses)[None],time_s=clock["solver_time_s"])
        raw["stored_k_stretch_n_m"]=np.asarray(raw["stretch_ea_n"])*rest[None]
        raw["rest_quaternion_norm_n_m"]=4*np.asarray(raw["bend_ei_n_m2"])/rest[None]
        packet["native_parameters"]=validate_snapshot(raw,expected_num_envs=1,expected_nodes=41,expected_time_s=clock["solver_time_s"])
        packet["reward_radius_sample"]["source"]["identity"]=clock
        packet["reward_radius_sample"]["value"]=float(raw["contact_radius_m"])
        return packet
    class Getter:
        def __init__(self,*args,**kwargs):pass
        def capture_reset(self,row,**kwargs):return parameter_packet(row)
        def capture(self,row,**kwargs):return parameter_packet(row)
    monkeypatch.setattr(parameters,"NativeParameterEndpoints",Getter)
    monkeypatch.setitem(sys.modules,"ropesim._ropesim",SimpleNamespace())
    source=m.NativeContactTripodSources(owner,contract,scope="cpu_fixture");owner.sources=source
    monkeypatch.setattr(source,"_health",lambda clock:make_solver_status(clock,0,source_scope="cpu_fixture",
        source_instance_id=id(c.rope),backend_sha256=profile.binary_sha256))
    complete=ep._exchange._complete_substep
    def contacted(pending,sdk):
        row=complete(pending,sdk)
        hand_links = ("finger14_link","finger23_link","finger33_link") if wire_trace_geometry else None
        data=build(proximal=True,table=table_at is not None and row["rod"]["physics_step"]>=table_at,load=.125,
                   hand_contact_links=hand_links)
        if controls != 1:
            # The actual getter exports float32 node force buffers. Construct
            # this CPU buffer in that dtype before recording or snapshotting.
            data["distributed_contact"]=np.asarray(data["distributed_contact"],np.float32)
        for key in ("distributed_contact","named_wrenches","table_wrench_world",
                    "external_support_wrenches_world","rod_mesh_contact_pairs"):
            row["rod"][key]=deepcopy(data[key])
        row["rod"]["rod_mesh_contact_pairs"].update(physics_step=row["rod"]["physics_step"],rod_tick=row["rod"]["rod_tick"])
        row["rod"]["source_scope"]="cpu_fixture"
        return row
    ep._exchange._complete_substep=contacted
    reset["reset"]["rod_reset"]["source_scope"]="cpu_fixture"
    with profile_reader_scope(profile):
        source.after_reset(reset)
        if bad_prime_parameter:
            original_prime=pm.prime_supported_sources
            def changed_parameter(*args,**kwargs):
                result=original_prime(*args,**kwargs)
                source.parameters=deepcopy(source.parameters)
                source.parameters["source_instance_id"]=-1
                return result
            monkeypatch.setattr(pm,"prime_supported_sources",changed_parameter)
        source.initialize_context(reset)
    config,*_=model_setup("tripod")
    model=NativeContextTeacher(config,backend_profile=profile,context_mode=PROFILE)
    trainer=DeformablePPOTrainer(model,PPOConfig(gamma=.995,epochs=1,minibatches=1,lr=.0001),
        rollout_steps=controls,num_envs=1)
    if lift_diagnostic:
        name = "go2_pvcnn.control.dh116.tabletop_teacher_lift_reward"
        assert importlib.util.find_spec(name) is not None, "Lift diagnostic reward is missing"
        from go2_pvcnn.control.dh116.tabletop_teacher_lift_reward import SupportedLiftDiagnosticReward
        from go2_pvcnn.control.dh116.tabletop_supported_lift_observation import FixedLiftReference
        reference = FixedLiftReference.from_rod_reset(reset["reset"]["rod_reset"], radius_m=.005, table_top_m=.3)
        reward = SupportedLiftDiagnosticReward(reward_config, contact_owner=owner.context,
            reference=reference, priming_event=history.priming_event, prelift_stable_s=1/480)
    else:
        reward=TrainingReward(reward_config,contact_owner=owner.context)
    collector=NativeTeacherCollector(owner,trainer,reward,SimpleNamespace())
    return collector,c,runtime,rows


def test_original_collector_uses_primed_source_then_real_ppo_update(monkeypatch,tmp_path,backend_profile):
    collector,c,budget,rows=arrange(monkeypatch,tmp_path,backend_profile)
    assert c.sim.current_time_step_index==3 and budget.solver_calls==1
    prime=next(json.loads(line)["payload"]["distance"] for line in collector.journal.path.read_text().splitlines()
               if json.loads(line)["operation"]=="primed_sources")
    assert prime["auxiliary"]["samples"]["approach_distance_m"]["source"].endswith(prime["source_sha256"])
    assert not len(collector.trainer.rollout)
    packet=collector.collect()
    assert c.sim.current_time_step_index==11 and budget.solver_calls==9
    assert packet.post_context["contact_s"]==pytest.approx(8/480)
    assert collector.context.current.values[0,6]==pytest.approx((8/480)/.25)
    assert packet.terms["contact"]["raw"]==1. and packet.event==0.
    assert collector.transitions==1 and not collector.quarantined
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_inputs import audit_teacher_input_sources
    audit=audit_teacher_input_sources(collector.journal.path.read_bytes(),backend_profile=backend_profile,
        scope="cpu_fixture",controls=1,grasp="tripod",profile=PROFILE)
    assert audit["decision_packs"]==audit["bootstrap_packs"]==1
    changed=[json.loads(line) for line in collector.journal.path.read_text().splitlines()]
    prime=next(row["payload"] for row in changed if row["operation"]=="owner_event"
               and row["payload"].get("kind")=="priming_physical")
    prime["exchange"]["rod"]["mesh_bindings"]["left.base_link"]["capture_sha256"]="c"*64
    # Reject the incompatible prime layout before trusting any later receipt hash.
    with pytest.raises(ValueError,match="mid-step contact source bindings changed"):
        audit_teacher_input_sources(b"".join((json.dumps(row)+"\n").encode() for row in changed),
            backend_profile=backend_profile,scope="cpu_fixture",controls=1,grasp="tripod",profile=PROFILE)
    damaged=[json.loads(line) for line in collector.journal.path.read_text().splitlines()]
    prime_row=next(row for row in damaged if row["operation"]=="owner_event"
                   and row["payload"].get("kind")=="priming_physical")
    prime_row["payload"]["kind"]="unknown_not_priming"
    bad=b"".join((json.dumps(row)+"\n").encode() for row in damaged)
    with pytest.raises(ValueError,match="prim"):
        audit_teacher_input_sources(bad,backend_profile=backend_profile,scope="cpu_fixture",
            controls=1,grasp="tripod",profile=PROFILE)
    # A self-consistent forged source pack and reward timer must still fail
    # independent replay of the original contact rows.
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned,_digest
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_inputs import _context,_rebuild
    forged=[json.loads(line) for line in collector.journal.path.read_text().splitlines()]
    settlement=next(row["payload"] for row in forged if row["operation"]=="settlement")
    bootstrap=settlement["bootstrap_source"]
    pack=forged[bootstrap["source_pack_sequence"]]["payload"]
    buffers=decode_owned(pack["buffers"])
    buffers["context"]["values"][0,6]+=1/480/.25
    buffers["observation"]=_rebuild(buffers,backend_profile,_context(buffers["context"]),PROFILE)
    pack["buffers"]=encode_owned(buffers)
    pack["sha256"]={key:_digest(value) for key,value in buffers.items()}
    bootstrap["observation"]=encode_owned(buffers["observation"])
    bootstrap["observation_sha256"]=_digest(buffers["observation"])
    settlement["post_context"]=encode_owned(buffers["context"])
    settlement["reward"]["post_context"]["contact_s"]+=1/480
    forged_raw=b"".join((json.dumps(row)+"\n").encode() for row in forged)
    with pytest.raises(ValueError,match="post-contact Context differs"):
        audit_teacher_input_sources(forged_raw,backend_profile=backend_profile,scope="cpu_fixture",
            controls=1,grasp="tripod",profile=PROFILE)
    stats=collector.update()
    assert stats["samples"]==1 and collector.trainer.counters["updates"]==1
    assert not packet.teacher_admission


def test_table_third_policy_substep_is_persisted_before_history_and_reward_terminal(monkeypatch,tmp_path,backend_profile):
    collector,c,budget,rows=arrange(monkeypatch,tmp_path,backend_profile,table_at=6)
    packet=collector.collect()
    assert packet.terminated and packet.event==-40. and packet.endpoint_phi==0.
    assert packet.duration_s==pytest.approx(3/480)
    assert c.sim.current_time_step_index==6 and budget.solver_calls==4
    assert collector.trainer.rollout.batch()["bootstrap_mask"].item() is False
    event=[e for e in rows if e["kind"]=="physical"][-1]
    assert event["exchange"]["guard_failure"]=="native_teacher_table_contact"
    assert not collector.quarantined and collector.episode_done
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_inputs import audit_teacher_input_sources
    audit=audit_teacher_input_sources(collector.journal.path.read_bytes(),backend_profile=backend_profile,
        scope="cpu_fixture",controls=1,grasp="tripod",profile=PROFILE)
    assert audit["decision_packs"]==1 and audit["bootstrap_packs"]==0


def test_missing_real_middle_distance_poisoning_never_fills_a_ppo_transition(monkeypatch,tmp_path,backend_profile):
    collector,c,budget,rows=arrange(monkeypatch,tmp_path,backend_profile,bad_middle=True)
    with pytest.raises((ValueError,RuntimeError),match="distance"):
        collector.collect()
    assert collector.quarantined and collector.trainer._pending is not None
    assert len(collector.trainer.rollout)==0 and c.sim.current_time_step_index==4
    with pytest.raises(RuntimeError):
        collector.update()


def test_foreign_parameter_packet_is_rejected_before_contact_duration_advances(monkeypatch,tmp_path,backend_profile):
    collector,c,budget,rows=arrange(monkeypatch,tmp_path,backend_profile)
    source=collector.owner.sources
    observe=source.observe
    def foreign(event):
        source.parameters=deepcopy(source.parameters)
        source.parameters["source_instance_id"]=-1
        return observe(event)
    source.observe=foreign
    with pytest.raises((ValueError,RuntimeError),match="parameter"):
        collector.collect()
    assert collector.context._observed==1  # Only the real source prime, no invalid policy contact.
    assert collector.quarantined and len(collector.trainer.rollout)==0


def test_last_substep_table_failure_cannot_be_forged_as_live_bootstrapped_window(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned,_digest
    from go2_pvcnn.control.dh116.tabletop_deformable_rollout_admission import gae_boundary
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_inputs import audit_teacher_input_sources
    collector,c,budget,rows=arrange(monkeypatch,tmp_path,backend_profile,table_at=11)
    packet=collector.collect()
    assert packet.terminated and packet.event==-40. and c.sim.current_time_step_index==11
    # Explicit hostile CPU artifact fabrication, never a live episode continuation.
    episode=collector.owner.episode
    state=episode._state
    try:
        episode._state="active"
        pre=episode.pre_action_state()
    finally:
        episode._state=state
    observation,binding=collector.owner.sources.pack_recorded(collector.trainer.model,pre,
        collector.context.current,purpose="bootstrap")
    damaged=[json.loads(line) for line in collector.journal.path.read_text().splitlines()]
    forged_pack=damaged.pop()
    assert forged_pack["operation"]=="teacher_input_source"
    index=next(i for i,row in enumerate(damaged) if row["operation"]=="settlement")
    damaged.insert(index,forged_pack)
    for seq,row in enumerate(damaged):
        row["sequence"]=seq
    settlement=damaged[index+1]["payload"]
    decision=next(row["payload"] for row in damaged if row["operation"]=="decision")
    last=next(row["payload"] for row in reversed(damaged) if row["operation"]=="physical")
    endpoint={k:last["exchange"]["rod"][k] for k in
        ("env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s")}
    endpoint["phase"]="post_action_pre_reset"
    settlement["reward"].update(terminated=False,event=0.,event_reasons=[])
    settlement["bootstrap_source"]=dict(collection_token=decision["collection_token"],identity=endpoint,
        source_pack_sequence=index,pre_action_sha256=_digest(pre),
        observation=encode_owned(observation),observation_sha256=_digest(observation))
    settlement["gae_boundary"]=gae_boundary(False,False,endpoint_identity=endpoint,next_value_identity=endpoint)
    raw=b"".join((json.dumps(row)+"\n").encode() for row in damaged)
    with pytest.raises(ValueError,match="table contact requires"):
        audit_teacher_input_sources(raw,backend_profile=backend_profile,scope="cpu_fixture",
            controls=1,grasp="tripod",profile=PROFILE)


def test_priming_parameter_source_cannot_differ_from_recorded_getter(monkeypatch,tmp_path,backend_profile):
    with pytest.raises((ValueError,RuntimeError),match="parameter"):
        arrange(monkeypatch,tmp_path,backend_profile,bad_prime_parameter=True)

def test_original_collector_consumes_measured_lift_diagnostic_reward(monkeypatch, tmp_path, backend_profile):
    collector, fixture, _, _ = arrange(monkeypatch, tmp_path, backend_profile, lift_diagnostic=True)
    original_step = fixture.rope.step_free

    def lifted_fixture_step(dt, substeps):
        original_step(dt, substeps)
        if fixture.sim.current_time_step_index == 8:
            fixture.rope.nodes[..., 2] += .015

    fixture.rope.step_free = lifted_fixture_step
    packet = collector.collect()
    assert packet.profile == "supported_lift_diagnostic_v2"
    assert packet.post_context["lift_phase"] == "Lift"
    assert packet.terms["lift"]["contribution"] == pytest.approx(2., abs=1e-5)
    assert packet.reward == pytest.approx(.995 * packet.phi_post_raw - packet.phi_pre - packet.integrated_cost)
    assert packet.event == 0 and not packet.terminated
    assert packet.post_context["hold_s"] is None
    assert packet.post_context["lift_shaping_delta"] == pytest.approx(1.99, abs=1e-5)
    assert len(collector.trainer.rollout) == 1
    metrics = collector.trainer.update()
    assert metrics["minibatch_updates"] == 1



def test_wire_trace_binds_previous_control_to_bootstrap_and_next_decision(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_provider import _trace_actual_wire_feedback

    collector,c,budget,_=arrange(monkeypatch,tmp_path,backend_profile,controls=2,wire_trace_geometry=True)
    with _trace_actual_wire_feedback(collector,tmp_path,profile=PROFILE):
        first=collector.collect()
        second=collector.collect()
    journal=[json.loads(line) for line in (tmp_path/"wire_control_feedback.jsonl").read_text().splitlines()]
    assert all(row["scope"]=="cpu_fixture" and row["operation"]=="wire_control_feedback" for row in journal)
    traces=[row["payload"] for row in journal]
    assert first.done is False and second.truncated is True
    assert collector.transitions == 2 and budget.solver_calls == 17
    decision, bootstrap, next_decision, final_bootstrap = traces
    assert [t["purpose"] for t in traces] == ["decision","bootstrap","decision","bootstrap_truncated"]
    assert [t["clock"]["physics_step"] for t in traces] == [3,11,11,19]
    assert decision["status"] == "no_previous_control_window"
    assert not np.any(decision["observation"]["reaction_valid"])
    assert np.isfinite(decision["observation"]["position_in_palm_m"]).all()
    assert bootstrap["status"] == next_decision["status"] == "measured"
    assert bootstrap["pre_action_sha256"] == next_decision["pre_action_sha256"]
    assert bootstrap["observation_sha256"] == next_decision["observation_sha256"]
    assert bootstrap["source_window"]["end_sdk_step"] == 11
    assert bootstrap["source_window"]["first_sdk_step"] == 4
    assert final_bootstrap["source_window"]["first_sdk_step"] == 12
    batch=collector.trainer.rollout.batch()
    assert {"common371","common_valid","rod_nodes","rod_mask","task_context","task_context_valid"} == set(batch["actor_obs"])
    assert batch["actions"].shape[-1] == 24


def test_wire_trace_rejects_disagreeing_palms_without_serving_a_feedback_packet(monkeypatch,tmp_path,backend_profile):
    collector,_,_,_=arrange(monkeypatch,tmp_path,backend_profile,wire_trace_geometry=True,wire_pose_error_m=.001)
    records=[]
    with collector.trace_wire_feedback(lambda _,payload:records.append(deepcopy(payload))):
        packet=collector.collect()
    assert not packet.terminated and not collector.quarantined
    decision,bootstrap=records
    assert decision["status"] == "no_previous_control_window"
    assert bootstrap["status"] == "rejected"
    assert "palm pose differs" in bootstrap["reason"]
    assert bootstrap["observation"] is None


def test_wire_trace_never_publishes_a_partial_terminal_window(monkeypatch,tmp_path,backend_profile):
    collector,_,budget,_=arrange(monkeypatch,tmp_path,backend_profile,table_at=6,wire_trace_geometry=True)
    records=[]
    with collector.trace_wire_feedback(lambda _,payload:records.append(deepcopy(payload))):
        packet=collector.collect()
    assert packet.terminated is True and budget.solver_calls == 4
    assert len(records) == 1 and records[0]["status"] == "no_previous_control_window"



def test_wire_trace_clears_partial_source_window_when_the_collector_quarantines(monkeypatch,tmp_path,backend_profile):
    collector,_,budget,_=arrange(monkeypatch,tmp_path,backend_profile,wire_trace_geometry=True)
    source=collector.sources
    observe=source.observe

    def fail_third(row):
        if row["exchange"]["substep"] == 2:
            raise ValueError("fixture unavailable on third physical row")
        return observe(row)

    monkeypatch.setattr(source,"observe",fail_third)
    traces=[]
    with collector.trace_wire_feedback(lambda _,payload:traces.append(deepcopy(payload))):
        wire_trace=collector._wire_feedback
        with pytest.raises(ValueError,match="fixture unavailable"):
            collector.collect()
        assert not wire_trace._rows and wire_trace._bootstrap is None
    assert budget.solver_calls == 4 and collector.quarantined
    assert len(traces) == 1 and traces[0]["status"] == "no_previous_control_window"
