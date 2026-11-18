"""Controlled CPU continuation through original control and guarded exchange."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import DT
from go2_pvcnn.control.dh116.tabletop_native_training_clock import ClockPolicyV2
from go2_pvcnn.control.dh116.tabletop_mesh_reset_readback_v2 import MeasuredMeshResetV2
from go2_pvcnn.control.dh116.tabletop_native_teacher_session import SessionBudget
from go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard import ClearanceGuardedExchange


@dataclass
class LongEpisodeFixture:
    sim: object
    rod: object
    owners: dict
    seed: dict
    rows: list
    clock_policy: object
    episode: object
    mesh: object
    budget: object
    original: object
    tmp_path: object

    def reset(self, uid):
        from go2_pvcnn.control.dh116.tabletop_native_episode_v2 import (
            FixedS0PhysicalEpisodeV2, guarded_exchange_factory_v2)
        old = getattr(self, "episode", None)
        self.episode = FixedS0PhysicalEpisodeV2(self.sim, self.rod, self.original._bindings,
            backend_artifact=self.tmp_path/"solver.so",
            backend_sha256=hashlib.sha256((self.tmp_path/"solver.so").read_bytes()).hexdigest(),
            inspect_geometry=self.original._inspect_geometry, physics_budget_end=3000,
            record=self.record, reset_factory=MeasuredMeshResetV2,
            exchange_factory=guarded_exchange_factory_v2, clock_policy=self.clock_policy,
            mesh_plan=self.mesh.mesh_plan)
        self.budget.reserve_reset(SimpleNamespace(sim=self.sim,rod=self.rod))
        from go2_pvcnn.control.dh116.tabletop_native_teacher_action_history import NativeAction24HistoryOwner
        from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseOwnerV2
        prior=self.rod.clock() if self.rod.clock()["reset_id"] else None
        self.history=NativeAction24HistoryOwner(initial_action24=torch.zeros((1,24)),
            seed_source="explicit CPU fixture zero controller memory",physics_budget_end=3000,
            solver_substeps=12,previous_reset=prior)
        owners=dict(self.owners)
        controller=self.history.wrap_controller_reset(owners["controller"])
        if not hasattr(self,"phase"):self.phase=PhaseOwnerV2(prelift_stable_s=.25)
        def reset_memory(context):
            self.phase.reset(episode_uid=uid,physics_tick=context["physics_step"])
            return controller(context)
        owners["controller"]=reset_memory
        evidence = self.episode.reset(self.seed,episode_uid=uid,owner_resets=owners)
        self.history.record(evidence)
        self.reset_context()
        self.old_episode=old
        return evidence["reset"]

    def reset_context(self):
        from pathlib import Path
        from test_native_teacher_context_v2 import fields
        from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
        from go2_pvcnn.control.dh116.tabletop_native_teacher_context_v2 import TaskContextOwnerV2,ContextIdentity
        profile=BackendProfile.load(Path(__file__).resolve().parents[1])
        if hasattr(self,"context"):self.context.close()
        self.context=TaskContextOwnerV2(backend_profile=profile,mode="fixed_s0_v2",
            scope="cpu_fixture",episode_cap_s=30,external_cap_s=600,prelift_scale_s=.25)
        identity=ContextIdentity.from_clock(self.episode._read_state_clock())
        state=dict(identity=identity,source_scope="cpu_fixture",
            backend_sha256=profile.binary_sha256,
            node_positions=torch.as_tensor(self.rod.rope.positions()).clone(),
            distance_mean_m=.02,rest_length_m=.5,target_reference_uid="cpu-fixture-target",
            fixed_s0_manifest_sha256="1"*64,
            external_budget=dict(remaining_s=self.budget.remaining_s(),cap_s=600.,
                source="external_budget_clock",identity=identity),
            physical_fields=fields(identity,self.phase))
        self.context.reset_from_verified_state(state,history_owner=self.history,phase_owner=self.phase)

    def record(self, row):
        if row["kind"]=="physical":
            self.budget.charge(row["exchange"],12)
        self.rows.append(deepcopy(row))

    def control_zero(self):
        return self.episode.control(torch.zeros((1,24)))

    def advance_to(self, target):
        assert self.sim.current_time_step_index==2
        count={240:29,258:32,2402:300}[target]
        for _ in range(count):
            result=self.control_zero()
            assert result["physical_samples"]==8 and result["termination_reason"] is None
        if target==240:
            step=self.sim.step
            def terminal(*,render):
                step(render=render)
                if self.sim.current_time_step_index==240:
                    hand=self.mesh.hands[1]
                    hand.measured[0,hand.joint_names.index("finger12")]=-.05
            self.sim.step=terminal
            try:
                result=self.control_zero()
            finally:
                self.sim.step=step
            assert result["physical_samples"]==6
            assert result["termination_reason"]=="right:joint_limit"
        assert self.sim.current_time_step_index==target


def long_episode_fixture(monkeypatch,tmp_path):
    # Use precisely the original physical-episode hand/rod/SDK fixture.
    from test_dh116_dual_hand_episode import setup
    from test_mesh_reset_v2 import long_mesh_fixture
    import test_mesh_reset_order
    original,rod,hands,sim,seed,owners,_=setup(monkeypatch,tmp_path)
    for hand in hands:
        hand.data.body_link_pose_w=hand.link_poses[..., [0,1,2,6,3,4,5]]
        hand.data.body_lin_vel_w=torch.zeros((1,23,3))
        hand.data.body_ang_vel_w=torch.zeros((1,23,3))
        hand.root_physx_view.get_link_velocities=lambda:torch.zeros((1,23,6))
        update=hand.update
        def refresh(dt,hand=hand,update=update):
            update(dt)
            hand.data.body_link_pose_w=hand.link_poses[..., [0,1,2,6,3,4,5]]
        hand.update=refresh
    step=sim.step
    monkeypatch.setattr(test_mesh_reset_order,"setup",lambda mp:(
        hands[0].events,rod,hands,original._bindings,sim,seed,owners))
    mesh=long_mesh_fixture(monkeypatch)
    sim.step=step  # declared double-DT fixture; real increments, never endpoint repair
    budget=SessionBudget(dict(max_sdk_steps=3000,max_solver_calls=3000,
        max_resets=8,max_episode_count=8,wall_timeout_s=600))
    owner=SimpleNamespace(sim=sim,rod=rod)
    # setup() starts from original two-step boundary; record its actual endpoint once.
    budget.record_setup(owner,[dict(kind="setup",sdk_step=0,sdk_time_s=0.),
        dict(kind="setup",sdk_step=sim.current_time_step_index,sdk_time_s=sim.current_time)])
    c=LongEpisodeFixture(sim,rod,owners,seed,[],
        ClockPolicyV2(9600,"float64_dt_float64_sum"),None,mesh,budget,original,tmp_path)
    c.reset("fixture:first")
    return c


@pytest.mark.parametrize("target",[240,258,2402])
def test_reset_after_real_fixture_progress_then_next_episode(monkeypatch,tmp_path,target):
    c=long_episode_fixture(monkeypatch,tmp_path)
    c.advance_to(target)
    before=(c.sim.current_time_step_index,c.sim.current_time,c.rod.rope.time,c.rod.rope.steps)
    cost=(c.budget.sdk_steps,c.budget.solver_calls,c.budget.internal_solver_substeps)
    epoch=c.mesh.mesh_epoch
    receipt=c.reset("fixture:next")
    assert before==(c.sim.current_time_step_index,c.sim.current_time,c.rod.rope.time,c.rod.rope.steps)
    assert cost==(c.budget.sdk_steps,c.budget.solver_calls,c.budget.internal_solver_substeps)
    assert cost==(target,target-2,(target-2)*12)
    assert c.mesh.mesh_epoch==epoch+1
    assert receipt["mesh_reset"]["new_sdk_steps"]==0
    assert receipt["mesh_reset"]["borrowed_mesh_views_invalidated"] is True
    assert receipt["after_clock"]["reset_id"]==2 and receipt["after_clock"]["rod_tick"]==0
    assert receipt["after_clock"]["episode_uid"]=="fixture:next"
    assert type(c.episode._exchange) is ClearanceGuardedExchange
    assert c.episode._control_index==0
    assert c.episode.pre_action_state()["common"]["identity"]["reset_id"]==2
    c.control_zero()
    assert c.sim.current_time_step_index==target+8
    assert c.budget.sdk_steps==target+8 and c.budget.solver_calls==target+6
    assert c.budget.internal_solver_substeps==(target+6)*12
    assert c.budget.resets==c.budget.episodes==2 and c.budget.setup_sdk_steps==2
    assert all(row["native_qualification"] is False for row in c.rows)
    import json
    print(json.dumps(dict(scope="cpu_fixture_original_control_continuation",
        target=target,actual_sdk_before_reset=dict(step=before[0],timestamp_s=before[1]),
        solver_before_reset=dict(time_s=before[2],calls=before[3]),
        reset_new_sdk_steps=receipt["new_sdk_steps"],reset_new_solver_steps=receipt["new_solver_steps"],
        next_episode_sdk_step=c.sim.current_time_step_index,
        next_episode_actual_sdk_timestamp_s=c.sim.current_time,
        next_episode_rod_nominal_timestamp_s=c.rod.clock()["timestamp_s"],
        session_sdk_steps=c.budget.sdk_steps,session_solver_calls=c.budget.solver_calls,
        session_internal_solver_substeps=c.budget.internal_solver_substeps,
        native_steps=0,native_ready=False),sort_keys=True))
    if target==240:
        with pytest.raises(RuntimeError,match="active"):c.old_episode.control(torch.zeros((1,24)))


def test_v2_inherits_original_control_and_pre_action_algorithm():
    from go2_pvcnn.control.dh116.tabletop_native_episode_v2 import FixedS0PhysicalEpisodeV2
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
    assert FixedS0PhysicalEpisodeV2.control is DualHandPhysicalEpisode.control
    assert FixedS0PhysicalEpisodeV2.pre_action_state is DualHandPhysicalEpisode.pre_action_state


def test_post_sdk_clearance_guard_stops_third_step_without_fourth(monkeypatch,tmp_path):
    c=long_episode_fixture(monkeypatch,tmp_path)
    step=c.sim.step
    def lower(*,render):
        step(render=render)
        if c.sim.current_time_step_index==5:
            c.mesh.hands[1].link_poses[0,:,:3][:,2]=.25
    c.sim.step=lower
    result=c.control_zero()
    assert result["physical_samples"]==3
    assert result["termination_reason"]=="post_step_source_clearance:right"
    assert c.sim.current_time_step_index==5 and c.rod.rope.steps==3
    assert c.budget.solver_calls==3 and c.budget.internal_solver_substeps==36
    with pytest.raises(RuntimeError,match="active"):c.control_zero()


@pytest.mark.parametrize("bad",["wrapper","bare","reset","source"])
def test_factory_and_source_rejected_before_sdk_or_reset(monkeypatch,tmp_path,bad):
    from go2_pvcnn.control.dh116 import tabletop_native_episode_v2 as v2
    c=long_episode_fixture(monkeypatch,tmp_path)
    before=(c.sim.current_time_step_index,c.rod.rope.resets)
    kw=dict(reset_factory=MeasuredMeshResetV2,exchange_factory=v2.guarded_exchange_factory_v2)
    if bad=="wrapper":
        kw["exchange_factory"]=lambda *a,**k:v2.guarded_exchange_factory_v2(*a,**k)
    if bad=="bare":
        from go2_pvcnn.control.dh116.tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
        kw["exchange_factory"]=FlexiblePhysicalExchange
    if bad=="reset":kw["reset_factory"]=lambda *a,**k:MeasuredMeshResetV2(*a,**k)
    if bad=="source":monkeypatch.setattr(v2,"_PINNED_GUARD_SOURCES",{})
    with pytest.raises(ValueError,match="factory|source"):
        v2.FixedS0PhysicalEpisodeV2(c.sim,c.rod,c.original._bindings,
            backend_artifact=tmp_path/"solver.so",
            backend_sha256=hashlib.sha256((tmp_path/"solver.so").read_bytes()).hexdigest(),
            inspect_geometry=c.original._inspect_geometry,physics_budget_end=3000,
            record=c.record,mesh_plan=c.mesh.mesh_plan,clock_policy=c.clock_policy,**kw)
    assert before==(c.sim.current_time_step_index,c.rod.rope.resets)


def test_preflight_rejects_caller_manifest_without_touching_scene():
    from go2_pvcnn.control.dh116.tabletop_native_reset_preflight_v2 import ResetPreflightV2
    with pytest.raises(ValueError,match="ResolvedResetManifest"):
        ResetPreflightV2.require_reset_admitted(object(),
            clock_policy=ClockPolicyV2(9600,"float64_dt_float64_sum"),manifest={})


def test_quarantined_advanced_row_charged_once_and_never_refunded(monkeypatch,tmp_path):
    c=long_episode_fixture(monkeypatch,tmp_path)
    def quarantine(row):
        c.record(row)
        if row["kind"]=="physical":
            raise OSError("fixture durable row quarantined after charge")
    c.episode._record=quarantine
    with pytest.raises(OSError,match="quarantined"):c.control_zero()
    assert (c.budget.sdk_steps,c.budget.solver_calls,c.budget.internal_solver_substeps)==(3,1,12)
    before=(c.sim.current_time_step_index,c.rod.rope.steps,c.budget.sdk_steps)
    c.reset("fixture:after-quarantine")
    assert before==(c.sim.current_time_step_index,c.rod.rope.steps,c.budget.sdk_steps)
    c.control_zero()
    assert (c.budget.sdk_steps,c.budget.solver_calls,c.budget.internal_solver_substeps)==(11,9,108)


@pytest.fixture
def actual_manifest():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import candidate_config,load_reset_manifest
    config=candidate_config()
    config["resolved_reset_manifest"]=dict(
        path="/home/xk/.local/share/dh116-autonomous-3c80166/evidence/c06-retained-s0-resolution-rjljko77/resolved-reset-manifest.json",
        sha256="a39de62de8ea97b3b154af2f910ba9dc097ccf64f9d650dad627ee2d8516cdce")
    return load_reset_manifest(config)


def test_actual_s0_source_manifest_and_v2_clock_admission_remain_distinct(
        monkeypatch,tmp_path,actual_manifest):
    from go2_pvcnn.control.dh116.tabletop_native_reset_preflight_v2 import ResetPreflightV2,validate_live_reset_v2
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    from pathlib import Path
    c=long_episode_fixture(monkeypatch,tmp_path)
    owner=SimpleNamespace(episode=c.episode,sim=c.sim,rod=c.rod,bindings=c.episode._bindings)
    gate=ResetPreflightV2.require_reset_admitted(owner,
        clock_policy=c.clock_policy,manifest=actual_manifest)
    assert gate["reset_clock_admitted"] and gate["reset_epoch_next"]==2
    assert not gate["native_ready"]
    owner.backend_profile=BackendProfile.load(Path(__file__).resolve().parents[1])
    receipt=c.rows[0]["reset"]
    # A valid historical S0 plus CPU numeric packets cannot masquerade as a live native Reset.
    with pytest.raises(ValueError):
        validate_live_reset_v2(owner=owner,receipt=receipt,
            numeric_audit=receipt["numeric_reset_audit"],mesh_plan=c.mesh.mesh_plan,
            manifest=actual_manifest,clock_policy=c.clock_policy)


def test_reset_rechecks_factory_binding_before_any_mutation(monkeypatch,tmp_path):
    c=long_episode_fixture(monkeypatch,tmp_path)
    from go2_pvcnn.control.dh116 import tabletop_native_episode_v2 as v2
    episode=v2.FixedS0PhysicalEpisodeV2(c.sim,c.rod,c.original._bindings,
        backend_artifact=tmp_path/"solver.so",
        backend_sha256=hashlib.sha256((tmp_path/"solver.so").read_bytes()).hexdigest(),
        inspect_geometry=c.original._inspect_geometry,physics_budget_end=3000,
        record=c.record,mesh_plan=c.mesh.mesh_plan,clock_policy=c.clock_policy,
        reset_factory=MeasuredMeshResetV2,exchange_factory=v2.guarded_exchange_factory_v2)
    episode._exchange_factory=lambda *a,**k:None
    before=(c.sim.current_time_step_index,c.rod.rope.resets)
    with pytest.raises(ValueError,match="factory"):
        episode.reset(c.seed,episode_uid="fixture:bad-late",owner_resets=c.owners)
    assert before==(c.sim.current_time_step_index,c.rod.rope.resets)


def test_real_w3_cpu_memory_owners_are_fresh_after_long_reset(monkeypatch,tmp_path):
    c=long_episode_fixture(monkeypatch,tmp_path)
    old_history=c.history
    old_context=c.context
    c.advance_to(258)
    c.reset("fixture:w3-next")
    assert c.history._owner_uid!=old_history._owner_uid
    assert c.context.owner_uid!=old_context.owner_uid and old_context._closed
    packet=c.context.current
    assert packet.metadata.identities[0].episode_uid=="fixture:w3-next"
    assert packet.metadata.identities[0].reset_id==2
    assert torch.equal(packet.values[0,:4],torch.tensor([1.,0.,0.,0.]))
    assert packet.values[0,7]==packet.values[0,50]==0
    assert packet.values[0,43]==0
    assert not packet.valid[0,8:10].any()
    assert torch.equal(packet.values[:,12:36],torch.zeros((1,24)))
    assert c.context._anchor_source is None if hasattr(c.context,"_anchor_source") else True
    assert c.history._reset["identity"]["reset_id"]==2
    assert not packet.native_qualified


def test_episode_factory_does_not_accept_caller_clock_ready(monkeypatch,tmp_path):
    c=long_episode_fixture(monkeypatch,tmp_path)
    assert c.episode.clock_consumers_ready is False
    assert c.episode.native_ready is False
    with pytest.raises(AttributeError):c.episode.clock_consumers_ready=True
    gate=c.episode.clock_readiness()
    assert gate["source_inventory"]==c.episode.source_inventory
    assert any(x.startswith("long_clock_consumer_incompatible:") for x in gate["blocked_reasons"])
