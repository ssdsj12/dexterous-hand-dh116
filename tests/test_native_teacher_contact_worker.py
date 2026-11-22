"""Contact worker/profile contracts over explicit CPU transport, never an execution lease."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from test_native_teacher_context import backend_profile
from test_native_teacher_contact_sources import arrange,PROFILE,ROOT
from test_native_teacher_worker_contract import recipe
from test_native_teacher_worker_semantic import config as semantic_config,binding
from test_native_teacher_checkpoint import contract
from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_contract as worker

CONTACT_FILES={
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_source_priming.py",
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_contact_context.py",
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_contact_measurement.py",
    "src/go2_pvcnn/control/dh116/tabletop_power_contact_evidence.py",
    "configs/dh116_training/approach_contact_prelift_v1.json",
    "tests/test_native_teacher_contact_worker.py",
}


def contact_recipe(tmp_path,controls=8):
    spec=recipe(tmp_path)
    pin=spec["teacher_contract"]
    pin.update(profile=PROFILE,style="tripod")
    measured=8*controls
    limits=dict(worker.FIXED_LIMITS,controls=controls,measured_sdk_steps=measured,
        warmup_sdk_steps=1,max_solver_calls=1+measured,max_internal_solver_substeps=12*(1+measured),
        inner_wall_timeout_s=180,jsonl_limits=dict(worker.JSONL_LIMITS))
    if controls==8:
        limits["jsonl_limits"].update({"physics.jsonl":64,"transitions.jsonl":8,
            "ownership.jsonl":128,"training_events.jsonl":768})
    pin["limits"]=limits
    spec["budget"]["physics_steps"]=3+measured
    spec["validator"]["line_counts"]={"physics.jsonl":measured,"transitions.jsonl":controls}
    spec["source_files"]=sorted(set(spec["source_files"])|worker.TRIPOD_SOURCE_FILES|CONTACT_FILES)
    return spec


@pytest.mark.parametrize("controls",[8,32])
def test_contact_recipe_counts_a_separate_zero_action_source_prime_and_closes_sources(tmp_path,controls):
    spec=contact_recipe(tmp_path,controls)
    assert worker.validate_teacher_recipe_v2(spec)==spec
    for style in ("pinch","power"):
        damaged=deepcopy(spec);damaged["teacher_contract"]["style"]=style
        with pytest.raises(ValueError):
            worker.validate_teacher_recipe_v2(damaged)
    damaged=deepcopy(spec);damaged["budget"]["physics_steps"]-=1
    with pytest.raises(ValueError):
        worker.validate_teacher_recipe_v2(damaged)
    damaged=deepcopy(spec);damaged["source_files"].remove(next(iter(CONTACT_FILES)))
    with pytest.raises(ValueError,match="source|closure"):
        worker.validate_teacher_recipe_v2(damaged)


def test_contact_training_semantics_cannot_borrow_smoke_budget_or_namespace(tmp_path):
    from test_native_teacher_tripod_admission import config as tripod_config
    cfg,kw=semantic_config(tmp_path)
    pre=cfg["preflight"]
    pre.update(profile=PROFILE,style="tripod",resolved_reset_manifest=tripod_config()["resolved_reset_manifest"],
        reward_config=binding(ROOT/"configs/dh116_training/approach_contact_prelift_v1.json"))
    pre["budget"].update(max_sdk_steps=67,max_solver_calls=65)
    cfg["rollout_steps"]=8
    cfg["model_config"]=json.loads((ROOT/"configs/dh116_training/privileged_deformable_teacher_tripod_v1.json").read_text())
    semantic=worker.training_semantic_config(cfg,**kw)
    assert semantic["preflight"]["profile"]==PROFILE
    assert semantic["preflight"]["style"]=="tripod"
    bad=deepcopy(cfg);bad["preflight"]["budget"].update(max_sdk_steps=66,max_solver_calls=64)
    with pytest.raises(ValueError,match="prim|contact"):
        worker.training_semantic_config(bad,**kw)


def test_original_contact_collector_runs_eight_controls_and_seals_real_checkpoint(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_update import run_fresh_update
    from go2_pvcnn.control.dh116.tabletop_native_teacher_checkpoint import audit_checkpoint_bytes
    collector,c,budget,rows=arrange(monkeypatch,tmp_path,backend_profile,controls=8)
    identity=contract(collector.trainer)
    result=run_fresh_update(collector,checkpoint_contract=identity,controls=8)
    assert result["physical"]==dict(start_sdk_step=3,end_sdk_step=67,start_rod_tick=12,end_rod_tick=780)
    assert collector.consumed_sdk_steps==64 and budget.sdk_steps==67 and budget.solver_calls==65
    assert collector.trainer.counters=={"updates":1,"environment_steps":8}
    metadata=audit_checkpoint_bytes((tmp_path/"events.jsonl").parent.joinpath("teacher_clean.pt").read_bytes(),
        collector.trainer,identity,expected_origin={"kind":"fresh"})
    assert metadata["counters"]==collector.trainer.counters
    assert not result["teacher_admission"] and not result["native_qualification"]
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_artifacts as artifacts
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    journal=[json.loads(line) for line in collector.journal.path.read_text().splitlines()]
    monkeypatch.setattr(artifacts,"_required_native_source_scope",lambda:"cpu_fixture")
    result_training={"training":{k:result[k] for k in
        ("updates","accepted_minibatch_updates","parameter_delta_l2")}}
    artifacts._journal(journal,result_training,controls=8,grasp="tripod",profile=PROFILE)
    evidence=artifacts._physical_journal(journal,reward_scope="cpu_fixture",controls=8,
        expected_backend_sha256=backend_profile.binary_sha256,profile=PROFILE)
    assert evidence["first_sdk_endpoint"]==4 and evidence["last_sdk_endpoint"]==67
    assert evidence["measured_sdk_steps"]==64
    assert evidence["source_priming_sdk_steps"]==1
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_artifacts as actual
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_inputs as inputs
    setup=actual._audit_fresh_setup(journal,profile=PROFILE,end_sdk_step=67)
    assert setup["first_policy_sdk_step"]==3
    # Explicit synthetic close records test pure artifact derivation; no App or
    # process has run and outer exit verification remains false.
    lifecycle={
        "result.json":dict(acquisition=result,close_mode="normal",failure=None,
            process_exit_verified=False,native_qualification=False,lifetime_released_before_close=True),
        "cleanup_stages.json":dict(close_mode="normal",failure=None,qualification_granted=False,
            stages={stage:dict(attempted=True,returned=True,error=None) for stage in actual._CLOSE_STAGES}),
        "app_close_returned.json":dict(app_close_returned=True,close_mode="normal",
            process_exit_verified=False,qualification_granted=False),
        "stop_commit.json":dict(stop_commit_returned=True,explicit_physics_steps_requested=0)}
    raw_lifecycle={name:(json.dumps(value)+"\n").encode() for name,value in lifecycle.items()}
    raw_lifecycle["ownership.jsonl"]=b'{"scope":"cpu_fixture"}\n'
    spec=contact_recipe(tmp_path)
    files,derived,recorded,epoch=actual._derived_payloads(spec,raw_lifecycle,
        collector.journal.path.read_bytes(),(tmp_path/"teacher_clean.pt").read_bytes())
    assert len(files["physics.jsonl"].splitlines())==64
    assert len(files["transitions.jsonl"].splitlines())==8
    assert derived["process_exit_verified"] is False and derived["native_qualification"] is False
    monkeypatch.setattr(inputs,"SCOPE","cpu_fixture")  # Keep original CPU rows explicitly CPU.
    seal=actual._audit_sealed_packs(dict(rows=journal,result=derived,raw_journal=collector.journal.path.read_bytes()),
        backend_profile,controls=8,grasp="tripod",context_mode=PROFILE)
    assert seal["source_scope"]=="cpu_fixture" and seal["priming_packs"]==1


def test_contact_failure_keeps_partial_ledger_and_never_emits_clean_checkpoint(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_update import run_fresh_update,TrainingStopped
    collector,c,budget,rows=arrange(monkeypatch,tmp_path,backend_profile,controls=8,table_at=6)
    with pytest.raises(TrainingStopped):
        run_fresh_update(collector,checkpoint_contract=contract(collector.trainer),controls=8)
    assert c.sim.current_time_step_index==6 and budget.solver_calls==4
    assert collector.trainer.counters["updates"]==0
    assert not (tmp_path/"teacher_clean.pt").exists()
    events=[json.loads(line) for line in collector.journal.path.read_text().splitlines()]
    assert any(row["operation"]=="training_stopped" for row in events)
    assert not any(row["operation"]=="fresh_update_checkpoint_sealed" for row in events)


def test_provider_source_binder_recognizes_contact_sdk3_without_constructing_cuda(monkeypatch):
    # Typed CPU boundary doubles only: this does not construct a native getter,
    # qualify a source, issue an action or claim a real CUDA observation.
    import sys
    import weakref
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_provider import _prepare_source_pack
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import NativeContactTripodSources
    from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import NativeParameterEndpoints
    class Rope:pass
    rope=Rope()
    reader=object.__new__(NativeParameterEndpoints)
    reader.device="cpu";reader._rope_ref=weakref.ref(rope)
    clock=dict(run_uuid="cpu",env_id=0,reset_id=1,rod_uid="rod",physics_step=3,rod_tick=12,
        timestamp_s=3/480,solver_time_s=1/480,episode_uid="cpu:wiring")
    owner=SimpleNamespace(episode_uid="cpu:wiring",rod=SimpleNamespace(rope=rope,clock=lambda:dict(clock)),
        sim=SimpleNamespace(device="cpu",current_time_step_index=3),
        bindings=[SimpleNamespace(hand=SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(device="cpu")))) for _ in range(2)],
        history=SimpleNamespace(priming={"primed_identity":dict(clock)}))
    source=object.__new__(NativeContactTripodSources)
    source.owner=owner;source.scope="owner_source_unqualified";source.reader=reader
    source.parameters={"source_instance_id":id(rope)}
    owner.sources=source
    profile=object()
    model_contract=SimpleNamespace(config={"grasp":"tripod"},backend_profile=profile,context_mode=PROFILE)
    monkeypatch.setitem(sys.modules,"ropesim._ropesim",SimpleNamespace(RopeBatch=Rope))
    bind=_prepare_source_pack(owner,model_contract,device="cpu")
    bind(SimpleNamespace(_backend_profile=profile,context_mode=PROFILE))
    owner.sim.current_time_step_index=2
    with pytest.raises(ValueError,match="source pack"):
        _prepare_source_pack(owner,model_contract,device="cpu")


def test_contact_update_rejects_missing_prime_charge_before_policy_sampling(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_update as update
    collector,c,budget,rows=arrange(monkeypatch,tmp_path,backend_profile,controls=8)
    budget.solver_calls=0
    monkeypatch.setattr(update,"run_updates",lambda *a,**kw:pytest.fail("invalid prime reached PPO"))
    with pytest.raises(ValueError,match="prim"):
        update.run_fresh_update(collector,checkpoint_contract=contract(collector.trainer),controls=8)
    assert c.sim.current_time_step_index==3 and collector.consumed_sdk_steps==0
    assert not len(collector.trainer.rollout)


def test_legacy_preflight_does_not_misreport_the_implemented_contact_adapter(monkeypatch,tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_training as training
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as preflight
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_model as model
    monkeypatch.setattr(preflight,"static_preflight",lambda c:{"blocked_reasons":[]})
    monkeypatch.setattr(BackendProfile,"load",lambda:object())
    monkeypatch.setattr(model,"model_contract",lambda *a:{"grasp":"tripod"})
    cfg=training.candidate_training_config()
    cfg.update(model_config={},episode_cap_s=2.)
    cfg["preflight"].update(style="tripod",profile=PROFILE)
    result=training.preflight_training(cfg)
    assert "full_shared_contact_slip_hold_source_adapter_missing" not in result["blocked_reasons"]
    assert result["native_execution_ready"] is False
    assert "c06_source_bound_worker_lease_missing" in result["blocked_reasons"]


def test_priming_and_physical_audits_use_original_zero_event_and_exclude_prime_from_policy_count(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_artifacts as audit
    collector,c,budget,rows=arrange(monkeypatch,tmp_path,backend_profile,controls=8)
    collector.collect()
    journal=[json.loads(line) for line in collector.journal.path.read_text().splitlines()]
    prime=audit._contact_prime(journal,end_sdk_step=67)
    assert prime["clock"]["physics_step"]==3 and prime["clock"]["env_id"]==0
    evidence=audit._physical_journal(journal,reward_scope="cpu_fixture",controls=1,
        expected_backend_sha256=backend_profile.binary_sha256,profile=PROFILE)
    assert evidence["measured_sdk_steps"]==8 and evidence["source_priming_sdk_steps"]==1
    assert evidence["first_sdk_endpoint"]==4 and evidence["last_sdk_endpoint"]==11
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_artifacts import _audit_fresh_setup
    setup=_audit_fresh_setup(journal,profile=PROFILE,end_sdk_step=67)
    assert setup["first_policy_sdk_step"]==3 and setup["setup_sdk_steps"]==2
    # Re-sign all mutable receipt hashes, so these negatives exercise the
    # original action protocol rather than merely failing digest consistency.
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned,_digest
    for fault in ("version","contract","env_alias","backend"):
        damaged=deepcopy(journal)
        event_row=next(row for row in damaged if row["payload"].get("kind")=="priming_physical")
        event=decode_owned(event_row["payload"])
        trace=event["exchange"]["action_readback"]
        if fault=="version":trace["version"]="wrong-zero-action-protocol"
        elif fault=="contract":trace["request"]["action_contract"]["unbound_change"]=True
        elif fault=="env_alias":trace["env_ids"]=[False]
        else:event["rod_boundary"]["rod_snapshot"]["backend_sha256"]="0"*64
        event_row["payload"]=encode_owned(event)
        next(row for row in damaged if row["operation"]=="priming_exchange")["payload"]=encode_owned(event["exchange"])
        for operation in ("source_priming_complete","primed_sources"):
            next(row for row in damaged if row["operation"]==operation)["payload"]["receipt"]["priming_row_sha256"]=_digest(event)
        with pytest.raises(ValueError,match="prim"):
            audit._contact_prime(damaged,end_sdk_step=67)
    for fault in ("missing","ppo_sample","charge","old_history"):
        damaged=deepcopy(journal)
        if fault=="missing":
            damaged=[row for row in damaged if row["payload"].get("kind")!="priming_physical"]
        elif fault=="ppo_sample":
            row=next(row for row in damaged if row["operation"]=="source_priming_complete")
            row["payload"]["receipt"]["ppo_transitions"]=1
        elif fault=="charge":
            row=next(row for row in damaged if row["operation"]=="source_priming_complete")
            row["payload"]["budget_solver_calls"]=0
        else:
            row=next(row for row in damaged if row["operation"]=="decision")
            row["payload"]["history"]["version"]="dh116-native-action24-history-v2"
        with pytest.raises(ValueError):
            audit._physical_journal(damaged,reward_scope="cpu_fixture",controls=1,profile=PROFILE)


def test_contact_wall_budget_starts_after_prime_and_rejects_sdk2():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_artifacts import _audit_fresh_budget
    before=dict(phase="before_update",episode_uid="cpu:wall:0",sdk_step=3,
        episode_cap_s=2.,local_cap_s=50.,parent_deadline_s=100.,started_s=2.,
        observed_s=3.,remaining_s=49.)
    after=dict(before,phase="after_update",sdk_step=67,observed_s=4.,remaining_s=48.)
    rows=[dict(sequence=0,operation="fresh_update_wall_budget",payload=before),
        dict(sequence=1,operation="decision",payload={}),
        dict(sequence=2,operation="fresh_update_checkpoint_sealed",payload={}),
        dict(sequence=3,operation="fresh_update_wall_budget",payload=after)]
    envelope=SimpleNamespace(nonce="cpu:wall",issued_monotonic_s=1.,deadline_monotonic_s=100.)
    audit=_audit_fresh_budget(rows,envelope,episode_uid="cpu:wall:0",episode_cap_s=2.,
        local_cap_s=50.,end_sdk_step=67,profile=PROFILE)
    assert audit["remaining_before_s"]==49.
    before["sdk_step"]=2
    with pytest.raises(ValueError,match="budget"):
        _audit_fresh_budget(rows,envelope,episode_uid="cpu:wall:0",episode_cap_s=2.,
            local_cap_s=50.,end_sdk_step=67,profile=PROFILE)
