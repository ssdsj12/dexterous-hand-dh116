"""Bounded CPU fixtures over original physical owner and original PPO update."""
import hashlib
import importlib.util
import json
from pathlib import Path
from copy import deepcopy
import pytest
import torch
from test_native_teacher_context import backend_profile
from test_native_teacher_collector import fixture


def runner():
    name='go2_pvcnn.control.dh116.tabletop_native_teacher_training'
    assert importlib.util.find_spec(name) is not None, 'bounded native Teacher runner missing'
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_training as training
    return training


def test_original_update_and_loaded_additional_target(monkeypatch,tmp_path,backend_profile):
    r=runner();c,ep,rod,*_=fixture(monkeypatch,tmp_path,backend_profile,rollout_steps=2)
    before=deepcopy(c.trainer.model.state_dict());uid=rod.clock()['episode_uid'];reset=rod.clock()['reset_id']
    result=r.run_updates(c,additional_updates=1)
    assert result['target_updates']==1 and result['updates']==1
    assert result['accepted_minibatch_updates']>0 and result['parameter_delta_l2']>0
    assert result['scope']=='cpu_fixture' and c.trainer._pending is None and len(c.trainer.rollout)==0
    assert rod.clock()['episode_uid']==uid and rod.clock()['reset_id']==reset
    assert any(not torch.equal(v,c.trainer.model.state_dict()[k]) for k,v in before.items() if isinstance(v,torch.Tensor))
    result2=r.run_updates(c,additional_updates=1)
    assert result2['target_updates']==2 and result2['updates']==2 and rod.rope.steps==32


def test_all_kl_rejected_stops_without_resampling_or_checkpoint(monkeypatch,tmp_path,backend_profile):
    r=runner();c,ep,rod,*_=fixture(monkeypatch,tmp_path,backend_profile,rollout_steps=2)
    # Force original optimizer guard to reject; collect/update implementations remain real.
    monkeypatch.setattr(c.trainer,'_optimizer_step',lambda batch:(None,4))
    with pytest.raises(r.TrainingStopped,match='optimizer_update_rejected'):r.run_updates(c,additional_updates=1)
    assert rod.rope.steps==16 and c.trainer.counters['updates']==0
    assert len(c.trainer.rollout)==0 and not list(tmp_path.glob('*.pt'))


def test_source_failure_preserves_pending_no_update(monkeypatch,tmp_path,backend_profile):
    r=runner();c,ep,rod,*_=fixture(monkeypatch,tmp_path,backend_profile)
    c.sources.damage='nan'
    with pytest.raises(ValueError):r.run_updates(c,additional_updates=1)
    assert rod.rope.steps==3 and c.quarantined and c.trainer._pending is not None
    assert c.trainer.counters['updates']==0 and not list(tmp_path.glob('*.pt'))


def test_budget_boundary_stops_without_sampling(monkeypatch,tmp_path,backend_profile):
    r=runner();c,ep,rod,*_=fixture(monkeypatch,tmp_path,backend_profile)
    c.sources.remaining_substeps=lambda:7
    with pytest.raises(r.TrainingStopped,match='budget_exhausted'):r.run_updates(c,additional_updates=1)
    assert rod.rope.steps==0 and c.trainer._pending is None


def test_preflight_and_unleased_native_never_construct_model(monkeypatch,tmp_path):
    r=runner()
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import candidate_config
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_model as models
    monkeypatch.setattr(models,'NativeContextTeacher',lambda *a,**k:pytest.fail('preflight built policy'))
    config=r.candidate_training_config();assert config['preflight']==candidate_config()
    path=tmp_path/'config.json';path.write_text(json.dumps(config));out=tmp_path/'out'
    report=r.preflight_training(config)
    assert report['native_execution_ready'] is False
    assert 'c06_source_bound_worker_lease_missing' in report['blocked_reasons']
    assert report['policy_constructed'] is False
    assert r.main(['--config',str(path),'--output',str(out),'--preflight'])==2
    assert not list(out.glob('*.pt'))
    assert r.main(['--config',str(path),'--output',str(tmp_path/'out-native'),'--execute-native'])==2


def test_preflight_checks_model_contract_without_construction(monkeypatch):
    r=runner();config=r.candidate_training_config()
    config['model_config']={'grasp':'tripod'}
    report=r.preflight_training(config)
    assert any('model_contract_invalid' in reason for reason in report['blocked_reasons'])
    assert report['policy_constructed'] is False


def test_terminal_without_explicit_reset_factory_stops_cleanly(monkeypatch,tmp_path,backend_profile):
    r=runner();c,ep,rod,*_=fixture(monkeypatch,tmp_path,backend_profile,rollout_steps=2,cap_controls=1)
    with pytest.raises(r.TrainingStopped,match='fresh_verified_reset_factories_required'):
        r.run_updates(c,additional_updates=1)
    assert rod.rope.steps==8 and len(c.trainer.rollout)==1 and c.trainer.counters['updates']==0


def test_rejected_update_latches_session_and_cannot_save_previous_clean(monkeypatch,tmp_path,backend_profile):
    r=runner();c,ep,rod,*_=fixture(monkeypatch,tmp_path,backend_profile,rollout_steps=2)
    r.run_updates(c,additional_updates=1)
    from test_native_teacher_checkpoint import checkpoint,contract
    checkpoint().save_checkpoint(tmp_path/'last-clean.pt',c.trainer,contract(c.trainer))
    monkeypatch.setattr(c.trainer,'_optimizer_step',lambda batch:(None,4))
    with pytest.raises(r.TrainingStopped,match='optimizer_update_rejected'):r.run_updates(c,additional_updates=1)
    with pytest.raises(r.TrainingStopped,match='failed_session_last_clean_checkpoint_only'):r.run_updates(c,additional_updates=1)
    with pytest.raises(RuntimeError):checkpoint().save_checkpoint(tmp_path/'bad-clean.pt',c.trainer,contract(c.trainer))
    assert rod.rope.steps==32 and c.trainer.counters['updates']==1
    assert not (tmp_path/'bad-clean.pt').exists()


def test_preflight_reports_new_budget_authority_and_resource_reservation_separately():
    r=runner();report=r.preflight_training(r.candidate_training_config())
    assert {'new_native_budget_missing','new_native_authorization_missing',
        'kernel_resource_reservation_missing'}<=set(report['blocked_reasons'])
    assert report['execution_requirements']['budget'] is None
    assert report['execution_requirements']['fresh_worker_lease_verified'] is False


def test_rollout32_r0_update_retains_all_source_bindings(monkeypatch, tmp_path, backend_profile):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest

    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig

    approved = json.loads((Path(__file__).resolve().parents[1] /
                           "configs/dh116_training/approach_wiring_smoke_v1.json").read_text())["ppo"]
    ppo = PPOConfig(lr=approved["learning_rate"], gamma=approved["gamma"],
                    lam=approved["gae_lambda"], clip=approved["clip"], entropy_coef=approved["entropy"],
                    value_coef=approved["value"], max_grad_norm=approved["max_grad_norm"],
                    epochs=approved["epochs"], minibatches=approved["minibatches"],
                    target_kl=approved["target_kl"], kl_backtrack_steps=approved["kl_backtrack_steps"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(116)
        collector, _, rod, _, sim, _ = fixture(monkeypatch, tmp_path, backend_profile,
                                              rollout_steps=32, physics_budget_end=258, ppo_config=ppo)

    def recorded(model, pre, context, *, purpose):
        observation = collector.sources.pack(model, pre, context)
        sequence = collector.journal.emit("teacher_input_source", {
            "purpose": purpose,
            "step": sim.current_time_step_index,
            "observation_sha256": _digest(observation),
        })
        return observation, sequence

    collector.sources.pack_recorded = recorded
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_update import run_fresh_update
    from test_native_teacher_checkpoint import contract

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(117)
        result = run_fresh_update(collector, checkpoint_contract=contract(collector.trainer))
    checkpoint = tmp_path / "teacher_clean.pt"
    assert checkpoint.is_file()
    assert result["clean_checkpoint"]["path"] == checkpoint.name
    assert result["clean_checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result["scope"] == "cpu_fixture"
    assert result["native_qualification"] is result["teacher_admission"] is False
    rows = [json.loads(line) for line in collector.journal.path.read_text().splitlines()]
    decisions = [row for row in rows if row["operation"] == "decision"]
    settlements = [row for row in rows if row["operation"] == "settlement"]
    inputs = [row for row in rows if row["operation"] == "teacher_input_source"]
    physical = [row for row in rows if row["operation"] == "physical"]
    assert rows[-1]["operation"] == "fresh_update_checkpoint_sealed"
    assert rows[-1]["payload"] == result
    assert len(decisions) == len(settlements) == 32
    assert len(inputs) == 64
    assert len(physical) == rod.rope.steps == 256
    assert sim.current_time_step_index == 258
    for index, (decision, settlement) in enumerate(zip(decisions, settlements)):
        first = rows[decision["payload"]["source_pack_sequence"]]
        bootstrap = settlement["payload"]["bootstrap_source"]
        second = rows[bootstrap["source_pack_sequence"]]
        assert first["sequence"] < decision["sequence"] < second["sequence"] < settlement["sequence"]
        assert first["payload"]["purpose"] == "decision" and first["payload"]["step"] == 2 + 8 * index
        assert second["payload"]["purpose"] == "bootstrap" and second["payload"]["step"] == 10 + 8 * index
        assert second["payload"]["observation_sha256"] == bootstrap["observation_sha256"]
    assert result["updates"] == 1 and result["accepted_minibatch_updates"] > 0
    assert result["parameter_delta_l2"] > 0
    exact_kl = result["metrics"]["epoch_trace"][-1]["exact_full_rollout_kl"]
    assert exact_kl <= approved["target_kl"]
    print(f"rows={len(rows)} sources={len(inputs)} controls={len(decisions)} physics={len(physical)} "
          f"accepted={result['accepted_minibatch_updates']} l2={result['parameter_delta_l2']:.9g} exact_kl={exact_kl:.9g}")


@pytest.mark.parametrize("grasp,reader_missing",[("tripod",False),("power",True)])
def test_training_preflight_names_only_missing_independent_distance_sources(grasp,reader_missing):
    """CLI diagnostics retain the lease gate without reporting implemented Tripod missing."""
    r=runner();config=r.candidate_training_config()
    config["preflight"]["style"]=grasp
    report=r.preflight_training(config)
    assert (grasp+"_distance_reader_implementation_pending" in report["blocked_reasons"]) is reader_missing
    assert "c06_source_bound_worker_lease_missing" in report["blocked_reasons"]
    assert report["native_execution_ready"] is False
    assert report["policy_constructed"] is False


@pytest.mark.parametrize("provide_factories,solver_budget,expected",[
    (False,8,"budget_exhausted_before_complete_window"),
    (True,8,"budget_exhausted_before_complete_window"),
    (True,64,"cpu_restart_boundary"),
])
def test_terminal_restart_obeys_session_budget_before_episode_end(
        monkeypatch,tmp_path,backend_profile,provide_factories,solver_budget,expected):
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import NativeFactories,SessionBudget
    r=runner()
    collector,_,rod,_,sim,_=fixture(monkeypatch,tmp_path,backend_profile,
        rollout_steps=2,cap_controls=1,physics_budget_end=10)
    runtime=SessionBudget(dict(max_sdk_steps=solver_budget+2,max_solver_calls=solver_budget,
        max_resets=2,max_episode_count=2,wall_timeout_s=180.))
    # Synthetic CPU scene setup is already complete. The real existing
    # observer charges this real SessionBudget for each delivered fixture row.
    runtime.record_setup(SimpleNamespace(sim=sim),[
        dict(kind="setup",sdk_step=0,sdk_time_s=0.),
        dict(kind="setup",sdk_step=sim.current_time_step_index,sdk_time_s=sim.current_time)])
    collector.owner.contract=SimpleNamespace(runtime=runtime)
    calls=[]
    def restart(*args,**kwargs):
        calls.append("restart")
        raise r.TrainingStopped("cpu_restart_boundary")
    # Observe the restart boundary without building any second environment.
    monkeypatch.setattr(collector,"restart_episode",restart)
    factories=NativeFactories(restart,restart,restart,restart) if provide_factories else None
    with pytest.raises(r.TrainingStopped,match=expected):
        r.run_updates(collector,additional_updates=1,factories=factories)
    assert calls==(["restart"] if expected=="cpu_restart_boundary" else [])
    assert rod.rope.steps==8 and runtime.sdk_steps==10 and runtime.solver_calls==8
    assert collector.episode_done and len(collector.trainer.rollout)==1
    assert collector.trainer.counters["updates"]==0 and collector.trainer._pending is None
    assert not list(tmp_path.glob("*.pt"))
    stopped=[json.loads(line) for line in collector.journal.path.read_text().splitlines()
             if json.loads(line)["operation"]=="training_stopped"]
    assert stopped[-1]["payload"]["reason"]==expected
