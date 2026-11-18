"""Bounded CPU fixtures over original physical owner and original PPO update."""
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

    collector, _, rod, _, sim, _ = fixture(monkeypatch, tmp_path, backend_profile,
                                          rollout_steps=32, physics_budget_end=258)

    def recorded(model, pre, context, *, purpose):
        observation = collector.sources.pack(model, pre, context)
        sequence = collector.journal.emit("teacher_input_source", {
            "purpose": purpose,
            "step": sim.current_time_step_index,
            "observation_sha256": _digest(observation),
        })
        return observation, sequence

    collector.sources.pack_recorded = recorded
    result = runner().run_updates(collector)
    rows = [json.loads(line) for line in collector.journal.path.read_text().splitlines()]
    decisions = [row for row in rows if row["operation"] == "decision"]
    settlements = [row for row in rows if row["operation"] == "settlement"]
    inputs = [row for row in rows if row["operation"] == "teacher_input_source"]
    physical = [row for row in rows if row["operation"] == "physical"]
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
    print(f"rows={len(rows)} sources={len(inputs)} controls={len(decisions)} physics={len(physical)} "
          f"accepted={result['accepted_minibatch_updates']} l2={result['parameter_delta_l2']:.9g}")
