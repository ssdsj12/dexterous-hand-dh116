"""Session composition and budget regression; no native execution."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from test_native_teacher_context import backend_profile
from test_native_teacher_session import arranged
from test_native_teacher_actual_factories import boundaries

PROFILE="approach_contact_prelift_v1"


def test_session_forwards_same_primed_owner_to_loaded_reward_factory(monkeypatch):
    m,events,owner,factories=arranged(monkeypatch)
    live=SimpleNamespace(model_contract=object(),reward_contract=SimpleNamespace(profile=PROFILE))
    owner.context=object()
    monkeypatch.setattr(m,"verify_reset_and_sources",lambda *a:live)
    calls=[]
    def reward(contract,*,contact_owner):
        calls.append((contract,contact_owner))
        return object()
    factories=replace(factories,reward=reward)
    assert m.create_native_collector({},factories)=="constructed"
    assert calls==[(live.reward_contract,owner.context)]


def test_actual_owner_selects_primed_history_and_contact_source_for_tripod(monkeypatch,tmp_path,backend_profile):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_sources as sources
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import NativeEpisodeOwner,SessionBudget
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
    from go2_pvcnn.control.dh116.tabletop_native_teacher_source_priming import PrimedNativeActionHistory
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
    sim=SimpleNamespace(current_time_step_index=2,current_time=2/480)
    rod=SimpleNamespace(rope=object(),_base=SimpleNamespace(cfg={"rod":{"solver_substeps":12}}))
    ep=object.__new__(DualHandPhysicalEpisode)
    ep._state="new";ep._end=67;ep._sim=sim;ep._rod=rod;ep._bindings=()
    original=lambda row:None
    ep._record=original
    budget=dict(max_sdk_steps=67,max_solver_calls=65,max_resets=1,max_episode_count=1,wall_timeout_s=180.)
    contract=SimpleNamespace(backend_profile=backend_profile,runtime=SessionBudget(budget),
        config=SimpleNamespace(preflight=dict(style="tripod",profile=PROFILE)))
    journal=NativeJournal(tmp_path/"owner.jsonl",scope="cpu_fixture")
    seen=[]
    def contact_source(owner,con):
        seen.append((type(owner.history),con))
        return SimpleNamespace(owner=owner,_error=None)
    assert hasattr(sources,"NativeContactTripodSources"),"contact source missing"
    monkeypatch.setattr(sources,"NativeContactTripodSources",contact_source)
    owner=NativeEpisodeOwner(ep,contract,episode_uid="cpu:owner",owner_resets={"controller":lambda c:c},
        mesh_plan=object(),ledger=journal,
        setup_events=[dict(kind="setup",sdk_step=0,sdk_time_s=0.),dict(kind="setup",sdk_step=2,sdk_time_s=2/480)],
        initial_geometry=lambda:())
    assert seen==[(PrimedNativeActionHistory,contract)]
    assert contract.runtime.sdk_steps==2 and contract.runtime.solver_calls==0
    assert isinstance(owner.history,PrimedNativeActionHistory)
    owner.close()
    assert ep._record is original


def test_contact_rollout_rejects_old_smoke_budget_that_omits_one_prime():
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_session as session
    assert hasattr(session,"validate_contact_rollout"),"contact priming budget audit missing"
    budget=dict(max_sdk_steps=67,max_solver_calls=65,max_resets=1,max_episode_count=1)
    session.validate_contact_rollout(8,budget,32)
    with pytest.raises(ValueError,match="prim"):
        session.validate_contact_rollout(8,dict(budget,max_sdk_steps=66,max_solver_calls=64),32)


def test_actual_factories_allow_tripod_contact_but_refuse_old_context_before_model(boundaries):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_factories import build_actual_factories
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import ModelContract
    from go2_pvcnn.control.dh116.tabletop_training_reward import load_config
    from test_native_teacher_context import ROOT
    from test_dh116_flexible_measured_common_teacher import setup
    fs,lease,owner,static,calls,events=boundaries
    model_config,*_=setup("tripod")
    config=replace(static.config,preflight=dict(style="tripod",profile=PROFILE),model_config=model_config)
    static=replace(static,config=config,reward_config=load_config(ROOT/"configs/dh116_training/approach_contact_prelift_v1.json"))
    owner.contract=static
    lease._config=dict(lease.process_config,preflight=config.preflight,model_config=model_config)
    fs=build_actual_factories(lease=lease,episode_factory=lambda contract:owner,
        prepare_source_pack=lambda *args:pytest.fail("unprimed old Context reached policy binder"))
    assert fs.episode(static) is owner
    owner.episode._state="active"
    owner.context._current=object()  # Explicit old-context boundary double, not a native witness.
    with pytest.raises(ValueError,match="live"):
        fs.policy(ModelContract(model_config,static.backend_profile,PROFILE))

def test_contact_static_config_uses_real_reward_gamma_but_cannot_borrow_pinch_sources(backend_profile):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as preflight
    from go2_pvcnn.control.dh116.tabletop_training_reward import load_config
    from test_native_teacher_context import ROOT
    config=preflight.candidate_config(style="tripod",profile=PROFILE)
    reward=load_config(ROOT/"configs/dh116_training/approach_contact_prelift_v1.json",ppo_gamma=config["ppo"]["gamma"])
    assert reward.gamma==.995
    tripod=preflight.static_preflight(config,project=ROOT,backend_profile=backend_profile)
    contact={"required_pad_contact","opposition","bilateral_support","off_table","no_external_support","continuous_history"}
    components={item["name"]:item for item in tripod["components"]}
    assert contact <= set(tripod["required_capabilities"])
    assert all(components[key]["valid"] and components[key]["source_sha256"] for key in contact)
    assert not tripod["native_qualification"] and not tripod["teacher_admission"]
    report=preflight.static_preflight(dict(config,style="pinch"),project=ROOT,backend_profile=backend_profile)
    assert not report["static_ready"]
    assert any("supported17_contact_source" in reason for reason in report["blocked_reasons"])
    assert "fixed_material_slip" not in report["required_capabilities"]
