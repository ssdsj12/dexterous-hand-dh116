"""Tripod Worker composition: CPU-only owners/getters, no execution lease grant."""
from dataclasses import replace
from types import SimpleNamespace
from weakref import ref

import pytest

from test_native_teacher_actual_factories import boundaries, backend_profile, _finished_live_check, SMOKE
from test_dh116_flexible_measured_common_teacher import setup


def test_actual_factories_keep_tripod_model_context_identity_and_original_ppo(boundaries):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_factories import build_actual_factories
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import ModelContract
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import TaskContextOwner
    _,lease,owner,static,calls,_=boundaries
    model,*_=setup("tripod")
    cfg=replace(static.config,preflight=dict(style="tripod",profile=SMOKE),model_config=model)
    static=replace(static,config=cfg)
    owner.contract=static
    owner.context=TaskContextOwner(backend_profile=static.backend_profile,mode=SMOKE,grasp="tripod",
        scope="owner_source_unqualified",episode_cap_s=2.,external_cap_s=5.)
    lease._config.update(preflight=cfg.preflight,model_config=model)
    fs=build_actual_factories(lease=lease,episode_factory=lambda c:owner,
        prepare_source_pack=lambda held,c:lambda policy:calls.append((held,policy)))
    assert fs.episode(static) is owner
    _finished_live_check(owner)
    live=ModelContract(model,static.backend_profile,SMOKE)
    policy=fs.policy(live)
    trainer=fs.trainer(policy,static.ppo_config)
    assert policy.model_identity["grasp"]=="tripod"
    assert trainer.model is policy and trainer.counters["updates"]==0
    assert calls==[(owner,policy)]
    owner.context.grasp="pinch"
    with pytest.raises(ValueError,match="same live episode"):
        fs.policy(live)


def test_provider_dispatches_tripod_to_fresh17_factory_before_policy_or_update(monkeypatch,tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_provider as provider
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_factories as factories
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_session as session
    assert hasattr(session,"build_fresh_tripod_owner"),"independent Tripod factory missing"
    failure=RuntimeError("CPU observed fresh17 dispatch")
    static=object()
    sim,owners,lease=(object() for _ in range(3))
    trace=SimpleNamespace(event=lambda *a,**kw:None)
    attempt=SimpleNamespace(project=tmp_path,output=tmp_path,session_uid="cpu17-dispatch",
        device=dict(logical_device="cuda:0"))
    def right(c,**kw):
        assert c is static and kw["sim"] is sim and kw["owners"] is owners
        assert kw["ownership"] is trace and kw["episode_uid"]=="cpu17-dispatch:0"
        assert kw["mesh_plan"] is None and kw["ledger"].path==tmp_path/"training_events.jsonl"
        assert not kw["ledger"]._stream.closed
        raise failure
    def wrong(*a,**kw):pytest.fail("Tripod was routed through the Pinch scene factory")
    monkeypatch.setattr(session,"build_fresh_tripod_owner",right)
    monkeypatch.setattr(session,"build_fresh_pinch_owner",wrong)
    monkeypatch.setattr(factories,"build_actual_factories",lambda **kw:kw["episode_factory"])
    monkeypatch.setattr(session,"create_native_collector",lambda config,factory:factory(static))
    config=dict(preflight=dict(style="tripod"),model_config={},episode_cap_s=1.,rollout_steps=8)
    with pytest.raises(RuntimeError) as caught:
        provider._acquire(sim,owners,trace,attempt=attempt,config=config,lease=lease)
    assert caught.value is failure


def test_tripod_source_pack_refuses_cross_grasp_and_shares_exact_owned_getters(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_provider import _prepare_source_pack
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import NativeTripodSources
    from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import NativeParameterEndpoints
    import sys
    class CpuRopeIdentity:pass
    rope=CpuRopeIdentity()
    owner=SimpleNamespace(rod=SimpleNamespace(rope=rope,
        clock=lambda:dict(physics_step=2,rod_tick=0,episode_uid="cpu17:0")),
        sim=SimpleNamespace(device="cuda:0",current_time_step_index=2),episode_uid="cpu17:0",
        bindings=tuple(SimpleNamespace(hand=SimpleNamespace(data=SimpleNamespace(
            joint_pos=SimpleNamespace(device="cuda:0")))) for _ in range(2)))
    reader=object.__new__(NativeParameterEndpoints)
    reader.device="cuda:0";reader._rope_ref=ref(rope)
    source=object.__new__(NativeTripodSources)
    source.owner=owner;source.reader=reader;source.parameters=dict(source_instance_id=id(rope))
    owner.sources=source
    backend=SimpleNamespace(RopeBatch=CpuRopeIdentity)
    monkeypatch.setitem(sys.modules,"ropesim._ropesim",backend)
    contract=SimpleNamespace(config=dict(grasp="tripod"),backend_profile=object(),context_mode=SMOKE)
    bind=_prepare_source_pack(owner,contract,device="cuda:0")
    bind(SimpleNamespace(_backend_profile=contract.backend_profile,context_mode=SMOKE))
    with pytest.raises(ValueError,match="source pack"):
        _prepare_source_pack(owner,replace_contract(contract,grasp="pinch"),device="cuda:0")
    source.parameters["source_instance_id"]+=1
    with pytest.raises(ValueError,match="source pack"):
        _prepare_source_pack(owner,contract,device="cuda:0")


def replace_contract(contract,**config):
    return SimpleNamespace(config=dict(contract.config,**config),
        backend_profile=contract.backend_profile,context_mode=contract.context_mode)
