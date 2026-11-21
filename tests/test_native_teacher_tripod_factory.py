"""Borrowed Tripod scene factory: no App creation, stepping, or qualification."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_native_teacher_tripod_admission import ROOT, config


@pytest.fixture(scope="module")
def manifest():
    from go2_pvcnn.control.dh116.tabletop_tripod_reset_manifest import load_tripod_reset_manifest
    return load_tripod_reset_manifest(config(),project=ROOT)


def test_tripod_factory_binds_frozen_task_seed_and_one_original_fresh_episode(monkeypatch,tmp_path,manifest):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_session as session
    assert hasattr(session,"build_fresh_tripod_owner"),"independent fresh17 Teacher factory missing"
    from go2_pvcnn.control.dh116 import tabletop_fresh_supported_builder as builder
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    data=manifest.data()
    profile=BackendProfile.load(ROOT)
    journal=NativeJournal(tmp_path/"teacher.jsonl",scope="cpu_fixture")
    cfg=session.NativeSessionConfig(dict(style="tripod",profile="approach_wiring_smoke_v1"),{},
        journal.path,1.,8,ROOT)
    contract=session.StaticContract(cfg,profile,manifest,data["reset_seed"],None,None,
        dict(max_sdk_steps=66),None,"CPU-factory-binding")
    sim,owners,trace,prepared=(object() for _ in range(4))
    original_episode=SimpleNamespace(_state="new")
    plan=dict(version="fixture17",sha256="f"*64)
    initial=lambda:()
    setup=tuple(dict(kind="setup",sdk_step=i,sdk_time_s=i/480) for i in (0,2))
    events=[]
    ownership=SimpleNamespace(event=lambda kind,**kw:events.append((kind,kw)))
    def build(actual_sim,actual_owners,**kw):
        assert actual_sim is sim and actual_owners is owners
        assert kw["prepared"] is prepared and kw["ownership"] is ownership
        assert kw["backend_profile"] is profile and kw["physics_budget_end"]==66
        assert str(kw["task_config"])==data["task_config"]["path"]
        assert kw["task_config_sha256"]==data["task_config"]["sha256"]
        assert kw["extra_sources"]==data["extra_sources"]
        assert kw["run_uuid"]=="cpu-run" and kw["episode_uid"]=="cpu-episode"
        for row in setup:kw["record"](row)
        return builder.FreshSupportedEpisode(original_episode,setup,initial,
            data["reset_seed"],data["reset_seed"]["sha256"],plan,data["task_config"]["sha256"])
    returned=object()
    def lease(episode,static,**kw):
        assert episode is original_episode and static is contract and episode._state=="new"
        assert kw["ledger"] is journal and kw["mesh_plan"] is plan
        assert kw["initial_geometry"] is initial and kw["setup_events"]==setup
        assert kw["episode_uid"]=="cpu-episode" and kw["owner_resets"]=={}
        return returned
    monkeypatch.setattr(builder,"build_fresh_supported_owned",build)
    monkeypatch.setattr(session,"NativeEpisodeOwner",lease)
    try:
        assert session.build_fresh_tripod_owner(contract,sim=sim,owners=owners,prepared=prepared,
            ownership=ownership,ledger=journal,run_uuid="cpu-run",episode_uid="cpu-episode",
            owner_resets={},mesh_plan=None) is returned
        assert len(events)==2 and all(k=="tripod_scene_setup" for k,_ in events)
        assert not journal.path.read_text() # EpisodeOwner exclusively persists setup.
        with pytest.raises(ValueError,match="fresh|plan"):
            session.build_fresh_tripod_owner(contract,sim=sim,owners=owners,prepared=prepared,
                ownership=ownership,ledger=journal,run_uuid="cpu-run",episode_uid="cpu-episode",
                owner_resets={},mesh_plan=plan)
        assert len(events)==2
        bad=replace(contract,reset_seed=dict(data["reset_seed"],sha256="0"*64))
        with pytest.raises(ValueError,match="seed"):
            session.build_fresh_tripod_owner(bad,sim=sim,owners=owners,prepared=prepared,
                ownership=ownership,ledger=journal,run_uuid="cpu-run",episode_uid="cpu-episode",
                owner_resets={},mesh_plan=None)
        assert len(events)==2
    finally:journal.close()
