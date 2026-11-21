"""CPU admission/lease contracts; actual saved evidence is not a new native run."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from test_tripod_reset_manifest import ROOT, RUN

MANIFEST = RUN.parent/"tripod-teacher-resolved-reset-v1.json"
MANIFEST_SHA = "ed086c283c177e52a5d9358c2f4887ccaea3d5c568e4770b19dc517d5f1f6d5c"


def config():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import candidate_config
    value = candidate_config(style="tripod")
    value["resolved_reset_manifest"] = dict(path=str(MANIFEST),sha256=MANIFEST_SHA)
    reward = ROOT/"configs/dh116_training/approach_wiring_smoke_v1.json"
    value["reward_config"] = dict(path=str(reward),sha256=hashlib.sha256(reward.read_bytes()).hexdigest())
    return value


@contextmanager
def cpu_budget(config,tmp_path,monkeypatch):
    """Temporary kernel lock and CPU files, never the real singleton or launch."""
    from scripts import run_dh116_work_packages as scheduler
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as pre
    config["budget"]=dict(max_sdk_steps=66,max_resets=1,max_solver_calls=64,
        max_episode_count=1,wall_timeout_s=180.)
    auth=dict(schema="dh116-native-teacher-budget-authorization-v1",
        scope="new_native_teacher_ppo",profile=config["profile"],style=config["style"],
        budget=config["budget"],max_ppo_updates=1,config_sha256=pre.authorization_config_digest(config),
        session_uid="CPU-only-Tripod-admission",owner_pid=os.getpid())
    def write(name,value):
        path=tmp_path/name
        path.write_text(json.dumps(value))
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    config["authorization"]=write("cpu-authorization.json",auth)
    config["resource_lock"]=write("cpu-reservation.json",dict(
        schema="dh116-native-resource-reservation-v1",session_uid=auth["session_uid"],
        resource_class="isaac_native_singleton",authorization_sha256=config["authorization"]["sha256"],pid=os.getpid()))
    kernel=tmp_path/"cpu.lock";kernel.touch()
    monkeypatch.setattr(scheduler,"NATIVE_LOCK",kernel)
    with scheduler.native_file_lock(kernel):
        yield


def test_real_tripod_manifest_enters_static_contract_without_pinch_or_full_success_transfer(monkeypatch,tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as pre
    from go2_pvcnn.control.dh116.tabletop_tripod_reset_manifest import TripodResetManifest
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import NativeSessionConfig,verify_static_contract
    from test_dh116_flexible_measured_common_teacher import setup
    value=config()
    loaded=pre.load_reset_manifest(value,project=ROOT)
    assert type(loaded) is TripodResetManifest
    with cpu_budget(value,tmp_path,monkeypatch):
        result=pre.static_preflight(value,project=ROOT)
        assert result["static_ready"] is True,result["blocked_reasons"]
        parts={x["name"]:x for x in result["components"]}
        assert parts["same_clock_distance"]["valid"] is True
        assert "tripod" in parts["same_clock_distance"]["reader"].lower()
        assert parts["retained_reset"]["applicable"] is False
        assert result["new_native_steps"]==result["ppo_updates"]==0
        assert result["live_reset_verified"] is result["teacher_admission"] is False
        model,*_=setup("tripod")
        contract=verify_static_contract(NativeSessionConfig(value,model,tmp_path/"events.jsonl",1.,8,ROOT))
        assert contract.config.model_config["grasp"]=="tripod"
        assert type(contract.reset_manifest) is TripodResetManifest
        assert contract.budget["max_sdk_steps"]==66
    value["resolved_reset_manifest"]=None
    missing=pre.static_preflight(value,project=ROOT)
    assert missing["static_ready"] is False
    assert not next(x for x in missing["components"] if x["name"]=="same_clock_distance")["valid"]


def test_supported_live_mesh_reopens_all17_and_refuses_drift_without_advancement(monkeypatch):
    from test_supported_mesh_reset import case,execute
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import validate_live_mesh
    c,module,plan=case(monkeypatch)
    _,receipt=execute(c,module,plan)
    cost=(c.sim.current_time_step_index,c.rope.steps,c.rope.resets)
    good=validate_live_mesh(c.sim,c.rod,reset_receipt=receipt,mesh_plan=plan)
    assert good["seventeen_reset_meshes_observed"] is True
    assert good["native_qualification"] is good["teacher_admission"] is False
    for fault in ("current","target","owner","clock"):
        row=deepcopy(receipt)
        original_current=c.rope.current[16][0].copy()
        original_target=c.rope.targets[16][0].copy()
        step=c.sim.current_time_step_index
        if fault=="current":c.rope.current[16][0][0]+=.001
        elif fault=="target":c.rope.targets[16][0][0]+=.001
        elif fault=="owner":row["mesh_state_readback"]["after"]["source_instance_id"]+=1
        else:c.sim.current_time_step_index+=1
        try:
            with pytest.raises(ValueError):
                validate_live_mesh(c.sim,c.rod,reset_receipt=row,mesh_plan=plan)
        finally:
            c.rope.current[16][0][:]=original_current
            c.rope.targets[16][0][:]=original_target
            c.sim.current_time_step_index=step
    assert cost==(c.sim.current_time_step_index,c.rope.steps,c.rope.resets)


def test_tripod_owner_reset_uses_real_supported_transaction_and_independent_journal_sink(monkeypatch,tmp_path):
    from test_fresh_supported_episode import episode_case
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_session as session
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
    c,module,plan,episode,physical_rows,_=episode_case(monkeypatch,tmp_path)
    owner=object.__new__(session.NativeEpisodeOwner)
    owner.episode=episode;owner.sim=c.sim;owner.rod=c.rod;owner.bindings=c.bindings
    owner.mesh_plan=plan;owner._closed=False;owner.backend_profile=object()
    owner.contract=SimpleNamespace(config=SimpleNamespace(preflight=dict(style="tripod")),
        runtime=session.SessionBudget(dict(max_sdk_steps=18,max_solver_calls=16,max_resets=1,
            max_episode_count=1,wall_timeout_s=180.)))
    owner.ledger=NativeJournal(tmp_path/"reset-events.jsonl",scope="cpu_fixture")
    owner._initial_geometry=lambda:()
    owner.history=SimpleNamespace(record=lambda row:None)
    after=[]
    owner.sources=SimpleNamespace(after_reset=after.append)
    # Hardware/source checks are owned by their stronger existing tests.
    # The actual original episode and supported transaction execute unchanged.
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_profile as profile_module
    monkeypatch.setattr(profile_module,"profile_reader_scope",lambda profile:nullcontext())
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_numeric_state as numeric
    @contextmanager
    def cpu_numeric_scope(**kwargs):
        audit={"scope":"cpu_fixture","native_qualification":False}
        yield audit
        kwargs["record"](audit)
    monkeypatch.setattr(numeric,"numeric_reset_state_scope",cpu_numeric_scope)
    before=(c.sim.current_time_step_index,c.rope.steps,c.rope.resets)
    try:
        result=owner.reset(c.seed,episode_uid="CPU-only-Tripod-lease",owner_resets=c.owners)
        assert len(after)==1 and after[0] is result
        assert result["reset"]["mesh_state_readback"]["complete"] is True
        assert len(result["reset"]["rod_reset"]["mesh_names"])==17
        rows=[json.loads(x) for x in owner.ledger.path.read_text().splitlines()]
        readbacks=[x for x in rows if x["operation"]=="supported_reset_readback"]
        assert len(readbacks)==1
        assert readbacks[0]["payload"]["complete"] is True
        assert len(physical_rows)==1 and physical_rows[0]["kind"]=="reset"
        assert c.sim.current_time_step_index==before[0] and c.rope.steps==before[1]
        assert c.rope.resets==before[2]+1
        assert owner.contract.runtime.resets==1
    finally:owner.ledger.close()
