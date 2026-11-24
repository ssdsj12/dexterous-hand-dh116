"""CPU-only diagnostic command/ownership tests; these never claim native contact."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
import pytest
import torch

from test_tensor_training_reward import start, config, row, batch


def api():
    path = Path(__file__).resolve().parents[1] / "scripts/smoke_dh116_tensor_contact_replay.py"
    assert path.is_file(), "independent archived-command replay runner is missing"
    spec = importlib.util.spec_from_file_location("tensor_contact_replay_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packet(*, hand=True, force=True):
    mesh = 1 if hand else 0
    w = torch.zeros(2, 2, 6, dtype=torch.float64)
    if force:
        w[0, mesh, 0] = 2.
    return dict(counts=torch.tensor([1, 0]), env_ids=torch.tensor([0]),
        indices=torch.tensor([[0, 1, mesh]]), mesh_wrenches=w)


def commands():
    return [np.full(24, i / 1000, dtype=np.float64) for i in range(26)] + [np.zeros(24)]


def fixture(*, failure_step=None):
    reward, _, initial = start(config(), n=2)
    c = SimpleNamespace(n=2, controls=0, device=torch.device("cpu"),
        runtime=SimpleNamespace(physics_step=2, mesh_names=("table", "left.pad"),
            _reaction=None, resets=[1, 1], ticks=(0, 0)),
        bindings={"left.pad": {"side": "left"}},
        reward=reward, last=batch(initial)[1], ages=torch.zeros(2, dtype=torch.long),
        reset_proofs=[], snapshots={}, actions=[], previous=[])
    # Real reward begins at fixture tick zero; runtime SDK offset is independent.
    def forbidden(*args, **kwargs):
        raise AssertionError("archived diagnostic must never enter model/PPO APIs")
    c.trainer=SimpleNamespace(counters={"updates": 0}, rollout=[],
        collect_step=forbidden, record_step=forbidden, update=forbidden)
    c.model=SimpleNamespace(value=forbidden)
    c.observation=forbidden
    c.source=SimpleNamespace(set_previous_request=lambda a,t: c.previous.append((a.clone(),t)))
    from go2_pvcnn.control.dh116.tabletop_tensor_collector import probe_copy
    c.snapshot=lambda name,value: c.snapshots.__setitem__(name,probe_copy(value))
    def sdk(actions, end):
        c.runtime.physics_step += 1
        step = c.runtime.physics_step - 2
        c.actions.append(actions.clone())
        c.last=batch([row(config(), i, step, episode=int(c.runtime.resets[i]>1),
            failure=(failure_step==step and i==0)) for i in range(2)])[1]
        c.runtime._reaction=packet()
        c.runtime.ticks=tuple(t+12 for t in c.runtime.ticks)
        return c.reward.observe(c.last)
    c._sdk_impl=sdk
    c.sim=SimpleNamespace(current_time_step_index=2)
    advance = c._sdk_impl
    def measured_sdk(actions,end):
        result = advance(actions,end)
        c.sim.current_time_step_index=c.runtime.physics_step
        return result
    c._sdk_impl=measured_sdk
    from go2_pvcnn.control.dh116.tabletop_tensor_physics_runtime import NativeTensorPhysicsRuntime
    for name in ("_assert_usable", "_initialize_window_state", "_window", "begin_action_window",
                 "sdk", "record_window_request", "complete_action_window", "_notify_capture"):
        setattr(c, name, MethodType(getattr(NativeTensorPhysicsRuntime,name),c))
    c.ids=(0,1)
    c._initialize_window_state()
    c.ctrl=SimpleNamespace(pending=None)
    c.source._boundary=lambda:SimpleNamespace(sdk_step=c.runtime.physics_step)
    def reset(ids):
        assert ids==(0,) and c.controls==26
        assert c.reward._pending is None
        peer=deepcopy(c.reward.schemas[1])
        c.runtime.resets[0]+=1
        c.runtime.ticks=(0,c.runtime.ticks[1])
        c.runtime._reaction["mesh_wrenches"][0]=0
        c.runtime._reaction["env_ids"]=torch.empty(0,dtype=torch.long)
        c.runtime._reaction["indices"]=torch.empty(0,3,dtype=torch.long)
        c.runtime._reaction["counts"][0]=0
        selected, frame=batch([row(config(),0,208,episode=1)])
        c.reward.reset_selected(ids, selected, frame, previous_request=torch.zeros(1,24),
            tokens=("reset:1",),policy_uids=("controller_reset_zero",))
        assert c.reward.schemas[1]==peer
        c.reset_proofs.append(dict(selected=[0],peers=[1],sdk_step=210))
        assert c._reset_allowed==(0,)
        c._reset_allowed=()
    c.reset=reset
    return c


def test_archived_commands_stay_exact_float64_and_never_enter_ppo():
    mod=api()
    c=fixture()
    original=commands()
    result=mod.run_replay(c,original,218,source_sha="a"*64)
    assert len(c.actions)==216 and c.controls==27 and result["updates"]==0
    for i, action in enumerate(c.actions):
        assert action.shape==(2,24) and action.dtype==torch.float64
        np.testing.assert_array_equal(action[0].numpy(), original[i//8])
        assert not action[1].count_nonzero()
    assert result["action_source"]=="archived_command_replay"
    assert result["ppo_rollout_admission"] is False and len(c.trainer.rollout)==0
    assert len(c.previous)==27 and all(t.startswith("archived_command_replay:") for _,t in c.previous)
    assert c.runtime.resets==[2,1] and c.runtime.ticks==(96,2592)
    assert result["contact_coverage"]["hand_pair_samples"]==[216,0]
    assert result["post_reset_next_sdk"]["sdk_step"]==211
    assert result["post_reset_next_sdk"]["applied"]["nonzero_hand_force"]==[False,False]
    assert "window_27" in c.snapshots


@pytest.mark.parametrize("bad", [np.zeros(23), np.full(24,np.nan), np.full(24,1.01)])
def test_invalid_archive_action_fails_before_any_sdk(bad):
    mod=api()
    c=fixture()
    archive=commands()
    archive[10]=bad
    with pytest.raises(ValueError,match="archived.*24D"):
        mod.run_replay(c,archive,218,source_sha="a"*64)
    assert not c.actions and c.reward._pending is None


def test_terminal_guard_stops_on_failing_substep_before_another_sdk():
    mod=api()
    c=fixture(failure_step=3)
    with pytest.raises(ValueError,match="terminal lane"):
        mod.run_replay(c,commands(),218,source_sha="a"*64)
    assert c.runtime.physics_step==5 and len(c.actions)==3
    assert c.controls==0 and not c.previous and not c.trainer.rollout


def test_table_only_pairs_cannot_qualify_hand_reaction():
    mod=api()
    data=mod.contact_stats(packet(hand=False), ("table","left.pad"), {"left.pad":{}}, n=2)
    assert data["hand_pair_samples"]==[0,0]
    assert data["table_pair_samples"]==[1,0]
    with pytest.raises(ValueError,match="hand.*pair.*reaction"):
        mod.require_contact(dict(hand_pair_samples=[0,0],nonzero_captured_hand_steps=[0,0],
            nonzero_applied_hand_steps=[0,0]))


def test_table_only_attempt_retains_actual_coverage_for_failure_receipt():
    mod=api()
    c=fixture()
    advance=c.sdk
    def table_sdk(actions,end):
        ready=advance(actions,end)
        c.runtime._reaction=packet(hand=False)
        return ready
    c.sdk=table_sdk
    with pytest.raises(ValueError,match="hand.*pair.*reaction"):
        mod.run_replay(c,commands(),218,source_sha="a"*64)
    assert c.diagnostic_coverage["hand_pair_samples"]==[0,0]
    assert c.diagnostic_coverage["table_pair_samples"]==[216,0]
    assert len(c.diagnostic_steps)==216 and c.controls==27


def test_nonzero_hand_force_without_actual_hand_pair_is_rejected():
    mod=api()
    p=packet(hand=False,force=False)
    p["mesh_wrenches"][0,1,0]=1.
    with pytest.raises(ValueError,match="real hand pair"):
        mod.contact_stats(p,("table","left.pad"),{"left.pad":{}},n=2)


def test_n2_shape_and_all_archive_windows_are_required_before_sdk():
    mod=api()
    c=fixture()
    c.n=128
    with pytest.raises(ValueError,match="N2"):
        mod.run_replay(c,commands(),218,source_sha="a"*64)
    c.n=2
    with pytest.raises(ValueError,match="27.*218"):
        mod.run_replay(c,commands()[:-1],218,source_sha="a"*64)
    assert not c.actions


def test_original_archive_loader_binds_pair_and_receipt_inventory(monkeypatch,tmp_path):
    mod=api()
    import scripts.smoke_dh116_teacher_batch_dynamic as original
    import go2_pvcnn.control.dh116.tabletop_native_teacher_profile as profile
    source=tmp_path/"outer.json"
    source.write_text('{"artifact_sha256":{"frozen_plan.json":"plan-sha","physical.jsonl":"ledger-sha"}}')
    pair={"sha256":"original-pair"}
    issued=commands()
    monkeypatch.setattr(profile,"checked_file",lambda path,sha: source.read_bytes())
    monkeypatch.setattr(original,"archived_contact_prefix",lambda path,sha:(pair,issued))
    archive=SimpleNamespace(pair={"sha256":"wrong-reset-receipt-pair"})
    loaded, metadata=mod.bind_replay_source(archive,source,mod.CONTACT_RECEIPT_SHA256)
    assert archive.pair is pair and loaded is issued
    assert metadata["contact_receipt_sha256"]==mod.CONTACT_RECEIPT_SHA256
    assert metadata["source_artifact_sha256"]["physical.jsonl"]=="ledger-sha"
    with pytest.raises(ValueError,match="locked.*contact receipt"):
        mod.bind_replay_source(archive,source,"b"*64)


def reaction_row():
    source=dict(parent_solver_id=13,parent_num_envs=2,run_uuid="actual-run",sdk_step=210,
        rod_ticks=(0,2496),episode_uids=("new:0","old:1"),reset_ids=(2,1),solver_time_s=.4)
    pending=packet()
    pending.update(vectors=torch.zeros(1,12),normal_force=torch.ones(1),node_forces=torch.zeros(2,41,3))
    reaction=dict(pending,source_solver_id=13,parent_num_envs=2,run_uuid="actual-run",
        source_physics_step=210,applied_physics_step=211,rod_ticks=(0,2496),
        episode_uids=("new:0","old:1"),reset_ids=(2,1),solver_time_s=.4)
    return dict(pre=source,reaction=reaction),pending


@pytest.mark.parametrize("fault", ["skip_sdk","stale_epoch","other_solver","cross_lane","altered_force"])
def test_cpu_replay_rejects_wrong_pending_reaction_ownership(fault):
    mod=api()
    row,pending=reaction_row()
    row=deepcopy(row)
    if fault=="skip_sdk": row["reaction"]["applied_physics_step"]=212
    if fault=="stale_epoch": row["reaction"]["reset_ids"]=(1,1)
    if fault=="other_solver": row["reaction"]["source_solver_id"]=14
    if fault=="cross_lane": row["reaction"]["env_ids"]=torch.tensor([1])
    if fault=="altered_force": row["reaction"]["mesh_wrenches"][0,1,0]=3.
    with pytest.raises((ValueError,AssertionError)):
        mod.verify_reaction_link(row,pending,{})


def test_cpu_replay_retains_exact_peer_pairs_and_force_across_selected_reset():
    mod=api()
    row,pending=reaction_row()
    mod.verify_reaction_link(row,pending,{})
    pending["env_ids"]=torch.tensor([0,1])
    pending["indices"]=pending["indices"].repeat(2,1)
    pending["vectors"]=pending["vectors"].repeat(2,1)
    pending["normal_force"]=pending["normal_force"].repeat(2)
    pending["counts"]=torch.ones(2,dtype=torch.long)
    pending["mesh_wrenches"][1,1,0]=7.
    retained=mod.reset_pending_reaction(pending,(0,))
    assert retained["env_ids"].tolist()==[1] and retained["counts"].tolist()==[0,1]
    assert not retained["mesh_wrenches"][0].count_nonzero()
    assert torch.equal(retained["mesh_wrenches"][1],pending["mesh_wrenches"][1])
    assert pending["env_ids"].tolist()==[0,1] and pending["mesh_wrenches"][0,1,0]==2.
