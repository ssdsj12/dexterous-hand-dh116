"""CPU owner doubles, never native qualification."""
from copy import deepcopy
import pytest
from test_deformable_r0_replay import setup, advance
from go2_pvcnn.control.dh116.tabletop_r0_live_observer import LiveR0Observer
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest

def live(monkeypatch,tmp_path):
    episode,rod,hands,sim,rows,old=setup(monkeypatch,tmp_path,mode="empty")
    obs=LiveR0Observer(**deepcopy(old._inputs))
    return episode,rod,hands,sim,rows,obs

def test_completed_transaction_initializes_original_r0_without_reset_callbacks(monkeypatch,tmp_path):
    ep,rod,hands,sim,rows,obs=live(monkeypatch,tmp_path)
    before=sim.current_time_step_index,rod.rope.steps
    receipt=obs.complete_physical_reset(rows[0],reset_approach=None)
    assert (sim.current_time_step_index,rod.rope.steps)==before
    assert receipt["initializer"]["reset_callbacks_executed"]==0
    assert receipt["physical_transaction_sha256"]==_digest(rows[0]["reset"])
    assert obs._state=="active"
    result=advance(ep,rows,obs)
    assert result["control_receipt"]["physical_samples"]==8
    assert result["training_reward"] is None
    assert result["r0_native_reset_complete"] is False
    assert result["live_post_reset_initialization"]["completed_transaction_observed"]
    assert result["physical_rows"][0]["measurements"]["stable"]["value"] is None
    assert result["physical_rows"][0]["r0"]["terms"]["hold_progress"]["raw"] is None

def test_saved_reset_cannot_enter_inherited_replay_initialization(monkeypatch,tmp_path):
    *_,rows,obs=live(monkeypatch,tmp_path)
    with pytest.raises(RuntimeError,match="completed transaction"):obs.record(rows[0])

@pytest.mark.parametrize("damage",["status","joint","owner","owner_uid","owner_source","clock"])
def test_incomplete_reset_evidence_refused(monkeypatch,tmp_path,damage):
    *_,rows,obs=live(monkeypatch,tmp_path)
    row=deepcopy(rows[0])
    if damage=="status":row["reset"]["status"]="partial"
    elif damage=="joint":row["reset"]["hands"].pop("right")
    elif damage=="owner":row["reset"]["owner_resets"].pop("filter")
    elif damage=="owner_uid":row["reset"]["owner_resets"]["filter"]["episode_uid"]="other"
    elif damage=="owner_source":row["reset"]["owner_resets"]["filter"]["source"]=""
    else:row["reset"]["after_clock"]["physics_step"]+=1
    with pytest.raises((ValueError,KeyError)):obs.complete_physical_reset(row,reset_approach=None)
    assert obs._state=="failed"

def test_original_joint_terminal_once_and_physical_sample_retained(monkeypatch,tmp_path):
    ep,rod,hands,sim,rows,obs=live(monkeypatch,tmp_path)
    obs.complete_physical_reset(rows[0],reset_approach=None)
    original=sim.step
    def fail(*,render):
        original(render=render)
        hands[1].measured[0,hands[1].joint_names.index("finger12")]=-.05
    sim.step=fail
    result=advance(ep,rows,obs)
    assert result["control_receipt"]["physical_samples"]==1
    assert result["control_receipt"]["terminal_reasons"]==["joint_limit"]
    assert len(result["physical_rows"])==1
    with pytest.raises(RuntimeError):obs.end({})
