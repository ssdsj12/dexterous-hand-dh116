"""Real exchange wrapper with synthetic rows, no Isaac/native qualification."""
import importlib.util
from copy import deepcopy
import numpy as np
import pytest
from scripts import prepare_dh116_dual_pinch_director_interval as director
from scripts import prepare_dh116_tripod_owned_approach as plan
from test_tripod_quiet_gate import make

def module():
    name="scripts.prepare_dh116_tripod_native_interval"
    assert importlib.util.find_spec(name) is not None,"Tripod live quiet admission missing"
    return __import__(name,fromlist=["TripodExchange"])

def setup(monkeypatch, fault=None):
    m=module();g,rows=make()
    calls=[]
    def physics(self,action,substep):
        i=len(calls);calls.append((self._control,substep))
        if i<8:
            r=deepcopy(rows[i])
            if fault=="rod_speed" and i==7:r["velocities"][0,0]=.01
            if fault=="source" and i==3: r=None
        else:r=None
        return dict(physics_step=i+3,policy_step=self._control,substep=substep,
                    guard_failure="joint_limit" if fault=="original" and i==3 else None,
                    synthetic_measurement=r)
    monkeypatch.setattr(director.DirectorExchange,"_step",physics)
    monkeypatch.setattr(director,"director_observation_failure",lambda r:r["quiet_prior_guard_failure"])
    monkeypatch.setattr(m,"make_quiet_gate",lambda self:g)
    monkeypatch.setattr(g,"observe_exchange",lambda r:g.observe_measurement(r["synthetic_measurement"]))
    x=object.__new__(m.TripodExchange);x._control=0
    return m,x,calls

def test_eight_current_rows_before_first_nonzero_action(monkeypatch):
    m,x,calls=setup(monkeypatch)
    schedule=plan.action_schedule()
    for i in range(8):
        row=x._step(schedule[0][None],i)
        assert row["guard_failure"] is None
        assert row["tripod_quiet"]["status"]==("passed" if i==7 else "pending")
    x._control=1
    row=x._step(schedule[1][None],0)
    assert calls==[(0,i) for i in range(8)]+[(1,0)]
    assert row["tripod_quiet"]["status"]=="passed"
    assert m.observation_failure(row) is None

@pytest.mark.parametrize("fault,steps,expected",[
    ("rod_speed",8,"tripod_quiet_admission_failed"),
    ("source",4,"tripod_quiet_source_invalid"),
    ("original",4,"joint_limit")])
def test_failure_retains_completed_row_and_prevents_approach(monkeypatch,fault,steps,expected):
    m,x,calls=setup(monkeypatch,fault);schedule=plan.action_schedule()
    for i in range(steps):row=x._step(schedule[0][None],i)
    assert row["physics_step"]==steps+2 and row["guard_failure"]==expected
    assert m.observation_failure(row)==expected
    x._control=1
    with pytest.raises(RuntimeError):x._step(schedule[1][None],0)
    assert len(calls)==steps

@pytest.mark.parametrize("fault",["skip","wrong_action","repeat"])
def test_unbudgeted_sequence_is_rejected_before_sdk(monkeypatch,fault):
    m,x,calls=setup(monkeypatch);schedule=plan.action_schedule()
    if fault=="skip":x._control=1
    if fault=="repeat":x._step(schedule[0][None],0)
    action=schedule[x._control][None].clone()
    if fault=="wrong_action":action[0,7]=.2
    before=len(calls)
    with pytest.raises((RuntimeError,ValueError)):x._step(action,0 if fault!="repeat" else 0)
    assert len(calls)==before

def test_tampered_saved_quiet_receipt_is_rejected(monkeypatch):
    m,x,calls=setup(monkeypatch);schedule=plan.action_schedule()
    row=x._step(schedule[0][None],0)
    row["tripod_quiet"]["status"]="passed"
    with pytest.raises(ValueError):m.observation_failure(row)
