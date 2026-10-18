import importlib
import numpy as np
import pytest


def gate(stage='lift'):
    module = importlib.import_module('go2_pvcnn.control.dh116.restore_gate')
    return module.RestoreGate(stage, minimum_loads_n=[.8]*4,
                              saved_anchors_local=np.zeros((4, 3)))


def sample(**changes):
    result = dict(safety_ok=True, active_constraints=[], settled=True,
                  approach_ok=True, loads_n=[1.]*4, valid_contacts=[True]*4,
                  normals=[[1,0,0],[-1,0,0],[1,0,0],[-1,0,0]],
                  anchors_local=np.zeros((4,3)))
    result.update(changes)
    return result


@pytest.mark.parametrize('stage', ['approach', 'close', 'hold', 'lift'])
def test_new_continuous_time_required(stage):
    g = gate(stage)
    steps = 10 if stage in ('approach', 'close') else 100
    for i in range(steps):
        assert not g.update(i*.01, **sample())['ready']
    result = g.update(steps*.01, **sample())
    assert result['ready']
    assert 'grasp_lift_passed' not in result


@pytest.mark.parametrize('changes', [dict(safety_ok=False),
    dict(active_constraints=['endpoint_binding']),
    dict(anchors_local=np.full((4,3), .002))])
def test_safety_failure_latches(changes):
    g = gate()
    assert g.update(0., **sample(**changes))['aborted']
    for i in range(1, 105):
        r = g.update(i*.01, **sample())
        assert r['aborted'] and not r['ready']


def test_preload_loss_restarts_fresh_hold_timer():
    g = gate()
    for i in range(90):
        g.update(i*.01, **sample())
    assert not g.update(.90, **sample(loads_n=[.79]*4))['ready']
    for i in range(91,191):
        assert not g.update(i*.01, **sample())['ready']
    assert g.update(1.91, **sample())['ready']


@pytest.mark.parametrize('changes', [dict(valid_contacts=[False]*4),
    dict(normals=[[1,0,0]]*4),dict(approach_ok=False)])
def test_missing_required_evidence_blocks_release(changes):
    g = gate('close' if 'approach_ok' in changes else 'lift')
    for i in range(105):
        assert not g.update(i*.01, **sample(**changes))['ready']


@pytest.mark.parametrize('time', [1., -.01, float('nan')])
def test_bad_initial_time_rejected(time):
    with pytest.raises(ValueError):
        gate().update(time, **sample())


def test_missing_interval_and_nan_rejected():
    g = gate()
    g.update(0., **sample())
    with pytest.raises(ValueError):
        g.update(.02, **sample())
    with pytest.raises(ValueError):
        g.update(.01, **sample(loads_n=[float('nan')]*4))


@pytest.mark.parametrize('stage', ['finish', 'lower', 'settle'])
def test_unsupported_entry_rejected(stage):
    with pytest.raises(ValueError):
        gate(stage)
