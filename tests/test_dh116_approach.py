import hashlib
import json
import numpy as np
import pytest
from go2_pvcnn.control.dh116.approach import approach_target,load_approach


def pair():
    a={s:np.zeros(13) for s in ('left','right')}
    b={s:np.full(13,.05) for s in a}
    return a,b


def test_smooth_approach_stays_within_prechecked_joint_segment():
    a,b=pair()
    assert np.allclose(approach_target(a,b,.4)['left'],a['left'])
    assert np.allclose(approach_target(a,b,2.)['left'],.025)
    assert np.allclose(approach_target(a,b,4.)['left'],b['left'])
    dt=.001
    targets=np.array([approach_target(a,b,t)['left'] for t in np.arange(0,4,dt)])
    assert np.abs(np.diff(targets,axis=0)/dt).max()<=.1


def test_rejects_overfast_or_invalid_reference():
    a,b=pair();b['left'][0]=1.
    with pytest.raises(ValueError,match='speed'):approach_target(a,b,1.)
    with pytest.raises(ValueError):approach_target(a,b,np.nan)


def artifact(tmp_path,**changes):
    a,b=pair()
    data=dict(static_wire_source_sweep_passed=True,motion_qualified=False,layers={'asset':'pin'},
        input={'waypoints':[{s:v.tolist() for s,v in x.items()} for x in (a,a,b,b)]},
        covered_intervals=[dict(leg=1,start=0.,end=1.,minimum_lower_bound_m=.0011)])
    data.update(changes);path=tmp_path/'certificate.json';path.write_text(json.dumps(data))
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_certificate_pin_and_coverage_required(tmp_path):
    p,h=artifact(tmp_path);a,b=load_approach(p,h,{'asset':'pin'})
    assert np.allclose(b['right'],.05)
    with pytest.raises(ValueError):load_approach(p,'0'*64,{'asset':'pin'})
    with pytest.raises(ValueError):load_approach(p,h,{'asset':'changed'})
    p,h=artifact(tmp_path,covered_intervals=[])
    with pytest.raises(ValueError,match='coverage'):load_approach(p,h,{'asset':'pin'})
    p,h=artifact(tmp_path,static_wire_source_sweep_passed=False)
    with pytest.raises(ValueError):load_approach(p,h,{'asset':'pin'})
