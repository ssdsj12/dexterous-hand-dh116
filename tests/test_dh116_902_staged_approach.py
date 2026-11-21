"""Staged commands preserve the 902 terminal target without moving while closing."""
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT

def api():
    from scripts import compile_dh116_902_staged_approach as module
    return module

def fixture():
    base=np.zeros((66,24))
    base[:,8]=.05;base[:,9]=-.1
    base[:,15]=.03;base[:,22]=-.02
    return base,np.array([.4,.3,.4,.9,.4,.4])

def test_three_phases_preserve_endpoint_and_keep_other_hand_fixed():
    base,q=fixture()
    out=api().compile_candidate(base,q,extra_middle_open_rad=.02,phase_windows=(16,66,54))
    a=out["actions"];scale=8/480*ACTION_CONTRACT["joint_rate_rad_s"]
    assert a.shape==(136,24) and np.max(abs(a))<=1.
    assert np.max(abs(a[:,list(range(6))+list(range(12,15))+list(range(18,21))]))==0.
    assert not np.any(a[:16,[15,16,17,21,22,23]])
    assert not np.any(a[16:82,6:12])
    assert not np.any(a[82:,[15,16,17,21,22,23]])
    assert abs(a[:16,9].sum()*scale-(base[:,9].sum()*scale-.02))<1e-6
    assert abs(a[82:,9].sum()*scale-.02)<1e-6
    measured=q.copy()
    for action in a:
        for _ in range(8):measured=(measured+action[6:12]*scale/8).astype(np.float32).astype(float)
    assert np.max(abs(measured-(q+base[:,6:12].sum(0)*scale)))<1e-6
    assert out["requires_measured_wrist_settling"] is True
    assert out["native_admissible"] is False
    assert out["close_success_requires"]=="opposed_contact_and_delivered_reaction"
    assert out["phase_ranges"]=={"open_middle":[0,16],"move_wrist":[16,82],"close_fixed_wrist":[82,136]}

@pytest.mark.parametrize("bad",["held_hand","nan","too_large","wrong_count","bool_count","negative_relief","over_relief","initial_limit","intermediate_limit"])
def test_rejects_invalid_or_limit_violating_candidate(bad):
    base,q=fixture();kw=dict(extra_middle_open_rad=.02,phase_windows=(16,66,54))
    if bad=="held_hand":base[0,0]=.01
    if bad=="nan":base[0,8]=np.nan
    if bad=="too_large":base[:,8]=1.
    if bad=="wrong_count":kw["phase_windows"]=(9,66,0)
    if bad=="bool_count":kw["phase_windows"]=(True,66,54)
    if bad=="negative_relief":kw["extra_middle_open_rad"]=-.02
    if bad=="over_relief":kw["extra_middle_open_rad"]=.061
    if bad=="initial_limit":q[1]=0.
    if bad=="intermediate_limit":
        base[:,9]=-.001;q[3]=.01
    with pytest.raises(ValueError):api().compile_candidate(base,q,**kw)

def test_rejects_opening_phase_without_original_middle_opening():
    base,q=fixture();base[:,9]=.01
    with pytest.raises(ValueError):api().compile_candidate(base,q,extra_middle_open_rad=.02,phase_windows=(16,66,54))

def test_invalid_path_is_not_silently_clipped():
    base,q=fixture()
    with pytest.raises(ValueError):api().compile_candidate(base,q,extra_middle_open_rad=.02,phase_windows=(1,1,1))

@pytest.mark.parametrize("start",[15,21])
@pytest.mark.parametrize("stage",["source","compressed"])
def test_wrist_vector_limit_rejects_diagonal_hidden_normalization(start,stage):
    base,q=fixture();base[:,15:18]=0.;base[:,21:24]=0.
    base[:,start:start+2]=.8 if stage=="source" else .1
    counts=(16,66 if stage=="source" else 8,54)
    with pytest.raises(ValueError,match="vector"):
        api().compile_candidate(base,q,extra_middle_open_rad=.02,phase_windows=counts)
