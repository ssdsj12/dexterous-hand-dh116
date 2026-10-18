import copy
import pytest
from go2_pvcnn.control.dh116.reaction_pair import compare_pair


def case():
    base=dict(passed=True,steps=1,asset_sha256='a',run_config={'dt':.01,'steps':1},
        initial_state={'left':[0],'right':[0]},contact=dict(reaction_applied=False,peak_force_n=1.))
    active=copy.deepcopy(base);active['contact']['reaction_applied']=True
    row=dict(step=0,passed=True,sides={s:dict(joint_names=['joint'],actual_joint_positions=[0.],
        actual_joint_velocities=[0.]) for s in ('left','right')})
    other=copy.deepcopy(row);other['sides']['left']['actual_joint_positions']=[.001]
    other['sides']['left']['actual_joint_velocities']=[.01]
    return base,active,[row],[other]


def test_measured_response_is_required():
    b,a,br,ar=case();r=compare_pair(b,a,br,ar)
    assert r['passed'] and r['max_joint_delta_rad']==pytest.approx(.001)
    assert not r['grasp_qualified']
    assert not compare_pair(b,a,br,br)['passed']


@pytest.mark.parametrize('fault',['config','initial','asset','unfinished','disabled','no_contact'])
def test_pair_rejects_incomparable_or_invalid_trials(fault):
    b,a,br,ar=case()
    if fault=='config':a['run_config']['dt']=.02
    if fault=='initial':a['initial_state']['left']=[1]
    if fault=='asset':a['asset_sha256']='b'
    if fault=='unfinished':ar=[]
    if fault=='disabled':a['contact']['reaction_applied']=False
    if fault=='no_contact':a['contact']['peak_force_n']=0.
    with pytest.raises(ValueError):compare_pair(b,a,br,ar)
