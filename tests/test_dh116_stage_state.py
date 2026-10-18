import copy
import hashlib
import importlib
import numpy as np
import pytest


def api():
    return importlib.import_module('go2_pvcnn.control.dh116.stage_state')


def state():
    return dict(schema='dh116-stage-state-v1', entry_stage='close',
                pins={'asset': 'a'*64, 'config': 'b'*64},
                source=dict(run='trial', step=40000, passed_stages=['settle','approach']),
                robot={s:dict(joint_names=[f'{s}_panda_joint{i}' for i in range(1,8)]+
                    ['finger11','finger12','finger13','finger21','finger22','finger31',
                     'finger32','finger41','finger42','finger51','finger52'],
                    q=[.1]*18, qd=[0.]*18, target=[.1]*18,
                    lower=[0.]*18, upper=[1.]*18,
                    root_state=[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]) for s in ('left','right')},
                rod=dict(model_sha256='c'*64,dynamic=dict(rod_position=np.array([np.linspace(0,.5,41),np.zeros(41),np.ones(41)*.308]).tolist(),
                    rod_velocity=np.zeros((3,41)).tolist(),rod_director=np.repeat(np.eye(3)[:,:,None],40,axis=2).tolist(),
                    rod_omega=np.zeros((3,40)).tolist()),
                    reference={'rest_lengths':[.0125]*40, 'rest_voronoi_lengths':[.0125]*39,
                               'rest_sigma':np.zeros((3,40)).tolist(),'rest_kappa':np.zeros((3,39)).tolist()}),
                material_ids=[36,36,4,4], anchors_local=np.zeros((4,3)).tolist(),
                settled_heights_m=[.308,.308], target_loads_n=[1.]*4)


def test_roundtrip_and_external_digest(tmp_path):
    m=api(); p=tmp_path/'state.json'; s=state()
    digest=m.save_state(p,s)
    assert digest==hashlib.sha256(p.read_bytes()).hexdigest()
    assert m.load_state(p,digest,s['pins'])==s
    with pytest.raises(FileExistsError):m.save_state(p,s)
    with pytest.raises(ValueError):m.load_state(p,'0'*64,s['pins'])
    with pytest.raises(ValueError):m.load_state(p,digest,{'asset':'c'*64})


@pytest.mark.parametrize('mutation', [
    lambda s:s.update(entry_stage='lower'),
    lambda s:s['source'].update(passed_stages=['settle']),
    lambda s:s['robot']['left']['q'].__setitem__(0,2.),
    lambda s:s['robot']['left']['qd'].__setitem__(0,float('nan')),
    lambda s:s['robot']['right']['joint_names'].__setitem__(0,'wrong'),
    lambda s:s['rod']['reference'].pop('rest_sigma'),
    lambda s:s['rod'].pop('model_sha256'),
    lambda s:s['rod']['dynamic']['rod_director'][0][0].__setitem__(0,2.),
    lambda s:s.update(material_ids=[4,4,4,4]),
    lambda s:s['pins'].update(asset='missing'),
])
def test_invalid_payload_rejected_before_writing(tmp_path,mutation):
    s=state();mutation(s);p=tmp_path/'bad.json'
    with pytest.raises((ValueError,KeyError)):api().save_state(p,s)
    assert not p.exists()


def test_validation_returns_independent_state():
    s=state();r=api().validate_state(s,s['pins'])
    r['robot']['left']['q'][0]=.4
    assert s['robot']['left']['q'][0]==.1


def test_all_entry_predecessors_required():
    for stage, phases in [('approach',['settle']),('hold',['settle','approach','close']),
                          ('lift',['settle','approach','close','hold'])]:
        s=state();s['entry_stage']=stage;s['source']['passed_stages']=phases
        assert api().validate_state(s,s['pins'])['entry_stage']==stage
