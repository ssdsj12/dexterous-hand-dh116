import hashlib,json
import numpy as np
import pytest
from go2_pvcnn.control.dh116.pose_artifact import load_active_pose


def write(tmp_path,**updates):
    data=dict(schema='dh116-static-pose-v1',asset_sha256='a'*64,layers={'source':'b'*64},
        states={s:[0,-.6,0,-2.2,0,1.8,.7]+[.2]*6 for s in ('left','right')},
        grasp_qualified=False)
    data.update(updates);p=tmp_path/'pose.json';p.write_text(json.dumps(data))
    return p,hashlib.sha256(p.read_bytes()).hexdigest()


def test_valid_pinned_static_pose(tmp_path):
    p,sha=write(tmp_path);r=load_active_pose(p,sha,'a'*64,{'source':'b'*64})
    assert set(r)=={'left','right'} and r['left'].shape==(13,)


@pytest.mark.parametrize('bad',['sha','asset','layer','side','hand','claim'])
def test_invalid_pose_evidence_rejected(tmp_path,bad):
    changes={}
    if bad=='asset':changes['asset_sha256']='c'*64
    if bad=='layer':changes['layers']={'source':'c'*64}
    if bad=='side':changes['states']={'left':[.2]*13}
    if bad=='hand':changes['states']={s:[.2]*7+[0]*6 for s in ('left','right')}
    if bad=='claim':changes['grasp_qualified']=True
    p,sha=write(tmp_path,**changes)
    with pytest.raises(ValueError):load_active_pose(p,'0'*64 if bad=='sha' else sha,'a'*64,{'source':'b'*64})
