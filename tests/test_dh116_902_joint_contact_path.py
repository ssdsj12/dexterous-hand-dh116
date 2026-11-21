import copy,hashlib,json
from pathlib import Path
import pytest
from scripts import certify_dh116_902_joint_contact_path as m
from scripts.search_dh116_902_joint_contact import ROOT,ORIGINAL_REQUEST,ORIGINAL_REQUEST_SHA256,decode_candidate

def fixture(tmp_path):
    target=json.loads((ROOT/ORIGINAL_REQUEST).read_bytes())["target"]
    pose,q=decode_candidate([0]*9,target)
    row=dict(ordinal=0,x=[0]*9,pose_wxyz=pose.tolist(),active_q_rad=q.tolist(),endpoint_candidate_passed=True)
    search=dict(source_request_sha256=ORIGINAL_REQUEST_SHA256,receipt_sha256="a"*64,rows=[row])
    p=tmp_path/"search.json";p.write_text(json.dumps(search))
    cfg=dict(version=m.VERSION,source_request=ORIGINAL_REQUEST,source_request_sha256=ORIGINAL_REQUEST_SHA256,
        search_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),receipt_sha256="a"*64,
        selected_ordinal=0,pose_wxyz=row["pose_wxyz"],active_q_rad=row["active_q_rad"],
        native_admissible=False,teacher_admission=False)
    return cfg,p

def test_candidate_cannot_change_target_after_search(tmp_path):
    cfg,p=fixture(tmp_path)
    assert m.select_target(cfg,p,"a"*64)["ordinal"]==0
    cfg=copy.deepcopy(cfg);cfg["active_q_rad"][4]+=.01
    with pytest.raises(ValueError):m.select_target(cfg,p,"a"*64)

def test_candidate_rejects_changed_search_or_wrong_receipt(tmp_path):
    cfg,p=fixture(tmp_path)
    with pytest.raises(ValueError):m.select_target(cfg,p,"b"*64)
    p.write_text(p.read_text()+" ")
    with pytest.raises(ValueError):m.select_target(cfg,p,"a"*64)
