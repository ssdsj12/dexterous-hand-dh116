from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts import audit_dh116_pinch_dynamic_fk as audit
from go2_pvcnn.control.dh116.linear_coupling import JOINTS,expand

ROOT=Path(__file__).resolve().parents[1]

def toy():
    a=np.eye(4);b=np.eye(4);b[0,3]=.02
    return SimpleNamespace(root="/base_link",edges=[("finger13","/base_link","/finger14_link",a,b,np.array([0.,0.,1.]))])

def test_full_fk_uses_measured_passive_angle_without_mimic_replacement():
    q=expand(np.array([.2,.2,.4,.4,.4,.4]));root=np.eye(4)
    a=audit.full_named_poses(toy(),root,dict(zip(JOINTS,q)))
    q[JOINTS.index("finger13")]+=.01
    b=audit.full_named_poses(toy(),root,dict(zip(JOINTS,q)))
    assert np.linalg.norm(a["finger14_link"][:3,3]-b["finger14_link"][:3,3])>0
    assert audit.pose_error(a["finger14_link"],b["finger14_link"])["rotation_rad"]==pytest.approx(.01)
    assert np.array_equal(root,np.eye(4))

@pytest.mark.parametrize("fault",["missing","nan","bad_root","nonrigid","unknown_axis"])
def test_malformed_named_fk_rejected(fault):
    q=dict(zip(JOINTS,expand(np.array([.2,.2,.4,.4,.4,.4]))));root=np.eye(4);fk=toy()
    if fault=="missing":q.pop("finger13")
    elif fault=="nan":q["finger13"]=float("nan")
    elif fault=="bad_root":root=np.zeros((3,3))
    elif fault=="nonrigid":root[0,0]=2
    else:fk.edges[0]=(*fk.edges[0][:-1],np.array([0.,0.,2.]))
    with pytest.raises(ValueError):audit.full_named_poses(fk,root,q)

def test_fk_and_error_are_world_rigid_transform_invariant():
    q=dict(zip(JOINTS,expand(np.array([.2,.2,.4,.4,.4,.4]))))
    root=np.eye(4);t=np.eye(4);t[:3,:3]=Rotation.from_euler("xyz",[.2,-.3,.1]).as_matrix();t[:3,3]=[1,2,3]
    first=audit.full_named_poses(toy(),root,q);second=audit.full_named_poses(toy(),t,q)
    for n in first:assert np.allclose(t@first[n],second[n],atol=1e-14)
    ref=first["finger14_link"].copy();ref[2,3]+=.001
    assert audit.pose_error(ref,first["finger14_link"])["position_m"]==pytest.approx(
        audit.pose_error(t@ref,second["finger14_link"])["position_m"])

@pytest.fixture(scope="module")
def actual():return audit.run(ROOT)

def test_actual_16_native_endpoints_keep_three_different_sources(actual):
    assert len(actual["rows"])==16
    assert [r["identity"]["physics_step"] for r in actual["rows"]]==list(range(3,19))
    assert actual["new_physics_steps"]==actual["optimizer_updates"]==0
    assert actual["teacher_admission"] is False and actual["native_qualification"] is False
    for row in actual["rows"]:
        assert row["identity"]["episode_uid"]
        assert row["action24"]==row["issued_action24"]
        assert set(row["hands"])=={"left","right"}
        for h in row["hands"].values():
            assert len(h["q_actual_si"])==len(h["qd_actual_si_per_s"])==17
            assert set(h["models"])=={"actual_full11","actual_active_mimic","sdk_active_target_mimic"}
            for m in h["models"].values():
                assert len(m["link_errors"])==17
                assert set(m["required_pad_sdf"])=={"finger14_link","finger23_link"}
            assert h["contact_inferred_from_model"] is False

def test_actual_fk_consistency_is_separate_from_unreached_sdk_target(actual):
    for side in ("left","right"):
        s=actual["summary"][side]
        assert s["actual_full11"]["position_m"]["max"]<2e-6
        assert s["actual_full11"]["rotation_rad"]["max"]<1e-4
        # This archived nonzero interval has a real command tracking lag.
        assert s["sdk_active_target_mimic"]["position_m"]["max"]>s["actual_full11"]["position_m"]["max"]*5
    last=actual["rows"][-1]
    assert last["identity"]["physics_step"]==18
    assert last["raw_hand_rod_pair_samples"]==0

def test_stats_keep_finite_values_and_do_not_fabricate_empty_measurements():
    assert audit.statistics([]) is None
    with pytest.raises(ValueError):audit.statistics([float("nan")])
    assert audit.statistics([1,2,3])["max"]==3

@pytest.fixture(scope="module")
def measured_input():
    import json
    r=json.loads(audit.evidence.read_pinned(audit.evidence.RECEIPT,audit.evidence.RECEIPT_SHA))
    rows=[audit.decode_owned(json.loads(l)) for l in audit.evidence.read_pinned(
        Path(r["output"])/"physical.jsonl",r["artifact_sha256"]["physical.jsonl"]).splitlines()]
    event=next(r["payload"]["exchange"] for r in rows if r.get("operation")=="physical")
    geometry=audit.MeasuredClosureGeometry(ROOT)
    models={s:audit.load_geometry(ROOT,side=s) for s in ("left","right")}
    return event,geometry,models

@pytest.mark.parametrize("fault",["declared","submitted_target","policy","radius","pose","side_model","joint_q"])
def test_archived_identity_geometry_and_target_corruption_rejected(measured_input,fault):
    e,g,models=measured_input;e=deepcopy(e);models=dict(models)
    declared=deepcopy(e["action_readback"]["request"]["action24"])
    if fault=="declared":declared[0,1]=.2
    elif fault=="submitted_target":
        e["action_readback"]["submission"]["expected_backend_targets"]["right"]["values"][0,0]+=.1
    elif fault=="policy":e["policy_step"]+=1
    elif fault=="radius":e["material_director"]["reward_radius_sample"]["value"]*=2
    elif fault=="pose":e["closure_geometry"]["raw"]["hands"]["right"]["link_poses_xyzw"][0,0,0]+=.01
    elif fault=="side_model":models["right"]=models["left"]
    elif fault=="joint_q":e["action_readback"]["post"]["hands"]["right"]["q"][0,0]+=.01
    with pytest.raises(ValueError):audit.evaluate_exchange(e,declared,g,models)

def test_existing_output_is_rejected_before_any_archive_read(tmp_path,monkeypatch):
    output=tmp_path/"result.json";output.write_text("existing")
    def forbidden():raise AssertionError("must not read/replay before overwrite refusal")
    monkeypatch.setattr(audit,"run",forbidden)
    with pytest.raises(SystemExit):audit.main(["--output",str(output)])
    assert output.read_text()=="existing"
