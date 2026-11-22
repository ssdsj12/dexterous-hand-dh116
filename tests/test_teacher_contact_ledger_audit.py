"""Offline audit must preserve actual contact/support/action clock evidence."""
import hashlib
import importlib.util
import json
from copy import deepcopy
import numpy as np
import pytest

from test_dh116_power_palm_contact_evidence import build
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned


def module():
    name = "scripts.audit_dh116_teacher_contact_ledger"
    assert importlib.util.find_spec(name) is not None, "missing offline teacher contact ledger auditor"
    return __import__(name, fromlist=["audit_file"])


def exchange():
    rod = build(proximal=True, table=True, load=.125)
    rod["velocities"] = np.zeros((1, 41, 3))
    keys = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid", "physics_step", "rod_tick")
    post = {k: rod[k] for k in keys}
    pre = dict(post, physics_step=2, rod_tick=0)
    return dict(physics_step=3, rod=rod, guard_failure=None,
        action_readback=dict(pre=dict(identity=[pre], hands={}), post=dict(identity=[post], hands={}),
                             request=dict(action24=np.zeros((1, 24)), action_contract="normalized_command24")))


def source(tmp_path, row):
    path = tmp_path/"physics.jsonl"
    path.write_text(json.dumps(encode_owned(dict(raw_sequence=15, payload=dict(exchange=row))))+"\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_contact_audit_retains_support_and_missing_joint_sources_without_granting_success(tmp_path):
    m = module()
    path, sha = source(tmp_path, exchange())
    report = m.audit_file(path, expected_sha256=sha, grasp="tripod")
    assert report["rows"] == 1 and report["bilateral_rows"] == 1
    assert report["support_free_rows"] == 0 and report["table_contact_rows"] == 1
    first = report["ledger"][0]
    assert first["contact"]["per_hand"]["right"]["required_mesh_ids"] == [6, 7, 8]
    assert max(first["contact"]["external_support_force_norm_n"]["support_0"]) == pytest.approx(.125)
    assert first["joint_state"]["left"]["valid"] is False
    assert first["joint_state"]["left"]["q"] is None
    assert first["joint_state"]["left"]["reason"] == "missing_joint_source"
    assert report["native_runs"] == report["ppo_updates"] == 0
    assert report["teacher_admission"] is report["grasp_qualified"] is False


@pytest.mark.parametrize("fault", ["raw_sha", "action_tick", "capture"])
def test_contact_audit_rejects_corrupt_or_cross_clock_evidence(tmp_path, fault):
    m = module()
    row = exchange()
    if fault == "action_tick":
        row["action_readback"]["post"]["identity"][0]["physics_step"] = 2
    elif fault == "capture":
        row["rod"]["rod_mesh_contact_pairs"]["solver_buffer"]["overflow"] = True
    path, sha = source(tmp_path, row)
    with pytest.raises(ValueError):
        m.audit_file(path, expected_sha256="a"*64 if fault == "raw_sha" else sha, grasp="tripod")


def test_contact_audit_refuses_power_alias_and_repeated_physical_tick(tmp_path):
    m = module()
    row = exchange()
    path, sha = source(tmp_path, row)
    with pytest.raises(ValueError, match="pinch or tripod"):
        m.audit_file(path, expected_sha256=sha, grasp="power")
    with path.open("a") as f:
        f.write(json.dumps(encode_owned(dict(raw_sequence=16, payload=dict(exchange=deepcopy(row)))))+"\n")
    with pytest.raises(ValueError, match="consecutive"):
        m.audit_file(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), grasp="tripod")


def tracked_exchange():
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, MIMIC
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    row=exchange()
    # Deliberately non-USD order: physical slot order must never become action order.
    names=["finger22","wrist_z",*ACTIVE,"wrist_x","wrist_y","wrist_roll","wrist_pitch","wrist_yaw"]
    n=len(names)
    for phase in ("pre","post"):
        for hi,side in enumerate(("left","right")):
            q=np.full((1,n),.32 if phase=="post" else .30)
            q[0,names.index("finger22")]=.8
            targets=np.full((1,n),.33 if phase=="post" else .31)
            targets[0,names.index("finger22")]=0.
            row["action_readback"][phase]["hands"][side]=dict(
                joint_names=list(names),active_indices=[names.index(name) for name in ACTIVE],
                q=q,qd=np.zeros((1,n)),backend_targets=targets,
                measured_source="physx_backend_get_dof_positions/get_dof_velocities",
                backend_source="physx_backend_get_dof_position_targets",
                instance_object_id=10+hi,view_object_id=20+hi,prim_paths=[f"/World/{side}/wrist"],
                row_indices=[0],units_by_name={name:"m" if name in ("wrist_x","wrist_y","wrist_z") else "rad" for name in names})
    request=row["action_readback"]["request"]
    request.update(action_contract=deepcopy(ACTION_CONTRACT),
                   requested_finger_targets_rad=np.full((1,12),.34))
    return row


def test_tracking_audit_binds_names_and_targets_without_treating_passive_zero_as_error(tmp_path):
    row=tracked_exchange()
    path,sha=source(tmp_path,row)
    result=module().audit_file(path,expected_sha256=sha,grasp="tripod")
    audit=result["ledger"][0]["joint_tracking"]
    for hi,side in enumerate(("left","right")):
        assert audit[side]["valid"] is True
        joints={x["joint"]:x for x in audit[side]["joints"]}
        active=joints["finger12"]
        assert active["action_index"]==hi*6+1
        assert active["position_delta"]==pytest.approx(.02)
        assert active["actual_minus_sdk_target"]==pytest.approx(-.01)
        assert active["sdk_minus_requested_target"]==pytest.approx(-.01)
        assert active["sdk_target_delta"]==pytest.approx(.02)
        passive=joints["finger22"]
        assert passive["joint_role"]=="passive_finger"
        assert passive["sdk_target"] is None and passive["actual_minus_sdk_target"] is None
        assert passive["target_reason"]=="passive_joint_not_commanded"
        wrist=joints["wrist_z"]
        assert wrist["position_unit"]=="m" and wrist["actual_minus_sdk_target"]==pytest.approx(-.01)
        assert wrist["requested_target"] is None
        assert wrist["requested_target_reason"]=="wrist_request_requires_coordinate_and_projection_mapping"
    assert result["teacher_admission"] is False


def test_tracking_audit_retains_unknown_channels_as_na(tmp_path):
    row=tracked_exchange()
    del row["action_readback"]["request"]["requested_finger_targets_rad"]
    del row["action_readback"]["post"]["hands"]["left"]["backend_targets"]
    path,sha=source(tmp_path,row)
    audit=module().audit_file(path,expected_sha256=sha,grasp="tripod")["ledger"][0]["joint_tracking"]
    assert audit["left"]["valid"] is False and audit["left"]["reason"]=="missing_backend_target_source"
    right={j["joint"]:j for j in audit["right"]["joints"]}["finger12"]
    assert right["actual_minus_sdk_target"]==pytest.approx(-.01)
    assert right["requested_target"] is None and right["sdk_minus_requested_target"] is None
    assert right["requested_target_reason"]=="missing_requested_finger_targets"


@pytest.mark.parametrize("fault",["joint_order","active_mapping","nonfinite","target_bool","target_text","instance"])
def test_tracking_audit_refuses_cross_source_or_corrupt_target_values(tmp_path,fault):
    row=tracked_exchange()
    hand=row["action_readback"]["post"]["hands"]["right"]
    if fault=="joint_order":
        hand["joint_names"]=list(reversed(hand["joint_names"]))
    elif fault=="active_mapping":
        hand["active_indices"][0]=False
    elif fault=="nonfinite":
        hand["backend_targets"][0,0]=np.inf
    elif fault=="target_bool":
        hand["backend_targets"]=hand["backend_targets"].astype(bool)
    elif fault=="target_text":
        hand["backend_targets"]=hand["backend_targets"].astype(str).tolist()
    else:
        hand["instance_object_id"]=999
    path,sha=source(tmp_path,row)
    with pytest.raises(ValueError,match="tracking|target"):
        module().audit_file(path,expected_sha256=sha,grasp="tripod")


def two_tracked_rows():
    first=tracked_exchange()
    def advance(value):
        if isinstance(value,dict):
            return {k:v+1 if k=="physics_step" else v+12 if k=="rod_tick" else v+1/480 if k in ("timestamp_s","solver_time_s") else advance(v)
                    for k,v in value.items()}
        if isinstance(value,list):
            return [advance(v) for v in value]
        return deepcopy(value)
    return first,advance(first)


@pytest.mark.parametrize("fault",["alias_sides","mid_episode_identity"])
def test_tracking_audit_preserves_distinct_continuous_hand_owners(tmp_path,fault):
    first,second=two_tracked_rows()
    rows=[dict(raw_sequence=15+i,payload=dict(exchange=x)) for i,x in enumerate((first,second))]
    path=tmp_path/"physics.jsonl"
    def write():
        path.write_text("".join(json.dumps(encode_owned(x))+"\n" for x in rows))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    assert module().audit_file(path,expected_sha256=write(),grasp="tripod")["rows"]==2
    if fault=="alias_sides":
        for sample in (first,second):
            for phase in ("pre","post"):
                a=sample["action_readback"][phase]["hands"]
                for key in ("instance_object_id","view_object_id","prim_paths"):
                    a["right"][key]=deepcopy(a["left"][key])
    else:
        for phase in ("pre","post"):
            second["action_readback"][phase]["hands"]["right"]["instance_object_id"]=999
    with pytest.raises(ValueError,match="tracking.*identity|tracking.*owners"):
        module().audit_file(path,expected_sha256=write(),grasp="tripod")


def test_target_readback_gap_preserves_contact_and_marks_partial_interval(tmp_path):
    first,second=two_tracked_rows()
    rows=[dict(raw_sequence=15+i,payload=dict(exchange=x)) for i,x in enumerate((first,second))]
    path=tmp_path/"physics.jsonl"
    def run():
        path.write_text("".join(json.dumps(encode_owned(x))+"\n" for x in rows))
        return module().audit_file(path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),grasp="tripod")
    assert run()["tracking_summary"]["left"]["full_interval_coverage"] is True
    del second["action_readback"]["post"]["hands"]["left"]["backend_targets"]
    result=run()
    summary=result["tracking_summary"]["left"]
    assert result["rows"]==result["bilateral_rows"]==2
    assert summary["valid_rows"]==1 and summary["missing_rows"]==1
    assert summary["full_interval_coverage"] is False and summary["missing_sdk_steps"]==[4]
    assert summary["residual_scope"]=="valid_rows_only"
    assert result["ledger"][1]["joint_tracking"]["left"]["joints"] is None
    assert result["teacher_admission"] is False
