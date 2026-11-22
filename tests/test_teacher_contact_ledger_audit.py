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
