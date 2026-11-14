"""CPU fixtures only: partial readbacks never become complete Teacher inputs."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits
from go2_pvcnn.control.dh116.tabletop_teacher_preaction_readback import (
    SOURCE, VERSION, adapt_named_hands, audit_legacy_pinch, capture_named_hand,
    require_teacher_common,
)


def clock(**changes):
    result = dict(run_uuid="fixture", env_id=0, episode_uid="ep0", reset_id=1,
                  rod_uid="rod0", physics_step=2, rod_tick=0, policy_step=0,
                  timestamp_s=2 / 480, phase="pre_action_before_target_write")
    return {**result, **changes}


def packet(side="left", names=JOINTS):
    q = np.linspace(.02, .4, 11)
    qd = np.linspace(-.2, .3, 11)
    order = [JOINTS.index(n) for n in names]
    return dict(version=VERSION, source=SOURCE, side=side, identity=clock(),
                joint_names=list(names), q=q[order].tolist(), qd=qd[order].tolist(),
                units={"q": "rad", "qd": "rad/s"}, source_valid=True,
                dof_types=[0]*11, dof_type_source="physx_backend_get_dof_types")


def test_exact_indices_and_normalization_allow_sdk_name_permutation():
    p = packet(names=JOINTS[::-1])
    result = adapt_named_hands([p], expected_identity=clock())
    q = dict(zip(p["joint_names"], p["q"]))
    qd = dict(zip(p["joint_names"], p["qd"]))
    lo, hi = active_limits()
    np.testing.assert_allclose(result["values"][:6],
        2 * (np.array([q[n] for n in ACTIVE]) - lo) / (hi - lo) - 1)
    np.testing.assert_allclose(result["values"][12:18], [qd[n] / .3 for n in ACTIVE])
    np.testing.assert_allclose(result["values"][78:83],
        [(q[f] - ratio*q[m] - offset)/.02 for f,m,ratio,offset in MIMIC])
    assert np.flatnonzero(result["valid"]).tolist() == list(range(6)) + list(range(12,18)) + list(range(78,83))
    assert np.isnan(result["values"][~result["valid"]]).all()
    assert result["qualified_for_teacher_ppo"] is False
    assert result["new_physics_steps"] == result["optimizer_updates"] == 0


def test_two_named_hands_still_leave_337_channels_unknown():
    result = adapt_named_hands([packet(), packet("right")], expected_identity=clock())
    assert result["valid"].sum() == 34
    assert np.flatnonzero(result["valid"]).tolist() == list(range(24)) + list(range(78,88))
    with pytest.raises(ValueError, match="337"):
        require_teacher_common(result)


@pytest.mark.parametrize("edit", [
    lambda p: p.pop("joint_names"),
    lambda p: p["joint_names"].__setitem__(1, p["joint_names"][0]),
    lambda p: p["q"].__setitem__(0, float("nan")),
    lambda p: p["qd"].append(0.),
    lambda p: p.update(source="physx_backend_get_dof_position_targets"),
    lambda p: p.update(source="RopeBatch.mesh_wrenches"),
    lambda p: p.update(source_valid=False),
    lambda p: p.update(source_valid=1),
    lambda p: p["units"].update(q="deg"),
    lambda p: p.update(side="both"),
    lambda p: p["identity"].update(physics_step=3),
    lambda p: p["identity"].update(rod_tick=12),
    lambda p: p["identity"].update(timestamp_s=0.),
    lambda p: p["identity"].update(episode_uid="ep1"),
    lambda p: p["identity"].update(reset_id=2),
    lambda p: p["identity"].update(run_uuid="other"),
    lambda p: p["identity"].update(phase="post_action"),
])
def test_rejects_unknown_source_order_units_and_stale_identity(edit):
    p = packet()
    edit(p)
    with pytest.raises(ValueError):
        adapt_named_hands([p], expected_identity=clock())


def test_duplicate_side_is_not_a_second_hand():
    with pytest.raises(ValueError, match="duplicate"):
        adapt_named_hands([packet(), packet()], expected_identity=clock())


def test_capture_owns_copies_and_reads_real_names_not_targets():
    p = packet(names=JOINTS[::-1])
    q, qd = np.array([p["q"]]), np.array([p["qd"]])
    hand = SimpleNamespace(joint_names=p["joint_names"], data=SimpleNamespace(
        joint_pos=q, joint_vel=qd, joint_pos_target=np.full_like(q, 99.)))
    hand.root_physx_view=SimpleNamespace(get_dof_types=lambda:np.zeros((1,11),dtype=int))
    captured = capture_named_hand(hand, side="left", read_identity=clock, data_env_index=0)
    assert captured["q"] == p["q"]
    q[:] = 99.
    qd[:] = 99.
    assert captured["q"] == p["q"] and captured["qd"] == p["qd"]
    assert captured["identity"] == clock()


def test_capture_rejects_clock_advance_during_copy():
    p=packet()
    hand=SimpleNamespace(joint_names=p["joint_names"],data=SimpleNamespace(
        joint_pos=np.array([p["q"]]),joint_vel=np.array([p["qd"]])))
    hand.root_physx_view=SimpleNamespace(get_dof_types=lambda:np.zeros((1,11),dtype=int))
    readings=iter([clock(),clock(physics_step=3,timestamp_s=3/480)])
    with pytest.raises(ValueError, match="changed"):
        capture_named_hand(hand, side="left",read_identity=lambda:next(readings),data_env_index=0)


def legacy_row():
    return dict(phase_step=1, physics_step=3, sdk_step=3, timestamp_s=3/480,
        sdk_time_s=3/480, dt_s=1/480, rod_tick=12, rod_substep=12,
        policy_step=0,substep=0, env_id=0,episode_uid="ep0",reset_id=1,rod_uid="rod0",
        side="left",pre_state={"q":[0.]*11,"qd":[0.]*11},
        post_state={"q":[1.]*11,"qd":[1.]*11},
        rod_nodes_w=[[0.,0.,0.]], rod_node_contact_forces_w=[[0.,0.,0.]],
        rod_mesh_wrenches_w={"table":[0.]*6}, canonical_action24=[0.]*24,
        full_state371_bound=False)


def test_historical_unnamed_arrays_are_unknown_not_assigned_constant_joint_order():
    result=audit_legacy_pinch([legacy_row()])
    assert result["rows"] == 1 and result["admitted_371d_channels"] == 0
    assert result["row_audits"][0]["pre_state_sdk_step_derived"] == 2
    assert result["row_audits"][0]["post_state_sdk_step"] == 3
    assert result["unknown_indices"] == list(range(371))
    assert result["qualified_for_teacher_ppo"] is False
    assert "missing_actual_sdk_joint_names" in result["blockers"]


@pytest.mark.parametrize("edit", [
    lambda r:r.update(sdk_step=4),
    lambda r:r.update(rod_tick=0),
    lambda r:r.update(timestamp_s=0.),
    lambda r:r.update(full_state371_bound=True),
    lambda r:r["pre_state"].pop("qd"),
])
def test_legacy_audit_rejects_clock_or_source_corruption(edit):
    row=legacy_row()
    edit(row)
    with pytest.raises(ValueError):
        audit_legacy_pinch([row])


def test_zeros_or_forged_mask_cannot_promote_partial_result():
    result=adapt_named_hands([packet()],expected_identity=clock())
    result["values"][~result["valid"]]=0.
    result["valid"][:]=True
    with pytest.raises(ValueError, match="source"):
        require_teacher_common(result)



def test_file_audit_binds_sources_and_never_overwrites(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from scripts.audit_dh116_teacher_preaction_sources import audit_files
    from go2_pvcnn.control.dh116.tabletop_teacher_preaction_readback import CONTRACT_SHA256, COUPLING_SHA256
    root=Path(__file__).resolve().parents[1]
    ledger=tmp_path/"physics.jsonl"
    ledger.write_text(json.dumps(legacy_row())+"\n")
    plan=tmp_path/"plan.json"
    plan.write_text(json.dumps({"sources":{"local_sources":{
        "docs/contracts/stereo_privileged_state_contract.json":CONTRACT_SHA256,
        "src/go2_pvcnn/control/dh116/linear_coupling.py":COUPLING_SHA256}},
        "state371_policy_binding":False,"native_execution_performed":True}))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    output=tmp_path/"audit.json"
    result=audit_files(ledger,sha(ledger),plan,sha(plan),output,root)
    assert result["inputs"]["ledger"]["sha256"] == sha(ledger)
    assert sum(b["slice"][1]-b["slice"][0] for b in result["blocks"]) == 371
    assert all(b["valid_channels"] == 0 for b in result["blocks"])
    assert result["admitted_371d_channels"] == 0
    saved=output.read_bytes()
    with pytest.raises(FileExistsError):
        audit_files(ledger,sha(ledger),plan,sha(plan),output,root)
    assert output.read_bytes() == saved
    with pytest.raises(ValueError,match="SHA256"):
        audit_files(ledger,"0"*64,plan,sha(plan),tmp_path/"bad.json",root)
    assert not (tmp_path/"bad.json").exists()
    wrong=json.loads(plan.read_text())
    wrong["sources"]["local_sources"]["docs/contracts/stereo_privileged_state_contract.json"]="0"*64
    plan.write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match="contract"):
        audit_files(ledger,sha(ledger),plan,sha(plan),tmp_path/"bad.json",root)


def test_capture_requires_actual_angular_dof_readback():
    p=packet()
    hand=SimpleNamespace(joint_names=p["joint_names"],data=SimpleNamespace(
        joint_pos=np.array([p["q"]]),joint_vel=np.array([p["qd"]])))
    with pytest.raises(ValueError,match="DOF"):
        capture_named_hand(hand,side="left",read_identity=clock,data_env_index=0)
    hand.root_physx_view=SimpleNamespace(get_dof_types=lambda:np.ones((1,11),dtype=int))
    with pytest.raises(ValueError,match="DOF"):
        capture_named_hand(hand,side="left",read_identity=clock,data_env_index=0)


@pytest.mark.parametrize("edit", [
    lambda p:p.pop("dof_types"),
    lambda p:p.update(dof_types=[1]*11),
    lambda p:p.update(dof_types=[0.]*11),
    lambda p:p.update(dof_type_source="joint_name_guess"),
])
def test_admission_requires_native_angular_type_provenance(edit):
    p=packet()
    edit(p)
    with pytest.raises(ValueError,match="DOF"):
        adapt_named_hands([p],expected_identity=clock())
