"""Regression coverage for the causal, specimen-independent grasp feedback."""
from copy import deepcopy

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_wire_grasp_feedback import pack_wire_grasp_feedback


def sample():
    clock = dict(run_uuid="run", env_id=0, episode_uid="episode", reset_id=1,
                 rod_uid="rod", physics_step=10, rod_tick=96, timestamp_s=10/480,
                 solver_time_s=8/480, clock_valid=True)
    hand = dict(xyz_in_palm_m=[0., .012, -.016], tangent_in_palm=[0., 1., 0.],
                reaction={"finger14_link": dict(minimum_normal_n=0., mean_normal_n=.2,
                                                mean_force_on_hand_palm_n=[0., 0., -.1]),
                          "finger23_link": dict(minimum_normal_n=0., mean_normal_n=0.,
                                                mean_force_on_hand_palm_n=[0., 0., 0.])})
    return dict(clock=clock, measured_hands={side:deepcopy(hand) for side in ("left", "right")},
                reaction_valid=True, readback={"weight_n": .4}, spec_sha256="old")


def test_grasp_feedback_uses_palm_geometry_and_measured_force_without_spec_identity():
    sample_a = sample()
    a = pack_wire_grasp_feedback(sample_a, sample_a["clock"])
    sample_b = deepcopy(sample_a)
    sample_b["spec_sha256"] = "held-out"
    sample_b["readback"].update(axial_stiffness_n=960., material="unseen", weight_n=.5)
    assert a == pack_wire_grasp_feedback(sample_b, sample_b["clock"])
    assert a["position_in_palm_m"][0] == [0., .012, -.016]
    assert a["reaction_valid"][0][:3] == [True, True, False]
    np.testing.assert_allclose(a["reaction_force_n"][0][0], [0., .2, 0., 0., -.1])
    assert a["reaction_force_n"][0][1] == [0.]*5
    sample_b["measured_hands"]["left"]["xyz_in_palm_m"][1] += .002
    sample_b["measured_hands"]["right"]["reaction"]["finger14_link"]["mean_normal_n"] *= 2
    b = pack_wire_grasp_feedback(sample_b, sample_b["clock"])
    assert b["position_in_palm_m"][0][1] == pytest.approx(.014)
    assert b["reaction_force_n"][1][0][1] == pytest.approx(.4)


def test_grasp_feedback_rejects_stale_or_cross_reset_sources_and_masks_invalid_reaction():
    data = sample()
    for key in ("physics_step", "rod_tick", "reset_id"):
        clock = dict(data["clock"], **{key: data["clock"][key] + 1})
        if key == "physics_step":
            clock["timestamp_s"] = clock["physics_step"]/480
        with pytest.raises(ValueError, match="same completed"):
            pack_wire_grasp_feedback(data, clock)
    data["reaction_valid"] = False
    a = pack_wire_grasp_feedback(data, data["clock"])
    assert not np.any(a["reaction_valid"])
    assert not np.any(a["reaction_force_n"])
    assert a["position_in_palm_m"][0] == [0., .012, -.016]


def test_grasp_feedback_rejects_nonfinite_trusted_reaction():
    data = sample()
    data["measured_hands"]["left"]["reaction"]["finger14_link"]["mean_normal_n"] = float("nan")
    with pytest.raises(ValueError, match="finite measured"):
        pack_wire_grasp_feedback(data, data["clock"])

def test_distal_and_power_sources_use_same_canonical_row():
    data = sample()
    distal = deepcopy(data)
    for hand in distal["measured_hands"].values():
        hand["reaction"] = {name: pad for name, pad in zip(
            ("thumb", "index"), hand["reaction"].values(), strict=True)}
    expected = pack_wire_grasp_feedback(data, data["clock"])
    assert pack_wire_grasp_feedback(distal, distal["clock"]) == expected
    distal["measured_hands"]["left"]["reaction"]["finger14_link"] = (
        data["measured_hands"]["left"]["reaction"]["finger14_link"])
    with pytest.raises(ValueError, match="unique canonical"):
        pack_wire_grasp_feedback(distal, distal["clock"])

def test_grasp_feedback_rejects_empty_valid_hand_source():
    data = sample()
    data["measured_hands"]["left"]["reaction"] = {}
    with pytest.raises(ValueError, match="missing measured reaction"):
        pack_wire_grasp_feedback(data, data["clock"])



def _control_window():
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES

    end = dict(sample()["clock"], policy_step=1)
    rows = []
    for i in range(8):
        sdk = 3 + i
        rod = dict(end, physics_step=sdk, timestamp_s=sdk/480,
                   rod_tick=12*(sdk-2), solver_time_s=(sdk-2)/480,
                   mesh_names=list(PROXIMAL_POWER_MESH_NAMES), contact_valid=True)
        del rod["policy_step"]
        rod["node_positions"] = np.zeros((1, 41, 3))
        rod["node_positions"][0, :, 1] = np.linspace(-.20, .20, 41)
        poses = {side+".base_link": (np.zeros(3), np.array([0., 0., 0., 1.]))
                 for side in ("left", "right")}
        rows.append(dict(physics_step=sdk, substep=i, policy_step=0, rod=rod,
                         sdk_hand_geometry=dict(physics_step=sdk, poses=poses)))
    rows[-1]["sdk_hand_geometry"]["poses"]["left.base_link"] = (
        np.array([0., .13, 0.]), np.array([0., 0., 2**-.5, 2**-.5]))
    return rows, end


def test_control_feedback_aggregates_all_steps_then_expresses_force_in_final_palm(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_wire_grasp_feedback as feedback
    from go2_pvcnn.control.dh116.tabletop_wire_grasp_feedback import pack_wire_control_feedback

    def trusted_pairs(rod):
        n = float(rod["physics_step"])
        def hand():
            return {"per_substep": [{"links": {
                "finger14_link": dict(normal_load_n=n+s, force_on_rod_world_n=[n, 0., 0.]),
                "base_link": dict(normal_load_n=0., force_on_rod_world_n=[0., 0., 0.]),
            }} for s in range(12)]}
        return dict(hands={"left": hand(), "right": hand()}, wrong_side_contacts=[])

    monkeypatch.setattr(feedback, "measure_power_contact", trusted_pairs)
    rows, clock = _control_window()
    data = pack_wire_control_feedback(rows, clock, grasp="power", grasp_material_ids={"left": 35, "right": 5})
    assert set(data) == {"position_in_palm_m", "tangent_in_palm", "reaction_valid", "reaction_force_n"}
    np.testing.assert_allclose(data["position_in_palm_m"][0], [.02, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(data["reaction_force_n"][0][0], [3., 12., 0., 6.5, 0.], atol=1e-12)
    np.testing.assert_allclose(data["reaction_force_n"][1][0], [3., 12., -6.5, 0., 0.])
    assert data["reaction_valid"][0] == [True, False, False, False, False, True]
    assert data["reaction_force_n"][0][5] == [0.] * 5


def test_control_feedback_requires_complete_order_and_masks_invalid_contact(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_wire_grasp_feedback as feedback
    from go2_pvcnn.control.dh116.tabletop_wire_grasp_feedback import pack_wire_control_feedback

    rows, clock = _control_window()
    def pack(items, when=clock):
        return pack_wire_control_feedback(items, when, grasp="power",
                                          grasp_material_ids={"left": 35, "right": 5})

    with pytest.raises(ValueError, match="complete"):
        pack(rows[1:])
    swapped = list(rows)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    with pytest.raises(ValueError, match="consecutive"):
        pack(swapped)
    wrong_epoch = deepcopy(rows)
    wrong_epoch[2]["rod"]["reset_id"] = 2
    with pytest.raises(ValueError, match="consecutive"):
        pack(wrong_epoch)
    with pytest.raises(ValueError, match="decision boundary"):
        pack(rows, dict(clock, physics_step=11, timestamp_s=11/480))
    unkeyed = dict(clock)
    del unkeyed["policy_step"]
    with pytest.raises(ValueError, match="decision policy index"):
        pack(rows, unkeyed)
    wrong_control = deepcopy(rows)
    for row in wrong_control:
        row["policy_step"] = False
    with pytest.raises(ValueError, match="control policy index"):
        pack(wrong_control)
    missing_supports = deepcopy(rows)
    for row in missing_supports:
        row["rod"]["mesh_names"] = row["rod"]["mesh_names"][:13]
    with pytest.raises(ValueError, match="supported contact layout"):
        pack(missing_supports)
    rows[0]["rod"]["contact_valid"] = False
    def invalid_source(rod):
        pytest.fail("a declared missing contact source should not be read")
    monkeypatch.setattr(feedback, "measure_power_contact", invalid_source)
    unmeasured = pack(rows)
    assert not np.any(unmeasured["reaction_valid"])
    assert unmeasured["position_in_palm_m"][0][0] == pytest.approx(.02)



def test_control_feedback_replay_rejects_rows_changed_after_native_exit(tmp_path, monkeypatch):
    import hashlib
    import json

    from scripts import replay_dh116_wire_control_feedback as replay
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned

    rows, clock = _control_window()
    native = tmp_path / "native"
    native.mkdir()
    hashes = {}
    for row in rows:
        p = native / f'power_control_{row["physics_step"]}.json'
        p.write_text(json.dumps(encode_owned(row)))
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = tmp_path / "outer.json"
    receipt.write_text(json.dumps(dict(exit_code=0, debugger_inferior_exit_code=0, artifact_sha256=hashes)))
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    monkeypatch.setattr(replay, "pack_wire_control_feedback", lambda exchanges, end, **kwargs: {
        "first": exchanges[0]["physics_step"], "last": end["physics_step"]})
    args = dict(grasp="power", sdk_end=10, grasp_material_ids={"left": 35, "right": 5})
    result = replay.replay_control_window(receipt, receipt_sha, **args)
    assert result["observation"] == {"first": 3, "last": clock["physics_step"]}
    assert result["teacher_admission"] is False
    (native / "power_control_3.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        replay.replay_control_window(receipt, receipt_sha, **args)
