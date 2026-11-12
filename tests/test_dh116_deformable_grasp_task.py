import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import (
    R0_TERMS, FlexibleGraspEpisode, load_flexible_grasp_task,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json"


def test_load_locks_original_action_state_backend_and_three_grasps(tmp_path):
    task = load_flexible_grasp_task(CONFIG, ROOT)
    assert task["contracts"]["action"]["size"] == 24
    assert task["contracts"]["state"]["size"] == 371
    assert task["scene"]["table_top_m"] == pytest.approx(.30)
    assert task["rod"]["pinned_node_ids"] == []
    assert task["rod"]["length_m"] is None
    assert task["rod"]["gravity_world_m_s2"] is None
    assert task["native_qualification"] is False
    assert "rod.gravity_world_m_s2" in task["native_blockers"]
    assert "scene.self_collision_enabled" in task["native_blockers"]
    assert "reward.coefficients" in task["native_blockers"]
    assert not task["reward"]["r2_enabled"]
    assert {name: len(profile["required_pads"]["left"]) for name, profile in task["grasps"].items()} == {
        "pinch": 2, "tripod": 3, "power": 5,
    }
    for candidate, error in (
        ({**task["rod"], "pinned_node_ids": [0, 20]}, "free rod"),
        ({**task["rod"], "gravity_world_m_s2": [0, 0, 0]}, "downward gravity"),
        ({**task["rod"], "radius_m": .005}, "unconfirmed"),
    ):
        conf = json.loads(CONFIG.read_text())
        conf["rod"] = candidate
        path = tmp_path / "config.json"
        path.write_text(json.dumps(conf))
        with pytest.raises(ValueError, match=error):
            load_flexible_grasp_task(path, ROOT)


def test_loader_rejects_changed_grasp_and_source_digest(tmp_path):
    config = json.loads(CONFIG.read_text())
    path = tmp_path / "config.json"
    config["grasps"]["tripod"]["required_pads"]["right"].pop()
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="pad"):
        load_flexible_grasp_task(path, ROOT)
    config = json.loads(CONFIG.read_text())
    config["sources"]["action"]["sha256"] = "0" * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="SHA256"):
        load_flexible_grasp_task(path, ROOT)


@pytest.fixture
def task():
    return load_flexible_grasp_task(CONFIG, ROOT)


def measurement(value, source="cpu_fixture"):
    return {"value": value, "source": source}


def snapshot(task, step, *, uid="ep-0", reset_id=0, dz=0., bend=False, rod_uid="rod0"):
    y = np.linspace(0, 1, 5)
    points = np.stack((np.zeros(5), y, np.full(5, .31 + dz)), axis=1)
    if bend:
        points[:, 0] = [0, .02, .03, .02, 0]
    return {"rod_uid": rod_uid, "rod_tick": step * 12, "rod_snapshot": {
        "task_kind": "deformable_rod", "version": "dh116-deformable-rod-readback-v1",
        "coordinate_frame": "world", "source_scope": "cpu_fixture",
        "backend_sha256": task["backend"]["extension_sha256"],
        "env_ids": [0], "episode_uids": [uid], "reset_ids": [reset_id],
        "physics_step": step, "timestamp_s": step / 480,
        "node_positions": points[None], "rest_lengths_m": np.full((1, 4), .25),
    }}


def signals(task, *, step=1, uid="ep-0", reset_id=0, shift=0., loss=False, support=False, include_external=True):
    values = dict(radius_m=measurement(.005), approach_distance_m=measurement([.01, .01]),
                  table_contact=measurement(False), joint_limit=measurement(False),
                  stable=measurement(True))
    values["sample_identity"] = dict(env_id=0, episode_uid=uid, reset_id=reset_id, rod_uid="rod0",
                                     physics_step=step, rod_tick=step * 12, timestamp_s=step / 480)
    if include_external:
        values.update(external_support=measurement(support), authored_force=measurement(False),
                      attachment=measurement(False))
    for side, station in (("left", .25 + shift), ("right", .75)):
        values[side] = measurement({"rod_uid": "rod0",
            "pads": [] if loss and side == "right" else task["grasps"]["pinch"]["required_pads"][side],
            "opposed": not (loss and side == "right"), "material_s_m": station,
            "material_uncertainty_m": .00001, "tangential_slip_m": 0.,
            "tangential_uncertainty_m": .00001})
    return values


def begin(task, *, diagnostic_coefficients=None, **changes):
    episode = FlexibleGraspEpisode(task, "pinch", env_id=0, diagnostic_coefficients=diagnostic_coefficients)
    state = {"rod": True, "joint": True, "controller": True, "filter": True,
             "contact": True, "camera_history": True}
    def read():
        assert not any(state.values())
        return snapshot(task, 0, **changes), measurement(.005)
    reset = {name: lambda name=name: state.__setitem__(name, False) for name in state}
    episode.reset(reset, read)
    return episode, reset


def test_step_uses_fixed_material_station_not_world_bend_and_rejects_double_count(task):
    episode, _ = begin(task)
    first = episode.step(snapshot(task, 1, dz=.035), signals(task))
    assert first["hold_steps"] == 0 and first["phase"] == "hold"
    assert first["body_net_lift_m"] == pytest.approx(.035)
    assert first["terms"]["material_anchor_slip"]["raw"] == pytest.approx(0)
    bent = episode.step(snapshot(task, 2, dz=.035, bend=True), signals(task, step=2))
    assert bent["hold_steps"] == 1
    assert bent["terms"]["material_anchor_slip"]["raw"] == pytest.approx(0)
    with pytest.raises(ValueError, match="clock"):
        episode.step(snapshot(task, 2, dz=.035), signals(task, step=2))
    slipped = episode.step(snapshot(task, 3, dz=.035), signals(task, step=3, shift=.0006))
    assert slipped["hold_steps"] == 0
    assert slipped["terms"]["material_anchor_slip"]["raw"] == pytest.approx(.0006)
    assert slipped["training_reward"] is None
    assert slipped["native_qualification"] is False


def test_hold_is_continuous_and_missing_measurements_never_become_zero_reward(task):
    episode, _ = begin(task)
    before = episode.step(snapshot(task, 1, dz=.035), signals(task))
    assert before["hold_steps"] == 0 and before["phase"] == "hold"
    unknown = episode.step(snapshot(task, 2, dz=.035), signals(task, step=2, include_external=False))
    assert unknown["hold_steps"] == 0
    assert unknown["terms"]["external_support_violation"]["raw"] is None
    assert unknown["terms"]["external_support_violation"]["valid"] is False
    assert unknown["training_reward"] is None
    assert unknown["term_stats"]["external_support_violation"]["validity_count"] == 1
    for step in range(3, 2403):
        last = episode.step(snapshot(task, step, dz=.035), signals(task, step=step))
    assert last["hold_steps"] == 2399 and not last["success_candidate"]
    last = episode.step(snapshot(task, 2403, dz=.035), signals(task, step=2403))
    assert last["hold_steps"] == 2400 and last["success_candidate"]
    assert last["native_qualification"] is False


def test_reset_clears_old_identity_anchor_and_reward_stats(task):
    episode, reset = begin(task)
    first = episode.step(snapshot(task, 1, dz=.035), signals(task))
    assert first["hold_steps"] == 0 and first["phase"] == "hold"
    episode.reset(reset, lambda: (snapshot(task, 0, uid="ep-1", reset_id=1), measurement(.005)))
    with pytest.raises(ValueError, match="episode"):
        episode.step(snapshot(task, 1, dz=.035), signals(task))
    data = episode.step(snapshot(task, 1, uid="ep-1", reset_id=1, dz=.035), signals(task, uid="ep-1", reset_id=1, shift=.002))
    assert data["hold_steps"] == 0 and data["phase"] == "hold"
    assert data["terms"]["material_anchor_slip"]["raw"] == pytest.approx(0)
    assert data["term_stats"]["required_contact"]["validity_count"] == 1
    assert data["terms"]["approach_progress"]["valid"] is False


def test_bilateral_contacts_must_use_one_rod_and_distinct_material_coordinates(task):
    episode, _ = begin(task)
    changed = signals(task)
    changed["right"]["value"]["rod_uid"] = "other"
    with pytest.raises(ValueError, match="same rod"):
        episode.step(snapshot(task, 1, dz=.035), changed)
    same = signals(task)
    same["right"]["value"]["material_s_m"] = .25
    output = episode.step(snapshot(task, 1, dz=.035), same)
    assert output["hold_steps"] == 0
    assert output["terms"]["material_anchor_slip"]["raw"] is None
    assert output["terms"]["material_anchor_slip"]["valid"] is False


def test_cpu_diagnostic_r0_uses_signed_terms_once_and_never_returns_training_reward(task):
    episode, _ = begin(task, diagnostic_coefficients={name: 1. for name in R0_TERMS})
    episode.step(snapshot(task, 1, dz=.035), signals(task))
    step = episode.step(snapshot(task, 2, dz=.035), signals(task, step=2, shift=.0006, support=True))
    assert step["terms"]["external_support_violation"]["contribution"] == -1.
    assert step["terms"]["material_anchor_slip"]["contribution"] == pytest.approx(-.0006)
    assert step["diagnostic_reward"] == pytest.approx(sum(row["contribution"] for row in step["terms"].values()))
    assert step["training_reward"] is None


def test_rejects_stale_measurements_after_reset_even_when_rod_uid_is_reused(task):
    episode, hooks = begin(task)
    episode.reset(hooks, lambda: (snapshot(task, 0, uid="ep-1", reset_id=1), measurement(.005)))
    next_rod = snapshot(task, 1, uid="ep-1", reset_id=1, dz=.035)
    with pytest.raises(ValueError, match="measurement identity"):
        episode.step(next_rod, signals(task))
    updated = episode.step(next_rod, signals(task, uid="ep-1", reset_id=1))
    assert updated["hold_steps"] == 0 and updated["phase"] == "hold"


def test_slip_with_unresolvable_precision_is_masked(task):
    episode, _ = begin(task)
    episode.step(snapshot(task, 1, dz=.035), signals(task))
    measured = signals(task, step=2)
    measured["left"]["value"]["material_uncertainty_m"] = .000495
    data = episode.step(snapshot(task, 2, dz=.035), measured)
    assert not data["terms"]["material_anchor_slip"]["valid"]
    assert data["material_anchor_slip_m"] is None
    assert data["hold_steps"] == 0 and not data["hold_valid"]


def test_authored_force_or_attachment_breaks_hold_without_implying_table_contact(task):
    episode, _ = begin(task)
    prior = episode.step(snapshot(task, 1, dz=.035), signals(task))
    assert prior["hold_steps"] == 0 and prior["phase"] == "hold"
    applied = signals(task, step=2)
    applied["authored_force"] = measurement(True, "cpu_authored_force_audit")
    result = episode.step(snapshot(task, 2, dz=.035), applied)
    assert result["hold_steps"] == 0
    assert result["terms"]["external_support_violation"]["raw"] == 1.
    assert "cpu_authored_force_audit" in result["terms"]["external_support_violation"]["source"]
    attached = signals(task, step=3)
    attached["attachment"] = measurement(True)
    result = episode.step(snapshot(task, 3, dz=.035), attached)
    assert result["hold_steps"] == 0
    assert result["terms"]["external_support_violation"]["raw"] == 1.


def test_original_hand_material_assumptions_are_locked_and_not_rod_material(task):
    source = task["scene"]["original_hand_material_contract"]
    assert source["static_friction"] == 1. and source["dynamic_friction"] == .8
    assert source["pad_compliance"]["stiffness_n_m"] == 10000.
    assert source["pad_compliance"]["damping_n_s_m"] == 30.
    assert source["calibration"] == "training_assumption_not_measured"
    assert task["rod"]["friction_static"] is None and task["rod"]["friction_dynamic"] is None


def test_median_lift_cannot_hide_a_rod_end_at_its_initial_height(task):
    episode, _ = begin(task)
    rod = snapshot(task, 1, dz=.04)
    rod["rod_snapshot"]["node_positions"][0, 0, 2] = .31
    result = episode.step(rod, signals(task))
    assert result["body_net_lift_m"] == pytest.approx(.04)
    assert result["min_node_net_lift_m"] == 0.
    assert result["phase"] == "lift" and result["hold_steps"] == 0
    assert result["terms"]["lift_progress"]["raw"] == 0.
