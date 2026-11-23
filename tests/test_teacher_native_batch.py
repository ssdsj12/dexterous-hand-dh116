"""CPU checks of the actual batch worker's shared physics and lane setup."""
import numpy as np
import pytest

from test_dual_hand_reset import setup


def test_lane_seed_shifts_both_world_poses_without_reauthoring_joint_targets(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_batch import translated_seed

    *_, original, _ = setup(monkeypatch)
    offset = np.array([3., 1., 0.])
    moved = translated_seed(original, offset)
    for name in ("authored_root_seed_w", "reset_root_pose_w"):
        a, b = map(np.asarray, (original[name], moved[name]))
        np.testing.assert_allclose(b[:, :3] - a[:, :3], np.tile(offset, (2, 1)))
        np.testing.assert_array_equal(a[:, 3:], b[:, 3:])
    assert moved["joint_positions"] == original["joint_positions"]
    assert moved["joint_velocities"] == original["joint_velocities"]
    assert moved["sha256"] != original["sha256"]


def test_one_sdk_tick_and_two_solver_calls_are_separate_on_partial_failure():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_batch import BatchBudget

    budget = BatchBudget([0, 1], start_sdk_step=2, max_sdk_step=18)
    budget.reset_lane(0, {"env_id": 0, "physics_step": 2, "rod_tick": 0, "reset_id": 1})
    budget.reset_lane(1, {"env_id": 1, "physics_step": 2, "rod_tick": 0, "reset_id": 1})
    budget.advanced_sdk(3)
    budget.advanced_rod({"env_id": 0, "physics_step": 3, "rod_tick": 12, "reset_id": 1})
    budget.advanced_rod({"env_id": 1, "physics_step": 3, "rod_tick": 12, "reset_id": 1})
    budget.advanced_sdk(4)
    budget.advanced_rod({"env_id": 0, "physics_step": 4, "rod_tick": 24, "reset_id": 1})
    counters = budget.snapshot()
    assert counters["measured_sdk_steps"] == 2
    assert counters["lanes"][0] == dict(solver_calls=2, rod_tick=24, resets=1, reset_id=1)
    assert counters["lanes"][1] == dict(solver_calls=1, rod_tick=12, resets=1, reset_id=1)
    with pytest.raises(ValueError, match="synchronized"):
        budget.require_window()


def test_parameter_reset_keeps_nonzero_native_lane_identity(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as native
    from go2_pvcnn.control.dh116 import tabletop_material_director_readback as director

    row = dict(env_id=3, episode_uid="owned:3:1", run_uuid="owned", rod_uid="rod:3",
               physics_step=2, rod_tick=0, reset_id=1, solver_time_s=0.,
               timestamp_s=2/480, clock_valid=True, contact_valid=False)
    nodes = np.stack((np.arange(41, dtype=np.float32) / 80, np.zeros(41), np.zeros(41)), axis=1)
    monkeypatch.setattr(native, "source_basis", dict)
    monkeypatch.setattr(director, "shared_clock", lambda f: f().copy())
    reader = native.NativeParameterEndpoints(nodes, device="cuda:0", env_id=3)
    monkeypatch.setattr(reader, "_read", lambda row, **kwargs: {
        "identity": row.copy(), "node_positions_m": reader._nodes[None],
    })
    with pytest.raises(ValueError, match="owning environment"):
        reader.capture_reset(dict(row, env_id=1), material_reference_uid="wrong", rope=None,
                             backend=None, sim=None, read_clock=lambda: row)
    packet = reader.capture_reset(row, material_reference_uid="owned", rope=None,
                                  backend=None, sim=None, read_clock=lambda: row)
    assert packet["identity"]["env_id"] == 3 and packet["boundary"] == "completed_reset_endpoint"


def test_translated_collision_binding_preserves_original_sdf_and_lane_path():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_lane_layout import verify_translated_bindings
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest

    canonical = {"left.base_link": dict(side="left", body_name="base_link", body_index=5,
                                         prim_path="/World/leftHand/base_link", sdf_sha256="a" * 64)}
    copied = {key: dict(row, prim_path="/World/lane_9/leftHand/base_link") for key, row in canonical.items()}
    assert verify_translated_bindings(canonical, copied, "/World/lane_9", _digest(canonical)) == _digest(copied)
    changed = {"left.base_link": dict(copied["left.base_link"], sdf_sha256="b" * 64)}
    with pytest.raises(ValueError, match="lane collision"):
        verify_translated_bindings(canonical, changed, "/World/lane_9", _digest(canonical))
    with pytest.raises(ValueError, match="lane collision"):
        verify_translated_bindings(canonical, copied, "/World/lane_8", _digest(canonical))
