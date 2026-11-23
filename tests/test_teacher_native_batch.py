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



def test_binary_window_owns_typed_rows_and_keeps_failure_boundary(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_batch_evidence import (
        BatchEvidence, read_batch_evidence,
    )

    evidence = BatchEvidence(tmp_path, lane_count=2)
    a, b = evidence.lane(0), evidence.lane(1)
    values = np.array([[1., float("nan")], [3., -4.]], dtype=np.float32)
    assert a.emit("physical", {"n": values, "pending": [False, None]}) == 0
    assert b.emit("physical", {"contact": np.array([], dtype=np.int64)}) == 0
    values[0, 0] = 9.
    assert not list(tmp_path.glob("*.dhb"))
    p = evidence.seal("rejected")
    frame = read_batch_evidence(p)
    assert frame["outcome"] == "rejected"
    first, second = frame["rows"]
    assert first["lane"] == 0 and second["lane"] == 1
    assert first["payload"]["n"].dtype == np.float32
    np.testing.assert_equal(first["payload"]["n"], [[1., np.nan], [3., -4.]])
    assert second["payload"]["contact"].shape == (0,)
    assert first["payload"]["pending"] == [False, None]
    assert a.emit("owner_lease_closed", {}) == 1
    p2 = evidence.seal("cleanup")
    assert p2 != p and read_batch_evidence(p2)["rows"][0]["sequence"] == 1

    damaged = tmp_path / "damaged.dhb"
    damaged.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        read_batch_evidence(damaged)


def test_empty_batch_window_does_not_write_a_file(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_batch_evidence import BatchEvidence

    evidence = BatchEvidence(tmp_path, lane_count=2)
    assert evidence.seal("cleanup") is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_tensor_readbacks_match_original_guards_and_reward(device):
    import torch
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits
    from go2_pvcnn.control.dh116.tabletop_batch_geometry import collision_clearance
    from go2_pvcnn.control.dh116.tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
    from go2_pvcnn.control.dh116.tabletop_teacher_tensor_math import (
        TensorSourceClearance, joint_force_guard, reward_window,
    )
    from go2_pvcnn.control.dh116.tabletop_training_reward import control_total

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA parity requires a GPU")
    low, high = active_limits()
    q = np.zeros((5, 2, len(JOINTS)), np.float64)
    q[:, :, [JOINTS.index(n) for n in ACTIVE]] = (low + high) / 2.
    for follower, master, ratio, offset in MIMIC:
        q[:, :, JOINTS.index(follower)] = ratio * q[:, :, JOINTS.index(master)] + offset
    q[1, 1, JOINTS.index(ACTIVE[0])] = high[0] + .021
    q[2, 0, JOINTS.index(MIMIC[0][0])] += .021
    q[3, 0, JOINTS.index(ACTIVE[0])] = np.nan
    force = np.zeros((5, 2, 5, 3))
    force[4, 0, 1, 0] = 30.01
    ids = [JOINTS.index(n) for n in ACTIVE]
    sides = ("left", "right")
    expected = [FlexiblePhysicalExchange._guard_failure(None,
        {side: dict(q=qs[i:i+1], joint_names=JOINTS, active_indices=ids)
         for i, side in enumerate(sides)},
        {side: dict(forces=fs[i:i+1]) for i, side in enumerate(sides)})
        for qs, fs in zip(q, force, strict=True)]
    # Guard code preserves original left-to-right reason precedence.
    reasons = (None, "left:nonfinite_joint_state", "left:joint_limit", "left:mimic_error",
               "right:nonfinite_joint_state", "right:joint_limit", "right:mimic_error", "excess_force")
    tensor = lambda v: torch.as_tensor(v, device=device)
    observed = joint_force_guard(tensor(q), tensor(force), JOINTS).tolist()
    assert [reasons[x] for x in observed] == expected
    poses = np.zeros((5, 2, 2, 7), np.float64)
    poses[..., 3] = 1.
    poses[:, :, :, 2] = .3021
    poses[0, 0, 1, 2] = .2
    shapes = [(0, 0, np.zeros((3, 3))), (0, 1, np.zeros((3, 3))),
              (1, 1, np.array([[.1, .2, .03], [.03, .2, .005]]))]
    expected_clearance, expected_index = collision_clearance(poses, shapes)
    clearance, index = TensorSourceClearance(shapes, device=device)(tensor(poses))
    np.testing.assert_allclose(clearance.cpu(), expected_clearance, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(index.cpu(), expected_index)
    phi = [.5, .4]
    costs = np.arange(64).reshape(8, 8)[:2] / 40
    event, end = [0., -10.], [False, True]
    actual = reward_window(tensor(phi), tensor(phi[::-1]), tensor(costs), .995,
                           tensor(event), tensor(end)).cpu().numpy()
    expected = [control_total(a, b, c, .995, d, e)
                for a,b,c,d,e in zip(phi, phi[::-1], costs.tolist(), event, end, strict=True)]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)



def test_compact_evidence_discards_success_raw_rows_and_seals_failed_window(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_batch_evidence import BatchEvidence, read_batch_evidence

    evidence = BatchEvidence(tmp_path, lane_count=2, full_windows=False)
    lane = evidence.lane(1)
    lane.emit("physical", {"tick": 3, "raw": np.ones((41, 3))})
    assert evidence.seal("committed") is None
    assert not list(tmp_path.iterdir())

    lane.emit("physical", {"tick": 4, "raw": np.zeros((41, 3))})
    frame = read_batch_evidence(evidence.seal("rejected"))
    assert len(frame["rows"]) == 1
    row, = frame["rows"]
    assert (row["lane"], row["sequence"], row["payload"]["tick"]) == (1, 1, 4)
    np.testing.assert_equal(row["payload"]["raw"], np.zeros((41, 3)))



def test_live_geometry_handoff_only_consumes_the_exact_captured_event():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_pinch_distance import LiveGeometryHandoff

    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure

    handoff = LiveGeometryHandoff()
    packet = {"source_complete": True, "sides": {}}
    row = {"closure_geometry": packet}
    handoff.capture(row)
    with pytest.raises(SourceFailure, match="same owned live geometry"):
        handoff.take(dict(row))
    assert handoff.take(row) is packet
    with pytest.raises(SourceFailure, match="same owned live geometry"):
        handoff.take(row)
    handoff.capture(row)
    row["closure_geometry"] = dict(packet)
    with pytest.raises(SourceFailure, match="same owned live geometry"):
        handoff.take(row)
