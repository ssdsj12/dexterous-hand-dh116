import numpy as np
import pytest

from scripts import audit_dh116_isaac_batch_scene as audit


@pytest.mark.parametrize("count", [32, 64])
def test_power_layout_names_and_spaces_original_tables(count):
    lanes = audit.scene_lanes(count, [0.6, 0.8, 0.3])
    assert len(lanes) == count
    assert len({path for path, _ in lanes}) == count
    assert lanes[0] == ("/World/DH116BatchAudit/lane_000", (0.0, 0.0, 0.0))
    for i, (_, origin) in enumerate(lanes):
        assert origin[2] == 0
        for _, other in lanes[:i]:
            assert abs(origin[0] - other[0]) >= 0.6 or abs(origin[1] - other[1]) >= 0.8


def test_one_lane_write_must_leave_all_other_native_views_unchanged():
    before = np.zeros((32, 2, 17))
    after = before.copy()
    after[31, 0, 0] = 0.001
    audit.verify_isolated_write(before, after, 31)
    with pytest.raises(ValueError, match="another lane"):
        after[0, 0, 0] = 1e-6
        audit.verify_isolated_write(before, after, 31)



def test_batch_ownership_preserves_every_witness_including_live_tail():
    from weakref import ref

    class Owner:
        pass

    dead = ref(Owner())
    tail = Owner()
    references = {str(i): dead for i in range(512)}
    references["last"] = ref(tail)
    assert hasattr(audit, "batch_snapshot"), "lifetime audit must cover the full 64-lane stage"
    summary = audit.batch_snapshot(references)
    assert set(summary["resources"]) == set(references)
    assert summary["resources"]["last"]["alive"] is True
    assert all(summary["resources"][str(i)]["alive"] is False for i in range(512))
    assert summary["native_release_verified"] is False



@pytest.mark.parametrize(("style", "count"), [("pinch", 11), ("tripod", 17), ("power", 17)])
def test_each_grasp_binds_its_original_mesh_slot_order(style, count):
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import MESH_NAMES
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES

    names = MESH_NAMES if count == 11 else PROXIMAL_POWER_MESH_NAMES
    audit.verify_mesh_layout(style, names)
    with pytest.raises(ValueError, match="mesh layout"):
        audit.verify_mesh_layout(style, tuple(reversed(names)))
    if style == "pinch":
        with pytest.raises(ValueError, match="mesh layout"):
            audit.verify_mesh_layout(style, PROXIMAL_POWER_MESH_NAMES)

@pytest.mark.parametrize(("style", "pose_key"), [
    ("pinch", "table_pose"), ("tripod", "static_pose"), ("power", "static_pose"),
])
def test_original_static_geometry_is_translated_from_its_grasp_archive(style, pose_key):
    from scripts.audit_dh116_batch_rod_exchange import translated_static_pose

    center = np.array([.2, -.3, .4], dtype=np.float32)
    quaternion = np.array([0., 0., 0., 1.], dtype=np.float32)
    translated, rotation = translated_static_pose(
        style, {pose_key: (center, quaternion)}, (2., 3., 0.))
    np.testing.assert_allclose(translated, [2.2, 2.7, .4], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(rotation, quaternion)
    np.testing.assert_array_equal(center, np.array([.2, -.3, .4], dtype=np.float32))
    wrong_key = "static_pose" if style == "pinch" else "table_pose"
    with pytest.raises(ValueError, match="static pose"):
        translated_static_pose(style, {wrong_key: (center, quaternion)}, (2., 3., 0.))


def test_startup_rebuild_uses_archived_sdf_and_rejects_stale_native_mesh():
    from scripts.audit_dh116_batch_rod_exchange import rebuild_reset_meshes

    class Rope:
        def clear_meshes(self):
            self.meshes = []

        def add_mesh(self, values, origin, spacing, *, pos, quat, mu_s, mu_k):
            self.meshes.append((values, origin, spacing, pos, quat, mu_s, mu_k))
            return len(self.meshes) - 1

        def mesh_state_snapshot(self):
            positions = np.stack([mesh[3] for mesh in self.meshes])[None]
            rotations = np.stack([mesh[4] for mesh in self.meshes])[None]
            return dict(device_state_source="cuda_device_copy",
                        mesh_count=len(self.meshes), pending_set=False, moving=False,
                        pending_teleport=np.zeros((1, len(self.meshes)), dtype=np.uint8),
                        pair_ready=self.stale, pair_capacity=24,
                        **{name+"_positions_world_m": positions for name in ("current", "target", "solver")},
                        **{name+"_quaternions_xyzw": rotations for name in ("current", "target", "solver")})

    rope = Rope()
    rope.stale = False
    grid = np.ones((2, 2, 2), dtype=np.float32)
    quaternion = np.array([0., 0., 0., 1.], dtype=np.float32)
    entry = dict(name="table", values=grid, origin=np.zeros(3), spacing=.01,
                 mu_s=.2, mu_k=.1, table_pose=(np.zeros(3), quaternion))
    rebuild_reset_meshes(rope, "pinch", [entry], {}, (2., 3., 0.), 24)
    assert rope.meshes[0][0] is grid
    np.testing.assert_array_equal(rope.meshes[0][3], [2., 3., 0.])
    assert rope.meshes[0][-2:] == (.2, .1)
    rope.stale = True
    with pytest.raises(ValueError, match="reset mesh state"):
        rebuild_reset_meshes(rope, "pinch", [entry], {}, (2., 3., 0.), 24)


def test_reset_fingerprint_reads_numeric_cuda_views(monkeypatch):
    from scripts.audit_dh116_batch_rod_exchange import reset_rod_fingerprint
    from go2_pvcnn.control.dh116 import tabletop_material_director_readback as endpoint
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_numeric_state as numeric

    class Rope:
        last_error = 0

        def parameter_snapshot(self):
            return {"time_s": 0., "contact_radius_m": .005}

    rope = Rope()
    positions = np.zeros((1, 41, 3), dtype=np.float32)
    orientations = np.zeros((1, 41, 4), dtype=np.float32)
    orientations[..., 3] = 1.
    velocities = np.zeros_like(positions)
    monkeypatch.setattr(endpoint, "_copy_endpoint", lambda owner, device: (positions.copy(), orientations.copy()))
    monkeypatch.setattr(numeric, "_copy_velocity", lambda owner, device: velocities.copy())

    before = reset_rod_fingerprint(rope, "cuda:0")
    assert reset_rod_fingerprint(rope, "cuda:0") == before
    positions[0, 0, 0] = .001
    assert reset_rod_fingerprint(rope, "cuda:0") != before
    positions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        reset_rod_fingerprint(rope, "cuda:0")
