"""CPU numerical parity; synthetic inputs are never native grasp evidence."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.verify_dh116_power_shared_rod import check_supported_shape
from scripts.screen_dh116_tripod_supported_reset import archived_tube
from scripts.audit_dh116_certified_segment_distance import ENVELOPE_REFINEMENT_VERSION
from go2_pvcnn.control.dh116.tabletop_batch_geometry import rotated_points


def subject(shapes, *, origins=None):
    from go2_pvcnn.control.dh116.tabletop_tensor_tripod_guard import TensorTripodSupportGuard
    if origins is None:
        origins = torch.zeros(3, 3, dtype=torch.float64)
    return TensorTripodSupportGuard(shapes, [
        dict(name="support_0", center_m=[2, 0, .6], size_m=[.02, .02, .02]),
        dict(name="support_1", center_m=[-2, 0, .6], size_m=[.02, .02, .02])],
        env_origins=origins, device="cpu")


def cube():
    return np.array([[x, y, z] for x in (-.001, .001)
                     for y in (-.001, .001) for z in (-.001, .001)])


def inputs(n=3):
    poses = torch.zeros(n, 2, 23, 7, dtype=torch.float64)
    poses[..., 6] = 1
    poses[..., 2] = .6
    poses[:, 0, :, 0], poses[:, 1, :, 0] = -1, 1
    nodes = torch.zeros(n, 41, 3, dtype=torch.float64)
    nodes[..., 1] = torch.linspace(-.02, .02, 41)
    nodes[..., 2] = .6
    return poses, nodes


def oracle(owner, shapes, poses, nodes):
    failed = []
    for lane in range(len(poses)):
        boxes = [(box["name"], np.array(box["center_m"])+owner.origins[lane].numpy()-np.array(box["size_m"])/2,
                  np.array(box["center_m"])+owner.origins[lane].numpy()+np.array(box["size_m"])/2)
                 for box in owner.supports]
        row = []
        for side, link, key, vertices in shapes:
            pose = poses[lane, side, link].numpy()
            world = rotated_points(pose[[0, 1, 2, 6, 3, 4, 5]], vertices)
            target = key.split(":", 1)[1] in ("finger14_link", "finger23_link", "finger33_link")
            result = check_supported_shape(world, boxes, None if target else archived_tube(nodes[lane:lane+1].numpy()),
                contact_shape=target, contact_penetration_m=.00015 if target else None,
                reference_nodes=nodes[lane:lane+1].numpy(),
                envelope_refinement=ENVELOPE_REFINEMENT_VERSION if target else None)
            row.append(result is not None)
        failed.append(row)
    return failed


def test_raw_convex_151um_penetration_and_non_target_250um_match_original():
    # SDK190's historical -0.151mm failure motivates this synthetic regression;
    # this is not the historical SDK frame or a new native measurement.
    shapes = [(0, 0, "source:finger33_link", cube()),
              (0, 1, "cooked:finger33_link", cube()),
              (1, 0, "source:finger43_link", cube())]
    owner = subject(shapes)
    poses, nodes = inputs()
    poses[:, 0, :2, 0] = torch.tensor([.005849, .005851, .02])[:, None]
    poses[:, 1, 0, 0] = torch.tensor([.006249, .006251, .02])
    result = owner.capture(poses, nodes, frame=("fixture", 190))
    assert result.failures.tolist() == oracle(owner, shapes, poses, nodes)
    assert result.failures.tolist() == [[True, True, True], [False, False, False], [False, False, False]]
    assert result.passed.tolist() == [False, True, True]
    assert result.fallback_rows.tolist() == [True, True, False]
    assert result.details[0, 0]["rod_gap_lower_m"] < -.00015
    assert not result.native_qualification


def test_support_sat_fallback_preserves_raw_quaternion_and_environment_origin():
    shapes = [(0, 0, "source:finger33_link", cube()), (1, 0, "source:finger43_link", cube())]
    origins = torch.tensor([[0., 0, 0], [5., 1, 0], [-3., 2, 0]], dtype=torch.float64)
    owner = subject(shapes, origins=origins)
    poses, nodes = inputs()
    poses[..., :3] += origins[:, None, None]
    nodes += origins[:, None]
    poses[1, 0, 0, :3] = torch.tensor([7., 1, .6])
    # Within native source norm tolerance, deliberately not normalized.
    poses[2, 0, 0, 3:] = torch.tensor([0., 0., .7071074, .7071074])
    poses[2, 0, 0, :3] = origins[2]+torch.tensor([2.011000002, 0., .6], dtype=torch.float64)
    result = owner.capture(poses, nodes, frame=1)
    assert result.failures.tolist() == oracle(owner, shapes, poses, nodes)
    assert result.failures.tolist() == [[False, False], [True, False], [True, False]]
    normalized = poses.clone()
    normalized[..., 3:] /= normalized[..., 3:].norm(dim=-1, keepdim=True)
    assert not oracle(owner, shapes, normalized, nodes)[2][0]
    assert "support_clearance" in result.details[1, 0]["conditions"]
    assert owner.require_ready(frame=1) is result
    with pytest.raises(ValueError, match="frame"):
        owner.require_ready(frame=2)
    with pytest.raises(ValueError):
        owner.capture(poses, nodes * torch.nan, frame=2)
    with pytest.raises(ValueError, match="frame"):
        owner.require_ready(frame=1)


def test_clear_ordinary_rows_stay_on_device(monkeypatch):
    shapes = [(0, 0, "source:finger33_link", cube()), (1, 0, "source:finger43_link", cube())]
    owner = subject(shapes)
    poses, nodes = inputs()
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary support guard copied rows to host")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        result = owner.capture(poses, nodes, frame=1)
    assert result.passed.all() and not result.fallback_rows.any()


def route_fixture():
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES, contact_links
    links = ("finger14_link", "finger23_link", "finger33_link", "finger43_link", "finger53_link", "base_link", "finger52_link")
    names = (*links, *("unused"+str(i) for i in range(16)))
    bindings = {side+"."+name: dict(side=side, body_name=name, body_index=i,
        prim_path="/env/"+side+"Hand/"+name, source_usd_sha256="a"*64,
        cooked_geometry_sha256="b"*64, sdf_sha256="c"*64)
        for side in ("left", "right") for i, name in enumerate(links)}
    n = 2
    p, nodes = inputs(n)
    com = p.clone()
    com[..., :3] += torch.tensor([.001, .002, -.003])
    wrench = torch.arange(n*17*6, dtype=torch.float64).reshape(n, 17, 6)*.001
    # Synthetic pair rows obey the same native reductions, with distinct table
    # and both support totals; no empty contact set invents a nonzero wrench.
    env_ids = torch.arange(n).repeat_interleave(17)
    indices = torch.zeros(n*17, 3, dtype=torch.int64)
    indices[:, 2] = torch.arange(17).repeat(n)
    vectors = torch.zeros(n*17, 12, dtype=torch.float64)
    vectors[:, 6:9] = -12*wrench[..., :3].reshape(-1, 3)
    vectors[:, 3:6] = vectors[:, 6:9]/vectors[:, 6:9].norm(dim=-1, keepdim=True)
    vectors[:, 9:12] = 12*wrench[..., 3:].reshape(-1, 3)
    nodes_force = torch.zeros(n, 41, 3, dtype=torch.float64)
    nodes_force[:, 0] = -wrench[..., :3].sum(1)
    packet = dict(mesh_wrenches=wrench, node_forces=nodes_force,
        pair_lane_meta=torch.tensor([[0, 0, 8364, 17]]*n, dtype=torch.int32),
        counts=torch.full((n,), 17, dtype=torch.int64), env_ids=env_ids,
        indices=indices, vectors=vectors, pair_owner_time_s=1/480)
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_source import SharedSourceIdentity
    pre = SharedSourceIdentity(parent_solver_id=123, parent_num_envs=n, env_rows=(0, 1),
        run_uuid="cpu_fixture", episode_uids=("a", "b"), rod_uids=("r0", "r1"), reset_ids=(1, 1),
        rod_ticks=(0, 0), sdk_step=1, sdk_time_s=1/480, sdk_dt_s=1/480, sdk_timer_dt_s=1/480, policy_time_s=1/480,
        solver_time_s=0., profile_sha256="d"*64, scope="cpu_fixture")
    post = replace(pre, sdk_step=2, rod_ticks=(12, 12), solver_time_s=1/480, sdk_time_s=2/480, policy_time_s=2/480)
    trans = dict(stamp=pre, phase="finished")
    return bindings, (names, names), PROXIMAL_POWER_MESH_NAMES, dict(link_pose_xyzw=p, body_com_pose_w=com), packet, trans, post


def test_seven_world_com_routes_and_external_supports_match_original_and_reject_missing_source():
    from go2_pvcnn.control.dh116.tabletop_tensor_tripod_guard import TensorTripodRoutes
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import route_world_wrenches
    bindings, names, meshes, state, packet, trans, stamp = route_fixture()
    owner = TensorTripodRoutes(bindings, names, meshes, device="cpu")
    result = owner.validate_substep(state, trans, packet, stamp)
    for lane in range(2):
        poses = {key: (state["link_pose_xyzw"][lane, (0 if row["side"]=="left" else 1), row["body_index"], :3].numpy(),
                       state["link_pose_xyzw"][lane, (0 if row["side"]=="left" else 1), row["body_index"], 3:].numpy())
                 for key, row in bindings.items()}
        centers = {key: state["body_com_pose_w"][lane, (0 if row["side"]=="left" else 1), row["body_index"], :3].numpy()
                   for key, row in bindings.items()}
        named = {key: packet["mesh_wrenches"][lane, meshes.index(key)].numpy() for key in (*bindings, "table")}
        expected, _ = route_world_wrenches(named, bindings, poses, centers)
        for side, label in enumerate(("left", "right")):
            np.testing.assert_allclose(result.forces[lane, side], expected[label]["forces"][0], atol=1e-15, rtol=0)
            np.testing.assert_allclose(result.torques[lane, side], expected[label]["torques"][0], atol=1e-15, rtol=0)
    for field in ("pair_lane_meta", "pair_owner_time_s"):
        bad = dict(packet); bad.pop(field)
        with pytest.raises(ValueError, match="source|allocation|clock"):
            owner.validate_substep(state, trans, bad, stamp)
    packet["pair_lane_meta"][0, 2] -= 1
    with pytest.raises(ValueError, match="allocation"):
        owner.validate_substep(state, trans, packet, stamp)
    packet["pair_lane_meta"][0, 2] += 1
    packet["mesh_wrenches"][0, 0, 0] = 30.000001
    with pytest.raises(ValueError, match="30 N"):
        owner.validate_substep(state, trans, packet, stamp)


def test_route_source_rejects_an_untyped_identity_even_with_matching_values():
    from go2_pvcnn.control.dh116.tabletop_tensor_tripod_guard import TensorTripodRoutes
    from dataclasses import asdict
    bindings, names, meshes, state, packet, trans, stamp = route_fixture()
    owner = TensorTripodRoutes(bindings, names, meshes, device="cpu")
    unowned = SimpleNamespace(**asdict(stamp))
    with pytest.raises(ValueError, match="source"):
        owner.validate_substep(state, trans, packet, unowned)
