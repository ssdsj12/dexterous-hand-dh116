"""CPU setup fixtures exercise source contracts and real regex row remapping."""
from copy import deepcopy
import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from test_dh116_regex_articulation import Hand
from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state
from go2_pvcnn.control.dh116.tabletop_regex_articulation import RegexArticulationBatch
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import MESH_NAMES, PADS
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES

MODULE = "go2_pvcnn.control.dh116.tabletop_regex_task_setup"


def implementation():
    assert importlib.util.find_spec(MODULE) is not None, "reusable regex task setup is missing"
    return importlib.import_module(MODULE)


def source_seed():
    a = 2 ** -.5
    return prepare_reset_state(hand_roots={s: [.5, y, .6, a, 0., 0., a]
        for s, y in (("left", .1), ("right", -.1))}, origin=[0., 0., 0.],
        root_xy=[[0., 0.], [0., 0.]], initial_offset_m=.04,
        joint_positions=[expand(np.full(6, .4)).tolist()] * 2,
        joint_velocities=[np.zeros(11).tolist()] * 2, source_description="CPU source-bound fixture")


@pytest.fixture
def setup_sources(monkeypatch):
    impl = None
    from go2_pvcnn.control.dh116 import tabletop_regex_articulation as regex
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as pinch
    assets, supports, support_material, task = [ModuleType("go2_pvcnn.control.dh116." + name) for name in (
        "tabletop_dual_hand_scene_preparation", "tabletop_end_supported_scene",
        "tabletop_deformx_end_supported_scene", "tabletop_power_supported_task")]
    assets.BUILDER_OFFSET_M = .04
    task.POWER_SCOPE, task.TRIPOD_SCOPE = "power_scope", "tripod_scope"
    for module in (assets, supports, support_material, task):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    seed = source_seed()
    nodes = np.column_stack((np.linspace(0, 1, 41), np.zeros(41), np.full(41, .7)))
    origins = np.asarray([[0., 0., 0.], [3., 0., 0.]])
    bodies = ["base_link", *PADS, "finger52_link"]
    bodies += [f"extra_{i}" for i in range(23 - len(bodies))]
    hands = [Hand((1, 0), "left"), Hand((0, 1), "right")]
    for hand in hands:
        hand.body_names = bodies.copy()
        hand.root_physx_view.link_paths = [
            [p.rsplit("/", 1)[0] + "/" + name for name in bodies]
            for p in hand.root_physx_view.prim_paths]
        com = torch.zeros(2, 23, 7)
        com[..., 0] = .1
        com[..., 6] = 1
        hand.root_physx_view.get_coms = lambda com=com: com
    events = []
    sim = SimpleNamespace(stage=object(), current_time_step_index=0)
    def refresh():
        events.append("refresh")
        for hand in hands:
            hand.root_physx_view.links.copy_(hand.root_physx_view.root[:, None].expand(-1, 23, -1))
    sim.physics_sim_view = SimpleNamespace(update_articulations_kinematic=refresh)
    def reset():
        events.append("sim.reset")
        sim.current_time_step_index += 2
    sim.reset = reset
    owners = SimpleNamespace(resources={})
    owners.keep = lambda name, value: owners.resources.setdefault(name, value)
    watched = []
    trace = SimpleNamespace(watch=lambda name, value, **kwargs: watched.append(name))
    args = object()
    layout = {"hand_roots": {s: np.array(seed["hand_roots"][i]) for i, s in enumerate(("left", "right"))},
              "active_joints": {s: np.full(6, .4) for s in ("left", "right")}}
    prepared = {"table": {"center_m": [0., 0., .2], "size_m": [1., 1., .2]},
        "candidate_physics": {"hand": {"active_effort_limit_nm": .2}, "table": {"mu_s": .8, "mu_k": .6}}}
    monkeypatch.setattr(assets, "spawn_parameters", lambda project, passed: (args, deepcopy(layout), {"left": "left.usd", "right": "right.usd"}), raising=False)
    monkeypatch.setattr(assets, "_bind_materials", lambda stage, scoped: events.append(("material", scoped["prefix"], scoped["table"]["center_m"])), raising=False)
    monkeypatch.setattr(assets, "inspect_source_stage", lambda stage, scoped: {"original_colliders": 34}, raising=False)
    monkeypatch.setattr(assets, "validate_simulation_settings", lambda sim, prepared: events.append("validate_settings"), raising=False)
    monkeypatch.setattr(pinch, "_paired_sources", lambda project, pair, prepared: events.append("paired_sources"))
    support_calls = []
    monkeypatch.setattr(supports, "author_end_supports", lambda stage, prefix, support: support_calls.append((prefix, deepcopy(support))) or [prefix + "/support"], raising=False)
    monkeypatch.setattr(support_material, "_bind_support_material", lambda *args: events.append(("support_material", args[-2:])), raising=False)
    monkeypatch.setattr(task, "prepare_initial_state", lambda config, table, expected_scope: (deepcopy(seed), np.full((2, 6), .4), nodes.copy(), None), raising=False)
    fake_vec = ModuleType("go2_pvcnn.control.dh116.tabletop_vec_isaac")
    fake_vec.isolate_scenes = lambda stage, scenes: events.append("isolate")
    monkeypatch.setitem(sys.modules, fake_vec.__name__, fake_vec)
    authored_layouts = []
    def factory(sim, passed_args, passed_layout, sources, env_origins):
        assert passed_args is args and sources == {"left": "left.usd", "right": "right.usd"}
        np.testing.assert_array_equal(env_origins, origins)
        authored_layouts.append(deepcopy(passed_layout))
        scenes = [dict(prefix=f"/World/envs/env_{i}", origin=origin, hands=[None, None]) for i, origin in enumerate(origins)]
        return RegexArticulationBatch(hands, (0, 1), tuple(s["prefix"] for s in scenes), ("/wrist_fixed",) * 2, scenes)
    monkeypatch.setattr(regex, "spawn_regex_articulations", factory)

    def archive(style):
        names = MESH_NAMES if style == "pinch" else PROXIMAL_POWER_MESH_NAMES
        bindings = {name: dict(side=name.split(".")[0], body_name=name.split(".")[1],
            body_index=bodies.index(name.split(".")[1]), prim_path="/original/" + name,
            source_usd_sha256="a" * 64, cooked_geometry_sha256="b" * 64, sdf_sha256="c" * 64)
            for name in names if "." in name}
        entries = [dict(name=name, **({} if name in bindings else {
            "table_pose" if style == "pinch" else "static_pose": ([.25, .5, .3], [0., 0., 0., 1.])})) for name in names]
        return SimpleNamespace(style=style, pair={"reset_seed": deepcopy(seed), "rod_seed": {"proposed_nodes_m": nodes.tolist()}},
            mesh_plan={"bindings": bindings, "entries": entries}, task_seed_sha256=seed["sha256"],
            task=None if style == "pinch" else {"support_scene": {"supports": [{"center_m": [0., 0., .5], "size_m": [.1, .1, .1]}]}})
    return SimpleNamespace(impl=impl, sim=sim, owners=owners, trace=trace, prepared=prepared,
        archive=archive, seed=seed, origins=origins, nodes=nodes, events=events, watched=watched,
        support_calls=support_calls, authored_layouts=authored_layouts, hands=hands,
        lanes=[("/unused/lane0", origins[0]), ("/unused/lane1", origins[1])])


@pytest.mark.parametrize("style", ["pinch", "tripod", "power"])
def test_reusable_setup_preserves_original_assets_and_returns_device_state(setup_sources, style):
    f = setup_sources
    f.impl = implementation()
    archive = f.archive(style)
    saved_bindings = deepcopy(archive.mesh_plan["bindings"])
    batch, seed, nodes, bindings, initial = f.impl.prepare_regex_task(
        f.sim, f.owners, f.trace, "/project", f.prepared, archive, f.lanes)
    assert f.sim.current_time_step_index == 2 and f.events.count("sim.reset") == 1
    assert len(batch.hands) == 2 and len(f.watched) == 8
    assert f.owners.resources["regex_batch"] is batch
    assert seed == f.seed and archive.mesh_plan["bindings"] == saved_bindings
    np.testing.assert_array_equal(nodes, f.nodes)
    assert all(isinstance(value, torch.Tensor) and value.device.type == "cpu" for value in initial.values())
    assert initial["q"].shape == (2, 2, 17)
    assert initial["body_com_velocity_w"].shape == (2, 2, 23, 6)
    # Raw +90deg Z quaternion rotates the real local COM +X offset into +Y.
    torch.testing.assert_close(initial["body_com_pose_w"][..., :3] - initial["link_pose_wxyz"][..., :3],
                               torch.tensor([0., .1, 0.], dtype=torch.float64).expand(2, 2, 23, 3), atol=2e-7, rtol=0)
    assert initial["base_link_pose_w"].shape == (2, 2, 7)
    assert initial["mesh_positions_world_m"].shape == (2, len(archive.mesh_plan["entries"]), 3)
    for key, row in bindings.items():
        assert row["source_usd_sha256"] == saved_bindings[key]["source_usd_sha256"]
        assert row["prim_path"] == row["prim_paths"][0]
        for env_id in (0, 1):
            assert row["prim_paths"][env_id] == f"/World/envs/env_{env_id}/{row['side']}Hand/{row['body_name']}"
    table_slot = [entry["name"] for entry in archive.mesh_plan["entries"]].index("table")
    torch.testing.assert_close(initial["mesh_positions_world_m"][:, table_slot],
        torch.tensor([[.25, .5, .3], [3.25, .5, .3]]), atol=0, rtol=0)
    if style == "pinch":
        assert "paired_sources" in f.events and not f.support_calls
    else:
        assert len(f.support_calls) == 2
        np.testing.assert_array_equal(f.support_calls[1][1]["supports"][0]["center_m"], [3., 0., .5])
        assert f.support_calls[1][1]["supports"][0]["size_m"] == [.1, .1, .1]


def test_selected_restore_uses_original_seed_and_preserves_peer_targets_and_composers(setup_sources):
    f = setup_sources
    f.impl = implementation()
    batch, _, _, _, initial = f.impl.prepare_regex_task(f.sim, f.owners, f.trace, "/project", f.prepared, f.archive("pinch"), f.lanes)
    before = {}
    for side, owner in batch.sides.items():
        owner.hand.root_physx_view.q.add_(.07)
        owner.hand.root_physx_view.targets.add_(.08)
        owner.hand.permanent_wrench_composer.composed_force_as_torch.fill_(.3)
        owner.hand.instantaneous_wrench_composer.composed_force_as_torch.fill_(.5)
        row = int(owner.rows([0])[0])
        before[side] = tuple(t[row].clone() for t in (owner.hand.root_physx_view.q,
            owner.hand.root_physx_view.targets, owner.hand.permanent_wrench_composer.composed_force_as_torch,
            owner.hand.instantaneous_wrench_composer.composed_force_as_torch))
    f.impl.restore_regex_seed(f.sim, batch, initial, env_ids=[1], active_effort_limit_nm=.2)
    assert f.sim.current_time_step_index == 2
    for side, owner in batch.sides.items():
        row = int(owner.rows([0])[0])
        now = tuple(t[row] for t in (owner.hand.root_physx_view.q, owner.hand.root_physx_view.targets,
            owner.hand.permanent_wrench_composer.composed_force_as_torch,
            owner.hand.instantaneous_wrench_composer.composed_force_as_torch))
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(now, before[side]))
        selected = int(owner.rows([1])[0])
        torch.testing.assert_close(owner.hand.root_physx_view.q[selected], initial["seed_q"][1, 0 if side == "left" else 1], atol=0, rtol=0)
        assert owner.hand.instantaneous_wrench_composer.composed_force_as_torch[selected].count_nonzero() == 0
        assert not any(call[0] == "write" for call in owner.hand.calls)


def test_height_is_explicit_and_never_moves_rod_or_authored_assets(setup_sources):
    f = setup_sources
    f.impl = implementation()
    archive = f.archive("tripod")
    _, seed, nodes, _, initial = f.impl.prepare_regex_task(f.sim, f.owners, f.trace, "/project", f.prepared, archive, f.lanes,
        seed_height=dict(delta_z_m=[.0025, .002], evidence_sha256="a" * 64))
    assert seed["sha256"] != f.seed["sha256"]
    np.testing.assert_array_equal(nodes, f.nodes)
    np.testing.assert_allclose(np.asarray(seed["reset_root_pose_w"])[:, 2] - np.asarray(f.seed["reset_root_pose_w"])[:, 2], [.0025, .002], atol=1e-15, rtol=0)
    for index, side in enumerate(("left", "right")):
        expected = np.asarray(f.seed["authored_root_seed_w"])[index].copy()
        expected[2] -= .04
        np.testing.assert_array_equal(f.authored_layouts[0]["hand_roots"][side], expected)
    torch.testing.assert_close(initial["rod_positions_w"][1], torch.tensor(f.nodes + [3., 0., 0.], dtype=torch.float32), atol=0, rtol=0)


def test_capture_reuses_actual_local_com_cache_with_fresh_link_pose(setup_sources):
    f = setup_sources
    f.impl = implementation()
    half = 2 ** -.5
    for hand in f.hands:
        local = hand.root_physx_view.get_coms()
        local[..., 3] = half
        local[..., 6] = half
    batch, _, _, _, initial = f.impl.prepare_regex_task(f.sim, f.owners, f.trace, "/project", f.prepared, f.archive("pinch"), f.lanes)
    torch.testing.assert_close(initial["body_com_pose_w"][..., 3:], torch.full((2, 2, 23, 4), .5, dtype=torch.float64), atol=2e-7, rtol=0)
    cached = initial["body_com_pose_b_xyzw"].clone()
    for hand in f.hands:
        hand.root_physx_view.root[:, 0] += 2
        hand.root_physx_view.get_coms = lambda: pytest.fail("static COM was fetched again")
    f.sim.physics_sim_view.update_articulations_kinematic()
    fresh = f.impl.capture_regex_state(batch, body_com_pose_b_xyzw=cached)
    torch.testing.assert_close(fresh["body_com_pose_w"][..., 0], initial["body_com_pose_w"][..., 0] + 2, atol=0, rtol=0)
    torch.testing.assert_close(fresh["body_com_pose_b_xyzw"], cached, atol=0, rtol=0)

def test_world_com_uses_original_float64_raw_quaternion_arithmetic(setup_sources):
    from go2_pvcnn.control.dh116.tabletop_deformable_live_geometry import rotation_wxyz
    f=setup_sources
    impl=implementation()
    batch,_,_,_,initial=impl.prepare_regex_task(f.sim,f.owners,f.trace,"/project",f.prepared,f.archive("pinch"),f.lanes)
    for hand in f.hands:
        hand.root_physx_view.links[...,0]=1000.125
    local=initial["body_com_pose_b_xyzw"].clone()
    local[...,0]=.000023
    result=impl.capture_regex_state(batch,body_com_pose_b_xyzw=local)
    link=result["link_pose_xyzw"].numpy()
    expected=np.empty(link.shape[:-1]+(3,))
    for index in np.ndindex(link.shape[:-1]):
        p=link[index].astype(float)
        expected[index]=p[:3]+rotation_wxyz(p[[6,3,4,5]])@local[index][:3].numpy().astype(float)
    assert result["body_com_pose_w"].dtype==torch.float64
    np.testing.assert_allclose(result["body_com_pose_w"][...,:3].numpy(),expected,rtol=0,atol=2e-15)

@pytest.mark.skipif(__import__("os").environ.get("DH116_TENSOR_SETUP_CUDA")!="1",reason="explicit CUDA device test")
def test_startup_cpu_com_buffer_is_copied_once_before_cuda_row_permutation():
    from go2_pvcnn.control.dh116.tabletop_regex_task_setup import capture_regex_state
    device=torch.device("cuda:0")
    links=torch.zeros(2,23,7,device=device);links[...,6]=1
    local=torch.zeros(2,23,7);local[...,0]=.01;local[...,6]=1
    fields={key:torch.zeros(2,17,device=device) for key in ("q","qd","backend_targets")}
    fields.update({key:links[:,0].clone() for key in ("root_pose_wxyz","root_pose_xyzw")})
    fields.update(root_velocity=torch.zeros(2,6,device=device),link_pose_wxyz=links,link_pose_xyzw=links,
        link_velocity=torch.zeros(2,23,6,device=device))
    calls=[]
    def get_coms():
        calls.append("startup_static_read")
        return local
    batch=SimpleNamespace(read=lambda:{"left":fields,"right":fields},
        sides={s:SimpleNamespace(device=device,row_indices=torch.tensor([1,0],device=device),
            body_names=["base_link"]+[f"link{i}" for i in range(22)],
            hand=SimpleNamespace(root_physx_view=SimpleNamespace(get_coms=get_coms))) for s in ("left","right")})
    actual=capture_regex_state(batch)
    assert actual["body_com_pose_b_xyzw"].device==device and len(calls)==2
    capture_regex_state(batch,body_com_pose_b_xyzw=actual["body_com_pose_b_xyzw"])
    assert len(calls)==2
    torch.testing.assert_close(actual["body_com_pose_w"][...,:3],local[...,:3].to(device).double()[:,None].expand(2,2,23,3),atol=0,rtol=0)
