"""CPU guards for the independent raised-rod Power native scene."""
import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116 import tabletop_power_shared_rod as power
from test_dual_hand_rod_runtime import Grid, runtime


def test_attach_both_palms_and_supports_after_original_reset_once(monkeypatch):
    original, base, bindings, poses, _ = runtime(monkeypatch, pairs=True)
    original.require_pairs = True
    original.reset(physics_step=2, episode_uid="reset-cpu:0", reset_id=1, pad_poses=poses)
    clock = original.clock()
    palms = {}
    for side in power.SIDES:
        key = f"{side}.base_link"
        palms[key] = dict(bindings[f"{side}.finger14_link"], body_name="base_link", body_index=6,
                          prim_path=f"/World/{side}Hand/base_link")
    pose = (np.zeros(3), np.array([0., 0., 0., 1.]))
    caps = []
    base.rope.enable_contact_pair_capture = lambda capacity: caps.append(capacity)
    values = dict(palm_bindings=palms, palm_grids={key:Grid() for key in palms},
                  palm_poses={key:pose for key in palms}, expected_clock=clock,
                  support_grids={key:Grid() for key in ("support_0", "support_1")},
                  support_poses={key:pose for key in ("support_0", "support_1")})
    attached = power.PowerSharedRodRuntime.from_original_reset(original, **values)
    assert attached.clock() == clock and original.clock()["clock_valid"] is False
    assert attached.rope is base.rope and base.rope.resets == 1 and base.rope.steps == 0
    assert caps == [len(base.nodes)*15*12]
    assert attached.mesh_names[11:] == ("left.base_link", "right.base_link", "support_0", "support_1")
    with pytest.raises(ValueError):
        power.PowerSharedRodRuntime.from_original_reset(original, **values)
    assert len(caps) == 1 and len(base.rope.slots) == 15


@pytest.mark.parametrize("first_motion_m", [0., 2.e-6])
def test_supported_stationary_first_step_never_sweeps_the_queued_reset(monkeypatch, first_motion_m):
    original, base, bindings, poses, _ = runtime(monkeypatch, pairs=True)
    original.require_pairs = True
    original.reset(physics_step=2, episode_uid="stationary-cpu:0", reset_id=1, pad_poses=poses)
    palms = {f"{side}.base_link": dict(bindings[f"{side}.finger14_link"],
        body_name="base_link", body_index=6, prim_path=f"/World/{side}Hand/base_link")
        for side in power.SIDES}
    identity = (np.zeros(3), np.array([0., 0., 0., 1.]))
    base.rope.enable_contact_pair_capture = lambda capacity: None
    attached = power.PowerSharedRodRuntime.from_original_reset(original,
        palm_bindings=palms, palm_grids={key:Grid() for key in palms},
        palm_poses={key:identity for key in palms}, expected_clock=original.clock(),
        support_grids={key:Grid() for key in power.END_SUPPORT_NAMES},
        support_poses={key:identity for key in power.END_SUPPORT_NAMES})
    reference = {**poses, **{key:identity for key in palms}}
    readback = dict(reference)
    key = "left.finger53_link"
    pos, quat = reference[key]
    readback[key] = (pos + [first_motion_m, 0., 0.], -quat)
    centers = {name:value[0] for name,value in readback.items()}
    indices = np.array([[step, 0, slot] for step in range(12) for slot in (0, 6)])
    vectors = np.zeros((24, 12))
    vectors[:, 5] = 1.
    vectors[:, 8] = np.tile([1., 2.], 12)
    base.rope.contact_pair_samples = lambda: dict(substeps=12, mesh_count=15,
        capacity=len(base.nodes)*15*12, attempted=24, overflow=False,
        indices=indices, vectors=vectors, node_forces_on_rod_world_n=base.rope.force[0],
        mesh_wrenches_on_mesh_world=base.rope.wrenches[0])
    base.rope.poses.clear()
    if first_motion_m:
        with pytest.raises(ValueError, match="stationary first"):
            attached.step(readback, centers, physics_step=3, stationary_reset_poses=reference)
        assert base.rope.steps == 0
        assert not base.rope.poses
        return

    attached.step(readback, centers, physics_step=3, stationary_reset_poses=reference)
    assert len(base.rope.poses) == 12 and all(p[3] for p in base.rope.poses)
    assert base.rope.steps == 1
    base.rope.poses.clear()
    attached.step(readback, centers, physics_step=4)
    assert len(base.rope.poses) == 12 and all(not p[3] for p in base.rope.poses)
    assert base.rope.steps == 2
    with pytest.raises(ValueError, match="fresh supported Reset"):
        attached.step(readback, centers, physics_step=5, stationary_reset_poses=reference)
    assert base.rope.steps == 2


def test_native_support_guard_rejects_one_micrometre_intrusion():
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import guard_hand_supports

    xyz = np.array([[x,y,z] for x in (0.01,0.02) for y in (-.002,.002) for z in (-.002,.002)])
    support = {"supports":[{"center_m":[0.,0.,0.], "size_m":[.02,.02,.02]},
                           {"center_m":[0.,.4,0.], "size_m":[.02,.02,.02]}]}
    clean = xyz + [1.e-6,0,0]
    bad = xyz - [1.e-6,0,0]
    assert guard_hand_supports({"left.source.test":clean}, support)["minimum_gap_m"] == pytest.approx(1.e-6)
    with pytest.raises(ValueError, match="left.source.test"):
        guard_hand_supports({"left.source.test":bad}, support)


def test_frozen_power_seed_keeps_free_rod_on_two_supports():
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/dh116_tasks/flexible_power_supported_contact_v1.json").read_text())
    table = {"center_m":[.65,0.,.275], "size_m":[.8,1.,.05]}
    seed, active, nodes, poses = prepare_initial_state(cfg, table)
    assert seed["joint_velocities"] == [[0.]*11]*2
    assert seed["reset_root_pose_w"] == [cfg["hands"][side]["pose_wxyz"] for side in power.SIDES]
    assert active.shape == (2,6) and nodes.shape == (41,3)
    assert all(pose[1].tolist() == [0.,0.,0.,1.] for pose in poses.values())
    cfg["fixed_rod_nodes"] = [0]
    with pytest.raises(ValueError, match="free Power rod"):
        prepare_initial_state(cfg, table)


def test_freeze_does_not_run_offline_usd_scene_preparation(tmp_path, monkeypatch):
    from argparse import Namespace
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene_api
    from scripts import verify_dh116_power_supported_task as pilot

    def fail_if_scene_prepared(*args, **kwargs):
        raise AssertionError("offline USD must not be loaded into the future AppLauncher process")

    monkeypatch.setattr(scene_api, "prepare_scene", fail_if_scene_prepared)
    monkeypatch.setattr(pilot.reset, "freeze_plan", lambda output: {"budget": {}, "frozen_files": {}})
    monkeypatch.setattr(pilot.sources, "checked_json", lambda path, sha: {"palms": {"left": {}, "right": {}}})
    monkeypatch.setattr(pilot.sources, "fixture", lambda side, rays, bake: (
        None, None, {"gpu_row_sha256": "a"*64}, None))
    paths = {"output": tmp_path / "native", "rays": tmp_path / "rays.json",
             "settle_without_hand_contact": False}
    paths["rays"].write_text("{}")
    for side in ("left", "right"):
        bake = tmp_path / f"{side}-bake"
        bake.mkdir()
        (bake / "result.json").write_text("{}")
        (bake / f"{side}.base_link.npz").write_bytes(b"test")
        (tmp_path / f"{side}_base_link.json").write_text("{}")
        paths[f"{side}_bake"] = bake
    plan, _ = pilot.freeze(Namespace(**paths))
    assert plan["budget"]["sdk_steps"] == 10 and plan["budget"]["solver_steps"] == 96
    assert str(pilot.CONFIG) in plan["frozen_files"]
    paths["settle_without_hand_contact"] = True
    settled, config = pilot.freeze(Namespace(**paths))
    original = json.loads(pilot.CONFIG.read_bytes())
    probe, parked = pilot.freeze(Namespace(**dict(paths,verify_supported_action_handoff=True)))
    assert parked == config
    assert probe["budget"]["sdk_steps"]==248 and probe["budget"]["control_steps"]==1
    assert probe["budget"]["solver_steps"]==246*12
    assert [i for i,value in enumerate(probe["action24"]) if value] == [12,15]
    from go2_pvcnn.control.dh116 import tabletop_supported_rod_reference as parked_reference

    draft = json.loads(pilot.NEAR_OPEN.read_bytes())
    parent, screen = tmp_path/"parent.json", tmp_path/"screen.json"
    parent.write_text("{}")
    screen.write_text("".join(json.dumps(dict(side=side, candidates=[candidate])) + "\n"
                              for side,candidate in draft["candidate"].items()))
    source_sha = pilot.native.sha
    monkeypatch.setattr(pilot.native, "sha",
        lambda path: draft["candidate_geometry_sha256"] if path == screen else source_sha(path))
    witnessed = []
    monkeypatch.setattr(parked_reference, "load_parked_reference",
        lambda path,digest,step: witnessed.append((path,digest,step)))
    near, initial = pilot.freeze(Namespace(**dict(paths,verify_near_open_settling=True,
        near_open_parent_receipt=parent,near_open_screen=screen)))
    assert witnessed == [(parent,draft["source_parked_outer_sha256"],240)]
    assert near["budget"]["sdk_steps"] == 240 and near["budget"]["control_steps"] == 0
    assert near["budget"]["mode"] == "power_supported_near_open_no_contact_settling"
    assert initial["rod_nodes_w_m"] == config["rod_nodes_w_m"]
    assert str(pilot.NEAR_OPEN) in near["frozen_files"]
    assert all(initial["hands"][side]["q_rad"] != original["hands"][side]["q_rad"] for side in power.SIDES)
    screen.write_text("".join(json.dumps(dict(side=side, candidates=[])) + "\n"
                              for side in draft["candidate"]))
    with pytest.raises(ValueError, match="geometry precheck"):
        pilot.freeze(Namespace(**dict(paths,verify_near_open_settling=True,
            near_open_parent_receipt=parent,near_open_screen=screen)))
    assert settled["budget"]["sdk_steps"] == 240 and settled["budget"]["solver_steps"] == 238*12
    assert str(pilot.SETTLE) in settled["frozen_files"]
    assert config["rod_nodes_w_m"] == original["rod_nodes_w_m"]
    assert config["support_scene"] == original["support_scene"]
    for side in power.SIDES:
        assert config["hands"][side]["pose_wxyz"][0] - original["hands"][side]["pose_wxyz"][0] == pytest.approx(.105)


@pytest.mark.parametrize(("isolated", "hand_pair"), [(False, False), (True, False), (True, True)])
def test_supported_worker_records_all_exchanges_and_independent_support_loads(
        tmp_path, monkeypatch, isolated, hand_pair):
    from types import SimpleNamespace as NS
    from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_LINEAR
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_reset as reset_api
    from go2_pvcnn.control.dh116 import tabletop_power_contact_evidence as evidence
    from scripts import verify_dh116_power_shared_rod as shared

    sim = NS(sdk=2)
    monkeypatch.setattr(shared.native, "clock", lambda ctx: {"sdk_step": ctx.sdk})
    def advance(ctx, previous, *, budget_end):
        assert ctx.sdk == previous and previous < budget_end
        ctx.sdk += 1
        return ctx.sdk
    monkeypatch.setattr(shared, "step_thumb_physics", advance)
    monkeypatch.setattr(shared, "hand_safety", lambda binding: None)
    monkeypatch.setattr(power, "write_composer_reactions", lambda *args, **kw: None)
    monkeypatch.setattr(reset_api, "read_joint_target_evidence",
        lambda hand: {"backend": {"available": True, "values": np.zeros((1, 17)), "source": "test"}})
    monkeypatch.setattr(evidence, "measure_power_contact",
        lambda row: {"all_substeps_power_contacts": False})
    names = power.SUPPORTED_POWER_MESH_NAMES
    hand_names = power.HAND_NAMES + power.PALM_NAMES
    meshes = {key: dict(body_index=index, side=key.split(".")[0])
              for index, key in enumerate(hand_names)}
    poses = {key: (np.zeros(3), np.array([0., 0., 0., 1.])) for key in hand_names}
    monkeypatch.setattr(shared, "measure_links", lambda *args, **kw: (poses, {}, np.ones(2)))
    monkeypatch.setattr(shared, "bind_palm_sources",
        lambda *args, **kw: ({key: meshes[key] for key in power.PALM_NAMES}, {}))
    hands = []
    for side in power.SIDES:
        view = NS(get_dof_positions=lambda: np.zeros((1, 17)), get_dof_velocities=lambda: np.zeros((1, 17)))
        joint_names = list(WRIST_LINEAR) + [f"other{i}" for i in range(17-len(WRIST_LINEAR))]
        hand = NS(permanent_wrench_composer=object(), root_physx_view=view,
                  joint_names=joint_names, write_data_to_sim=lambda: None, update=lambda _: None)
        hands.append(NS(side=side, hand=hand))
    clock = lambda: dict(physics_step=sim.sdk, rod_tick=(sim.sdk-2)*12)
    original = NS(bindings={key: meshes[key] for key in power.HAND_NAMES}, clock=clock, rope=object())
    owners = NS(resources={"runtime": original,
        "dual_episode": NS(_bindings=hands, _exchange=NS(_read_geometry=NS(_shapes={})))},
        keep=lambda name, value: value)
    fixture = NS(_base=NS(nodes=np.zeros((41, 3))), rope=original.rope, clock=clock, mesh_names=names,
                 bindings=meshes)
    vectors = {side: {"forces": np.zeros((1,6,3))} for side in power.SIDES}
    pair_names = ["support_0", "support_1"]
    if hand_pair:
        pair_names.append("left.finger14_link")
    fixture.step = lambda *args, **kw: dict(hand_wrenches_com_world=vectors, pair_force_residual_n=0.,
        rod_mesh_contact_pairs={"pairs":[{"mesh_name": name} for name in pair_names]},
        external_support_wrenches_world={"table": np.zeros(6),
            "support_0": np.array([0.,0.,.15, 0.,0.,0.]), "support_1": np.array([0.,0.,.17, 0.,0.,0.])})
    def attach(orig, **kwargs):
        assert orig is original and "support_poses" in kwargs
        return fixture
    monkeypatch.setattr(power.PowerSharedRodRuntime, "from_original_reset", attach)

    result = shared.run_shared(sim, owners, None, tmp_path, profile=None,
        args=NS(measure_contact_evidence=True), require_no_hand_contact=isolated,
        supported_acquire=lambda *a, **kw: ({"clock": dict(physics_step=2)}, {"support_poses": {}}))
    if hand_pair:
        assert "recorded a hand pair" in result["failure"]
        assert result["guarded_stop_sdk_step"] == 3
        assert (tmp_path/"power_exchange_3.json").exists()
        return
    assert result.get("failure") is None, result
    assert result["solver_substeps"] == 96
    assert result["native_qualification"] is False
    assert result["peak_support_force_n"] == {"table":0., "support_0":.15,"support_1":.17}
    assert len(list(tmp_path.glob("power_exchange_*.json"))) == 8
    binding = json.loads((tmp_path/"power_binding.json").read_text())
    assert binding["pair_capacity"] == 41*15*12 and binding["supported_task"] is True

    row = json.loads((tmp_path/"power_exchange_3.json").read_text())
    assert row["native_joint_state"]["left"]["position_units"][:3] == ["m"]*3
    assert row["native_joint_state"]["left"]["velocity_units"][-1] == "rad/s"
