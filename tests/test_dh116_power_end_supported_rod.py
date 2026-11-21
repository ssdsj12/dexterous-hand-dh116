"""A free rod with the original hand slots and two independent static supports."""
import numpy as np
import pytest

from test_dual_hand_rod_runtime import FakeRope, Grid, PADS, setup
from go2_pvcnn.control.dh116 import tabletop_power_shared_rod as power

SUPPORTS = ("support_0", "support_1")


def supported_fixture(monkeypatch, *, incomplete=False, nodes=None):
    base, bindings, poses, centers = setup(monkeypatch, nodes=nodes)
    for side in ("left", "right"):
        key = f"{side}.base_link"
        bindings[key] = dict(bindings[f"{side}.finger14_link"], body_name="base_link",
                             body_index=6, prim_path=f"/World/{side}Hand/base_link")
        poses[key] = (np.zeros(3), np.array([0., 0., 0., 1.]))
        centers[key] = np.zeros(3)

    support_poses = {name:(np.array([0., .2 * index, .33]), np.array([0., 0., 0., 1.]))
                     for index,name in enumerate(SUPPORTS)}
    support_names = SUPPORTS[:-1] if incomplete else SUPPORTS
    return base, poses, centers, dict(bindings=bindings,
        right_grids={name:Grid() for name in PADS},
        right_poses={name:poses[f"right.{name}"] for name in PADS},
        palm_grids={key:Grid() for key in power.PALM_NAMES},
        palm_poses={key:poses[key] for key in power.PALM_NAMES},
        support_grids={name:Grid() for name in support_names}, support_poses=support_poses,
        require_pairs=True, run_uuid="power-supported-cpu", env_id=0)


def test_only_complete_support_bindings_can_append_any_mesh(monkeypatch):
    base, _, _, kwargs = supported_fixture(monkeypatch, incomplete=True)
    with pytest.raises(ValueError, match="support"):
        power.PowerSharedRodRuntime(base, **kwargs)
    assert len(base.rope.slots) == 6 and base.rope.steps == 0


@pytest.mark.parametrize("bad_capture", [None, "capacity", "overflow"])
def test_power_static_supports_keep_clock_and_stay_out_of_hand_wrench_routes(monkeypatch, bad_capture):
    base, poses, centers, kwargs = supported_fixture(monkeypatch)
    obj = power.PowerSharedRodRuntime(base, **kwargs)
    names = (*power.POWER_MESH_NAMES, *SUPPORTS)
    assert obj.mesh_names == names and obj.rope is base.rope
    assert len(obj.rope.slots) == 15 and not obj.rope.pinned.any()
    assert obj.rope.time == 0.0 and obj.rope.steps == 0

    def step(dt, substeps):
        FakeRope.step_free(base.rope, dt, substeps)
        base.rope.wrenches[0,13,2] = -7.
        base.rope.wrenches[0,14,2] = -9.
        base.rope.force[0,0,2] += 16.

    def capture():
        slots = [0, 6, 13, 14]
        vectors = [[0., 0., 0., 0., 0., 1., *(-12 * base.rope.wrenches[0,slot,:3]), 0., 0., 0.]
                    for slot in slots]
        return dict(substeps=12, mesh_count=15,
            capacity=12*len(base.nodes)*(13 if bad_capture=="capacity" else 15),
            attempted=len(slots), overflow=bad_capture=="overflow",
            indices=np.array([[0,0,slot] for slot in slots]), vectors=np.array(vectors),
            node_forces_on_rod_world_n=base.rope.force[0], mesh_wrenches_on_mesh_world=base.rope.wrenches[0])

    obj.rope.step_free = step
    obj.rope.contact_pair_samples = capture
    reset = obj.reset(physics_step=2, episode_uid="cpu:0", reset_id=1, pad_poses=poses)
    assert reset["external_support_wrenches_world"] is None
    assert {slot for slot,*_ in base.rope.poses} == set(range(13))-{5}
    base.rope.poses.clear()
    if bad_capture:
        with pytest.raises(ValueError):
            obj.step(poses, centers, physics_step=3)
        assert obj.clock()["clock_valid"] is False
        return
    row = obj.step(poses, centers, physics_step=3)
    assert row["rod_tick"] == 12 and row["pair_force_residual_n"] == pytest.approx(0)
    assert {slot for slot,*_ in base.rope.poses} == set(range(13))-{5}
    np.testing.assert_allclose(row["external_support_wrenches_world"]["support_0"], [0,0,-7,0,0,0])
    np.testing.assert_allclose(row["external_support_wrenches_world"]["support_1"], [0,0,-9,0,0,0])
    assert np.array_equal(row["external_support_wrenches_world"]["table"], np.zeros(6))
    routed = sum(packet["forces"][0,:,2].sum() for packet in row["hand_wrenches_com_world"].values())
    assert routed == pytest.approx(-3)
    assert row["rod_mesh_contact_pairs"]["solver_buffer"]["mesh_count"] == 15
    assert not row["native_qualification"] and not row["teacher_admission"]


@pytest.mark.parametrize("broken", [None, "binding", "reordered", "support_buffer"])
def test_loaded_static_support_is_reported_separately_from_grasp(broken):
    from types import SimpleNamespace
    from test_dh116_power_palm_contact_evidence import build
    from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import read_pair_capture
    from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import measure_power_contact

    rod = build()
    names = (*power.POWER_MESH_NAMES, *SUPPORTS)
    old = rod["rod_mesh_contact_pairs"]["pairs"]
    indices = [[x["substep"], x["node_index"], x["mesh_index"]] for x in old]
    vectors = [[*x["contact_point_world_m"], *x["outward_normal_world"], *x["force_on_rod_world_n"],
                *x["torque_on_mesh_about_origin_world_nm"]] for x in old]
    meshes = np.array([*rod["named_wrenches"].values(), np.zeros(6), np.zeros(6)])
    for step in range(12):
        indices.append([step, 20, 13])
        # Support 0 has a lateral load in addition to its upward normal.
        vectors.append([0.,0.,0.,0.,0.,1.,.2,0.,.4,0.,0.,0.])
    rod["distributed_contact"][0,20] += [.2,0.,.4]
    meshes[13,:3] = [-.2,0.,-.4]
    raw = dict(substeps=12, mesh_count=15, capacity=41*15*12, attempted=len(indices), overflow=False,
               indices=np.asarray(indices), vectors=np.asarray(vectors),
               node_forces_on_rod_world_n=rod["distributed_contact"][0], mesh_wrenches_on_mesh_world=meshes)
    capture = read_pair_capture(SimpleNamespace(contact_pair_samples=lambda:raw), dt_s=1/480,
                                physics_step=3, rod_tick=12, mesh_names=names)
    capture["solver_buffer"] = {k:raw[k] for k in ("substeps","mesh_count","capacity","attempted","overflow")}
    rod.update(mesh_names=list(names), named_wrenches=dict(zip(names,meshes,strict=True)), rod_mesh_contact_pairs=capture,
               external_support_wrenches_world={name:meshes[names.index(name)].copy() for name in ("table",*SUPPORTS)})
    if broken == "binding":
        rod["external_support_wrenches_world"].pop("support_1")
    elif broken == "reordered":
        rod["mesh_names"][13:] = reversed(SUPPORTS)
    elif broken == "support_buffer":
        rod["external_support_wrenches_world"]["support_0"][2] = 0.
    if broken:
        with pytest.raises(ValueError):
            measure_power_contact(rod)
        return
    result = measure_power_contact(rod)
    assert result["all_substeps_power_contacts"] and result["zero_table_load"]
    assert result["zero_external_support_load"] is False and result["native_qualification"] is False
    assert result["external_support_force_norm_n"]["support_0"] == pytest.approx([np.hypot(.2,.4)]*12)
    assert result["external_support_force_norm_n"]["support_1"] == [0.]*12
