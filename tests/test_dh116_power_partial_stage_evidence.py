from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import read_pair_capture
from scripts import verify_dh116_power_partial_stage as stage
from test_dh116_power_palm_contact_evidence import build


LINKS = ("base_link", "finger14_link", "finger53_link")
STATIONS = {f"{side}.{link}": (node,) for side, node in (("left", 33), ("right", 7)) for link in LINKS}
SCENE = {"supports": [], "wire_radius_m": .005}


def pair_exchanges(*, left_little=True, extra=None):
    rod = build(proximal=True)
    names = rod["mesh_names"]
    selected = set(STATIONS)
    if not left_little:
        selected.remove("left.finger53_link")
    if extra is not None:
        selected.add(extra)
    pairs = [pair for pair in rod["rod_mesh_contact_pairs"]["pairs"] if pair["mesh_name"] in selected]
    forces, wrenches = np.zeros((41,3)), np.zeros((len(names),6))
    indices, vectors = [], []
    for pair in pairs:
        step, node, mesh = (pair[key] for key in ("substep","node_index","mesh_index"))
        force = pair["force_on_rod_world_n"]
        forces[node] += np.asarray(force) / 12
        wrenches[mesh, :3] -= np.asarray(force) / 12
        indices.append([step,node,mesh])
        vectors.append([*pair["contact_point_world_m"], *pair["outward_normal_world"],
                        *force, *pair["torque_on_mesh_about_origin_world_nm"]])
    raw = dict(substeps=12, mesh_count=len(names), capacity=41*len(names)*12,
               attempted=len(indices), overflow=False, indices=np.asarray(indices), vectors=np.asarray(vectors),
               node_forces_on_rod_world_n=forces, mesh_wrenches_on_mesh_world=wrenches)
    capture = read_pair_capture(SimpleNamespace(contact_pair_samples=lambda: raw), dt_s=1/480,
                                physics_step=3, rod_tick=12, mesh_names=names)
    capture["solver_buffer"] = {key:raw[key] for key in ("substeps","mesh_count","capacity","attempted","overflow")}
    named = dict(zip(names,wrenches))
    routes = {}
    for side in ("left","right"):
        links = [key for key in rod["mesh_bindings"] if key.startswith(side+".")]
        routes[side] = dict(body_ids=[rod["mesh_bindings"][key]["body_index"] for key in links],
                            is_global=True, forces=np.asarray([[named[key][:3] for key in links]]),
                            torques=np.zeros((1,len(links),3)))
    rod.update(solver_time_s=1/480, distributed_contact=forces[None], named_wrenches=named,
               table_wrench_world=named["table"], rod_mesh_contact_pairs=capture,
               external_support_wrenches_world={name:named[name] for name in ("table","support_0","support_1")},
               hand_wrenches_com_world=routes)
    current_rod = deepcopy(rod)
    current_rod.update(physics_step=4, rod_tick=24, timestamp_s=4/480, solver_time_s=2/480)
    current_rod["rod_mesh_contact_pairs"].update(physics_step=4, rod_tick=24)
    return dict(physics_step=3, rod=rod), dict(physics_step=4, rod=current_rod, reaction_source_physics_step=3,
                                            issued_composer_wrench=deepcopy(routes))


def test_local_reaction_must_be_measured_on_both_target_pads_and_delivered_in_the_next_tick():
    first, second = pair_exchanges()
    evidence = stage.confirm_local_opposition(first, second, STATIONS, scene=SCENE)
    assert evidence["local_opposition_with_reaction"]
    assert all(phase["all_substeps_opposed"] for phase in evidence["hands"].values())
    assert not evidence["all_substeps_power_contacts"]
    assert not evidence["grasp_qualified"] and not evidence["teacher_admission"]
    second["issued_composer_wrench"]["left"]["forces"][0,0,0] += .1
    with pytest.raises(ValueError, match="applied world wrench"):
        stage.confirm_local_opposition(first, second, STATIONS, scene=SCENE)


def test_missing_selected_pad_and_unexpected_distal_never_count_as_local_opposition():
    first, second = pair_exchanges(left_little=False)
    evidence = stage.confirm_local_opposition(first, second, STATIONS, scene=SCENE)
    assert not evidence["local_opposition_with_reaction"]
    assert not evidence["hands"]["left"]["all_substeps_opposed"]
    assert evidence["hands"]["right"]["all_substeps_opposed"]
    first, second = pair_exchanges(extra="left.finger43_link")
    evidence = stage.confirm_local_opposition(first, second, STATIONS, scene=SCENE)
    assert not evidence["local_opposition_with_reaction"]
    assert evidence["unexpected_pairs"] == ["left.finger43_link:33"]


def test_a_contact_cannot_be_assigned_to_the_other_side_or_previous_episode():
    first, second = pair_exchanges()
    stations = dict(STATIONS, **{"left.finger53_link": (7,)})
    with pytest.raises(ValueError, match="assigned wire half"):
        stage.confirm_local_opposition(first, second, stations, scene=SCENE)
    second["rod"]["reset_id"] += 1
    with pytest.raises(ValueError, match="same consecutive Power exchange"):
        stage.confirm_local_opposition(first, second, STATIONS, scene=SCENE)
