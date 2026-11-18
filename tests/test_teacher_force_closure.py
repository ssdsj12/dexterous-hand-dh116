"""CPU tests using reconciled solver pairs, with no posture input."""
import copy
import importlib
import importlib.util
import json

import numpy as np
import pytest

MODULE = "go2_pvcnn.control.dh116.tabletop_teacher_force_closure"


def reduce(packet, grasp="Pinch", **kwargs):
    assert importlib.util.find_spec(MODULE) is not None, "force closure API missing"
    return importlib.import_module(MODULE).reduce_force_closure(packet, grasp=grasp, **kwargs)


def packet(rows=(), substeps=2):
    """Rows are (substep, node, mesh, normal ON_ROD, force ON_ROD)."""
    indices = np.array([r[:3] for r in rows], dtype=np.int64).reshape(-1, 3)
    vectors = np.array([[0., 0., 0., *r[3], *r[4], 0., 0., 0.] for r in rows]).reshape(-1, 12)
    nodes, meshes = np.zeros((41, 3)), np.zeros((13, 6))
    for s, node, mesh, normal, force in rows:
        nodes[node] += np.array(force) / substeps
        meshes[mesh, :3] -= np.array(force) / substeps
    return dict(substeps=substeps, mesh_count=13, capacity=512, attempted=len(rows), overflow=False,
                indices=indices, vectors=vectors, node_forces_on_rod_world_n=nodes,
                mesh_wrenches_on_mesh_world=meshes)


def loaded_rows(pads=2, substeps=2, load=.2):
    return [(s, 0, base + i, (1., 0., 0.) if i == 0 else (-1., 0., 0.),
             (load, 0., 0.) if i == 0 else (-load, 0., 0.))
            for s in range(substeps) for base in (0, 6) for i in range(pads)]


def test_bilateral_pinch_uses_normal_reactions_and_is_json_serializable():
    r = reduce(packet(loaded_rows()))
    assert r["bilateral"] and r["all_solver_substeps_loaded"]
    assert r["source_scope"] == "deformx_simulated_contact_reaction_not_hardware"
    assert r["native_success"] is False
    assert json.loads(json.dumps(r, allow_nan=False))["bilateral"] is True
    for side, ids in (("left", [0, 1]), ("right", [6, 7])):
        hand = r["per_hand"][side]
        assert hand["loaded_substeps"] == [0, 1]
        assert hand["worst_opposition_cos"] == pytest.approx(-1)
        assert hand["required_mesh_ids"] == ids
        assert hand["total_normal_load_n"] == pytest.approx(.4)
        assert hand["per_pad"]["thumb"]["normal_load_n"] == pytest.approx(.2)
        assert hand["per_pad"]["thumb"]["force_on_rod_world_n"] == pytest.approx([.2, 0, 0])
        assert len(hand["per_substep"]) == 2


def test_no_force_has_no_closure_even_if_no_posture_information_is_available():
    r = reduce(packet())
    assert not r["bilateral"]
    assert r["per_hand"]["left"]["loaded_substeps"] == []


@pytest.mark.parametrize("mode", ["one_hand", "one_pad", "same_direction", "flicker", "opposition_flicker", "tangent", "support_only"])
def test_missing_or_ineffective_reaction_fails_closed(mode):
    rows = loaded_rows()
    if mode == "one_hand":
        rows = [r for r in rows if r[2] < 5]
    elif mode == "one_pad":
        rows = [r for r in rows if r[2] in (0, 6)]
    elif mode == "same_direction":
        rows = [(s, n, m, (1., 0., 0.), (.2, 0., 0.)) for s, n, m, _, _ in rows]
    elif mode == "flicker":
        rows = [(s, n, m, normal, tuple(2*x for x in force)) for s, n, m, normal, force in rows if s == 0]
    elif mode == "opposition_flicker":
        rows = [(s, n, m, normal if s == 0 else (1., 0., 0.), force if s == 0 else (.2, 0., 0.)) for s, n, m, normal, force in rows]
    elif mode == "tangent":
        rows = [(s, n, m, normal, (0., 20., 0.)) for s, n, m, normal, force in rows]
    else:
        rows = [(s, 0, m, (0., 0., 1.), (0., 0., 100.)) for s in range(2) for m in (5, 11, 12)]
    r = reduce(packet(rows))
    assert not r["bilateral"] and not r["all_solver_substeps_loaded"]
    if mode in ("flicker", "opposition_flicker"):
        assert r["per_hand"]["left"]["loaded_substeps"] == [0]
    if mode == "support_only":
        assert r["external_support"]["total_normal_load_n"] == pytest.approx(300.)
        assert r["per_hand"]["left"]["total_normal_load_n"] == 0


@pytest.mark.parametrize("load", [.05, .15])
def test_strict_pad_and_total_load_thresholds(load):
    assert not reduce(packet(loaded_rows(load=load)))["bilateral"]


def test_normal_weighted_normals_use_compression_not_tangential_force_direction():
    rows = loaded_rows()
    rows += [(s, 1, 0, (0., 1., 0.), (100., 0., 0.)) for s in range(2)]
    r = reduce(packet(rows))
    assert r["bilateral"]
    thumb = r["per_hand"]["left"]["per_pad"]["thumb"]
    assert thumb["normal_on_rod_world"] == pytest.approx([1., 0., 0.])
    assert thumb["normal_load_n"] == pytest.approx(.2)


def test_tripod_requires_third_pad_on_an_opposing_face():
    assert reduce(packet(loaded_rows(pads=3)), grasp="Tripod")["bilateral"]
    rows = [(s, n, m, (0., 1., 0.), (0., .2, 0.)) if m in (2, 8) else (s, n, m, normal, force)
            for s, n, m, normal, force in loaded_rows(pads=3)]
    assert not reduce(packet(rows), grasp="Tripod")["bilateral"]


def test_power_always_blocked_without_actual_palm_proximal_source():
    r = reduce(packet(loaded_rows(pads=5)), grasp="Power")
    assert not r["bilateral"]
    for h in r["per_hand"].values():
        assert "power_palm_proximal_source_missing" in h["reasons"]
        assert h["required_mesh_ids"] in ([0, 1, 2, 3, 4], [6, 7, 8, 9, 10])


@pytest.mark.parametrize("edit", [
    lambda p: p.update(overflow=True),
    lambda p: p.update(mesh_count=12),
    lambda p: p.update(attempted=p["attempted"] + 1),
    lambda p: p["node_forces_on_rod_world_n"].__setitem__((0, 0), 2.),
    lambda p: p["mesh_wrenches_on_mesh_world"].__setitem__((0, 0), 2.),
    lambda p: p.pop("indices"),
    lambda p: p.update(vectors="broken"),
    lambda p: p["vectors"].__setitem__((0, 0), float("nan")),
])
def test_malformed_incomplete_and_mismatched_packets_raise_value_error(edit):
    p = copy.deepcopy(packet(loaded_rows()))
    edit(p)
    with pytest.raises(ValueError):
        reduce(p)


@pytest.mark.parametrize("grasp", [None, "unknown", "PINCH"])
def test_unknown_grasp_fails_closed(grasp):
    with pytest.raises(ValueError):
        reduce(packet(loaded_rows()), grasp=grasp)


def test_uses_original_packet_and_existing_validator(monkeypatch):
    module = importlib.import_module(MODULE)
    original = module.validate_pair_capture
    p = packet(loaded_rows())
    seen = []
    def validate(raw, **kwargs):
        seen.append(raw)
        return original(raw, **kwargs)
    monkeypatch.setattr(module, "validate_pair_capture", validate)
    assert reduce(p)["bilateral"]
    assert len(seen) == 1 and seen[0] is p


@pytest.mark.parametrize("grasp,pads", [("pinch", 2), ("tripod", 3), ("power", 5)])
def test_existing_runtime_lowercase_grasp_names(grasp, pads):
    result = reduce(packet(loaded_rows(pads=pads)), grasp=grasp)
    assert result["bilateral"] is (grasp != "power")
    if grasp == "power":
        assert all("power_palm_proximal_source_missing" in h["reasons"]
                   for h in result["per_hand"].values())
