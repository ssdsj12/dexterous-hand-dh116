import numpy as np

from scripts.audit_dh116_cooked_hull_device_parity import (
    compare_hull_topology, physics_device_matches,
)


def _shape(vertices, rings):
    indices = []
    polygons = []
    for ring in rings:
        polygons.append({"index_base": len(indices), "num_vertices": len(ring)})
        indices.extend(ring)
    return {"vertices": vertices, "indices": indices, "polygons": polygons}


def test_device_check_requires_runtime_gpu_scene_and_tensor_ordinal():
    gpu = {"requested_device": "cuda:0", "manager_device": "cuda:0",
           "tensor_device_ordinal": 0, "gpu_dynamics_enabled": True,
           "broadphase": "GPU", "physx_device_override": -1}
    assert physics_device_matches(gpu)
    for key, value in (("manager_device", "cpu"), ("tensor_device_ordinal", -1),
                       ("gpu_dynamics_enabled", False), ("broadphase", "MBP"),
                       ("physx_device_override", 0)):
        assert not physics_device_matches({**gpu, key: value})
    cpu = {"requested_device": "cpu", "manager_device": "cpu",
           "tensor_device_ordinal": -1, "gpu_dynamics_enabled": False,
           "broadphase": "MBP", "physx_device_override": -1}
    assert physics_device_matches(cpu)
    assert not physics_device_matches({**cpu, "tensor_device_ordinal": 0})


def test_parity_is_invariant_to_vertex_and_polygon_permutations():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rings = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    permutation = [3, 2, 0, 1]
    remap = {v: i for i, v in enumerate(permutation)}
    new_rings = [[remap[v] for v in ring[1:] + ring[:1]] for ring in rings[::-1]]
    a = _shape(vertices, rings)
    b = _shape(vertices[permutation], new_rings)
    comparison = compare_hull_topology(a, b)
    assert comparison["parity_passed"]
    assert comparison["max_vertex_delta_m"] == 0
    assert not compare_hull_topology(a, _shape(vertices[permutation], new_rings[:-1]))["parity_passed"]
    assert not compare_hull_topology(a, _shape(vertices[permutation] + .001, new_rings))["parity_passed"]
    assert not compare_hull_topology(a, _shape(vertices[permutation], [r[::-1] for r in new_rings]))["parity_passed"]
