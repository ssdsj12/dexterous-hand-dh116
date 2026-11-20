import numpy as np
import pytest

from scripts.audit_dh116_pinch_thumb_projections import measure_step, summarize


def pair(mesh, node, point, normal, force, substep=0):
    return dict(mesh_name=mesh, mesh_index=node, node_index=node, substep=substep,
                contact_point_world_m=point, outward_normal_world=normal, force_on_rod_world_n=force)


def make_exchange(thumb_tangent):
    pairs = []
    for sub in range(12):
        pairs.extend((
            pair("left.finger23_link", 34, [0.,0.,0.], [1.,0.,0.], [.1,0.,0.], sub),
            pair("left.finger14_link", 35, [.01,0.,0.], [-.6,.8,0.], [-.06,.08,0.] if thumb_tangent is None
                 else [-.06 + .8*thumb_tangent, .08 + .6*thumb_tangent, 0.], sub),
        ))
    return dict(physics_step=299, rod=dict(rod_mesh_contact_pairs=dict(pairs=pairs)),
                stage_reaction=dict(per_substep=[dict(substep=sub) for sub in range(12)]))


def test_tangential_opposition_loss_is_distinct_from_normal_contact_loss():
    rows = measure_step(make_exchange(.1))
    assert all(x["normal_n"][1] > 0 and x["inward_n"][1] <= 0 for x in rows)
    assert rows[0]["material_nodes"] == [34, 35]
    result = summarize(rows)
    assert result["negative_thumb_inward_count"] == result["sign_losses"] == 12
    assert result["nodes_hist"] == [[34, 35, 12]]
    np.testing.assert_allclose(result["thumb_inward_min_n"], -.02)


def test_missing_or_ambiguous_substep_evidence_rejected():
    exchange = make_exchange(None)
    pairs = exchange["rod"]["rod_mesh_contact_pairs"]["pairs"]
    pairs.append(pairs[0])
    with pytest.raises(ValueError, match="multiple"):
        measure_step(exchange)
    pairs.pop()
    pairs.pop()
    with pytest.raises(ValueError, match="missing"):
        measure_step(exchange)


def test_positive_contact_vectors_are_preserved():
    result = summarize(measure_step(make_exchange(None)))
    assert result["sign_losses"] == 0
    assert result["negative_thumb_inward_count"] == 0
    assert result["substeps"] == 12
