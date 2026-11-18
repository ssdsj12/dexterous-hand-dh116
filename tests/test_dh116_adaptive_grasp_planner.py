import numpy as np

from go2_pvcnn.control.dh116.adaptive_grasp_planner import rank_grasp_candidates


def square_contacts():
    return np.array([[.02, .02, 0.], [-.02, .02, 0.],
                     [-.02, -.02, 0.], [.02, -.02, 0.]]), np.tile([0., 0., 1.], (4, 1))


def test_planner_rejects_collision_and_support_failures_and_ranks_alternates():
    points, normals = square_contacts()
    candidates = [
        dict(candidate_id="unsafe", points=points, normals=normals,
             material_ids=[1, 2, 3, 4], com=points.mean(axis=0),
             reach_cost=0.1, collision_risk=1.0),
        dict(candidate_id="good", points=points, normals=normals,
             material_ids=[5, 6, 7, 8], com=points.mean(axis=0),
             reach_cost=0.2, collision_risk=0.1,
             rod_bending_cost=0.1),
        dict(candidate_id="heavy", points=points, normals=normals,
             material_ids=[9, 10, 11, 12], com=points.mean(axis=0), mass_kg=2.0, reach_cost=0.0,
             collision_risk=0.0),
    ]
    ranked = rank_grasp_candidates(
        np.c_[np.linspace(0., .5, 5), np.zeros(5), np.full(5, .1)],
        candidates, max_load_n=.5,
        collision_check=lambda c: c["candidate_id"] != "unsafe",
    )
    assert [item.candidate_id for item in ranked] == ["good"]
    assert ranked[0].support["feasible"]
    assert not ranked[0].support["grasp_qualified"]
    assert ranked[0].alternates == ("good",)


def test_planner_requires_four_contacts_and_finite_wire():
    points, normals = square_contacts()
    bad = dict(candidate_id="bad", points=points[:3], normals=normals[:3],
               material_ids=[1, 2, 3])
    try:
        rank_grasp_candidates(np.zeros((2, 3)), [bad])
    except ValueError as exc:
        assert "four" in str(exc)
    else:
        raise AssertionError("invalid contact candidate was accepted")

    with np.testing.assert_raises(ValueError):
        rank_grasp_candidates(np.array([[np.nan, 0., 0.], [0., 0., 0.]]), [])
