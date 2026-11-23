from scripts.audit_dh116_loaded_pair_continuity import contact_runs


def test_only_adjacent_loaded_samples_of_the_same_pair_extend_a_run():
    loaded = {
        ("left.finger23_link", 35): [7, 8, 12],
        ("left.finger23_link", 36): [9, 10, 11],
    }
    runs = contact_runs(loaded)
    assert [(x["node_index"], x["first_rod_tick"], x["last_rod_tick"]) for x in runs] == [
        (35, 7, 8), (35, 12, 12), (36, 9, 11),
    ]
    assert [x["solver_steps"] for x in runs] == [2, 1, 3]


def test_empty_observation_has_no_pair_or_load_run():
    assert contact_runs({}) == []


def test_repeated_pair_tick_refuses_unverified_continuity():
    import pytest

    with pytest.raises(ValueError, match="strictly increasing"):
        contact_runs({("left.finger23_link", 35): [7, 7]})
