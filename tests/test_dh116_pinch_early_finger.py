import numpy as np
import pytest

from scripts import screen_dh116_pinch_early_finger as screen


def fixture():
    actions = np.zeros((9, 24))
    actions[1:, [1, 2, 7, 8]] = [.5, .625, .625, .875]
    return actions


def test_only_left_pad_flexion_changes_and_cumulative_targets_are_in_radians():
    original = fixture()
    amended = screen.compile_variant(original, thumb_rad=.0026, index_rad=.007)
    expected = original.copy()
    expected[4:, 1] += .104
    expected[4:, 2] += .28
    np.testing.assert_allclose(amended, expected)
    progress = screen.additional_target_rad(original, amended)
    assert progress.shape == (72, 2)
    np.testing.assert_array_equal(progress[:32], np.zeros((32, 2)))
    np.testing.assert_allclose(progress[-1], [.0026, .007])
    assert np.all(original[:, 2] == fixture()[:, 2])


def test_rejects_commands_beyond_original_rate_and_non_finger_changes():
    original = fixture()
    with pytest.raises(ValueError, match="rate"):
        screen.compile_variant(original, thumb_rad=.0026, index_rad=.02)
    changed = original.copy()
    changed[-1, 7] = .2
    with pytest.raises(ValueError, match="left"):
        screen.additional_target_rad(original, changed)


def test_terminal_acceptance_requires_each_sample_and_every_boundary():
    accepted = dict(table_lower_m=screen.CLEARANCE_M, non_target_lower_m=.00001,
                    terminal_cases=[dict(passes_nominal_gate=True)] * 5)
    assert screen.qualified_preflight([accepted], terminal_row_count=1)
    assert not screen.qualified_preflight([accepted], terminal_row_count=2)
    assert not screen.qualified_preflight([dict(accepted, table_lower_m=.001)], terminal_row_count=1)
    assert not screen.qualified_preflight([dict(accepted, non_target_lower_m=0.)], terminal_row_count=1)
    assert not screen.qualified_preflight([dict(accepted, terminal_cases=[dict(passes_nominal_gate=True),
                                                                        dict(passes_nominal_gate=False)])],
                                           terminal_row_count=1)
