import numpy as np
import pytest

from scripts.smoke_dh116_shared_rod_native import command_for_tick


def test_eight_sdk_ticks_replay_exact_24d_and_neighbor_holds():
    commands = [np.linspace(-1, 1, 24), np.linspace(1, -1, 24)]
    for step in range(11, 27):
        actual = command_for_tick(commands, physics_step=step, start_step=10, env_id=0)
        np.testing.assert_array_equal(actual, commands[(step - 11)//8])
        np.testing.assert_array_equal(command_for_tick(commands,
            physics_step=step, start_step=10, env_id=1), np.zeros(24))
    with pytest.raises(ValueError):
        command_for_tick(commands, physics_step=27, start_step=10, env_id=0)
    commands[0][0] = np.nan
    with pytest.raises(ValueError):
        command_for_tick(commands, physics_step=11, start_step=10, env_id=0)
