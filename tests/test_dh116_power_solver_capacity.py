import numpy as np
import pytest

from scripts import audit_dh116_power_solver_capacity as probe


@pytest.mark.parametrize("layout_args", [[], ["--shared-sdf"]])
def test_capacity_can_request_1024(monkeypatch, capsys, layout_args):
    monkeypatch.setattr(
        "sys.argv", ["probe", "--receipt", "outer.json", "--receipt-sha256", "0"*64,
                     "--gpu-index", "2", "--num-envs", "1024", *layout_args])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="matching CUDA_VISIBLE_DEVICES"):
        probe.main()
    assert not capsys.readouterr().err


def test_allocation_stops_before_enqueuing_another_owner_under_reserve():
    available_mib = iter([2200, 2100, 2000])
    allocated = []
    assert hasattr(probe, "allocate_owners"), "a memory-reserved allocation path is required"
    result = probe.allocate_owners(1024, allocated.append, lambda: next(available_mib))
    assert allocated == [0, 1]
    assert result == ([None, None], 2000)



def test_shared_sdf_expands_only_the_archived_environment_axis():
    seed = np.zeros((1, 41, 3), dtype=np.float32)
    seed[0, :, 2] = 0.1

    def inspect_rope(batch, offsets):
        assert batch.shape == (2, 41, 3)
        assert offsets.shape == (2, 3)
        np.testing.assert_allclose(batch[1] - offsets[1], seed[0])
        np.testing.assert_allclose(batch[0] - offsets[0], seed[0] + [2, 0, 0])
        raise RuntimeError("seed witness")

    with pytest.raises(RuntimeError, match="seed witness"):
        probe.probe_shared_sdf(2, seed, inspect_rope, None, 8364, 17, 0, 0,
                               None, lambda: 3000, None)
