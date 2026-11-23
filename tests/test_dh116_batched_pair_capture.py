"""Cross-lane contact attribution and exact legacy reduction regression."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_batched_pair_capture import compact_pair_capture
from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import validate_pair_capture


def packet():
    lanes = []
    for env_id in range(3):
        count = 0 if env_id == 0 else 2
        force = float(env_id + 1)
        lanes.append(dict(schema="dh116.sdf_pair_samples_lane.v1", env_id=env_id,
            substeps=2, mesh_count=2, capacity=8, attempted=count, overflow=False,
            last_error=0, owner_time_s_before=1., owner_time_s_after=1.,
            indices=np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int32)[:count],
            vectors=np.array([[0, 0, 1, 0, 0, 1, 0, 0, force, 0, -1, 0],
                              [1, 0, 1, 0, 0, 1, 0, 0, force, 0, 0, 0]],
                             dtype=np.float32)[:count]))
    nodes = np.zeros((3, 1, 3), dtype=np.float32)
    wrenches = np.zeros((3, 2, 6), dtype=np.float32)
    for i in (1, 2):
        nodes[i, 0, 2] = i + 1
        wrenches[i, :, 2] = -(i + 1) / 2
        wrenches[i, 0, 4] = -.5
    return dict(schema="dh116.sdf_pair_samples_batch.v1", lanes=lanes,
                node_forces_on_rod_world_n=nodes, mesh_wrenches_on_mesh_world=wrenches)


def test_compact_reduction_matches_legacy_and_keeps_empty_lane():
    raw = packet()
    actual = compact_pair_capture(raw, dt_s=1/480, device="cpu")
    assert actual["env_ids"].tolist() == [1, 1, 2, 2]
    for i, lane in enumerate(raw["lanes"]):
        expected = validate_pair_capture(lane, dt_s=1/480,
            node_forces=raw["node_forces_on_rod_world_n"][i],
            mesh_wrenches=raw["mesh_wrenches_on_mesh_world"][i])
        assert len(expected["pairs"]) == int(actual["counts"][i])
    torch.testing.assert_close(actual["node_forces"], torch.from_numpy(raw["node_forces_on_rod_world_n"]).double())
    torch.testing.assert_close(actual["mesh_wrenches"], torch.from_numpy(raw["mesh_wrenches_on_mesh_world"]).double())
    assert actual["vectors"].shape == (4, 12)


@pytest.mark.parametrize("case", ["identity", "overflow", "duplicate", "nan", "swapped_force",
    "normal", "count", "clock", "range", "solver_error"])
def test_rejects_corruption_before_exposing_reactions(case):
    raw = deepcopy(packet())
    lane = raw["lanes"][1]
    if case == "identity":
        lane["env_id"] = 2
    elif case == "overflow":
        lane["overflow"] = True
    elif case == "duplicate":
        lane["indices"][1] = lane["indices"][0]
    elif case == "nan":
        lane["vectors"][0, 0] = np.nan
    elif case == "swapped_force":
        raw["node_forces_on_rod_world_n"][[1, 2]] = raw["node_forces_on_rod_world_n"][[2, 1]]
    elif case == "normal":
        lane["vectors"][0, 5] = .5
    elif case == "count":
        lane["attempted"] = 3
    elif case == "clock":
        lane["owner_time_s_after"] = 2.
    elif case == "range":
        lane["indices"][0, 1] = 1
    elif case == "solver_error":
        lane["last_error"] = 1
    with pytest.raises(ValueError):
        compact_pair_capture(raw, dt_s=1/480, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_reductions_match_cpu_without_changing_input():
    raw = packet()
    cpu = compact_pair_capture(raw, dt_s=1/480, device="cpu")
    gpu = compact_pair_capture(raw, dt_s=1/480, device="cuda")
    for key in cpu:
        torch.testing.assert_close(cpu[key], gpu[key].cpu(), rtol=0, atol=1e-5)
    assert raw["lanes"][0]["attempted"] == 0


def test_all_empty_batches_still_reconcile_every_lane():
    raw = packet()
    for lane in raw["lanes"]:
        lane["attempted"] = 0
        lane["indices"] = lane["indices"][:0]
        lane["vectors"] = lane["vectors"][:0]
    raw["node_forces_on_rod_world_n"].fill(0)
    raw["mesh_wrenches_on_mesh_world"].fill(0)
    actual = compact_pair_capture(raw, dt_s=1/480, device="cpu")
    assert actual["vectors"].shape == (0, 12)
    assert actual["counts"].tolist() == [0, 0, 0]
    raw["node_forces_on_rod_world_n"][2, 0, 0] = .001
    with pytest.raises(ValueError, match="reconcile"):
        compact_pair_capture(raw, dt_s=1/480, device="cpu")


def test_compact_result_owns_its_arrays_after_native_buffer_reuse():
    raw = packet()
    actual = compact_pair_capture(raw, dt_s=1/480, device="cpu")
    saved = {key: value.clone() for key, value in actual.items()}
    for lane in raw["lanes"]:
        lane["indices"].fill(0)
        lane["vectors"].fill(0)
    raw["node_forces_on_rod_world_n"].fill(0)
    raw["mesh_wrenches_on_mesh_world"].fill(0)
    for key in saved:
        torch.testing.assert_close(actual[key], saved[key])


def test_atomic_reader_binds_current_owner_and_separate_lane_clocks():
    from go2_pvcnn.control.dh116.tabletop_batched_pair_capture import read_compact_pair_capture

    class Solver:
        last_error = 0
        calls = 0

        def contact_pair_samples_batched_v1(self):
            self.calls += 1
            return packet()

    solver = Solver()
    kwargs = dict(physics_step=18, rod_ticks=(12, 24, 36), mesh_names=("pad", "table"),
                  expected_owner_time_s=1., device="cpu")
    result = read_compact_pair_capture(solver, **kwargs)
    assert solver.calls == 1
    assert result["physics_step"] == 18 and result["rod_ticks"] == (12, 24, 36)
    assert result["mesh_names"] == ("pad", "table")
    with pytest.raises(ValueError):
        read_compact_pair_capture(solver, **{**kwargs, "expected_owner_time_s":.5})
    solver.last_error = 1
    with pytest.raises(ValueError):
        read_compact_pair_capture(solver, **kwargs)
