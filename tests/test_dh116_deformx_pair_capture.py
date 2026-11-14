import copy

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import validate_pair_capture


def packet():
    return dict(substeps=2, mesh_count=2, capacity=8, attempted=2, overflow=False,
                # substep, env-local node, mesh, position, normal, force ON rod, torque ON mesh
                indices=np.array([[1, 0, 1], [0, 0, 0]], dtype=np.int64),
                vectors=np.array([[0, 0, 1, 0, 0, 1, 0, 0, 4, 0, 0, 0],
                                  [1, 0, 1, 0, 0, 1, 0, 0, 2, 0, -1, 0]], dtype=np.float32))


def test_pairs_are_independent_and_reconcile_to_original_net_wrenches():
    p = packet()
    result = validate_pair_capture(p, dt_s=1/480, node_forces=[[0, 0, 3]],
                                   mesh_wrenches=[[0, 0, -1, 0, -.5, 0], [0, 0, -2, 0, 0, 0]])
    assert [r["mesh_index"] for r in result["pairs"]] == [0, 1]
    assert result["pairs"][0]["normal_force_on_rod_n"] == 2
    assert result["pairs"][1]["impulse_on_rod_ns"] == [0, 0, 4/960]


@pytest.mark.parametrize("edit", [
    lambda p: p.update(overflow=True),
    lambda p: p.update(attempted=3),
    lambda p: p["indices"].__setitem__((0, 0), 2),
    lambda p: p["indices"].__setitem__(0, p["indices"][1]),
    lambda p: p["vectors"].__setitem__((1, 4), .5),
    lambda p: p["vectors"].__setitem__((0, 8), 100),
    lambda p: p["vectors"].__setitem__((0, 0), float("nan")),
])
def test_incomplete_or_unphysical_pairs_cannot_admit_contact(edit):
    p = copy.deepcopy(packet())
    edit(p)
    with pytest.raises(ValueError):
        validate_pair_capture(p, dt_s=1/480, node_forces=[[0, 0, 3]],
                              mesh_wrenches=[[0, 0, -1, 0, -.5, 0], [0, 0, -2, 0, 0, 0]])


def test_capture_consumes_one_atomic_host_snapshot_with_exact_binding():
    from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import read_pair_capture

    p = packet()
    p.update(node_forces_on_rod_world_n=[[0, 0, 3]],
             mesh_wrenches_on_mesh_world=[[0, 0, -1, 0, -.5, 0], [0, 0, -2, 0, 0, 0]])
    calls = []

    class Solver:
        def contact_pair_samples(self):
            calls.append("snapshot")
            return p

    result = read_pair_capture(Solver(), dt_s=1/480, physics_step=3, rod_tick=12,
                               mesh_names=("left.finger14_link", "table"))
    assert calls == ["snapshot"]
    assert result["physics_step"] == 3 and result["rod_tick"] == 12
    assert [x["mesh_name"] for x in result["pairs"]] == ["left.finger14_link", "table"]
    assert result["physx_contact_pairs"] is None
    assert result["native_qualification"] is False
    with pytest.raises(ValueError, match="mesh"):
        read_pair_capture(Solver(), dt_s=1/480, physics_step=3, rod_tick=12,
                          mesh_names=("table", "table"))


def test_cpu_preflight_and_ungated_native_leave_no_output(tmp_path, monkeypatch, capsys):
    import hashlib
    import json
    import sys
    from scripts import verify_dh116_deformx_pair_capture as cli

    extension = tmp_path / "ropesim" / "_ropesim.cpython-310-test.so"
    extension.parent.mkdir()
    extension.write_bytes(b"cpu test")
    output = tmp_path / "output"
    argv = ["verify", "--output", str(output), "--extension", str(extension),
            "--extension-sha256", hashlib.sha256(extension.read_bytes()).hexdigest()]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv(cli.WORKER_ENV, raising=False)

    def cannot_construct(*args):
        pytest.fail("CPU or ungated path reached CUDA")

    monkeypatch.setattr(cli, "run_calibration", cannot_construct)
    cli.main()
    assert json.loads(capsys.readouterr().out)["native_execution_performed"] is False
    monkeypatch.setattr(sys, "argv", argv + ["--execute-native"])
    with pytest.raises(ValueError, match="registered GPU 3 scheduler worker"):
        cli.main()
    assert not output.exists()


def test_actual_imported_extension_must_match_candidate_before_constructing(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts import verify_dh116_deformx_pair_capture as cli

    expected = tmp_path / "ropesim" / "_ropesim.cpython-310-expected.so"
    expected.parent.mkdir()
    alternate = expected.with_name("_ropesim.cpython-310-alternate.so")
    alternate.touch()

    def cannot_construct(*args):
        pytest.fail("mismatched binary reached solver construction")

    module = SimpleNamespace(RopeBatch=SimpleNamespace(from_positions=cannot_construct),
                             _ropesim=SimpleNamespace(__file__=str(alternate)), sdf=None)
    monkeypatch.setitem(sys.modules, "ropesim", module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(ValueError, match="loaded solver binary differs"):
        cli.run_calibration(expected, "0" * 64)


def test_reopen_packet_detects_tampered_result(tmp_path):
    from scripts import verify_dh116_deformx_pair_capture as cli

    raw = dict(substeps=12, mesh_count=1, capacity=36, attempted=1, overflow=False,
               indices=[[0, 0, 0]], vectors=[[0, 0, 1, 0, 0, 1, 0, 0, 12, 0, 0, 0]],
               node_forces_on_rod_world_n=[[0, 0, 1], [0, 0, 0], [0, 0, 0]],
               mesh_wrenches_on_mesh_world=[[0, 0, -1, 0, 0, 0]])
    report = validate_pair_capture(raw, dt_s=cli.DT,
        node_forces=raw["node_forces_on_rod_world_n"], mesh_wrenches=raw["mesh_wrenches_on_mesh_world"])
    report.update(solver_clock_before_s=0., solver_clock_after_s=cli.DT, stale_reset_rejected=True)
    cli.write_new(tmp_path / "raw_pairs.json", raw)
    cli.write_new(tmp_path / "pairs.json", report)
    cli.write_new(tmp_path / "result.json", {"pair_count": 1})
    cli.validate_artifacts(tmp_path)
    (tmp_path / "result.json").write_text('{"pair_count": 2}')
    with pytest.raises(ValueError, match="does not reconcile"):
        cli.validate_artifacts(tmp_path)
