"""CPU contract tests; fixtures do not qualify the CUDA getter."""
import hashlib
from pathlib import Path
import numpy as np
import pytest

from scripts import prepare_dh116_deformx_parameter_getters as patch
from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot

def packet():
    rest = np.array([[.01, .02]], dtype=np.float32)
    return {
        "schema": "dh116.native_parameter_snapshot.v1", "num_envs": 1, "nodes_per_env": 3,
        "time_s": 0., "metadata_source": "native_host_metadata",
        "segment_source": "cuda_device_copy", "hardware_measured": False,
        "contact_radius_m": .005, "gravity_world_m_s2": np.array([0, 0, -9.81], np.float32),
        "contact_mu_static": .5, "contact_mu_kinetic": .4,
        "contact_v_slip_m_s": .001, "contact_kappa": 1.,
        "rest_lengths_m": rest, "stored_k_stretch_n_m": rest * 800,
        "rest_quaternion_norm_n_m": 4 * .018 / rest,
        "stretch_ea_n": np.full((1, 2), 800, np.float32),
        "bend_ei_n_m2": np.full((1, 2), .018, np.float32),
        "inverse_masses_kg_inv": np.full((1, 3), 100, np.float32),
    }

def test_native_fields_and_derived_values_keep_separate_sources():
    p = packet()
    r = validate_snapshot(p, expected_num_envs=1, expected_nodes=3, expected_time_s=0.)
    assert r["fields"]["contact_radius_m"]["source"] == "native_host_metadata"
    assert r["fields"]["rest_lengths_m"]["source"] == "cuda_device_copy"
    assert r["fields"]["bend_ei_n_m2"]["source"] == "derived_from_cuda_device_copy"
    assert r["fields"]["contact_radius_m"]["uncertainty"] is None
    assert r["hardware_measured"] is False and r["teacher_qualified"] is False
    assert r["fields"]["stretch_ea_n"]["value"] == [[800., 800.]]
    p["rest_lengths_m"][0, 0] = 9
    assert r["fields"]["rest_lengths_m"]["value"][0][0] < .011

@pytest.mark.parametrize("key,value", [
    ("schema", "configuration.v1"), ("metadata_source", "config"),
    ("segment_source", "host_mirror"), ("hardware_measured", True),
    ("contact_radius_m", float("nan")), ("contact_radius_m", 0.),
    ("gravity_world_m_s2", [0, 0]), ("contact_mu_static", -1),
    ("contact_kappa", 0), ("contact_v_slip_m_s", 0),
    ("rest_lengths_m", [[.01, .02, .03]]),
    ("rest_lengths_m", [[.01, 0]]),
    ("stored_k_stretch_n_m", [[8, float("nan")]]),
    ("rest_quaternion_norm_n_m", [[0, 1]]),
    ("bend_ei_n_m2", [[.018, .18]]),
    ("stretch_ea_n", [[80, 800]]),
    ("inverse_masses_kg_inv", [[-1, 100, 100]]),
    ("time_s", .01), ("num_envs", 2), ("nodes_per_env", 4),
])
def test_invalid_snapshot_is_not_qualified(key, value):
    p = packet(); p[key] = value
    with pytest.raises(ValueError):
        validate_snapshot(p, expected_num_envs=1, expected_nodes=3, expected_time_s=0.)

def test_fixed_nodes_are_reported_not_hidden():
    p = packet(); p["inverse_masses_kg_inv"][0, 1] = 0
    r = validate_snapshot(p, expected_num_envs=1, expected_nodes=3, expected_time_s=0.)
    assert r["fixed_node_indices"] == [[0, 1]]

def test_patch_requires_exact_post_pair_source_sha(monkeypatch):
    original = "// fixture\n" + patch.ANCHOR
    monkeypatch.setattr(patch, "POST_PAIR_BINDINGS_SHA", hashlib.sha256(original.encode()).hexdigest())
    modified = patch.patch_bindings(original)
    assert modified.count('.def("parameter_snapshot"') == 1
    assert modified.endswith(patch.ANCHOR)
    with pytest.raises(ValueError):
        patch.patch_bindings(modified)
    with pytest.raises(ValueError):
        patch.patch_bindings(original + "// drift")

def test_getter_copies_device_rest_quaternions_and_never_uploads():
    code = patch.GETTER
    assert "m.d_qRests" in code and "cudaMemcpyDeviceToHost" in code
    assert "std::lock_guard<std::mutex>" in code
    assert "cudaStreamSynchronize" in code
    assert "V - 1" in code
    for forbidden in ("->download(", "->upload(", "step_free(", "keep_alive", "CudaView", "set_contact"):
        assert forbidden not in code
    assert "m.contactRadius" in code and "m.radius" not in code

def test_non_worktree_and_dirty_source_rejected_before_write(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="separate"):
        patch.apply(tmp_path)
    (tmp_path / ".git").rmdir()
    (tmp_path / ".git").write_text("gitdir: dummy")
    monkeypatch.setattr(patch, "git_output", lambda root, *args: patch.UPSTREAM if args[0] == "rev-parse" else " M file")
    with pytest.raises(ValueError, match="clean"):
        patch.apply(tmp_path)

def test_plan_is_all_or_nothing_on_wrong_anchor(tmp_path, monkeypatch):
    target = tmp_path / "csrc/bindings.cpp"; target.parent.mkdir()
    target.write_text("unchanged")
    monkeypatch.setattr(patch, "PAIR_PATCHES", [("csrc/bindings.cpp", "unchanged", "changed"),
                                                ("csrc/missing.cpp", "x", "y")])
    with pytest.raises((ValueError, FileNotFoundError)):
        patch.planned_changes(tmp_path)
    assert target.read_text() == "unchanged"

def test_build_output_cannot_replace_source_sdk_or_existing_directory(tmp_path):
    from scripts.build_dh116_deformx_parameter_getters import validate_output
    source = tmp_path / "source"; source.mkdir()
    sdk = tmp_path / "sdk"; sdk.mkdir()
    for output in (source, source / "nested", sdk / "nested", tmp_path):
        with pytest.raises(ValueError):
            validate_output(source, sdk, output)
    assert validate_output(source, sdk, tmp_path / "new-build") == tmp_path / "new-build"

def test_source_audit_trusts_only_explicit_worktree_for_child(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import build_dh116_deformx_parameter_getters as build
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"source_inventory_available": True}),
                               stderr="")
    monkeypatch.setattr(build.subprocess, "run", run)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")
    assert build.audit_worktree(tmp_path)["source_inventory_available"]
    env = calls[0][1]["env"]
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert env["GIT_CONFIG_KEY_1"] == "safe.directory"
    assert env["GIT_CONFIG_VALUE_1"] == str(tmp_path.resolve())
    assert env["GIT_CONFIG_VALUE_0"] == "false"
    assert build.os.environ["GIT_CONFIG_COUNT"] == "1"
