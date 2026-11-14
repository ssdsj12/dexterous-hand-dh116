"""CPU tests for the independently frozen left fixed-base Pinch successor."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_dh116_pinch_v3 import BLOCKER, cpu
from scripts import verify_dh116_pinch_candidate_v4 as entry
from scripts import run_dh116_pinch_work_packages_v4 as wrapper
from go2_pvcnn.control.dh116 import tabletop_deformable_grasp_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]


def test_new_worker_preflight_in_real_cpu_process_stays_before_usd():
    process = cpu(BLOCKER + """
from pathlib import Path
from scripts import verify_dh116_pinch_candidate_v4 as entry
from scripts import run_dh116_pinch_work_packages_v4
contract,cfg=entry.preflight(Path.cwd())
entry.prior.assert_clean_import_boundary()
assert 'ropesim' not in sys.modules and 'isaaclab' not in sys.modules
assert contract['version']==entry.VERSION
assert cfg['action']['thumb_velocity_normalized']!=cfg['action']['index_velocity_normalized']
assert contract['policy_wrist_clearance_qualified'] is False
print('V4_BYTE_ONLY_PREFLIGHT')
""")
    assert process.returncode == 0, process.stdout + process.stderr
    assert process.stdout.strip() == "V4_BYTE_ONLY_PREFLIGHT"


@pytest.mark.parametrize("damage", ["action", "material", "base", "seed", "limit", "quiet"])
def test_only_frozen_successor_pose_and_separate_closure_change(damage):
    contract, original = entry.prior.preflight(ROOT)
    cfg = entry.read(ROOT / entry.POSE)
    proof = entry.read(entry.PROOF)
    if damage == "action":
        cfg["action"]["thumb_velocity_normalized"] = .25
    elif damage == "material":
        cfg["rod"]["mu_s"] = .6
    elif damage == "base":
        cfg["initial_pose"]["base_pose_wxyz"][2] += .002
    elif damage == "seed":
        cfg["initial_pose"]["active_q_rad"][3] = 1.16
    elif damage == "limit":
        proof["action"]["active_velocity_limit_rad_s"] = .4
    elif damage == "quiet":
        proof["quiet_counterfactual"]["predicted_contact_predicates"] = 1
    with pytest.raises(ValueError):
        entry.validate_candidate(cfg, original, proof, contract)


def test_exact_quiet_two_digits_and_settle_use_original_kernel():
    contract, cfg = entry.preflight(ROOT)
    target = np.array(cfg["initial_pose"]["active_q_rad"])
    start = target.copy()
    for step in range(1, 239):
        action = runtime.candidate_action(cfg, step)
        if step <= 48 or step > 208:
            assert not action.any()
        else:
            assert np.flatnonzero(action).tolist() == [1, 2]
        target = runtime.integrate_pinch_action(target, target, action, side="left", dt=1/480)
    proof = entry.read(entry.PROOF)
    np.testing.assert_allclose(target, proof["target_end_q_rad"], rtol=0, atol=1e-12)
    assert (start != target).sum() == 2
    assert contract["native_execution_performed"] is False


def test_frozen_source_rebinding_restored_on_acquisition_failure(monkeypatch, tmp_path):
    _, cfg = entry.preflight(ROOT)
    native = entry.native
    previous = runtime.validate_pinch_ledger
    acquire = native.build_native
    args = SimpleNamespace(output=tmp_path, cleanup="explicit")
    events = []

    def fake_execute(args, contract, cfg, project):
        assert runtime.validate_pinch_ledger is entry.validate_ledger
        assert native.VERSION == entry.VERSION
        assert native.build_native is not acquire
        assert native.run_lifecycle is not old_lifecycle
        events.append(cfg)
        raise RuntimeError("controlled acquire failure")

    old_lifecycle = native.run_lifecycle
    monkeypatch.setattr(native, "execute", fake_execute)
    monkeypatch.setattr(entry.prior, "assert_clean_import_boundary",
                        lambda: dict(loaded_usd_modules=[]))
    with pytest.raises(RuntimeError, match="controlled"):
        entry.execute(args, {}, cfg, ROOT)
    assert events == [cfg]
    assert runtime.validate_pinch_ledger is previous
    assert native.build_native is acquire and native.run_lifecycle is old_lifecycle
    assert json.loads((tmp_path / "before_app_launcher.json").read_text()) == {"loaded_usd_modules": []}


def test_measured_geometry_enforces_existing_fixed_base_floor(tmp_path):
    row = dict(table=dict(top_clearance_m=.001899, intersects=False))
    packet = dict(initial_geometry_passed=True, source_colliders={"example": row},
                  locked_pads={"example": deepcopy(row)})
    with pytest.raises(ValueError, match="1.9 mm"):
        entry.require_table_floor(packet)
    row["table"]["top_clearance_m"] = .00196
    packet["locked_pads"] = {"example": deepcopy(row)}
    assert entry.require_table_floor(packet) == .00196


def test_wrapper_rejects_foreign_budget_and_fingerprint(monkeypatch):
    monkeypatch.setattr(wrapper, "original_case", lambda _: "explicit")
    sha = "a" * 64
    spec = dict(budget=dict(physics_steps=240, updates=0, environments=1,
        gpu_index=3, timeout_seconds=180),
        argv=["--approved-code-config-sha256", sha],
        validator=dict(equals=dict(version=entry.VERSION, code_config_sha256=sha)))
    assert wrapper.cleanup_case(spec) == "explicit"
    spec["budget"]["physics_steps"] = 241
    with pytest.raises(ValueError, match="exact Pinch v4"):
        wrapper.cleanup_case(spec)
    spec["budget"]["physics_steps"] = 240
    spec["validator"]["equals"]["code_config_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="exact Pinch v4"):
        wrapper.cleanup_case(spec)
