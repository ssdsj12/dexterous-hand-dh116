"""Actual Adam/RNG resume across two fresh CPU fixture processes."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_two_process_resume_keeps_a_and_advances_b_once(tmp_path):
    root = Path(__file__).resolve().parents[1]
    parent_env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(root), str(root / "src"), str(root / "tests"), os.environ.get("PYTHONPATH", "")]
    )}
    results = {}
    for role in ("a", "b"):
        p = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), role, str(tmp_path)],
                           cwd=root, env=parent_env, text=True, capture_output=True, timeout=100)
        assert p.returncode == 0, p.stdout + p.stderr
        results[role] = json.loads(p.stdout.strip().splitlines()[-1])

    a, b = results.values()
    assert a["pid"] != b["pid"] and a["episode"] != b["episode"]
    assert a["update"] == 1 and a["steps"] == 8
    assert b["update"] == 2 and b["steps"] == 16 and b["physical_steps"] == 64
    assert b["training_loaded"] == 1 and b["restored_adam"] == a["accepted"]
    assert b["accepted"] > a["accepted"] and b["delta_l2"] > 0.0
    assert b["restored_rng_sha256"] == a["rng_sha256"]
    assert b["loaded_weights_sha256"] == a["weights_sha256"]
    assert b["weights_sha256"] != a["weights_sha256"]
    assert b["origin"] == {"kind": "resume", "checkpoint_sha256": a["checkpoint_sha256"]}
    assert hashlib.sha256((tmp_path / "a" / "teacher_clean.pt").read_bytes()).hexdigest() == a["checkpoint_sha256"]
    assert a["native_qualification"] is b["native_qualification"] is False


def _stage(role, work):
    import random
    import numpy as np
    import torch

    from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_update import run_fresh_update, run_resumed_update
    from go2_pvcnn.control.dh116.tabletop_native_teacher_checkpoint import _rng_state, checkpoint_trainer_factory
    from test_native_teacher_checkpoint import contract
    from test_native_teacher_collector import fixture
    from test_native_teacher_context import backend_profile

    torch.set_num_threads(1)
    seed = 116 if role == "a" else 117
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    folder = work / role
    folder.mkdir()
    origin = json.loads((work / "a.json").read_bytes()) if role == "b" else None
    captured = {}
    uid = f"cpu-two-process-{os.getpid()}-{role}"
    original_reset = DualHandPhysicalEpisode.reset

    def reset(self, *args, episode_uid, **kwargs):
        return original_reset(self, *args, episode_uid=uid, **kwargs)

    def configure(trainer, owner):
        assert owner.context._current is not None
        assert owner.episode._state == "active"
        path = work / "a" / "teacher_clean.pt"
        restore = checkpoint_trainer_factory(lambda: trainer, path, contract,
                                             expected_sha256=origin["checkpoint_sha256"])
        assert restore() is trainer
        captured.update(restored_rng_sha256=_digest(_rng_state()),
                        loaded_weights_sha256=_digest(trainer.model.state_dict()),
                        restored_adam=int(next(iter(trainer.optimizer.state.values()))["step"].item()))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DualHandPhysicalEpisode, "reset", reset)
        collector, _, rod, *_ = fixture(patch, folder, backend_profile.__wrapped__(),
                                        rollout_steps=8, configure_trainer=configure if role == "b" else None)
        checked = contract(collector.trainer)
        if role == "a":
            result = run_fresh_update(collector, checkpoint_contract=checked, controls=8)
        else:
            origin = dict(checkpoint_path=str(work / "a" / "teacher_clean.pt"),
                          checkpoint_sha256=origin["checkpoint_sha256"],
                          policy_weights_sha256=origin["weights_sha256"],
                          rng_state_sha256=origin["rng_sha256"],
                          loaded_accepted_minibatch_updates=origin["accepted"],
                          producer_episode_uid=origin["episode"])
            with torch.random.fork_rng(devices=[]):
                torch.rand(1)
                with pytest.raises(ValueError, match="restored optimizer RNG differs"):
                    run_resumed_update(collector, checkpoint_contract=checked, origin=origin, controls=8)
                assert collector.transitions == 0 and rod.rope.steps == 0
                assert not (folder / "teacher_clean.pt").exists()
            result = run_resumed_update(collector, checkpoint_contract=checked, origin=origin, controls=8)
        events = [json.loads(line)["operation"] for line in collector.journal.path.read_text().splitlines()]
        assert events.count("optimizer_update_accepted") == 1
        assert events.count("fresh_update_checkpoint_sealed") == int(role == "a")
        assert events.count("resume_update_checkpoint_sealed") == int(role == "b")
        saved = torch.load(folder / "teacher_clean.pt", map_location="cpu", weights_only=True)
        if role == "b":
            assert result["restored_rng_sha256"] == captured["restored_rng_sha256"]
        info = dict(pid=os.getpid(), episode=rod.clock()["episode_uid"], update=collector.trainer.counters["updates"],
                    steps=collector.trainer.counters["environment_steps"], physical_steps=rod.rope.steps,
                    training_loaded=result["loaded_updates"], delta_l2=result["parameter_delta_l2"],
                    accepted=result["accepted_minibatch_updates"],
                    checkpoint_sha256=result["clean_checkpoint"]["sha256"],
                    weights_sha256=result["clean_checkpoint"]["policy_weights_sha256"],
                    rng_sha256=_digest(saved["rng"]), origin=saved["origin"],
                    native_qualification=result["native_qualification"], **captured)
        (work / (role + ".json")).write_text(json.dumps(info, allow_nan=False))
        print(json.dumps(info, allow_nan=False))


if __name__ == "__main__":
    _stage(sys.argv[1], Path(sys.argv[2]).resolve())
