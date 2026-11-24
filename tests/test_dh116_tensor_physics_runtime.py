"""Shared physics/window/capture contracts with CPU producers; no native claim."""
from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from test_tensor_training_reward import batch, config, row, start

MODULE = "go2_pvcnn.control.dh116.tabletop_tensor_physics_runtime"


def implementation():
    assert importlib.util.find_spec(MODULE) is not None, "shared physical runtime is missing"
    return importlib.import_module(MODULE)


@dataclass(frozen=True)
class Boundary:
    sdk_step: int
    solver_time_s: float
    reset_ids: tuple = (0, 0)
    episode_uids: tuple = ("ep:0:0", "ep:1:0")
    rod_ticks: tuple = (0, 0)


def physical_fixture(monkeypatch):
    module = implementation()
    c = module.NativeTensorPhysicsRuntime.__new__(module.NativeTensorPhysicsRuntime)
    c.n, c.ids, c.device, c.controls = 2, (0, 1), torch.device("cpu"), 0
    c.probe_dir, c.geometry, c.first_sdk = None, None, 0
    c.timings, c.ages, c.reset_proofs = {}, torch.zeros(2, dtype=torch.long), []
    c._initialize_window_state()
    c.sim = SimpleNamespace(current_time_step_index=0)
    c.state = {"q": torch.zeros(2, 2, 17), "qd": torch.zeros(2, 2, 17),
        "base_link_pose_w": torch.zeros(2, 2, 7), "link_pose_xyzw": torch.zeros(2, 2, 23, 7),
        "link_pose_wxyz": torch.zeros(2, 2, 23, 7)}
    c.nodes, c.velocity, c.clear = torch.zeros(2, 41, 3), torch.zeros(2, 41, 3), torch.ones(2, 2)
    c.initial = {"body_com_pose_b_xyzw": torch.zeros(2, 2, 23, 7)}
    c.reward, _, prior = start(config(), n=2)
    c.last = batch(prior)[1]
    c.config = config()
    c.events = []
    c.runtime = SimpleNamespace(physics_step=0, solver=SimpleNamespace(time=0., last_error=0),
        episodes=["ep:0:0", "ep:1:0"], resets=[0, 0], ticks=[0, 0])
    def boundary():
        return Boundary(c.runtime.physics_step, c.runtime.physics_step / 480,
                        tuple(c.runtime.resets), tuple(c.runtime.episodes), tuple(c.runtime.ticks))
    c.source = SimpleNamespace(_boundary=boundary,
        set_previous_request=lambda a, t: c.events.append(("request", t)))
    c.ctrl = SimpleNamespace(pending=None, epoch=0, _failed=False,
        palms=torch.zeros(2, 2, 3), angles=torch.zeros(2, 2, 3))
    def prepare(actions, *args, **kwargs):
        c.events.append("prepare")
        c.ctrl.pending = {"phase": "prepared", "projection": {}, "expected": []}
        return c.ctrl.pending
    def submit(trans, **kwargs):
        c.events.append("submit")
        trans["phase"] = "submitted"
    def finish(trans, state, **kwargs):
        c.events.append("finish")
        trans["phase"] = "finished"
        c.ctrl.pending = None
    c.ctrl.prepare, c.ctrl.submit, c.ctrl.finish = prepare, submit, finish
    def reaction(**kwargs):
        c.events.append("reaction")
        return {"mesh_wrenches": torch.zeros(2, 2, 6)}
    c.runtime.initial_reaction = c.runtime.consume = reaction
    def advance(*, physics_step):
        c.events.append("rod")
        c.runtime.physics_step = physics_step
        c.runtime.solver.time = physics_step / 480
        c.runtime.ticks[:] = [v + 12 for v in c.runtime.ticks]
        return {"mesh_wrenches": torch.zeros(2, 2, 6)}
    c.runtime.advance = advance
    def sdk_step(sim, pre, budget_end):
        c.events.append("sdk")
        assert sim.current_time_step_index == pre
        sim.current_time_step_index += 1
        return sim.current_time_step_index
    sdk_module = importlib.import_module("scripts.verify_dh116_deformx_thumb_contact")
    monkeypatch.setattr(sdk_module, "step_thumb_physics", sdk_step)
    c.batch = SimpleNamespace(update=lambda dt: c.events.append("update"), sides={})
    def acquire(*args, **kwargs):
        c.events.append("read")
        return {key: value.clone() for key, value in c.state.items()}
    monkeypatch.setattr(module, "capture_regex_state", acquire)
    c.raw_clearance = lambda poses: c.events.append("clearance") or torch.ones(2, 2)
    c.source_guards = lambda: c.events.append("source_guard")
    c.bridge = SimpleNamespace(submit=lambda *a: c.events.append("mesh"))
    c.distance = lambda stamp: c.events.append("distance") or torch.full((2,), .02)
    c.frame = lambda stamp, distance: batch([row(c.config, i, stamp.sdk_step) for i in range(2)])[1]
    return c


def begin(c):
    actions = torch.zeros(2, 24)
    c.begin_action_window(actions, token="student-token", policy_uids=("student:7",) * 2, end=8)
    return actions


class Capture:
    def __init__(self):
        self.events, self.resets = [], []
    def on_measurement(self, event):
        self.events.append(event)
        return {i: {"image": torch.full((1, 2, 3), float(i)), "phase": event.phase} for i in (0, 1)}
    def reset_selected(self, env_ids, event):
        self.resets.append((tuple(env_ids), event.stamp.reset_ids))


def test_shared_module_import_does_not_load_tensor_teacher_or_ppo():
    implementation()
    code = ("import sys; import " + MODULE + "; "
        "assert 'go2_pvcnn.control.dh116.tabletop_tensor_native_teacher' not in sys.modules; "
        "assert 'go2_pvcnn.control.dh116.tabletop_device_ppo' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout


def test_student_actions_use_same_ordered_sdk_chain_and_original_reward_identity(monkeypatch):
    c = physical_fixture(monkeypatch)
    actions = begin(c)
    c.advance_action_window(actions, 8)
    expected = ["prepare", "reaction", "submit", "sdk", "update", "read", "clearance", "source_guard", "mesh", "rod", "finish", "distance"]
    assert c.events == expected * 8
    c.record_window_request()
    packet = c.complete_action_window(buffer_full=torch.ones(2, dtype=torch.bool))
    assert packet.policy_uids == ("student:7",) * 2
    assert packet.tokens == ("student-token",) * 2
    assert packet.substeps.tolist() == [8, 8]
    assert c.controls == 1 and c.runtime.physics_step == 8
    assert not any(hasattr(c, name) for name in ("model", "trainer", "critic"))
    measured = c.measurement("inspection")
    assert measured.stamp.sdk_step == 8
    measured.state["q"].fill_(42)
    assert not c.state["q"].any()
    assert not hasattr(measured, "reference") and not hasattr(measured, "critic")


def test_capture_defaults_to_window_owned_snapshots_and_selected_history_reset(monkeypatch):
    c = physical_fixture(monkeypatch)
    capture = Capture()
    c.attach_capture(capture)
    actions = begin(c)
    c.advance_action_window(actions, 8)
    c.record_window_request()
    c.complete_action_window(truncated=torch.tensor([False, True]))
    assert [(e.phase, e.stamp.sdk_step) for e in capture.events] == [("pre", 0), ("post", 8)]
    capture.events[0].state["q"].fill_(99)
    assert not c.state["q"].any()
    assert len(c.capture_history(0)) == len(c.capture_history(1)) == 2
    returned = c.capture_history(0)
    returned[0]["image"].fill_(100)
    assert not c.capture_history(0)[0]["image"].any()
    # Exercise the real Reset transaction with only physical operations stubbed.
    c.peer_state = lambda peers: {"q": c.state["q"][list(peers)].clone()}
    c.source.previous_request_tokens = ("peer-token", "student-token")
    c.runtime.solver.mesh_state_snapshot = lambda: {}
    c.effort = 1.
    def restore(*args, **kwargs):
        assert kwargs["env_ids"] == [1]
        return c.state
    monkeypatch.setattr(implementation(), "restore_regex_seed", restore)
    monkeypatch.setattr(implementation(), "mesh_reset", lambda *a, **k: None)
    def reset_lanes(ids, episode_uids):
        assert tuple(ids) == (1,)
        c.runtime.episodes[1] = episode_uids[0]
        c.runtime.resets[1] += 1
        c.runtime.ticks[1] = 0
    c.runtime.reset_lanes = reset_lanes
    c.ctrl.invalidate = lambda ids: None
    c._reset_refs = lambda ids: None
    c.reset((1,))
    assert len(c.capture_history(0)) == 2 and c.capture_history(1) == ()
    assert capture.resets == [((1,), (0, 1))]
    assert c.runtime.physics_step == c.sim.current_time_step_index == 8


@pytest.mark.parametrize("violation", ["clock", "transaction", "physical_tensor", "exception"])
def test_capture_violation_is_fatal_and_never_advances_an_sdk(monkeypatch, violation):
    c = physical_fixture(monkeypatch)
    capture = Capture()
    def broken(event):
        if violation == "clock": c.sim.current_time_step_index += 1
        elif violation == "transaction": c.ctrl.epoch += 1
        elif violation == "physical_tensor": c.state["q"].fill_(3)
        else: raise RuntimeError("capture failed")
        return {0: None, 1: None}
    capture.on_measurement = broken
    c.attach_capture(capture)
    with pytest.raises((ValueError, RuntimeError), match="capture"):
        begin(c)
    assert "sdk" not in c.events
    with pytest.raises(ValueError, match="failed"):
        c.reset((1,))


def test_guard_failure_cannot_be_swallowed_by_capture_and_midwindow_reset_is_forbidden(monkeypatch):
    c = physical_fixture(monkeypatch)
    capture = Capture()
    c.attach_capture(capture, cadence="substep")
    actions = begin(c)
    with pytest.raises(ValueError, match="settled"):
        c.reset((1,))
    # A rejected Reset poisons this scope, so use a new producer for guard test.
    c = physical_fixture(monkeypatch)
    capture = Capture()
    c.attach_capture(capture, cadence="substep")
    actions = begin(c)
    def guard(): raise ValueError("original source guard failed")
    c.source_guards = guard
    with pytest.raises(ValueError, match="original source guard"):
        c.sdk(actions, 8)
    assert c.runtime.physics_step == 0 and c.sim.current_time_step_index == 1
    assert [e.phase for e in capture.events] == ["pre"]
    with pytest.raises(ValueError, match="failed"):
        c.sdk(actions, 8)


def test_shared_timeout_keeps_original_truncation_and_blocks_unreset_lanes(monkeypatch):
    c = physical_fixture(monkeypatch)
    c.ages.fill_(31)
    actions = begin(c)
    c.advance_action_window(actions, 8)
    packet = c.complete_action_window(buffer_full=True)
    assert packet.truncated.tolist() == [True, True]
    assert packet.done.tolist() == [True, True] and not packet.buffer_full.any()
    assert c._reset_allowed == (0, 1)
    with pytest.raises(ValueError, match="Reset is unfinished"):
        c.begin_action_window(actions, token="next", policy_uids=("student:8",)*2, end=16)


def test_startup_measurement_excludes_seed_reference_and_critic_fields(monkeypatch):
    c = physical_fixture(monkeypatch)
    # Real startup state aliases the complete initial packet until first SDK.
    forbidden = ("root_seed_w", "seed_q", "seed_qd", "seed_root_pose_w", "critic_context", "previous_request")
    c.state.update({key: torch.ones(2, 1) for key in forbidden})
    event = c.measurement("pre")
    assert not set(forbidden).intersection(event.state)
    assert "q" in event.state and "link_pose_xyzw" in event.state
