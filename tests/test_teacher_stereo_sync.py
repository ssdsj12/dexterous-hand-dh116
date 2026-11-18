import ast
from pathlib import Path

import pytest

from go2_pvcnn.control.dh116.tabletop_teacher_grasp_preview_runtime import compile_runtime
from go2_pvcnn.control.dh116.tabletop_teacher_stereo_sync import ActionBoundary, sync_row, timing_summary

ROOT = Path(__file__).resolve().parents[1]


def test_opt_in_observer_brackets_actual_sdk_integration():
    import hashlib

    source = (ROOT / "scripts/train_dh116_gpu_teacher_ppo.py").read_text()
    sha = hashlib.sha256(source.encode()).hexdigest()
    base = compile_runtime(source, sha)
    observed = compile_runtime(source, sha, observe_actions=True)
    step = next(node for node in observed.body[0].body if isinstance(node, ast.FunctionDef)
                and node.name == "step_env")
    substeps = next(node for node in step.body if isinstance(node, ast.For))
    lines = [ast.unparse(node) for node in substeps.body]
    at = lines.index("sim.step(render=False)")
    assert lines[at-2:at] == ["right_hands.write_data_to_sim()",
                               "if _sub == 0:\n    step_observer('issued', sim, act)"]
    assert lines[at+1] == "if _sub == 0:\n    step_observer('integrated', sim, act)"
    del substeps.body[at+1]
    del substeps.body[at-1]
    observed.body[0].args.args.pop()
    assert ast.dump(observed) == ast.dump(base)


def example(tick, rgb_offset=0.0, act_skew=0.0, wall_ns=100_000_000):
    step = 2 + tick * 8
    stamp = step / 480
    camera = dict(render_frame_number=tick+1, simulation_time=stamp+rgb_offset)
    receipt = {"identity": {"policy_tick": tick, "sdk_step": step, "sdk_timestamp_s": stamp},
               "render_timing": {"capture_id": tick+1, "raw_left_clock": camera,
                                 "raw_right_clock": dict(camera)}}
    issued = dict(sdk_step=step, sim_time_s=stamp+act_skew, wall_ns=wall_ns+5_000_000)
    integrated = dict(sdk_step=step+1, sim_time_s=stamp+1/480, wall_ns=wall_ns+7_000_000)
    return sync_row(receipt, issued, integrated, capture_start_ns=wall_ns-1_000_000,
                    rgb_ready_ns=wall_ns, final_step=step+8,
                    actor_action=[.2]*24, applied_action=[.2]*24)


def test_raw_rgb_skew_and_wall_jitter_are_measured_separately():
    rows = [example(0), example(1, wall_ns=116_000_000),
            example(2, rgb_offset=.0021, wall_ns=149_000_000)]
    out = timing_summary(rows)
    assert out["rgb_phys_under_2ms"] is False
    assert out["rgb_phys_abs_ms"]["max"] == pytest.approx(2.1)
    assert out["frame_wall_interval_ms"]["min"] == pytest.approx(16)
    assert out["frame_wall_interval_ms"]["max"] == pytest.approx(33)
    assert out["action_sim_offset_constant"] is True
    assert out["action_same_sdk_boundary"] is True
    assert out["reprojection_px"] is None


def test_same_sdk_tick_does_not_hide_a_timestamp_shift():
    r = example(0, act_skew=.001)
    assert timing_summary([r])["action_same_sdk_boundary"] is False
    assert timing_summary([r])["action_sim_offset_constant"] is None


def test_action_hook_rejects_wrong_issue_order():
    from types import SimpleNamespace

    probe = ActionBoundary()
    sim = SimpleNamespace(current_time_step_index=2, current_time=2/480)
    with pytest.raises(ValueError, match="action hook order"):
        probe("integrated", sim, None)

def test_freeze_records_one_mode_and_hard_control_budget(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts import record_dh116_teacher_stereo_sync as recorder

    path = tmp_path / recorder.MODEL_CONFIG
    path.parent.mkdir(parents=True)
    model = tmp_path / "model"
    model.write_bytes(b"frozen checkpoint bytes")
    path.write_text(json.dumps(dict(checkpoints={"tripod": "model", "pinch": "model"},
        checkpoint_sha256={mode: hashlib.sha256(model.read_bytes()).hexdigest()
                           for mode in ("tripod", "pinch")},
        source_sha256="teacher-source")))
    script = tmp_path/"scripts/record_dh116_teacher_stereo_sync.py"
    script.parent.mkdir()
    script.write_bytes(b"frozen recorder bytes")
    monkeypatch.setattr(recorder, "ROOT", tmp_path)
    monkeypatch.setattr(recorder, "ORIGINAL", tmp_path)
    monkeypatch.setattr(recorder.stereo, "freeze_plan", lambda root, output, bindings: dict(
        output=str(output), teacher_source_sha256="teacher-source", budget=dict(ppo_updates=0, soft_wall_seconds=180),
        frozen_files={}, teacher_stereo={}))
    plan = recorder.freeze_plan(tmp_path/"out", "tripod", 500)
    assert plan["budget"]["max_total_sdk_steps"] == 4002
    assert plan["budget"]["capture_wall_seconds"] == 10
    assert plan["budget"]["solver_steps"] == 48000
    assert plan["budget"]["ppo_updates"] == 0
    assert not plan["budget"]["webrtc"]
    recorder.check_plan(plan, recorder.legacy.digest(plan))
    live = recorder.freeze_plan(tmp_path/"live", "pinch", 16, live_webrtc=True)
    assert live["budget"]["webrtc"] is True
    assert live["budget"]["max_total_sdk_steps"] == 130
    assert live["budget"]["max_capture_renders"] == 64
    assert live["budget"]["ppo_updates"] == 0
    assert live["budget"]["live_hold_seconds"] == 60
    assert live["budget"]["connection_wall_seconds"] == 35
    assert live["budget"]["max_connection_updates"] == 700
    recorder.check_plan(live, recorder.legacy.digest(live))
    with pytest.raises(ValueError, match="at most 16"):
        recorder.freeze_plan(tmp_path/"live-long", "pinch", 17, live_webrtc=True)
    warmup = recorder.freeze_plan(tmp_path/"warmup", "pinch", 1, clock_warmup=True)
    assert warmup["clock_warmup_compare"]
    assert warmup["teacher_stereo"]["capture_pairs"] == 2
    assert warmup["budget"]["max_total_sdk_steps"] == 10
    assert warmup["budget"]["solver_steps"] == 96
    assert warmup["budget"]["teacher_queries"] == warmup["budget"]["model_loads"] == 0
    assert warmup["budget"]["max_capture_renders"] == 8
    recorder.check_plan(warmup, recorder.legacy.digest(warmup))
    with pytest.raises(ValueError, match="one Pinch control window"):
        recorder.freeze_plan(tmp_path/"invalid", "pinch", 2, clock_warmup=True)
    recheck = recorder.freeze_plan(tmp_path/"recheck", "pinch", 1, clock_warmup=True,
                                   reset_recheck=True)
    assert recheck["clock_reset_recheck"]
    assert recheck["teacher_stereo"]["capture_pairs"] == 3
    assert recheck["budget"]["episode_resets"] == 2
    assert recheck["budget"]["max_capture_renders"] == 12
    assert recheck["budget"]["max_total_sdk_steps"] == warmup["budget"]["max_total_sdk_steps"]
    recorder.check_plan(recheck, recorder.legacy.digest(recheck))
    with pytest.raises(ValueError, match="requires clock warmup"):
        recorder.freeze_plan(tmp_path/"invalid", "pinch", 1, reset_recheck=True)
    model.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint or source changed"):
        recorder.check_plan(plan, recorder.legacy.digest(plan))
    with pytest.raises(ValueError, match="at most 500"):
        recorder.freeze_plan(tmp_path/"invalid", "tripod", 501)



@pytest.mark.parametrize("sdk_steps,solver_steps,admitted", [
    (8, 96, True), (7, 96, False), (8, 95, False),
])
def test_zero_action_clock_window(sdk_steps, solver_steps, admitted):
    from types import SimpleNamespace

    from scripts import record_dh116_teacher_stereo_sync as recorder

    boundary = ActionBoundary()
    sim = SimpleNamespace(current_time_step_index=2, current_time=2/480, is_playing=lambda: True)
    rope = SimpleNamespace(time=0.)
    seen = []

    def step(action):
        seen.append(action.tolist())
        boundary("issued", sim, action)
        sim.current_time_step_index += 1
        sim.current_time += 1/480
        boundary("integrated", sim, action)
        sim.current_time_step_index += sdk_steps-1
        sim.current_time += (sdk_steps-1)/480
        rope.time += solver_steps/(480*12)

    rt = SimpleNamespace(sim=sim, rope=rope, device="cpu", step_env=step)
    if admitted:
        result = recorder._zero_action_clock_window(rt, boundary)
        assert result["sdk_after"]["sdk_step"] == 10
        assert result["solver_elapsed_s"] == pytest.approx(8/480)
    else:
        with pytest.raises(RuntimeError, match="frozen clock budget"):
            recorder._zero_action_clock_window(rt, boundary)
    assert seen == [[[0.]*24]]




@pytest.mark.parametrize("step_delta", [0, 1])
def test_second_reset_preserves_sdk_clock_and_captures_its_own_solver_origin(step_delta):
    from types import SimpleNamespace

    from scripts import record_dh116_teacher_stereo_sync as recorder

    sim = SimpleNamespace(current_time_step_index=10, current_time=10/480)
    rope = SimpleNamespace(time=8/480)
    def reset():
        sim.current_time_step_index += step_delta
        sim.current_time += step_delta / 480
        rope.time = 0.
    rt = SimpleNamespace(sim=sim, rope=rope, reset_all=reset)
    if step_delta:
        with pytest.raises(RuntimeError, match="reset advanced the SDK clock"):
            recorder._reset_clock_boundary(rt)
    else:
        result = recorder._reset_clock_boundary(rt)
        assert result["sdk_before"] == result["sdk_after"]
        assert result["solver_before_s"] == 8/480
        assert result["solver_after_s"] == 0.

def test_live_preview_keeps_clock_rejection_diagnostic():
    from scripts import record_dh116_teacher_stereo_sync as recorder

    skewed = {"student_sync_usable": False, "rgb_sdk_delta_ms": -4.166667}
    assert recorder._allow_teacher_step(skewed, live_webrtc=False) is False
    assert recorder._allow_teacher_step(skewed, live_webrtc=True) is True


def test_live_preview_prepares_two_separate_rgba_images():
    import torch
    from scripts import record_dh116_teacher_stereo_sync as recorder

    left = torch.zeros((1, 3, 2, 3))
    right = left.clone()
    right[:, 1] = 1
    images = recorder._preview_pixels(dict(left_rgb=left, right_rgb=right), 3, 2)
    assert images["left"].shape == images["right"].shape == (2, 3, 4)
    assert images["left"][0, 0].tolist() == [0, 0, 0, 255]
    assert images["right"][0, 0].tolist() == [0, 255, 0, 255]
    with pytest.raises(ValueError, match="calibrated RGB shape"):
        recorder._preview_pixels(dict(left_rgb=left, right_rgb=right[..., :2]), 3, 2)
