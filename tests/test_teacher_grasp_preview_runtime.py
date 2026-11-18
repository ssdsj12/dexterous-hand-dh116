import ast,hashlib
from pathlib import Path
import pytest
from go2_pvcnn.control.dh116.tabletop_teacher_grasp_preview_runtime import compile_runtime

ROOT=Path(__file__).resolve().parents[1]


def test_preview_extraction_has_real_environment_step_and_per_env_seeds():
    source=(ROOT/"scripts/train_dh116_gpu_teacher_ppo.py").read_text()
    tree=compile_runtime(source,hashlib.sha256(source.encode()).hexdigest())
    text=ast.unparse(tree)
    assert "def step_env" in text and "rope.step_free" in text
    assert "seed_q[hi].clone()" in text and "preview_seeds" in text
    assert "device = args.device" in text
    assert "backward" not in text and "os._exit" not in text
    compile(tree,"preview-runtime","exec")


def test_preview_rejects_changed_teacher_source_before_execution():
    with pytest.raises(ValueError,match="changed"):
        compile_runtime("print('changed')","0"*64)


def test_preview_seeds_preserve_training_roots_and_passive_coupling():
    import json,numpy as np
    from go2_pvcnn.control.dh116.tabletop_teacher_grasp_preview_runtime import load_seeds
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116.linear_coupling import expand
    seeds=load_seeds(ROOT)
    original=builder.load_pair_proposal()["reset_seed"]
    np.testing.assert_array_equal(seeds[0]["roots"],original["reset_root_pose_w"])
    np.testing.assert_array_equal(seeds[0]["joints"],original["joint_positions"])
    styles=json.loads((ROOT/"configs/dh116_training/gpu_teacher_style_seeds_v1.json").read_text())["styles"]
    for i,mode in enumerate(("tripod","power"),1):
        for hand,side in enumerate(("left","right")):
            np.testing.assert_array_equal(seeds[i]["roots"][hand],styles[mode][side]["base_pose_wxyz"])
            np.testing.assert_array_equal(seeds[i]["joints"][hand],expand(styles[mode][side]["active_q_rad"]))


def test_preview_button_switches_view_without_mutating_physics_or_models():
    source=(ROOT/"scripts/view_dh116_teacher_grasps_webrtc.py").read_text()
    tree=ast.parse(source)
    funcs={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    select=ast.unparse(funcs["select_view"])
    assert "reset_all" not in select and "set_camera_view" in select
    assert "selected" in select
    runtime=(ROOT/"scripts/train_dh116_gpu_teacher_ppo.py").read_text()
    extracted=ast.unparse(compile_runtime(runtime,hashlib.sha256(runtime.encode()).hexdigest()))
    assert "use_fabric=False" not in extracted
    assert "0.3 * DT" in extracted and "0.01 * DT" in extracted


def test_viewer_drives_only_isolated_visual_copy():
    text=(ROOT/"scripts/view_dh116_teacher_grasps_webrtc.py").read_text()
    assert "ReadOnlyHandDisplay" in text
    assert "display.sync" in text
    assert "set_physx_update_transformations_settings(" not in text
    assert "get_physx_interface" not in text


def test_display_export_uses_native_api_without_advancing_physics():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_teacher_grasp_preview_runtime import export_physics_display
    events=[]
    sim=SimpleNamespace(current_time_step_index=8,current_time=8/480,
        physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda:events.append("fk")))
    interface=SimpleNamespace(update_transformations=lambda *args:events.append(args))
    receipt=export_physics_display(sim,interface)
    assert events==["fk",(True,True,True,False)]
    assert receipt["physics_clock_unchanged"] is True


def test_display_export_rejects_unexpected_physics_advance():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_teacher_grasp_preview_runtime import export_physics_display
    sim=SimpleNamespace(current_time_step_index=8,current_time=8/480,
        physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda:None))
    def bad_export(*args):
        sim.current_time_step_index+=1
    with pytest.raises(RuntimeError,match="clock"):
        export_physics_display(sim,SimpleNamespace(update_transformations=bad_export))
