"""Source-wiring CPU checks, not native contact validation."""
import ast,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_supported_runtime_only_appends_support_wiring_and_preserves_action_limits():
    from go2_pvcnn.control.dh116.tabletop_teacher_supported_runtime import compile_supported_runtime
    source=(ROOT/"scripts/train_dh116_gpu_teacher_ppo.py").read_text()
    tree=compile_supported_runtime(source,hashlib.sha256(source.encode()).hexdigest())
    text=ast.unparse(tree)
    assert "bind_native_end_supports" in text
    assert "extend_mesh_pose_buffers" in text
    assert "support_plan['scene']['dz_m']" in text
    assert "0.3 * DT" in text and "0.01 * DT" in text
    assert "0.15" in text
    assert "def set_settled_lift_reference" in text
    assert "os._exit" not in text and ".backward(" not in text
    compile(tree,"supported-preview","exec")

def test_supported_runtime_rejects_changed_source_before_execution():
    import pytest
    from go2_pvcnn.control.dh116.tabletop_teacher_supported_runtime import compile_supported_runtime
    with pytest.raises(ValueError,match="changed"):
        compile_supported_runtime("pass","0"*64)


def test_settled_reference_can_clear_buffers_created_in_inference_mode():
    import torch
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_teacher_supported_runtime import compile_supported_runtime
    source=(ROOT/"scripts/train_dh116_gpu_teacher_ppo.py").read_text()
    tree=compile_supported_runtime(source,hashlib.sha256(source.encode()).hexdigest())
    setter=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="set_settled_lift_reference")
    body=ast.parse("""def probe():
    import torch
    from types import SimpleNamespace
    with torch.inference_mode():
        prev_act=torch.ones(3,24)
        step_in_ep=torch.ones(3,dtype=torch.long)
        hold10_s=torch.ones(3)
        hold30_s=torch.ones(3)
        stage1_awarded=torch.ones(3,dtype=torch.bool)
        stage2_awarded=torch.ones(3,dtype=torch.bool)
    init_med_z=None
    init_min_z=None
    rope_pos_view=torch.tensor([[[0.,0.,.3],[0.,0.,.4]]]*3)
    sim=SimpleNamespace(current_time_step_index=482,current_time=1.)
""").body[0]
    body.body.append(setter)
    body.body.extend(ast.parse("""result=set_settled_lift_reference()
assert not prev_act.any()
assert not step_in_ep.any()
return result""").body)
    module=ast.fix_missing_locations(ast.Module(body=[body],type_ignores=[]))
    ns={};exec(compile(module,"settle-inference-regression","exec"),ns)
    result=ns["probe"]()
    assert result["sdk_step"]==482
    assert len(result["initial_min_z_m"])==3
