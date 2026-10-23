import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _source():
    return (ROOT / "scripts/preview_native_dual_panda_dh116.py").read_text()


def test_native_runner_exposes_reference_views_and_sequence_mode():
    source = _source()
    assert "--capture-views" in source
    assert "--sequence" in source
    assert "camera_preset" in source
    assert "reference_wire_points" in source


def test_native_runner_has_explicit_synchronized_lift_path():
    source = _source()
    tree = ast.parse(source)
    assert "apply_root_delta_z" in source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_root_delta_z"
        for node in ast.walk(tree)
    )


def test_integrated_runner_uses_shared_perspective_camera():
    source = (ROOT / "scripts/verify_fixed_dual_panda_dh116.py").read_text()
    assert "camera_preset" in source
    assert 'camera_preset("perspective")' in source

