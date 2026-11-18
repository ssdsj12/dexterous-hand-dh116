from pathlib import Path
import go2_pvcnn.control.dh116.stage_runtime as runtime

ROOT=Path(__file__).resolve().parents[1]

def test_import_is_from_new_source():
    assert Path(runtime.__file__).resolve().is_relative_to(ROOT/'src')

def test_runner_backend_and_module_pins_are_local():
    code=(ROOT/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert '/home/xk/coding/DeformX' not in code
    assert "'src/go2_pvcnn/control/dh116'" in code

def test_no_load_uses_current_manifest_digest():
    code=(ROOT/'scripts/verify_dh116_no_load.py').read_text()
    assert "pins = dict(left=" not in code
    assert "record['sha256']" in code
