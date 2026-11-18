import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_preview_warp_readback import adapt_pointer_function,nonowning_constructor,binding_scope

def test_modern_constructor_receives_nonowning_pointer_without_owner_keyword():
    calls=[]
    def modern(**kwargs):calls.append(kwargs);return kwargs
    legacy=nonowning_constructor(modern)
    result=legacy(ptr=123,shape=(3,),dtype="float32",device="cpu",owner=False)
    assert result["deleter"] is None and "owner" not in result
    assert result["ptr"]==123
    with pytest.raises(ValueError):legacy(ptr=123,owner=True)

def test_binding_scope_restores_original_after_readback_error():
    def original():pass
    def replacement():pass
    module=SimpleNamespace(_reshape_output_ptr=original)
    with pytest.raises(RuntimeError,match="read error"):
        with binding_scope(module,replacement):
            assert module._reshape_output_ptr is replacement
            raise RuntimeError("read error")
    assert module._reshape_output_ptr is original

def test_real_installed_isaac_wrapper_replays_against_current_warp_on_cpu():
    import warp as wp
    file=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.replicator.core-1.11.35+106.5.0.lx64.r.cp310/omni/replicator/core/scripts/utils/annotator_utils.py")
    if not file.is_file():pytest.skip("installed Isaac 4.5 source unavailable")
    tree=ast.parse(file.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_reshape_output_ptr")
    node.decorator_list=[]
    namespace=dict(wp=wp,np=np,carb=SimpleNamespace(profiler=SimpleNamespace(begin=lambda *a:None,end=lambda *a:None)))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),"exec"),namespace)
    raw=namespace["_reshape_output_ptr"];data=np.array([1.,2.,3.],np.float32)
    args=dict(ptr=data.ctypes.data,height=0,width=0,strides=(4,),buffer_size=data.nbytes,elem_count=1,
        source_device="cpu",target_device="cpu",wp_dtype=wp.float32,np_dtype=np.dtype(np.float32))
    with pytest.raises(TypeError,match="owner"):raw(**args)
    adapted=adapt_pointer_function(raw,wp)
    output=adapted(**args)
    value=output[0] if isinstance(output,tuple) else output
    assert np.array_equal(value,data)
    assert namespace["wp"] is wp
    del output,value;gc.collect()
    assert np.array_equal(data,[1.,2.,3.])

def test_owned_reader_scopes_compatibility_and_restores_on_rgb_exception(monkeypatch):
    from contextlib import contextmanager
    import go2_pvcnn.control.dh116.tabletop_preview_warp_readback as adapter
    from go2_pvcnn.control.dh116.tabletop_stereo_rgb_owned_native import OwnedNativeRGBPairReader
    from go2_pvcnn.control.dh116.tabletop_stereo_camera import _NativeRGBPairReader
    events=[]
    @contextmanager
    def scope():
        events.append("enter")
        try:yield
        finally:events.append("restore")
    def fail(self):events.append("read");raise RuntimeError("rgb failed")
    monkeypatch.setattr(adapter,"isaac45_rgb_readback_scope",scope)
    monkeypatch.setattr(_NativeRGBPairReader,"read",fail)
    reader=object.__new__(OwnedNativeRGBPairReader)
    with pytest.raises(RuntimeError,match="rgb failed"):reader.read()
    assert events==["enter","read","restore"]

def test_bundled_warp_uses_the_real_original_nonowning_call():
    import os
    import subprocess
    import sys

    ext = Path(sys.prefix) / "lib/python3.10/site-packages/isaacsim/extscache"
    bundled = ext / "omni.warp.core-1.5.0+lx64"
    script = """
import ast
import sys
import types
from pathlib import Path
import numpy as np
import warp as wp
from go2_pvcnn.control.dh116 import tabletop_preview_warp_readback as adapter

path = next(Path(sys.argv[1]).glob("omni.replicator.core-*/omni/replicator/core/scripts/utils/annotator_utils.py"))
module = types.ModuleType("omni.replicator.core.scripts.utils.annotator_utils")
module.__file__ = str(path)
module.wp, module.np = wp, np
module.carb = types.SimpleNamespace(profiler=types.SimpleNamespace(begin=lambda *a:None,end=lambda *a:None))
node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name=="_reshape_output_ptr")
node.decorator_list = []
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),vars(module))
sys.modules[module.__name__] = module
raw = module._reshape_output_ptr
source = np.array([1., 2., 3.], np.float32)
with adapter.isaac45_rgb_readback_scope():
    assert module._reshape_output_ptr is raw
    value = raw(ptr=source.ctypes.data,height=0,width=0,strides=(4,),buffer_size=source.nbytes,elem_count=1,
        source_device="cpu",target_device="cpu",wp_dtype=wp.float32,np_dtype=np.dtype(np.float32))
    result = value[0] if isinstance(value,tuple) else value
    assert np.array_equal(result,source)
assert module._reshape_output_ptr is raw
"""
    if not bundled.is_dir():
        pytest.skip("installed Isaac bundled Warp unavailable")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,(bundled, root/"src"))), CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-B", "-c", script, str(ext)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout
