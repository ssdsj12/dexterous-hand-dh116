import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def test_host_object_relative_keeps_precision_at_large_grid_coordinates():
    path = Path('src/go2_pvcnn/control/dh116/tabletop_gpu_isaac.py')
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'measure_batch')
    expression = next(k.value for n in ast.walk(method) if isinstance(n, ast.Call)
                      for k in n.keywords if k.arg == 'object_relative')
    roots = np.array([[[101.502094, 101.188617, .428541],
                       [101.522105, 100.802814, .419162]]], dtype=np.float32)
    wp = np.array([[101.512, 101., .308]], dtype=np.float32)
    actual = eval(compile(ast.Expression(expression), str(path), 'eval'),
                  {'np': np, 'self': SimpleNamespace(wp=wp, roots=roots)})
    expected = wp.astype(np.float64) - roots[:, :, :3].astype(np.float64).mean(axis=1)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
