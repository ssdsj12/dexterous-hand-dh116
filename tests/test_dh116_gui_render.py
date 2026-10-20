import ast
from pathlib import Path
from types import SimpleNamespace
from go2_pvcnn.control.dh116.render_schedule import render_due


def test_physics_step_renders_in_gui_but_not_headless():
    source = Path(__file__).parents[1] / 'scripts/verify_fixed_dual_panda_dh116.py'
    tree = ast.parse(source.read_text())
    steps = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'step'
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'sim']
    assert len(steps) == 1
    expr = next(k.value for k in steps[0].keywords if k.arg == 'render')
    assert ast.unparse(expr) == 'do_render'
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'do_render' for t in n.targets)]
    assert len(assignments) == 1
    compiled = compile(ast.Expression(assignments[0].value), '<render-flag>', 'eval')
    for headless in (False, True):
        for interval in (1, 10):
            actual = [eval(compiled, {'args': SimpleNamespace(headless=headless, render_every=interval),
                                     'step': s, 'render_due': render_due}) for s in range(31)]
            assert actual == [not headless and s % interval == 0 for s in range(31)]
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.target) == 'step')
    assert any(isinstance(n, ast.Expr) and n.value is steps[0] for n in loop.body), 'Physics must remain unconditional'


def test_loop_wire_visual_publication_is_only_on_render_steps():
    source = Path(__file__).parents[1] / 'scripts/verify_fixed_dual_panda_dh116.py'
    tree = ast.parse(source.read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.target) == 'step')
    publications = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'publish_wire'
                    and ast.unparse(n.args[0]) == 'snapshot']
    assert len(publications) == 1
    guarded = [n for n in ast.walk(loop) if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'do_render'
               and any(child is publications[0] for item in n.body for child in ast.walk(item))]
    assert len(guarded) == 1, 'visual mesh update must be guarded by the render schedule'
    assignments = [n for n in ast.walk(loop) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'do_render' for t in n.targets)]
    assert assignments[0].lineno < guarded[0].lineno
