import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_production_tree_has_bounded_entrypoints_and_tests():
    scripts = [p for p in (ROOT / 'scripts').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    tests = list((ROOT / 'tests').rglob('*.py'))
    assert len(scripts) == 6
    assert len(tests) == 15
    assert all(p.match('test_core_*.py') for p in tests)

def test_production_source_never_imports_repository_scripts_or_compiles_graphs():
    for p in (ROOT / 'src').rglob('*.py'):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('scripts')
            elif isinstance(node, ast.Import):
                assert all((not n.name.startswith('scripts') for n in node.names))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'compile':
                    assert not (isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch')

def test_no_retired_production_modules_remain():
    package=ROOT/"src/dh116"
    assert package.is_dir()
    assert {p.name for p in package.iterdir() if p.is_dir() and p.name!="__pycache__"}=={"env","physics","rl","hand","cuda_measurements"}
    assert [p.name for p in package.glob("*.py")]==["__init__.py"]
    sources = [p for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    assert len(sources) == 32
    assert not [p for p in package.rglob("*.py") if p.name.startswith(("tabletop_","hand_only_")) or "mpc" in p.name]
