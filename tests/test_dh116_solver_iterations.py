"""CLI and wiring checks; not evidence of physical convergence."""
import argparse
import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/verify_fixed_dual_panda_dh116.py'


def iteration_parser():
    tree = ast.parse(SOURCE.read_text())
    statements = [node for node in tree.body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call)
                  and any(isinstance(arg, ast.Constant) and
                          arg.value == '--solver-position-iterations'
                          for arg in node.value.args)]
    assert len(statements) == 1, 'Missing explicit position iteration option'
    parser = argparse.ArgumentParser()
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(SOURCE), 'exec'),
         {'parser': parser})
    return parser


@pytest.mark.parametrize('count', [1, 4, 16])
def test_position_iteration_diagnostic_options(count):
    assert iteration_parser().parse_args(
        ['--solver-position-iterations', str(count)]).solver_position_iterations == count


def test_default_preserves_sixteen():
    assert iteration_parser().parse_args([]).solver_position_iterations == 16


@pytest.mark.parametrize('count', ['0', '-1', '256', '1.5'])
def test_invalid_iterations_rejected(count):
    with pytest.raises(SystemExit):
        iteration_parser().parse_args(['--solver-position-iterations', count])


def test_iteration_count_reaches_articulation_and_evidence():
    tree = ast.parse(SOURCE.read_text())
    values = {name: [] for name in ('solver_position_iteration_count', 'solver_position_iterations')}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in values:
            values[node.arg].append(ast.unparse(node.value))
    assert values['solver_position_iteration_count'] == ['args.solver_position_iterations']
    assert 'args.solver_position_iterations' in values['solver_position_iterations']
