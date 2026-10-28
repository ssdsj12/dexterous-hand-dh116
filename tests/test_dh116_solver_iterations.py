"""Keep the contact-solver fix in scene creation and checkpoint provenance."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def test_spawn_authors_requested_velocity_iterations_before_physx_creation():
    tree = ast.parse((ROOT/'src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py').read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'ArticulationRootPropertiesCfg')
    # Evaluate the actual config call using a constructor stand-in; no Isaac
    # runtime or source-text matching is needed to inspect the authored values.
    for count in (None, 4, 16, 64):
        args = SimpleNamespace(position_iterations=64)
        if count is not None:
            args.velocity_iterations = count
        cfg = eval(compile(ast.Expression(call), '<scene config>', 'eval'),
                   dict(args=args,
                        sim_utils=SimpleNamespace(ArticulationRootPropertiesCfg=lambda **kw: kw)))
        assert cfg['solver_velocity_iteration_count'] == (4 if count is None else count)
        assert cfg['solver_position_iteration_count'] == 64


def test_gpu_cli_accepts_explicit_velocity_convergence_probe():
    import argparse
    tree = ast.parse((ROOT/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and n.args and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == '--velocity-iterations')
    parser = argparse.ArgumentParser()
    eval(compile(ast.Expression(call), '<velocity argument>', 'eval'), dict(parser=parser))
    assert parser.parse_args(['--velocity-iterations', '64']).velocity_iterations == 64
    assert parser.parse_args([]).velocity_iterations == 4


def test_only_validated_solver_migration_is_allowed():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    old = dict(physics_hz=480, solver_type=0, position_iterations=64, num_envs=4096)
    new = dict(old, velocity_iterations=16)
    assert checkpoint_contract_changes(old, new) == ['contact_solver_velocity_4_to_16']
    assert checkpoint_contract_changes(new, new) == []
    for bad in (dict(new, velocity_iterations=8), dict(new, velocity_iterations=64), dict(new, position_iterations=128),
                dict(new, physics_hz=240), dict(new, num_envs=1024), dict(new, solver_type=1)):
        with pytest.raises(ValueError, match='contract mismatch'):
            checkpoint_contract_changes(old, bad)
    with pytest.raises(ValueError, match='contract mismatch'):
        checkpoint_contract_changes(new, old)


@pytest.mark.parametrize('enabled', [False, True])
def test_gpu_scene_receives_explicit_articulation_solver_order(enabled):
    tree = ast.parse((ROOT/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'PhysxCfg')
    cfg = eval(compile(ast.Expression(call), '<physics config>', 'eval'),
               dict(args=SimpleNamespace(solver_type=0, solve_articulation_contact_last=enabled),
                    sim_utils=SimpleNamespace(PhysxCfg=lambda **kw: kw)))
    assert cfg.get('solve_articulation_contact_last') is enabled
    assert cfg['solver_type'] == 0


def test_solver_order_readback_rejects_missing_or_unapplied_setting():
    from go2_pvcnn.control.dh116.tabletop_solver_order import verify_solver_order
    for enabled in (False, True):
        context = SimpleNamespace(get_solve_articulation_contact_last=lambda: enabled)
        assert verify_solver_order(context, enabled) == dict(
            requested=enabled, actual=enabled, verified=True)
        for actual in (None, not enabled):
            context = SimpleNamespace(get_solve_articulation_contact_last=lambda: actual)
            with pytest.raises(RuntimeError, match='solver order'):
                verify_solver_order(context, enabled)


def test_solver_order_is_checkpointed_and_cannot_silently_migrate():
    from go2_pvcnn.control.dh116.tabletop_solver_order import CONTRACT
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    tree = ast.parse((ROOT/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Attribute)
                 and n.test.attr == 'solve_articulation_contact_last')
    for enabled in (False, True):
        contract = dict(physics_hz=480, num_envs=4096)
        scope = dict(args=SimpleNamespace(solve_articulation_contact_last=enabled), contract=contract)
        exec(compile(ast.Module(body=[block], type_ignores=[]), '<contract>', 'exec'), scope)
        assert contract.get('articulation_solver_order') == (CONTRACT if enabled else None)
    old = dict(physics_hz=480, num_envs=4096)
    new = dict(old, articulation_solver_order=CONTRACT)
    assert checkpoint_contract_changes(new, new) == []
    for before, after in ((old, new), (new, old)):
        with pytest.raises(ValueError, match='contract mismatch'):
            checkpoint_contract_changes(before, after)
