"""Cumulative diagnostic pilot budgets, without a simulator or Torch."""
import copy
import importlib

import pytest


def budget(saved=None, **changes):
    module = importlib.import_module('go2_pvcnn.control.dh116.tabletop_surface_student_budget')
    helper = getattr(module, 'diagnostic_budget', None)
    assert callable(helper), 'diagnostic_budget helper must be implemented'
    options = dict(requested_max_iterations=4, requested_timesteps=None,
                   episodes_per_env=None, num_envs=2, rollout_steps=32,
                   admitted=False, limit=64)
    options.update(changes)
    return helper(saved, **options)


def test_diagnostic_budget_module_exists():
    name = 'go2_pvcnn.control.dh116.tabletop_surface_student_budget'
    assert importlib.util.find_spec(name) is not None, 'pure diagnostic budget module is required'


def test_fresh_pilot_accepts_exact_limit_and_returns_already_used_updates():
    assert budget(requested_max_iterations=64) == 0
    with pytest.raises(ValueError, match='limit|budget'):
        budget(requested_max_iterations=65)


def test_four_update_resumes_share_one_lifetime_limit():
    for iteration in range(4, 64, 4):
        assert budget({'iteration': iteration}, requested_max_iterations=iteration + 4) == iteration
    with pytest.raises(ValueError, match='limit|budget'):
        budget({'iteration': 64}, requested_max_iterations=68)


def test_explicit_diagnostic_counter_takes_priority_and_saved_stays_unchanged():
    saved = dict(iteration=1000, timesteps=999999,
                 surface_student_diagnostic_updates=60, metadata={'preserved': [1, 2]})
    before = copy.deepcopy(saved)
    assert budget(saved, requested_max_iterations=1004) == 60
    with pytest.raises(ValueError):
        budget(saved, requested_max_iterations=1005)
    assert saved == before


def test_noop_targets_and_missing_saved_counts():
    assert budget({}) == 0
    assert budget({'iteration': 64}, requested_max_iterations=60) == 64
    assert budget({'iteration': 64}, requested_max_iterations=64) == 64
    with pytest.raises(ValueError):
        budget({'iteration': 65}, requested_max_iterations=65)


@pytest.mark.parametrize('target,delta', [(0, 0), (64, 1), (65, 2), (4096, 64)])
def test_timesteps_use_ceiling_of_vector_rollouts(target, delta):
    assert budget(requested_max_iterations=None, requested_timesteps=target, limit=delta) == 0
    if delta:
        with pytest.raises(ValueError):
            budget(requested_max_iterations=None, requested_timesteps=target, limit=delta - 1)


def test_timestep_resume_subtracts_saved_timesteps_before_ceiling():
    saved = dict(iteration=20, timesteps=1000, surface_student_diagnostic_updates=62)
    assert budget(saved, requested_max_iterations=None, requested_timesteps=1128) == 62
    with pytest.raises(ValueError):
        budget(saved, requested_max_iterations=None, requested_timesteps=1129)
    assert budget(saved, requested_max_iterations=None, requested_timesteps=999) == 62


def test_iteration_target_has_precedence_over_timestep_target():
    assert budget(requested_max_iterations=4, requested_timesteps=10**30) == 0


def test_timestep_ceiling_preserves_large_integer_precision():
    saved = dict(iteration=0, timesteps=10**30, surface_student_diagnostic_updates=63)
    assert budget(saved, requested_max_iterations=None, requested_timesteps=10**30 + 64) == 63
    with pytest.raises(ValueError):
        budget(saved, requested_max_iterations=None, requested_timesteps=10**30 + 65)


def test_unadmitted_episode_or_unbounded_budgets_are_rejected():
    with pytest.raises(ValueError, match='episode'):
        budget(episodes_per_env=1)
    with pytest.raises(ValueError, match='budget|finite'):
        budget(requested_max_iterations=None)


def test_admitted_student_is_not_limited_to_diagnostic_updates():
    saved = dict(iteration=100, surface_student_diagnostic_updates=64)
    assert budget(saved, requested_max_iterations=10**30, admitted=True) == 64
    assert budget(saved, requested_max_iterations=None, episodes_per_env=2048, admitted=True) == 64
    assert budget(saved, requested_max_iterations=None, admitted=True) == 64


@pytest.mark.parametrize('field,value', [
    ('iteration', -1), ('iteration', True), ('iteration', 4.),
    ('timesteps', -1), ('timesteps', '64'),
    ('surface_student_diagnostic_updates', -1),
    ('surface_student_diagnostic_updates', True),
    ('surface_student_diagnostic_updates', 4.),
])
def test_invalid_saved_counts_are_rejected_even_for_admitted_student(field, value):
    for admitted in (False, True):
        with pytest.raises(ValueError):
            budget({field: value}, admitted=admitted)


@pytest.mark.parametrize('changes', [
    dict(requested_max_iterations=-1), dict(requested_max_iterations=True),
    dict(requested_max_iterations=4.), dict(requested_timesteps=-1),
    dict(requested_timesteps=float('inf')), dict(episodes_per_env=0),
    dict(episodes_per_env=True), dict(num_envs=0), dict(num_envs=True),
    dict(rollout_steps=0), dict(rollout_steps=32.), dict(limit=-1),
    dict(limit=True), dict(admitted='yes'),
])
def test_invalid_requested_counts_and_dimensions_are_rejected(changes):
    with pytest.raises(ValueError):
        budget(**changes)


@pytest.mark.parametrize('saved', [False, [], 'checkpoint'])
def test_saved_must_be_a_checkpoint_dictionary_or_none(saved):
    with pytest.raises(ValueError):
        budget(saved)
