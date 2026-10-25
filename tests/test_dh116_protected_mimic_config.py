"""Argument checks for an opt-in failure lifecycle, before Isaac startup."""
import argparse
import pytest
from go2_pvcnn.control.dh116 import tabletop_vec_config as config


def test_protected_mimic_arguments_default_to_old_lifecycle():
    parser = argparse.ArgumentParser()
    config.add_protected_mimic_arguments(parser)
    args = parser.parse_args([])
    assert not args.protected_mimic_resets
    assert not args.allow_protected_mimic_init
    args = parser.parse_args(['--protected-mimic-resets', '--allow-protected-mimic-init'])
    assert args.protected_mimic_resets and args.allow_protected_mimic_init


@pytest.mark.parametrize('kwargs', [
    {},
    dict(protected_mimic_resets=True),
    dict(protected_mimic_resets=True, resume='model.zip'),
    dict(protected_mimic_resets=True, initialize_from='model.zip', allow_protected_mimic_init=True),
])
def test_valid_lifecycle_options(kwargs):
    config.validate_protected_mimic_options(physics_hz=480, mimic_frequency=200, **kwargs)


@pytest.mark.parametrize('overrides', [
    dict(physics_hz=240),
    dict(mimic_frequency=400),
    dict(allow_protected_mimic_init=True),
    dict(initialize_from='source.zip'),
    dict(initialize_from='source.zip', allow_protected_mimic_init=True, resume='other.zip'),
    dict(initialize_from='source.zip', allow_protected_mimic_init=True, allow_physics_refinement=True),
    dict(initialize_from='source.zip', allow_protected_mimic_init=True, allow_mimic_refinement=True),
    dict(protected_mimic_resets=False, initialize_from='source.zip', allow_protected_mimic_init=True),
])
def test_invalid_lifecycle_options_fail_before_launch(overrides):
    kwargs = dict(protected_mimic_resets=True, physics_hz=480, mimic_frequency=200)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        config.validate_protected_mimic_options(**kwargs)
