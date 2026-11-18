"""Close seeds may refine the physics step without changing their provenance."""
from copy import deepcopy

import pytest


def provenance(physics_hz=240):
    return dict(pose_sha256='pose', source_sha256={'left': 'left', 'right': 'right'},
                physics_hz=physics_hz, control_hz=60, solver_type=0, position_iterations=16)


@pytest.mark.parametrize('physics_hz', [240, 480])
def test_matching_seed_provenance_needs_no_refinement_metadata(physics_hz):
    from go2_pvcnn.control.dh116.tabletop_vec_config import validate_seed_provenance
    source = dict(provenance(physics_hz), observation_version='historical_seed')
    assert validate_seed_provenance(source, provenance(physics_hz)) == {}


def test_only_supported_refinement_is_recorded_without_mutating_seed_contract():
    from go2_pvcnn.control.dh116.tabletop_vec_config import validate_seed_provenance
    source, requested = provenance(), provenance(480)
    before_source, before_requested = deepcopy(source), deepcopy(requested)
    assert validate_seed_provenance(source, requested) == {'seed_physics_hz': 240}
    assert source == before_source and requested == before_requested


@pytest.mark.parametrize('key,value', [
    ('pose_sha256', 'other_pose'),
    ('source_sha256', {'left': 'left', 'right': 'other_asset'}),
    ('control_hz', 120),
    ('solver_type', 1),
    ('position_iterations', 32),
])
def test_physics_refinement_does_not_permit_other_provenance_changes(key, value):
    from go2_pvcnn.control.dh116.tabletop_vec_config import validate_seed_provenance
    requested = provenance(480)
    requested[key] = value
    with pytest.raises(ValueError, match='provenance'):
        validate_seed_provenance(provenance(), requested)


@pytest.mark.parametrize('source_hz,requested_hz', [(480, 240), (240, 120),
                                                (240, 960), (120, 240), (240, 481)])
def test_reverse_or_other_frequency_changes_are_rejected(source_hz, requested_hz):
    from go2_pvcnn.control.dh116.tabletop_vec_config import validate_seed_provenance
    with pytest.raises(ValueError, match='provenance'):
        validate_seed_provenance(provenance(source_hz), provenance(requested_hz))


@pytest.mark.parametrize('missing_from', ['source', 'requested'])
def test_missing_required_provenance_cannot_bypass_validation(missing_from):
    from go2_pvcnn.control.dh116.tabletop_vec_config import validate_seed_provenance
    source, requested = provenance(), provenance(480)
    del (source if missing_from == 'source' else requested)['solver_type']
    with pytest.raises(ValueError, match='provenance'):
        validate_seed_provenance(source, requested)


def test_extra_requested_provenance_is_compared_but_source_training_fields_are_not():
    from go2_pvcnn.control.dh116.tabletop_vec_config import validate_seed_provenance
    source = dict(provenance(), observation_size=104, future_physics_setting='same')
    requested = dict(provenance(480), future_physics_setting='same')
    assert validate_seed_provenance(source, requested) == {'seed_physics_hz': 240}
    requested['future_physics_setting'] = 'changed'
    with pytest.raises(ValueError, match='provenance'):
        validate_seed_provenance(source, requested)


def mimic_options(**overrides):
    result = dict(physics_hz=240, mimic_frequency=200, initialize_from=None,
                  resume=None, allow_mimic_refinement=False,
                  allow_physics_refinement=False)
    result.update(overrides)
    return result


def test_mimic_cli_defaults_preserve_200_and_require_an_explicit_refinement_flag():
    import argparse
    from go2_pvcnn.control.dh116.tabletop_vec_config import add_mimic_arguments
    parser = argparse.ArgumentParser()
    add_mimic_arguments(parser)
    default = parser.parse_args([])
    assert default.mimic_frequency == 200
    assert default.allow_mimic_refinement is False
    requested = parser.parse_args(['--mimic-frequency', '400', '--allow-mimic-refinement'])
    assert requested.mimic_frequency == 400
    assert requested.allow_mimic_refinement is True


@pytest.mark.parametrize('frequency', ['100', '300', '800', 'nan'])
def test_mimic_cli_rejects_unvalidated_frequencies(frequency):
    import argparse
    from go2_pvcnn.control.dh116.tabletop_vec_config import add_mimic_arguments
    parser = argparse.ArgumentParser()
    add_mimic_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(['--mimic-frequency', frequency])


def test_rigid_cli_requires_explicit_load_profile_validation():
    import argparse
    from go2_pvcnn.control.dh116.tabletop_vec_config import add_mimic_arguments, validate_mimic_options
    parser = argparse.ArgumentParser()
    add_mimic_arguments(parser)
    args = parser.parse_args(['--mimic-frequency', '0'])
    with pytest.raises(ValueError):
        validate_mimic_options(physics_hz=480,mimic_frequency=args.mimic_frequency)
    validate_mimic_options(physics_hz=480,mimic_frequency=args.mimic_frequency,allow_rigid=True)


@pytest.mark.parametrize('overrides', [
    {},
    dict(physics_hz=480),
    dict(physics_hz=480, mimic_frequency=400),
    dict(physics_hz=480, mimic_frequency=400, resume='same_contract_checkpoint.zip'),
    dict(physics_hz=480, mimic_frequency=400, initialize_from='same_contract_checkpoint.zip'),
    dict(physics_hz=480, mimic_frequency=400, initialize_from='source_200.zip',
         allow_mimic_refinement=True),
    dict(physics_hz=480, initialize_from='source_240hz.zip', allow_physics_refinement=True),
])
def test_valid_mimic_runtime_and_explicit_transfer_options(overrides):
    from go2_pvcnn.control.dh116.tabletop_vec_config import validate_mimic_options
    options = mimic_options(**overrides)
    before = deepcopy(options)
    assert validate_mimic_options(**options) is None
    assert options == before


@pytest.mark.parametrize('overrides,error', [
    (dict(mimic_frequency=400), '480'),
    (dict(mimic_frequency=300), '200.*400'),
    (dict(physics_hz=480, mimic_frequency=400, allow_mimic_refinement=True), 'initialize'),
    (dict(physics_hz=480, mimic_frequency=400, allow_mimic_refinement=True,
          resume='checkpoint.zip'), 'initialize'),
    (dict(physics_hz=480, mimic_frequency=400, allow_mimic_refinement=True,
          resume='checkpoint.zip', initialize_from='source.zip'), 'resume'),
    (dict(physics_hz=480, mimic_frequency=400, allow_mimic_refinement=True,
          allow_physics_refinement=True, initialize_from='source.zip'), 'mutually exclusive'),
    (dict(physics_hz=480, allow_mimic_refinement=True,
          initialize_from='source.zip'), '400'),
])
def test_invalid_mimic_options_fail_before_simulation(overrides, error):
    from go2_pvcnn.control.dh116.tabletop_vec_config import validate_mimic_options
    with pytest.raises(ValueError, match=error):
        validate_mimic_options(**mimic_options(**overrides))
