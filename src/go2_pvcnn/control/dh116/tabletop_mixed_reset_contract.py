"""Checkpoint provenance for two qualified reset styles in one PPO batch."""

from copy import deepcopy
import json

from .tabletop_surface_contract import semantic_sha256, validate_contract
from .tabletop_surface_imitation import validate_reset_family


_RESET_FIELDS = ('close_seed_sha256', 'grasp_initialization_sha256')


def _reset_hashes(contract):
    return {field: contract.get(field) for field in _RESET_FIELDS}


def _refresh_digest(contract):
    contract['surface_contract_sha256'] = semantic_sha256(contract)
    return contract


def _member(family, label):
    if not isinstance(label, str) or not label:
        raise ValueError('mixed reset requires a named reset family member')
    if not isinstance(family, list):
        raise ValueError('mixed reset requires a declared reset family')
    matches = [row for row in family if isinstance(row, dict) and row.get('label') == label]
    if len(matches) != 1 or not isinstance(matches[0].get('contract'), dict):
        raise ValueError(f'mixed reset requires exactly one contract for {label}')
    return matches[0]['contract']


def _same(left, right):
    # JSON distinguishes booleans, integers, and floats in checkpoint metadata.
    return (json.dumps(left, sort_keys=True, allow_nan=False)
            == json.dumps(right, sort_keys=True, allow_nan=False))


def mixed_reset_contract(contract, family, *, primary_label, secondary_label, secondary_sha256,
                         num_envs):
    """Bind the equal contiguous slot groups and seal the complete 371D contract."""
    validate_contract(contract)
    if 'mixed_reset' in contract:
        raise ValueError('mixed reset requires a single-style primary contract')
    if type(num_envs) is not int or num_envs < 2 or num_envs % 2:
        raise ValueError('mixed reset requires an even number of environments')
    if type(contract.get('num_envs')) is not int or contract['num_envs'] != num_envs:
        raise ValueError('mixed reset environment count differs from the primary contract')
    if primary_label == secondary_label:
        raise ValueError('mixed reset requires distinct primary and secondary labels')

    primary = _reset_hashes(_member(family, primary_label))
    secondary = _reset_hashes(_member(family, secondary_label))
    if primary != _reset_hashes(contract):
        raise ValueError('mixed reset primary initialization does not match its declared style')
    if secondary_sha256 != secondary['grasp_initialization_sha256'] or secondary == primary:
        raise ValueError('mixed reset secondary initialization does not match its declared style')

    secondary_contract = _refresh_digest(dict(contract, **secondary))
    validate_reset_family(family, contract, secondary_contract)
    mixed = deepcopy(contract)
    mixed['mixed_reset'] = dict(
        version='dh116_mixed_reset_v1', layout='contiguous_primary_then_secondary',
        groups=[
            dict(label=primary_label, num_envs=num_envs // 2, **primary),
            dict(label=secondary_label, num_envs=num_envs // 2, **secondary),
        ],
    )
    return _refresh_digest(mixed)


def single_reset_contract(contract):
    """Project the qualified primary reset for validation of existing 371D demonstrations."""
    validate_contract(contract)
    single = deepcopy(contract)
    if 'mixed_reset' in single:
        del single['mixed_reset']
        _refresh_digest(single)
    return single


def mixed_reset_resume_changes(saved, current, family):
    """Allow an exact mixed resume or an explicit single-to-mixed reset migration."""
    if 'mixed_reset' not in current:
        raise ValueError('mixed reset checkpoint cannot resume as single-style training')
    primary = single_reset_contract(current)
    record = current['mixed_reset']
    groups = record.get('groups') if isinstance(record, dict) else None
    if (not isinstance(groups, list) or len(groups) != 2
            or any(not isinstance(group, dict) for group in groups)):
        raise ValueError('mixed reset contract requires two ordered groups')

    expected = mixed_reset_contract(
        primary, family, primary_label=groups[0].get('label'),
        secondary_label=groups[1].get('label'),
        secondary_sha256=groups[1].get('grasp_initialization_sha256'),
        num_envs=current.get('num_envs'),
    )
    if not _same(current, expected):
        raise ValueError('mixed reset layout or group provenance differs from its canonical record')

    if 'mixed_reset' in saved:
        if not _same(saved, current):
            raise ValueError('mixed reset checkpoint layout or training contract differs')
        return []

    validate_reset_family(family, saved, primary)
    if not any(_reset_hashes(saved) == _reset_hashes(group) for group in groups):
        raise ValueError('mixed reset checkpoint must be from one of its two declared styles')
    aligned = _refresh_digest(dict(saved, **_reset_hashes(primary)))
    if not _same(aligned, primary):
        raise ValueError('mixed reset resume requires unchanged non-reset training contract')
    return ['surface_mixed_reset']
