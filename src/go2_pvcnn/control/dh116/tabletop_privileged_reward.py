"""Native reward passthrough and explicit, evidence-gated optional shaping."""
import hashlib
import json
import math
from pathlib import Path

import torch

VERSION = 'dh116-privileged-teacher-reward-v1'
PROFILES = ('teacher_r0_existing', 'teacher_r1_geometry', 'teacher_r2_support_recovery')


def _number(value, reference, name):
    if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
            or value.device != reference.device or not value.is_floating_point()):
        raise ValueError(name + ' must match native reward shape/device')
    return value


def _qualification(provenance, key, expected_identity):
    entry = provenance.get(key) if isinstance(provenance, dict) else None
    if (not isinstance(entry, dict) or entry.get('qualified') is not True
            or entry.get('evidence_kind') != 'native_isaac_physx'
            or entry.get('schema_only') is not False):
        raise ValueError('qualified native provenance required for ' + key)
    path, expected = entry.get('proof_path'), entry.get('proof_sha256')
    if not isinstance(path, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('SHA-bound qualification provenance required')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('qualification proof SHA mismatch')
    proof = json.loads(raw)
    fields = ('qualified', 'evidence_kind', 'schema_only', 'state_contract_sha256', 'action_contract_sha256')
    if any(proof.get(name) != entry.get(name) for name in fields):
        raise ValueError('qualification report identity mismatch')
    for name in ('state_contract_sha256', 'action_contract_sha256'):
        value = entry.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError('qualification requires current state/action identity')
        if not isinstance(expected_identity, dict) or expected_identity.get(name) != value:
            raise ValueError('qualified provenance does not match current runtime identity')
    # A Boolean does not qualify force data. Bind the actual qualification run
    # and its immutable raw native evidence, then check the measured-source schema.
    if proof.get('process_exit_code') != 0 or type(proof.get('sample_count')) is not int or proof['sample_count'] < 1:
        raise ValueError('completed native qualification run required')
    evidence_path, evidence_sha = proof.get('native_evidence_path'), proof.get('native_evidence_sha256')
    if (not isinstance(evidence_path, str) or not isinstance(evidence_sha, str)
            or hashlib.sha256(Path(evidence_path).read_bytes()).hexdigest() != evidence_sha):
        raise ValueError('SHA-bound raw native evidence required for qualification')
    required = ('geometry_source',) if key == 'geometry' else (
        'force_side', 'force_units', 'force_frame', 'partner_attribution', 'per_digit_source')
    if any(not proof.get(field) for field in required):
        raise ValueError('qualification report lacks measured source: ' + key)
    return entry


def compose_native_reward(native_reward, native_terms, *, profile='teacher_r0_existing',
                          shaping=None, provenance=None, dt=None, expected_identity=None):
    if profile not in PROFILES:
        raise ValueError('unknown privileged Teacher reward profile')
    if (not isinstance(native_reward, torch.Tensor) or native_reward.ndim != 1
            or not native_reward.is_floating_point() or not bool(torch.isfinite(native_reward).all())):
        raise ValueError('finite per-environment native reward required')
    if not isinstance(native_terms, dict) or not native_terms:
        raise ValueError('native reward terms required for contribution closure')
    terms = {}
    for name, value in native_terms.items():
        if not isinstance(name, str):
            raise ValueError('native reward term names must be strings')
        _number(value, native_reward, name)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('native reward term must be finite: ' + name)
        terms[name] = dict(raw=value.detach().clone(), normalized=value.detach().clone(), coefficient=1.,
            raw_kind='native_contribution_not_original_physical_measurement',
            raw_unit='points', normalized_unit='points', coefficient_unit='identity',
            physical_raw=None, physical_raw_unavailable_reason='native extras provides final contributions; source formula in Phase0 audit',
            audit_reference='docs/privileged_teacher_reward_audit.md',
            dt_scaling='already_scaled_by_native_task', valid_mask=torch.ones_like(value, dtype=torch.bool),
            valid_mask_kind='finite_contribution;physical_eligibility_is_owned_by_native_task',
            source='native_TensorEnv.extras.terms.' + name,
            per_env_contribution=value.detach().clone(), total_contribution=value.detach().sum())
    total = torch.stack(list(native_terms.values())).sum(0)
    if not torch.allclose(total, native_reward, atol=1e-5, rtol=1e-5):
        raise ValueError('native reward term sum does not close to native total')
    reward = native_reward.clone()
    if profile != 'teacher_r0_existing':
        _qualification(provenance, 'geometry', expected_identity)
        if not isinstance(shaping, dict):
            raise ValueError('qualified geometry measurements required')
        before = _number(shaping.get('geometry_potential_before'), native_reward, 'geometry before')
        after = _number(shaping.get('geometry_potential_after'), native_reward, 'geometry after')
        mask = shaping.get('geometry_valid')
        if not isinstance(mask, torch.Tensor) or mask.shape != reward.shape or mask.dtype != torch.bool or mask.device != reward.device:
            raise ValueError('explicit geometry validity mask required')
        if bool((mask & (~torch.isfinite(before) | ~torch.isfinite(after))).any()):
            raise ValueError('qualified geometry marked valid contains NA')
        if any('teacher_geometry' in key or key == 'grasp_acquisition' for key in native_terms):
            raise ValueError('geometry shaping overlaps an existing native reward term')
        raw = after - before
        contribution = torch.where(mask, raw, 0.)
        terms['teacher_geometry_progress'] = dict(raw=raw.detach().clone(), normalized=raw.detach().clone(),
            coefficient=1., dt_scaling='potential_difference_no_dt', valid_mask=mask.detach().clone(),
            raw_unit='qualified_geometry_potential', normalized_unit='qualified_geometry_potential',
            coefficient_unit='points_per_potential_unit',
            source=provenance['geometry'], per_env_contribution=contribution.detach().clone(),
            total_contribution=contribution.detach().sum())
        reward = reward + contribution
    if profile == 'teacher_r2_support_recovery':
        _qualification(provenance, 'support', expected_identity)
        if any('support_deficit' in key for key in native_terms):
            raise ValueError('support deficit overlaps existing native reward; select one audited owner')
        if isinstance(dt, bool) or not isinstance(dt, (float, int)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('positive measured physical dt required')
        support = _number(shaping.get('support_n'), native_reward, 'signed attributable support')
        weight = _number(shaping.get('weight_n'), native_reward, 'runtime object weight')
        mask = shaping.get('support_valid')
        if not isinstance(mask, torch.Tensor) or mask.shape != reward.shape or mask.dtype != torch.bool or mask.device != reward.device:
            raise ValueError('explicit support validity mask required')
        finite = torch.isfinite(support) & torch.isfinite(weight) & (weight > 0)
        if bool((mask & ~finite).any()):
            raise ValueError('qualified support marked valid contains NA or invalid weight')
        safe_weight = torch.where(finite, weight, 1.)
        normalized = ((weight-support)/safe_weight).clamp(0., 1.)
        contribution = torch.where(mask, -1.5*normalized*dt, 0.)
        terms['teacher_support_deficit'] = dict(raw=(weight-support).detach().clone(),
            normalized=normalized.detach().clone(), coefficient=-1.5, dt_scaling=float(dt),
            raw_unit='N', normalized_unit='weight_fraction', coefficient_unit='points_per_second',
            valid_mask=mask.detach().clone(), source=provenance['support'],
            per_env_contribution=contribution.detach().clone(), total_contribution=contribution.detach().sum())
        reward = reward + contribution
    return reward, dict(version=VERSION, profile=profile, terms=terms,
        native_contribution=native_reward.detach().clone(), total_contribution=reward.detach().clone(),
        all_env_total=reward.detach().sum(), qualification='native_sources_require_separate_qualification')
