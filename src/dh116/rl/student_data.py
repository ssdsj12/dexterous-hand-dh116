from dh116.env.contracts import source_file
'Source-bound stereo dataset/collection/frozen-Teacher binding, BC/DAgger losses/trainers and strict actor checkpoint/export.'
import math as stere_bc__math
from copy import deepcopy as stere_check__deepcopy
import copy as stere_dagge_datas__copy
import torch as stere_dagge_train__torch
import copy as stere_losse__copy
from copy import deepcopy as stere_sourc_bindi__deepcopy
import copy as stere_super_datas__copy
import copy as stere_teach_colle__copy
import torch as stere_bc__torch
from dataclasses import asdict
from dh116.env.contracts import make_contract as stere_dagge_datas__make_contract
import hashlib as stere_losse__hashlib
import hashlib as stere_sourc_bindi__hashlib
import json as stere_super_datas__json
import hashlib as stere_teach_colle__hashlib
import hashlib as stere_check__hashlib
import json as stere_losse__json
import io as stere_sourc_bindi__io
from pathlib import Path as stere_super_datas__Path
import io as stere_teach_colle__io
import io as stere_check__io
from pathlib import Path as stere_losse__Path
import json as stere_sourc_bindi__json
import torch as stere_super_datas__torch
from pathlib import Path as stere_teach_colle__Path
import math as stere_check__math
import re
from pathlib import Path as stere_sourc_bindi__Path
from dh116.env.contracts import make_contract as stere_super_datas__make_contract
from dataclasses import dataclass
from pathlib import Path as stere_check__Path
import torch as stere_losse__torch
import torch as stere_sourc_bindi__torch
from dh116.env.contracts import stere_contr__validate_contract as stere_super_datas__validate_contract
from dataclasses import field
import random
from torch import nn
import dh116.rl.qualification as promotion
from dh116.env.contracts import validate_identity as stere_super_datas__validate_identity
from dataclasses import fields
import numpy as np
from torch.nn import functional as F

def _same(a, b):
    """Compare tensor storage bytes, including NaN payloads, and typed trees."""
    if isinstance(a, stere_sourc_bindi__torch.Tensor) or isinstance(b, stere_sourc_bindi__torch.Tensor):
        if not isinstance(a, stere_sourc_bindi__torch.Tensor) or not isinstance(b, stere_sourc_bindi__torch.Tensor):
            return False
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        left = a.detach().cpu().contiguous().reshape(-1).view(stere_sourc_bindi__torch.uint8)
        right = b.detach().cpu().contiguous().reshape(-1).view(stere_sourc_bindi__torch.uint8)
        return stere_sourc_bindi__torch.equal(left, right)
    if isinstance(a, dict) or isinstance(b, dict):
        return type(a) is type(b) and a.keys() == b.keys() and all((_same(a[key], b[key]) for key in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all((_same(x, y) for (x, y) in zip(a, b)))
    return type(a) is type(b) and a == b
from dh116.env.contracts import require_aligned as stere_super_datas__require_aligned
import math as stere_teach_colle__math
import torch as stere_check__torch
STATE_LOSS_VERSION = 'dh116-stereo-canonical-field-loss-v2'

def _validate_bindings(bindings):
    """Validate only reference syntax; references never create qualification."""
    if not isinstance(bindings, list) or not bindings:
        raise ValueError('formal data requires trusted source bindings')
    for binding in bindings:
        if not isinstance(binding, dict) or binding.keys() != {'proof_path', 'proof_sha256', 'row_indices'}:
            raise ValueError('exact trusted source bindings required')
        if not isinstance(binding['proof_path'], str) or not stere_sourc_bindi__Path(binding['proof_path']).is_absolute():
            raise ValueError('absolute native source proof required')
        promotion.stere_promo__require_sha(binding['proof_sha256'])
        indices = binding['row_indices']
        if not isinstance(indices, list) or not indices or any((type(index) is not int or index < 0 for index in indices)) or (indices != sorted(set(indices))):
            raise ValueError('ordered unique raw record row selection required')
    return bindings
from dh116.env.contracts import require_sha as stere_super_datas__require_sha
import torch as stere_teach_colle__torch
from dh116.env.contracts import action_sha256

def masked_group_normalized_loss(predictions, targets, valid_mask):
    if predictions.ndim != 2 or predictions.shape[-1] != 24 or targets.shape != predictions.shape or (valid_mask.shape != predictions.shape) or (valid_mask.dtype != stere_losse__torch.bool) or (valid_mask.device != predictions.device) or (targets.device != predictions.device):
        raise ValueError('matching actions/targets and boolean mask [N,24] required')
    if not bool(stere_losse__torch.isfinite(predictions).all()) or bool((valid_mask & ~stere_losse__torch.isfinite(targets)).any()):
        raise ValueError('finite predictions and valid targets required')
    safe_targets = stere_losse__torch.where(valid_mask, targets, predictions.detach())
    error = (predictions - safe_targets).square()
    (group_losses, active) = ([], [])
    for (start, stop) in ((0, 12), (12, 18), (18, 24)):
        count = valid_mask[:, start:stop].sum(-1)
        group_losses.append((error[:, start:stop] * valid_mask[:, start:stop]).sum(-1) / count.clamp_min(1))
        active.append(count > 0)
    (losses, enabled) = (stere_losse__torch.stack(group_losses, -1), stere_losse__torch.stack(active, -1))
    per_row = losses.sum(-1) / enabled.sum(-1).clamp_min(1)
    rows = enabled.any(-1)
    return (per_row * rows).sum() / rows.sum().clamp_min(1)

def verify_stereo_dataset_sources(data):
    """Replay exact formal rows from outputs owned by one actual completion."""
    if not isinstance(data, dict):
        raise ValueError('versioned stereo dataset required')
    if data.get('schema_only') is True:
        return
    bindings = _validate_bindings(data.get('source_bindings'))
    from dh116.rl.student_data import rows_from_records as rows_from_records
    expected_rows = []
    for binding in bindings:
        try:
            proof_path = stere_sourc_bindi__Path(binding['proof_path'])
            if not proof_path.is_file():
                raise ValueError('regular native source proof required')
            proof_bytes = proof_path.read_bytes()
            if stere_sourc_bindi__hashlib.sha256(proof_bytes).hexdigest() != binding['proof_sha256']:
                raise ValueError('native source proof SHA mismatch')
            proof = stere_sourc_bindi__json.loads(proof_bytes)
            if not isinstance(proof, dict):
                raise ValueError('native source proof object required')
        except (OSError, TypeError, ValueError) as error:
            raise ValueError('native source proof unavailable or changed') from error
        base = proof_path.parent
        promotion._native_run(proof, base)
        promotion._process_output(proof, base, proof.get('raw_artifact'))
        (manifest_bytes, manifest_path) = promotion._process_output(proof, base, proof.get('stereo_source_manifest_artifact'))
        try:
            manifest = stere_sourc_bindi__json.loads(manifest_bytes)
        except (UnicodeError, ValueError) as error:
            raise ValueError('native stereo source manifest identity differs') from error
        if not isinstance(manifest, dict) or manifest.keys() != {'version', 'teacher_checkpoint_sha256', 'native_raw_artifact', 'records_artifact'} or manifest['version'] != 'dh116-stereo-source-manifest-v1' or (manifest['teacher_checkpoint_sha256'] != data.get('source_sha256')) or (manifest['native_raw_artifact'] != proof.get('raw_artifact')):
            raise ValueError('native stereo source manifest identity differs')
        (records_bytes, _) = promotion._process_output(proof, base, manifest['records_artifact'], binding_base=manifest_path.parent)
        packet = stere_sourc_bindi__torch.load(stere_sourc_bindi__io.BytesIO(records_bytes), map_location='cpu', weights_only=True)
        if not isinstance(packet, dict) or packet.get('version') != 'dh116-stereo-teacher-candidate-records-v1' or packet.get('teacher_checkpoint_sha256') != data.get('source_sha256') or (not isinstance(packet.get('records'), list)):
            raise ValueError('actual sealed stereo records required')
        records = stere_sourc_bindi__deepcopy(packet['records'])
        if data.get('purpose') == 'successful_bc':
            for record in records:
                record['mode'] = 'teacher_reference'
                record['teacher_qualification'] = stere_sourc_bindi__deepcopy(data.get('qualification'))
                record['candidate_only'] = False
        elif data.get('purpose') != 'dagger_visited_states':
            raise ValueError('explicit formal source purpose required')
        rows = rows_from_records(records)
        indices = binding['row_indices']
        if any((index >= len(rows) for index in indices)):
            raise ValueError('ordered unique raw record row selection required')
        expected_rows.extend((rows[index] for index in indices))
    if not _same(data.get('rows'), expected_rows):
        raise ValueError('dataset rows differ from sealed native records')
from dh116.env.contracts import stere_contr__file_sha256 as stere_super_datas__file_sha256
from dh116.env.contracts import require_aligned as stere_teach_colle__require_aligned
from dh116.env.contracts import stere_contr__digest as digest

def _force_proof_verified(binding):
    """Proof/producer/raw-byte verification is shared with the promotion gate."""
    try:
        from dh116.rl.qualification import verify_force_qualification
        return bool(verify_force_qualification(binding))
    except (ValueError, FileNotFoundError, ImportError):
        return False
from dh116.rl.student_model import StereoHistoryBatch as stere_super_datas__StereoHistoryBatch
from dh116.env.contracts import validate_identity as stere_teach_colle__validate_identity
from dh116.env.contracts import stere_contr__file_sha256 as stere_check__file_sha256

def _mask(mask, reference):
    if not isinstance(mask, stere_losse__torch.Tensor) or mask.dtype != stere_losse__torch.bool or mask.device != reference.device:
        raise ValueError('boolean mask tensor on the prediction device required')
    if mask.shape == reference.shape[:-1]:
        mask = mask[..., None].expand_as(reference)
    if mask.shape != reference.shape:
        raise ValueError('mask shape must match the supervised tensor')
    return mask
from dh116.env.contracts import require_sha as stere_teach_colle__require_sha
from dh116.env.contracts import require_sha as stere_check__require_sha

def _safe_pair(prediction, target, mask):
    if not isinstance(target, stere_losse__torch.Tensor) or prediction.shape != target.shape or prediction.device != target.device or (not prediction.is_floating_point()) or (not target.is_floating_point()):
        raise ValueError('matching floating target/prediction shape and device required')
    mask = _mask(mask, prediction)
    if bool((mask & (~stere_losse__torch.isfinite(prediction) | ~stere_losse__torch.isfinite(target))).any()):
        raise ValueError('valid predictions and targets must be finite')
    return (stere_losse__torch.where(mask, prediction, 0.0), stere_losse__torch.where(mask, target.detach(), 0.0), mask)
from dh116.rl.student_model import StereoHistoryBatch as stere_teach_colle__StereoHistoryBatch
from dh116.env.contracts import stere_contr__validate_contract as stere_check__validate_contract

def _reduce(values, mask, weights=None):
    weights = stere_losse__torch.ones_like(values) if weights is None else stere_losse__torch.as_tensor(weights, device=values.device, dtype=values.dtype)
    if bool((~stere_losse__torch.isfinite(weights) | (weights < 0)).any()):
        raise ValueError('finite nonnegative loss weights required')
    weights = stere_losse__torch.broadcast_to(weights, values.shape) * mask
    return (values * weights).sum() / weights.sum().clamp_min(1e-08)

def clone_tree(value, *, cpu=False):
    if isinstance(value, stere_teach_colle__torch.Tensor):
        value = value.detach()
        return (value.cpu() if cpu else value).clone()
    if isinstance(value, dict):
        return {k: clone_tree(v, cpu=cpu) for (k, v) in value.items()}
    if isinstance(value, list):
        return [clone_tree(v, cpu=cpu) for v in value]
    if isinstance(value, tuple):
        return tuple((clone_tree(v, cpu=cpu) for v in value))
    return stere_teach_colle__copy.deepcopy(value)
from dh116.rl.student_model import stere_stude_ppo__INPUT_KEYS as INPUT_KEYS

def masked_smooth_l1(prediction, target, mask, weights=None):
    (p, t, m) = _safe_pair(prediction, target, mask)
    return _reduce(F.smooth_l1_loss(p, t, reduction='none'), m, weights)

def _tensor(value, shape, name, *, boolean=False, finite=True):
    if not isinstance(value, stere_teach_colle__torch.Tensor) or value.shape != shape or (value.dtype != stere_teach_colle__torch.bool if boolean else not value.is_floating_point()):
        raise ValueError(f'{name} must be ' + ('boolean' if boolean else 'floating') + f' tensor {shape}')
    if finite and (not boolean) and (not bool(stere_teach_colle__torch.isfinite(value).all())):
        raise ValueError(name + ' must be finite')
    return value
from dh116.rl.student_model import make_stereo_actor

def masked_bce(prediction, target, mask):
    (p, t, m) = _safe_pair(prediction, target, mask)
    if bool((m & ((t < 0) | (t > 1))).any()):
        raise ValueError('classification labels must lie in [0,1]')
    return _reduce(F.binary_cross_entropy_with_logits(p, t, reduction='none'), m)

def _actions(value, n, name):
    _tensor(value, (n, 24), name)
    if bool((value.abs() > 1).any()):
        raise ValueError(name + ' must be canonical normalized24 in [-1,1]')
    return value
from dh116.rl.student_model import normalize_network_config

def quaternion_geodesic_loss(prediction, target, mask):
    if prediction.shape[-1] != 4 or mask.shape != prediction.shape[:-1]:
        raise ValueError('wxyz groups [...,4] and group validity [...] required')
    (p, t, m) = _safe_pair(prediction, target, mask)
    identity = stere_losse__torch.zeros_like(p)
    identity[..., 0] = 1.0
    pnorm = p.norm(dim=-1, keepdim=True)
    tnorm = t.norm(dim=-1, keepdim=True)
    if bool((mask & (tnorm[..., 0] <= 1e-08)).any()):
        raise ValueError('valid target quaternion must be nonzero')
    p = stere_losse__torch.where(mask[..., None] & (pnorm > 1e-08), p / pnorm.clamp_min(1e-08), identity)
    t = stere_losse__torch.where(mask[..., None], t / tnorm.clamp_min(1e-08), identity)
    t = stere_losse__torch.where(((p * t).sum(-1) < 0)[..., None], -t, t)
    angle = 4 * stere_losse__torch.atan2((p - t).norm(dim=-1), (p + t).norm(dim=-1))
    return _reduce(angle, mask)

@dataclass
class PreActionState:
    sensor_history: stere_teach_colle__StereoHistoryBatch
    privileged: stere_teach_colle__torch.Tensor
    privileged_valid: stere_teach_colle__torch.Tensor
    identities: list
    extra_labels: dict = field(default_factory=dict)
    label_validity: dict = field(default_factory=dict)
    state_sources: dict = field(default_factory=dict)
    privileged_identities: list | None = None
    force_qualification: dict | None = None

    def validate(self, *, max_image_age_s=1 / 60):
        if type(self.sensor_history) is not stere_teach_colle__StereoHistoryBatch:
            raise TypeError('typed sensor history required')
        h = self.sensor_history.validate()
        n = len(h)
        _tensor(self.privileged, (n, 371), 'privileged', finite=False)
        _tensor(self.privileged_valid, (n, 371), 'privileged_valid', boolean=True)
        if self.privileged.device != h.proprio.device or self.privileged_valid.device != h.proprio.device:
            raise ValueError('snapshot sensors and privileged labels must share a device')
        if bool((self.privileged_valid & ~stere_teach_colle__torch.isfinite(self.privileged)).any()):
            raise ValueError('valid 371D labels must be finite')
        if not isinstance(self.identities, list) or len(self.identities) != n:
            raise ValueError('one identity per environment required')
        seen = set()
        for (i, row) in enumerate(self.identities):
            stere_teach_colle__validate_identity(row)
            key = (row['run_uuid'], row['env_id'])
            if key in seen:
                raise ValueError('duplicate environment identity')
            seen.add(key)
            if self.privileged_identities is not None:
                if len(self.privileged_identities) != n:
                    raise ValueError('privileged identity count mismatch')
                stere_teach_colle__require_aligned(row, self.privileged_identities[i])
        if not bool(h.history_valid[:, -1].all()) or not bool(h.image_valid[:, -1].all()) or bool((h.image_age[:, -1] > max_image_age_s).any()):
            raise ValueError('stale/missing current stereo image or history frame')
        for name in ('extra_labels', 'label_validity', 'state_sources'):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(name + ' dictionary required')
        return self

    def clone(self):
        return type(self)(**{f.name: self.sensor_history.clone() if f.name == 'sensor_history' else clone_tree(getattr(self, f.name)) for f in fields(self)})
stere_check__VERSION = 'dh116-stereo-rgb-student-checkpoint-v1'

def per_dimension_metrics(prediction, target, mask):
    """Masked MAE/RMSE/P95; no valid labels => NA (NaN), never fake zero error."""
    (p, t, m) = _safe_pair(prediction, target, mask)
    if p.ndim != 2:
        raise ValueError('metrics require [B,D]')
    error = (p - t).abs()
    count = m.sum(0)
    denominator = count.clamp_min(1)
    nan = stere_losse__torch.full_like(denominator, float('nan'), dtype=p.dtype)
    mae = stere_losse__torch.where(count > 0, (error * m).sum(0) / denominator, nan)
    rmse = stere_losse__torch.where(count > 0, ((error.square() * m).sum(0) / denominator).sqrt(), nan)
    p95 = stere_losse__torch.stack([stere_losse__torch.quantile(error[:, i][m[:, i]], 0.95) if bool(m[:, i].any()) else nan[i] for i in range(p.shape[1])])
    return dict(count=count, mae=mae, rmse=rmse, p95=p95)

@dataclass
class StepResult:
    next_state: PreActionState
    reward: stere_teach_colle__torch.Tensor
    done: stere_teach_colle__torch.Tensor
    info: dict
    issued_actions: stere_teach_colle__torch.Tensor
    applied_targets: dict | None = None
DEPLOYMENT_VERSION = 'dh116-stereo-rgb-deployment-actor-v1'

class StereoEstimationLoss(nn.Module):

    def __init__(self, state_contract_path=None, *, state_contract=None, head_weights=None, expected_contract_sha256=None):
        super().__init__()
        self.loss_version = STATE_LOSS_VERSION
        if state_contract is None:
            path = stere_losse__Path(state_contract_path) if state_contract_path is not None else stere_losse__Path(__file__).resolve().parents[3] / 'configs/contracts/privileged_state.json'
            state_contract = stere_losse__json.loads(path.read_text(encoding='utf-8'))
        self.state_contract = stere_losse__copy.deepcopy(state_contract)
        self.state_contract_sha256 = stere_losse__hashlib.sha256(stere_losse__json.dumps(self.state_contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if expected_contract_sha256 is not None and expected_contract_sha256 != self.state_contract_sha256:
            raise ValueError('state contract SHA256 mismatch')
        rows = self.state_contract.get('dimensions', [])
        if self.state_contract.get('size') != 371 or [r.get('index') for r in rows] != list(range(371)):
            raise ValueError('complete canonical 371D state contract required')
        common = set(self.state_contract.get('runtime_mask_contract', {}).get('required_common_keys', []))
        if not common or self.state_contract.get('runtime_mask_contract', {}).get('default') != 'fail_closed' or any((not isinstance(r.get('required_validity_keys'), list) or not common.issubset(r['required_validity_keys']) or r.get('loss_type') not in ('regression', 'classification', 'none') or (r.get('supervision_status') not in ('NA', 'conditional')) for r in rows)):
            raise ValueError('explicit per-dimension fail-closed validity contract required')
        allowed = [r.get('deployment_allowed') is True and r.get('loss_type') != 'none' and (r.get('supervision_status') != 'NA') and (r.get('student_source') != 'unobservable') for r in rows]
        self.register_buffer('state_allowed', stere_losse__torch.tensor(allowed, dtype=stere_losse__torch.bool))
        self.register_buffer('state_classification', stere_losse__torch.tensor([r['loss_type'] == 'classification' for r in rows], dtype=stere_losse__torch.bool), persistent=False)
        self.required = {key: [r['index'] for r in rows if key in r.get('required_validity_keys', [])] for key in {k for r in rows for k in r.get('required_validity_keys', [])}}
        self.rotations = self.state_contract.get('rotation_groups', [])
        self.head_weights = {} if head_weights is None else dict(head_weights)
        if any((not isinstance(v, (float, int)) or isinstance(v, bool) or v < 0 or (not stere_losse__torch.isfinite(stere_losse__torch.tensor(v))) for v in self.head_weights.values())):
            raise ValueError('finite nonnegative head weights required')

    def state_mask(self, prediction, validity):
        if prediction.ndim != 2 or prediction.shape[1] != 371:
            raise ValueError('state prediction shape must be [B,371]')
        supplied = validity.get('privileged_state', stere_losse__torch.zeros_like(prediction, dtype=stere_losse__torch.bool))
        mask = _mask(supplied, prediction) & self.state_allowed[None].to(prediction.device)
        sources = validity.get('state_sources', {})
        if not isinstance(sources, dict):
            raise ValueError('state_sources must contain explicit source masks')
        for (key, indices) in self.required.items():
            value = sources.get(key)
            if value is None:
                mask[:, indices] = False
                continue
            if not isinstance(value, stere_losse__torch.Tensor) or value.dtype != stere_losse__torch.bool or value.device != prediction.device or (value.shape not in (prediction.shape, prediction.shape[:1])):
                raise ValueError('state source mask must be boolean [B] or [B,371]: ' + key)
            mask[:, indices] &= value[:, None] if value.ndim == 1 else value[:, indices]
        return mask

    def canonical_state_loss(self, prediction, target, mask):
        """Mixed canonical mathematics; callers qualify source/symmetry masks first."""
        if prediction.ndim != 2 or prediction.shape[1] != 371:
            raise ValueError('state prediction shape must be [B,371]')
        (_, _, mask) = _safe_pair(prediction, target, mask)
        scalar_mask = mask.clone()
        rotation_losses = []
        for group in self.rotations:
            (lo, hi) = group['slice']
            group_mask = mask[:, lo:hi].all(-1)
            scalar_mask[:, lo:hi] = False
            rotation_losses.append(quaternion_geodesic_loss(prediction[:, lo:hi], target[:, lo:hi], group_mask))
        classification = self.state_classification[None].to(prediction.device)
        regression_loss = masked_smooth_l1(prediction, target, scalar_mask & ~classification)
        classification_loss = masked_bce(prediction, target, scalar_mask & classification)
        return regression_loss + classification_loss + sum(rotation_losses, prediction.new_zeros(()))

    @staticmethod
    def _force_mask(qualification, prediction):
        empty = stere_losse__torch.zeros_like(prediction, dtype=stere_losse__torch.bool)
        if qualification is None:
            return (empty, 'missing_force_source_evidence')
        if not isinstance(qualification, dict):
            raise TypeError('force qualification dictionary required')
        masks = []
        for key in ('source_valid', 'contact_valid', 'attributable'):
            value = qualification.get(key)
            if value is not None and (not isinstance(value, stere_losse__torch.Tensor) or value.dtype != stere_losse__torch.bool or value.shape != prediction.shape or (value.device != prediction.device)):
                raise ValueError('force qualification mask shape must be [B,2,3]: ' + key)
            if value is None:
                return (empty, 'missing_force_mask_' + key)
            masks.append(value)
        required = ('source_sha256', 'units', 'coordinate_frame', 'partner_attribution', 'per_digit_source', 'sign_convention', 'evidence_scope')
        missing = [key for key in required if key not in qualification]
        if missing:
            return (empty, 'missing_force_metadata_' + ','.join(missing))
        if qualification['evidence_scope'] != 'native_qualified':
            return (empty, 'force_evidence_scope_not_native_qualified')
        digest = qualification['source_sha256']
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-fA-F]{64}', digest):
            return (empty, 'invalid_force_source_sha256')
        if qualification['units'] != 'N' or qualification['coordinate_frame'] != 'accepted_pad_normal':
            return (empty, 'unsupported_force_units_or_frame')
        if qualification['partner_attribution'] != 'verified_object_partner':
            return (empty, 'unverified_force_partner_attribution')
        if qualification['sign_convention'] != 'compressive_positive_on_object':
            return (empty, 'unsupported_force_sign_convention')
        sources = qualification['per_digit_source']
        if not isinstance(sources, list) or len(sources) != 2 or any((not isinstance(row, list) or len(row) != 3 or any((not isinstance(x, str) or not x.strip() for x in row)) for row in sources)):
            return (empty, 'missing_per_digit_force_source')
        if not _force_proof_verified(qualification):
            return (empty, 'force_native_proof_not_verified')
        return (masks[0] & masks[1] & masks[2], None)

    def forward(self, predictions, targets, validity, *, force_qualification=None):
        if not isinstance(predictions, dict) or not predictions:
            raise ValueError('named estimator predictions required')
        first = next(iter(predictions.values()))
        if not isinstance(first, stere_losse__torch.Tensor):
            raise TypeError('tensor predictions required')
        zero = stere_losse__torch.where(stere_losse__torch.isfinite(first), first, 0.0).sum() * 0.0
        result = {'total': zero, 'coverage': {}, 'blocked_reasons': {}}
        allowed_heads = ('privileged_state', 'geometry', 'velocity', 'deformation', 'contact_logits', 'force_proxy', 'slip_logits', 'slip_velocity')
        for name in allowed_heads:
            if name not in predictions or name not in targets:
                continue
            (pred, target) = (predictions[name], targets[name])
            supplied = validity.get(name, stere_losse__torch.zeros_like(pred, dtype=stere_losse__torch.bool))
            mask = _mask(supplied, pred)
            if name == 'privileged_state':
                mask = self.state_mask(pred, validity)
                for group in self.rotations:
                    (lo, hi) = group['slice']
                    group_mask = mask[:, lo:hi].all(-1)
                    if group.get('symmetry_mask_required'):
                        symmetry = validity.get('state_sources', {}).get('rotation_symmetry_valid')
                        if symmetry is None:
                            group_mask &= False
                        elif not isinstance(symmetry, stere_losse__torch.Tensor) or symmetry.shape != (len(pred),) or symmetry.dtype != stere_losse__torch.bool or (symmetry.device != pred.device):
                            raise ValueError('rotation_symmetry_valid must be boolean [B]')
                        else:
                            group_mask &= symmetry
                    mask[:, lo:hi] = group_mask[:, None]
                loss = self.canonical_state_loss(pred, target, mask)
            elif name == 'force_proxy':
                if pred.shape[1:] != (2, 3):
                    raise ValueError('force prediction shape must be [B,2,3]')
                (qualified, reason) = self._force_mask(force_qualification, pred)
                mask = mask & qualified
                if reason:
                    result['blocked_reasons'][name] = reason
                if target.shape == pred.shape and bool((mask & (target < 0)).any()):
                    raise ValueError('qualified compressive-positive pad-normal targets must be nonnegative')
                loss = masked_smooth_l1(pred, target, mask)
            elif name in ('contact_logits', 'slip_logits'):
                loss = masked_bce(pred, target, mask)
            else:
                loss = masked_smooth_l1(pred, target, mask)
            result[name] = loss
            result['coverage'][name] = float(mask.float().mean().detach())
            result['total'] = result['total'] + self.head_weights.get(name, 1.0) * loss
        return result

def require_teacher(qualification, source_sha256):
    if qualification is None:
        raise ValueError('qualified Teacher native proof is required before formal collection')
    stere_teach_colle__require_sha(source_sha256, 'Teacher checkpoint SHA256')
    from dh116.rl.qualification import require_teacher_qualification
    return require_teacher_qualification(qualification, source_sha256)
stere_bc__StereoEstimationLoss = StereoEstimationLoss
stere_bc__masked_smooth_l1 = masked_smooth_l1
stere_bc__masked_group_normalized_loss = masked_group_normalized_loss
ACTOR_CONTRACT_VERSION = 'dh116-stereo-rgb-actor-contract-v1'

def _teacher_state_sha256(state):
    from dh116.env.contracts import canonical_json
    hasher = stere_teach_colle__hashlib.sha256()

    def part(raw):
        hasher.update(len(raw).to_bytes(8, 'big'))
        hasher.update(raw)
    for (name, value) in sorted(state.items()):
        part(name.encode())
        if isinstance(value, stere_teach_colle__torch.Tensor):
            if not bool(stere_teach_colle__torch.isfinite(value).all()):
                raise ValueError('finite Teacher checkpoint weights required')
            tensor = value.detach().cpu().contiguous()
            part(canonical_json([str(tensor.dtype), list(tensor.shape)]).encode())
            part(tensor.reshape(-1).view(stere_teach_colle__torch.uint8).numpy().tobytes())
        else:
            part(canonical_json(value).encode())
    return hasher.hexdigest()

def _cpu(value):
    if isinstance(value, stere_check__torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for (key, item) in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple((_cpu(item) for item in value))
    return stere_check__deepcopy(value)

def _teacher_versions(teacher):
    return tuple(((name, id(value), value._version) for (name, value) in list(teacher.named_parameters()) + list(teacher.named_buffers())))

def _finite(value):
    if isinstance(value, stere_check__torch.Tensor):
        if not bool(stere_check__torch.isfinite(value).all()):
            raise ValueError('nonfinite checkpoint tensor')
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite(item)

class FrozenTeacherBinding:
    """Checkpoint-verified exact current model; freezing grants no qualification."""

    def __init__(self, teacher, checkpoint_path, *, expected_sha256):
        from dh116.rl.teacher import TensorNativeTeacher
        from dh116.rl.device import tenso_check__VERSION as VERSION
        if type(teacher) is not TensorNativeTeacher:
            raise ValueError('actual current TensorNativeTeacher model required')
        stere_teach_colle__require_sha(expected_sha256, 'Teacher checkpoint SHA256')
        path = stere_teach_colle__Path(checkpoint_path).resolve()
        raw = path.read_bytes()
        if stere_teach_colle__hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('actual Teacher checkpoint SHA mismatch')
        payload = stere_teach_colle__torch.load(stere_teach_colle__io.BytesIO(raw), map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or payload.get('version') != VERSION or payload.get('identity', {}).get('model') != teacher.model_identity or (not isinstance(payload.get('policy'), dict)):
            raise ValueError('current Teacher checkpoint model identity/state required')
        saved = _teacher_state_sha256(payload['policy'])
        if saved != _teacher_state_sha256(teacher.state_dict()):
            raise ValueError('Teacher weights differ from checkpoint policy state')
        teacher.eval().requires_grad_(False)
        self.teacher = teacher
        self.path = path
        self.sha256 = expected_sha256
        self.state_sha256 = saved
        self.versions = _teacher_versions(teacher)

    def verify(self, teacher, source_sha256):
        from dh116.env.contracts import stere_contr__file_sha256 as file_sha256
        if teacher is not self.teacher or source_sha256 != self.sha256:
            raise ValueError('Teacher model/checkpoint differs from frozen binding')
        if file_sha256(self.path) != self.sha256 or any((module.training for module in teacher.modules())) or any((parameter.requires_grad for parameter in teacher.parameters())) or (_teacher_versions(teacher) != self.versions) or (_teacher_state_sha256(teacher.state_dict()) != self.state_sha256):
            raise ValueError('frozen Teacher checkpoint/state/weights or parameter version changed')
        return self

def tensor_state_sha256(state):
    if not isinstance(state, dict) or not state:
        raise ValueError('nonempty model tensor state required')
    hash_ = stere_check__hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not isinstance(value, stere_check__torch.Tensor):
            raise ValueError('named tensor state required')
        _finite(value)
        cpu = value.detach().cpu().contiguous()
        hash_.update((key + '|' + str(cpu.dtype) + '|' + str(tuple(cpu.shape)) + '|').encode())
        hash_.update(cpu.numpy().tobytes())
    return hash_.hexdigest()

def bind_frozen_teacher(teacher, checkpoint_path, *, expected_sha256):
    """Bind exact inference state after its normal current checkpoint loader.

    Optimizer/RNG validation stays owned by that loader. The supplied model is
    then frozen for Student label queries; this does not grant qualification.
    """
    return FrozenTeacherBinding(teacher, checkpoint_path, expected_sha256=expected_sha256).verify(teacher, expected_sha256)

def _source_sha():
    names = ('rl/student_data.py', 'rl/student_model.py', 'physics/vision.py', 'physics/wire.py', 'physics/stereo_match.py', 'env/contracts.py')
    return {name: stere_check__hashlib.sha256(source_file(name).read_bytes()).hexdigest() for name in names}

def _verify_teacher_binding(binding, teacher, source_sha256):
    if type(binding) is not FrozenTeacherBinding:
        raise ValueError('actual checkpoint-verified frozen Teacher binding required')
    return binding.verify(teacher, source_sha256)

def _training_source_sha():
    names = ('rl/ppo.py', 'rl/device.py', 'rl/training.py')
    return {name: stere_check__hashlib.sha256(source_file(name).read_bytes()).hexdigest() for name in names}

def _model_device(model, fallback):
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return fallback

def _validate_ppo_configuration(contract, actual):
    from dh116.rl.ppo import PPOConfig
    configured = asdict(PPOConfig(**contract['config'].get('ppo', {})))
    if configured != actual:
        raise ValueError('experiment PPO config differs from actual optimizer configuration')

def _teacher_input(before, teacher, callback, *, schema_only):
    device = _model_device(teacher, before.privileged.device)
    if callback is None:
        if not schema_only:
            raise ValueError('explicit six-key teacher_inputs callback required for current collection')
        return before.privileged.to(device).clone()
    if not callable(callback):
        raise TypeError('teacher_inputs callback required')
    from dh116.rl.teacher import tenso_teach_schem__INPUT_KEYS as INPUT_KEYS
    binding = callback(before.clone())
    if not isinstance(binding, dict) or set(binding) != {'inputs', 'identities'}:
        raise ValueError('teacher_inputs must bind inputs and same pre-action identities')
    identities = binding['identities']
    if not isinstance(identities, list) or len(identities) != len(before.identities):
        raise ValueError('one Teacher pre-action identity per environment required')
    for (sensor, teacher_identity) in zip(before.identities, identities):
        stere_teach_colle__require_aligned(sensor, teacher_identity)
    inputs = binding['inputs']
    if not isinstance(inputs, dict) or set(inputs) != set(INPUT_KEYS):
        raise ValueError('current Teacher requires exact six input keys')
    for (key, expected) in (('common371', before.privileged), ('common_valid', before.privileged_valid)):
        value = inputs[key]
        if not isinstance(value, stere_teach_colle__torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype or (value.device != expected.device):
            raise ValueError('Teacher common snapshot shape/dtype/device differs')
        equal = stere_teach_colle__torch.equal(value, expected) if value.dtype == stere_teach_colle__torch.bool else bool(stere_teach_colle__torch.isclose(value, expected, rtol=0, atol=0, equal_nan=True).all())
        if not equal:
            raise ValueError('Teacher common snapshot differs; raw NaN/masks must be preserved')
    if any((not isinstance(value, stere_teach_colle__torch.Tensor) or len(value) != len(before.sensor_history) for value in inputs.values())):
        raise ValueError('six-key Teacher tensors must share the pre-action batch')
    return {key: value.detach().to(device).clone() for (key, value) in inputs.items()}

def _actor_contract(contract, network, *, training_provenance=None):
    from dh116.rl.qualification import validate_training_provenance
    calibration = stere_check__require_sha(contract.get('calibration_sha256'), 'calibration SHA')
    provenance = validate_training_provenance(contract['config'].get('provenance') if training_provenance is None else training_provenance, require_artifacts=True)
    processing = stere_check__deepcopy(contract['config'].get('preprocessor'))
    if processing is not None:
        from dh116.physics.vision import validate_processing_contract
        validate_processing_contract(processing)
        if processing['calibration_sha256'] != calibration:
            raise ValueError('actor preprocessing/calibration contract mismatch')
    result = dict(version=ACTOR_CONTRACT_VERSION, input_keys=list(INPUT_KEYS), actor_input='StereoHistoryBatch_only', action_size=24, action_contract_sha256=action_sha256(), training_kind=contract['kind'], native_qualified=False, network_config_sha256=digest(network), calibration_sha256=calibration, training_contract_sha256=contract['contract_sha256'], state_schema_sha256=contract['state_schema_sha256'], training_provenance=provenance, preprocessor=dict(binding='bound_raw_stereo' if processing is not None else 'history_input_only_unspecified', processing=processing, rgb='synchronized_left_right_RGB', depth='computed_from_synchronized_left_right_RGB', points='calibrated_stereo_world_points', proprio='measured62_with_previous_issued24', missing='explicit_boolean_masks;no_ground_truth_fallback'), source_sha256=_source_sha())
    result['contract_sha256'] = digest(result)
    return result

def _validate_step(before, result, submitted, max_age):
    n = len(before.sensor_history)
    if type(result) is not StepResult:
        raise TypeError('env.step must return StepResult')
    _tensor(result.reward, (n,), 'reward')
    _tensor(result.done, (n,), 'done', boolean=True)
    _actions(result.issued_actions, n, 'actually issued actions')
    if not stere_teach_colle__torch.equal(result.issued_actions.to(submitted), submitted):
        raise ValueError('environment changed whole-action owner request without declared shielding')
    if not isinstance(result.info, dict):
        raise ValueError('step info dictionary required')
    execution = result.info.get('execution')
    if not isinstance(execution, list) or len(execution) != n:
        raise ValueError('actual per-environment physics execution range required')
    after = result.next_state.validate(max_image_age_s=max_age)
    for (i, (old, new, run)) in enumerate(zip(before.identities, after.identities, execution)):
        if not isinstance(run, dict) or run.get('physics_tick_start') != old['physics_tick_end'] or type(run.get('physics_tick_end')) is not int or (run['physics_tick_end'] <= run['physics_tick_start']) or (run.get('simulation_timestamp_start') != old['simulation_timestamp']) or (not isinstance(run.get('simulation_timestamp_end'), (float, int))) or (not stere_teach_colle__math.isfinite(run['simulation_timestamp_end'])) or (run['simulation_timestamp_end'] <= old['simulation_timestamp']):
            raise ValueError('execution physics/time range must begin at the preserved pre-action state')
        if old['run_uuid'] != new['run_uuid'] or old['env_id'] != new['env_id'] or old['calibration_sha256'] != new['calibration_sha256']:
            raise ValueError('next-state run/environment/calibration identity mismatch')
        if new['physics_tick_start'] < run['physics_tick_end'] or new['simulation_timestamp'] < run['simulation_timestamp_end']:
            raise ValueError('next state precedes actual execution completion')
        h = after.sensor_history
        if bool(result.done[i]):
            if new['episode_uid'] == old['episode_uid'] or new['reset_id'] <= old['reset_id']:
                raise ValueError('terminal reset must change episode/reset identity')
            if int(h.history_valid[i].sum()) != 1 or not bool((h.proprio[i, -1, 38:] == 0).all()):
                raise ValueError('reset history contains an earlier episode or previous action')
        else:
            if new['episode_uid'] != old['episode_uid'] or new['reset_id'] != old['reset_id']:
                raise ValueError('episode identity changed without a terminal reset')
            if new['policy_tick'] <= old['policy_tick'] or new['frame_id'] <= old['frame_id']:
                raise ValueError('next pre-action identity clocks must be increasing')
            if not stere_teach_colle__torch.equal(h.proprio[i, -1, 38:].to(submitted), submitted[i]):
                raise ValueError('history previous action differs from actual issued action')
    return execution

def _validate_actor_contract(value, network, calibration):
    if not isinstance(value, dict) or value.get('version') != ACTOR_CONTRACT_VERSION:
        raise ValueError('independent actor input contract version required')
    from dh116.env.contracts import state_contract
    if value.get('state_schema_sha256') != digest(state_contract()):
        raise ValueError('current371 schema differs from actor package field semantics')
    if value.get('input_keys') != list(INPUT_KEYS) or value.get('actor_input') != 'StereoHistoryBatch_only' or value.get('action_size') != 24 or (value.get('action_contract_sha256') != action_sha256()):
        raise ValueError('actor input/action contract mismatch; truth keys forbidden')
    if value.get('training_kind') not in ('student', 'bc', 'dagger') or value.get('native_qualified') is not False:
        raise ValueError('explicit independent actor training role required; export does not grant native qualification')
    from dh116.rl.qualification import validate_training_provenance
    if not isinstance(value.get('training_provenance'), dict):
        raise ValueError('explicit training provenance metadata required')
    validate_training_provenance(value['training_provenance'])
    if value.get('network_config_sha256') != digest(network):
        raise ValueError('network config SHA mismatch')
    if value.get('calibration_sha256') != calibration:
        raise ValueError('calibration contract mismatch')
    stere_check__require_sha(calibration, 'calibration SHA')
    preprocessor = value.get('preprocessor')
    if not isinstance(preprocessor, dict) or preprocessor.get('binding') not in ('bound_raw_stereo', 'history_input_only_unspecified'):
        raise ValueError('explicit raw-stereo or history-input-only preprocessor binding required')
    if preprocessor['binding'] == 'bound_raw_stereo':
        from dh116.physics.vision import validate_processing_contract
        processing = validate_processing_contract(preprocessor.get('processing'))
        if processing['calibration_sha256'] != calibration:
            raise ValueError('processing/calibration contract mismatch')
    elif preprocessor.get('processing') is not None:
        raise ValueError('unspecified history package cannot claim a raw processing contract')
    if value.get('source_sha256') != _source_sha():
        raise ValueError('deployment implementation source SHA mismatch')
    if value.get('contract_sha256') != digest({k: v for (k, v) in value.items() if k != 'contract_sha256'}):
        raise ValueError('actor contract SHA mismatch')
    return value

@stere_teach_colle__torch.no_grad()
def collect_step(env, student, teacher, *, teacher_probability=1.0, schema_only=False, teacher_qualification=None, source_sha256=None, generator=None, max_image_age_s=1 / 60, record_teacher_sample=False, teacher_inputs=None, teacher_binding=None):
    if type(schema_only) is not bool or not stere_teach_colle__math.isfinite(teacher_probability) or (not 0 <= teacher_probability <= 1):
        raise ValueError('explicit schema mode and Teacher probability in [0,1] required')
    binding = None if schema_only else require_teacher(teacher_qualification, source_sha256)
    if not schema_only and getattr(env, 'evidence_scope', None) != 'native_runtime_capture_unqualified':
        raise ValueError('formal collection requires the native runtime path; CPU fixtures require schema_only=True')
    if not schema_only or teacher_binding is not None:
        _verify_teacher_binding(teacher_binding, teacher, source_sha256)
    before = env.snapshot()
    if type(before) is not PreActionState:
        raise TypeError('env.snapshot must return PreActionState')
    before = before.clone().validate(max_image_age_s=max_image_age_s)
    n = len(before.sensor_history)
    device = before.privileged.device
    teacher_input = _teacher_input(before, teacher, teacher_inputs, schema_only=schema_only)
    teacher_mean = _actions(teacher.deterministic(teacher_input), n, 'Teacher deterministic mean').to(device)
    teacher_mu = teacher.latent_mean(teacher_input).to(device) if hasattr(teacher, 'latent_mean') else None
    teacher_std = teacher.log_std.exp().expand(n, 24).to(device) if hasattr(teacher, 'log_std') else None
    if teacher_mu is not None:
        _tensor(teacher_mu, (n, 24), 'Teacher latent mean')
    if teacher_std is not None:
        _tensor(teacher_std, (n, 24), 'Teacher latent std')
        if bool((teacher_std <= 0).any()):
            raise ValueError('Teacher latent std must be positive')
    teacher_sample = None
    if record_teacher_sample:
        if not hasattr(teacher, 'sample'):
            raise ValueError('Teacher does not expose a sampled policy action')
        teacher_sample = _actions(teacher.sample(teacher_input)['actions'], n, 'Teacher sampled action').to(device)
    nominal = student_mu = None
    if student is not None:
        sensor = before.sensor_history.to(_model_device(student, device)).clone()
        nominal = _actions(student.deterministic(sensor), n, 'Student nominal action').to(device)
        student_mu = student.latent_mean(sensor).to(device) if hasattr(student, 'latent_mean') else None
        if student_mu is not None:
            _tensor(student_mu, (n, 24), 'Student latent mean')
    if teacher_probability < 1 and nominal is None:
        raise ValueError('Student policy required for Student-owned execution')
    if teacher_probability in (0.0, 1.0):
        owner = stere_teach_colle__torch.full((n,), teacher_probability == 1.0, device=device, dtype=stere_teach_colle__torch.bool)
    else:
        owner = (stere_teach_colle__torch.rand(n, generator=generator, device='cpu') < teacher_probability).to(device)
    submitted = teacher_mean.clone() if nominal is None else stere_teach_colle__torch.where(owner[:, None], teacher_mean, nominal)
    owner_names = ['teacher' if value else 'student' for value in owner.tolist()]
    if hasattr(env, 'set_action_owners'):
        env.set_action_owners(owner_names.copy())
    current = env.snapshot()
    if type(current) is not PreActionState:
        raise TypeError('current env.snapshot must return PreActionState')
    current.validate(max_image_age_s=max_image_age_s)
    if len(current.sensor_history) != n or len(current.identities) != len(before.identities):
        raise ValueError('current pre-action environment batch/count differs')
    for (preserved, actual) in zip(before.identities, current.identities):
        stere_teach_colle__require_aligned(preserved, actual)
    if not schema_only or teacher_binding is not None:
        _verify_teacher_binding(teacher_binding, teacher, source_sha256)
    result = env.step(submitted.clone())
    if not schema_only or teacher_binding is not None:
        _verify_teacher_binding(teacher_binding, teacher, source_sha256)
    execution = _validate_step(before, result, submitted, max_image_age_s)
    if not schema_only and result.info.get('evidence_scope') != 'native_runtime_capture_unqualified':
        raise ValueError('formal execution lacks the matching native runtime evidence scope')
    mode = 'teacher_reference' if teacher_probability == 1 else 'dagger_student' if teacher_probability == 0 else 'dagger_mixed'
    return clone_tree(dict(sensor_history=before.sensor_history.as_dict(), privileged=before.privileged, privileged_valid=before.privileged_valid, identities=before.identities, privileged_identities=before.identities if before.privileged_identities is None else before.privileged_identities, extra_labels=before.extra_labels, label_validity=before.label_validity, state_sources=before.state_sources, force_qualification=before.force_qualification, actor_raw=dict(teacher=teacher_mu, student=student_mu), teacher_mean=teacher_mean, teacher_sample=teacher_sample, teacher_std=teacher_std, student_nominal=nominal, issued_actions=result.issued_actions, applied_targets=result.applied_targets, execution=execution, reward=result.reward, done=result.done, info=result.info, action_owner=owner_names, mode=mode, teacher_probability=float(teacher_probability), dagger_student_samples=int((~owner).sum()), teacher_query_count=n, schema_only=schema_only, native_qualified=False, teacher_qualification=binding), cpu=True)

def _write_new(path, payload):
    path = stere_check__Path(path)
    if path.exists():
        raise FileExistsError(path)
    _finite(payload)
    stream = stere_check__io.BytesIO()
    stere_check__torch.save(payload, stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as out:
        out.write(stream.getvalue())
    return dict(path=str(path), sha256=stere_check__hashlib.sha256(path.read_bytes()).hexdigest(), version=payload['version'])

class TeacherCollector:

    def __init__(self, **options):
        self.options = options

    def run(self, env, student, teacher, steps, **options):
        if type(steps) is not int or steps < 1:
            raise ValueError('positive bounded collection steps required')
        return [collect_step(env, student, teacher, **self.options | options) for _ in range(steps)]

def _boundary(trainer, history_reset):
    if history_reset != 'clear':
        raise ValueError('explicit history_reset=clear required')
    if not trainer.clean_boundary:
        raise RuntimeError('clean PPO boundary required; pending/nonempty rollout forbidden')

def run(env, student, teacher, steps, **options):
    return TeacherCollector(**options).run(env, student, teacher, steps)

def _validate_state(state, model):
    current = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(current) or any((not isinstance(state[k], stere_check__torch.Tensor) or state[k].shape != current[k].shape or state[k].dtype != current[k].dtype for k in current)):
        raise ValueError('model topology/input version mismatch')
    _finite(state)
stere_super_datas__clone_tree = clone_tree
stere_super_datas__require_teacher = require_teacher

def _training_payload(path):
    payload = stere_check__torch.load(stere_check__Path(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('version') != stere_check__VERSION:
        raise ValueError('independent Student checkpoint required; legacy visual3844 conversion forbidden')
    (contract, network) = (payload.get('contract'), payload.get('network_config'))
    stere_check__validate_contract(contract, kind='student')
    if network != normalize_network_config(network):
        raise ValueError('normalized network config required')
    configured = contract['config'].get('network')
    if configured is None or normalize_network_config(configured) != network:
        raise ValueError('experiment/network config mismatch')
    _validate_actor_contract(payload.get('actor_contract'), network, contract.get('calibration_sha256'))
    _validate_ppo_configuration(contract, payload.get('ppo_config'))
    if payload.get('training_source_sha256') != _training_source_sha():
        raise ValueError('PPO training source SHA mismatch')
    if payload.get('policy_sha256') != tensor_state_sha256(payload.get('policy')):
        raise ValueError('Student policy tensor SHA mismatch')
    _finite(payload)
    return payload
stere_super_datas__VERSION = 'dh116-stereo-supervision-v1'

def save_student_checkpoint(path, trainer, contract, *, network_config, history_reset='clear'):
    _boundary(trainer, history_reset)
    stere_check__validate_contract(contract, kind='student')
    _validate_ppo_configuration(contract, asdict(trainer.config))
    network = normalize_network_config(network_config)
    if network != trainer.model.network_config or normalize_network_config(contract['config'].get('network')) != network:
        raise ValueError('Student model/network/contract mismatch')
    policy = _cpu(trainer.model.state_dict())
    payload = dict(version=stere_check__VERSION, contract=stere_check__deepcopy(contract), network_config=network, actor_contract=_actor_contract(contract, network, training_provenance=getattr(trainer, 'training_provenance', None)), policy=policy, policy_sha256=tensor_state_sha256(policy), training_source_sha256=_training_source_sha(), optimizer=_cpu(trainer.optimizer.state_dict()), ppo_config=asdict(trainer.config), policy_version=trainer.policy_version, counters=stere_check__deepcopy(trainer.counters), rollout_steps=trainer.rollout_steps, num_envs=trainer.num_envs, history_reset=history_reset, rng=dict(torch=stere_check__torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(), cuda=stere_check__torch.cuda.get_rng_state_all() if stere_check__torch.cuda.is_initialized() else []), auxiliary_bc_metadata=stere_check__deepcopy(getattr(getattr(trainer, 'auxiliary_bc', None), 'metadata', None)))
    return _write_new(path, payload)

def _sample(tree, index, n, *, keep_batch=False):
    if isinstance(tree, stere_super_datas__torch.Tensor):
        if tree.ndim and tree.shape[0] == n:
            return tree[index:index + 1].clone() if keep_batch else tree[index].clone()
        return tree.clone()
    if isinstance(tree, dict):
        return {k: _sample(v, index, n, keep_batch=keep_batch) for (k, v) in tree.items()}
    if isinstance(tree, list) and len(tree) == n:
        return stere_super_datas__copy.deepcopy(tree[index])
    return stere_super_datas__copy.deepcopy(tree)

def _validate_optimizer(state, trainer, updates):
    groups = state.get('param_groups') if isinstance(state, dict) else None
    parameters = list(trainer.model.parameters())
    if not isinstance(groups, list) or len(groups) != 1 or groups[0].get('params') != list(range(len(parameters))):
        raise ValueError('complete named-order Student Adam group required')
    if groups[0].get('lr') != trainer.config.lr or not isinstance(state.get('state'), dict):
        raise ValueError('Student Adam config mismatch')
    expected = trainer.optimizer.state_dict()['param_groups'][0]
    if any((groups[0].get(key) != value for (key, value) in expected.items() if key != 'params')):
        raise ValueError('Student Adam hyperparameter contract mismatch')
    if updates and (not state['state']):
        raise ValueError('updated Student Adam state missing')
    for (index, entry) in state['state'].items():
        if type(index) is not int or not 0 <= index < len(parameters) or (not isinstance(entry, dict)):
            raise ValueError('unknown Adam parameter identity')
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= entry.keys():
            raise ValueError('incomplete Student Adam moments')
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key in entry and (not isinstance(entry[key], stere_check__torch.Tensor) or entry[key].shape != parameters[index].shape or entry[key].dtype != parameters[index].dtype):
                raise ValueError('Student Adam moment shape/dtype mismatch')
        if not isinstance(entry['step'], stere_check__torch.Tensor) or entry['step'].numel() != 1 or entry['step'].item() < 0:
            raise ValueError('Student Adam counter mismatch')

def rows_from_records(records):
    rows = []
    for record in records:
        n = len(record['identities'])
        for i in range(n):
            row = dict(history=_sample(record['sensor_history'], i, n, keep_batch=True), identity=stere_super_datas__copy.deepcopy(record['identities'][i]), privileged_identity=stere_super_datas__copy.deepcopy(record['privileged_identities'][i]), execution=stere_super_datas__copy.deepcopy(record['execution'][i]), mode=record['mode'], schema_only=record['schema_only'], action_owner=record['action_owner'][i], teacher_qualification=stere_super_datas__copy.deepcopy(record['teacher_qualification']))
            for key in ('privileged', 'privileged_valid', 'extra_labels', 'label_validity', 'state_sources', 'actor_raw', 'teacher_mean', 'teacher_sample', 'teacher_std', 'student_nominal', 'issued_actions', 'applied_targets', 'reward', 'done', 'info'):
                row[key] = _sample(record.get(key), i, n)
            binding = record.get('force_qualification')
            row['force_qualification'] = None if binding is None else {key: _sample(value, i, n) if isinstance(value, stere_super_datas__torch.Tensor) else stere_super_datas__copy.deepcopy(value) for (key, value) in binding.items()}
            rows.append(row)
    return stere_super_datas__clone_tree(rows, cpu=True)

def load_student_checkpoint(path, trainer, expected_contract, *, history_reset='clear', restore_rng=True):
    _boundary(trainer, history_reset)
    payload = _training_payload(path)
    stere_check__validate_contract(payload['contract'], kind='student', expected=expected_contract)
    if payload['network_config'] != trainer.model.network_config or payload['ppo_config'] != asdict(trainer.config):
        raise ValueError('Student optimizer/network configuration mismatch')
    if payload.get('history_reset') != history_reset or payload.get('num_envs') != trainer.num_envs or payload.get('rollout_steps') != trainer.rollout_steps:
        raise ValueError('Student rollout/history contract mismatch')
    _validate_state(payload['policy'], trainer.model)
    counters = payload.get('counters')
    if not isinstance(counters, dict) or set(counters) != {'environment_steps', 'updates'} or any((type(v) is not int or v < 0 for v in counters.values())):
        raise ValueError('complete Student counters required')
    if type(payload.get('policy_version')) is not int or payload['policy_version'] < 0:
        raise ValueError('Student policy version counter required')
    _validate_optimizer(payload.get('optimizer'), trainer, counters['updates'])
    noise = payload['policy']['actor.log_std']
    if bool(((noise < stere_check__math.log(trainer.config.min_std)) | (noise > stere_check__math.log(trainer.config.max_std))).any()):
        raise ValueError('Student noise contract mismatch')
    from dh116.rl.device import check_state___validate_rng as _validate_rng
    _validate_rng(payload.get('rng'))
    candidate = stere_check__deepcopy(trainer.model)
    candidate.load_state_dict(payload['policy'], strict=True)
    candidate_optimizer = stere_check__torch.optim.Adam(candidate.parameters(), lr=trainer.config.lr)
    candidate_optimizer.load_state_dict(payload['optimizer'])
    trainer.model.load_state_dict(payload['policy'], strict=True)
    trainer.optimizer.load_state_dict(payload['optimizer'])
    (trainer.policy_version, trainer.counters) = (payload['policy_version'], stere_check__deepcopy(counters))
    trainer.training_provenance = stere_check__deepcopy(payload['actor_contract']['training_provenance'])
    trainer._new_rollout()
    trainer.project_noise()
    if restore_rng:
        rng = payload['rng']
        stere_check__torch.set_rng_state(rng['torch'])
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        if rng['cuda']:
            stere_check__torch.cuda.set_rng_state_all(rng['cuda'])
    return dict(version=stere_check__VERSION, contract=stere_check__deepcopy(payload['contract']), network_config=stere_check__deepcopy(payload['network_config']), history_reset=history_reset, policy_version=trainer.policy_version, counters=stere_check__deepcopy(counters), auxiliary_bc_metadata=stere_check__deepcopy(payload.get('auxiliary_bc_metadata')))

def identity_key(row):
    return tuple((row[k] for k in ('run_uuid', 'env_id', 'episode_uid', 'reset_id', 'policy_tick')))

def export_actor(checkpoint_path, output_path):
    payload = _training_payload(checkpoint_path)
    state = {key[len('actor.'):]: value for (key, value) in payload['policy'].items() if key.startswith('actor.')}
    actor = make_stereo_actor(payload['network_config'])
    _validate_state(state, actor)
    input_contract = stere_check__deepcopy(payload['actor_contract'])
    input_contract['training_provenance']['exported_from_checkpoint_artifact'] = dict(path=str(stere_check__Path(checkpoint_path).resolve()), sha256=stere_check__file_sha256(checkpoint_path))
    input_contract['contract_sha256'] = digest({key: value for (key, value) in input_contract.items() if key != 'contract_sha256'})
    result = dict(version=DEPLOYMENT_VERSION, actor=state, actor_sha256=tensor_state_sha256(state), actor_contract=input_contract, network_config=stere_check__deepcopy(payload['network_config']), calibration_sha256=payload['contract']['calibration_sha256'], preprocessor=stere_check__deepcopy(payload['actor_contract']['preprocessor']))
    return _write_new(output_path, result)

def _row_action(value, name, required=True):
    if value is None and (not required):
        return
    if not isinstance(value, stere_super_datas__torch.Tensor) or value.shape != (24,) or (not value.is_floating_point()) or (not bool(stere_super_datas__torch.isfinite(value).all())) or bool((value.abs() > 1).any()):
        raise ValueError('finite canonical24 ' + name + ' required')

def save_actor_package(path, actor, contract, *, network_config):
    """Export actual BC/DAgger/Student actor weights without inventing PPO state."""
    stere_check__validate_contract(contract)
    if contract.get('kind') not in ('student', 'bc', 'dagger'):
        raise ValueError('only independent Student/BC/DAgger actor contracts may deploy')
    from dh116.rl.student_model import StereoStudentActor as StereoStudentActor
    if type(actor) is not StereoStudentActor:
        raise ValueError('actual independent StereoStudentActor required; legacy visual3844 conversion forbidden')
    network = normalize_network_config(network_config)
    if normalize_network_config(contract['config'].get('network')) != network:
        raise ValueError('BC/DAgger actor network config differs from complete source contract')
    if actor.input_mode != network['actor_input_mode'] or actor.task_state_config != network['task_state_config'] or actor.estimator.point_coordinate_mode != network['point_coordinate_mode'] or (actor.estimator.point_source_encoding != network.get('point_source_encoding', False)):
        raise ValueError('actor field/coordinate configuration semantics differ from export contract')
    with stere_check__torch.random.fork_rng(devices=[]):
        candidate = make_stereo_actor(network)
    state = _cpu(actor.state_dict())
    _validate_state(state, candidate)
    if actor.action_contract_sha256 != action_sha256():
        raise ValueError('actor action semantics SHA mismatch')
    input_contract = _actor_contract(contract, network)
    payload = dict(version=DEPLOYMENT_VERSION, actor=state, actor_sha256=tensor_state_sha256(state), actor_contract=input_contract, network_config=network, calibration_sha256=contract['calibration_sha256'], preprocessor=stere_check__deepcopy(input_contract['preprocessor']))
    return _write_new(path, payload)

def validate_rows(rows, *, contract, schema_only):
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty synchronized dataset rows required')
    seen = set()
    previous = {}
    for row in rows:
        identity = stere_super_datas__validate_identity(row['identity'])
        stere_super_datas__require_aligned(identity, row['privileged_identity'])
        stere_super_datas__require_sha(identity.get('physical_initial_condition_sha256'), 'physical initial condition SHA256')
        if not isinstance(identity.get('source_episode_group'), str) or not identity['source_episode_group']:
            raise ValueError('source episode group required for independent holdout')
        if identity['calibration_sha256'] != contract['calibration_sha256']:
            raise ValueError('dataset calibration contract mismatch')
        if row.get('schema_only') is not schema_only:
            raise ValueError('schema/native evidence scope mismatch')
        if not schema_only and row.get('info', {}).get('evidence_scope') != 'native_runtime_capture_unqualified':
            raise ValueError('formal dataset cannot admit CPU/non-native executed rows')
        key = identity_key(identity)
        if key in seen:
            raise ValueError('duplicate sample identity')
        seen.add(key)
        history = stere_super_datas__StereoHistoryBatch(**row['history']).validate()
        if len(history) != 1 or not bool(history.history_valid[0, -1]):
            raise ValueError('one current sensor history per row required')
        if bool((history.image_age[:, -1] > 1 / 60).any()) or not bool(history.image_valid[:, -1].all()):
            raise ValueError('stale stereo sample')
        for name in ('teacher_mean', 'issued_actions'):
            _row_action(row.get(name), name)
        for name in ('teacher_sample', 'student_nominal'):
            _row_action(row.get(name), name, False)
        (privileged, valid) = (row.get('privileged'), row.get('privileged_valid'))
        if not isinstance(privileged, stere_super_datas__torch.Tensor) or privileged.shape != (371,) or (not privileged.is_floating_point()) or (not isinstance(valid, stere_super_datas__torch.Tensor)) or (valid.shape != (371,)) or (valid.dtype != stere_super_datas__torch.bool) or (valid.device != privileged.device):
            raise ValueError('371D labels and explicit per-dimension validity required')
        if bool((valid & ~stere_super_datas__torch.isfinite(privileged)).any()):
            raise ValueError('valid 371D labels must be finite')
        if row.get('action_owner') not in ('teacher', 'student'):
            raise ValueError('whole-action owner required')
        expected = row['teacher_mean'] if row['action_owner'] == 'teacher' else row.get('student_nominal')
        if expected is None or not stere_super_datas__torch.equal(row['issued_actions'], expected):
            raise ValueError('issued action differs from declared complete owner')
        execution = row.get('execution', {})
        if execution.get('physics_tick_start') != identity['physics_tick_end'] or execution.get('physics_tick_end', -1) <= identity['physics_tick_end'] or execution.get('simulation_timestamp_start') != identity['simulation_timestamp'] or (execution.get('simulation_timestamp_end', -1) <= identity['simulation_timestamp']):
            raise ValueError('executed physics/time range must follow the preserved pre-action identity')
        env = (identity['run_uuid'], identity['env_id'])
        old = previous.get(env)
        if old is not None:
            old_id = old['identity']
            same = identity['episode_uid'] == old_id['episode_uid']
            if identity['simulation_timestamp'] <= old_id['simulation_timestamp'] or identity['frame_id'] <= old_id['frame_id'] or identity['physics_tick_start'] < old['execution']['physics_tick_end']:
                raise ValueError('sample time/frame/physics clocks must remain ordered')
            if same:
                if bool(old['done']) or identity['reset_id'] != old_id['reset_id'] or identity['policy_tick'] != old_id['policy_tick'] + 1:
                    raise ValueError('episode ticks must be ordered without crossing terminal/reset')
                if ('episode_policy_tick' in identity or 'episode_policy_tick' in old_id) and ('episode_policy_tick' not in identity or 'episode_policy_tick' not in old_id or identity['episode_policy_tick'] != old_id['episode_policy_tick'] + 1):
                    raise ValueError('episode-local policy ticks must increase exactly once')
                if identity.get('trial_episode_index') != old_id.get('trial_episode_index'):
                    raise ValueError('trial episode index changed within the same episode UID')
                if not stere_super_datas__torch.equal(history.proprio[0, -1, 38:], old['issued_actions']):
                    raise ValueError('previous actually-issued action mismatch')
            elif not bool(old['done']) or identity['reset_id'] <= old_id['reset_id'] or int(history.history_valid.sum()) != 1 or (not bool((history.proprio[0, -1, 38:] == 0).all())):
                raise ValueError('reset boundary has contaminated sensor history or episode label')
        previous[env] = row
    return rows

def load_deployment_actor(path, expected_calibration_sha256=None):
    payload = stere_check__torch.load(stere_check__Path(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('version') != DEPLOYMENT_VERSION:
        raise ValueError('independent stereo deployment version required; legacy visual3844 forbidden')
    allowed = {'version', 'actor', 'actor_sha256', 'actor_contract', 'network_config', 'calibration_sha256', 'preprocessor'}
    if set(payload) != allowed:
        raise ValueError('actor-only package must contain no critic/Teacher/optimizer/env')
    network = payload['network_config']
    calibration = payload['calibration_sha256']
    _validate_actor_contract(payload['actor_contract'], network, calibration)
    if payload['preprocessor'] != payload['actor_contract']['preprocessor']:
        raise ValueError('preprocessor contract mismatch')
    if expected_calibration_sha256 is not None and calibration != expected_calibration_sha256:
        raise ValueError('deployment calibration SHA mismatch')
    if payload['actor_sha256'] != tensor_state_sha256(payload['actor']):
        raise ValueError('actor tensor SHA mismatch')
    actor = make_stereo_actor(network)
    _validate_state(payload['actor'], actor)
    actor.load_state_dict(payload['actor'], strict=True)
    actor.eval()
    actor.deployment_contract = stere_check__deepcopy(payload['actor_contract'])
    actor.network_config = stere_check__deepcopy(network)
    actor.calibration_sha256 = calibration
    actor.deployment_package_artifact = dict(path=str(stere_check__Path(path).resolve()), sha256=stere_check__file_sha256(path))
    return actor

def _success_pool(data):
    stere_super_datas__require_teacher(data.get('qualification'), data['source_sha256'])
    from dh116.rl.qualification import qualified_episode_identities
    qualified = qualified_episode_identities(data['qualification'])
    groups = {}
    for row in data['rows']:
        identity = row['identity']
        key = tuple((identity[k] for k in ('run_uuid', 'env_id', 'episode_uid', 'reset_id')))
        if key not in qualified:
            raise ValueError('collected episode is not bound to native success proof')
        groups.setdefault(key, []).append(row)
    for episode in groups.values():
        if episode[0]['identity'].get('episode_policy_tick', episode[0]['identity']['policy_tick']) != 0 or not bool(episode[-1]['done']) or any((bool(row['done']) for row in episode[:-1])):
            raise ValueError('complete first episode from reset through terminal required')
        for row in episode:
            info = row['info']
            if not bool(info.get('first_episode', False)) or bool(info.get('recovered', True)) or bool(info.get('restored', True)):
                raise ValueError('successful BC requires first native episode without recovery/restore')
        if not bool(episode[-1]['info'].get('success', False)):
            raise ValueError('failed/incomplete episode cannot become a success BC pool')

def validate_dataset(data, *, allow_schema_only=False):
    if not isinstance(data, dict) or data.get('version') != stere_super_datas__VERSION:
        raise ValueError('versioned stereo dataset required')
    stere_super_datas__validate_contract(data.get('contract'))
    if data['contract'].get('kind') not in ('bc', 'dagger'):
        raise ValueError('BC/DAgger dataset contract required')
    stere_super_datas__require_sha(data.get('source_sha256'), 'Teacher checkpoint/source SHA256')
    if type(data.get('schema_only')) is not bool:
        raise ValueError('explicit schema-only flag required')
    if data['schema_only'] and (not allow_schema_only):
        raise ValueError('schema-only CPU fixture cannot qualify native dataset')
    if (data['schema_only'] or data.get('purpose') == 'dagger_visited_states') and data.get('successful_demonstrations') is not False:
        raise ValueError('schema fixtures and visited failure states cannot be successful demonstrations')
    if data.get('native_qualified') is not False:
        raise ValueError('dataset schema validation must not claim native qualification')
    validate_rows(data.get('rows'), contract=data['contract'], schema_only=data['schema_only'])
    if not data['schema_only']:
        if data.get('purpose') == 'successful_bc':
            _success_pool(data)
        elif data.get('purpose') == 'dagger_visited_states':
            stere_super_datas__require_teacher(data.get('qualification'), data['source_sha256'])
        else:
            raise ValueError('explicit production dataset purpose required')
    return data

def validate_sealed_stereo_dataset(data, *, allow_schema_only=False):
    """Validate completed disk sources before the existing qualification gates."""
    if not isinstance(data, dict):
        raise ValueError('versioned stereo dataset required')
    if data.get('schema_only') is not True:
        _stere_super_datas_import__verify_stereo_dataset_sources = verify_stereo_dataset_sources
        _stere_super_datas_import__verify_stereo_dataset_sources(data)
    return validate_dataset(data, allow_schema_only=allow_schema_only)

def build_dataset(records, *, source_sha256, contract=None, schema_only=False, qualification=None, purpose='successful_bc', source_bindings=None):
    rows = rows_from_records(list(records))
    if not rows:
        raise ValueError('nonempty executed records required')
    if contract is None:
        contract = stere_super_datas__make_contract('dagger' if purpose == 'dagger_visited_states' else 'bc', {}, calibration_sha256=rows[0]['identity']['calibration_sha256'])
    data = dict(version=stere_super_datas__VERSION, contract=stere_super_datas__copy.deepcopy(contract), source_sha256=source_sha256, schema_only=schema_only, native_qualified=False, validation_scope='schema_only_not_native_success', purpose=purpose, successful_demonstrations=False if schema_only or purpose != 'successful_bc' else True, qualification=str(qualification) if isinstance(qualification, stere_super_datas__Path) else stere_super_datas__copy.deepcopy(qualification), source_bindings=stere_super_datas__copy.deepcopy(source_bindings), rows=rows)
    return validate_dataset(data, allow_schema_only=schema_only)

def save_dataset(data, path):
    validate_dataset(data, allow_schema_only=data.get('schema_only') is True)
    path = stere_super_datas__Path(path)
    with path.open('xb') as stream:
        stere_super_datas__torch.save(stere_super_datas__clone_tree(data, cpu=True), stream)
    return dict(path=str(path), sha256=stere_super_datas__file_sha256(path), samples=len(data['rows']), native_qualified=False)

def load_dataset(path, *, allow_schema_only=False, expected_sha256=None, expected_contract=None):
    if expected_sha256 is not None and stere_super_datas__file_sha256(path) != stere_super_datas__require_sha(expected_sha256):
        raise ValueError('dataset bytes SHA mismatch')
    data = stere_super_datas__torch.load(path, map_location='cpu', weights_only=True)
    validate_sealed_stereo_dataset(data, allow_schema_only=allow_schema_only)
    if expected_contract is not None:
        stere_super_datas__validate_contract(data['contract'], expected=expected_contract)
    return data

def _stack_trees(rows):
    if not rows:
        return {}
    keys = set.intersection(*(set(row) for row in rows))
    result = {}
    for key in keys:
        values = [row[key] for row in rows]
        if all((isinstance(x, stere_super_datas__torch.Tensor) for x in values)):
            result[key] = stere_super_datas__torch.stack(values)
        elif all((isinstance(x, dict) for x in values)):
            result[key] = _stack_trees(values)
    return result

def batch_dataset(data, indices, *, device='cpu'):
    validate_dataset(data, allow_schema_only=data.get('schema_only') is True)
    indices = [int(i) for i in indices]
    if not indices or any((i < 0 or i >= len(data['rows']) for i in indices)):
        raise ValueError('nonempty valid dataset indices required')
    rows = [data['rows'][i] for i in indices]
    history = stere_super_datas__StereoHistoryBatch.cat([stere_super_datas__StereoHistoryBatch(**row['history']) for row in rows]).to(device)
    targets = _stack_trees([row['extra_labels'] for row in rows])
    targets['privileged_state'] = stere_super_datas__torch.stack([row['privileged'] for row in rows])
    validity = _stack_trees([row['label_validity'] for row in rows])
    validity['privileged_state'] = stere_super_datas__torch.stack([row['privileged_valid'] for row in rows])
    validity['state_sources'] = _stack_trees([row['state_sources'] for row in rows])

    def move(tree):
        return {k: v.to(device) if isinstance(v, stere_super_datas__torch.Tensor) else move(v) for (k, v) in tree.items()}
    force = None
    bindings = [row.get('force_qualification') for row in rows]
    if all((isinstance(binding, dict) for binding in bindings)):
        mask_keys = ('source_valid', 'contact_valid', 'attributable')
        metadata = [{k: v for (k, v) in binding.items() if k not in mask_keys} for binding in bindings]
        if all((item == metadata[0] for item in metadata[1:])):
            force = stere_super_datas__copy.deepcopy(metadata[0])
            for key in mask_keys:
                if not all((isinstance(binding.get(key), stere_super_datas__torch.Tensor) for binding in bindings)):
                    force = None
                    break
                force[key] = stere_super_datas__torch.stack([binding[key] for binding in bindings]).to(device)
    return dict(history=history, teacher_actions=stere_super_datas__torch.stack([row['teacher_mean'] for row in rows]).detach().to(device), action_valid=stere_super_datas__torch.ones(len(rows), 24, dtype=stere_super_datas__torch.bool, device=device), targets=move(targets), validity=move(validity), force_qualification=force, schema_only=data['schema_only'], native_qualified=False, _dataset=data, indices=indices)

def split_by_initial_condition(data, *, heldout_initial_condition_sha256=()):
    validate_dataset(data, allow_schema_only=data.get('schema_only') is True)
    heldout = set(heldout_initial_condition_sha256)
    known = {row['identity']['physical_initial_condition_sha256'] for row in data['rows']}
    if heldout - known:
        raise ValueError('unknown physical initial condition holdout SHA')
    test_hashes = set(heldout)
    test_sources = set()
    changed = True
    while changed:
        old = (len(test_hashes), len(test_sources))
        for row in data['rows']:
            ident = row['identity']
            if ident['physical_initial_condition_sha256'] in test_hashes or ident['source_episode_group'] in test_sources:
                test_hashes.add(ident['physical_initial_condition_sha256'])
                test_sources.add(ident['source_episode_group'])
        changed = old != (len(test_hashes), len(test_sources))
    train = []
    test = []
    for (i, row) in enumerate(data['rows']):
        (test if row['identity']['physical_initial_condition_sha256'] in test_hashes else train).append(i)
    if not train:
        raise ValueError('holdout consumed every independent initial-condition/source group')
    return dict(train_indices=train, heldout_indices=test, split_unit='physical_initial_condition_and_source_episode_connected_groups')
stere_bc__batch_dataset = batch_dataset
stere_bc__validate_dataset = validate_dataset
stere_dagge_datas__VERSION = stere_super_datas__VERSION
stere_dagge_datas__rows_from_records = rows_from_records
stere_dagge_datas__validate_dataset = validate_dataset
stere_dagge_datas__save_dataset = save_dataset
stere_dagge_datas__load_dataset = load_dataset
stere_dagge_datas__batch_dataset = batch_dataset
DEFAULT_CONFIG = dict(lr=0.0003, lambda_action=1.0, lambda_state=1.0, lambda_depth=0.0, lambda_temporal=0.0, max_grad_norm=0.5)

class DAggerDataset:

    def __init__(self, *, source_sha256, contract=None, schema_only=False, qualification=None):
        self.source_sha256 = source_sha256
        self.contract = stere_dagge_datas__copy.deepcopy(contract)
        self.schema_only = schema_only
        self.qualification = stere_dagge_datas__copy.deepcopy(qualification)
        self._rows = []

    def __len__(self):
        return len(self._rows)

    def _payload(self, rows, contract):
        return dict(version=stere_dagge_datas__VERSION, contract=stere_dagge_datas__copy.deepcopy(contract), source_sha256=self.source_sha256, schema_only=self.schema_only, native_qualified=False, validation_scope='schema_only_not_native_success', purpose='dagger_visited_states', successful_demonstrations=False, qualification=stere_dagge_datas__copy.deepcopy(self.qualification), rows=stere_dagge_datas__copy.deepcopy(rows))

    def append(self, record):
        records = record if isinstance(record, list) else [record]
        if any((row.get('mode') == 'teacher_reference' for row in records)):
            raise ValueError('D0 Teacher-only reference belongs in a reference pool, not Student DAgger')
        incoming = stere_dagge_datas__rows_from_records(records)
        if not incoming:
            raise ValueError('nonempty executed visited-state rows required')
        contract = self.contract
        if contract is None:
            contract = stere_dagge_datas__make_contract('dagger', {}, calibration_sha256=incoming[0]['identity']['calibration_sha256'])
        candidate = self._payload(self._rows + incoming, contract)
        stere_dagge_datas__validate_dataset(candidate, allow_schema_only=self.schema_only)
        self.contract = stere_dagge_datas__copy.deepcopy(contract)
        self._rows = candidate['rows']
        return len(incoming)

    def payload(self):
        value = self._payload(self._rows, self.contract)
        return stere_dagge_datas__validate_dataset(value, allow_schema_only=self.schema_only)

    def batch(self, indices, *, device='cpu'):
        return stere_dagge_datas__batch_dataset(self.payload(), indices, device=device)

    def save(self, path):
        return stere_dagge_datas__save_dataset(self.payload(), path)

    @classmethod
    def load(cls, path, *, allow_schema_only=False, expected_sha256=None):
        payload = stere_dagge_datas__load_dataset(path, allow_schema_only=allow_schema_only, expected_sha256=expected_sha256)
        if payload.get('purpose') != 'dagger_visited_states':
            raise ValueError('DAgger visited-state pool required')
        result = cls(source_sha256=payload['source_sha256'], contract=payload['contract'], schema_only=payload['schema_only'], qualification=payload['qualification'])
        result._rows = stere_dagge_datas__copy.deepcopy(payload['rows'])
        return result

def _equal(a, b):
    if isinstance(a, stere_bc__torch.Tensor) and isinstance(b, stere_bc__torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and bool(stere_bc__torch.isclose(a, b.to(a.device), rtol=0, atol=0, equal_nan=True).all())
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all((_equal(a[k], b[k]) for k in a))
    return a == b

class StereoBCTrainer:

    def __init__(self, actor, config=None, *, ppo_trainer=None, schema_only=False):
        (self.actor, self.ppo_trainer, self.schema_only) = (actor, ppo_trainer, schema_only)
        self.config_metadata = {} if config is None else dict(config)
        if self.config_metadata.get('schema_version', 'dh116-stereo-bc-v1') != 'dh116-stereo-bc-v1':
            raise ValueError('versioned stereo BC configuration required')
        if any((key.startswith('lambda_') and key not in DEFAULT_CONFIG for key in self.config_metadata)):
            raise ValueError('unknown BC loss weight')
        self.config = DEFAULT_CONFIG | {k: v for (k, v) in self.config_metadata.items() if k in DEFAULT_CONFIG}
        if any((not isinstance(v, (int, float)) or not stere_bc__math.isfinite(v) or v < 0 for v in self.config.values())):
            raise ValueError('finite nonnegative BC hyperparameters required')
        if self.config['lr'] <= 0 or self.config['max_grad_norm'] <= 0:
            raise ValueError('positive learning rate and gradient bound required')
        variance = getattr(actor, 'log_std', None)
        self.parameters = [p for p in actor.parameters() if p.requires_grad and p is not variance]
        self.optimizer = stere_bc__torch.optim.Adam(self.parameters, lr=self.config['lr'])
        self.state_loss = stere_bc__StereoEstimationLoss()
        self.steps = 0

    def step(self, batch):
        modes = [(module, module.training) for module in self.actor.modules()]
        self.actor.train()
        try:
            return self._step(batch)
        finally:
            for (module, training) in modes:
                module.training = training

    def _step(self, batch):
        if self.ppo_trainer is not None and (not self.ppo_trainer.clean_boundary):
            raise RuntimeError('BC requires a clean empty PPO rollout boundary')
        if not isinstance(batch, dict) or '_dataset' not in batch:
            raise ValueError('batch must bind its admitted source dataset')
        dataset = stere_bc__validate_dataset(batch['_dataset'], allow_schema_only=self.schema_only)
        if dataset['schema_only'] and (not self.schema_only):
            raise ValueError('CPU fixture BC requires explicit schema_only=True')
        device = next(self.actor.parameters()).device
        expected = stere_bc__batch_dataset(dataset, batch['indices'], device=device)
        for name in ('teacher_actions', 'action_valid', 'targets', 'validity'):
            if not _equal(batch[name], expected[name]):
                raise ValueError('BC batch differs from the admitted dataset: ' + name)
        if not _equal(batch['history'].as_dict(), expected['history'].as_dict()):
            raise ValueError('BC sensor history differs from admitted dataset')
        history = expected['history']
        target = expected['teacher_actions'].detach()
        output = None
        if hasattr(self.actor, 'estimator'):
            output = self.actor.estimator(history)
            features = self.actor.features_from_estimates(output, history)
            nominal = self.actor.action_net(features).tanh()
            state = self.state_loss(output, expected['targets'], expected['validity'], force_qualification=expected['force_qualification'])
            state_value = state['total']
        else:
            nominal = self.actor.deterministic(history)
            state_value = nominal.sum() * 0.0
            state = {'coverage': {}, 'blocked_reasons': {}}
        action = stere_bc__masked_group_normalized_loss(nominal, target, expected['action_valid'])
        terms = dict(action=action, state=state_value)
        terms['depth'] = terms['temporal'] = nominal.sum() * 0.0
        if self.config['lambda_depth'] > 0:
            if output is None or 'depth_m' not in output or 'depth_m' not in expected['targets'] or ('depth_m' not in expected['validity']):
                raise ValueError('depth loss requires an actual depth prediction head and qualified target/validity; no GT fallback')
            terms['depth'] = stere_bc__masked_smooth_l1(output['depth_m'], expected['targets']['depth_m'], expected['validity']['depth_m'])
        if self.config['lambda_temporal'] > 0:
            if output is None or 'velocity' not in expected['targets'] or 'velocity' not in expected['validity'] or ('temporal_correspondence' not in expected['validity']):
                raise ValueError('temporal loss requires qualified velocity and cross-frame point correspondence masks')
            previous = history.clone()
            previous.history_valid[:, -1] = False
            prior_geometry = self.actor.estimator(previous)['geometry']
            predicted_velocity = (output['geometry'] - prior_geometry) / history.dt[:, -1, None, None]
            correspondence = expected['validity']['temporal_correspondence']
            if correspondence.shape == predicted_velocity.shape[:-1]:
                correspondence = correspondence[..., None].expand_as(predicted_velocity)
            if correspondence.shape != predicted_velocity.shape or correspondence.dtype != stere_bc__torch.bool:
                raise ValueError('temporal correspondence mask must match point IDs [B,128] or [B,128,3]')
            valid = expected['validity']['velocity'] & correspondence & previous.history_valid.any(1)[:, None, None]
            terms['temporal'] = stere_bc__masked_smooth_l1(predicted_velocity, expected['targets']['velocity'], valid)
        total = sum((self.config['lambda_' + name] * value for (name, value) in terms.items()))
        norms = {}
        for (name, value) in terms.items():
            if value.requires_grad:
                gradient = stere_bc__torch.autograd.grad(value, self.parameters, retain_graph=True, allow_unused=True)
                norm = sum((g.detach().square().sum() for g in gradient if g is not None), value.new_zeros(())).sqrt()
                norms[name] = float(norm)
            else:
                norms[name] = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = stere_bc__torch.nn.utils.clip_grad_norm_(self.parameters, self.config['max_grad_norm'], error_if_nonfinite=True)
        self.optimizer.step()
        self.steps += 1
        return dict(steps=self.steps, state_loss_version=self.state_loss.loss_version, losses={k: float(v.detach()) for (k, v) in terms.items()} | {'total': float(total.detach())}, gradient_norms=norms, gradient_norm_before_clip=float(norm), coverage=state['coverage'], blocked_reasons=state['blocked_reasons'], schema_only=dataset['schema_only'], native_qualified=False, ppo_storage_modified=False)
stere_dagge_train__StereoBCTrainer = StereoBCTrainer

class StereoDAggerTrainer(stere_dagge_train__StereoBCTrainer):

    def train(self, dataset, *, batch_size=32, epochs=1, generator=None):
        if type(batch_size) is not int or batch_size < 1 or type(epochs) is not int or (epochs < 1):
            raise ValueError('positive DAgger epochs and minibatch size required')
        if not len(dataset):
            raise ValueError('nonempty independently aggregated visited-state dataset required')
        reports = []
        for epoch in range(epochs):
            order = stere_dagge_train__torch.randperm(len(dataset), generator=generator).tolist()
            for start in range(0, len(order), batch_size):
                report = self.step(dataset.batch(order[start:start + batch_size]))
                report['epoch'] = epoch + 1
                reports.append(report)
        return reports
