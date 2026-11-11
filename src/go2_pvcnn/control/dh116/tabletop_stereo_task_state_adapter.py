"""Field-wise deployable task state, bound to the canonical 371D schema.

``privileged_state`` is the historical name of an *estimator output head*.
This adapter accepts that prediction and a strict sensor history, never a
critic observation or simulator/task mapping. Configured source eligibility
does not establish native supervision qualification or measurement accuracy.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from .tabletop_stereo_observation import StereoHistoryBatch


EXTRA_SHAPES = {'geometry': (128,3), 'velocity': (128,3), 'deformation': (128,),
                'contact_logits': (2,3), 'slip_logits': (2,),
                'slip_velocity': (2,), 'force_proxy': (2,3)}
_VISUAL_HEADS = {'geometry_head','velocity_head','contact_head','slip_head'}
_KNOWN_KEYS = {'joint_lower','joint_upper','reset_joint_reference',
               'grip_drive_stiffness','wrist_reset_xyz'}


def field_history_frames(row):
    """Number of valid frames required by a canonical field's runtime gate."""
    return 2 if (row['estimator_head'] in ('velocity_head','slip_head')
                 or row['block'].startswith('interval_')) else 1


class TaskStateAdapter(nn.Module):
    """Select declared fields in schema index order and preserve autograd.

    ``allowlist`` contains canonical indices, semantic names, or block names.
    ``known_config`` contains explicitly declared raw SI configuration values;
    joint limits are training normalization constants, not hardware approval.
    ``source_qualification`` maps schema validity keys to booleans or boolean
    [B]/[B,371] eligibility masks. Missing keys fail closed. Optional estimator
    sidebands ``state_valid`` and ``state_uncertainty`` are [B,371]; uncertainty
    is nonnegative in each normalized field's units. Absent uncertainty uses 1
    as an explicit unknown sentinel, never a zero error claim.

    Extra visual heads are opt-in and append to the selected 371D fields.
    Their units/coordinates are explicitly unbound until ``extra_head_metadata``
    declares them; no extra node is relabelled as a canonical 371D quantity.
    """

    def __init__(self, *, allowlist=None, known_config=None, source_qualification=None,
                 extra_heads=(), extra_head_metadata=None):
        super().__init__()
        path = Path(__file__).resolve().parents[4]/'docs/contracts/stereo_privileged_state_contract.json'
        raw = path.read_bytes()
        schema = json.loads(raw.decode('utf-8-sig'))
        self.state_contract_sha256 = hashlib.sha256(raw).hexdigest()
        rows = schema['dimensions']
        if schema.get('size') != 371 or [r['index'] for r in rows] != list(range(371)):
            raise ValueError('complete canonical 371D schema required')
        self._rows = copy.deepcopy(rows)
        self.known_config = {} if known_config is None else copy.deepcopy(dict(known_config))
        unknown = set(self.known_config)-_KNOWN_KEYS
        if unknown:
            raise ValueError('unknown known_config fields: '+','.join(sorted(unknown)))
        self._validate_known_config()
        self.source_qualification = {} if source_qualification is None else dict(source_qualification)
        source_keys = {key for row in rows for key in row['required_validity_keys']}
        unknown = set(self.source_qualification)-(source_keys|{'rotation_symmetry_valid','force_proxy'})
        if unknown:
            raise ValueError('unknown source qualification keys: '+','.join(sorted(unknown)))
        for key,value in self.source_qualification.items():
            if key == 'force_proxy':
                if not isinstance(value,dict): raise TypeError('force_proxy raw qualification mapping required')
            elif type(value) is not bool and not (isinstance(value,torch.Tensor) and value.dtype == torch.bool):
                raise ValueError('source eligibility must be bool or boolean tensor: '+key)
        if allowlist is None:
            indices = [r['index'] for r in rows if self._deployable(r) and
                       (r['estimator_head'] in _VISUAL_HEADS or
                        r['block'] in ('joint_position','joint_velocity','last_submitted_action24'))]
        else:
            if isinstance(allowlist,(str,bytes)): raise TypeError('allowlist must be a sequence of fields')
            indices = []
            for field in allowlist:
                if type(field) is int:
                    matches = [field] if 0 <= field < 371 else []
                elif isinstance(field,str):
                    matches = [r['index'] for r in rows if field in (r['semantic_name'],r['block'])]
                else:
                    raise TypeError('allowlist fields must be canonical indices or names')
                if not matches: raise ValueError('unknown schema allowlist field: '+str(field))
                if any(not self._deployable(rows[i]) for i in matches):
                    raise ValueError('NA/forbidden schema field is not deployable: '+str(field))
                indices.extend(matches)
        self.field_indices = tuple(sorted(set(indices)))
        self.field_order = tuple(rows[i]['semantic_name'] for i in self.field_indices)
        if isinstance(extra_heads,str): raise TypeError('extra_heads must be a sequence')
        self.extra_heads = tuple(extra_heads)
        if len(set(self.extra_heads)) != len(self.extra_heads) or any(k not in EXTRA_SHAPES for k in self.extra_heads):
            raise ValueError('unknown or duplicate extra estimator head')
        self.extra_head_metadata = {} if extra_head_metadata is None else copy.deepcopy(dict(extra_head_metadata))
        if set(self.extra_head_metadata)-set(self.extra_heads):
            raise ValueError('extra_head_metadata requires a selected extra head')
        self.value_size = len(self.field_indices)+sum(math.prod(EXTRA_SHAPES[k]) for k in self.extra_heads)
        if not self.value_size: raise ValueError('at least one task state field required')
        self.input_size = self.feature_size = 3*self.value_size

    @staticmethod
    def _deployable(row):
        return (row['deployment_allowed'] is True and row['supervision_status'] != 'NA'
                and row['student_source'] != 'unobservable' and 'supervision_forbidden' not in row['required_validity_keys'])

    def _validate_known_config(self):
        shapes = {'joint_lower':((6,),(12,)), 'joint_upper':((6,),(12,)),
                  'reset_joint_reference':((12,),), 'grip_drive_stiffness':((2,),),
                  'wrist_reset_xyz':((2,3),)}
        for key,value in self.known_config.items():
            value = torch.as_tensor(value,dtype=torch.float32)
            if tuple(value.shape) not in shapes[key] or not bool(torch.isfinite(value).all()):
                raise ValueError('known_config requires finite raw SI values and declared shape: '+key)
            self.known_config[key] = value.tolist()
        if ('joint_lower' in self.known_config) != ('joint_upper' in self.known_config):
            raise ValueError('joint_lower and joint_upper must both be explicitly supplied')
        if 'joint_lower' in self.known_config:
            low,high = (torch.tensor(self.known_config[k]).repeat(2 if len(self.known_config[k])==6 else 1)
                        for k in ('joint_lower','joint_upper'))
            if not bool((high>low).all()): raise ValueError('joint normalization upper must exceed lower')

    @staticmethod
    def _prediction(value, shape, reference, name):
        if (not isinstance(value,torch.Tensor) or tuple(value.shape) != tuple(shape)
                or not value.is_floating_point() or value.device != reference.device):
            raise ValueError(name+' must be floating prediction with shape '+str(tuple(shape))+' on history device')
        return value

    def _eligibility(self, key, reference, index=None):
        value = self.source_qualification.get(key,False)
        if type(value) is bool: return torch.full((reference.shape[0],),value,dtype=torch.bool,device=reference.device)
        value = value.to(reference.device)
        if value.shape == reference.shape[:1]: return value
        if value.shape == (reference.shape[0],371) and index is not None: return value[:,index]
        raise ValueError('source eligibility mask shape must be [B] or [B,371]: '+key)

    def _known(self, key, reference):
        return torch.as_tensor(self.known_config[key],device=reference.device,dtype=reference.dtype)

    def _joint_normalization(self, value, reference):
        if 'joint_lower' not in self.known_config: return None
        low,high = (self._known(key,reference) for key in ('joint_lower','joint_upper'))
        if low.numel()==6: low=low.repeat(2)
        if high.numel()==6: high=high.repeat(2)
        return 2*(value-low)/(high-low)-1

    def _passthrough(self, row, history, prediction):
        """Return a field only where the 62D sensor contract supplies its source."""
        p = history.proprio[:,-1]
        block,col = row['block'],row['block_column']
        if block == 'joint_position':
            q = self._joint_normalization(p[:,:12],prediction)
            return (None if q is None else q[:,col]), 'history.proprio.q; explicit normalization limits'
        if block == 'joint_velocity': return p[:,12+col]/.3, 'history.proprio.qd'
        if block == 'last_submitted_action24': return p[:,38+col], 'history.proprio.previous_actually_issued_action24'
        if block == 'reset_joint_reference' and block in self.known_config:
            value = self._joint_normalization(self._known(block,prediction),prediction)
            if value is not None: return value[col].expand(len(p)), 'declared known_config.reset_joint_reference'
        if block == 'grip_drive_stiffness' and block in self.known_config:
            return ((self._known(block,prediction)[col]-10)/10).expand(len(p)), 'declared known_config.grip_drive_stiffness'
        if block == 'wrist_pose_from_reset':
            return None, 'sensor palm_link FK is not canonical wrist root; same-frame palm-to-root calibration unavailable'
        # The history has no measured mount joint angles, authored target,
        # native wrist velocity origin, or completed native interval boundaries.
        # Pose differencing/Euler extraction cannot silently stand in for these.
        return None, 'unavailable in declared 62D sensor history/configuration'

    def _canonical_fields(self, estimates, history, prediction):
        b = len(history)
        supplied = estimates.get('state_valid')
        if supplied is None: supplied=torch.ones_like(prediction,dtype=torch.bool)
        if (not isinstance(supplied,torch.Tensor) or supplied.dtype != torch.bool
                or supplied.shape != prediction.shape or supplied.device != prediction.device):
            raise ValueError('state_valid mask must be boolean [B,371] on prediction device')
        uncertainty = estimates.get('state_uncertainty')
        if uncertainty is not None:
            self._prediction(uncertainty,prediction.shape,prediction,'state_uncertainty')
            if bool((~torch.isfinite(uncertainty)|(uncertainty<0)).any()):
                raise ValueError('state_uncertainty must be finite and nonnegative')
        current = history.history_valid[:,-1]
        pair = history.image_valid[:,-1].all(-1)
        observed = history.point_valid[:,-1].any(-1) | history.depth_valid[:,-1].flatten(1).any(-1)
        values,masks,uncertainties,metadata = [],[],[],[]
        blocked = {}
        for index in self.field_indices:
            row = self._rows[index]
            visual = row['estimator_head'] in _VISUAL_HEADS
            mask = current.clone()
            for key in row['required_validity_keys']: mask = mask & self._eligibility(key,prediction,index)
            if visual:
                value = prediction[:,index]
                mask = mask & pair & observed & supplied[:,index]
                if field_history_frames(row) == 2:
                    mask=mask & ((history.history_valid & history.image_valid.all(-1)).sum(-1)>=2)
                if row['block'] == 'object_quaternion':
                    mask = mask & (prediction[:,64:68].norm(dim=-1)>1e-8)
                    mask = mask & self._eligibility('rotation_symmetry_valid',prediction,index)
                # Mask before nonlinear operations: invalid NaNs never create a
                # NaN gradient through sigmoid or downstream Actor arithmetic.
                mask = mask & torch.isfinite(value)
                value = torch.where(mask,value,0.)
                if row['loss_type']=='classification': value = value.sigmoid()
                source = 'estimated371.'+row['estimator_head']
                uncert = torch.ones(b,device=prediction.device,dtype=prediction.dtype) if uncertainty is None else uncertainty[:,index]
                uncertainty_source = 'unknown sentinel 1; not calibrated' if uncertainty is None else 'estimated normalized-field uncertainty sideband; calibration unverified'
            else:
                value,source = self._passthrough(row,history,prediction)
                if value is None:
                    value = torch.zeros(b,device=prediction.device,dtype=prediction.dtype)
                    mask = mask & False
                    blocked[row['semantic_name']] = source
                mask = mask & torch.isfinite(value)
                # Sensor accuracy is not available in the 62D payload.
                uncert = torch.ones_like(value)
                uncertainty_source = 'unknown sensor/configuration uncertainty sentinel 1'
            values.append(torch.where(mask,value,0.))
            masks.append(mask)
            uncertainties.append(torch.where(mask,uncert,torch.ones_like(uncert)))
            metadata.append(dict(index=index, semantic_name=row['semantic_name'],unit=row['unit'],
                coordinate_frame=row['coordinate_frame'],normalization=row['normalization'],
                source=source,validity=list(row['required_validity_keys']),
                validity_description=row['validity_mask'],uncertainty=uncertainty_source,
                role='estimated_contact_probability_proxy' if visual and row['loss_type']=='classification' else
                     'estimated_task_field' if visual else 'declared_sensor_or_configuration_passthrough'))
        return values,masks,uncertainties,metadata,blocked

    def _extra_fields(self, estimates, history, prediction):
        values,masks,uncertainties,metadata,blocked = [],[],[],[],{}
        b = len(history)
        for name in self.extra_heads:
            shape = (b,*EXTRA_SHAPES[name]); width=math.prod(EXTRA_SHAPES[name])
            if name not in estimates: raise ValueError('missing selected extra prediction: '+name)
            raw = self._prediction(estimates[name],shape,prediction,name)
            gate = history.history_valid[:,-1] & history.image_valid[:,-1].all(-1)
            gate = gate & history.point_valid[:,-1].any(-1)
            declared=self.extra_head_metadata.get(name,{})
            if not isinstance(declared,dict):raise ValueError('extra head metadata must be a mapping')
            if name in ('geometry','velocity','deformation','slip_velocity') and not all(
                    isinstance(declared.get(k),str) and declared[k].strip() for k in ('unit','coordinate_frame')):
                gate=gate & False
                blocked[name]='extra visual units and coordinate_frame are not declared'
            if name in ('velocity','slip_velocity','slip_logits'):
                gate=gate & ((history.history_valid & history.image_valid.all(-1)).sum(-1)>=2)
            for key in ('source_valid','identity_aligned','timestamp_valid','visibility_valid','correspondence_valid'):
                gate = gate & self._eligibility(key,prediction)
            valid = gate.reshape(b,*([1]*(raw.ndim-1))).expand_as(raw) & torch.isfinite(raw)
            if name == 'force_proxy':
                # Same raw artifact/proof verifier as supervision. Declaration,
                # a digest string, or a CPU fixture cannot grant qualification.
                from .tabletop_stereo_losses import StereoEstimationLoss
                qualification = self.source_qualification.get('force_proxy')
                qualified,reason = StereoEstimationLoss._force_mask(qualification,raw)
                valid = valid & qualified
                if reason: blocked[name]=reason
            supplied = estimates.get(name+'_valid')
            if supplied is not None:
                if not isinstance(supplied,torch.Tensor) or supplied.dtype!=torch.bool or supplied.shape!=raw.shape or supplied.device!=raw.device:
                    raise ValueError('extra validity mask shape must match prediction: '+name)
                valid = valid & supplied
            value = torch.where(valid,raw,0.)
            if name in ('contact_logits','slip_logits'): value=value.sigmoid()
            value=torch.where(valid,value,0.)
            uncertainty=estimates.get(name+'_uncertainty')
            if uncertainty is None: uncertainty=torch.ones_like(raw)
            else:
                self._prediction(uncertainty,shape,prediction,name+'_uncertainty')
                if bool((~torch.isfinite(uncertainty)|(uncertainty<0)).any()):
                    raise ValueError('extra uncertainty must be finite nonnegative: '+name)
            values.extend(value.reshape(b,width).unbind(-1))
            masks.extend(valid.reshape(b,width).unbind(-1))
            uncertainties.extend(torch.where(valid,uncertainty,torch.ones_like(uncertainty)).reshape(b,width).unbind(-1))
            declared=self.extra_head_metadata.get(name,{})
            if not isinstance(declared,dict): raise ValueError('extra head metadata must be a mapping')
            unit = 'N' if name=='force_proxy' else 'probability' if name.endswith('logits') else declared.get('unit','unbound estimator units')
            frame = 'accepted_pad_normal' if name=='force_proxy' else declared.get('coordinate_frame','unbound estimator head coordinates')
            for col in range(width):
                metadata.append(dict(index=None,semantic_name=f'{name}[{col}]',unit=unit,
                    coordinate_frame=frame,normalization=declared.get('normalization','no additional normalization'),
                    source='estimated_extra_head.'+name,validity='sensor/eligibility and explicit per-head mask; raw proof additionally required for force',
                    uncertainty='unknown sentinel 1 unless explicit per-head uncertainty supplied; calibration unverified',
                    role='additional_estimated_feature; not canonical371 field'))
        return values,masks,uncertainties,metadata,blocked

    def forward(self, estimates, history):
        if type(history) is not StereoHistoryBatch:
            raise TypeError('TaskStateAdapter requires StereoHistoryBatch; privileged mappings forbidden')
        history.validate()
        if not isinstance(estimates,dict): raise TypeError('named estimator outputs required')
        allowed={'latent','privileged_state','state_valid','state_uncertainty'}|set(EXTRA_SHAPES)
        allowed |= {key+suffix for key in EXTRA_SHAPES for suffix in ('_valid','_uncertainty')}
        if set(estimates)-allowed: raise ValueError('unknown/privileged estimator inputs forbidden')
        prediction = self._prediction(estimates.get('privileged_state'),(len(history),371),history.proprio,'estimated371')
        values,masks,uncertainties,fields,blocked=self._canonical_fields(estimates,history,prediction)
        extra=self._extra_fields(estimates,history,prediction)
        values.extend(extra[0]);masks.extend(extra[1]);uncertainties.extend(extra[2]);fields.extend(extra[3]);blocked.update(extra[4])
        value=torch.stack(values,-1);valid=torch.stack(masks,-1);uncertainty=torch.stack(uncertainties,-1)
        coverage=[]
        selected=set(self.field_indices)
        for row in self._rows:
            status='NA' if not self._deployable(row) else 'selected' if row['index'] in selected else 'excluded'
            coverage.append(dict(index=row['index'],semantic_name=row['semantic_name'],status=status,
                reason='schema forbids deployment/supervision' if status=='NA' else
                       'declared allowlist; per-sample validity required' if status=='selected' else 'outside configured task-state profile'))
        metadata=dict(fields=fields,schema_coverage=coverage,blocked_reasons=blocked,
            state_contract_file_sha256=self.state_contract_sha256,field_order=tuple(f['semantic_name'] for f in fields),
            feature_layout=('values','valid','uncertainty'),
            evidence_scope='runtime_configuration_not_native_qualification',
            uncertainty_policy='sentinel 1 is unknown; no calibrated uncertainty supplied by the estimator',
            missing_value_policy='NA in values; neutral finite encoding only in masked actor_features')
        return dict(values=torch.where(valid,value,float('nan')),valid=valid,uncertainty=uncertainty,metadata=metadata,
                    actor_features=torch.cat((value,valid.to(value.dtype),uncertainty),-1))
