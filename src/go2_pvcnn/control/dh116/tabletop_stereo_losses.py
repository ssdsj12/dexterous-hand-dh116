"""Source-qualified supervision; no truth is accepted by the Actor forward."""
import copy
import hashlib
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F


def _force_proof_verified(binding):
    """Proof/producer/raw-byte verification is shared with the promotion gate."""
    try:
        from .tabletop_stereo_promotion import verify_force_qualification
        return bool(verify_force_qualification(binding))
    except (ValueError,FileNotFoundError,ImportError):
        return False


def _mask(mask,reference):
    if not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool or mask.device!=reference.device:
        raise ValueError('boolean mask tensor on the prediction device required')
    if mask.shape==reference.shape[:-1]: mask=mask[...,None].expand_as(reference)
    if mask.shape!=reference.shape: raise ValueError('mask shape must match the supervised tensor')
    return mask


def _safe_pair(prediction,target,mask):
    if (not isinstance(target,torch.Tensor) or prediction.shape!=target.shape
            or prediction.device!=target.device or not prediction.is_floating_point()
            or not target.is_floating_point()):
        raise ValueError('matching floating target/prediction shape and device required')
    mask=_mask(mask,prediction)
    if bool((mask & (~torch.isfinite(prediction)|~torch.isfinite(target))).any()):
        raise ValueError('valid predictions and targets must be finite')
    return torch.where(mask,prediction,0.),torch.where(mask,target.detach(),0.),mask


def _reduce(values,mask,weights=None):
    weights=torch.ones_like(values) if weights is None else torch.as_tensor(weights,device=values.device,dtype=values.dtype)
    if bool((~torch.isfinite(weights)|(weights<0)).any()): raise ValueError('finite nonnegative loss weights required')
    weights=torch.broadcast_to(weights,values.shape)*mask
    return (values*weights).sum()/weights.sum().clamp_min(1e-8)


def masked_smooth_l1(prediction,target,mask,weights=None):
    p,t,m=_safe_pair(prediction,target,mask)
    return _reduce(F.smooth_l1_loss(p,t,reduction='none'),m,weights)


def masked_bce(prediction,target,mask):
    p,t,m=_safe_pair(prediction,target,mask)
    if bool((m & ((t<0)|(t>1))).any()): raise ValueError('classification labels must lie in [0,1]')
    return _reduce(F.binary_cross_entropy_with_logits(p,t,reduction='none'),m)


def quaternion_geodesic_loss(prediction,target,mask):
    if prediction.shape[-1]!=4 or mask.shape!=prediction.shape[:-1]:
        raise ValueError('wxyz groups [...,4] and group validity [...] required')
    p,t,m=_safe_pair(prediction,target,mask)
    identity=torch.zeros_like(p);identity[...,0]=1.
    pnorm=p.norm(dim=-1,keepdim=True);tnorm=t.norm(dim=-1,keepdim=True)
    if bool((mask & (tnorm[...,0]<=1e-8)).any()): raise ValueError('valid target quaternion must be nonzero')
    p=torch.where((mask[...,None]&(pnorm>1e-8)),p/pnorm.clamp_min(1e-8),identity)
    t=torch.where(mask[...,None],t/tnorm.clamp_min(1e-8),identity)
    t=torch.where(((p*t).sum(-1)<0)[...,None],-t,t)
    # This equivalent SO(3) formula has finite derivatives at q==+-target;
    # acos(abs(dot)) otherwise has an infinite endpoint derivative.
    angle=4*torch.atan2((p-t).norm(dim=-1),(p+t).norm(dim=-1))
    return _reduce(angle,mask)


def per_dimension_metrics(prediction,target,mask):
    """Masked MAE/RMSE/P95; no valid labels => NA (NaN), never fake zero error."""
    p,t,m=_safe_pair(prediction,target,mask)
    if p.ndim!=2: raise ValueError('metrics require [B,D]')
    error=(p-t).abs(); count=m.sum(0)
    denominator=count.clamp_min(1)
    nan=torch.full_like(denominator,float('nan'),dtype=p.dtype)
    mae=torch.where(count>0,(error*m).sum(0)/denominator,nan)
    rmse=torch.where(count>0,((error.square()*m).sum(0)/denominator).sqrt(),nan)
    p95=torch.stack([torch.quantile(error[:,i][m[:,i]],.95) if bool(m[:,i].any()) else nan[i]
                     for i in range(p.shape[1])])
    return dict(count=count,mae=mae,rmse=rmse,p95=p95)


class StereoEstimationLoss(nn.Module):
    def __init__(self,state_contract_path=None,*,state_contract=None,head_weights=None,expected_contract_sha256=None):
        super().__init__()
        if state_contract is None:
            path=Path(state_contract_path) if state_contract_path is not None else Path(__file__).resolve().parents[4]/'docs/contracts/stereo_privileged_state_contract.json'
            state_contract=json.loads(path.read_text(encoding='utf-8'))
        self.state_contract=copy.deepcopy(state_contract)
        self.state_contract_sha256=hashlib.sha256(json.dumps(self.state_contract,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if expected_contract_sha256 is not None and expected_contract_sha256!=self.state_contract_sha256:
            raise ValueError('state contract SHA256 mismatch')
        rows=self.state_contract.get('dimensions',[])
        if self.state_contract.get('size')!=371 or [r.get('index') for r in rows]!=list(range(371)):
            raise ValueError('complete canonical 371D state contract required')
        common=set(self.state_contract.get('runtime_mask_contract',{}).get('required_common_keys',[]))
        if (not common or self.state_contract.get('runtime_mask_contract',{}).get('default')!='fail_closed'
                or any(not isinstance(r.get('required_validity_keys'),list)
                    or not common.issubset(r['required_validity_keys'])
                    or r.get('loss_type') not in ('regression','classification','none')
                    or r.get('supervision_status') not in ('NA','conditional') for r in rows)):
            raise ValueError('explicit per-dimension fail-closed validity contract required')
        allowed=[r.get('deployment_allowed') is True and r.get('loss_type')!='none'
            and r.get('supervision_status')!='NA' and r.get('student_source')!='unobservable' for r in rows]
        self.register_buffer('state_allowed',torch.tensor(allowed,dtype=torch.bool))
        self.register_buffer('state_classification', torch.tensor(
            [r['loss_type']=='classification' for r in rows], dtype=torch.bool), persistent=False)
        self.required={key:[r['index'] for r in rows if key in r.get('required_validity_keys',[])]
            for key in {k for r in rows for k in r.get('required_validity_keys',[])}}
        self.rotations=self.state_contract.get('rotation_groups',[])
        self.head_weights={} if head_weights is None else dict(head_weights)
        if any(not isinstance(v,(float,int)) or isinstance(v,bool) or v<0 or not torch.isfinite(torch.tensor(v))
               for v in self.head_weights.values()): raise ValueError('finite nonnegative head weights required')

    def state_mask(self,prediction,validity):
        if prediction.ndim!=2 or prediction.shape[1]!=371: raise ValueError('state prediction shape must be [B,371]')
        supplied=validity.get('privileged_state',torch.zeros_like(prediction,dtype=torch.bool))
        mask=_mask(supplied,prediction)&self.state_allowed[None].to(prediction.device)
        sources=validity.get('state_sources',{})
        if not isinstance(sources,dict): raise ValueError('state_sources must contain explicit source masks')
        for key,indices in self.required.items():
            value=sources.get(key)
            if value is None: mask[:,indices]=False;continue
            if (not isinstance(value,torch.Tensor) or value.dtype!=torch.bool or value.device!=prediction.device
                    or value.shape not in (prediction.shape,prediction.shape[:1])):
                raise ValueError('state source mask must be boolean [B] or [B,371]: '+key)
            mask[:,indices]&=value[:,None] if value.ndim==1 else value[:,indices]
        return mask

    @staticmethod
    def _force_mask(qualification,prediction):
        empty=torch.zeros_like(prediction,dtype=torch.bool)
        if qualification is None: return empty,'missing_force_source_evidence'
        if not isinstance(qualification,dict): raise TypeError('force qualification dictionary required')
        masks=[]
        for key in ('source_valid','contact_valid','attributable'):
            value=qualification.get(key)
            if value is not None and (not isinstance(value,torch.Tensor) or value.dtype!=torch.bool or value.shape!=prediction.shape or value.device!=prediction.device):
                raise ValueError('force qualification mask shape must be [B,2,3]: '+key)
            if value is None: return empty,'missing_force_mask_'+key
            masks.append(value)
        required=('source_sha256','units','coordinate_frame','partner_attribution','per_digit_source','sign_convention','evidence_scope')
        missing=[key for key in required if key not in qualification]
        if missing: return empty,'missing_force_metadata_'+','.join(missing)
        if qualification['evidence_scope']!='native_qualified': return empty,'force_evidence_scope_not_native_qualified'
        digest=qualification['source_sha256']
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-fA-F]{64}',digest): return empty,'invalid_force_source_sha256'
        if qualification['units']!='N' or qualification['coordinate_frame']!='accepted_pad_normal': return empty,'unsupported_force_units_or_frame'
        if qualification['partner_attribution']!='verified_object_partner': return empty,'unverified_force_partner_attribution'
        if qualification['sign_convention']!='compressive_positive_on_object': return empty,'unsupported_force_sign_convention'
        sources=qualification['per_digit_source']
        if (not isinstance(sources,list) or len(sources)!=2 or any(not isinstance(row,list) or len(row)!=3
            or any(not isinstance(x,str) or not x.strip() for x in row) for row in sources)):
            return empty,'missing_per_digit_force_source'
        if not _force_proof_verified(qualification):return empty,'force_native_proof_not_verified'
        return masks[0]&masks[1]&masks[2],None

    def forward(self,predictions,targets,validity,*,force_qualification=None):
        if not isinstance(predictions,dict) or not predictions: raise ValueError('named estimator predictions required')
        first=next(iter(predictions.values()))
        if not isinstance(first,torch.Tensor): raise TypeError('tensor predictions required')
        zero=torch.where(torch.isfinite(first),first,0.).sum()*0.
        result={'total':zero,'coverage':{},'blocked_reasons':{}}
        allowed_heads=('privileged_state','geometry','velocity','deformation','contact_logits','force_proxy','slip_logits','slip_velocity')
        for name in allowed_heads:
            if name not in predictions or name not in targets: continue
            pred,target=predictions[name],targets[name]
            supplied=validity.get(name,torch.zeros_like(pred,dtype=torch.bool))
            mask=_mask(supplied,pred)
            if name=='privileged_state':
                mask=self.state_mask(pred,validity)
                scalar_mask=mask.clone(); rotations=[]
                for group in self.rotations:
                    lo,hi=group['slice']; group_mask=mask[:,lo:hi].all(-1)
                    if group.get('symmetry_mask_required'):
                        symmetry=validity.get('state_sources',{}).get('rotation_symmetry_valid')
                        if symmetry is None: group_mask&=False
                        elif not isinstance(symmetry,torch.Tensor) or symmetry.shape!=(len(pred),) or symmetry.dtype!=torch.bool or symmetry.device!=pred.device:
                            raise ValueError('rotation_symmetry_valid must be boolean [B]')
                        else: group_mask&=symmetry
                    scalar_mask[:,lo:hi]=False
                    mask[:,lo:hi]=group_mask[:,None]
                    rotations.append(quaternion_geodesic_loss(pred[:,lo:hi],target[:,lo:hi],group_mask))
                loss=masked_smooth_l1(pred,target,scalar_mask)+sum(rotations,zero)
            elif name=='force_proxy':
                if pred.shape[1:]!=(2,3): raise ValueError('force prediction shape must be [B,2,3]')
                qualified,reason=self._force_mask(force_qualification,pred)
                mask=mask&qualified
                if reason: result['blocked_reasons'][name]=reason
                if target.shape==pred.shape and bool((mask&(target<0)).any()):
                    raise ValueError('qualified compressive-positive pad-normal targets must be nonnegative')
                loss=masked_smooth_l1(pred,target,mask)
            elif name in ('contact_logits','slip_logits'):
                loss=masked_bce(pred,target,mask)
            else: loss=masked_smooth_l1(pred,target,mask)
            result[name]=loss
            result['coverage'][name]=float(mask.float().mean().detach())
            result['total']=result['total']+self.head_weights.get(name,1.)*loss
        return result
