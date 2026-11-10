"""Independent Student BC Adam; no PPO storage, latent sample or log-prob writes."""
import math
import torch

from .tabletop_privileged_dataset import masked_group_normalized_loss
from .tabletop_stereo_losses import StereoEstimationLoss,masked_smooth_l1
from .tabletop_stereo_supervision_dataset import batch_dataset,validate_dataset


DEFAULT_CONFIG=dict(lr=3e-4,lambda_action=1.,lambda_state=1.,lambda_depth=0.,lambda_temporal=0.,max_grad_norm=.5)


def _equal(a,b):
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        return a.shape==b.shape and a.dtype==b.dtype and bool(torch.isclose(a,b.to(a.device),rtol=0,atol=0,equal_nan=True).all())
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(_equal(a[k],b[k]) for k in a)
    return a==b


class StereoBCTrainer:
    def __init__(self,actor,config=None,*,ppo_trainer=None,schema_only=False):
        self.actor,self.ppo_trainer,self.schema_only=actor,ppo_trainer,schema_only
        self.config_metadata={} if config is None else dict(config)
        if self.config_metadata.get('schema_version','dh116-stereo-bc-v1')!='dh116-stereo-bc-v1':
            raise ValueError('versioned stereo BC configuration required')
        if any(key.startswith('lambda_') and key not in DEFAULT_CONFIG for key in self.config_metadata):
            raise ValueError('unknown BC loss weight')
        self.config=DEFAULT_CONFIG|{k:v for k,v in self.config_metadata.items() if k in DEFAULT_CONFIG}
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in self.config.values()):raise ValueError('finite nonnegative BC hyperparameters required')
        if self.config['lr']<=0 or self.config['max_grad_norm']<=0:raise ValueError('positive learning rate and gradient bound required')
        variance=getattr(actor,'log_std',None)
        self.parameters=[p for p in actor.parameters() if p.requires_grad and p is not variance]
        self.optimizer=torch.optim.Adam(self.parameters,lr=self.config['lr'])
        self.state_loss=StereoEstimationLoss();self.steps=0

    def step(self,batch):
        # Actor packages load in eval mode. Supervised optimization still needs
        # the encoder's training-time activation checkpointing at full camera
        # resolution; restore every prior mode before the next collection call.
        modes=[(module,module.training) for module in self.actor.modules()]
        self.actor.train()
        try:
            return self._step(batch)
        finally:
            for module,training in modes:
                module.training=training

    def _step(self,batch):
        if self.ppo_trainer is not None and not self.ppo_trainer.clean_boundary:
            raise RuntimeError('BC requires a clean empty PPO rollout boundary')
        if not isinstance(batch,dict) or '_dataset' not in batch:raise ValueError('batch must bind its admitted source dataset')
        dataset=validate_dataset(batch['_dataset'],allow_schema_only=self.schema_only)
        if dataset['schema_only'] and not self.schema_only:raise ValueError('CPU fixture BC requires explicit schema_only=True')
        device=next(self.actor.parameters()).device
        expected=batch_dataset(dataset,batch['indices'],device=device)
        for name in ('teacher_actions','action_valid','targets','validity'):
            if not _equal(batch[name],expected[name]):raise ValueError('BC batch differs from the admitted dataset: '+name)
        if not _equal(batch['history'].as_dict(),expected['history'].as_dict()):raise ValueError('BC sensor history differs from admitted dataset')
        history=expected['history'];target=expected['teacher_actions'].detach()
        output=None
        if hasattr(self.actor,'estimator'):
            output=self.actor.estimator(history)
            nominal=self.actor.action_net(output['latent']).tanh()
            state=self.state_loss(output,expected['targets'],expected['validity'],force_qualification=expected['force_qualification'])
            state_value=state['total']
        else:
            nominal=self.actor.deterministic(history);state_value=nominal.sum()*0.;state={'coverage':{},'blocked_reasons':{}}
        action=masked_group_normalized_loss(nominal,target,expected['action_valid'])
        # Canonical groups are already normalized by their physical scales:
        # ((a_s*scale)-(a_t*scale))/scale == a_s-a_t, then equal group means.
        terms=dict(action=action,state=state_value)
        terms['depth']=terms['temporal']=nominal.sum()*0.
        if self.config['lambda_depth']>0:
            if output is None or 'depth_m' not in output or 'depth_m' not in expected['targets'] or 'depth_m' not in expected['validity']:
                raise ValueError('depth loss requires an actual depth prediction head and qualified target/validity; no GT fallback')
            terms['depth']=masked_smooth_l1(output['depth_m'],expected['targets']['depth_m'],expected['validity']['depth_m'])
        if self.config['lambda_temporal']>0:
            if (output is None or 'velocity' not in expected['targets'] or 'velocity' not in expected['validity']
                    or 'temporal_correspondence' not in expected['validity']):
                raise ValueError('temporal loss requires qualified velocity and cross-frame point correspondence masks')
            previous=history.clone();previous.history_valid[:,-1]=False
            prior_geometry=self.actor.estimator(previous)['geometry']
            predicted_velocity=(output['geometry']-prior_geometry)/history.dt[:,-1,None,None]
            correspondence=expected['validity']['temporal_correspondence']
            if correspondence.shape==predicted_velocity.shape[:-1]:correspondence=correspondence[...,None].expand_as(predicted_velocity)
            if correspondence.shape!=predicted_velocity.shape or correspondence.dtype!=torch.bool:
                raise ValueError('temporal correspondence mask must match point IDs [B,128] or [B,128,3]')
            valid=expected['validity']['velocity']&correspondence&previous.history_valid.any(1)[:,None,None]
            terms['temporal']=masked_smooth_l1(predicted_velocity,expected['targets']['velocity'],valid)
        total=sum(self.config['lambda_'+name]*value for name,value in terms.items())
        norms={}
        for name,value in terms.items():
            if value.requires_grad:
                gradient=torch.autograd.grad(value,self.parameters,retain_graph=True,allow_unused=True)
                norm=sum((g.detach().square().sum() for g in gradient if g is not None),value.new_zeros(())).sqrt()
                norms[name]=float(norm)
            else:norms[name]=0.
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.parameters,self.config['max_grad_norm'],error_if_nonfinite=True)
        self.optimizer.step();self.steps+=1
        return dict(steps=self.steps,losses={k:float(v.detach()) for k,v in terms.items()}|{'total':float(total.detach())},
            gradient_norms=norms,gradient_norm_before_clip=float(norm),coverage=state['coverage'],blocked_reasons=state['blocked_reasons'],
            schema_only=dataset['schema_only'],native_qualified=False,ppo_storage_modified=False)
